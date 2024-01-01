import fire
import os
import numpy as np
import torch
import random
from scipy import linalg
from vqmap.config.config import parse_config
from vqmap.logging.logger import PythonLogger
from vqmap.datasets.loaders import prepare_dataloaders, collate_fn, collate_fn_mocap
from vqmap.trainer.trainer import TrainerEngine
from vqmap.models import get_model
from vqmap.comparisons.evaluate.tools import save_metrics, format_metrics

def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_from_checkpoint(checkpoint):
    # load parameters from training directory for consistency
    expdir = os.path.dirname(checkpoint)
    config_path = os.path.join(expdir, 'parameters.yaml')
    config = parse_config(config_path)
    
    logger = PythonLogger(name=os.path.join(expdir, 'eval.txt'))
    
    # create model
    engine = TrainerEngine(device='cuda')
    engine.set_logger(logger)
    engine.create(config)
    engine.model_to_device()
    engine.load_state_dict(checkpoint, load_keys=['model'])
    engine.model.eval()
    
    return config, engine.model

def _data_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    
    return batch

def extract_gt_features(classifer, dataset, all_indices):
    z = []
    n_samples = len(all_indices)
    for i in range(n_samples//10):
        indices = np.arange(i*10, (i+1)*10)

        if not isinstance(dataset, torch.Tensor):
            samples = [dataset[all_indices[index]] for index in indices]
            batch = collate_fn_mocap(samples)
            batch = _data_to_device(batch)
        else:
            samples = torch.stack([dataset[all_indices[index]] for index in indices], 0)
            batch = samples.cuda()
        feats = classifer.latent_return(batch)
        z.append(feats.detach().cpu())
    
    z = torch.cat(z, 0).mean(-1)
    z = z.reshape(z.shape[0], -1).numpy()
    
    return z

def extract_features(classifer, gens):
    z = []
    n_samples = len(gens)
    for i in range(n_samples//10):
        indices = np.arange(i*10, (i+1)*10)
        batch = {"motion": gens[indices].cuda()}
        feats = classifer.latent_return(batch)
        z.append(feats.detach().cpu())
    
    z = torch.cat(z, 0).mean(-1)
    z = z.reshape(z.shape[0], -1).numpy()
    
    return z

def extract_gens(generator, n_samples, duration):
    gens = []
    for i in range(n_samples):
        gen = generator.generate(duration)
        gens.append(gen.detach().cpu())
    gens = torch.stack(gens, 0)
    gens = gens.reshape(*gens.shape[:2], -1)

    return gens

def get_vq_model(config):
    vq_root = os.path.dirname(config.dataloader.root)
    vq_checkpoint = os.path.join(
        vq_root,
        'model_last.pth'
    )
    vq_config_path = os.path.join(
        vq_root,
        'parameters.yaml'
    )
    vq_config = parse_config(vq_config_path).model
    vq_model = get_model(vq_config)
    vq_model.load_state_dict(
        torch.load(vq_checkpoint)["model"]
    )
    vq_model.eval()
    vq_model.cuda()
    
    return vq_model

def extract_gens_gpt(config, generator, vq_model, n_samples, duration):
    gens = []
    for i in range(n_samples // 2):
        start = torch.tensor(np.random.choice(config.model.gpt.vocab_size, 1)).unsqueeze(1)
        start = start.cuda()

        generated_idx, _ = generator.generate_multimodal(
            start, duration//16-1, num_samples=2, 
        )
        code_b, code_t = generated_idx//16, generated_idx%16
        N, T = code_b.shape

        code_b, code_t = code_b.view(-1), code_t.view(-1)

        quant_t = vq_model.quantizer_t.codebook[code_t].reshape(N, T, -1).permute(0, 2, 1)
        quant_b = vq_model.quantizer_b.codebook[code_b].reshape(N, T, -1).permute(0, 2, 1)

        quant_t = vq_model.upsample_t[0](quant_t)
        dec = vq_model.decoder_b(torch.cat((quant_t, quant_b), 1))
        dec = dec.detach().cpu()
        gens.append(dec)

    gens = torch.cat(gens, 0)
    gens = gens[np.random.permutation(n_samples)]
    
    return gens

def extract_gens_gpt_base(config, generator, vq_model, n_samples, duration):
    gens = []
    for i in range(n_samples // 2):
        start = torch.tensor(np.random.choice(config.model.gpt.vocab_size, 1)).unsqueeze(1)
        start = start.cuda()

        generated_idx, _ = generator.generate_multimodal(
            start, duration//16-1, num_samples=2, 
        )
        code = generated_idx
        N, T = code.shape
        code = code.view(-1)
        quant = vq_model.quantizer.codebook[code].reshape(N, T, -1).permute(0, 2, 1)
        dec = vq_model.decoder(quant)
        dec = dec.detach().cpu()
        gens.append(dec)

    gens = torch.cat(gens, 0)
    gens = gens[np.random.permutation(n_samples)]
    
    return gens


def compute_statistics(code):
    return np.mean(code, axis=0), np.cov(code, rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

# from action2motion
def calculate_diversity(activations, seed=None):
    diversity_times = 200
    # changes here: since we don't have action labels!
    num_motions = len(activations)

    diversity = 0

    if seed is not None:
        np.random.seed(seed)
        
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):

        diversity += np.linalg.norm(
            activations[first_idx, :]- activations[second_idx, :], 
        )
        # diversity += torch.dist(activations[first_idx, :],
                                # activations[second_idx, :])
    diversity /= diversity_times

    return diversity #.item()

def main(
    classifer_checkpoint,
    generator_checkpoint=None,
    n_samples=2000,
    niter=20,
):
    saveroot = os.path.dirname(generator_checkpoint)
    logger = PythonLogger(name=os.path.join(saveroot, 'eval_metrics.txt'))
    
    classifer_config, classifer = load_from_checkpoint(classifer_checkpoint)
    dataset = prepare_dataloaders(classifer_config.dataloader, split='val', logger=logger)['val'].dataset
    duration = classifer_config.dataloader.seqlen
    
    print("Dataset loaded")
    logger.log("GT dataset: ", dataset.__len__())
    
    dataset_gen = None
    if generator_checkpoint.endswith('.npy'):
        dataset_gen = np.load(generator_checkpoint, allow_pickle=True)[()]["out"]
        dataset_gen = dataset_gen[89952*24:]
        dataset_gen = torch.from_numpy(dataset_gen.reshape((-1, duration, dataset_gen.shape[-1]*dataset_gen.shape[-2])))
        logger.log("Reconstruction dataset: {}".format(dataset_gen.shape))
    else:
        generator_config, generator = load_from_checkpoint(generator_checkpoint)
        
        gpt = False
        if generator_config.model.name == "posegpt":
            vq_model = get_vq_model(generator_config)
            gpt = True

    metrics = {"gt-FID": [], "gen-FID": [], "gt-DIV": [], "gen-DIV": []}
    allseeds = list(range(niter))
    for seed in allseeds:
        fixseed(seed)
        
        gt_indices = np.random.permutation(len(dataset))[:n_samples*2]
        gt_indices_0, gt_indices_1 = gt_indices[:n_samples], gt_indices[n_samples:]
        
        gt_feats_0 = extract_gt_features(classifer, dataset, gt_indices_0)
        gt_feats_1 = extract_gt_features(classifer, dataset, gt_indices_1)
        mu_gt_0, sigma_gt_0 = compute_statistics(gt_feats_0)
        mu_gt_1, sigma_gt_1 = compute_statistics(gt_feats_1)

        fid_gt_gt = calculate_frechet_distance(mu_gt_0, sigma_gt_0, mu_gt_1, sigma_gt_1)
        print("gt-gt FID: ", fid_gt_gt)
        metrics['gt-FID'].append(fid_gt_gt)
        div_gt = calculate_diversity(gt_feats_1)
        
        print("DIV gt: ", div_gt)
        metrics['gt-DIV'].append(div_gt)
        
        if dataset_gen is not None:
            indices = np.random.permutation(len(dataset_gen))[:n_samples]
            gen_feats = extract_gt_features(classifer, dataset_gen, indices)
        else:
            if gpt:
                if hasattr(vq_model, "quantizer_t"):
                    gens = extract_gens_gpt(generator_config, generator, vq_model, n_samples, duration)
                else:
                    gens = extract_gens_gpt_base(generator_config, generator, vq_model, n_samples, duration)
            else:          
                gens = extract_gens(generator, n_samples, duration)

            gen_feats = extract_features(classifer, gens)
        mu_gens, sigma_gens = compute_statistics(gen_feats)
        fid_gt_gen = calculate_frechet_distance(mu_gt_0, sigma_gt_0, mu_gens, sigma_gens)
        print("gt-gen FID: ", fid_gt_gen)
        metrics['gen-FID'].append(fid_gt_gen)
        
        div_gen = calculate_diversity(gen_feats)
        print("DIV gen: ", div_gen)
        metrics['gen-DIV'].append(div_gen)
        

    savename = f"evaluation_metrics_{n_samples}_{niter}.npy"
    np.save(os.path.join(saveroot, savename), metrics)
    print(f"Saving to {saveroot}")
    
    print("====== SUMMARY ======")
    for k, v in metrics.items():
        avg = np.mean(v)
        std = np.std(v)
        print(k, avg, std)
    


if __name__ == '__main__':
    fire.Fire(main)