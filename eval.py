import fire
import os
import numpy as np
import torch
from loguru import logger
from vqmap.config.config import parse_config
from vqmap.trainer.trainer import TrainerEngine
from vqmap.datasets.loaders import prepare_dataloaders, collate_fn, collate_fn_mocap
from vqmap.datasets import initialize_dataset
from vqmap.utils.visualize import visualize, visualize2d
from tqdm import tqdm
import matplotlib.pyplot as plt
# import imageio
from copy import deepcopy
from vqmap.utils.skeleton import *


def skeleton_data_conversion(animal_skeleton, inputs, n_joints=23, seqlen=None):
    n_samples, seqlen, _ = inputs.shape
    inputs = inputs.reshape(n_samples*seqlen, n_joints, -1)
    n_chan = inputs.shape[-1]

    # quaternion
    if n_chan == 4:
        inputs = animal_skeleton.forward_kinematics_np(
            inputs, 
            np.zeros((inputs.shape[0], 3)), do_root_R=True
        )
    # continuous 6D rotation
    elif n_chan == 6:
        inputs = animal_skeleton.forward_kinematics_cont6d_np(
            inputs, 
            np.zeros((inputs.shape[0], 3)), do_root_R=True
        )
    # otherwise keep xyz
    inputs = inputs.reshape((n_samples, seqlen, n_joints, -1))
    return inputs
    

def main(
    checkpoint, mode,
    dataset='mocap', split='train',
    seed=1234, n_samples=6,
    augment=None, 
    **kwargs
):
    # load parameters from training directory for consistency
    expdir = os.path.dirname(checkpoint)
    config_path = os.path.join(expdir, 'parameters.yaml')
    config = parse_config(config_path)
    
    # logger = PythonLogger(name=os.path.join(expdir, 'eval.txt'))

    # dataset for inference
    config_dataset = config.dataset
    # kind = config_dataset.get("kind", "xyz")
    
    # replace root directory for local inference
    if not os.path.exists(config_dataset.root):
        config_dataset.root = config_dataset.root.replace(
            '/hpc/group/tdunn',
            '/home/tianqingli/dl-projects/duke-cluster'
        )

    # specify experiment groups
    if dataset == "amphe":
        config_dataset.root = '/media/mynewdrive/datasets/dannce/social_rat/LONGEVANS_M_SOC8'
        config_dataset.name = "mocap_unannot"     

    elif dataset == 'mocapephys':
        config_dataset.valid_sessions = [31]
        config_dataset.sampling = False
            
    # config_dataset["inference"] = True

    if mode != "sample":
        # dataloaders = prepare_dataloaders(config_dataset, split=split, logger=logger)
        dataloaders = initialize_dataset(config_dataset, splits=['val'])
    # skeleton information
    animal_skeleton = skeleton_initialize()
    n_joints = get_njoints(config_dataset)
    
    # create model
    engine = TrainerEngine(device='cuda')
    engine.set_logger(logger)
    engine.create(config)
    engine.model_to_device()
    engine.load_state_dict(checkpoint, load_keys=['model'])
    engine.model.eval()

    # select mode
    if mode == "latent":
        inputs, latent_vars, indices = engine.retrieve_latents(dataloader=dataloaders[split], augment=augment)
        results = {'code_indices': indices} #, "inputs": inputs}
        dataset_size = dataloaders[split].dataset.__len__()
        np.save(os.path.join(expdir, f'{os.path.basename(config_dataset.root)}_n{dataset_size}_{augment}_results'), results)
    
    elif mode == "infer_from_latent":
        latents = np.load(os.path.join(expdir, 'cluster_profile.npy'))
        latents = torch.from_numpy(latents).unsqueeze(-1).float().to(engine.device)
        with torch.no_grad():
            if hasattr(engine.model, "upsample_t"):
                quant_b, quant_t = latents[:, :32, :], latents[:, 32:, :]
                upsample_t = engine.model.upsample_t(quant_t)
                quant = torch.cat([upsample_t, quant_b], 1)
                out = engine.model.decoder_b(quant)
            else:
                out = engine.model.decoder(latents)
        out = out.detach().cpu().numpy()
        out = skeleton_data_conversion(animal_skeleton, out, n_joints)
        np.save(os.path.join(expdir, 'cluster_results'), out)
     
    elif mode == "inference":
        outputs, inputs = engine.inference(dataloader=dataloaders[split])
        outputs = skeleton_data_conversion(animal_skeleton, outputs, n_joints)
        inputs = skeleton_data_conversion(animal_skeleton, inputs, n_joints)
        outputs = outputs.reshape((-1, *outputs.shape[2:]))
        inputs = inputs.reshape(outputs.shape)
        losses = np.sqrt(np.sum((outputs - inputs)**2, axis=-1))
        losses = np.mean(losses, 1)
        np.save(os.path.join(expdir, f'{dataset}_{split}_outputs'), {"out": outputs, "loss": losses})
    
    elif mode == 'vq':
        vq_results, prequant = engine.retrieve_vq(dataloader=dataloaders[split])
        n_items = len(vq_results[0])
        vq_results = [[item[i] for item in vq_results] for i in range(n_items)]
        np.save(os.path.join(expdir, f'{dataset}_{split}_vq'), vq_results)
        print(prequant.shape)
        np.save(os.path.join(expdir, 'prequant_embeddings.npy'), prequant)
    
    elif mode == "visualize":
        # visualize a subset of samples
        np.random.seed(seed)

        try:
            _dataset = dataloaders[split].dataset
        except:
            _dataset = dataloaders[split]

        indices = np.random.choice(len(_dataset), n_samples)
        samples = [_dataset[index] for index in indices]

        if dataset == "mocapephys":
            batch = collate_fn(samples)
        else:
            batch = collate_fn_mocap(samples)

        batch = engine._data_to_device(batch)
        out = engine.model(batch)[0]

        if isinstance(out, list):
            out = [o.detach().cpu().numpy() for o in out]
        else:
            out = [out.detach().cpu().numpy()]

        for i, o in enumerate(out):
            out[i] = skeleton_data_conversion(animal_skeleton, o, n_joints)
        ref = skeleton_data_conversion(animal_skeleton, batch['ref'].detach().cpu().numpy(), n_joints)

        savepath = os.path.join(expdir, f'visualize_{dataset}_{split}_{seed}.mp4')

        if config_dataset.name == 'calms21':
            vis_fcn = visualize2d
        else:
            vis_fcn = visualize
        anim = vis_fcn(
            [ref, *out],
            batch["length"][0],
            savepath,
            ["GT"] + ["Recon"]*len(out)
        )

    elif mode == "sample":
        if config.model.vae:
            latent = torch.randn(64, config.model.latent_dim, 1).to(engine.device)

            out_params = engine.model.decoder(latent).detach().cpu().numpy()
            out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)

            np.save(os.path.join(expdir, 'vae_rand_sampling.npy'), {"params": out_params, "out": out})
            return
        
        if config.model.name == 'vqvae2':
            out = []
            num_codes_b, num_codes_t = config.model.vq.num_codes
            for i in range(num_codes_t):
                for j in range(num_codes_b):
                    dec = engine.model.decode_latent(i, j).detach().cpu().numpy()
                    out.append(dec)
            out_params = np.concatenate(out, 0)
            out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)
            out = out.reshape(num_codes_t, num_codes_b, *out.shape[1:])
            
            np.save(os.path.join(expdir, 'codebook_decodes.npy'), {"params": out_params, "out": out})
            return
        
        def cross_perturb(dist1, dist2):
            latents = []
            for i in range(dist1.shape[0]):
                # pertube codebook 1 by imposing codebook 0 embedding 0
                latent0 = torch.tile(dist1[i:i+1], (dist2.shape[0], 1))
                latent1 = dist2.clone()
                latent = torch.cat((latent0, latent1), 1)
                latents.append(latent)
            
            # for i in range(dist2.shape[0]):
            #     latent0 = torch.tile(dist2[i:i+1], (dist1.shape[0], 1))
            #     latent1 = dist1.clone()
            #     latent = torch.cat((latent1, latent0), 1)
            #     latents.append(latent)
            
            latents = torch.cat(latents, 0).unsqueeze(-1)
            return latents
        
        if config.model.name == 'style':
            c_key = [key for key in engine.model.state_dict() if 'content_vq' in key][0]
            s_key = [key for key in engine.model.state_dict() if 'style_vq' in key][0]
            codebook_c = engine.model.state_dict()[c_key].clone()
            codebook_s = engine.model.state_dict()[s_key].clone()
            codebook_c = codebook_c.view(-1, codebook_c.shape[-1])
            codebook_s = codebook_s.view(-1, codebook_s.shape[-1])

            # additive
            latents = []
            for vec_s in codebook_s:
                # latent = codebook_c + vec_s.unsqueeze(0)
                latent = torch.cat((codebook_c, vec_s.unsqueeze(0).repeat(codebook_c.shape[0], 1)), dim=1)
                latents.append(latent)

            latents = torch.cat(latents, 0).unsqueeze(-1)
            
            out_params = engine.model.dec(latents).detach().cpu().numpy()
            out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)

            np.save(os.path.join(expdir, 'sample_repeats.npy'), {"params": out_params, "out": out})
            
            latents = []
            for vec_c in codebook_c:
                # latent = codebook_c + vec_s.unsqueeze(0)
                latent = torch.cat((vec_c.unsqueeze(0).repeat(codebook_s.shape[0], 1), codebook_s), dim=1)
                latents.append(latent)

            latents = torch.cat(latents, 0).unsqueeze(-1)
            
            out_params = engine.model.dec(latents).detach().cpu().numpy()
            out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)

            np.save(os.path.join(expdir, 'sample_repeats_variants.npy'), {"params": out_params, "out": out})
            

            out_params_base = engine.model.dec(codebook_c.unsqueeze(-1)).detach().cpu().numpy()
            out_base = skeleton_data_conversion(animal_skeleton, out_params_base, n_joints)

            np.save(os.path.join(expdir, 'sample_repeats_content.npy'), {"out": out_base}) 
            
            out_params_base = engine.model.dec(codebook_s.unsqueeze(-1)).detach().cpu().numpy()
            out_base = skeleton_data_conversion(animal_skeleton, out_params_base, n_joints)

            np.save(os.path.join(expdir, 'sample_repeats_style.npy'), {"out": out_base}) 
            
            return
                      
        elif 'n_groups' in config.model.vq:
            codebooks = [
                engine.model.state_dict()[f'quantizer.sub_quantizers.{i}.codebook']
                for i in range(config.model.vq.n_groups)
            ]
            codebooks = [cb.view(-1, cb.shape[-1]) for cb in codebooks] #[N**2, D]
            latents = cross_perturb(codebooks[0], codebooks[1])

            out_params = engine.model.decoder(latents).detach().cpu().numpy()
            out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)

            out = out.reshape((codebooks[0].shape[0], codebooks[1].shape[0], *out.shape[1:]))
            print("Decodes: ", out.shape)
            np.save(os.path.join(expdir, 'codebook_decodes.npy'), {"params": out_params, "out": out})
            return

        # workaround naming differences
        try:
            codebook = engine.model.state_dict()["quantizer.embeddings"]
        except:
            codebook = engine.model.state_dict()["quantizer.codebook"]

        codebook = codebook.clone().reshape(-1, codebook.shape[-1])
        num_codes = codebook.shape[0]
        # latent = torch.tile(codebook.unsqueeze(-1), (1, 1, 6))

        latent = codebook.unsqueeze(-1)
            
        out_params = engine.model.decoder(latent).detach().cpu().numpy()
        out = skeleton_data_conversion(animal_skeleton, out_params, n_joints)
        
        # interpolation
        latent = latent.detach().cpu().numpy()
        latent_interpolation = np.concatenate(
            (np.repeat(latent, num_codes, axis=0), np.tile(latent, (num_codes, 1, 1))),
            axis=-1
        )
        latent_interpolation = torch.from_numpy(latent_interpolation).to(engine.device)
        out_params_inter = engine.model.decoder(latent_interpolation).detach().cpu().numpy()
        out_interpolation = skeleton_data_conversion(animal_skeleton, out_params_inter, n_joints)

        np.save(os.path.join(expdir, 'codebook_decodes.npy'), {"params": out_params, "out": out, "interpolation": out_interpolation})
        return
        
        saveroot = os.path.join(expdir, 'sample_repeats_single_rot')
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        for index in tqdm(range(codebook.shape[0])):
            savepath = os.path.join(saveroot, f'{index}.mp4')
            anim = visualize(
                [out[index:index+1]],
                out.shape[1],
                savepath,
                [str(index)],
            )
        
        imgs = []
        for index in range(codebook.shape[0]):
            vid = imageio.get_reader(os.path.join(saveroot, f'{index}.mp4'), 'ffmpeg')
            img = vid.get_data(0)
            imgs.append(img)
            vid.close()
        
        n_col = 10
        for i in range(codebook.shape[0] // n_col):
            indices = list(np.arange(i*n_col, (i+1)*n_col))
            to_plot = [imgs[idx] for idx in indices]
            to_plot = np.concatenate(to_plot, axis=1)
            plt.imsave(os.path.join(saveroot, f'combine_{i}.png'), to_plot)
        
        imgs = [f for f in os.listdir(saveroot) if 'combine' in f]
        imgs = [plt.imread(os.path.join(saveroot, im)) for im in imgs]
        img_full = np.concatenate(imgs, axis=0)
        plt.imsave(os.path.join(saveroot, 'combine.png'), img_full)
    
    elif mode == "encode_vqvae2":
        assert config.model.name == 'vqvae2'

        all_codes_t, all_codes_b = [], []
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        for batch in tqdm(dataloaders[split]):                
            batch = engine._data_to_device(batch)
            
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            # if augment == 'LR':
            #     motion = batch['motion']
            #     motion = motion.reshape(*motion.shape[:2], -1, 3)
            #     motion[:, :, :, 1] = -motion[:, :, :, 1]
            #     batch['motion'] = motion.reshape(*motion.shape[:2], -1)
            
            info_t, info_b = engine.model.encode(batch["motion"].to(engine.device))[-2:]
            # info_t, info_b = engine.model.encode(batch["motion"].to(engine.device))[-3:-1]
                     
        # end.record()

        # # Waits for everything to finish running
        # torch.cuda.synchronize()

        # print(start.elapsed_time(end))
        # breakpoint()
        # return
            
            if engine.model.continuous_top:
                codes_t = engine.model.sample_from_distribution(info_t)
                codes_t = codes_t.detach().cpu().numpy()
                codes_b = engine.model.sample_from_distribution(info_b)
                codes_b = codes_b.detach().cpu().numpy()
                
                # codes_b = info_b[1].detach().cpu().numpy()
            else:
                codes_t, codes_b = info_t[1].detach().cpu().numpy(), info_b[1].detach().cpu().numpy()
            # codes_b = codes_b.reshape((-1, 2**config.model.down_t[1]))
            all_codes_t.append(codes_t)
            all_codes_b.append(codes_b)
        all_codes_t, all_codes_b = np.concatenate(all_codes_t), np.concatenate(all_codes_b)
        np.save(os.path.join(expdir, f'{os.path.basename(config_dataset.root)}_{augment}_results'), 
        {"top": all_codes_t, "bottom": all_codes_b})
    elif mode == 'vq_vqvae2':
        assert config.model.name == 'vqvae2'
        prequant = []
        for batch in tqdm(dataloaders[split]):
            batch = engine._data_to_device(batch)
            inputs = batch['motion']
            
            enc_b = engine.model.encoder_b(inputs)
            enc_t = engine.model.encoder_t(enc_b.permute(0, 2, 1))
            
            enc = torch.cat((enc_b, enc_t), 1).permute(0, 2, 1).detach().cpu().numpy()
            prequant.append(enc)
        prequant = np.concatenate(prequant, 0)
        prequant = np.reshape(prequant, (-1, prequant.shape[-1]))
        print(prequant.shape)
        np.save(os.path.join(expdir, 'prequant_embeddings.npy'), prequant)
    
    return

if __name__ == "__main__":
    fire.Fire(main)