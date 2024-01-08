import fire
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from loguru import logger
from vqmap.trainer import initialize_trainer
from vqmap.datasets import initialize_dataset, collate_fn_mocap
from vqmap.utils.visualize import visualize, visualize_latent_space
from vqmap.utils.run import set_random_seed
from tqdm import tqdm
import matplotlib.pyplot as plt
from vqmap.utils.skeleton import *


def main(
    ckpt_path, mode, seed=1024, split='inference',
    n_samples=4,
):
    assert os.path.exists(ckpt_path), f"{ckpt_path} does exist"
    
    # must load parameters from train
    expdir = os.path.dirname(ckpt_path)    
    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    set_random_seed(seed)
    logger.info(f"Set random seed to: {seed}\n")
    
    # create new logger
    logger.add(
        os.path.join(expdir, f"stats_eval.log"),
        format="{time:YYYY-MM-DD HH:mm} | {level} | {message}"
    )
    
    # skeleton
    skeleton = skeleton_initialize_v2()
    
    # specify inference mode
    modes = ["sample", "visualize", "code"]
    assert mode in ["all", "sample", "visualize", "code"]
    config_dataset = config.dataset
    dataloader = None
    if mode != "sample":
        dataloaders = initialize_dataset(config_dataset, splits=[split])
        dataloader = dataloaders[split]
    
    # initialize trainer
    engine = initialize_trainer(config)
    engine.create(config)
    engine.model_to_device()
    engine.load_state_dict(ckpt_path, load_keys=['model'])
    engine.model.eval()
    
    args = {
        "dataloader": dataloader,
        "engine": engine,
        "skeleton": skeleton,
        "expdir": expdir,
        "seed": seed,
        "config": config,
    }

    def _inference(mode):
        if mode == "visualize":
            _visualize(**args, n_samples=n_samples)
        elif mode == "code":
            _code(**args)
        elif mode == "sample":
            _sample(**args)
    
    if mode == "all":
        logger.info("Running full inference pipeline ...")
        _modes = modes
    else:
        _modes = [mode]
    for m in _modes:
        _inference(m)
    
    return


def _visualize(
    dataloader, engine, skeleton,
    expdir, seed, config,
    n_samples=4
):  
    logger.info("Running: visualization of reconstruction quality")
    # visualize a subset of reconstructed samples
    np.random.seed(seed)
    try:
        _dataset = dataloader.dataset
    except:
        _dataset = dataloader

    indices = np.random.choice(len(_dataset), n_samples)
    samples = [_dataset[index] for index in indices]

    batch = collate_fn_mocap(samples)
    batch = engine._data_to_device(batch)
    out = engine.model(batch)[0]

    if isinstance(out, list):
        out = [o.detach().cpu().numpy() for o in out]
    else:
        out = [out.detach().cpu().numpy()]

    for i, o in enumerate(out):
        out[i] = skeleton.convert_to_euclidean(o)
    ref = skeleton.convert_to_euclidean(batch['ref'].detach().cpu().numpy())

    savepath = os.path.join(expdir, f'vis_seqs_seed{seed}.mp4')
    anim = visualize(
        skeleton,
        [ref, *out],
        batch["length"][0],
        savepath,
        ["GT"] + ["Recon"]*len(out)
    )


def _code(
    dataloader, engine, skeleton,
    expdir, seed, config,
):
    logger.info("Running: VQ code extraction")
    vq_results, _ = engine.retrieve_vq(dataloader=dataloader)
    dataroot = config.dataset.root.split('/')[-1]
    datapath = dataloader.dataset.datapath
    fname = os.path.join(expdir, f'vq_code_{dataroot}.npy')
    data = {
        'code': vq_results,
        'datapath': datapath,
        'config': config
    }
    np.save(fname, data)
    

def _sample(
    dataloader, engine, skeleton,
    expdir, seed, config,
):
    logger.info("Running: latent space decoding and visualization")
    # workaround quantizer naming differences TODO
    try:
        codebook = engine.model.state_dict()["quantizer.embeddings"]
    except:
        codebook = engine.model.state_dict()["quantizer.codebook"]

    codebook = codebook.clone().reshape(-1, codebook.shape[-1])
    codebook = codebook.detach().cpu().numpy()

    num_codes = config.model.bottleneck.args.nb_code
    if isinstance(num_codes, list):
        num_codes_0, num_codes_1 = num_codes
        for i in range(num_codes_1):
            for j in range(num_codes_0):
                dec = engine.model.decode_latent(i, j).detach().cpu().numpy()
                out.append(dec)
        out_params = np.concatenate(out, 0)
    else:
        out_params = np.concatenate(
            [engine.model.decode_latent(idx).detach().cpu().numpy() for idx in range(num_codes)]
        )
    out = skeleton.convert_to_euclidean(out_params)
    
    # save
    fname = os.path.join(expdir, f'vq_codebook.npy')
    data = {
        "codebook": codebook,
        "decodes": out_params,
        "code_sequences": out,
    }
    np.save(fname, data)

    # visualize the latent space
    savepath = os.path.join(expdir, 'vis')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    visualize_latent_space(
        skeleton,
        out,
        savepath,
    )

if __name__ == "__main__":
    fire.Fire(main)
 