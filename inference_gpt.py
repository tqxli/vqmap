import fire
import os
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from loguru import logger
from vqmap.trainer import initialize_trainer
from vqmap.datasets import initialize_dataset, collate_fn_mocap
from vqmap.models import initialize_model
from vqmap.utils.visualize import *
from vqmap.utils.run import set_random_seed
import matplotlib.pyplot as plt
from vqmap.utils.skeleton import *


def main(
    ckpt_path, mode, seed=1024,
):
    assert os.path.exists(ckpt_path), f"{ckpt_path} does exist"
    
    # must load parameters from train
    expdir = os.path.dirname(ckpt_path)    
    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    set_random_seed(seed)
    logger.info(f"Set random seed to: {seed}\n")
    
    logger.add(
        os.path.join(expdir, f"stats_eval.log"),
        format="{time:YYYY-MM-DD HH:mm} | {level} | {message}"
    )
    skeleton = skeleton_initialize_v2()
    
    # initialize trainer
    engine = initialize_trainer(config)
    engine.create(config)
    engine.model_to_device()
    engine.load_state_dict(ckpt_path, load_keys=['model'])
    engine.model.eval()
    
    # need an additional VAE decoder
    # in order to convert codes into motion
    vq_code_path = config.dataset.root
    vq_ckpt_path = np.load(vq_code_path, allow_pickle=True)[()]["config"].checkpoint
    vq_ckpt = torch.load(vq_ckpt_path)
    vq_config = vq_ckpt["config"]
    vq_model = initialize_model(vq_config.model)
    vq_model.load_state_dict(vq_ckpt["model"])
    vq_model.eval()
    vq_model.to(engine.device)
    
    # specify inference mode
    modes = ["transition", "single_condition"]
    assert mode in modes
    
    args = {
        "engine": engine,
        "vq_model": vq_model,
        "skeleton": skeleton,
        "expdir": expdir,
        "config": config,
        "vq_config": vq_config,
    }

    def _inference(mode):
        if mode == "single_condition":
            _single_condition(**args)
        elif mode == "transition":
            _transition(**args)

    if mode == "all":
        logger.info("Running full inference pipeline ...")
        _modes = modes
    else:
        _modes = [mode]
    for m in _modes:
        _inference(m)


def _transition(
    engine, vq_model, skeleton,
    expdir, config, vq_config,
):
    idx = torch.arange(config.model.vocab_size).unsqueeze(-1).to(engine.device)
    temperature = 1.0
    with torch.no_grad():
        logits, loss, attns = engine.model(idx)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
    
    probs = probs.detach().cpu().numpy()
    savepath = os.path.join(expdir, 'vis')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # save as heatmap
    savepath = os.path.join(savepath, 'conditional_probs.png')
    plot_heatmap(probs, savepath)


def _single_condition(
    engine, vq_model, skeleton,
    expdir, config, vq_config, 
    num_samples=4,
    **kwargs):
    start = np.arange(config.model.vocab_size)
    start = torch.tensor(start).unsqueeze(1).to(engine.device)

    with torch.no_grad():
        generated_idx, attentions = engine.model.generate_multimodal(
            start, 5, num_samples=num_samples,
        )
    N, T = generated_idx.shape

    if isinstance(vq_config.model.latent_dim, list):
        code_b, code_t = generated_idx//16, generated_idx%16
        code_b, code_t = code_b.view(-1), code_t.view(-1)
        quant_t = vq_model.quantizer_t.codebook[code_t].reshape(N, T, -1).permute(0, 2, 1)
        quant_b = vq_model.quantizer_b.codebook[code_b].reshape(N, T, -1).permute(0, 2, 1)
        dec = vq_model.decode(quant_t, quant_b)
    else:
        z = vq_model.quantizer.codebook[generated_idx.view(-1)]
        z = z.reshape(N, T, -1).permute(0, 2, 1)
        dec = vq_model.decode(z)

    dec = dec.detach().cpu().numpy()  
    dec = skeleton.convert_to_euclidean(dec)
    
    dec = dec.reshape((-1, num_samples, *dec.shape[1:]))

    savepath = os.path.join(expdir, f'decodes_single_condition.npy')
    np.save(savepath, dec)
    
    # TODO: visualization
    

if __name__ == "__main__":
    fire.Fire(main)