import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig


def automatic_experiment_naming(cfg: DictConfig) -> str:
    """Generate experiment name automatically by training configuration.
    following cfg.expdir/cfg.datasets/AUTO_EXPNAME
    """
    name = "+".join(list(cfg.datasets.keys()))
    
    # model types
    name += f"/{cfg.model.encoder._target_.split('.')[-1]}"
    
    # encoder
    if hasattr(cfg.model.encoder, "channel_encoder"):
        encoder_cfg = cfg.model.encoder.channel_encoder
    elif hasattr(cfg.model.encoder, "encoder_shared"):
        encoder_cfg = cfg.model.encoder.encoder_shared
    else:
        encoder_cfg = cfg.model.encoder
    name += f"_enc{encoder_cfg.n_ds}_s{encoder_cfg.strides}_hd{encoder_cfg.hidden_dim}_oc{encoder_cfg.out_channels}_depth{encoder_cfg.depth}_dilation{encoder_cfg.dilation}"
    
    if hasattr(cfg.model.encoder, "query_size"):
        name += f"_nh{cfg.model.encoder.num_heads}_nq{cfg.model.encoder.query_size}"
    
    # decoder
    decoder_cfg = cfg.model.decoder.decoder_shared    
    name += f"_dec{decoder_cfg.n_ds}_s{decoder_cfg.strides}_hd{decoder_cfg.hidden_dim}_od{decoder_cfg.out_channels}_depth{decoder_cfg.depth}"

    # bottleneck
    if hasattr(cfg.model.bottleneck, "codebook_size"):
        name += f"_quantizer_cb{cfg.model.bottleneck.codebook_size}_dim{cfg.model.bottleneck.code_dim}"
    elif hasattr(cfg.model.bottleneck, "latent_vars"):
        name += f"_mixb_ic{cfg.model.bottleneck.in_channels}_dim{cfg.model.bottleneck.code_dim}_ndc{cfg.model.bottleneck.latent_vars.discrete.nb_code}"
    else:
        name += f"_vae{cfg.model.bottleneck.latent_dim}"

    # optimization
    loss_cfg = cfg.model.loss_cfg
    name += f"_recons{loss_cfg.recons}" #{loss_cfg.skew_factor}_{loss_cfg.normalization}_{loss_cfg.aggregation}"

    # augmentation
    name += f"_lraug{cfg.train.augmentation.lr_flip_prob}loss{cfg.model.lambdas.assignment}_delay{cfg.model.loss_cfg.assignment_delay}"
    name += f"_lraug_naive{cfg.train.augmentation.lr_flip_naive_prob}"
    name += f"_kptdrop{cfg.train.augmentation.kpt_dropout_prob}_{cfg.train.augmentation.kpt_dropout_max_parts}"
    name += f"_lr{cfg.optimizer.lr}"
    
    # remove whitespace and special characters
    name = name.replace(" ", "")
    name = name.replace("[", "")
    name = name.replace("]", "")
    name = name.replace(",", "+")

    return name


def check_config(cfg: DictConfig):
    # check encoder - bottleneck - decoder dim sizes
    code_dim = cfg.model.bottleneck.code_dim
    if isinstance(code_dim, ListConfig):
        latent_dim = sum(code_dim)
    elif isinstance(code_dim, int):
        latent_dim = code_dim
    else:
        raise TypeError("cfg.model.bottleneck.code_dim should be either int or list of int")

    if hasattr(cfg.model.encoder, "channel_encoder"):
        cfg_encoder = cfg.model.encoder.channel_encoder
        encoder_out_channels = cfg.model.encoder.latent_dim
    elif hasattr(cfg.model.encoder, "encoder_shared"):
        cfg_encoder = cfg.model.encoder.encoder_shared
        encoder_out_channels = cfg_encoder.out_channels
    else:
        cfg_encoder = cfg.model.encoder
        encoder_out_channels = cfg_encoder.out_channels

    assert encoder_out_channels == latent_dim, \
        f"Encoder output dim ({cfg.model.encoder.latent_dim}) does not match bottleneck latent dim ({latent_dim})"
    
    assert cfg.model.decoder.decoder_shared.in_channels == latent_dim, \
        f"Decoder input dim ({cfg.model.decoder.decoder_shared.in_channels}) does not match encoder latent dim ({latent_dim})"
    
    # check striding
    n_ds = cfg_encoder.n_ds
    strides_enc = cfg_encoder.strides
    
    ds_ratio = np.prod(strides_enc) if isinstance(strides_enc, ListConfig) else strides_enc**n_ds
    
    n_up = cfg.model.decoder.decoder_shared.n_ds
    strides_dec = cfg.model.decoder.decoder_shared.strides
    up_ratio = np.prod(strides_dec) if isinstance(strides_dec, ListConfig) else strides_dec**n_up
    assert ds_ratio == up_ratio, f"Encoder and decoder strides do not match: {ds_ratio} != {up_ratio}"

    return cfg


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def move_data_to_device(batch: Any, device: str = "cuda:0"):
    if isinstance(batch, list):
        batch_new = []
        for item in batch:
            if isinstance(item, list):
                batch_new.append([i.to(device) for i in item])
            else:
                batch_new.append(item.to(device))
        return batch_new

    elif isinstance(batch, torch.Tensor):
        return batch.to(device)

    elif isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                batch[k] = [item.to(device) for item in v]
        return batch
    else:
        raise NotImplementedError


def prepare_expdir(cfg: DictConfig):
    """Check validaity and create experiment directory (and necessary subdirectories).
    """
    if cfg.automatic_naming:
        cfg.expname = automatic_experiment_naming(cfg)

    if not cfg.overwrite:
        existing = [exp for exp in os.listdir(cfg.expdir) if cfg.expname in exp]
        if existing == 1:
            cfg.expname += "_version1"
        elif len(existing) > 1:
            versions = [int(exp.split("_version")[-1]) for exp in existing if "_version" in exp]
            cfg.expname += f"_version{max(versions)+1}"

    expdir = os.path.join(cfg.expdir, cfg.expname)
    if os.path.exists(expdir) and not cfg.overwrite:
        raise FileExistsError(f"Experiment {cfg.expname} already exists in {cfg.expdir}")

    os.makedirs(expdir, exist_ok=cfg.overwrite)
    os.makedirs(os.path.join(expdir, "tb"), exist_ok=cfg.overwrite)
    os.makedirs(os.path.join(expdir, "checkpoints"), exist_ok=cfg.overwrite)
        
    return expdir