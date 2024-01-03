import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from typing import List, Optional
from vqmap.losses.kl import KLLoss
import vqmap.models.quantizer as quantizers
import vqmap.models.encoder as encoders
import vqmap.models.decoder as decoders
from vqmap.models.gpt import Block
from copy import deepcopy


class SequentialVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.initialize()
        self.init_optimization()

    def get_encdec(self, config, nfeats, latent_dim, type='Encoder'):
        cls_name = config.get("cls", f"Conv1D{type}")
        registry = encoders if type == "Encoder" else decoders
        assert hasattr(registry, cls_name), f"Specified {type}-{cls_name} does not exist"
        cls = getattr(registry, cls_name)
        arch = cls(nfeats=nfeats, latent_dim=latent_dim, **config)
    
        return arch

    def initialize(self):
        self.latent_dim = latent_dim = self.cfg.latent_dim
        self.nfeats = nfeats = self.cfg.nfeats
        
        # encoder, decoder
        self.encoder = self.get_encdec(self.cfg.encoder, nfeats, latent_dim, 'Encoder')
        self.decoder = self.get_encdec(self.cfg.decoder, nfeats, latent_dim, 'Decoder')
        
        # bottleneck
        self._init_bottleneck()
    
    def _init_bottleneck(self):
        bn_cfg = self.cfg.bottleneck
        self.bn_cls = bn_cls = bn_cfg.type
        
        if bn_cls == 'ae':
            return
        elif bn_cls == 'vae':
            self.mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.logvar = nn.Linear(self.latent_dim, self.latent_dim)
            self.sample_mean, self.fact = None, None
        elif bn_cls == 'quantizer':
            vq = bn_cfg.get('args', {})
            vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
            args = {k:v for k, v in vq.items() if k != 'name'}
            if 'n_groups' in args:
                n_groups = args["n_groups"]
                args = {k:v for k, v in args.items() if k != 'n_groups'}
                self.quantizer = quantizers.MultiGroupQuantizer(
                    vq_type, args, n_groups,
                )
                self.code_dim = self.quantizer.code_dim
            else:
                self.quantizer = vq_type(**args)
                self.code_dim = vq.code_dim

    def compute_kl(self, distribution):
        # Create a centred normal distribution to compare with
        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl = self.kl_loss(distribution, distribution_ref)
        
        return kl

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)
        
    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector, distribution
    
    def _bottleneck_vae(self, latent):
        latent = latent.permute(0, 2, 1)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        std = logvar.exp().pow(0.5)
        distribution = torch.distributions.Normal(mu, std)
        
        latent = self.sample_from_distribution(distribution)
        latent = latent.permute(0, 2, 1)
        return latent, distribution

    def _bottleneck_quantizer(self, latent):
        if len(latent.shape) == 2:
            if self.code_dim < latent.shape[-1]:
                latent = latent.reshape(latent.shape[0], self.code_dim, -1)
            else:
                latent = latent.unsqueeze(-1)

        return self.quantize(latent)
    
    def _bottleneck(self, latent):
        if self.bn_cls == 'ae':
            return latent, None
        elif self.bn_cls == 'vae':
            return self._bottleneck_vae(latent)
        elif self.bn_cls == 'quantizer':
            return self._bottleneck_quantizer(latent)
        else:
            raise AssertionError
    
    def quantize(self, z):
        quant, emb_loss, info = self.quantizer(z)
        return quant, (emb_loss, info)

    def _prepare_objective(self):
        self.recons_loss = nn.MSELoss()
        self.kl_loss = KLLoss()
    
    def init_optimization(self):
        self._prepare_objective()
        self.lambdas = self.cfg.lambdas
    
    def compute_loss(self, batch, out, zinfo):
        recons = self.recons_loss(batch['motion'], out)
        loss = {"recons": recons}
        
        if self.bn_cls == 'vae':
            distribution = zinfo
            loss["kl"] = self.compute_kl(distribution)
        elif self.bn_cls == 'quantizer':
            emb_loss = zinfo[0]
            if isinstance(emb_loss, torch.Tensor):
                loss["commitment"] = emb_loss
            else:
                loss = {**loss, **emb_loss}         
        
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return total_loss, loss_dict

    def forward(self, batch):
        latent = self.encoder(batch['motion'])
        z, z_info = self._bottleneck(latent)
        out = self.decoder(z)
        return out, *self.compute_loss(batch, out, z_info)


if __name__ == "__main__":
    # load example config file
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/model.yaml')
    
    # generate random inputs
    seq = torch.randn((2, 64, 69)).cuda()
    batch = {"motion": seq}
       
    # create and test models
    model = SequentialVAE(cfg.model).cuda()
    out, total_loss, loss_dict = model(batch)
    print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")
    
    cfg.model.bottleneck.type = 'vae'
    model = SequentialVAE(cfg.model).cuda()
    out, total_loss, loss_dict = model(batch)
    print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")
    
    cfg.model.bottleneck.type = 'ae'
    model = SequentialVAE(cfg.model).cuda()
    out, total_loss, loss_dict = model(batch)
    print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")