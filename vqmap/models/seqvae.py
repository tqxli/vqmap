import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from typing import Optional
from vqmap.losses.kl import KLLoss
import vqmap.models.quantizer as quantizers
import vqmap.models.encoder as encoders
import vqmap.models.decoder as decoders


class SequentialVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for 1D sequential data
    """
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
    
    def _bottleneck_vae(self, latent):
        latent = latent.permute(0, 2, 1)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        std = logvar.exp().pow(0.5)
        distribution = torch.distributions.Normal(mu, std)
        
        latent = self.sample_from_distribution(distribution)
        latent = latent.permute(0, 2, 1)
        return latent, distribution

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
        
        bottleneck_loss = self.compute_bottleneck_loss(zinfo)        
        loss = {**bottleneck_loss, **loss}
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return total_loss, loss_dict

    def compute_bottleneck_loss(self, zinfo):
        loss = {}
        if self.bn_cls == 'vae':
            distribution = zinfo
            loss["kl"] = self.compute_kl(distribution)
        elif self.bn_cls == 'quantizer':
            emb_loss = zinfo[0]
            if isinstance(emb_loss, torch.Tensor):
                loss["commitment"] = emb_loss
            else:
                return emb_loss
        return loss

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

    def forward(self, batch):
        z, z_info, latent = self.encode(batch)
        out = self.decode(z)
        return out, *self.compute_loss(batch, out, z_info)
    
    def encode(self, batch):
        latent = self.encoder(batch['motion'])
        z, z_info = self._bottleneck(latent)
        return z, z_info, latent

    def decode(self, z):
        return self.decoder(z)
    
    def decode_latent(self, idx):
        z = self.quantizer.codebook[idx].unsqueeze(0).unsqueeze(-1)
        return self.decode(z)

class MultiBranchSeqVAE(SequentialVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._initialize_multi_branches()
    
    def _initialize_multi_branches(self):
        self.multi_branches = hasattr(self.cfg, "input_feats")
        if not self.multi_branches:
            return
    
        self.input_feats = input_feats = self.cfg.input_feats
        self.n_branches = len(input_feats)
            
        # processing layers for each inputs
        self.encoders_proc =  nn.ModuleList(
            [nn.Conv1d(input_feats[i], self.nfeats, 1)
                for i in range(self.n_branches)]
        )
        self.decoders_post = nn.ModuleList(
            [nn.Conv1d(self.nfeats, input_feats[i], 1)
                for i in range(self.n_branches)]
        )        

    def encode_proc(self, batches):
        if not self.multi_branches:
            return batches['motion']
        
        inputs = [
            encoder(batch['motion'].permute(0, 2, 1))
            for batch, encoder in zip(batches, self.encoders_proc)
        ]
        inputs = torch.cat(inputs, 0).permute(0, 2, 1)
        return inputs

    def decode_post(self, outs):
        if not self.multi_branches:
            return outs
        outs = outs.reshape(2, -1, *outs.shape[1:])
        outs = [
            decoder(latent.permute(0, 2, 1)).permute(0, 2, 1)
            for latent, decoder in zip(outs, self.decoders_post)
        ]
        return outs   
    
    def encode(self, batches):
        inputs = self.encode_proc(batches)
        latent = self.encoder(inputs)
        z, z_info = self._bottleneck(latent)
        return z, z_info, latent
    
    def decode(self, z):
        outs = self.decoder(z)
        outs = self.decode_post(outs)
        return outs
    
    def compute_loss(self, batches, outs, zinfo):
        if not self.multi_branches:
            return super().compute_loss(batches, outs, zinfo)
        
        loss = {
            f"recons{i}":self.recons_loss(batches[i]["motion"], out)
            for i, out in enumerate(outs)
        }
        
        bottleneck_loss = self.compute_bottleneck_loss(zinfo)        
        loss = {**bottleneck_loss, **loss}
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return total_loss, loss_dict        


class HierarchicalSeqVAE(MultiBranchSeqVAE):
    def initialize(self):
        # for different levels in the VAE
        self.latent_dim = latent_dim = self.cfg.latent_dim
        assert len(latent_dim) == 2, \
            "Must specify 2 latent dims in HierarchicalSeqVAE"
        self.down_ts = down_ts = self.cfg.get("down_t", [4, 0])

        self.nfeats = input_nfeats = self.cfg.nfeats
            
        # create a common encoder-decoder backbone (shared or not)
        self.encoder_layer0 = encoders.Conv1DEncoder(
            input_nfeats, latent_dim[0], width=self.cfg.encoder.width, down_t=down_ts[0],
        )
        self.encoder_layer1 = encoders.Conv1DEncoder(
            latent_dim[0], latent_dim[1], width=self.cfg.encoder.width, down_t=down_ts[1],
        )
        self.decoder_layer1 = decoders.Conv1DDecoder(
            latent_dim[1], latent_dim[1], width=self.cfg.decoder.width, down_t=down_ts[1],
        )
        self.decoder_layer0 = decoders.Conv1DDecoder(
            input_nfeats, latent_dim[0]*2, width=self.cfg.decoder.width, down_t=down_ts[0],
        )
        
        upsample = []
        if down_ts[1] != 0:
            upsample.append(nn.Upsample(scale_factor=2**down_ts[1]))
        upsample.append(nn.Conv1d(latent_dim[1], latent_dim[0], 1) )
        self.upsample = nn.Sequential(*upsample)
        
        # bottleneck
        bn_cfg = self.cfg.bottleneck
        self.bn_cls = bn_cfg.type
        vq = bn_cfg.get('args', {})
        vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
        num_codes = bn_cfg.num_codes
        args = {k:v for k, v in vq.items() if k != 'name' and k != 'num_codes'}
        
        self.quantizer0 = vq_type(num_codes[0], **args)
        self.quantizer1 = vq_type(num_codes[1], **args)
        
        self._initialize_multi_branches()

    def encode(self, batch):
        inputs = self.encode_proc(batch)
        enc_0 = self.encoder_layer0(inputs)
        enc_1 = self.encoder_layer1(enc_0.permute(0, 2, 1))

        quant_1, emb_1, info_1 = self.quantizer1(enc_1)
        quant_0, emb_0, info_0 = self.quantizer0(enc_0)
        
        return quant_1, quant_0, (emb_1+emb_0, info_1, info_0)

    def decode(self, quant_1, quant_0):
        # upsample the top latent map
        upsample_1 = self.upsample(quant_1)
        quant = torch.cat([upsample_1, quant_0], 1)
        dec = self.decoder_layer0(quant)
        dec = self.decode_post(dec)
        return dec

    def decode_latent(self, code_1_idx, code_0_idx):
        quant_t = self.quantizer1.codebook[code_1_idx].unsqueeze(0).unsqueeze(-1)
        quant_b = self.quantizer0.codebook[code_0_idx].unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2**self.down_ts[1])
        dec = self.decode(quant_t, quant_b)

        return dec

    def forward(self, batch):
        quant_1, quant_0, zinfo = self.encode(batch)
        outs = self.decode(quant_1, quant_0)
        
        return outs, *self.compute_loss(batch, outs, zinfo)


if __name__ == "__main__":
    # load example config file
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/model.yaml')
    
    # generate random inputs
    # seq = torch.randn((2, 64, 69)).cuda()
    # batch = {"motion": seq}
       
    # # create and test models
    # # loss types should be automatically adjusted depending on the bottleneck type
    # model = SequentialVAE(cfg.model).cuda()
    # out, total_loss, loss_dict = model(batch)
    # print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")
    
    # cfg.model.bottleneck.type = 'vae'
    # model = SequentialVAE(cfg.model).cuda()
    # out, total_loss, loss_dict = model(batch)
    # print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")
    
    # cfg.model.bottleneck.type = 'ae'
    # model = SequentialVAE(cfg.model).cuda()
    # out, total_loss, loss_dict = model(batch)
    # print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")
    
    # seq1 = torch.randn((2, 64, 69)).cuda()
    # seq2 = torch.randn((2, 64, 54)).cuda()
    # batches = [
    #     {'motion': seq1},
    #     {'motion': seq2}
    # ]
    
    # cfg = OmegaConf.load('configs/model_multibranch.yaml')
    # model = MultiBranchSeqVAE(cfg.model).cuda()
    # outs, total_loss, loss_dict = model(batches)
    # print(f"Output shape: {outs[0].shape} {outs[1].shape}")
    # print(f"Loss: {total_loss} {loss_dict.keys()}")
    
    # cfg.model.bottleneck.type = 'vae'
    # model = MultiBranchSeqVAE(cfg.model).cuda()
    # outs, total_loss, loss_dict = model(batches)
    # print(f"Output shape: {outs[0].shape} {outs[1].shape}")
    # print(f"Loss: {total_loss} {loss_dict.keys()}")
    
    # cfg = OmegaConf.load('configs/model_hier.yaml')
    # seq = torch.randn((2, 64, 69)).cuda()
    # batch = {"motion": seq}
       
    # create and test models
    # loss types should be automatically adjusted depending on the bottleneck type
    # model = HierarchicalSeqVAE(cfg.model).cuda()
    # outs, total_loss, loss_dict = model(batches)
    # print(f"Output shape: {outs[0].shape} {outs[1].shape}")
    # print(f"Loss: {total_loss} {loss_dict.keys()}")
    # print(f"Output shape: {out.shape} Loss: {total_loss} {loss_dict.keys()}")