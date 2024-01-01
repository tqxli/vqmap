import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from typing import List, Optional
from vqmap.losses.kl import KLLoss
import vqmap.models.vq as quantizers
import vqmap.models.encoder as encoders
import vqmap.models.decoder as decoders
from vqmap.models.gpt import Block
from copy import deepcopy


def get_encdec(config, nfeats, latent_dim, vae, type='Encoder'):
    cls_name = config.get("cls", f"Conv1D{type}")
    registry = encoders if type == "Encoder" else decoders
    assert hasattr(registry, cls_name), f"Specified {type}-{cls_name} does not exist"
    cls = getattr(registry, cls_name)
    layer = cls(nfeats=nfeats, latent_dim=latent_dim, vae=vae, **config)
    
    return layer


class SingleModalityVAE(nn.Module):
    def __init__(self,
        nfeats, latent_dim,
        enc_config, dec_config, lambdas, 
        modality='motion',
        vae=False,
        vq=None
    ):
        super().__init__()
        
        # architecture
        self.encoder = get_encdec(enc_config, nfeats, latent_dim, vae, 'Encoder')

        self.decoder = get_encdec(dec_config, nfeats, latent_dim, vae, 'Decoder')
        self.seq_modeling = "Transformer" in enc_config.get("cls", f"Conv1D{type}")
        
        # latent
        self.vae = vae
        if vae:
            self.mu = nn.Linear(self.encoder.width, latent_dim)
            self.logvar = nn.Linear(self.encoder.width, latent_dim)
        # conditioning on action classes
        self.action_cond = hasattr(self.encoder, "action_cond")
        if vae:
            self.sample_mean = False
            self.fact = None            
        
        self.vq = vq is not None
        if self.vq:
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
        # data modality
        self.modality = modality
        if self.modality == "neural":
            self.softplus = nn.Softplus()

        # loss weights
        self.lambdas = lambdas
        self._prepare_objective()


    def _prepare_objective(self):
        # if self.modality == "neural":
            # self.recons_loss = nn.PoissonNLLLoss(log_input=False, reduction='sum')
        # else:
        if self.modality == 'neural':
            self.recons_loss = nn.MSELoss(reduction='sum') #nn.L1Loss()
        else:
            self.recons_loss = nn.MSELoss()
        self.kl_loss = KLLoss()

    def _prepare_inputs(self, batch, dec=False):
        x = batch[self.modality]
        if self.seq_modeling:
            lengths = batch["length"]
            actions = batch.get("action", None)
            if dec:
                return lengths, actions
            return x, lengths, actions

        if dec:
            return []
        
        return [x]

    def forward(self, batch):
        latent = self.encoder(*self._prepare_inputs(batch))

        if self.vae:
            # normal distribution is returned, need reparam
            # distribution = latent
            # latent = self.sample_from_distribution(distribution)
            latent = latent.permute(0, 2, 1)
            mu = self.mu(latent)
            logvar = self.logvar(latent)

            std = logvar.exp().pow(0.5)
            distribution = torch.distributions.Normal(mu, std)
            
            latent = self.sample_from_distribution(distribution)
            latent = latent.permute(0, 2, 1)

        if self.vq:
            z = latent #[B, hidden_dim]
            
            if len(z.shape) == 2:
                if self.code_dim < z.shape[-1]:
                    z = z.reshape(z.shape[0], self.code_dim, -1)
                else:
                    z = z.unsqueeze(-1)

            latent, emb_loss, info = self.quantize(z)

        out = self.decoder(latent, *self._prepare_inputs(batch, dec=True))

        if self.modality == "neural":
            # keep all positives
            out = self.softplus(out)

        # compute loss
        recons = self.recons_loss(batch[self.modality], out)
        if self.modality == "neural":
            recons = recons / out.shape[0] / out.shape[1]
        loss = {"recons": recons}
        if self.vq:
            if isinstance(emb_loss, torch.Tensor):
                loss["commitment"] = emb_loss
            else:
                loss = {**loss, **emb_loss}
        if self.vae:
            loss["kl"] = self.compute_kl(distribution)
        
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return out, total_loss, loss_dict
    
    def encode(self, batch):
        latent = self.encoder(*self._prepare_inputs(batch))

        if self.vae:
            # normal distribution is returned, need reparam
            # distribution = latent
            # latent = self.sample_from_distribution(distribution)
            latent = latent.permute(0, 2, 1)
            mu = self.mu(latent)
            logvar = self.logvar(latent)

            std = logvar.exp().pow(0.5)
            distribution = torch.distributions.Normal(mu, std)
            
            latent = self.sample_from_distribution(distribution)
            latent = latent.permute(0, 2, 1)

        if self.vq:
            z = latent #[B, hidden_dim]
            if len(z.shape) == 2:
                if self.code_dim < z.shape[-1]:
                    z = z.reshape(z.shape[0], self.code_dim, -1)
                else:
                    z = z.unsqueeze(-1)
            
            latent, emb_loss, info = self.quantize(z)
            return latent, emb_loss, info, z
        
        return latent
    
    def decode(self, latent_vector, lengths, actions):
        return self.decoder(latent_vector, lengths, actions)
    
    def quantize(self, z):
        quant, emb_loss, info = self.quantizer(z)
        return quant, emb_loss, info
     
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
    
    def sample_prior(self, batch_size, seqlen):
        lengths = [seqlen] * batch_size
        latent_vector = torch.randn(batch_size, self.encoder.latent_dim)
        
        out = self.decoder(latent_vector, lengths)
        return out
        
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
        return latent_vector
    

class HierarchicalVQVAE(nn.Module):
    def __init__(
        self, 
        config,
        nfeats, latent_dim,
        lambdas,
        modality='motion',
    ):
        super().__init__()

        # architecture
        self.encoder_bottom = encoders.Conv1DEncoder(nfeats, latent_dim, **config.encoder.bottom)
        self.encoder_top = encoders.Conv1DEncoder(latent_dim, latent_dim*2, **config.encoder.top)
        self.decoder_top = decoders.Conv1DDecoder(nfeats, latent_dim*2, **config.decoder.top)
        self.decoder_bottom = decoders.Conv1DDecoder(nfeats, latent_dim, **config.decoder.bottom)
        
        vq_type = getattr(quantizers, config.vq.get("name", "QuantizeEMAReset"))
        args = {k:v for k, v in config.vq.items() if k != 'name'}
        self.code_dim = latent_dim
        self.quantizer_bottom = vq_type(**config.vq.bottom)
        self.quantizer_top = vq_type(**config.vq.top)
        
        # loss weights
        self.lambdas = lambdas
        self.recons_loss = nn.MSELoss()
        
        # data modality
        self.modality = modality
    
    def forward(self, batch):
        # latent_b = self.encoder_bottom(batch[self.modality]) #[bs, L//4, latent_dim]
        # latent_bq, emb_loss_b, info_b = self.quantizer_bottom(latent_b)

        # latent_t = self.encoder_top(latent_bq.permute(0, 2, 1))
        # latent_tq, emb_loss_t, info_t = self.quantizer_top(latent_t)

        # dec_t = self.decoder_top(latent_tq)
        # dec_b = self.decoder_bottom(latent_bq)
        
        # the original VQVAE-2 modeling
        inputs = batch[self.modality]
        latent_b = self.encoder_bottom(inputs)
        latent_t = self.encoder_top(latent_b)
        
        # vector quantization occurs in the decoding
        latent_tq, emb_loss_t, info_t = self.quantizer_top(latent_t)
        dec_t = self.decoder_top(latent_tq)
        enc_b = torch.cat([dec_t, latent_b], 1)
        quant_b, emb_loss_b, info_b = self.quantizer_bottom(enc_b)
        
        quant = torch.cat([dec_t, quant_b], 1)
        dec_b = self.dec(quant)

        # compute loss
        recons_t = self.recons_loss(dec_t, batch[self.modality])
        recons_b = self.recons_loss(dec_b, batch[self.modality])
        loss = {
            "recons_t": recons_t,
            "recons_b": recons_b,
            "commitment": emb_loss_b + emb_loss_t
        }
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return [dec_b, dec_t], total_loss, loss_dict

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)


class HierarchicalVQVAEv2(nn.Module):
    def __init__(self, config, nfeats):
        super().__init__()
        
        # loss weights
        self.lambdas = config.lambdas
        self.recons_loss = nn.MSELoss()
        
        # encoders
        down_ts = config.down_ts
        self.latent_dims = latent_dims = config.latent_dims
        self.n_levels = n_levels = len(down_ts)
        self.widths = widths = config.get("widths", [512]*n_levels)
        dilation_rates = config.get("dilations", [3]*n_levels)
        strides = config.get("strides", [2]*n_levels)
        
        self.encoders = nn.ModuleList()
        for level in range(n_levels):
            if level == 0:
                encoder = encoders.Conv1DEncoder(
                    nfeats, latent_dims[level], down_t=down_ts[0],
                    stride_t=strides[0], dilation_growth_rate=dilation_rates[0]
                )
            else:
                encoder = encoders.Conv1DEncoder(
                    latent_dims[level-1], latent_dims[level], down_t=down_ts[level],
                    stride_t=strides[level], dilation_growth_rate=dilation_rates[level]
                )
            self.encoders.append(encoder)
        
        self.decoders = nn.ModuleList()
        for level in range(n_levels):
            if level == n_levels - 1:
                decoder = decoders.Conv1DDecoder(
                    nfeats, latent_dims[0], down_t=down_ts[0], width=widths[0],
                    stride_t=strides[0], dilation_growth_rate=dilation_rates[0]
                )
            else:
                decoder = decoders.Conv1DDecoder(
                    latent_dims[-(level+2)], latent_dims[-(level+1)], down_t=down_ts[-(level+1)],
                    widths=widths[-(level+1)],
                    stride_t=strides[-(level+1)], dilation_growth_rate=dilation_rates[-(level+1)]
                )
            self.decoders.append(decoder)
            
        vq_type = getattr(quantizers, config.vq.get("name", "QuantizeEMAReset"))
        num_codes = config.vq.num_codes
        mu, beta = config.vq.mu, config.vq.beta
        self.quantizers = nn.ModuleList()
        for level in range(n_levels):
            code_dim = latent_dims[level]
            quantizer = vq_type(num_codes[level], code_dim, mu, beta)
            self.quantizers.append(quantizer)
    
    def forward(self, batch):
        inputs = latents = batch['motion']
        quantized = []
        emb_losses = []
        for level in range(self.n_levels):
            latents = self.encoders[level](latents)
            latents, emb_loss, info = self.quantizers[level](latents)
            quantized.append(latents)
            emb_losses.append(emb_loss)

            latents = latents.permute(0, 2, 1)

        decoded = []
        for level in range(self.n_levels):
            dec = quantized[-(level+1)]
            for i in range(level, self.n_levels, 1):
                dec = self.decoders[i](dec).permute(0, 2, 1)
            decoded.append(dec.permute(0, 2, 1)) #[bs, T, J*3]
        
        loss = {"commitment": sum(emb_losses)}
        for level in range(self.n_levels):
            idx = -(level+1)
            loss[f"recons_{level}"] = self.recons_loss(decoded[idx], inputs)
        
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return decoded, total_loss, loss_dict

    def encode(self, batch):
        inputs = latents = batch['motion']
        quantized = []
        infos = []
        for level in range(self.n_levels):
            latents = self.encoders[level](latents)
            latents, emb_loss, info = self.quantizers[level](latents)
            quantized.append(latents)
            infos.append(info)
            latents = latents.permute(0, 2, 1)
        return quantized, infos

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)      


class VQVAE2(nn.Module):
    def __init__(self, config, nfeats):
        super().__init__()
        
        # loss weights
        self.modality = 'motion'
        self.lambdas = config.lambdas
        self.recons_loss = nn.MSELoss()
        
        latent_dim_b, latent_dim_t = config.latent_dims
        width = config.get("width", 256)
        self.down_ts = down_ts = config.get("down_t", [2, 2])
        
        self.continuous_top = config.get("continuous_top", False)
        
        self.n_branches = 1 if isinstance(nfeats, int) else len(nfeats)
        self.multi = False
        input_nfeats = nfeats
        if self.n_branches > 1:
            self.multi = True
            self.common_feats = common_nfeats = config.get("common_feats", 128)
            # dataset specific data preproc
            self.encoders_proc =  nn.ModuleList(
                [nn.Conv1d(nfeats[i], common_nfeats, 1)
                 for i in range(self.n_branches)]
            )
            input_nfeats = common_nfeats
            self.decoders_post = nn.ModuleList(
                [nn.Conv1d(common_nfeats, nfeats[i], 1)
                 for i in range(self.n_branches)]
            )
        
        self.no_top_ds = config.get("no_top_ds", False)
         
        self.encoder_b = encoders.Conv1DEncoder(
            input_nfeats, latent_dim_b, width=width, down_t=down_ts[0], #stride_t=strides[0]
        )
        self.encoder_t = encoders.Conv1DEncoder(
            latent_dim_b, latent_dim_t, width=width, down_t=down_ts[1], #stride_t=strides[1]
        )
        self.decoder_t = decoders.Conv1DDecoder(
            latent_dim_t, latent_dim_t, width=width, down_t=down_ts[1], #stride_t=strides[1]
        )

        self.decoder_b = decoders.Conv1DDecoder(
            input_nfeats, latent_dim_b*2, width=width, down_t=down_ts[0], #stride_t=strides[0]
        )
        # self.upsample_t = nn.ConvTranspose1d(latent_dim_t, latent_dim_b, 4, 4) #only worked for 4x upsampling
        upsample_t = []
        if down_ts[1] != 0:
            upsample_t.append(nn.Upsample(scale_factor=2**down_ts[1]))
        upsample_t.append(nn.Conv1d(latent_dim_t, latent_dim_b, 1) )
        self.upsample_t = nn.Sequential(*upsample_t)
        
        vq_type = getattr(quantizers, config.vq.get("name", "QuantizeEMAReset"))
        num_codes = config.vq["num_codes"]
        mu, beta = config.vq["mu"], config.vq["beta"]
        
        # if self.continuous_top:
        #     self.mu = nn.Linear(latent_dim_t, latent_dim_t)
        #     self.logvar = nn.Linear(latent_dim_t, latent_dim_t)
        #     self.sample_mean = False
        #     self.fact = None
        #     self.kl_loss = KLLoss()
        if self.continuous_top:
            self.mu_t = nn.Linear(latent_dim_t, latent_dim_t)
            self.logvar_t = nn.Linear(latent_dim_t, latent_dim_t)
            self.mu_b = nn.Linear(latent_dim_b, latent_dim_b)
            self.logvar_b = nn.Linear(latent_dim_b, latent_dim_b)
            self.sample_mean = False
            self.fact = None
            self.kl_loss = KLLoss()
        else:
            self.quantizer_t = vq_type(num_codes[1], latent_dim_t, mu, beta)
            self.quantizer_b = vq_type(num_codes[0], latent_dim_b, mu, beta)
        
        self.merge_tb = config.get("merge_tb", True)
        if self.merge_tb:        
            self.quantize_conv_b = nn.Conv1d(latent_dim_b+latent_dim_t, latent_dim_b, 1)
    
    def forward(self, batch):
        # print(batch[0]['motion'].shape, batch[1]['motion'].shape)
        if self.multi:
            latents = [encoder(bat['motion'].permute(0, 2, 1)) for bat, encoder in zip(batch, self.encoders_proc)]
            inputs = torch.cat(latents, 0).permute(0, 2, 1)
        else:
            inputs = batch['motion']
            
        quant_t, quant_b, emb_loss, info_t, info_b = self.encode(inputs)
        outs = self.decode(quant_t, quant_b)

        if self.multi:
            outs = outs.reshape(2, -1, *outs.shape[1:])
            outs = [
                decoder(latent.permute(0, 2, 1)).permute(0, 2, 1)
                for latent, decoder in zip(outs, self.decoders_post)
            ]
            loss = {f"recons{i}":self.recons_loss(recon, batch[i]["motion"]) for i, recon in enumerate(outs)}
        else:
            loss = {"recons": self.recons_loss(outs, inputs)}
        
        if self.continuous_top:
            distribution = info_t
            loss["kl"] = self.compute_kl(distribution) + self.compute_kl(emb_loss)
        else:
            loss["commitment"] = emb_loss

        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return outs, total_loss, loss_dict

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(float(self.lambdas[key]) * lossval)
        
        return sum(total_loss)  
        
    def encode(self, inputs):
        enc_b = self.encoder_b(inputs)
        enc_t = self.encoder_t(enc_b.permute(0, 2, 1))

        if self.continuous_top:
            latent = enc_t.permute(0, 2, 1)
            mu = self.mu_t(latent)
            logvar = self.logvar_t(latent)
            std = logvar.exp().pow(0.5)
            distribution = torch.distributions.Normal(mu, std)
            
            latent = self.sample_from_distribution(distribution)
            quant_t = latent.permute(0, 2, 1)
        else:
            quant_t, emb_t, info_t = self.quantizer_t(enc_t)

        dec_t = self.decoder_t(quant_t).permute(0, 2, 1)
        if self.merge_tb:
            enc_b = torch.cat([dec_t, enc_b], 1)
            enc_b = self.quantize_conv_b(enc_b)
            
        if self.continuous_top:
            latent = enc_b.permute(0, 2, 1)
            mu = self.mu_b(latent)
            logvar = self.logvar_b(latent)
            std = logvar.exp().pow(0.5)
            distribution_b = torch.distributions.Normal(mu, std)
            
            latent_b = self.sample_from_distribution(distribution_b)
            quant_b = latent_b.permute(0, 2, 1)
            
            return quant_t, quant_b, distribution_b, distribution, None
        else:
            quant_b, emb_b, info_b = self.quantizer_b(enc_b)
        
        if self.continuous_top:
            return quant_t, quant_b, emb_b, distribution, info_b
        
        return quant_t, quant_b, emb_t+emb_b, info_t, info_b

    def _prepare_inputs(self, batch):
        return [batch['motion']]
    
    def decode(self, quant_t, quant_b):
        # upsample the top latent map
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.decoder_b(quant)
        
        return dec

    def decode_latent(self, code_t, code_b):
        quant_t = self.quantizer_t.codebook[code_t].unsqueeze(0).unsqueeze(-1)
        # quant_t = self.quantize_conv_b.
        quant_b = self.quantizer_b.codebook[code_b].unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2**self.down_ts[1])
        dec = self.decode(quant_t, quant_b)

        return dec
    
    def sample_prior(self, batch_size, seqlen):
        lengths = [seqlen] * batch_size
        latent_vector = torch.randn(batch_size, self.encoder.latent_dim)
        
        out = self.decoder(latent_vector, lengths)
        return out
        
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
        return latent_vector

    def compute_kl(self, distribution):
        # Create a centred normal distribution to compare with
        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl = self.kl_loss(distribution, distribution_ref)
        
        return kl


class VQPC(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, batch):
        inputs = batch['motion']


class MultiEncodersVAE(nn.Module):
    def __init__(
        self, 
        config,
        nfeats, latent_dim,
        lambdas,
        modality='motion',
        skeleton_biases=None,
    ):
        super().__init__()
        
        self.modality = modality
        self.lambdas = lambdas
        
        self.n_branches = len(nfeats)
        self.encoders = nn.ModuleList(
            [get_encdec(
                config.encoder, nfeats[i], latent_dim, False, 'Encoder',
            ) for i in range(self.n_branches)]
        )
        self.decoders = nn.ModuleList(
            [get_encdec(
                config.decoder, nfeats[i], latent_dim, False, 'Decoder',
            ) for i in range(self.n_branches)]
        )
        self.vq = self._prepare_vq(config.vq)
        self.skeleton_biases = None
        if skeleton_biases is not None:
            self.skeleton_biases = nn.Parameter(torch.randn(self.n_branches, latent_dim))

        self.recons_loss = nn.MSELoss()

    def forward(self, batches):
        latents = [encoder(*self._prepare_inputs(batch)
        ) for batch, encoder in zip(batches, self.encoders)]
        bs = [latent.shape[0] for latent in latents]

        latents = torch.cat(latents, 0)
        
        latents_q, emb_loss, info = self.vq(latents)
        
        latents_q = [latents_q[:bs[0]], latents_q[bs[0]:]]
        if self.skeleton_biases is not None:
            for i in range(self.n_branches):
                latents_q[i] += self.skeleton_biases[i].unsqueeze(0).unsqueeze(-1)
        
        outs = [decoder(latent) for latent, decoder in zip(latents_q, self.decoders)]

        # compute loss
        recons = [
            self.recons_loss(out, batch[self.modality]
        ) for out, batch in zip(outs, batches)]
        loss = {f"recons{i}": reconloss for i, reconloss in enumerate(recons)}

        if isinstance(emb_loss, torch.Tensor):
            loss["commitment"] = emb_loss
        else:
            loss = {**loss, **emb_loss}
        
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return outs, total_loss, loss_dict

    def _prepare_inputs(self, batch, dec=False):
        x = batch[self.modality]
        if dec:
            return []
        return [x]

    def _prepare_vq(self, vq):
        vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
        args = {k:v for k, v in vq.items() if k != 'name'}
        if 'n_groups' in args:
            n_groups = args["n_groups"]
            args = {k:v for k, v in args.items() if k != 'n_groups'}
            quantizer = quantizers.MultiGroupQuantizer(
                vq_type, args, n_groups,
            )
        else:
            quantizer = vq_type(**args)
        return quantizer

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)


class MultiEncoderVAEv2(nn.Module):
    def __init__(
        self, 
        config,
        nfeats, latent_dim,
        lambdas,
        modality='motion',
        vae=False
    ):
        super().__init__()
        
        self.modality = modality
        self.lambdas = lambdas
         
        self.n_branches = len(nfeats)
        # self.encoders_proc =  nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(nfeats[i], width, 3, 1, 1),
        #         nn.ReLU()
        #     ) for i in range(self.n_branches)
        # ])
        # self.encoder = encoders.Conv1DEncoderBackbone(latent_dim, **config.encoder)
        # self.decoder = decoders.Conv1DDecoderBackbone(latent_dim, **config.decoder)
        # self.decoders_post = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Conv1d(width, nfeats[i], 3, 1, 1)
        #     ) for i in range(self.n_branches)
        # )
        common_nfeats = config.get("common_nfeats", 128)
        skeleton_biases = config.get("skeleton_biases", None)
        depth = config.get("depth", 1)
        
        self.encoders_proc =  nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(nfeats[i], common_nfeats, 1),
            ) for i in range(self.n_branches)
        ])
        # self.encoders_proc = nn.ModuleList()
        # for i in range(self.n_branches):
        #     encoder_proc = [nn.Conv1d(nfeats[i], common_nfeats, 1)]
        #     if depth > 1:s
        #         encoder_proc += [nn.Conv1d(common_nfeats, common_nfeats) for j in range(depth-1)]
        #     encoder_proc = nn.Sequential(*encoder_proc)
        #     self.encoders_proc.append(encoder_proc)
            
        self.encoder = encoders.Conv1DEncoder(common_nfeats, latent_dim, **config.encoder)
        self.decoder = decoders.Conv1DDecoder(common_nfeats, latent_dim, **config.decoder)
        self.decoders_post = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(common_nfeats, nfeats[i], 1)
            ) for i in range(self.n_branches)
        )
        # self.decoders_proc = nn.ModuleList()
        # for i in range(self.n_branches):
        #     intermediate_feats = common_nfeats if depth > 1 else nfeats[i]
        #     decoder_proc = [nn.Conv1d(common_nfeats, intermediate_feats)]
        #     if depth > 1:
        #         decoder_proc += [nn.Conv1d(intermediate_feats, common_nfeats) for j in range(depth-1)]
        #     decoder_proc = nn.Sequential(*decoder_proc)
        #     self.decoders_proc.append(decoder_proc)
        
        self.vae = vae
        if vae:
            self.mu = nn.Linear(self.encoder.width, latent_dim)
            self.logvar = nn.Linear(self.encoder.width, latent_dim)
            self.sample_mean = False
            self.fact = None 
        else:
            self.vq = self._prepare_vq(config.vq)
        self.skeleton_biases = None
        if skeleton_biases is not None:
            self.skeleton_biases = nn.Parameter(torch.randn(self.n_branches, latent_dim))

        self.recons_loss = nn.MSELoss()
        self.kl_loss = KLLoss()

    def forward(self, batches):
        latents = [encoder(*self._prepare_inputs(batch)
        ) for batch, encoder in zip(batches, self.encoders_proc)]
        bs = [latent.shape[0] for latent in latents]

        latents = torch.cat(latents, 0).permute(0, 2, 1)
        latents = self.encoder(latents)

        if self.vae:
            latent = latents.permute(0, 2, 1)
            mu = self.mu(latent)
            logvar = self.logvar(latent)

            std = logvar.exp().pow(0.5)
            distribution = torch.distributions.Normal(mu, std)
            
            latent = self.sample_from_distribution(distribution)
            latents_q = latent.permute(0, 2, 1)
        else:
            latents_q, emb_loss, info = self.vq(latents)

        latents_q = [latents_q[:bs[0]], latents_q[bs[0]:]]
        
        # latents_q, emb_loss = [], []
        # for latent in latents:
        #     quantized = self.vq(latent)
        #     latents_q.append(quantized[0])
        #     emb_loss.append(quantized[1])

        if self.skeleton_biases is not None:
            for i in range(self.n_branches):
                latents_q[i] += self.skeleton_biases[i].unsqueeze(0).unsqueeze(-1)
        
        latents_q = torch.cat(latents_q, 0)
        latents_q = self.decoder(latents_q)
        
        latents_q = [latents_q[:bs[0]], latents_q[bs[0]:]]
        outs = [decoder(latent.permute(0, 2, 1)).permute(0, 2, 1) for latent, decoder in zip(latents_q, self.decoders_post)]

        # compute loss
        recons = [
            self.recons_loss(out, batch[self.modality]
        ) for out, batch in zip(outs, batches)]
        loss = {f"recons{i}": reconloss for i, reconloss in enumerate(recons)}

        # loss["commitment"] = emb_loss
        # loss.update({f"commitment{i}": embloss for i, embloss in enumerate(emb_loss)})
        
        if self.vae:
            loss["kl"] = self.compute_kl(distribution)
        else:
            if isinstance(emb_loss, torch.Tensor):
                loss["commitment"] = emb_loss
            else:
                loss = {**loss, **emb_loss}
            
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return outs, total_loss, loss_dict

    def _prepare_inputs(self, batch, dec=False):
        x = batch[self.modality].permute(0, 2, 1)
        if dec:
            return []
        return [x]

    def _prepare_vq(self, vq):
        vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
        args = {k:v for k, v in vq.items() if k != 'name'}
        if 'n_groups' in args:
            n_groups = args["n_groups"]
            args = {k:v for k, v in args.items() if k != 'n_groups'}
            quantizer = quantizers.MultiGroupQuantizer(
                vq_type, args, n_groups,
            )
        else:
            quantizer = vq_type(**args)
        return quantizer

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)
    
    def compute_kl(self, distribution):
        # Create a centred normal distribution to compare with
        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl = self.kl_loss(distribution, distribution_ref)
        
        return kl
    
    def sample_prior(self, batch_size, seqlen):
        lengths = [seqlen] * batch_size
        latent_vector = torch.randn(batch_size, self.encoder.latent_dim)
        
        out = self.decoder(latent_vector, lengths)
        return out
        
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
        return latent_vector


class MultiVQVAE(nn.Module):
    def __init__(
        self, 
        nfeats, latent_dim,
        downs_t, strides_t,
        lambdas, vq
    ):
        super().__init__()
        
        self.levels = levels = len(downs_t)
        
        self.vaes = nn.ModuleList()
        for level in range(levels):
            enc_config = {
                "down_t": downs_t[level],
                "stride_t": strides_t[level],
            }
            dec_config = enc_config
            self.vaes.append(
                SingleModalityVAE(nfeats, latent_dim, enc_config, dec_config, lambdas,
                                  vq=vq)
            )
    
    def forward(self, batch):
        outs, losses = [], []
        loss_dict_all = {}
        for level in range(self.levels):
            vae = self.vaes[level]
            out, total_loss, loss_dict = vae(batch)
            outs.append(out)
            losses.append(total_loss)
            
            for k, v in loss_dict.items():
                loss_dict_all[f"{level}_{k}"] = v
        
        losses = sum(losses)
        return outs, losses, loss_dict_all
    
    def encode(self, batch):
        latent, emb_loss, info = [], [], []
        for level in range(self.levels):
            vae = self.vaes[level]
            out = vae.encode(batch)
            latent.append(out[0])
            emb_loss.append(out[1])
            info.append(out[2])
        
        return latent, emb_loss, info


class StyleVQVAE(nn.Module):
    def __init__(
        self, 
        config,
        nfeats, latent_dim,
        lambdas,
        modality='motion',
    ):
        super().__init__()

        self.content_enc = get_encdec(
            config.content_enc, nfeats, latent_dim, False)
        self.style_enc = get_encdec(
            config.style_enc, nfeats, latent_dim, False)
        self.dec = get_encdec(
            config.dec, nfeats, latent_dim*2, False, 'Decoder')
        
        self.content_vq = self._prepare_vq(config.content_vq)
        self.style_vq = self._prepare_vq(config.style_vq)
        
        self.lambdas = lambdas
        self._prepare_objective()
        
        self.modality = modality
        if self.modality == "neural":
            self.softplus = nn.Softplus()

    def _prepare_objective(self):
        self.recons_loss = nn.MSELoss()

    def _prepare_inputs(self, batch, dec=False):
        x = batch[self.modality]
        if dec:
            return []
        return [x]
    
    def _prepare_vq(self, vq):
        vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
        args = {k:v for k, v in vq.items() if k != 'name'}
        if 'n_groups' in args:
            n_groups = args["n_groups"]
            args = {k:v for k, v in args.items() if k != 'n_groups'}
            quantizer = quantizers.MultiGroupQuantizer(
                vq_type, args, n_groups,
            )
        else:
            quantizer = vq_type(**args)
        return quantizer
    
    def forward(self, batch):
        inputs = self._prepare_inputs(batch)

        z_c = self.content_enc(*inputs) #[B, D, T//S]
        z_s = self.style_enc(*inputs) #[B, D, 1]
        
        latent_c, emb_loss_c, info_c = self.content_vq(z_c)
        latent_s, emb_loss_s, info_s = self.style_vq(z_s)
        
        # latent = latent_c + latent_s
        latent = torch.cat((latent_c, latent_s.repeat(1, 1, latent_c.shape[-1])), dim=1)

        out = self.dec(latent, *self._prepare_inputs(batch, dec=True))

        # compute loss
        recons = self.recons_loss(out, batch[self.modality])   
        loss = {"recons": recons}
        for emb_loss in [emb_loss_c, emb_loss_s]:
            if isinstance(emb_loss, torch.Tensor):
                if "commitment" in loss:
                    loss["commitment"] += emb_loss
                else:
                    loss["commitment"] = emb_loss
            else:
                for k, v in emb_loss.items():
                    loss[k] = loss[k] + v if k in loss else v
        
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return out, total_loss, loss_dict
    
    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)
    

class TransformerStep(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.latent_size = config.latent_dim
        self.condition_size = config.observation_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = config.trajectory_length #config.block_size//config.transition_dim-1
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.latent_step = config.latent_step
        self.state_conditional = config.state_conditional
        if "masking" in config:
            self.masking = config.masking
        else:
            self.masking = "none"
        
        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.vq = self._prepare_vq(config.vq)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))

        self.embed = nn.Linear(self.observation_dim, self.embedding_dim)

        self.predict = nn.Linear(self.embedding_dim, self.observation_dim)

        self.cast_embed = nn.Linear(self.embedding_dim, self.latent_size)
        self.latent_mixing = nn.Linear(self.latent_size, self.embedding_dim)
        if "bottleneck" not in config:
            self.bottleneck = "pooling"
        else:
            self.bottleneck = config.bottleneck

        if self.bottleneck == "pooling":
            self.latent_pooling = nn.MaxPool1d(self.latent_step, stride=self.latent_step)
        # elif self.bottleneck == "attention":
        #     self.latent_pooling = AsymBlock(config, self.trajectory_length // self.latent_step)
        #     self.expand = AsymBlock(config, self.trajectory_length)
        else:
            raise ValueError(f'Unknown bottleneck type {self.bottleneck}')

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.lambdas = config.lambdas

    def _prepare_vq(self, vq):
        vq_type = getattr(quantizers, vq.get("name", "QuantizeEMAReset"))
        args = {k:v for k, v in vq.items() if k != 'name'}
        if 'n_groups' in args:
            n_groups = args["n_groups"]
            args = {k:v for k, v in args.items() if k != 'n_groups'}
            quantizer = quantizers.MultiGroupQuantizer(
                vq_type, args, n_groups,
            )
        else:
            quantizer = vq_type(**args)
        return quantizer

    def encode(self, joined_inputs):
        b, t, joined_dimension = joined_inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.embed(joined_inputs)

        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.encoder(x)
        ## [ B x T x embedding_dim ]
        if self.bottleneck == "pooling":
            x = self.latent_pooling(x.transpose(1, 2)).transpose(1, 2)
        elif self.bottleneck == "attention":
            x = self.latent_pooling(x)
        else:
            raise ValueError()

        ## [ B x (T//self.latent_step) x embedding_dim ]
        x = self.cast_embed(x)
        return x.permute(0, 2, 1)
    
    def decode(self, latents):
        """
            latents: [B x (T//self.latent_step) x latent_size]
            state: [B x observation_dimension]
        """
        B, T, _ = latents.shape
        # state_flat = torch.reshape(state, shape=[B, 1, -1]).repeat(1, T, 1)
        # if not self.state_conditional:
        #     state_flat = torch.zeros_like(state_flat)
        # inputs = torch.cat([state_flat, latents], dim=-1)
        inputs = self.latent_mixing(latents)
        if self.bottleneck == "pooling":
            inputs = torch.repeat_interleave(inputs, self.latent_step, 1)
        elif self.bottleneck == "attention":
            inputs = self.expand(inputs)

        inputs = inputs + self.pos_emb[:, :inputs.shape[1]]
        x = self.decoder(inputs)
        x = self.ln_f(x)

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
        # joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        # joined_pred[:, :, :self.observation_dim] += torch.reshape(state, shape=[B, 1, -1])
        return joined_pred

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)

    def forward(self, batch):
        inputs, ref = batch["motion"], batch["ref"]
        trajectory_feature = self.encode(inputs)
        latents_q, emb_loss, info = self.vq(trajectory_feature)
        indices = info[-1].reshape(latents_q.shape[0], latents_q.shape[2])
        if self.bottleneck == "attention":
            if self.masking == "uniform":
                mask = torch.ones(latents_q.shape[0], latents_q.shape[1], 1).to(latents_q.device)
                mask_index = np.random.randint(0, latents_q.shape[1], size=[latents_q.shape[0]])
                for i, start in enumerate(mask_index):
                    mask[i, -start:, 0] = 0
                latents_q = latents_q * mask
                indices = indices * mask
                trajectory_feature = trajectory_feature * mask
            elif self.masking == "none":
                pass
            else:
                raise ValueError(f"Unknown masking type {self.masking}")
        out = self.decode(latents_q.permute(0, 2, 1))
        
        recons = F.mse_loss(out, ref)
        loss = {"recons": recons}
        if isinstance(emb_loss, torch.Tensor):
            loss["commitment"] = emb_loss
        else:
            loss = {**loss, **emb_loss}
        
        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return out, total_loss, loss_dict


if __name__ == "__main__":
    import munch
    # vae = False
    # motion_enc = {
    #     # "nfeats": 69,
    #     # "vae": False,
    #     "cls": "TransformerEncoder",
    #     "vae": vae
    # }
    # motion_dec = {
    #     "cls": "TransformerDecoder",
    #     "vae": vae
    # }
    # lambdas = {
    #     "recons": 1.0,
    #     "kl": 0.0001
    #     # "commitment": 1.0,
    # }

    # vq = None
    # vq = {
    #     "nb_code": 1024,
    #     "code_dim": 64,
    #     "mu": 0.99,
    #     "beta": 0.25
    # }
    # nfeats = 69
    # latent_dim = 64
    
    # # model = SingleModalityVAE(
    # #     nfeats, latent_dim,
    # #     motion_enc, motion_dec, lambdas, vae=vae, vq=vq)
    
    # enc_config = {
    #     "top": {"down_t": 2, "stride_t": 2},
    #     "bottom": {"down_t": 2, "stride_t": 2}
    # }
    # dec_config = {        
    #     "top": {"down_t": 2},
    #     "bottom": {"down_t": 2}
    # }
    # lambdas = {
    #     "recons": 1.0,
    #     "commitment": 1.0,
    # }
    # vq = {
    #     "name": "QuantizeEMAReset",
    #     "top": {
    #         "nb_code": 256,
    #         "code_dim": latent_dim*2,
    #         "beta": 0.25,
    #         "mu": 0.99,
    #     },
    #     "bottom": {
    #         "nb_code": 512,
    #         "code_dim": latent_dim,
    #         "beta": 0.25,
    #         "mu": 0.99,
    #     }
    # }
    # vq = {
    #     "name": "QuantizeEMAReset",
    #     "nb_code": 256,
    #     "code_dim": latent_dim,
    #     "beta": 0.25,
    #     "mu": 0.99,
    # }
    # vq = munch.munchify(vq)
    
    # model = HierarchicalVQVAE(
    #     nfeats, latent_dim,
    #     enc_config,
    #     dec_config,
    #     lambdas,
    #     vq
    # )
    # model = MultiVQVAE(nfeats, latent_dim, 2, lambdas, vq)

    
    # motion_seq = torch.randn(2, 256, 69)
    # batch = {
    #     "motion": motion_seq,
    #     "length": [motion_seq.shape[1]] * motion_seq.shape[0],
    #     "ref": motion_seq.clone()
    # }
    # out = model(batch)
    # try:
    #     print(out[0].shape)
    # except:
    #     for item in out[0]:
    #         print(item.shape)
    #     print(out[-1].keys())
    
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    
    # config = munch.munchify({})
    # config.latent_dim = 64
    # config.observation_dim = 69
    
    # config.n_embd = 128
    # config.bias = True
    # config.dropout = 0.1
    # config.n_layer = 4
    # config.n_head = 4
    # config.embd_pdrop = 0.1
    # config.trajectory_length = 256
    # config.block_size = 256
    # config.latent_step = 32
    # config.state_conditional = False
    # config.vq = {
    #     "name": "QuantizeEMAReset",
    #     "nb_code": 36,
    #     "code_dim": config.latent_dim,
    #     "beta": 0.25,
    #     "mu": 0.99,
    # }

    # model = TransformerStep(config)
    bs = 10
    T = 256
    nfeats = 69
    
    config = munch.munchify({})
    config.latent_dim = 32
    config.vq = {
        "name": "QuantizeEMAReset",
        "num_codes": [32, 48],
        "code_dim": config.latent_dim,
        "beta": 0.25,
        "mu": 0.99,
    }
    config.lambdas = {
        "recons": 1.0,
        "commitment": 0.02
    }
    
    # inputs = torch.randn(bs, config.trajectory_length, config.observation_dim)
    inputs = torch.randn(bs, T, nfeats)
    batch = {"motion": inputs}
    model = VQVAE2(config, nfeats)
    print("inputs", inputs.shape)
    out = model(batch)
    print(out[0].shape)
    print(out[1:])
    model.decode_latent(0, 0)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))