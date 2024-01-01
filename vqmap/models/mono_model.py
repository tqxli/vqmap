import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from typing import List, Optional
from vqmap.losses.kl import KLLoss

from vqmap.models.single_modal import get_encdec

class MONO(nn.Module):
    def __init__(self,
        nfeats, latent_dim,
        motion_enc, neuro_enc, motion_dec,
        vae=True,
        lambdas={},
    ):
        super().__init__()
        
        self.vae = vae

        self.motion_encoder = get_encdec(
            motion_enc, nfeats, latent_dim, vae, 'Encoder'
        )
        self.neuro_encoder = get_encdec(
            neuro_enc, nfeats, latent_dim, vae, 'Encoder'
        )
        self.motion_decoder = get_encdec(
            motion_dec, nfeats, latent_dim, vae, 'Decoder'
        )
        
        self.sample_mean = False
        self.fact = None
        self.lambdas = lambdas
        
        self._init_loss_funcs()
    
    # def forward(self, motion_seq, neural):
    #     motion_features= self.motion_encoder(motion_seq)
    #     neuro_features = self.neuro_encoder(neural)
        
    #     return motion_features, neuro_features

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

    def neuro_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.vae:
            distribution = self.neuro_encoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.neuro_encoder(text_sentences)

        # Decode the latent vector to a motion
        out = self.motion_decoder(latent_vector, lengths)

        if not return_latent:
            return out
        return out, latent_vector, distribution

    def motion_to_motion_forward(self, features,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):

        # Encode the motion to the latent space
        if self.vae:
            distribution = self.motion_encoder(features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motion_encoder(features, lengths)

        # Decode the latent vector to a motion
        out = self.motion_decoder(latent_vector, lengths)

        if not return_latent:
            return out
        return out, latent_vector, distribution
    
    def forward(self, batch):
        ret = self.neuro_to_motion_forward(batch["neural"],
                                          batch["length"],
                                          return_latent=True)
        neural_out, latent_from_neural, distribution_from_neural = ret


        ret = self.motion_to_motion_forward(batch["motion"],
                                            batch["length"],
                                            return_latent=True)
        motion_out, latent_from_motion, distribution_from_motion = ret

        # GT data
        motion_ref = batch["ref"]

        # Compare to a Normal distribution
        if self.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_neural.loc)
            scale_ref = torch.ones_like(distribution_from_neural.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None

        # Compute the losses
        loss = self.compute_losses(
            neural_out,
            motion_out,
            motion_ref,
            latent_from_neural,
            latent_from_motion,
            distribution_from_neural,
            distribution_from_motion,
            distribution_ref)

        total_loss = self.get_weighted_loss(loss)

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return total_loss, loss_dict

    def _init_loss_funcs(self):
        self.recons_loss = nn.L1Loss()
        self.sim_loss = nn.L1Loss()
        self.kl_loss = KLLoss()

    def compute_losses(self, 
                       neural_out, motion_out, motion_ref,
                       latent_from_neural, latent_from_motion,
                       dis_neural, dis_motion, dis_ref,
        ):

        recons_neural2mo = self.recons_loss(neural_out, motion_ref)
        recons_mo2mo = self.recons_loss(motion_out, motion_ref)
        latent_sim = self.sim_loss(latent_from_neural, latent_from_motion)
        kl_neural = self.kl_loss(dis_neural, dis_ref)
        kl_mo = self.kl_loss(dis_motion, dis_ref)
        kl_neural2mo = self.kl_loss(dis_neural, dis_motion)
        kl_mo2neural = self.kl_loss(dis_motion, dis_neural)
        
        return {
            "recons_neural2mo": recons_neural2mo,
            "recons_mo2mo": recons_mo2mo,
            "latent_sim": latent_sim,
            "kl_neural": kl_neural,
            "kl_mo": kl_mo,
            "kl_neural2mo": kl_neural2mo,
            "kl_mo2neural": kl_mo2neural
        }
        
    def get_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas.get(key, 1.0) * lossval)
        
        return sum(total_loss)


if __name__ == "__main__":
    motion_enc = {
        "nfeats": 69
    }
    neuro_enc = {
        "nfeats": 96
    }
    lambdas = {
        "recons_neural2mo": 1.0,
        "recons_mo2mo": 1.0,
        "latent_sim": 1e-5,
        "kl_neural": 1e-5,
        "kl_mo": 1e-5,
        "kl_neural2mo": 1e-5,
        "kl_mo2neural": 1e-5,
    }
    
    model = MONO(motion_enc, neuro_enc, motion_enc, True, lambdas)
    motion_seq = torch.randn(2, 10, 69)
    neural = torch.randn(2, 10, 96)
    motion_features, neuro_features = model(motion_seq, neural)
    print(motion_features, neuro_features)
    
    lengths = [len(feature) for feature in neural]
    out = model.neuro_to_motion_forward(neural, lengths)
    print(out.shape)

    lengths = [len(feature) for feature in motion_seq]
    out = model.motion_to_motion_forward(motion_seq, lengths)
    print(out.shape)
    
    ref = torch.randn_like(motion_seq)
    batch = {
        "neural": neural,
        "motion": motion_seq,
        "length": lengths,
        "ref": ref
    }
    total_loss, loss = model.translation_step(batch)
    for k, v in loss.items():
        print(k, v)
    print(total_loss)