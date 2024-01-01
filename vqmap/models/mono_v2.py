import torch
import torch.nn as nn
import torch.nn.functional as F
from vqmap.models.single_modal import get_encdec, SingleModalityVAE


class MONO_v2(nn.Module):
    def __init__(
        self, nfeats_mo, latent_dim_mo,
        nfeats_no, latent_dim_no,
        motion_enc, neuro_enc, motion_dec,
        lambdas, vq
    ):
        super().__init__()
        
        self.motion_encdec = SingleModalityVAE(
            nfeats_mo, latent_dim_mo,
            motion_enc, motion_dec, lambdas, vq=vq,
            modality='motion'
        )
        self.motion_encdec.requires_grad_(False)
        self.neuro_enc = get_encdec(
            neuro_enc, nfeats_no, latent_dim_no, False, 'Encoder'
        )
        n_codes = vq.nb_code
        self.weight_proj = nn.Linear(latent_dim_no, n_codes)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        self.lambdas = lambdas
        
    def forward(self, batch):
        # encode motion data
        # z_m, emb_loss, info = self.motion_encdec.encode(batch)
        # k = info[-1]

        # encode neural data
        z_n = self.neuro_enc(batch["neural"])
        z_n = z_n.permute(0, 2, 1)        
        z_n = self.weight_proj(z_n)

        N, T, C = z_n.shape
        logits = F.gumbel_softmax(z_n, hard=True) #[N, T, C]
        logits = logits.view(-1, C)

        # indexing codebooks
        z_n = logits @ self.motion_encdec.quantizer.codebook
        z_n = z_n.view(N, T, -1).permute(0, 2, 1)
        
        out = self.motion_encdec.decoder(z_n)
        
        # compute losses
        recons = F.mse_loss(out, batch["motion"])

        # ce = F.cross_entropy(logits, k)
        loss = {
            "recons": recons,
            # "commitment": emb_loss,
            # "crossentropy": ce,
        }

        total_loss = self.compute_weighted_loss(loss)
        
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return out, total_loss, loss_dict

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)
        
        return sum(total_loss)

        
if __name__ == "__main__":
    import os
    from copy import deepcopy
    from vqmap.config.config import parse_config
    
    ckpt_path = 'experiments/neural_motion_coembed/motion_embed_vq_n128_d64_ds16/model_last.pth'
    config_path = os.path.join(os.path.dirname(ckpt_path), 'parameters.yaml')
    config_motion = parse_config(config_path)["model"]
    
    nfeats = 69
    latent_dim = config_motion["latent_dim"]
    motion_enc, motion_dec = config_motion["motion_encoder"], config_motion["motion_decoder"]
    vq = config_motion["vq"]
    
    neuro_enc = deepcopy(motion_enc)
    
    lambdas = {
        "recons": 1.0,
        "commitment": 0.02,
        "crossentropy": 1.0,
    }
    
    model = MONO_v2(
        nfeats, latent_dim,
        96, 256,
        motion_enc, neuro_enc, motion_dec,
        lambdas=lambdas, vq=vq
    )
    model.motion_encdec.load_state_dict(
        torch.load(ckpt_path)["model"]
    )
    print("Loaded pretrained motion VQVAE")
    
    B = 4
    T = 64
    motion = torch.randn(B, T, 69)
    neural = torch.randn(B, T, 96)
    batch = {
        "motion": motion,
        "neural": neural
    }
    out, total_loss, loss_dict = model(batch)
    print(out.shape, total_loss)