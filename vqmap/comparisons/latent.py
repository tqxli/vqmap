import torch
import torch.nn as nn

from vqmap.comparisons.gru import Encoder_GRU, Decoder_GRU
from vqmap.comparisons.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from vqmap.comparisons.fc import Encoder_FC, Decoder_FC
from vqmap.comparisons.losses import get_loss_function

class LatentDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        self.device = device
        
        self.losses = list(self.lambdas) + ["mixed"]

    def _data_preproc(self, batch):
        x = batch["motion"]
        z = batch["action"]
        z = torch.stack(z, 0).to(x.device)
        bs, t, nfeats = z.shape

        lengths = (torch.ones(bs) * t).long().to(z.device)
        mask = lengths_to_mask(lengths).to(z.device)

        batch["motion"] = x.reshape(x.shape[0], x.shape[1], -1, 3).permute(0, 2, 3, 1)
        batch["x"] = z.permute(1, 0, 2)
        batch["lengths"] = lengths
        batch["mask"] = mask
        
        lengths = (torch.ones(x.shape[0]) * x.shape[1]).long().to(z.device)
        mask = lengths_to_mask(lengths).to(z.device)
        batch['motion_mask'] = mask

        return batch

    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        batch = self._data_preproc(batch)

        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)
        
        batch["mask"] = batch["motion_mask"]
        # decode
        batch.update(self.decoder(batch))
        batch["x"] = batch["motion"]
        mixed_loss, losses = self.compute_loss(batch)
        out = batch["output"].permute(0, 3, 1, 2)
        out = out.reshape(*out.shape[:2], -1)
        return out, mixed_loss, losses

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)

    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        # losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate(self, duration, fact=1):
        # y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        out = batch["output"][0]
        return out.permute(2, 0, 1)

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def get_latentvae(architecture, nfeats, latent_dim, lambdas, num_frames=64):
    encoder_name = f"Encoder_{architecture}"
    decoder_name = f"Decoder_{architecture}"
    encoder_cls = globals()[encoder_name]
    decoder_cls = globals()[decoder_name]
    
    n_dim = 3
    n_joints = nfeats // n_dim
    encoder = encoder_cls(
        'Encoder', latent_dim, 1, num_frames//16, latent_dim=latent_dim
    )

    decoder = decoder_cls(
        'Decoder', n_joints, n_dim, num_frames, latent_dim=latent_dim
    )
    model = LatentDecoder(
        encoder=encoder, decoder=decoder,
        device='cuda', lambdas=lambdas, latent_dim=latent_dim
    )
    return model

if __name__ == "__main__":

    n_joints = 23
    nfeats = 3
    num_frames = 64
    latent_dim = 256
    
    decoder = Decoder_TRANSFORMER(
        'Decoder', n_joints, nfeats, num_frames, latent_dim=latent_dim
    )
    bs = 10

    batch = {
        "motion": torch.randn(bs, num_frames, n_joints*nfeats),
        "latent": torch.randn(bs, num_frames//16, latent_dim),
    }

    lambdas = {
        "recons": 1.0,
    }
    model = LatentDecoder(
        decoder=decoder,
        device='cpu', lambdas=lambdas, latent_dim=latent_dim
    )
    print("input: ", batch["motion"].shape)
    batch = model(batch)
    for v in batch:
        try:
            print(v.shape)
        except:
            print(v)
    
    gen = model.generate(64)
    print(gen.shape)