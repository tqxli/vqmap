import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(edgeitems=5)

        
class DenoiseAutoencoder(nn.Module):
    def __init__(self, nfeats, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(nfeats, latent_dim//4,4,stride=4),
            nn.BatchNorm1d(latent_dim//4),
            nn.ReLU(),
            nn.Conv1d(latent_dim//4,latent_dim//2,3,dilation=2),
            nn.BatchNorm1d(latent_dim//2),
            nn.ReLU(),
            nn.Conv1d(latent_dim//2,latent_dim,2),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim,latent_dim//2,2),
            nn.BatchNorm1d(latent_dim//2),
            nn.ReLU(),
            nn.ConvTranspose1d(latent_dim//2,latent_dim//4,3,dilation=2),
            nn.BatchNorm1d(latent_dim//4),
            nn.ReLU(),
            nn.ConvTranspose1d(latent_dim//4,nfeats,4,stride=4)
            )

    def forward(self, batch):
        x = batch["motion"]
        latent_code = self.encoder(x.permute(0, 2, 1))
        x_recons = self.decoder(latent_code).permute(0, 2, 1)
        
        loss = F.mse_loss(x_recons, batch["ref"])
        loss_dict = {"recons": loss.item()}
        
        return x_recons, loss, loss_dict

    def latent_return(self, batch):
        try:
            x = batch["motion"]
        except:
            x = batch
        latent_code = self.encoder(x.permute(0, 2, 1))
        return latent_code


if __name__ == "__main__":
    inputs = torch.randn(10, 64, 69)
    model = DenoiseAutoencoder(69)
    out, z = model({"motion": inputs})
    print(out.shape)
    print(z.shape)