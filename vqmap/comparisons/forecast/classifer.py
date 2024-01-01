import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):
    def __init__(self, n_embd, n_cls, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj  = nn.Linear(4 * n_embd, n_cls, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class ForecastNet(nn.Module):
    def __init__(self, input_feats, n_cls,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 activation="gelu", **kargs):
        super().__init__()
        
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        self.n_cls = n_cls
        self.classifier = MLP(latent_dim, n_cls)

    def forward(self, batch):
        batch = self._data_preproc(batch)
        x, mask = batch["x"], batch["mask"]
        targets = batch["y"]
        x = x.permute((1, 0, 2))
        
        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # # only for ablation / not used in the final model
        # add positional encoding
        x = self.sequence_pos_encoder(x)
        
        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # get the average of the output
        z = final.mean(axis=0)

        logits = self.classifier(z)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        lossdict = {"ce_loss": loss.item()}
        return logits, loss, lossdict

    def _data_preproc(self, batch):
        x = batch["motion"]
        bs, t, nfeats = x.shape
        lengths = (torch.ones(bs) * t).long().to(x.device)
        mask = lengths_to_mask(lengths).to(x.device)
        batch["x"] = x
        batch["lengths"] = lengths
        batch["mask"] = mask
        
        return batch

def lengths_to_mask(lengths):
    max_len = max(lengths)
    if isinstance(max_len, torch.Tensor):
        max_len = max_len.item()
    index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
    mask = index < lengths.unsqueeze(1)
    return mask
    
if __name__ == "__main__":
    latent_dim = 64
    n_cls = 12
    inputs = torch.randn(10, 16, 1)
    targets = torch.randint(0, 12, (10,))
    model = ForecastNet(1, n_cls, latent_dim)
    
    batch = {
        "x": inputs,
        "y": targets,
        "mask": torch.ones(*inputs.shape[:2]).long()
    }
    
    logits = model(batch)
    print(logits.shape)