import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from vqmap.models.transformer import PositionalEncoding
from vqmap.utils.data import lengths_to_mask
from vqmap.models.resnet import Resnet1D

class TransformerEncoder(nn.Module):
    def __init__(self,
                 nfeats: int, vae: bool = True,
                 action_cond = False, num_classes=13, 
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()

        input_feats = nfeats
        self.embedding = nn.Linear(input_feats, latent_dim)

        self.vae = vae
        self.action_cond = action_cond
        if vae:
            if action_cond:
                self.num_classes = num_classes
                self.mu_token = nn.Parameter(torch.randn(self.num_classes, latent_dim))
                self.logvar_token = nn.Parameter(torch.randn(self.num_classes, latent_dim))
            else:
                self.mu_token = nn.Parameter(torch.randn(latent_dim))
                self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None, actions = None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.vae:
            if self.action_cond:
                # adding the mu and sigma queries
                assert actions is not None
                mu_token = self.mu_token[actions]
                logvar_token = self.logvar_token[actions]
            else:
                mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
                logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]
        

class Conv1DEncoder(nn.Module):
    def __init__(self, 
                 nfeats = 3,
                 latent_dim = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 drop_last=False,
                 **kwargs,
                 ):
        super().__init__()
        
        self.width = width
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(nfeats, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        if not drop_last:
            blocks.append(nn.Conv1d(width, latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x: [B, ]
        """
        x = x.permute(0, 2, 1)
        x = self.model(x)
        return x


class Conv1DEncoderBackbone(nn.Module):
    def __init__(self, 
                 latent_dim = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 **kwargs,
                 ):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x: [B, ]
        """
        # x = x.permute(0, 2, 1)
        x = self.model(x)
        return x

from vqmap.models.gpt import Block

if __name__ == "__main__":
    model = TransformerEncoder(69)
    inputs = torch.randn(1, 60, 69)
    outputs = model(inputs)
