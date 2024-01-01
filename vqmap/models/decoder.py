import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional
from torch import nn, Tensor

from vqmap.models.transformer import PositionalEncoding
from vqmap.utils.data import lengths_to_mask
from vqmap.models.resnet import Resnet1D


class TransformerDecoder(nn.Module):
    def __init__(self, nfeats: int,
                 action_cond = False, num_classes = 13, 
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)
        
        self.action_cond = action_cond
        if action_cond:
            self.actionBiases = nn.Parameter(torch.randn(num_classes, latent_dim))

    def forward(self, z: Tensor, lengths: List[int], actions=None):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        if self.action_cond:
            z = z + self.actionBiases[actions]
        z = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats


class Conv1DDecoder(nn.Module):
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
                 **kwargs,):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(latent_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, nfeats, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # (bs, Jx3, T)
        return self.model(x).permute(0, 2, 1)

class Conv1DDecoderBackbone(nn.Module):
    def __init__(self,
                #  nfeats = 3,
                 latent_dim = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 **kwargs,):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(latent_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        # blocks.append(nn.Conv1d(width, nfeats, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # (bs, Jx3, T)
        return self.model(x)#.permute(0, 2, 1)
    