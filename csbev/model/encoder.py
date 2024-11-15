from __future__ import annotations
from typing import Dict, Union, List
import torch
import torch.nn as nn

from csbev.model.resnet import Resnet1D
from csbev.model.attention import SetAttentiveBlock


class Conv1DEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        stride: int = 2,
        depth: int = 1,
        dilation: int = 3,
        activation: str = "relu",
        normalization: str = "LN",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ds = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2 if stride % 2 == 0 else stride // 2 + 1,
        )
        self.res_block = Resnet1D(
            hidden_dim, depth, dilation, activation=activation, norm=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ds(x)
        x = self.res_block(x)
        return x


class Conv1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_ds: int = 3,
        strides: Union[list, int] = 2,
        depth: int = 1,
        dilation: int = 3,
        activation: str = "relu",
        normalization: str = "LN",
        out_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.n_ds = n_ds

        if isinstance(strides, int):
            strides = [strides] * n_ds
        else:
            assert len(strides) == n_ds

        layers = []
        for i in range(n_ds):
            layers.append(
                Conv1DEncoderBlock(
                    in_channels=in_channels if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    stride=strides[i],
                    depth=depth,
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Conv1d(
            hidden_dim, out_channels, kernel_size=out_kernel_size, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor, pe_indices=None) -> torch.Tensor:
        bs, n_points, D_in, T = x.shape
        x = x.reshape(bs, n_points*D_in, T)
        x = self.backbone(x)
        x = self.out(x)
        return x


class ChannelInvariantEncoder(nn.Module):
    def __init__(
        self,
        channel_encoder: Conv1DEncoder,
        num_heads: int = 2,
        query_size: int | List[int] = 8,
        chan_mix_last: bool = False,
        num_points: int | None = None,
        latent_dim: int = 64,
        use_be: bool = False,
    ):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.data_dim = channel_encoder.in_channels
        self.chan_out_dim = channel_encoder.out_channels
        self.n_blocks = channel_encoder.n_ds

        if isinstance(query_size, int):
            query_size = [query_size] * self.n_blocks
        self.query_size = query_size

        attention_blocks = []
        for i in range(self.n_blocks):
            block = self.channel_encoder.backbone[i]
            chan_out_dim = block.ds.out_channels
            attention_blocks.append(
                SetAttentiveBlock(
                    dim_in=chan_out_dim,
                    dim_out=chan_out_dim,
                    num_heads=num_heads,
                    num_inds=query_size[i],
                )
            )
        self.attention_blocks = nn.Sequential(*attention_blocks)

        self.latent_dim = latent_dim
        self.chan_mix_last = chan_mix_last and num_points is not None
        if self.chan_mix_last:
            n_channels_out = num_points * self.chan_out_dim
        else:
            n_channels_out = query_size[-1] * self.chan_out_dim
        self.fc = nn.Linear(n_channels_out, latent_dim)

        self.use_be = use_be
        if use_be:
            self.be = nn.Embedding(6, channel_encoder.hidden_dim)

    def _forward(self, x: torch.Tensor, pe_indices=None, **kwargs) -> torch.Tensor:
        """Process sets of data points in a channel-invariant manner

        Args:
            x (torch.Tensor): [B, n_points, D, T], D is the data dimension (e.g., 3 for 3D Cartesian coordinates)

        Returns:
            out: torch.Tensor: [B, n_points, D_out, T']
            hidden_feats: List[torch.Tensor], [B*T', query_size, D]
        """
        bs, n_points, D_in, T = x.shape
        x = x.reshape(bs * n_points, D_in, T)

        query_feats = []
        for i in range(self.n_blocks):
            x = self.channel_encoder.backbone[i](x)  # --> [B*N, D, T//ds]

            x = x.reshape(bs, n_points, *x.shape[1:])  # --> [B, N, D, T//ds=T']
            x = x.permute(0, 3, 1, 2).flatten(0, 1)  # --> [B*T', N, D]

            x, h, attn1, attn2 = self.attention_blocks[i](x)

            query_feats.append(
                h.reshape(bs, -1, *h.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
            )  # --> [B*M, D, T']
            x = (
                x.reshape(bs, -1, *x.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
            )  # [B*T', D, N] --> [B*N, D, T']

            if i == 0 and self.use_be and pe_indices is not None:
                pe = self.be(pe_indices.to(x.device)).unsqueeze(-1)
                x = x.reshape(bs, -1, *x.shape[1:])
                x += pe
                x = x.flatten(0, 1)

        out = x if self.chan_mix_last else query_feats[-1]
        out = self.channel_encoder.out(out)
        out = out.reshape(bs, -1, *out.shape[1:])  # [B, N, D_out, T']
        out = out.flatten(1, 2)
        return out

    def forward(self, x: torch.Tensor, pe_indices=None, **kwargs):
        out = self._forward(x, pe_indices, **kwargs)  # --> [B, N, D_out, T']

        out = self.fc(out.permute(0, 2, 1))  # --> [B, T', D_latent]
        out = out.permute(0, 2, 1)

        return out

    def __str__(self):
        return "CI_MIEncoder"


class MIEncoder(nn.Module):
    def __init__(
        self,
        encoder_shared: nn.Module,
        in_channels: Dict[str, int],
        in_datadim: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_shared = encoder_shared
        self.hidden_dim = encoder_shared.in_channels

        input_layers = {}
        for tag, n_chan_in in in_channels.items():
            input_layers[tag] = nn.Conv1d(
                n_chan_in * in_datadim, self.hidden_dim, 1, 1
            )
        self.input_layers = nn.ModuleDict(input_layers)
    
    def forward(self, x: torch.Tensor, batch_tag: str, **kwargs) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): [B, D, 3, T]
            batch_tag (str): indicator for output layer

        Returns:
            torch.Tensor: [B, C, T]
        """
        x = x.flatten(1, 2)
        x = self.input_layers[batch_tag](x)
        x = x.unsqueeze(2) # [B, C, 1, T] for compatibility only
        x = self.encoder_shared(x)
        return x
    
    def __str__(self):
        return "CM_MIEncoder"


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from csbev.utils.run import count_parameters

    # channel-mixing encoder
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/encoder/encoder_tcn.yaml")

    model = instantiate(cfg.model.encoder)
    print(f"Model {str(model)} size: {count_parameters(model):.2f}M")

    bs = 5
    data_dim = 3
    T = 256
    
    in_channels = cfg.model.encoder.in_channels
    for tag, n_chan_in in in_channels.items():
        inputs = torch.randn(bs, n_chan_in, data_dim, T)
        x = model(inputs, tag)
        print(inputs.shape, x.shape)

    # channel-invariant encoder
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/encoder/encoder_ci.yaml")

    model = instantiate(cfg.model.encoder)
    print(f"Model {str(model)} size: {count_parameters(model):.2f}M")
    
    for n_points in [20, 23]:
        inputs = torch.randn(bs, n_points, data_dim, T)
        x = model(inputs, None)
        print(inputs.shape, x.shape)
    
    