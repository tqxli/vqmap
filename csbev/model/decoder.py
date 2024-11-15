from typing import Dict, Union
import torch
import torch.nn as nn

from csbev.model.resnet import Resnet1D


class Conv1DDecoder(nn.Module):
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
        input_kernel_size: int = 3,
        out_kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        if isinstance(strides, int):
            strides = [strides] * n_ds
        else:
            assert len(strides) == n_ds

        self.input_proc = nn.Conv1d(in_channels, hidden_dim, input_kernel_size, 1, 1)

        layers = []
        for i in range(n_ds):
            layers.append(
                Conv1DDecoderBlock(
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    stride=strides[i],
                    depth=depth,
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                    out_kernel_size=out_kernel_size,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Conv1d(
            hidden_dim, out_channels, kernel_size=out_kernel_size, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.input_proc(x)
        x = self.backbone(x)
        x = self.out(x)
        return x


class Conv1DDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        stride: int = 2,
        depth: int = 1,
        dilation: int = 3,
        activation: str = "relu",
        normalization: str = "LN",
        out_kernel_size: int = 3,
    ):
        super().__init__()

        self.res_block = Resnet1D(
            in_channels,
            depth,
            dilation,
            reverse_dilation=True,
            activation=activation,
            norm=normalization,
        )

        self.upsamp = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="nearest"),
            nn.Conv1d(hidden_dim, hidden_dim, out_kernel_size, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        x = self.upsamp(x)
        return x


class MODecoder(nn.Module):
    def __init__(
        self,
        decoder_shared: nn.Module,
        out_channels: Dict[str, int],
        out_datadim: int = 3,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.decoder_shared = decoder_shared
        self.hidden_dim = decoder_shared.out_channels

        output_layers = {}
        for tag, n_chan_out in out_channels.items():
            output_layers[tag] = nn.Conv1d(
                self.hidden_dim, n_chan_out * out_datadim, 1, 1
            )
        self.output_layers = nn.ModuleDict(output_layers)

    def forward(self, x: torch.Tensor, batch_tag: str):
        """_summary_

        Args:
            x (torch.Tensor): [B, D, T]
            batch_tag (str): indicator for output layer

        Returns:
            torch.Tensor: [B, C, T]
        """
        x = self.decoder_shared(x)
        x = self.output_layers[batch_tag](x)
        return x


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from csbev.utils.run import count_parameters

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/decoder/decoder_tcn.yaml")

    cfg.model.decoder.decoder_shared.depth = 1

    model = instantiate(cfg.model.decoder)
    print(f"Model size: {count_parameters(model):.2f}M")

    bs = 5
    latent_dim = 64
    T = 256 // 2 ** 3

    inputs = torch.randn(bs, latent_dim, T)
    print(inputs.shape)
    x = model(inputs, "rat23")
    print(x.shape)
    x = model(inputs, "mouse_demo")
    print(x.shape)
