from __future__ import annotations
from typing import Dict, List, Optional
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.distribution import Distribution
from omegaconf import OmegaConf


class VariationalBottleneck(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        min_capacity: float = 0.0,
        max_capacity: float = 0.5,
        max_iters: int = 5000,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.sample_mean, self.fact = None, None
        
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.max_iters = max_iters

    def forward(self, latent: torch.Tensor, cur_iters: int = 1):
        """Reparametrization for VAE

        Args:
            latent (torch.Tensor): [B, D, T]
        """
        latent = latent.permute(0, 2, 1)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        std = logvar.exp().pow(0.5)
        distribution = torch.distributions.Normal(mu, std)

        kl = self.compute_kl(distribution) #[B, T, D]
        
        cap_loss = self.compute_capacity_loss(kl, cur_iters)

        latent = self.sample_from_distribution(distribution)
        latent = latent.permute(0, 2, 1)
        return latent, {"KLDiv": kl.mean(), "capacity_loss": cap_loss}

    def sample_from_distribution(
        self,
        distribution: Distribution,
        *,
        fact: Optional[bool] = None,
        sample_mean: Optional[bool] = False,
    ) -> torch.Tensor:
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
        return latent_vector, distribution

    def compute_kl(self, distribution: Distribution):
        # Create a centred normal distribution to compare with
        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl = torch.distributions.kl_divergence(distribution, distribution_ref)
        return kl.mean()
        # return kl.sum() / kl.shape[0] / kl.shape[1]

    def compute_capacity_loss(
        self,
        kl: torch.Tensor,
        cur_iters: int,
    ):
        cur_capacity = min(self.max_capacity, cur_iters / self.max_iters * self.max_capacity)
        capacity_loss = torch.abs(cur_capacity - kl)
        return capacity_loss


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu, beta, transition=False):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.beta = beta

        self.freeze_codebook = False
        self.reinit_codebook = True
        self.reset_codebook()

        self.transition = transition
        if transition:
            self.init_prob()

    def init_prob(self):
        probs_raw = torch.zeros([self.nb_code, self.nb_code]).exp()
        probs_sum = probs_raw.sum(dim=-1, keepdim=True)
        self.probs = nn.Parameter(probs_raw / probs_sum)

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer(
            "codebook", torch.zeros(self.nb_code, self.code_dim).cuda()
        )

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        if self.reinit_codebook:
            print(f"Codebook initialized!")
            out = self._tile(x)
            self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def _gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def _normalized_probs(self):
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=-1, keepdim=True)
        return probs_pos / probs_sum
        # print(self.probs[0])
        # print(self.probs.requires_grad)

    def compute_transition_loss(self, code_idx, T):
        # NLL for transition probabilities, from SOM-VAE
        k = code_idx.reshape(-1, T)  # [N, T]
        k = k.transpose(1, 0)  # [T, N]
        k_old = torch.cat((k[:1], k[:-1]), axis=0)  # [T, N]
        k_stacked = torch.stack([k_old, k], dim=-1)  # [T, N, 2]
        k_stacked = k_stacked.flatten(0, 1)  # [T*N, 2]
        transitions_all = self._gather_nd(self._normalized_probs(), k_stacked)  # [T*N]
        prob_l = -torch.mean(torch.log(transitions_all))
        return prob_l

    def compute_smoothness_loss(self, code_idx, z_dist, T):
        # nodes w higher probability should be closer
        k = code_idx.reshape(-1, T)  # [N, T]
        k = k.transpose(1, 0)  # [T, N]
        k_old = torch.cat((k[:1], k[:-1]), axis=0)  # [T, N]
        k_stacked = torch.stack([k_old, k], dim=-1)  # [T, N, 2]
        k_stacked = k_stacked.transpose(1, 0).flatten(0, 1)  # [N*T, 2]
        # self._normalize_probs()
        out_prob_old = self._gather_nd(self._normalized_probs(), k_stacked)
        weighted_z_dist_prob = z_dist * out_prob_old.unsqueeze(-1)
        prob_z_l = torch.mean(weighted_z_dist_prob)
        return prob_z_l

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(
            self.nb_code, code_idx.shape[0], device=code_idx.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # print("Called update!")
        code_onehot = torch.zeros(
            self.nb_code, x.shape[0], device=x.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        # Update centres
        self.code_sum = (
            self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        )  # w, nb_code
        self.code_count = (
            self.mu * self.code_count + (1.0 - self.mu) * code_count
        )  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(
            self.nb_code, self.code_dim
        ) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w ** 2, dim=0, keepdim=True)
        )  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx, distance

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx, z_dist = self.quantize(x)

        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training and not self.freeze_codebook:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        loss = {}
        loss["commitment"] = F.mse_loss(x, x_d.detach())
        if self.transition:
            loss["transition"] = self.compute_transition_loss(code_idx, T)
            loss["smoothness"] = self.compute_smoothness_loss(code_idx, z_dist, T)

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, loss, (perplexity, code_idx)


class MultiGroupQuantizer(nn.Module):
    def __init__(
        self,
        sub_quantizer: nn.Module | List[nn.Module],
        n_groups: int = 2,
        codebook_size: List[int] = [8, 16],
        code_dim: List[int] | int = 64,
        additive_codebooks: bool = False,
        latent_correation_loss: bool = False,
    ):
        super().__init__()

        self.n_groups = n_groups
        self.codebook_size = OmegaConf.to_object(codebook_size)
        assert len(self.codebook_size) == n_groups

        self.additive_codebooks = additive_codebooks
        self.latent_correation_loss = latent_correation_loss

        if not isinstance(code_dim, int):
            assert len(code_dim) == n_groups
            self.code_dim = OmegaConf.to_object(code_dim)
        elif isinstance(code_dim, int):
            assert code_dim % n_groups == 0
            self.code_dim = [code_dim // n_groups] * n_groups
        else:
            raise ValueError("Invalid code_dim")

        sub_quantizers = []
        for i in range(n_groups):
            sub_quantizers.append(
                sub_quantizer(code_dim=self.code_dim[i], nb_code=codebook_size[i])
            )
        self.sub_quantizers = nn.ModuleList(sub_quantizers)

    def get_codevecs(self):
        combinations = product(*[range(nb_code) for nb_code in self.codebook_size])
        combinations = [combo for combo in combinations]

        codevecs = [
            torch.cat(
                [
                    self.sub_quantizers[i].codebook[combo[i]]
                    for i in range(self.n_groups)
                ],
                dim=0,
            )
            for combo in combinations
        ]
        codevecs = torch.stack(codevecs, dim=0).unsqueeze(-1)
        return codevecs, combinations

    def _merge_loss_dict(self, losses):
        loss = {}
        for k in losses[0]:
            loss[k] = sum([loss_dict[k] for loss_dict in losses])
        return loss

    def forward(self, x: torch.Tensor):
        # x: [B, D, T]
        x = torch.split(x, self.code_dim, dim=1)

        loss, info = [], []
        x_q = []
        for i in range(self.n_groups):
            x_d, c_loss, inf = self.sub_quantizers[i](x[i])
            x_q.append(x_d)
            loss.append(c_loss)
            info.append(inf)

        loss = self._merge_loss_dict(loss)
        if self.latent_correation_loss:
            assert len(x_q) == 2
            corr = F.cosine_similarity(
                x[0]-x[0].mean(dim=1, keepdim=True),
                x[1]-x[1].mean(dim=1, keepdim=True),
                dim=1,
            ).mean()
            corr_loss = 1 + corr
            loss["correlation"] = corr_loss

        if self.additive_codebooks:
            x_q = sum(x_q)
        else:
            x_q = torch.cat(x_q, dim=1)  # [B, D, T]

        return x_q, loss, info


class MixBottleneck(nn.Module):
    def __init__(
        self,
        code_dim: List[int],
        latent_vars: Dict[nn.Module],
        additive_codebooks: bool = False,
    ):
        super().__init__()

        self.code_dim = OmegaConf.to_object(code_dim)
        self.discrete_var = latent_vars["discrete"](code_dim=self.code_dim[0])
        self.continuous_var = latent_vars["continuous"](latent_dim=self.code_dim[1])

        if additive_codebooks:
            assert self.code_dim[0] == self.code_dim[1]
        self.additive_codebooks = additive_codebooks

    def forward(self, x: torch.Tensor):
        x = torch.split(x, self.code_dim, dim=1)
        x_dis, x_cont = x
        z_dis, loss_dis, info_dis = self.discrete_var(x_dis)
        z_cont, loss_cont = self.continuous_var(x_cont)

        if self.additive_codebooks:
            z = z_dis + z_cont
        else:
            z = torch.cat([z_dis, z_cont], dim=1)
        loss = {**loss_dis, **loss_cont}
        return z, loss, info_dis


class MixBottleneckv2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        code_dim: int,
        latent_vars: Dict[nn.Module],
    ):
        super().__init__()

        self.code_dim = OmegaConf.to_object(code_dim)

        self.l1 = nn.Linear(in_channels, code_dim[0])
        self.l2 = nn.Linear(in_channels, code_dim[1])
        self.discrete_var = latent_vars["discrete"](code_dim=code_dim[0])
        self.continuous_var = latent_vars["continuous"](latent_dim=code_dim[1])

    def forward(self, x: torch.Tensor, cur_iters: int = 1):
        """
        Args:
            x (torch.Tensor): [B, D, T]
        """
        z = x.permute(0, 2, 1)
        
        z_dis = self.l1(z).permute(0, 2, 1)
        z_cont = self.l2(z).permute(0, 2, 1)
        
        z_dis_q, loss_dis, info_dis = self.discrete_var(z_dis)
        z_cont, loss_cont = self.continuous_var(z_cont, cur_iters)
        
        # z = z_dis_q + z_cont
        z = torch.cat([z_dis_q, z_cont], dim=1)
        
        loss = {**loss_dis, **loss_cont}
        return z, loss, info_dis


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/bottleneck/mix_v2.yaml")
    print(cfg)
    bottleneck = instantiate(cfg.model.bottleneck)

    inputs = torch.randn(5, 64, 16)
    z, loss, info = bottleneck(inputs)
    print(z.shape)
    print(loss)
