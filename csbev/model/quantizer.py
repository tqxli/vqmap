from __future__ import annotations
import random
from typing import Dict, List, Optional
from itertools import product
from einops import repeat
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


class GumbelVectorQuantizer(nn.Module):
    """
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/gumbel_vector_quantizer.py
    """
    def __init__(
        self,
        nb_code,
        code_dim,
        input_dim=None,
        temp=(2, 0.5, 0.999995),
        time_first=False,
        groups=1,
        combine_groups=True,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        hard=True,
        std=0,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.input_dim = input_dim if input_dim is not None else code_dim
        self.groups = groups
        self.combine_groups = combine_groups
        self.nb_code = nb_code
        self.time_first = time_first
        self.hard = hard

        assert (
            code_dim % groups == 0
        ), f"dim {code_dim} must be divisible by groups {groups} for concatenation"

        var_dim = code_dim // groups
        self.n_groups = num_groups = groups #if not combine_groups else 1

        self.codebook = nn.Parameter(torch.FloatTensor(1, num_groups * nb_code, var_dim))
        if std == 0:
            nn.init.uniform_(self.codebook)
        else:
            nn.init.normal_(self.codebook, mean=0, std=std)

        self.weight_proj_depth = weight_proj_depth
        self.weight_proj_factor = weight_proj_factor
        self.activation = activation

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * nb_code),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * nb_code)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_codes)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.codebook.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_codes**self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_codes * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    # def codebook(self):
    #     indices = self.get_codebook_indices()
    #     return (
    #         self.codebook.squeeze(0)
    #         .index_select(0, indices)
    #         .view(self.num_codes**self.groups, -1)
    #     )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.codebook.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_codes**exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):
        result = {"num_vars": self.nb_code * self.n_groups}
        
        bsz, fsz, tsz = x.shape
        x = x.transpose(1, 2)  # BTC -> BCT 

        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        with torch.no_grad():
            _, k = x.max(-1)
            hard_x = (
                x.new_zeros(*x.shape)
                .scatter_(-1, k.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
            )
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(
                x
            )
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.codebook
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.nb_code, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        x = x.transpose(1, 2)  # BTC -> BCT

        emb_loss = {
            "code_penality": (result["num_vars"]-result["prob_perplexity"]) / result["num_vars"]
        }

        return x, emb_loss, result


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

    def set_num_updates(self, num_updates):
        for sub_quantizer in self.sub_quantizers:
            if hasattr(sub_quantizer, "set_num_updates"):
                sub_quantizer.set_num_updates(num_updates)

    def get_codevecs(self):
        combinations = product(*[range(nb_code) for nb_code in self.codebook_size])
        combinations = [combo for combo in combinations]
        
        # extract codebook
        if isinstance(self.sub_quantizers[0], ResidualVQ):
            codebooks = [sub_quantizer.codebooks[0] for sub_quantizer in self.sub_quantizers] #[num_cb, num_code, dim]
        else:
            codebooks = [sub_quantizer.codebook for sub_quantizer in self.sub_quantizers]

        codevecs = [
            torch.cat(
                [
                    codebooks[i][combo[i]]
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

    def quantize(self, x: torch.Tensor):
        x = torch.split(x, self.code_dim, dim=1)

        x_q = []
        for i in range(self.n_groups):
            _, q_res = self.sub_quantizers[i].quantize(x[i], return_latent=True)
            q_cumsum = torch.cumsum(q_res, dim=0) #[n_res_cb, B, D, T]
            x_q.append(q_cumsum)
        x_q = torch.cat(x_q, dim=2)
        return x_q


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
        

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout
        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed
        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_infos = []
        # go through the layers
        for quantizer_index, layer in enumerate(self.layers):
            quantized, loss, info = layer(residual)
            # don't use += or -= which modifies the tensor in place, causing gradient issues
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_infos.append(info)

        # stack all losses and indices
        all_losses = self._merge_loss_dict(all_losses)

        return quantized_out, all_losses, all_infos[0]

    def _merge_loss_dict(self, losses):
        loss = {}
        for k in losses[0]:
            loss[k] = sum([loss_dict[k] for loss_dict in losses])
        return loss
    
    def quantize(self, x, return_latent=False):
        all_infos = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):
            quantized, loss, info = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_infos.append(info)
            all_codes.append(quantized)

        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return all_infos, all_codes
        return all_infos


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from csbev.utils.run import count_parameters

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/bottleneck/quantizer_mg_res.yaml")

    bottleneck = instantiate(cfg.model.bottleneck)

    inputs = torch.randn(5, 64, 16)
    z, loss, info = bottleneck(inputs)
    print(z.shape)
    print(loss)
    # for k, v in info[0].items():
    #     if isinstance(v, torch.Tensor):
    #         print(k, v.shape)
    #     elif isinstance(v, list):
    #         print(k, len(v))
    #     else:
    #         print(k, v)
