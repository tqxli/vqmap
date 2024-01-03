## Code adapted from [Esser, Rombach 2021]: https://compvis.github.io/taming-transformers/
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, nb_code, code_dim, mu, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = nb_code #n_e
        self.e_dim = self.code_dim = code_dim #e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        #print('zshape', z.shape)
        # z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        #loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #    torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        #print(min_encodings.shape, self.embedding.weight.shape)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu, beta):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.beta = beta
        
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

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
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

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
        code_idx = self.quantize(x)

        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, (perplexity, code_idx)


class MultiGroupQuantizer(nn.Module):
    def __init__(self, vq_type, args, n_groups):
        super().__init__()

        self.n_groups = n_groups
        self.code_dim = args["code_dim"] // self.n_groups
        args["code_dim"] = self.code_dim
        
        num_codes = args["nb_code"]
        assert len(num_codes) == n_groups
        
        sub_quantizers = []
        for i in range(n_groups):
            sub_args = {"nb_code": num_codes[i]}
            for k, v in args.items():
                if k not in sub_args:
                    sub_args[k] = v
            
            sub_quantizers.append(vq_type(**sub_args))
        
        self.sub_quantizers = nn.ModuleList(sub_quantizers)
    
    def forward(self, x):
        # split into different groups
        x = x.view(x.shape[0], self.n_groups, -1, x.shape[2])
        x = x.transpose(1, 0) #[n_groups, bs, 64, 4]

        commit_loss = []
        info = []
        for i in range(self.n_groups):
            x_d, c_loss, inf = self.sub_quantizers[i](x[i])
            x[i] = x_d
            commit_loss.append(c_loss)
            info.append(inf)
        
        x = x.transpose(1, 0)
        
        x = x.view(x.shape[0], -1, x.shape[3])

        loss = sum(commit_loss)
        # loss = {k: v for k, v in commit_loss[0].items()}
        # for cl in commit_loss[1:]:
        #     for k, v in cl.items():
        #         loss[k] += v
        return x, loss, info
    

class SOMQuantizer(nn.Module):
    def __init__(
        self, code_dim,
        som_dim=[32, 32],
        device='cuda',
        transition=False,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.som_dim = som_dim
        self.nb_code = som_dim[0]*som_dim[1]
        # if ema:
        #     self.register_buffer(
        #         "embeddings", torch.zeros((som_dim[0], som_dim[1], code_dim)).to(device)
        #     )
        # else:
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((som_dim[0], som_dim[1], code_dim)),
                std=0.05, a=-0.1, b=0.1)
        )
        self.device = device
        self.mse_loss = nn.MSELoss()
        
        self.transition = transition

    def forward(self, x):
        N, C, T = x.shape
        z_e = self.preprocess(x) #[NT, C]
        z_q, z_dist, k = self._find_closest_embedding(z_e, batch_size=z_e.shape[0])
        z_q_neighbors = self._find_neighbors(z_q, k, batch_size=z_e.shape[0])
        
        loss = self.loss(z_e, z_q, z_q_neighbors, batch_size=N)
        
        return self.postprocess(z_e, z_q, N, T), loss, (z_q_neighbors, z_dist, k)

    def dequantize(self, k):
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_batch = torch.stack([k_1, k_2], dim=1)
        return self._gather_nd(self.embeddings, k_batch)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x
    
    def postprocess(self, z_e, z_q, N, T):
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.view(N, T, -1)
        z_q = z_q.permute(0, 2, 1).contiguous()
        return z_q

    def _get_coordinates_from_idx(self, k):
        k_1 = torch.div(k, self.som_dim[1], rounding_mode='floor')
        k_2 = k % self.som_dim[1]
        return k_1, k_2

    @staticmethod
    def _gather_nd(params, idx):
        """Similar to tf.gather_nd. Here: returns batch of params given the indices."""
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def _find_closest_embedding(self, z_e, batch_size=32):
        """Picks the closest embedding for every encoding."""
        z_dist = (z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2
        z_dist_sum = torch.sum(z_dist, dim=-1)
        z_dist_flat = z_dist_sum.view(batch_size, -1)
        k = torch.argmin(z_dist_flat, dim=-1)
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_batch = torch.stack([k_1, k_2], dim=1)
        return self._gather_nd(self.embeddings, k_batch), z_dist_flat, k
    
    def _find_neighbors(self, z_q, k, batch_size):
        k_1, k_2 = self._get_coordinates_from_idx(k)

        k1_not_top = k_1 < self.som_dim[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.som_dim[1] - 1
        k2_not_left = k_2 > 0

        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)

        z_q_up = torch.zeros(batch_size, self.code_dim).to(self.device)
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]

        z_q_down = torch.zeros(batch_size, self.code_dim).to(self.device)
        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]

        z_q_right = torch.zeros(batch_size, self.code_dim).to(self.device)
        z_q_right_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] = z_q_right_[k2_not_right == 1]

        z_q_left = torch.zeros(batch_size, self.code_dim).to(self.device)
        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]

        return torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)
    
    def _loss_commit(self, z_e, z_q):
        # commit_l = self.mse_loss(z_e, z_q)
        commit_l = torch.mean((z_q.detach()-z_e)**2) + \
                   torch.mean((z_q - z_e.detach()) ** 2)
        return commit_l

    @staticmethod
    def _loss_som(z_e, z_q_neighbors):
        z_e = z_e.detach()
        som_l = torch.mean((z_e.unsqueeze(1) - z_q_neighbors) ** 2)
        return som_l
    
    def _loss_z_smooth(self, z_q, batch_size, dist='l1'):
        z_q = z_q.view(batch_size, -1, z_q.shape[-1])
        # if dist == 'l1':
        return F.l1_loss(z_q[:, :-1], z_q[:, 1:], reduce='sum') / batch_size
    
    def _loss_smooth(self, k, batch_size):
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_1 = k_1.view(batch_size, -1, *k_1.shape[1:])
        k_2 = k_2.view(batch_size, -1, *k_2.shape[1:])
        return torch.abs(k_1 - k_2).sum() / batch_size

    def loss(self, z_e, z_q, z_q_neighbors, batch_size):
        commit_l = self._loss_commit(z_e, z_q)
        som_l = self._loss_som(z_e, z_q_neighbors)        
        loss = {
            "commitment": commit_l,
            "som": som_l
        }
        if self.transition:
            loss['transition'] = self._loss_z_smooth(z_q, batch_size)
        
        return loss


class GumbelVectorQuantizer(nn.Module):
    """
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/gumbel_vector_quantizer.py
    """
    def __init__(
        self,
        input_dim,
        num_codes,
        code_dim,
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

        self.input_dim = input_dim
        self.groups = groups
        self.combine_groups = combine_groups
        self.num_codes = num_codes
        self.time_first = time_first
        self.hard = hard

        # assert (
        #     vq_dim % groups == 0
        # ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        # var_dim = vq_dim // groups
        var_dim = code_dim
        num_groups = groups #if not combine_groups else 1

        self.codebook = nn.Parameter(torch.FloatTensor(1, num_groups * num_codes, var_dim))
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
                nn.Linear(inner_dim, groups * num_codes),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_codes)
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

        result = {"num_vars": self.num_codes * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
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
        x = x.view(bsz * tsz, self.groups, self.num_codes, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        # result["x"] = x
        emb_loss = {"diversity": (result["num_vars"]-result['prob_perplexity'])/result['num_vars']}

        return x, emb_loss, result


class HierarchicalQuantizer(GumbelVectorQuantizer):
    def __init__(
        self,
        apply_gumbel=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.apply_gumbel = apply_gumbel
        
        # split the codebook
        self.codebook = nn.Parameter(self.codebook.reshape(1, self.groups, self.num_codes, -1))
        
        if not self.apply_gumbel:
            self.weight_proj = self._weight_proj(
                self.weight_proj_depth, self.weight_proj_factor, self.activation,
                self.input_dim, self.groups
            )
            
            self.mse_mean = nn.MSELoss(reduction="mean")
            self.gamma = 0.25

    def _weight_proj(self,
                     weight_proj_depth,
                     weight_proj_factor,
                     activation,
                     input_dim,
                     output_dim,
        ):
        if weight_proj_depth > 1:
            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = input_dim * weight_proj_factor
            weight_proj = nn.Sequential(
                *[
                    block(input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, output_dim),
            )
        else:
            weight_proj = nn.Linear(input_dim, output_dim)
            nn.init.normal_(weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(weight_proj.bias)
        
        return weight_proj

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    def forward(self, x, produce_targets=False):
        result = {"num_vars": self.num_codes * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)

        z_e = x
        x = self.weight_proj(x) #[bsz*tsz, self.groups*num_codes]
        x = x.view(bsz * tsz, self.groups, -1)

        result["temp"] = self.curr_temp

        if not self.apply_gumbel:
            avg_probs = torch.softmax(x.float().squeeze(), dim=-1).mean(dim=0)
            result["prob_perplexity"] = torch.exp(
                -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
            ).sum()
            
            if self.training:
                x1 = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard, dim=1).type_as(x)
            else:
                k = x.max(1)[-1]
                hard_x = (
                    x.new_zeros(*x.shape[:2])
                    .scatter_(-1, k.view(-1, 1), 1.0)
                    .view(bsz * tsz, self.groups, -1)
                )
                x1 = hard_x
            
            embeddings = self.codebook * x1.unsqueeze(-1)
            embeddings = embeddings.sum(1) #[bsz*tsz, num_codes, D]
            d = (
                (z_e.unsqueeze(1) - embeddings)
                .norm(dim=-1, p=2)
            ) #[bsz*tsz, num_codes]
            idx = d.argmin(dim=1).unsqueeze(1)
            min_encodings = torch.zeros(idx.shape[0], self.num_codes).to(x)
            min_encodings.scatter_(1, idx, 1)

            z_q = (min_encodings.unsqueeze(-1) * embeddings).sum(1)
            
            assert z_e.shape == z_q.shape, (z_e.shape, z_q.shape)
            x = self._pass_grad(z_e, z_q)
            
            x = x.view(bsz, tsz, -1).permute(0, 2, 1)
            
            with torch.no_grad():
                hard_x = (
                    idx.new_zeros(bsz * tsz * self.groups, self.num_codes)
                    .scatter_(-1, idx.view(-1, 1), 1.0)
                    .view(bsz * tsz, self.groups, -1)
                )
                hard_probs = torch.mean(hard_x.float(), dim=0)
                result["code_perplexity"] = torch.exp(
                    -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
                ).sum()
            
            if produce_targets:
                result["targets"] = idx

            z_e = z_e.float()
            z_q = z_q.float()
            latent_loss = self.mse_mean(z_q, z_e.detach())
            commitment_loss = self.mse_mean(z_e, z_q.detach())
            
            if self.time_first:
                x = x.transpose(1, 2)  # BCT -> BTC
            
            result["x"] = x

            loss = {}
            loss["commitment"] = latent_loss + self.gamma * commitment_loss
            loss["diversity"] = (self.groups-result['prob_perplexity'])/self.groups
            
            return x, loss, result

        with torch.no_grad():
            if self.apply_gumbel:
                _, k1 = x.max(-1)
                _, k2 = x[k1].max(-1)
                hard_x = [k1, k2]

        avg_probs = torch.softmax(x.float(), dim=-1).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x1 = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard, dim=1).type_as(x)
            x2 = F.gumbel_softmax((x*x1).sum(1), tau=self.curr_temp, hard=self.hard)
            
            vars = self.codebook
            x = x.view(bsz * tsz, -1)
            x = x1.unsqueeze(-1) * vars
            x = x.sum(1) #[bsz*tsz, 1, self.num_codes, -1]
            x = x2.unsqueeze(-1) * x
            x = x.sum(1)
        else:
            x = hard_x

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        # result["x"] = x
        emb_loss = {"diversity": (result["num_vars"]-result['prob_perplexity'])/result['num_vars']}

        return x, emb_loss, result

if __name__ == "__main__":
    # N = 32
    # d_e = 64
    # T = 4
    # som_dim = [32, 32]
    
    # z = torch.randn(N, d_e, T)
    # print("Inputs: ", z.shape)
    # quantizer = SOMQuantizer(d_e, som_dim, device='cpu', transition=True)
    # z_q, loss, _ = quantizer(z)
    # print("Outputs: ", z_q.shape)
    # print(loss)
    
    N = 10
    code_dim = 32
    input_dim = code_dim
    T = 4
    z = torch.randn(N, input_dim, T)
    print("Inputs: ", z.shape)
    
    num_codes = 8
    num_groups = 4
    quantizer = HierarchicalQuantizer(
        input_dim=input_dim,
        num_codes=num_codes, code_dim=code_dim,
        groups=num_groups,
        apply_gumbel=False,
    )
    quantizer.eval()
    results = quantizer(z)
    print(results[0].shape, results[1])
    
    # results[1]['kmeans_loss'].backward()
        