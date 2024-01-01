import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def get_shape_list(tensor):
    shape = tensor.size()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape
    else:
        print('something wrong with static shaping')
        assert False


class CrossModalLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, modal_a_sequences, modal_b_sequences):
        """
        Parameters
        ----------
        modal_a_sequences : tensor
            the first modality (e.g. person 1 motion embedding)
        modal_b_sequences : tensor
            the second modality (e.g. person 2 embedding)
        """

        _, _, modal_a_width = get_shape_list(modal_a_sequences)
        merged_sequences = modal_a_sequences
        if modal_b_sequences is not None:
            _, _, modal_b_width = get_shape_list(modal_b_sequences)
            if modal_a_width != modal_b_width:
                raise ValueError(
                    "Modality hidden size (%d) does not match with (%d)" % (modal_a_width, modal_b_width))
            merged_sequences = torch.cat([merged_sequences, modal_b_sequences])
        
        return merged_sequences


class CrossModalAttention(nn.Module):
    """ Cross Modal Attention Layer

    Given 2 modalities (a, b), computes the K,V from modality b and Q from
    modality a.
    """

    def __init__(self, in_dim, dim, heads=8, in_dim2=None):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        if in_dim2 is not None:
            self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
        else:
            self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.to_q = nn.Linear(in_dim, dim, bias=False)
        if in_dim2 is not None:
            dim2 = int((in_dim + in_dim2*2) / 3)
        else:
            dim2 = dim
        self.to_out = nn.Linear(dim2, dim)

        self.rearrange_qkv = Rearrange(
            "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")

    def forward(self, x_data):
        x_a = x_data['x_a']
        x_b = x_data['x_b']

        kv = self.to_kv(x_b)
        q = self.to_q(x_a)

        qkv = torch.cat((q, kv), dim=-1)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return {'x_a':x_a, 'x_b':out}