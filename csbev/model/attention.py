import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim: int, mlp_dim: int, n_heads: int, dropout: float = 0.0,
    ):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True,
        )

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q: torch.Tensor):
        """Perform cross attention between x and query q
        Args:
            x (torch.Tensor): input tensor [B, N, D], batch first.
            q (torch.Tensor): query tensor [B, M, D]
        """
        # compute attention
        out = self.lnorm1(x)
        out, attn = self.attn(query=q, key=x, value=x, average_attn_weights=False)
        attn = attn.permute(1, 0, 2, 3)  # [B, H, N, M] --> [H, B, N, M]

        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out, attn


class SetAttentiveBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        # input and output dimensions of input points
        self.dim_in = dim_in
        self.dim_out = dim_out

        # num_inds M specifies the size of the learnable query array [1, M, D]
        self.num_inds = num_inds
        self.register_parameter(
            name="i", param=nn.Parameter(torch.randn(1, num_inds, dim_out)),
        )  # [1, M, D]
        nn.init.xavier_uniform_(self.i)

        if dim_in != dim_out:
            self.fc = nn.Linear(dim_in, dim_out)

        self.input_proc = hasattr(self, "fc")

        self.att1 = CrossAttention(dim_out, dim_out, num_heads, dropout_p)
        self.att2 = CrossAttention(dim_out, dim_out, num_heads, dropout_p)

    def project(self, x: torch.Tensor):
        # query with the learnable array (M << N)
        i = self.i.repeat(x.shape[0], 1, 1)  # [B, M, D]
        h, alpha = self.att1(x, i)  # [B, M, D], [num_heads, B, M, N]
        return h, alpha.transpose(2, 3)  # [B, M, D], [num_heads, B, N, M]

    def broadcast(self, h: torch.Tensor, x: torch.Tensor):
        # map back to the original input size by setting query = x
        o, alpha = self.att2(h, x)  # [B, N, D], [num_heads, B, N, M]
        return o, alpha

    def forward(self, x: torch.Tensor):
        """Perform attention on point set

        Args:
            x (torch.Tensor): [B, N, dim_in], the input point set, N is the number of points

        Returns:
            o (torch.Tensor): [B, N, dim_out], the output point set
            h (torch.Tensor): [B, M, dim_out], the projected set, M < N
            attn1 (torch.Tensor): [H, B, N, M], attention weights for the projection
            attn2 (torch.Tensor): [H, B, N, M], attention weights for the broadcast
        """
        if self.input_proc:
            x = self.fc(x)

        # step 1: project the input points to a smaller size h [B, M, dim_out]
        h, attn1 = self.project(x)
        # step 2: recover the original set size
        o, attn2 = self.broadcast(h, x)

        return o, h, attn1, attn2


if __name__ == "__main__":
    attnblock = SetAttentiveBlock(dim_in=16, dim_out=16, num_heads=2, num_inds=3)
    inputs = torch.randn(5, 10, 16)
    o, h, attn1, attn2 = attnblock(inputs)
    print(o.shape, h.shape, attn1.shape, attn2.shape)
