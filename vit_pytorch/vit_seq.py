import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from scipy.ndimage import gaussian_filter1d

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.skw = nn.ParameterList([nn.Parameter(torch.full((1,), 1.0)) for _ in range(2*depth)])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            #x = self.skw[2*i] * attn(x) + (2-self.skw[2*i]) * x
            #x = self.skw[2*i+1] * ff(x) + (2-self.skw[2*i+1]) * x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.dropout = nn.Dropout(dropout)

        self.avg_pool = nn.AdaptiveAvgPool1d(2)

    def forward(self, img):
        x = img
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x = self.transformer(x).permute(0, 2, 1)

        x = self.avg_pool(x)

        return x
