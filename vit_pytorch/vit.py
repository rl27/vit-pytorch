import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
            x = self.skw[2*i] * attn(x) + (2-self.skw[2*i]) * x
            x = self.skw[2*i+1] * ff(x) + (2-self.skw[2*i+1]) * x
        return x

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
        )

        '''
        self.fc = nn.Sequential(
            nn.Linear(800, 200),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(200, 2),
        )
        '''

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.conv0(x)
        residual1 = conv
        conv = self.conv1(conv) + residual1
        conv = self.relu(conv)
        residual2 = conv
        conv = self.conv2(conv) + residual2
        conv = self.relu(conv)
        conv = self.conv3(conv)

        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return conv

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        '''
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        '''

        patch_length = patch_size
        num_patches = 200 // patch_length

        patch_dim = 6 * patch_length

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (p n) -> b n (p c)', p = patch_length),
        #    nn.Linear(patch_dim, dim),
        #)

        num_patches = 6

        # (128, 3, 200) --> (128, 3, 200)
        '''
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=7, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True)
        )
        '''

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.pos_embedding = nn.Parameter(torch.randn(1, image_size + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.conv = Conv()

        self.pool = pool
        #self.to_latent = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(1500),
            nn.Linear(1500, 2)
        )

        #self.attention_pool = nn.Linear(dim, 1)

        self.apply(self.init_weight)


    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, img):
        #x = self.to_patch_embedding(img)
        x = img
        b, n, _ = x.shape

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        #x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        c = self.conv(x)
        x = self.transformer(x)

        x = torch.cat((x, c), -1)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #x = torch.matmul(torch.nn.functional.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        #x = self.to_latent(x)

        x = x.view(x.size(0), -1)

        return self.mlp_head(x)
