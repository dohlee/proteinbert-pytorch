import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.conv_narrow = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', dilation=1),
            nn.GELU(),
            Rearrange('b d l -> b l d')
        )
        self.conv_wide = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', dilation=5),
            nn.GELU(),
            Rearrange('b d l -> b l d')
        )
    
    def forward(self, x):
        return self.conv_narrow(x) + self.conv_wide(x)

class GlobalAttention(nn.Module):
    def __init__(self, d_local, d_global, n_heads, d_key):
        super().__init__()
        d_value = d_global // n_heads

        self.to_q = nn.Sequential(nn.Linear(d_global, d_key * n_heads, bias=False), nn.Tanh())
        self.to_k = nn.Sequential(nn.Linear(d_local, d_key * n_heads, bias=False), nn.Tanh())
        self.to_v = nn.Sequential(nn.Linear(d_local, d_value * n_heads, bias=False), nn.GELU())

        self.n_heads = n_heads
        self.d_key = d_key
    
    def forward(self, x_local, x_global):
        q = self.to_q(x_global)
        k = self.to_k(x_local)
        v = self.to_v(x_local)

        q = rearrange(q, 'b (h d) -> b h d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.n_heads)

        att = einsum('b h d, b l h d -> b h l', q, k) / math.sqrt(self.d_key)
        att = att.softmax(dim=-1)

        x_global = einsum('b h l, b l h d -> b h d', att, v)
        x_global = rearrange(x_global, 'b h d -> b (h d)')
        return x_global


class TransformerLikeBlock(nn.Module):
    def __init__(self, d_local, d_global):
        super().__init__()

        self.wide_and_narrow_conv1d = ConvBlock(d_local, d_local)
        self.dense_and_broadcast = nn.Sequential(
            nn.Linear(d_global, d_local),
            nn.GELU(),
            Rearrange('b d -> b () d')
        )
        self.local_ln1 = nn.LayerNorm(d_local)
        self.local_dense = nn.Sequential(
            Residual(nn.Sequential(nn.Linear(d_local, d_local), nn.GELU())),
            nn.LayerNorm(d_local),
        )

        self.global_dense1 = nn.Sequential(nn.Linear(d_global, d_global), nn.GELU())
        self.global_attention = GlobalAttention(d_local, d_global, n_heads=4, d_key=64)
        self.global_ln1 = nn.LayerNorm(d_global)
        self.global_dense2 = nn.Sequential(
            Residual(nn.Sequential(nn.Linear(d_global, d_global), nn.GELU())),
            nn.LayerNorm(d_global),
        )
    
    def forward(self, x_local, x_global):
        x_local = self.local_ln1(
            x_local + self.wide_and_narrow_conv1d(x_local) + self.dense_and_broadcast(x_global)
        )
        x_local = self.local_dense(x_local)

        x_global = self.global_ln1(
            x_global + self.global_dense1(x_global) + self.global_attention(x_local, x_global)
        )
        x_global = self.global_dense2(x_global)

        return x_local, x_global

class ProteinBERT(nn.Module):
    def __init__(
            self,
            vocab_size,
            ann_size,
            d_local=128,
            d_global=512,
        ):
        super().__init__()

        self.embed_local = nn.Embedding(vocab_size, d_local)
        self.embed_global = nn.Sequential(nn.Linear(ann_size, d_global), nn.GELU())

        self.blocks = nn.ModuleList([TransformerLikeBlock(d_local, d_global) for _ in range(6)])

        self.local_head = nn.Sequential(nn.Linear(d_local, vocab_size))  # NOTE: logits are returned
        self.global_head = nn.Sequential(nn.Linear(d_global, ann_size), nn.Sigmoid())
    
    def forward(self, x_local, x_global):
        x_local = self.embed_local(x_local)
        x_global = self.embed_global(x_global)

        for block in self.blocks:
            x_local, x_global = block(x_local, x_global)

        return self.local_head(x_local), self.global_head(x_global)

if __name__ == '__main__':
    vocab_size = 26
    ann_size = 8943
    bsz = 1

    model = ProteinBERT(vocab_size, ann_size)

    x_local = torch.randint(0, vocab_size, (bsz, 512))
    x_global = torch.rand(bsz, ann_size)

    x_local, x_global = model(x_local, x_global)
    print(x_local.shape, x_global.shape)

    # Print the number of parameters in the model.
    # NOTE: Must have ~16M parameters according to the paper.
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))