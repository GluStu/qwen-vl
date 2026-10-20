"""Position-aware cross-attention adapter (Qwen-VL, sec. 2.1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0)
    omega = 1.0 / 10000**omega
    out = torch.einsum("m,d->md", pos.reshape(-1).double(), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_2d(dim: int, gh: int, gw: int) -> torch.Tensor:
    """Fixed 2D absolute positional embedding of shape (gh*gw, dim). Requires dim % 4 == 0."""
    if dim % 4:
        raise ValueError(f"embedding dim must be divisible by 4, got {dim}")
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.cat([_sincos_1d(dim // 2, ys), _sincos_1d(dim // 2, xs)], dim=1).float()


class Resampler(nn.Module):
    """Single-layer cross-attention: `grid**2` learnable queries compress N patch tokens.

    2D positions are added to queries and keys (not values). Keys live on the ViT grid
    (e.g. 27x27), so the query-grid table is bicubically interpolated onto it, which keeps
    queries and keys in one shared coordinate frame.
    """

    def __init__(self, embed_dim: int, kv_dim: int, num_heads: int, key_grid: int, grid: int = 16):
        super().__init__()
        pos = sincos_2d(embed_dim, grid, grid)
        self.register_buffer("pos_q", pos, persistent=False)
        k = F.interpolate(
            pos.reshape(1, grid, grid, -1).permute(0, 3, 1, 2),
            size=(key_grid, key_grid),
            mode="bicubic",
            align_corners=False,
        )
        self.register_buffer("pos_k", k.permute(0, 2, 3, 1).reshape(key_grid**2, -1), persistent=False)
        self.query = nn.Parameter(torch.empty(grid * grid, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=embed_dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, kv_dim) -> (B, grid**2, D)
        x = self.ln_kv(self.kv_proj(x))
        q = (self.ln_q(self.query) + self.pos_q).unsqueeze(0).expand(x.size(0), -1, -1)
        k = x + self.pos_k.to(x.dtype)
        out, _ = self.attn(q, k, x, need_weights=False)
        return self.proj(self.ln_post(out))
