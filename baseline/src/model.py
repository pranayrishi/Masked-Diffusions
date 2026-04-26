"""Shared transformer for both MDM (bidirectional) and ARM (causal) models.

Architecture choices (binding, paper-notes §8):
  - Time-embedding-free denoiser: model takes only token IDs, no `t` argument.
  - Bidirectional attention for MDM (`causal=False`), causal triangular for ARM.
  - Positional embeddings: RoPE for L&O-NAE-SAT (paper Appendix C.2.1) or
    learnable absolute (paper §3.2 scaling-law experiments).
  - Pre-LayerNorm transformer blocks, GELU FFN, no dropout by default.

Sizes follow Ye et al. 2024 GPT-2 ratios (verify against their codebase configs):
  6M  : hidden=256, layers=6,  heads=4, ff=1024
  19M : hidden=384, layers=8,  heads=6, ff=1536
  42M : hidden=512, layers=10, heads=8, ff=2048

Smoke-test sized:
  tiny: hidden=32, layers=2, heads=2, ff=64
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """Rotary positional embedding (Su et al. 2021).

    Returns (cos, sin) tensors of shape (1, 1, L, head_dim).
    """

    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        # head_dim must be even for the rotation pairs to work
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, hidden: int, n_heads: int, ff: int, causal: bool, dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0, f"hidden {hidden} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.causal = causal
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.out = nn.Linear(hidden, hidden, bias=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden, ff, bias=True),
            nn.GELU(),
            nn.Linear(ff, hidden, bias=True),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cos=None, rope_sin=None) -> torch.Tensor:
        B, L, H = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, n_heads, L, head_dim)
        if rope_cos is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        if self.causal:
            mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(mask[None, None], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).reshape(B, L, H)
        y = self.out(y)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level transformer
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int
    hidden: int
    n_layers: int
    n_heads: int
    ff: int
    max_seq_len: int
    causal: bool                         # MDM: False, ARM: True
    pos_type: str = "rope"               # "rope" or "learned"
    dropout: float = 0.0
    weight_tie: bool = False             # tie token embedding ↔ output head


class Transformer(nn.Module):
    """Shared transformer used for both MDM and ARM.

    Time-embedding-free: forward signature is `forward(x: LongTensor[B, L])`.
    The model never sees the noise level `t` directly; for MDMs it infers it
    implicitly from the count of mask tokens in `x`.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden)
        if cfg.pos_type == "rope":
            self.rope = RoPE(cfg.hidden // cfg.n_heads, cfg.max_seq_len)
            self.pos = None
        elif cfg.pos_type == "learned":
            self.rope = None
            self.pos = nn.Embedding(cfg.max_seq_len, cfg.hidden)
        else:
            raise ValueError(f"unknown pos_type: {cfg.pos_type}")
        self.blocks = nn.ModuleList(
            [Block(cfg.hidden, cfg.n_heads, cfg.ff, causal=cfg.causal, dropout=cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        if cfg.weight_tie:
            self.head.weight = self.tok.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def num_parameters(self, *, count_embedding: bool = True) -> int:
        if count_embedding:
            return sum(p.numel() for p in self.parameters())
        return sum(p.numel() for n, p in self.named_parameters() if "tok" not in n and "pos" not in n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) LongTensor of token IDs. Returns logits (B, L, vocab_size)."""
        B, L = x.shape
        h = self.tok(x)
        cos = sin = None
        if self.rope is not None:
            cos, sin = self.rope(L, device=x.device, dtype=h.dtype)
        if self.pos is not None:
            h = h + self.pos(torch.arange(L, device=x.device))
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h = self.ln_f(h)
        return self.head(h)
