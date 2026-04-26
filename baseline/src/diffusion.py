"""Diffusion: noise schedule, forward (masking) process, MDM training loss.

References (paper-notes):
  §2.1   forward process: each token independently masked with prob (1 - α_t)
  §2.3   continuous-time loss: ∫ α'_t/(1-α_t) · E[Σ -log p_θ(x_0^i | x_t)] dt
  §12.2  practical implementation: discrete-time loss (Prop E.1, Eq. 6) with 1/n weighting
         per sample. Linear schedule α_t = 1-t  ⇒  α'_t = -1, weight = -1/(1-α_t) = -1/t,
         which combined with -log p_θ in the loss gives the positive (1/n) Σ -log p_θ.

Implementation chosen: discrete-time mask-count form.
  For each sample: sample n ~ Uniform{1, ..., maskable_len},
  uniformly choose n positions to mask, compute 1/n × Σ_{masked} -log p_θ,
  then mean over batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------

def alpha_t(t: torch.Tensor) -> torch.Tensor:
    """α_t = 1 - t for t ∈ [0, 1]. Paper §2.1."""
    return 1.0 - t


def alpha_prime_t(t: torch.Tensor) -> torch.Tensor:
    """dα_t / dt = -1 for the linear schedule."""
    return torch.full_like(t, -1.0)


# ---------------------------------------------------------------------------
# Forward (masking) process — vectorized, paper §2.1
# ---------------------------------------------------------------------------

def apply_mask(
    x0: torch.Tensor,
    n: torch.Tensor,
    mask_token_id: int,
    *,
    fixed_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask exactly n[b] random positions per sample (uniformly without replacement).

    Args:
        x0:           (B, L) clean sequences.
        n:            (B,)  number of positions to mask in each sample. Must be in [0, L].
        mask_token_id: integer mask-token id.
        fixed_mask:   optional (B, L) bool tensor; True positions are NEVER masked
                      (used for padding tokens, or for puzzle clues).
        generator:    optional torch RNG for deterministic masking.

    Returns:
        x_masked: (B, L) — copy of x0 with masked positions replaced by mask_token_id.
        mask:     (B, L) bool — True at masked positions.

    Mask construction uses the argsort-of-random-noise trick: for each sample we
    sort uniform random noise and take the first n[b] indices. Fully vectorized.
    """
    B, L = x0.shape
    device = x0.device

    # Generate one uniform noise tensor; the sort gives a uniform random permutation
    # of [0, L) per row. Setting the noise of fixed positions to +infinity guarantees
    # those positions go to the end of the sort, so the first n[b] selected indices
    # are always non-fixed.
    noise = torch.rand(B, L, device=device, generator=generator)
    if fixed_mask is not None:
        noise = noise.masked_fill(fixed_mask, float("inf"))

    sorted_idx = noise.argsort(dim=-1)                    # (B, L), permutation per row
    arange_L = torch.arange(L, device=device).expand(B, L)
    pos_in_sort = arange_L                                # 0..L-1 along last dim
    # mask = True iff (rank in sort) < n[b]
    rank = pos_in_sort < n.unsqueeze(-1)                  # (B, L) bool, by rank position
    # Convert rank-in-sort → original index mask
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    mask.scatter_(1, sorted_idx, rank)

    x_masked = x0.clone()
    x_masked[mask] = mask_token_id
    return x_masked, mask


# ---------------------------------------------------------------------------
# MDM loss — vectorized, 1/n per sample, paper §2.4 / §12.2
# ---------------------------------------------------------------------------

@dataclass
class MdmLossOutput:
    loss: torch.Tensor                  # scalar — the value to .backward()
    n_masked_per_sample: torch.Tensor   # (B,) integer counts of masked positions
    n_masked_total: int                 # sum(n_masked_per_sample), for diagnostics


def mdm_loss(
    logits: torch.Tensor,         # (B, L, V)
    x0: torch.Tensor,             # (B, L)
    mask: torch.Tensor,           # (B, L) bool — True at masked positions
) -> MdmLossOutput:
    """Compute the MDM training loss with 1/n weighting per sample.

    Loss per sample b:   (1 / n_b) × Σ_{i ∈ M_b} −log p_θ(x_0^{i,b} | x^b)
    Final loss:          mean over batch.
    """
    B, L, V = logits.shape
    # Per-token cross-entropy at every position (the reduction='none' form).
    # We zero out non-masked positions before per-sample summation.
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        x0.reshape(-1),
        reduction="none",
    ).reshape(B, L)
    ce = ce * mask.float()

    n_masked = mask.sum(dim=-1)                           # (B,)
    # Avoid division by zero for any sample with n=0 (shouldn't happen with n≥1 sampling
    # but guard anyway). For samples with n=0, contribute zero to the loss.
    safe_n = n_masked.clamp(min=1).float()
    per_sample = ce.sum(dim=-1) / safe_n                  # (B,)
    per_sample = per_sample * (n_masked > 0).float()

    loss = per_sample.mean()
    return MdmLossOutput(loss=loss, n_masked_per_sample=n_masked, n_masked_total=int(n_masked.sum().item()))


# ---------------------------------------------------------------------------
# Mask-count sampler (per-step convenience)
# ---------------------------------------------------------------------------

def sample_mask_counts(
    batch_size: int,
    maskable_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample n ~ Uniform{1, ..., maskable_len} per sample (paper §12.2)."""
    return torch.randint(
        1, maskable_len + 1, (batch_size,), device=device, generator=generator
    )
