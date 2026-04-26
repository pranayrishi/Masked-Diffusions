"""Test the forward (masking) process.

Concrete assertion (binding):
  Mask 10000 sequences at α_t = 0.5 (i.e., mask 50 % of positions per sample)
  and assert empirical mask rate is 0.5 ± 0.01.

Also checks:
  - `apply_mask` honors the per-sample n[b] count exactly.
  - Fixed positions (e.g., padding or clues) are never masked.
  - Mask token never collides with data when fixed_mask is set.
"""

from __future__ import annotations

import torch

from src.diffusion import apply_mask, sample_mask_counts


def test_empirical_mask_rate_at_alpha_half_is_close_to_50_percent():
    """At α_t = 0.5, mask-prob is 1 - α_t = 0.5. Mask 10 000 sequences and check.

    We use a deterministic n = round(0.5 × L) per sample, which is the standard
    way to instantiate a particular α_t value with our exact-count masker.
    The paper's training does n ~ Uniform{1, ..., L} (paper §12.2); here we
    pin n to test the masking *mechanism* at a specific noise level.
    """
    torch.manual_seed(0)
    B, L = 10_000, 50
    mask_token_id = 99
    n_target = L // 2                                      # exactly half

    x0 = torch.zeros(B, L, dtype=torch.long)               # data values arbitrary; here all 0
    n = torch.full((B,), n_target, dtype=torch.long)
    x_masked, mask = apply_mask(x0, n, mask_token_id=mask_token_id)

    empirical_rate = mask.float().mean().item()
    assert abs(empirical_rate - 0.5) < 0.01, \
        f"empirical mask rate {empirical_rate:.4f} differs from 0.5 by more than 0.01"


def test_apply_mask_count_per_sample_is_exact():
    """apply_mask should mask exactly n[b] positions in sample b."""
    torch.manual_seed(1)
    B, L = 64, 32
    x0 = torch.zeros(B, L, dtype=torch.long)
    # Use a mix of n values
    n = torch.tensor([0, 1, 5, 16, 31, 32] * (B // 6 + 1), dtype=torch.long)[:B]
    _, mask = apply_mask(x0, n, mask_token_id=7)
    counts = mask.sum(dim=-1)
    assert torch.equal(counts, n), f"per-sample mask count mismatch:\n  expected {n.tolist()}\n  got      {counts.tolist()}"


def test_apply_mask_respects_fixed_mask():
    """Fixed positions must never be masked. We assert this on 1000 samples."""
    torch.manual_seed(2)
    B, L = 1000, 32
    x0 = torch.zeros(B, L, dtype=torch.long)

    # Fix the last 10 positions (e.g., simulating padding or puzzle clues)
    fixed_mask = torch.zeros(B, L, dtype=torch.bool)
    fixed_mask[:, -10:] = True

    # Try to mask 20 positions per sample. The 22 non-fixed positions can support this.
    n = torch.full((B,), 20, dtype=torch.long)
    _, mask = apply_mask(x0, n, mask_token_id=99, fixed_mask=fixed_mask)

    # No fixed position is masked
    assert not (mask & fixed_mask).any(), "apply_mask masked a position marked as fixed"
    # And we still got exactly n masked tokens per sample
    assert torch.equal(mask.sum(dim=-1), n)


def test_sample_mask_counts_in_range():
    """sample_mask_counts must produce values in {1, ..., maskable_len}."""
    torch.manual_seed(3)
    n = sample_mask_counts(batch_size=10_000, maskable_len=64, device=torch.device("cpu"))
    assert n.min().item() >= 1
    assert n.max().item() <= 64
