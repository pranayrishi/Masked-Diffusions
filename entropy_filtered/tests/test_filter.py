"""Tests for the entropy filter module.

Concrete invariants (binding):

  1. Determinism: with fixed seed + fixed batch + fixed model state, `filter_batch`
     returns the same `decision.keep` and the same per-sample `H` byte-for-byte.

  2. Per-sample entropy is upper-bounded by log(V_effective). For V=4 with the mask
     token excluded, max entropy is log(3) ≈ 1.0986 nats.

  3. Warmup: for steps 1..warmup_steps inclusive, ALL samples are kept regardless
     of mode and threshold. At step warmup_steps+1, the filter fires.

  4. Percentile mode keeps approximately (pct_high - pct_low) fraction of the batch.

  5. "Mode = none" is a strict pass-through: keep == [True] * B.

  6. "Mode = top" with a finite H_high drops samples above the threshold.

  7. "Mode = bottom" with H_low > 0 drops samples below the threshold.

  8. "Mode = band" with feasible thresholds drops both ends.

  9. Defensive: if every sample's entropy lies outside an absolute band, percentile
     never wipes the entire batch (it falls back to keeping all).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from entropy_filtered.src.filter import (
    EntropyFilterConfig,
    FilterDecision,
    filter_batch,
    per_sample_entropy,
    select_kept_samples,
)


def _toy_logits_and_mask(B=8, L=10, V=4, mask_token_id=3, seed=0):
    """Construct logits with deliberately varied per-sample entropy.

    Sample b has logits scaled by `b * 0.5` at masked positions, so smaller b means
    flatter distribution (higher entropy) and larger b means sharper (lower entropy).
    """
    torch.manual_seed(seed)
    logits = torch.zeros(B, L, V)
    base = torch.randn(L, V)               # one shared random pattern
    for b in range(B):
        logits[b] = base * (0.1 + b * 0.5)
    # mask the second half of each row
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L // 2 :] = True
    return logits, mask, mask_token_id


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_filter_batch_is_deterministic():
    logits, mask, mid = _toy_logits_and_mask(seed=42)
    cfg = EntropyFilterConfig(mode="band", warmup_steps=0, H_low=0.5, H_high=1.0)
    d1 = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1000)
    d2 = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1000)
    assert torch.equal(d1.keep, d2.keep)
    assert torch.allclose(d1.H, d2.H)


# ---------------------------------------------------------------------------
# 2. Entropy upper bound
# ---------------------------------------------------------------------------

def test_entropy_bounded_by_log_v_effective():
    """With V=4 and mask token excluded, max entropy = log(3) ≈ 1.0986 nats."""
    logits, mask, mid = _toy_logits_and_mask(B=16, V=4, mask_token_id=3, seed=7)
    H = per_sample_entropy(
        logits, mask, mask_token_id=mid, reduction="mean", use_softmax_excluding_mask_token=True
    )
    upper = math.log(3) + 1e-6
    assert H.max().item() <= upper, f"max entropy {H.max().item()} exceeded log(3)"


def test_entropy_uniform_logits_attains_log_v_effective():
    """If all 3 non-mask logits are equal, the per-position entropy = log(3) exactly."""
    B, L, V, mid = 1, 4, 4, 3
    logits = torch.zeros(B, L, V)
    # Setting non-mask logits to 0 and mask to 0 gives equal probs over the 4 tokens;
    # but per_sample_entropy excludes mask, leaving 3 equal classes → entropy = log(3)
    mask = torch.tensor([[True, True, False, False]])
    H = per_sample_entropy(logits, mask, mask_token_id=mid, reduction="mean")
    expected = math.log(3)
    assert abs(H.item() - expected) < 1e-5, f"got {H.item()}, expected {expected}"


# ---------------------------------------------------------------------------
# 3. Warmup window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bottom", "top", "band", "percentile_band"])
def test_warmup_keeps_all_samples_inclusive(mode):
    logits, mask, mid = _toy_logits_and_mask(B=8, seed=11)
    cfg = EntropyFilterConfig(mode=mode, warmup_steps=10, H_low=0.5, H_high=1.0,
                              pct_low=0.25, pct_high=0.75)
    # Every step from 1 to 10 (inclusive) must keep all 8 samples
    for step in range(1, 11):
        d = filter_batch(logits, mask, cfg, mask_token_id=mid, step=step)
        assert d.n_kept == 8, f"step {step}: filter fired during warmup"


def test_warmup_filter_fires_at_warmup_plus_one():
    logits, mask, mid = _toy_logits_and_mask(B=8, seed=11)
    cfg = EntropyFilterConfig(mode="percentile_band", warmup_steps=10,
                              pct_low=0.25, pct_high=0.75)
    d_warmup = filter_batch(logits, mask, cfg, mask_token_id=mid, step=10)
    d_post = filter_batch(logits, mask, cfg, mask_token_id=mid, step=11)
    assert d_warmup.n_kept == 8
    assert d_post.n_kept < 8, "percentile_band did not fire at step warmup+1"


# ---------------------------------------------------------------------------
# 4. Percentile mode keep fraction
# ---------------------------------------------------------------------------

def test_percentile_band_keeps_approximately_half():
    """With pct_low=0.25 and pct_high=0.75, expect ~50% of a 64-sample batch."""
    torch.manual_seed(0)
    B, L, V = 64, 16, 4
    logits = torch.randn(B, L, V) * (0.1 + torch.arange(B).reshape(B, 1, 1).float())
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L // 2 :] = True

    cfg = EntropyFilterConfig(mode="percentile_band", warmup_steps=0,
                              pct_low=0.25, pct_high=0.75)
    d = filter_batch(logits, mask, cfg, mask_token_id=3, step=1000)
    # Allow a 10% slack because of percentile rounding
    assert abs(d.n_kept / B - 0.5) < 0.1, f"percentile_band kept {d.n_kept}/{B}"


# ---------------------------------------------------------------------------
# 5. mode=none is a pass-through
# ---------------------------------------------------------------------------

def test_mode_none_is_pass_through():
    logits, mask, mid = _toy_logits_and_mask(B=8, seed=21)
    cfg = EntropyFilterConfig(mode="none", warmup_steps=0, H_low=10.0, H_high=10.0)
    d = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1)
    assert d.n_kept == 8 and d.n_dropped == 0


# ---------------------------------------------------------------------------
# 6. mode=top drops above H_high
# ---------------------------------------------------------------------------

def test_top_mode_drops_high_entropy():
    """Construct a batch with known per-sample entropies; assert that mode='top'
    with H_high between two of them keeps only the low-entropy ones."""
    # Build logits where two samples have ~maximum entropy (uniform) and two have
    # ~minimum entropy (very peaked).
    B, L, V, mid = 4, 4, 4, 3
    logits = torch.zeros(B, L, V)
    # Samples 0, 1: peaked (low entropy)
    logits[0, :, 0] = 10.0
    logits[1, :, 1] = 10.0
    # Samples 2, 3: roughly uniform (high entropy)
    # leave logits at 0 → after excluding mask, distribution is (1/3, 1/3, 1/3, ε≈0)
    logits[2] = 0.0
    logits[3] = 0.0
    mask = torch.ones(B, L, dtype=torch.bool)

    cfg = EntropyFilterConfig(mode="top", warmup_steps=0, H_high=0.5)   # nats
    d = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1)
    # Samples 0, 1 should be kept (they're well below 0.5 nats); 2, 3 dropped
    assert d.keep.tolist() == [True, True, False, False], f"keep={d.keep.tolist()}, H={d.H.tolist()}"


def test_bottom_mode_drops_low_entropy():
    """Symmetric to the previous test for mode='bottom'."""
    B, L, V, mid = 4, 4, 4, 3
    logits = torch.zeros(B, L, V)
    logits[0, :, 0] = 10.0
    logits[1, :, 1] = 10.0
    # 2, 3 stay uniform (high entropy)
    mask = torch.ones(B, L, dtype=torch.bool)

    cfg = EntropyFilterConfig(mode="bottom", warmup_steps=0, H_low=0.5)
    d = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1)
    # Samples 0, 1 should be dropped (entropy << 0.5); 2, 3 kept
    assert d.keep.tolist() == [False, False, True, True], f"keep={d.keep.tolist()}, H={d.H.tolist()}"


def test_band_mode_drops_both_ends():
    """Mode='band' should keep only mid-entropy samples (drops both ends)."""
    B, L, V, mid = 6, 4, 4, 3
    logits = torch.zeros(B, L, V)
    # peaked (low entropy) → samples 0, 1
    logits[0, :, 0] = 10.0
    logits[1, :, 1] = 10.0
    # uniform (highest entropy ≈ log 3 ≈ 1.0986) → samples 2, 3 (logits all 0)
    # mid-entropy → samples 4, 5 (moderate bias gives ≈ 0.83 nats)
    logits[4, :, 0] = 1.5
    logits[5, :, 0] = 1.5
    mask = torch.ones(B, L, dtype=torch.bool)

    cfg = EntropyFilterConfig(mode="band", warmup_steps=0, H_low=0.5, H_high=1.05)
    d = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1)
    assert d.keep.tolist() == [False, False, False, False, True, True], \
        f"keep={d.keep.tolist()}, H={d.H.tolist()}"


# ---------------------------------------------------------------------------
# 7. Defensive: percentile mode never empties the batch
# ---------------------------------------------------------------------------

def test_percentile_band_never_empties_batch():
    """With pct_low == pct_high (= same percentile), the keep mask collapses to
    a single sample. With our defensive fallback (in `select_kept_samples`),
    even a degenerate config should not produce an empty keep mask."""
    torch.manual_seed(0)
    B, L, V, mid = 8, 4, 4, 3
    logits = torch.randn(B, L, V)
    mask = torch.ones(B, L, dtype=torch.bool)
    cfg = EntropyFilterConfig(mode="percentile_band", warmup_steps=0,
                              pct_low=0.5, pct_high=0.5)
    keep = select_kept_samples(
        per_sample_entropy(logits, mask, mask_token_id=mid),
        cfg, step=1,
    )
    assert keep.any(), "select_kept_samples produced an entirely-False keep mask"


def test_validate_rejects_bad_config():
    cfg = EntropyFilterConfig(mode="garbage")
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = EntropyFilterConfig(mode="band", H_low=2.0, H_high=1.0)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = EntropyFilterConfig(mode="percentile_band", pct_low=0.8, pct_high=0.2)
    with pytest.raises(ValueError):
        cfg.validate()
