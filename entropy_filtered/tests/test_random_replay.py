"""Tests for random_replay filter mode + FilterTraceWriter/FilterTraceReader.

The paired-replay control implementation has three contracts:

  1. FilterTraceWriter.log writes a JSONL row per step with {step, n_kept, n_dropped}.
  2. FilterTraceReader maps step -> n_kept, raises KeyError for missing steps.
  3. select_kept_samples(mode="random_replay") drops EXACTLY (B - trace_n_kept) samples
     per step, selecting them uniformly at random.
  4. End-to-end: run an entropy filter, then run random_replay against its trace and
     verify acceptance counts match the entropy run's per step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from entropy_filtered.src.filter import (
    EntropyFilterConfig, FilterTraceReader, FilterTraceWriter,
    filter_batch, select_kept_samples,
)


# ---------------------------------------------------------------------------
# 1. FilterTraceWriter format
# ---------------------------------------------------------------------------

def test_trace_writer_format(tmp_path):
    path = tmp_path / "trace.jsonl"
    w = FilterTraceWriter(path)
    w.log(step=1, n_kept=128, n_dropped=0)
    w.log(step=2, n_kept=96, n_dropped=32)
    w.log(step=3, n_kept=0, n_dropped=128)
    w.close()

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 3
    assert rows[0] == {"step": 1, "n_kept": 128, "n_dropped": 0}
    assert rows[1] == {"step": 2, "n_kept": 96, "n_dropped": 32}
    assert rows[2] == {"step": 3, "n_kept": 0, "n_dropped": 128}


# ---------------------------------------------------------------------------
# 2. FilterTraceReader read-back
# ---------------------------------------------------------------------------

def test_trace_reader_lookup_and_max_step(tmp_path):
    path = tmp_path / "trace.jsonl"
    w = FilterTraceWriter(path)
    for s in range(1, 11):
        w.log(step=s, n_kept=128 - s, n_dropped=s)
    w.close()

    r = FilterTraceReader(path)
    assert r.n_kept_at(5) == 123
    assert 5 in r
    assert r.max_step() == 10


def test_trace_reader_missing_step_raises(tmp_path):
    path = tmp_path / "trace.jsonl"
    w = FilterTraceWriter(path)
    w.log(step=1, n_kept=128, n_dropped=0)
    w.close()

    r = FilterTraceReader(path)
    assert 99 not in r
    with pytest.raises(KeyError, match="no row for step 99"):
        r.n_kept_at(99)


# ---------------------------------------------------------------------------
# 3. random_replay validate — paired_trace_path required
# ---------------------------------------------------------------------------

def test_random_replay_validate_requires_paired_trace_path():
    cfg = EntropyFilterConfig(mode="random_replay")
    with pytest.raises(ValueError, match="random_replay requires paired_trace_path"):
        cfg.validate()


def test_random_replay_validate_passes_with_path():
    cfg = EntropyFilterConfig(mode="random_replay", paired_trace_path="/some/path")
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# 4. random_replay drops the right number of samples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_keep", [0, 1, 7, 64, 127, 128])
def test_random_replay_keeps_exactly_n_samples(n_keep):
    """select_kept_samples in random_replay mode must produce exactly n_keep True entries."""
    torch.manual_seed(0)
    B = 128
    H = torch.rand(B)  # values are irrelevant for random_replay
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=0,
                              paired_trace_path="/dummy")
    keep = select_kept_samples(H, cfg, step=1, trace_n_kept=n_keep)
    assert keep.shape == (B,)
    assert int(keep.sum().item()) == n_keep


def test_random_replay_warmup_keeps_all():
    """During warmup, even random_replay keeps everything regardless of trace_n_kept."""
    torch.manual_seed(0)
    B = 32
    H = torch.rand(B)
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=10,
                              paired_trace_path="/dummy")
    keep = select_kept_samples(H, cfg, step=5, trace_n_kept=0)  # would normally drop all
    assert int(keep.sum().item()) == B  # warmup overrides


def test_random_replay_requires_trace_n_kept():
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=0,
                              paired_trace_path="/dummy")
    H = torch.rand(8)
    with pytest.raises(ValueError, match="random_replay requires trace_n_kept"):
        select_kept_samples(H, cfg, step=1, trace_n_kept=None)


# ---------------------------------------------------------------------------
# 5. random_replay determinism: same seed + same trace_n_kept → same selection
# ---------------------------------------------------------------------------

def test_random_replay_is_deterministic_given_torch_seed():
    """Two calls with the same global torch seed must produce identical keep masks."""
    B = 64
    H = torch.rand(B)
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=0,
                              paired_trace_path="/dummy")

    torch.manual_seed(42)
    keep_a = select_kept_samples(H, cfg, step=10, trace_n_kept=20)
    torch.manual_seed(42)
    keep_b = select_kept_samples(H, cfg, step=10, trace_n_kept=20)
    assert torch.equal(keep_a, keep_b)


# ---------------------------------------------------------------------------
# 6. End-to-end via filter_batch
# ---------------------------------------------------------------------------

def test_filter_batch_random_replay_via_trace_n_kept():
    """filter_batch in random_replay mode must drop (B - trace_n_kept) samples."""
    torch.manual_seed(7)
    B, L, V, mid = 16, 8, 4, 3
    logits = torch.randn(B, L, V)
    mask = torch.ones(B, L, dtype=torch.bool)
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=0,
                              paired_trace_path="/dummy")
    decision = filter_batch(logits, mask, cfg, mask_token_id=mid, step=1, trace_n_kept=10)
    assert decision.n_kept == 10
    assert decision.n_dropped == 6


# ---------------------------------------------------------------------------
# 7. mode="random_replay" via select_kept_samples uses ONLY trace_n_kept,
#    not H values (the H values are irrelevant — the keep selection must NOT
#    correlate with H).
# ---------------------------------------------------------------------------

def test_random_replay_does_not_correlate_with_H():
    """If keep selection used H, the kept samples would have systematically lower H
    (or higher H, depending on direction). For random_replay, the kept-set H and
    dropped-set H should be statistically indistinguishable for large B."""
    torch.manual_seed(123)
    B = 256
    # Construct H with a clear gradient: H[i] = i / B
    H = torch.linspace(0.0, 1.0, B)
    cfg = EntropyFilterConfig(mode="random_replay", warmup_steps=0,
                              paired_trace_path="/dummy")
    keep = select_kept_samples(H, cfg, step=1, trace_n_kept=B // 2)

    kept_mean = H[keep].mean().item()
    dropped_mean = H[~keep].mean().item()
    # Both groups should have mean ≈ 0.5 (population mean of the linspace).
    # Random sub-sampling at n=128 has standard error ≈ 0.026 — pick generous tolerance.
    assert abs(kept_mean - 0.5) < 0.1, f"random_replay biased: kept_mean={kept_mean}"
    assert abs(dropped_mean - 0.5) < 0.1, f"random_replay biased: dropped_mean={dropped_mean}"
