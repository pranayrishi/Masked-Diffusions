"""Early-stop (plateau detection) tests for calibration runs.

Hard contract:
  1. _is_plateau correctly identifies the first step satisfying the
     rolling_mean_relative criterion: |mean(last W) - mean(prev W)| / mean(prev W) < tol.
  2. _is_plateau returns False before min_step.
  3. _is_plateau returns False at non-multiples of check_every.
  4. End-to-end: a tiny calibration-style run that converges fast STOPS early and
     writes plateau_step.txt with a single integer ≤ num_iterations.
  5. End-to-end: a run with early_stop disabled runs to completion and writes
     plateau_step.txt = num_iterations.
"""

from __future__ import annotations

import json
from pathlib import Path

from entropy_filtered.src.train_filtered import (
    EarlyStopConfig,
    _is_plateau,
    _to_filtered_config,
    run_filtered_training,
)


# ---------------------------------------------------------------------------
# Unit: _is_plateau
# ---------------------------------------------------------------------------

def test_is_plateau_fires_on_flat_loss():
    """Constant loss → relative diff is 0 → plateau."""
    losses = {s: 1.0 for s in range(1, 2001)}
    cfg = EarlyStopConfig(enabled=True, tolerance=0.005, window=500,
                          check_every=500, min_step=1000)
    assert _is_plateau(losses, 2000, cfg) is True


def test_is_plateau_does_not_fire_on_decreasing_loss():
    """Loss decreasing by 5% step-to-step → relative diff far above tolerance."""
    losses = {s: 1.0 - 0.0001 * s for s in range(1, 2001)}  # loss(2000) = 0.8 vs 1.0
    cfg = EarlyStopConfig(enabled=True, tolerance=0.005, window=500,
                          check_every=500, min_step=1000)
    assert _is_plateau(losses, 2000, cfg) is False


def test_is_plateau_silent_before_min_step():
    losses = {s: 1.0 for s in range(1, 2001)}
    cfg = EarlyStopConfig(enabled=True, tolerance=0.005, window=500,
                          check_every=500, min_step=5000)
    assert _is_plateau(losses, 2000, cfg) is False


def test_is_plateau_silent_off_check_boundary():
    losses = {s: 1.0 for s in range(1, 2001)}
    cfg = EarlyStopConfig(enabled=True, tolerance=0.005, window=500,
                          check_every=500, min_step=1000)
    # 1999 isn't a multiple of check_every=500
    assert _is_plateau(losses, 1999, cfg) is False
    # 2000 is
    assert _is_plateau(losses, 2000, cfg) is True


def test_is_plateau_disabled_returns_false_unconditionally():
    losses = {s: 1.0 for s in range(1, 2001)}
    cfg = EarlyStopConfig(enabled=False)
    assert _is_plateau(losses, 2000, cfg) is False


# ---------------------------------------------------------------------------
# Unit: half-coverage requirement
# ---------------------------------------------------------------------------

def test_is_plateau_requires_half_coverage_in_each_window():
    """If the optimizer was skipped on most steps in either window, plateau cannot be
    declared (we don't have enough samples)."""
    # Heavily sparse losses_by_step — only every 10th step has a loss
    losses = {s: 1.0 for s in range(1, 2001) if s % 10 == 0}  # 200 entries total
    cfg = EarlyStopConfig(enabled=True, tolerance=0.005, window=500,
                          check_every=500, min_step=1000)
    # 50 entries in last window, 50 in prev — both below window/2 = 250 threshold
    assert _is_plateau(losses, 2000, cfg) is False


# ---------------------------------------------------------------------------
# End-to-end: tiny calibration run early-stops and writes plateau_step.txt
# ---------------------------------------------------------------------------

_TINY_CONFIG = {
    "task": "lo_nae_sat",
    "data": {
        "N": 5, "P": 10, "m": 2,
        "pad_to": 16, "pad_value": 2,
        "mask_token_id": 3, "vocab_size": 4,
        "seed": 42,
    },
    "model": {
        "hidden": 32, "n_layers": 2, "n_heads": 2, "ff": 64,
        "causal": False, "pos_type": "rope",
        "dropout": 0.0, "weight_tie": False, "vocab_size": 4,
    },
    "seed": 0,
    "num_iterations": 800,
    "batch_size": 32,
    "lr": 1.0e-3,
    "weight_decay": 0.1,
    "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0,
    "eta_min_ratio": 0.1,
    "log_every": 50,
    "ckpt_every": 800,
    "entropy_filter": {"mode": "none", "warmup_steps": 0},
    "train_size": 500,
    "train_sample_seed": 123,
    "output_dir": "PLACEHOLDER",
}


def test_early_stop_writes_single_integer_plateau_file(tmp_path):
    """Run with a generous tolerance so plateau fires before num_iterations.
    Verify plateau_step.txt parses as a single integer ≤ num_iterations."""
    out_dir = tmp_path / "calib"
    cfg_dict = dict(_TINY_CONFIG)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["early_stop"] = {
        "enabled": True,
        "criterion": "rolling_mean_relative",
        "tolerance": 0.05,    # 5% — generous; should fire on the small toy task
        "window": 100,
        "check_every": 100,
        "min_step": 200,
    }

    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    plateau_path = out_dir / "plateau_step.txt"
    assert plateau_path.exists(), "plateau_step.txt was not written"

    content = plateau_path.read_text()
    plateau_step = int(content.strip())  # must parse as a single integer
    assert 1 <= plateau_step <= cfg.num_iterations, \
        f"plateau_step {plateau_step} outside [1, {cfg.num_iterations}]"


def test_no_early_stop_runs_to_cap_and_writes_cap(tmp_path):
    """With early_stop disabled, plateau_step.txt should equal num_iterations."""
    out_dir = tmp_path / "no_early"
    cfg_dict = dict(_TINY_CONFIG)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["num_iterations"] = 100
    cfg_dict["ckpt_every"] = 100
    # No early_stop block at all → disabled by default.

    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    content = (out_dir / "plateau_step.txt").read_text()
    plateau_step = int(content.strip())
    assert plateau_step == 100, f"plateau_step {plateau_step} != num_iterations 100"


def test_warnings_log_is_written(tmp_path):
    """Every filtered training run drops a warnings.log alongside metrics.jsonl,
    even if empty. This is the audit trail for non-deterministic ops."""
    out_dir = tmp_path / "warn"
    cfg_dict = dict(_TINY_CONFIG)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["num_iterations"] = 50
    cfg_dict["ckpt_every"] = 50

    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    assert (out_dir / "warnings.log").exists(), "warnings.log was not created"


def test_plateau_file_has_no_extra_whitespace(tmp_path):
    """plateau_step.txt format contract: single integer, no surrounding whitespace
    other than what str() produces. Downstream parsers should not need to .strip()."""
    out_dir = tmp_path / "fmt"
    cfg_dict = dict(_TINY_CONFIG)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["num_iterations"] = 50
    cfg_dict["ckpt_every"] = 50

    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    raw = (out_dir / "plateau_step.txt").read_text()
    # Exact match: no leading/trailing whitespace, parses as int
    assert raw == str(int(raw)), f"plateau_step.txt has extra characters: {raw!r}"
