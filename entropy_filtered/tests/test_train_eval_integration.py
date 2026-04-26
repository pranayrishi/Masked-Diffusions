"""End-to-end integration test: training + held-out evaluation + eval_results.json.

A tiny calibration-style run with eval_num_samples > 0 should:
  1. Train to completion (or early-stop).
  2. Generate a held-out test set with seed != train_sample_seed.
  3. Run inference under each requested strategy.
  4. Write eval_results.json with one entry per strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

from entropy_filtered.src.train_filtered import _to_filtered_config, run_filtered_training


_TINY = {
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
    "num_iterations": 50,
    "batch_size": 32,
    "lr": 1.0e-3,
    "weight_decay": 0.1,
    "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0,
    "eta_min_ratio": 0.1,
    "log_every": 10,
    "ckpt_every": 50,
    "entropy_filter": {"mode": "none", "warmup_steps": 0},
    "train_size": 200,
    "train_sample_seed": 123,
    "output_dir": "PLACEHOLDER",
}


def test_eval_runs_at_end_and_writes_eval_results_json(tmp_path):
    out_dir = tmp_path / "run"
    cfg_dict = dict(_TINY)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["eval_num_samples"] = 5            # tiny, just to exercise the code path
    cfg_dict["eval_test_seed"] = 777            # != train_sample_seed (123)
    cfg_dict["eval_strategies"] = ["vanilla", "top_prob_margin"]
    cfg_dict["eval_num_steps"] = 5
    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    eval_path = out_dir / "eval_results.json"
    assert eval_path.exists(), "eval_results.json was not written"

    summary = json.loads(eval_path.read_text())
    # Required top-level fields
    assert "results" in summary
    assert "final_step" in summary
    assert summary["eval_test_seed"] == 777

    # One entry per strategy
    by_strategy = {r["strategy"]: r for r in summary["results"]}
    assert set(by_strategy) == {"vanilla", "top_prob_margin"}

    for s, r in by_strategy.items():
        assert r["num_samples"] == 5
        assert r["obs_total"] > 0
        assert 0.0 <= r["obs_accuracy"] <= 1.0
        assert r["obs_correct"] + (r["obs_total"] - r["obs_correct"]) == r["obs_total"]


def test_eval_skipped_when_num_samples_zero(tmp_path):
    """eval_num_samples == 0 is the default path for non-eval runs (calibration)."""
    out_dir = tmp_path / "noeval"
    cfg_dict = dict(_TINY)
    cfg_dict["output_dir"] = str(out_dir)
    # No eval_num_samples key → defaults to 0
    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg)

    assert not (out_dir / "eval_results.json").exists(), \
        "eval_results.json was created despite eval_num_samples=0"


def test_eval_rejects_test_seed_equal_to_train_seed(tmp_path):
    """Held-out evaluation requires eval_test_seed != train_sample_seed."""
    import pytest
    out_dir = tmp_path / "bad"
    cfg_dict = dict(_TINY)
    cfg_dict["output_dir"] = str(out_dir)
    cfg_dict["eval_num_samples"] = 3
    cfg_dict["eval_test_seed"] = cfg_dict["train_sample_seed"]   # bug: same seed
    cfg = _to_filtered_config(cfg_dict)
    with pytest.raises(ValueError, match="eval_test_seed must differ"):
        run_filtered_training(cfg)
