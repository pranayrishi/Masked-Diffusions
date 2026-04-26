"""Determinism tests — bit-exact reproducibility of training runs.

Reviewer-mandatory: a re-run of the same config with the same seed must produce
bit-identical loss trajectories. Without this, "filter A beats filter B by 0.7%"
is unfalsifiable — a re-run could land 0.7% the other way for free.

Two layers of assertion:

  1. `set_global_seed` enables the determinism flags. We check the flags are SET,
     not just that the result happens to be reproducible.

  2. Two fresh training runs with the same seed produce bit-equal losses at every
     logged step AND bit-equal final model parameters (within 1e-7, which is the
     float-comparison floor for what "bit-equal" means in practice on CPU FP32).

Note: this test runs on CPU. On CUDA, additional flags (CUBLAS_WORKSPACE_CONFIG,
cudnn.deterministic) come into play. Those are set in `set_global_seed` and
covered by `test_set_global_seed_enables_deterministic_algorithms`; the bit-exact
behavior on CUDA can only be verified on the cluster, not on this laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.train import _to_train_config, run_training
from src.utils import set_global_seed


_CONFIG = {
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
    "num_iterations": 60,
    "batch_size": 32,
    "lr": 1.0e-3,
    "weight_decay": 0.1,
    "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0,
    "eta_min_ratio": 0.1,
    "log_every": 10,
    "ckpt_every": 60,
    "train_size": 500,
    "train_sample_seed": 123,
    "output_dir": "PLACEHOLDER",
}


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. set_global_seed turns on the determinism flags
# ---------------------------------------------------------------------------

def test_set_global_seed_enables_deterministic_algorithms():
    set_global_seed(7)
    assert torch.are_deterministic_algorithms_enabled(), \
        "torch.use_deterministic_algorithms(True) was not honored"
    assert torch.backends.cudnn.deterministic, "cudnn.deterministic should be True"
    assert torch.backends.cudnn.benchmark is False, "cudnn.benchmark should be False"


def test_set_global_seed_sets_required_env_vars():
    """CUBLAS_WORKSPACE_CONFIG must be in env BEFORE the first cuBLAS call;
    PYTHONHASHSEED must be set so str-keyed dict iteration is deterministic."""
    import os
    set_global_seed(13)
    assert os.environ.get("PYTHONHASHSEED") == "13"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").startswith(":")  # ":4096:8" or shell-set value


# ---------------------------------------------------------------------------
# 2. Two fresh runs with the same seed → bit-equal loss + model params
# ---------------------------------------------------------------------------

def test_two_fresh_runs_produce_bit_identical_losses(tmp_path):
    """Run training to completion twice. Every logged loss must agree to 1e-7."""
    a_dir = tmp_path / "run_a"
    b_dir = tmp_path / "run_b"

    a_cfg_dict = dict(_CONFIG)
    a_cfg_dict["output_dir"] = str(a_dir)
    a_cfg = _to_train_config(a_cfg_dict)
    run_training(a_cfg)

    b_cfg_dict = dict(_CONFIG)
    b_cfg_dict["output_dir"] = str(b_dir)
    b_cfg = _to_train_config(b_cfg_dict)
    run_training(b_cfg)

    a_metrics = _read_metrics(a_dir / "metrics.jsonl")
    b_metrics = _read_metrics(b_dir / "metrics.jsonl")

    a_by_step = {r["step"]: r["loss"] for r in a_metrics}
    b_by_step = {r["step"]: r["loss"] for r in b_metrics}

    assert set(a_by_step.keys()) == set(b_by_step.keys()), \
        f"logged steps differ: {sorted(a_by_step.keys())} vs {sorted(b_by_step.keys())}"
    for step in sorted(a_by_step):
        diff = abs(a_by_step[step] - b_by_step[step])
        assert diff < 1e-7, \
            f"loss mismatch at step {step}: a={a_by_step[step]:.10f} b={b_by_step[step]:.10f} diff={diff:.2e}"


def test_two_fresh_runs_produce_bit_identical_final_params(tmp_path):
    """The final checkpoint's model_state must be parameter-by-parameter bit-equal
    (within 1e-7 abs diff) across two fresh runs with the same seed."""
    a_dir = tmp_path / "run_a"
    b_dir = tmp_path / "run_b"

    for d in (a_dir, b_dir):
        cfg_dict = dict(_CONFIG)
        cfg_dict["output_dir"] = str(d)
        cfg = _to_train_config(cfg_dict)
        run_training(cfg)

    a_ckpt = torch.load(str(a_dir / "ckpt_step60.pt"), map_location="cpu")
    b_ckpt = torch.load(str(b_dir / "ckpt_step60.pt"), map_location="cpu")

    assert a_ckpt["model_state"].keys() == b_ckpt["model_state"].keys()
    max_diff = 0.0
    for k in a_ckpt["model_state"]:
        diff = (a_ckpt["model_state"][k].float() - b_ckpt["model_state"][k].float()).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-7, \
        f"final-checkpoint params differ across fresh runs with same seed: max abs diff = {max_diff:.3e}"
