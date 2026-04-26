"""Checkpoint/resume round-trip determinism test.

Hard contract (binding for Bouchet scavenge_gpu preemption resilience):
  - Reference run: num_iterations=100; ckpt_every=50 saves checkpoints at steps 50 and 100.
  - Resumed run:   num_iterations=100, resume_from=<ref step-50 ckpt>; runs steps 51..100.
  - Assert: post-resume losses at steps 51..100 == reference losses at the same steps
            (bit-exact, abs diff < 1e-6 on CPU with fixed seeds).
  - Assert: final model state (step 100) matches reference within 1e-6.

The data-loader RNG is stateless: `batch_indices_for_step(seed, step, ...)` derives a
fresh `np.random.default_rng` from (seed, step), so resumed batches at step k match
ref's batches at step k. The torch RNG state is captured / restored in the checkpoint.

WHY THIS IS A BETTER TEST THAN "TRAIN HALF, THEN RESUME": if we change num_iterations
between the two `run_training` calls, the cosine LR scheduler (T_max=num_iterations)
produces different LRs at the same step. Resume into the same num_iterations as the
reference run is the only fair comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.train import _to_train_config, run_training


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
    "num_iterations": 100,
    "batch_size": 32,
    "lr": 1.0e-3,
    "weight_decay": 0.1,
    "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0,
    "eta_min_ratio": 0.1,
    "log_every": 10,
    "ckpt_every": 50,
    "train_size": 500,
    "train_sample_seed": 123,
    "output_dir": "PLACEHOLDER",
}


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_resume_produces_identical_loss_trajectory_and_model_state(tmp_path):
    # === Reference run: 1..100 in one go, with ckpt at step 50 ===
    ref_dir = tmp_path / "ref"
    ref_cfg_dict = dict(_CONFIG)
    ref_cfg_dict["output_dir"] = str(ref_dir)
    ref_cfg = _to_train_config(ref_cfg_dict)
    run_training(ref_cfg)

    ref_metrics = _read_metrics(ref_dir / "metrics.jsonl")
    ref_step100_ckpt = torch.load(str(ref_dir / "ckpt_step100.pt"), map_location="cpu")

    # === Resumed run: same num_iterations, resume from ref's step-50 ckpt ===
    # Run dir is fresh — it will only contain steps 51..100 in metrics.jsonl
    res_dir = tmp_path / "res"
    res_cfg_dict = dict(_CONFIG)
    res_cfg_dict["output_dir"] = str(res_dir)
    # IMPORTANT: keep num_iterations=100 to keep the cosine LR scheduler identical to ref.
    res_cfg = _to_train_config(res_cfg_dict)
    run_training(res_cfg, resume_from=str(ref_dir / "ckpt_step50.pt"))

    res_metrics = _read_metrics(res_dir / "metrics.jsonl")
    res_step100_ckpt = torch.load(str(res_dir / "ckpt_step100.pt"), map_location="cpu")

    # --- Assertion 1: every step that's logged in BOTH ref and res must agree (bit-exact). ---
    ref_by_step = {r["step"]: r["loss"] for r in ref_metrics}
    res_by_step = {r["step"]: r["loss"] for r in res_metrics}

    common = sorted(set(ref_by_step) & set(res_by_step))
    # The res run only logs steps > 50 (resumed from step 50, then ckpt_every=50 makes no
    # additional checkpoints until 100; log_every=10 logs steps 60, 70, ..., 100).
    assert len(common) >= 5, f"too few common log steps: {common}"
    for step in common:
        diff = abs(ref_by_step[step] - res_by_step[step])
        assert diff < 1e-6, \
            f"loss mismatch at step {step}: ref={ref_by_step[step]:.10f} vs res={res_by_step[step]:.10f} (diff={diff:.2e})"

    # --- Assertion 2: final model state matches within 1e-6 (CPU FP32 determinism) ---
    ref_state = ref_step100_ckpt["model_state"]
    res_state = res_step100_ckpt["model_state"]
    assert ref_state.keys() == res_state.keys()
    max_diff = 0.0
    for k in ref_state:
        diff = (ref_state[k].float() - res_state[k].float()).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-6, f"model parameters differ after resume: max abs diff = {max_diff:.3e}"


def test_resume_recovers_step_counter():
    """If resume_from points at step k, the resumed run starts at step k+1 (not at 1)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Train to step 50
        a = dict(_CONFIG)
        a["output_dir"] = f"{td}/a"
        a["num_iterations"] = 50
        run_training(_to_train_config(a))

        # Resume into a fresh dir, num_iterations=100. Should run steps 51..100.
        b = dict(_CONFIG)
        b["output_dir"] = f"{td}/b"
        b["num_iterations"] = 100
        run_training(_to_train_config(b), resume_from=f"{td}/a/ckpt_step50.pt")

        rows = _read_metrics(Path(f"{td}/b/metrics.jsonl"))
        steps = sorted({r["step"] for r in rows})
        # Should NOT contain any step <= 50 (resume started from 50 → next logged is 60)
        assert all(s > 50 for s in steps), f"resumed run logged steps <= 50: {steps}"
        # Should contain step 100
        assert 100 in steps, f"resumed run did not reach step 100: {steps}"
