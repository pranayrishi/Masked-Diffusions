"""Filter-state resume determinism test — cluster-preemption insurance.

Hard contract (binding for Bouchet scavenge_gpu preemption resilience):
  - Reference filtered run: num_iterations=60, ckpt_every=30 saves at steps 30 and 60.
  - Resumed filtered run:   num_iterations=60, resume_from=<ref step-30 ckpt>; runs 31..60.
  - Assert: every per-step filter-state field at the post-resume steps matches the
            reference run bit-exactly:
              - filter_n_kept, filter_n_dropped
              - filter_H_min, filter_H_max, filter_H_mean
              - filter_loss_kept_mean, filter_loss_dropped_mean
              - skipped_optim_step
  - Assert: model parameters at step 60 are bit-equivalent (max abs diff < 1e-6).

Why this matters: the entropy filter's decisions depend on (model state, batch
contents, mask sampling). All three are deterministic if and only if:
  - The model is restored bit-exactly (covered by checkpoint model_state).
  - The data sampling at step k is deterministic in (seed, step) (covered by
    `batch_indices_for_step`).
  - The mask sampling consumes a torch RNG state that is bit-equivalent to the
    reference run's at the start of step k (covered by `restore_rng_states`).

If any of these breaks under preemption, the filter could make different decisions
on the same step post-resume — producing a subtle bias that this test catches.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from entropy_filtered.src.train_filtered import _to_filtered_config, run_filtered_training


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
    "log_every": 5,
    "ckpt_every": 30,
    # percentile_band guarantees the filter fires at every step (after warmup),
    # so we exercise the most filter-state-rich code path.
    "entropy_filter": {
        "mode": "percentile_band",
        "warmup_steps": 0,
        "reduction": "mean",
        "H_low": 0.0,
        "H_high": 1e9,
        "pct_low": 0.25,
        "pct_high": 0.75,
        "use_softmax_excluding_mask_token": True,
        "eps": 1e-12,
    },
    "train_size": 500,
    "train_sample_seed": 123,
    "output_dir": "PLACEHOLDER",
}


# Filter-state fields that must agree bit-exactly between ref and resume
_FILTER_FIELDS_EXACT_INT = ("filter_n_kept", "filter_n_dropped", "skipped_optim_step")
_FILTER_FIELDS_EXACT_FLOAT = ("filter_H_min", "filter_H_max", "filter_H_mean")
_FILTER_FIELDS_NULLABLE_FLOAT = ("filter_loss_kept_mean", "filter_loss_dropped_mean")


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_filter_resume_preserves_per_step_decisions_and_diagnostics(tmp_path):
    # === Reference filtered run: 1..60 in one go, ckpt at step 30 and 60 ===
    ref_dir = tmp_path / "ref"
    ref_cfg = _to_filtered_config({**_CONFIG, "output_dir": str(ref_dir)})
    run_filtered_training(ref_cfg)

    ref_metrics = _read_metrics(ref_dir / "metrics.jsonl")
    ref_step60_ckpt = torch.load(str(ref_dir / "ckpt_step60.pt"), map_location="cpu")

    # === Resumed filtered run: same num_iterations, resume from ref's step-30 ckpt ===
    res_dir = tmp_path / "res"
    res_cfg = _to_filtered_config({**_CONFIG, "output_dir": str(res_dir)})
    run_filtered_training(res_cfg, resume_from=str(ref_dir / "ckpt_step30.pt"))

    res_metrics = _read_metrics(res_dir / "metrics.jsonl")
    res_step60_ckpt = torch.load(str(res_dir / "ckpt_step60.pt"), map_location="cpu")

    # --- Assertion 1: every common logged step has bit-equal filter state ---
    ref_by_step = {r["step"]: r for r in ref_metrics}
    res_by_step = {r["step"]: r for r in res_metrics}

    common = sorted(set(ref_by_step) & set(res_by_step))
    # res only contains steps > 30, so common = post-resume logged steps
    assert all(s > 30 for s in common), f"unexpected common steps: {common}"
    assert len(common) >= 4, f"too few common log steps: {common}"

    for step in common:
        ref_row = ref_by_step[step]
        res_row = res_by_step[step]

        # Loss should also match (covered by test_resume in baseline, but verify here too)
        diff = abs(ref_row["loss"] - res_row["loss"])
        assert diff < 1e-6, \
            f"loss mismatch at step {step}: ref={ref_row['loss']:.10f} res={res_row['loss']:.10f}"

        for f in _FILTER_FIELDS_EXACT_INT:
            assert ref_row[f] == res_row[f], \
                f"{f} mismatch at step {step}: ref={ref_row[f]} res={res_row[f]}"

        for f in _FILTER_FIELDS_EXACT_FLOAT:
            d = abs(ref_row[f] - res_row[f])
            assert d < 1e-6, \
                f"{f} mismatch at step {step}: ref={ref_row[f]:.10f} res={res_row[f]:.10f} diff={d:.2e}"

        for f in _FILTER_FIELDS_NULLABLE_FLOAT:
            ref_val = ref_row[f]
            res_val = res_row[f]
            if ref_val is None:
                assert res_val is None, \
                    f"{f} at step {step}: ref=None but res={res_val}"
            else:
                assert res_val is not None, \
                    f"{f} at step {step}: ref={ref_val} but res=None"
                d = abs(ref_val - res_val)
                assert d < 1e-6, \
                    f"{f} mismatch at step {step}: ref={ref_val:.10f} res={res_val:.10f} diff={d:.2e}"

    # --- Assertion 2: final model state matches across resume (1e-6 abs diff) ---
    ref_state = ref_step60_ckpt["model_state"]
    res_state = res_step60_ckpt["model_state"]
    assert ref_state.keys() == res_state.keys()
    max_diff = 0.0
    for k in ref_state:
        d = (ref_state[k].float() - res_state[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff < 1e-6, \
        f"model parameters differ after filter resume: max abs diff = {max_diff:.3e}"


def test_filter_resume_step_counter_advances_correctly(tmp_path):
    """If resume_from points at a checkpoint at step 30, the resumed run starts at 31."""
    a_dir = tmp_path / "a"
    cfg_a = _to_filtered_config({**_CONFIG, "output_dir": str(a_dir), "num_iterations": 30})
    run_filtered_training(cfg_a)

    b_dir = tmp_path / "b"
    cfg_b = _to_filtered_config({**_CONFIG, "output_dir": str(b_dir), "num_iterations": 60})
    run_filtered_training(cfg_b, resume_from=str(a_dir / "ckpt_step30.pt"))

    rows = _read_metrics(b_dir / "metrics.jsonl")
    steps = sorted({r["step"] for r in rows})
    assert all(s > 30 for s in steps), f"resumed run logged steps <= 30: {steps}"
    assert 60 in steps, f"resumed run did not reach step 60: {steps}"
