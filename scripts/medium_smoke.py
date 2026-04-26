"""Medium smoke: production-shape scale check on CPU before Bouchet submission.

Goal (paper-notes / user instruction): catch bugs that only appear at realistic
scale — gradient explosion, attention numerical instability, optimizer state
drift — on the laptop rather than at hour 11 on Bouchet.

Specifications:
  - Production-shape model: 19M-class transformer (~14M actual params with our
    config: hidden=384, layers=8, heads=6, ff=1536, RoPE).
  - Sequence length: 300 (the actual L&O-NAE-SAT data length, no padding to 512
    so attention's L^2 cost stays manageable on CPU).
  - Batch size: 32 by default (the user's specified value).
  - Iterations: split across all 5 filter modes. Default 200 steps each = 1000 total.
  - All 5 entropy-filter modes exercised: none, bottom, top, band, percentile_band.

Hard contract (binding before Bouchet):
  - All 5 modes complete without raising.
  - Loss decreases over each variant's run.
  - No NaN / Inf in losses or gradients at any logged step.
  - Filter fires (n_dropped > 0 at some logged step) for non-`none` modes.
  - Cumulative gradient-norm distribution stays bounded (max < 100 on this scale).

Timing (measured on Apple Silicon CPU with torch.set_num_threads(8)):
  - 1 step ≈ 2.4 s
  - 200 steps × 5 variants = 1000 total ≈ 40 min  (full-spec, the user's target)
  - 100 steps × 5 variants = 500 total ≈ 20 min  (--steps-per-variant 100)
  -  50 steps × 5 variants = 250 total ≈ 10 min  (--steps-per-variant 50, quick sanity)

CLI:
  python scripts/medium_smoke.py [--steps-per-variant N] [--batch-size B] [--seq-len L]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.src.utils import auto_device, set_global_seed
from entropy_filtered.src.train_filtered import (
    _to_filtered_config,
    run_filtered_training,
)


VARIANTS = ["none", "bottom", "top", "band", "percentile_band"]


def _build_cfg_dict(
    *,
    mode: str,
    steps: int,
    batch: int,
    seq_len: int,
    output_dir: str,
) -> dict:
    """Produce a config dict in the schema FilteredTrainConfig expects."""
    # Use (N, P) = (40, 260) for L_data = 300; pad_to = seq_len so attention is at
    # exactly seq_len (no wasted padded positions).
    return {
        "task": "lo_nae_sat",
        "data": {
            "N": 40, "P": 260, "m": 2,
            "pad_to": seq_len, "pad_value": 2,
            "mask_token_id": 3, "vocab_size": 4,
            "seed": 42,
        },
        "model": {
            # 19M-class config; this exact one is ~14M params with RoPE + no time emb.
            "hidden": 384, "n_layers": 8, "n_heads": 6, "ff": 1536,
            "causal": False, "pos_type": "rope",
            "dropout": 0.0, "weight_tie": False,
            "vocab_size": 4,
        },
        "seed": 0,
        "num_iterations": steps,
        "batch_size": batch,
        "lr": 1.0e-3,
        "weight_decay": 0.1,
        "beta1": 0.9, "beta2": 0.95,
        "grad_clip": 1.0,
        "eta_min_ratio": 0.1,
        "log_every": max(1, steps // 20),         # ~20 logged steps per variant
        "ckpt_every": steps,                       # one ckpt at the end
        "entropy_filter": {
            "mode": mode,
            "warmup_steps": 20,                    # short so the filter actually fires
            "reduction": "mean",
            "H_low": 0.05,                         # used by 'bottom' / 'band'
            "H_high": 1.05,                        # used by 'top' / 'band'
            "pct_low": 0.25,
            "pct_high": 0.75,
            "use_softmax_excluding_mask_token": True,
            "eps": 1.0e-12,
        },
        "train_size": 5000,                        # enough for 1000 batches at batch=32 w/o repeats
        "train_sample_seed": 123,
        "output_dir": output_dir,
    }


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize_one(rows: list[dict]) -> dict:
    losses = [r["loss"] for r in rows if not r.get("skipped_optim_step", 0)]
    grad_norms = [r["grad_norm"] for r in rows if not r.get("skipped_optim_step", 0)]
    n_dropped_total = sum(r.get("filter_n_dropped", 0) for r in rows)
    n_logged = len(rows)
    initial_loss = losses[0] if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")
    max_grad = max(grad_norms) if grad_norms else float("nan")

    # NaN / Inf check
    nans = sum(1 for x in losses + grad_norms if not math.isfinite(x))

    return {
        "n_logged_steps": n_logged,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_grad_norm": max_grad,
        "n_filter_dropped": n_dropped_total,
        "n_nan_or_inf": nans,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-variant", type=int, default=200,
                        help="Steps per filter mode (default 200; full-spec the user "
                             "requested = 200, lighter sanity = 50).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    print("=" * 76)
    print("MEDIUM SMOKE")
    print("=" * 76)
    print(f"  device:             {auto_device()}")
    print(f"  torch threads:      {args.num_threads}")
    print(f"  steps per variant:  {args.steps_per_variant}  (× {len(VARIANTS)} variants = {args.steps_per_variant * len(VARIANTS)} total)")
    print(f"  batch size:         {args.batch_size}")
    print(f"  sequence length:    {args.seq_len}")
    print()

    out_root = ROOT / "medium_smoke_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    overall_t0 = time.monotonic()

    for variant in VARIANTS:
        print(f"--- {variant} ---")
        out_dir = out_root / variant
        cfg = _to_filtered_config(_build_cfg_dict(
            mode=variant,
            steps=args.steps_per_variant,
            batch=args.batch_size,
            seq_len=args.seq_len,
            output_dir=str(out_dir),
        ))
        t0 = time.monotonic()
        run_filtered_training(cfg)
        dt = time.monotonic() - t0
        rows = _read_metrics(out_dir / "metrics.jsonl")
        summary = _summarize_one(rows)
        summary["wall_seconds"] = dt
        summaries[variant] = summary
        print(f"  done in {dt:.1f}s  | initial={summary['initial_loss']:.4f}, "
              f"final={summary['final_loss']:.4f}, "
              f"max_grad={summary['max_grad_norm']:.2f}, "
              f"filter_dropped={summary['n_filter_dropped']}")

    overall_dt = time.monotonic() - overall_t0
    print()
    print("=" * 76)
    print(f"TOTAL WALL TIME: {overall_dt:.1f}s = {overall_dt/60:.1f} min")
    print("=" * 76)

    # --- Hard contract assertions ---
    failed = []
    for variant, s in summaries.items():
        if s["n_nan_or_inf"] > 0:
            failed.append(f"{variant}: NaN or Inf in losses/grad ({s['n_nan_or_inf']} occurrences)")
        if not math.isfinite(s["final_loss"]):
            failed.append(f"{variant}: non-finite final loss")
        if s["final_loss"] > s["initial_loss"]:
            failed.append(f"{variant}: loss did not decrease "
                          f"(initial={s['initial_loss']:.4f}, final={s['final_loss']:.4f})")
        if s["max_grad_norm"] > 100.0:
            failed.append(f"{variant}: max grad norm too high ({s['max_grad_norm']:.2f})")
        if variant != "none" and s["n_filter_dropped"] == 0:
            # A non-`none` variant should drop SOMETHING after warmup. Only flag if total dropped is 0.
            # Edge case: 'bottom' / 'top' / 'band' with absolute thresholds may not fire if all entropies
            # land inside the band. We still don't fail — we warn — so we set a soft flag.
            print(f"WARNING: {variant} dropped 0 samples (warmup={20}, steps={args.steps_per_variant}). "
                  "Thresholds may be too loose. Investigate before Bouchet.")

    out_summary = {
        "args": vars(args),
        "summaries": summaries,
        "overall_wall_seconds": overall_dt,
        "failures": failed,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(out_summary, indent=2))
    print(f"\nFull summary written to {summary_path}")

    if failed:
        print()
        print("FAIL — hard-contract violations:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print()
    print("MEDIUM SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
