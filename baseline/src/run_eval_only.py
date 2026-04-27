"""CLI entry-point: load a checkpoint and run held-out evaluation at a given sample
count, writing eval_results.json into the checkpoint's directory or a custom output.

Used by Phase F (10K-sample camera-ready re-evaluation of headline conditions). The
training run has already saved ckpt_step*.pt + config.yaml; this script loads them,
generates a held-out test set with `eval_test_seed` (which the production runs locked
to 99999, distinct from train_sample_seed=123), and runs each strategy in
`--strategies`.

Usage:

    python -m baseline.src.run_eval_only \\
        --run-dir /path/to/results/phase_c/phase_c_25_275_top_065_seed0-12345_2 \\
        --num-samples 10000 \\
        --strategies vanilla,top_prob_margin \\
        --num-steps 50 \\
        --output-name eval_results_10k.json

The script picks the LATEST ckpt_step*.pt in --run-dir unless --ckpt is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

# Make `baseline` importable regardless of where this script is invoked from.
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent.parent
sys.path.insert(0, str(REPO))

from baseline.src.data import LoNaeSatConfig, generate_dataset           # noqa: E402
from baseline.src.evaluate import evaluate_lo_nae_sat                    # noqa: E402
from baseline.src.model import Transformer, TransformerConfig            # noqa: E402
from baseline.src.utils import auto_device, load_checkpoint, set_global_seed   # noqa: E402


def _build_model(model_cfg: dict, max_seq_len: int) -> Transformer:
    return Transformer(TransformerConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden=model_cfg["hidden"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        ff=model_cfg["ff"],
        max_seq_len=max_seq_len,
        causal=model_cfg.get("causal", False),
        pos_type=model_cfg.get("pos_type", "rope"),
        dropout=model_cfg.get("dropout", 0.0),
        weight_tie=model_cfg.get("weight_tie", False),
    ))


def _latest_ckpt(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("ckpt_step*.pt"),
                        key=lambda p: int(p.stem.replace("ckpt_step", "")))
    if not candidates:
        raise FileNotFoundError(f"no ckpt_step*.pt in {run_dir}")
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="Specific checkpoint path; default = latest in run-dir")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--strategies", default="vanilla,top_prob_margin")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--noise", default="gumbel")     # paper Table 1 uses Gumbel
    ap.add_argument("--noise-scale", type=float, default=0.5)
    ap.add_argument("--eval-test-seed", type=int, default=99999)
    ap.add_argument("--output-name", default="eval_results.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else _latest_ckpt(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    config_data = yaml.safe_load(cfg_path.read_text())
    train_sample_seed = int(config_data.get("train_sample_seed", 123))
    if args.eval_test_seed == train_sample_seed:
        raise ValueError(f"eval_test_seed must differ from train_sample_seed (both {train_sample_seed})")

    set_global_seed(int(config_data.get("seed", 0)))
    device = auto_device()

    # Build & load model
    data_cfg = LoNaeSatConfig(**config_data["data"])
    model = _build_model(config_data["model"], max_seq_len=data_cfg.L).to(device)
    state = load_checkpoint(str(ckpt_path))
    model.load_state_dict(state.model_state)
    print(f"[run_eval_only] loaded ckpt {ckpt_path} (step={state.step}); device={device}")

    # Generate held-out test set
    print(f"[run_eval_only] generating {args.num_samples} test samples (seed={args.eval_test_seed})")
    test_seqs, _ = generate_dataset(
        data_cfg, num_samples=args.num_samples, sample_seed=args.eval_test_seed,
    )

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    eval_t0 = time.monotonic()
    results = []
    for strategy in strategies:
        print(f"[run_eval_only] eval strategy={strategy} num_steps={args.num_steps}")
        ev_t0 = time.monotonic()
        res = evaluate_lo_nae_sat(
            model, test_seqs, data_cfg,
            strategy=strategy,
            num_steps=args.num_steps,
            noise=args.noise,
            noise_scale=args.noise_scale,
            num_eval=args.num_samples,
            device=device,
        )
        ev_elapsed = time.monotonic() - ev_t0
        print(f"[run_eval_only]  {strategy}: accuracy={res.obs_accuracy:.4%} "
              f"({res.obs_correct}/{res.obs_total}) in {ev_elapsed:.1f}s")
        results.append({
            "strategy": strategy,
            "num_steps": args.num_steps,
            "noise": args.noise,
            "noise_scale": args.noise_scale,
            "num_samples": res.num_eval_samples,
            "obs_correct": res.obs_correct,
            "obs_total": res.obs_total,
            "obs_accuracy": res.obs_accuracy,
            "wall_time_s": ev_elapsed,
        })

    eval_total = time.monotonic() - eval_t0
    summary = {
        "ckpt_step": state.step,
        "ckpt_path": str(ckpt_path),
        "eval_test_seed": args.eval_test_seed,
        "eval_total_wall_time_s": eval_total,
        "results": results,
    }
    out_path = run_dir / args.output_name
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[run_eval_only] wrote {out_path} (eval total {eval_total:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
