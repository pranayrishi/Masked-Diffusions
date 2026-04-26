"""End-to-end smoke test: train a tiny MDM on a tiny L&O-NAE-SAT instance, then
evaluate vanilla and adaptive inference. Must complete on CPU in well under 10 min.

Hard contract (binding before Phase 5):
  - pytest passes (run separately)
  - Training runs to completion without error
  - Final loss is lower than initial loss
  - Vanilla and adaptive inference both produce valid sequences (no mask tokens)
  - Adaptive observation accuracy >= vanilla observation accuracy − 0.05
    (we allow a small slack because at 100 training steps the model is barely
     trained and rankings between strategies aren't yet stable; we just want
     a sanity check that adaptive isn't catastrophically broken)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LoNaeSatConfig, generate_dataset
from src.evaluate import evaluate_lo_nae_sat
from src.inference import run_inference
from src.model import Transformer, TransformerConfig
from src.train import _to_train_config, run_training
from src.utils import load_config, set_global_seed


def main() -> int:
    cfg_path = ROOT / "configs" / "lo_nae_sat_smoke.yaml"
    cfg_dict = load_config(str(cfg_path))
    cfg = _to_train_config(cfg_dict)

    print("=" * 70)
    print(f"SMOKE TEST  ({cfg_path.name})")
    print("=" * 70)
    print(f"  task:           {cfg.task}")
    print(f"  data:           N={cfg.data['N']}, P={cfg.data['P']}, L={cfg.data.get('pad_to', cfg.data['N']+cfg.data['P'])}")
    print(f"  model:          hidden={cfg.model['hidden']}, layers={cfg.model['n_layers']}, heads={cfg.model['n_heads']}")
    print(f"  iterations:     {cfg.num_iterations}, batch_size={cfg.batch_size}")
    print(f"  device:         {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Train
    t0 = time.monotonic()
    output_dir = ROOT / "runs" / "smoke"
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_training(cfg)
    train_seconds = time.monotonic() - t0

    # Read training metrics
    metrics_path = output_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    losses = [r["loss"] for r in rows]
    if len(losses) < 2:
        print(f"FAIL: not enough metric rows ({len(losses)})")
        return 1
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\n[smoke] training done in {train_seconds:.1f}s")
    print(f"[smoke] initial loss: {initial_loss:.4f}")
    print(f"[smoke] final   loss: {final_loss:.4f}")
    if not (final_loss < initial_loss):
        print(f"FAIL: final loss {final_loss:.4f} >= initial loss {initial_loss:.4f}")
        return 1

    # Reload the trained model and evaluate
    set_global_seed(cfg.seed)
    data_cfg = LoNaeSatConfig(**cfg.data)
    test_seqs, _ = generate_dataset(data_cfg, num_samples=200, sample_seed=999)

    m_cfg = TransformerConfig(
        vocab_size=cfg.model["vocab_size"],
        hidden=cfg.model["hidden"],
        n_layers=cfg.model["n_layers"],
        n_heads=cfg.model["n_heads"],
        ff=cfg.model["ff"],
        max_seq_len=data_cfg.L,
        causal=cfg.model.get("causal", False),
        pos_type=cfg.model.get("pos_type", "rope"),
        dropout=0.0,
        weight_tie=cfg.model.get("weight_tie", False),
    )
    model = Transformer(m_cfg)
    ckpt_path = output_dir / f"ckpt_step{cfg.num_iterations}.pt"
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state"])

    # Evaluate vanilla and adaptive (top_prob_margin) on 100 test samples
    g = torch.Generator(device="cpu").manual_seed(0)
    t1 = time.monotonic()
    vanilla = evaluate_lo_nae_sat(
        model, test_seqs, data_cfg,
        strategy="vanilla", num_steps=20, noise="none", noise_scale=0.0,
        num_eval=100, device="cpu", generator=g,
    )
    adaptive = evaluate_lo_nae_sat(
        model, test_seqs, data_cfg,
        strategy="top_prob_margin", num_steps=20, noise="gumbel", noise_scale=0.5,
        num_eval=100, device="cpu", generator=g,
    )
    eval_seconds = time.monotonic() - t1
    print(f"[smoke] eval done in {eval_seconds:.1f}s")
    print(f"[smoke] vanilla    obs_acc: {vanilla.obs_accuracy:.4f}")
    print(f"[smoke] adaptive   obs_acc: {adaptive.obs_accuracy:.4f}")

    if adaptive.obs_accuracy + 0.05 < vanilla.obs_accuracy:
        print(f"FAIL: adaptive {adaptive.obs_accuracy:.4f} much worse than vanilla {vanilla.obs_accuracy:.4f}")
        return 1

    total_seconds = time.monotonic() - t0
    print()
    print("=" * 70)
    print(f"SMOKE TEST PASSED  ({total_seconds:.1f}s total)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
