"""End-to-end smoke for entropy_filtered/.

Hard contract:
  - All entropy-filter unit tests pass (run separately).
  - Filtered training runs to completion without error.
  - Filter actually fires (n_dropped > 0 at some logged step after warmup).
  - Final loss < initial loss.
  - Both vanilla and adaptive inference produce valid sequences (delegates to baseline).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baseline.src.data import LoNaeSatConfig, generate_dataset
from baseline.src.evaluate import evaluate_lo_nae_sat
from baseline.src.model import Transformer, TransformerConfig
from baseline.src.utils import load_config, set_global_seed
from entropy_filtered.src.train_filtered import (
    _to_filtered_config, run_filtered_training,
)


def main() -> int:
    cfg_path = ROOT / "entropy_filtered" / "configs" / "lo_nae_sat_smoke_band.yaml"
    cfg_dict = load_config(str(cfg_path))
    cfg = _to_filtered_config(cfg_dict)

    print("=" * 70)
    print(f"ENTROPY-FILTERED SMOKE  ({cfg_path.name})")
    print("=" * 70)
    print(f"  mode:           {cfg.entropy_filter['mode']}")
    print(f"  warmup_steps:   {cfg.entropy_filter['warmup_steps']}")
    print(f"  iterations:     {cfg.num_iterations}")
    print(f"  batch_size:     {cfg.batch_size}")
    print(f"  device:         {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    output_dir = ROOT / "entropy_filtered" / "runs" / "smoke_band"
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    run_filtered_training(cfg)
    train_seconds = time.monotonic() - t0

    rows = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        print("FAIL: not enough metric rows")
        return 1

    initial_loss = rows[0]["loss"]
    final_loss = rows[-1]["loss"]
    print(f"\n[smoke-filtered] training done in {train_seconds:.1f}s")
    print(f"[smoke-filtered] initial loss:  {initial_loss:.4f}")
    print(f"[smoke-filtered] final   loss:  {final_loss:.4f}")

    if not (final_loss < initial_loss):
        print("FAIL: loss did not decrease")
        return 1

    # The filter must have fired at least once after the warmup window
    post_warmup = [r for r in rows if r["step"] > cfg.entropy_filter["warmup_steps"]]
    if not post_warmup:
        print("FAIL: no logged steps after warmup window")
        return 1
    fired = any(r.get("filter_n_dropped", 0) > 0 for r in post_warmup)
    if not fired:
        print("FAIL: filter never dropped a single sample after warmup; misconfigured?")
        return 1
    n_dropped_total = sum(r.get("filter_n_dropped", 0) for r in post_warmup)
    print(f"[smoke-filtered] filter fired: total dropped (over logged steps) = {n_dropped_total}")

    # Inference sanity: model produces valid sequences
    set_global_seed(cfg.seed)
    data_cfg = LoNaeSatConfig(**cfg.data)
    test_seqs, _ = generate_dataset(data_cfg, num_samples=100, sample_seed=999)

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

    g = torch.Generator(device="cpu").manual_seed(0)
    vanilla = evaluate_lo_nae_sat(
        model, test_seqs, data_cfg,
        strategy="vanilla", num_steps=20, noise="none", noise_scale=0.0,
        num_eval=50, device="cpu", generator=g,
    )
    adaptive = evaluate_lo_nae_sat(
        model, test_seqs, data_cfg,
        strategy="top_prob_margin", num_steps=20, noise="gumbel", noise_scale=0.5,
        num_eval=50, device="cpu", generator=g,
    )
    print(f"[smoke-filtered] vanilla  obs_acc: {vanilla.obs_accuracy:.4f}")
    print(f"[smoke-filtered] adaptive obs_acc: {adaptive.obs_accuracy:.4f}")

    if adaptive.obs_accuracy + 0.1 < vanilla.obs_accuracy:
        print("FAIL: adaptive much worse than vanilla on filter-trained model")
        return 1

    total_seconds = time.monotonic() - t0
    print()
    print("=" * 70)
    print(f"ENTROPY-FILTERED SMOKE PASSED  ({total_seconds:.1f}s)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
