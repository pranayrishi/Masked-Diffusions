"""Select the best entropy-filter variant from Phase C results.

Reads the aggregate from aggregate_findings.discover_runs() and picks the variant
with highest cross-config mean accuracy (under a chosen inference strategy).

Writes:
  - <out_dir>/best_variant.txt  (single line: e.g., "top_065")
  - <out_dir>/variant_ranking.json (full ranking + per-config breakdown)

This drives Phase E (5-seed bump) submission.

Usage:
    python scripts/select_best_variant.py \\
        --results-root <project>/results \\
        --output-dir <project>/findings \\
        --metric vanilla    # or "adaptive"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Allow importing from the same scripts/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_findings import discover_runs, aggregate_by_condition  # noqa: E402


# Variants we consider as candidates. Random-replay variants are excluded — we want
# the best ENTROPY filter, not the best of any.
ENTROPY_VARIANTS = [
    "top_055", "top_060", "top_065", "top_070",
    "bottom_030",
    "band_030_055", "band_030_060", "band_030_065", "band_030_070",
    "percentile",
]

CONFIG_ORDER = ["(25, 275)", "(30, 270)", "(40, 260)", "(50, 250)", "(100, 200)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metric", choices=["vanilla", "adaptive"], default="vanilla")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_root)
    print(f"[select_best] discovered {len(runs)} runs across all phases")

    agg = aggregate_by_condition(runs)
    metric_key = f"{args.metric}_mean"

    # Per-variant cross-config mean (only Phase C entries)
    rankings: list[tuple[str, float, dict]] = []
    for v in ENTROPY_VARIANTS:
        per_config_acc = []
        per_config_breakdown = {}
        for c in CONFIG_ORDER:
            k = ("phase_c", c, v)
            if k in agg and not math.isnan(agg[k][metric_key]):
                per_config_acc.append(agg[k][metric_key])
                per_config_breakdown[c] = agg[k][metric_key]
            else:
                per_config_breakdown[c] = None
        if not per_config_acc:
            continue
        cross_mean = sum(per_config_acc) / len(per_config_acc)
        rankings.append((v, cross_mean, per_config_breakdown))

    if not rankings:
        print("[select_best] FAIL: no Phase C results found", file=sys.stderr)
        return 2

    rankings.sort(key=lambda x: -x[1])
    best_variant = rankings[0][0]
    best_acc = rankings[0][1]

    print(f"[select_best] using metric={args.metric}_mean across {len(CONFIG_ORDER)} configs")
    print(f"[select_best] ranked entropy variants:")
    for v, m, _ in rankings:
        print(f"   {v:14s}  {m:.4%}")
    print(f"[select_best] BEST: {best_variant} (cross-config mean = {best_acc:.4%})")

    # Write outputs
    (out_dir / "best_variant.txt").write_text(best_variant + "\n")

    ranking_data = {
        "metric": args.metric,
        "best_variant": best_variant,
        "best_cross_config_mean": best_acc,
        "rankings": [
            {"variant": v, "cross_config_mean": m, "per_config": breakdown}
            for v, m, breakdown in rankings
        ],
    }
    (out_dir / "variant_ranking.json").write_text(json.dumps(ranking_data, indent=2))
    print(f"[select_best] wrote {out_dir}/best_variant.txt and variant_ranking.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
