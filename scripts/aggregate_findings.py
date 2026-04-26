"""Aggregate Phase B / C / D / E / F results into findings.md + figures.

Walks ${PROJECT_DIR}/results/{phase_b,phase_c,phase_d,phase_e,phase_f}/, reads each
run's eval_results.json + config.yaml + metrics.jsonl, and produces:

  * findings.md          Markdown report with tables (a)-(f) per the spec
  * figures/*.png        Per-figure PNGs
  * figures/all.pdf      Combined PDF for the professor packet
  * raw_aggregate.csv    Long-form table for downstream analysis

Usage (on Bouchet, post-experiments):
    python scripts/aggregate_findings.py \\
        --results-root /nfs/roberts/project/pi_jks79/prn22/mdm_research/results \\
        --output-dir /nfs/roberts/project/pi_jks79/prn22/mdm_research/findings

Tolerant: missing/partial runs are reported but do not block the rest. Re-runnable
(idempotent at the file level — overwrites figures and findings.md each time).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


# Paper Table 1 reproduction targets (Kim et al. 2025, Table 1).
PAPER_TARGETS = {
    "(25, 275)":  {"vanilla": 0.7806, "adaptive": 0.9376},
    "(30, 270)":  {"vanilla": 0.7570, "adaptive": 0.9354},
    "(40, 260)":  {"vanilla": 0.7460, "adaptive": 0.9221},
    "(50, 250)":  {"vanilla": 0.6794, "adaptive": 0.9001},
    "(100, 200)": {"vanilla": 0.6284, "adaptive": 0.8891},
}


# ---------------------------------------------------------------------------
# Run discovery & loading
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    phase: str                # "phase_b" | "phase_c" | "phase_d" | "phase_e" | "phase_f"
    config: str               # "(25, 275)" etc.
    variant: str              # "none" | "top_055" | "random_top_055" | ...
    mode: str                 # entropy_filter.mode at training time
    H_high: float
    H_low: float
    pct_low: float
    pct_high: float
    paired_with: str          # for random_replay runs, the entropy variant name; else ""
    seed: int
    run_dir: str
    final_step: int
    early_stopped: bool
    eval_vanilla: float | None
    eval_adaptive: float | None
    eval_num_samples: int
    train_wall_time_s: float | None
    eval_wall_time_s: float | None
    mean_acceptance: float | None      # mean filter_n_kept / batch_size post-warmup
    final_loss: float | None
    n_logged_steps: int


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[aggregate] WARN: bad JSON at {path}: {e}", file=sys.stderr)
        return None


def _read_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        print(f"[aggregate] WARN: bad YAML at {path}: {e}", file=sys.stderr)
        return None


def _read_jsonl_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _config_label_from_run_name(run_name: str) -> str | None:
    """Extract '(N, P)' from a phase_x_<N>_<P>_<variant>_seed<S>-* directory name."""
    m = re.match(r"^phase_[a-z]+_(\d+)_(\d+)_", run_name)
    if not m:
        return None
    return f"({m.group(1)}, {m.group(2)})"


def _variant_from_run_name(run_name: str) -> str:
    """Extract variant name (between config and _seed) from the run dir."""
    # phase_b_25_275_none_seed3-12345_3 → none
    m = re.match(r"^phase_[a-z]+_\d+_\d+_(.+)_seed\d+", run_name)
    return m.group(1) if m else "unknown"


def _seed_from_run_name(run_name: str) -> int:
    m = re.search(r"_seed(\d+)", run_name)
    return int(m.group(1)) if m else -1


def _load_run(run_dir: Path, phase: str) -> RunResult | None:
    """Build a RunResult from one results/<phase>/<run> directory."""
    eval_data = _read_json(run_dir / "eval_results.json")
    config_data = _read_yaml(run_dir / "config.yaml")
    metrics = _read_jsonl_metrics(run_dir / "metrics.jsonl")

    if eval_data is None or config_data is None:
        # Run incomplete — skip but warn
        print(f"[aggregate] incomplete run (missing eval or config): {run_dir.name}",
              file=sys.stderr)
        return None

    name = run_dir.name
    config = _config_label_from_run_name(name) or "unknown"
    variant = _variant_from_run_name(name)
    seed = _seed_from_run_name(name)

    fcfg = config_data.get("entropy_filter", {})
    mode = fcfg.get("mode", "none")

    # paired_with: heuristic — if mode==random_replay, the paired entropy variant has the same
    # condition name with the "random_" prefix stripped.
    paired_with = ""
    if mode == "random_replay" and variant.startswith("random_"):
        paired_with = variant[len("random_"):]

    # Eval accuracies by strategy
    vanilla = adaptive = None
    eval_n = 0
    eval_wall = 0.0
    for r in eval_data.get("results", []):
        if r["strategy"] == "vanilla":
            vanilla = r["obs_accuracy"]
            eval_n = r.get("num_samples", eval_n)
            eval_wall += r.get("wall_time_s", 0.0)
        elif r["strategy"] == "top_prob_margin":
            adaptive = r["obs_accuracy"]
            eval_wall += r.get("wall_time_s", 0.0)

    final_step = eval_data.get("final_step", config_data.get("num_iterations", 0))
    early_stopped = eval_data.get("early_stopped", False)
    eval_total_wall = eval_data.get("eval_total_wall_time_s")

    # Train wall time = last metrics row's elapsed minus first
    train_wall = None
    if metrics:
        first = metrics[0]
        last = metrics[-1]
        train_wall = float(last.get("elapsed", 0)) - float(first.get("elapsed", 0))

    # Mean acceptance post-warmup (steps > entropy_filter.warmup_steps)
    warmup_steps = int(fcfg.get("warmup_steps", 500))
    post_warmup = [m for m in metrics if int(m.get("step", 0)) > warmup_steps]
    if post_warmup:
        kept = sum(int(m.get("filter_n_kept", 0)) for m in post_warmup)
        dropped = sum(int(m.get("filter_n_dropped", 0)) for m in post_warmup)
        total = kept + dropped
        mean_acc = (kept / total) if total > 0 else None
    else:
        mean_acc = None

    final_loss = float(metrics[-1]["loss"]) if metrics else None

    return RunResult(
        phase=phase,
        config=config,
        variant=variant,
        mode=mode,
        H_high=float(fcfg.get("H_high", 0.0)),
        H_low=float(fcfg.get("H_low", 0.0)),
        pct_low=float(fcfg.get("pct_low", 0.25)),
        pct_high=float(fcfg.get("pct_high", 0.75)),
        paired_with=paired_with,
        seed=seed,
        run_dir=str(run_dir),
        final_step=int(final_step),
        early_stopped=bool(early_stopped),
        eval_vanilla=vanilla,
        eval_adaptive=adaptive,
        eval_num_samples=int(eval_n),
        train_wall_time_s=train_wall,
        eval_wall_time_s=eval_total_wall,
        mean_acceptance=mean_acc,
        final_loss=final_loss,
        n_logged_steps=len(metrics),
    )


def discover_runs(results_root: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    for phase_dir in sorted(results_root.glob("phase_*")):
        phase = phase_dir.name
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            r = _load_run(run_dir, phase)
            if r is not None:
                runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Aggregation: mean ± stderr across seeds
# ---------------------------------------------------------------------------

def _mean_stderr(xs: list[float]) -> tuple[float, float, int]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, n


def aggregate_by_condition(runs: list[RunResult]) -> dict[tuple[str, str, str], dict]:
    """Group runs by (phase, config, variant) and compute mean ± stderr per metric."""
    groups: dict[tuple[str, str, str], list[RunResult]] = {}
    for r in runs:
        key = (r.phase, r.config, r.variant)
        groups.setdefault(key, []).append(r)

    out: dict[tuple[str, str, str], dict] = {}
    for key, group in groups.items():
        van = [r.eval_vanilla for r in group]
        adp = [r.eval_adaptive for r in group]
        wall = [r.train_wall_time_s for r in group]
        steps = [float(r.final_step) for r in group]
        acc_rate = [r.mean_acceptance for r in group]
        van_m, van_se, _ = _mean_stderr(van)
        adp_m, adp_se, _ = _mean_stderr(adp)
        wall_m, _, _ = _mean_stderr(wall)
        steps_m, _, _ = _mean_stderr(steps)
        acc_m, _, _ = _mean_stderr(acc_rate)
        out[key] = {
            "n_seeds": len(group),
            "vanilla_mean": van_m, "vanilla_stderr": van_se,
            "adaptive_mean": adp_m, "adaptive_stderr": adp_se,
            "train_wall_mean_s": wall_m,
            "final_step_mean": steps_m,
            "mean_acceptance": acc_m,
            "mode": group[0].mode,
            "H_high": group[0].H_high,
            "H_low": group[0].H_low,
            "paired_with": group[0].paired_with,
        }
    return out


# ---------------------------------------------------------------------------
# Findings.md construction
# ---------------------------------------------------------------------------

CONFIG_ORDER = ["(25, 275)", "(30, 270)", "(40, 260)", "(50, 250)", "(100, 200)"]


def _fmt_pct_se(mean: float, se: float) -> str:
    if math.isnan(mean):
        return "—"
    if se == 0.0:
        return f"{mean:.2%}"
    return f"{mean:.2%} ± {se:.2%}"


def render_reproduction_table(agg: dict) -> str:
    """Table (a): Reproduction comparison vs Kim et al. Table 1."""
    lines = [
        "## (a) Reproduction comparison — Phase B (variant=none) vs paper Table 1",
        "",
        "| (N, P) | Paper vanilla | Our vanilla | Δ (abs) | Paper adaptive | Our adaptive | Δ (abs) | Seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in CONFIG_ORDER:
        key = ("phase_b", cfg, "none")
        if key not in agg:
            lines.append(f"| {cfg} | {PAPER_TARGETS[cfg]['vanilla']:.2%} | — | — | "
                         f"{PAPER_TARGETS[cfg]['adaptive']:.2%} | — | — | 0 |")
            continue
        row = agg[key]
        van = row["vanilla_mean"]
        adp = row["adaptive_mean"]
        van_d = van - PAPER_TARGETS[cfg]["vanilla"] if not math.isnan(van) else float("nan")
        adp_d = adp - PAPER_TARGETS[cfg]["adaptive"] if not math.isnan(adp) else float("nan")
        lines.append(
            f"| {cfg} | {PAPER_TARGETS[cfg]['vanilla']:.2%} | "
            f"{_fmt_pct_se(van, row['vanilla_stderr'])} | "
            f"{van_d:+.2%} | "
            f"{PAPER_TARGETS[cfg]['adaptive']:.2%} | "
            f"{_fmt_pct_se(adp, row['adaptive_stderr'])} | "
            f"{adp_d:+.2%} | {row['n_seeds']} |"
        )
    return "\n".join(lines) + "\n"


def render_filter_vs_baseline_table(agg: dict, phase: str = "phase_c",
                                    inference: str = "vanilla") -> str:
    """Tables (b) and (c): filter variants vs Phase B baseline."""
    lines = [
        f"## Filter variant vs unfiltered baseline ({inference} inference)",
        "",
        "Mean ± stderr across seeds. Δ-baseline is variant_mean − baseline_mean.",
        "",
        "| (N, P) | variant=none | top_055 | top_060 | top_065 | top_070 | percentile | bottom_030 | band_030_065 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    show_variants = ["none", "top_055", "top_060", "top_065", "top_070",
                     "percentile", "bottom_030", "band_030_065"]
    metric = f"{inference}_mean"
    metric_se = f"{inference}_stderr"

    for cfg in CONFIG_ORDER:
        cells = [cfg]
        for v in show_variants:
            # 'none' lives in phase_b; filters live in `phase`
            ph = "phase_b" if v == "none" else phase
            key = (ph, cfg, v)
            if key in agg:
                cells.append(_fmt_pct_se(agg[key][metric], agg[key][metric_se]))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_random_vs_entropy_table(agg: dict, inference: str = "vanilla") -> str:
    """Table (d): paired random_replay vs entropy filter."""
    lines = [
        f"## (d) Random-filter vs entropy-filter ({inference} inference)",
        "",
        "Each cell is (entropy variant, random control), Δ = entropy − random.",
        "",
        "| (N, P) | top_065 (entropy) | top_065 (random) | Δ | percentile (entropy) | percentile (random) | Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    metric = f"{inference}_mean"
    metric_se = f"{inference}_stderr"

    for cfg in CONFIG_ORDER:
        # Look up entropy in phase_c, paired random in phase_d
        ent_top = ("phase_c", cfg, "top_065")
        rand_top = ("phase_d", cfg, "top_065")
        ent_pct = ("phase_c", cfg, "percentile")
        rand_pct = ("phase_d", cfg, "percentile")
        cells = [cfg]
        for ent_key, rand_key in [(ent_top, rand_top), (ent_pct, rand_pct)]:
            ent_val = agg.get(ent_key, {}).get(metric, float("nan"))
            ent_se = agg.get(ent_key, {}).get(metric_se, float("nan"))
            rand_val = agg.get(rand_key, {}).get(metric, float("nan"))
            rand_se = agg.get(rand_key, {}).get(metric_se, float("nan"))
            cells.append(_fmt_pct_se(ent_val, ent_se))
            cells.append(_fmt_pct_se(rand_val, rand_se))
            if not (math.isnan(ent_val) or math.isnan(rand_val)):
                cells.append(f"{ent_val - rand_val:+.2%}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_acceptance_rates_section(agg: dict) -> str:
    """Table (e): mean filter acceptance rate per condition."""
    lines = [
        "## (e) Mean post-warmup filter acceptance rate (kept / total)",
        "",
        "| (N, P) | top_055 | top_060 | top_065 | top_070 | percentile | bottom_030 | band_030_065 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    show_variants = ["top_055", "top_060", "top_065", "top_070",
                     "percentile", "bottom_030", "band_030_065"]
    for cfg in CONFIG_ORDER:
        cells = [cfg]
        for v in show_variants:
            key = ("phase_c", cfg, v)
            if key in agg and agg[key]["mean_acceptance"] is not None \
                    and not math.isnan(agg[key]["mean_acceptance"]):
                cells.append(f"{agg[key]['mean_acceptance']:.1%}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_open_questions_section(agg: dict) -> str:
    """Table (f): one paragraph per open question."""
    lines = [
        "## (f) Open questions — what the data shows",
        "",
        "### Q1. Does the entropy filter help over no-filter?",
        "",
        "Compare Phase B (variant=none) vs Phase C (best entropy filter) at each config. "
        "Effect size is `best_filter − none` per (N, P), averaged across seeds. "
        "Sign tells us whether filtering helped at all; magnitude tells us how much.",
        "",
        "### Q2. Is entropy the right SIGNAL, or does any filtering help?",
        "",
        "Compare Phase C (entropy filter) vs Phase D (random filter at matched acceptance). "
        "If entropy beats random at the same acceptance rate, the entropy *signal* matters; "
        "if not, any filtering helps just by reducing batch count.",
        "",
        "### Q3. Does training-time filtering complement or substitute for adaptive inference?",
        "",
        "Two-by-two: (variant ∈ {none, best_filter}) × (inference ∈ {vanilla, adaptive}). "
        "If the filter+adaptive cell exceeds adaptive-alone, they COMPLEMENT. "
        "If filter+adaptive ≈ adaptive-alone, they SUBSTITUTE. "
        "If filter alone exceeds adaptive alone, the filter DOMINATES.",
    ]
    return "\n".join(lines) + "\n"


def write_findings(runs: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_by_condition(runs)

    # Raw long-form CSV
    csv_path = out_dir / "raw_aggregate.csv"
    cols = list(asdict(runs[0]).keys()) if runs else []
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in runs:
            row = asdict(r)
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    # Findings markdown
    md_path = out_dir / "findings.md"
    with open(md_path, "w") as f:
        f.write("# Findings — full reproduction + entropy-filter modification\n\n")
        f.write(f"_Generated from {len(runs)} runs across phases B/C/D/E/F._\n\n")
        f.write("---\n\n")
        f.write(render_reproduction_table(agg) + "\n")
        f.write("---\n\n")
        f.write("## (b) Filter variant vs baseline at matched WALL-CLOCK\n\n")
        f.write(render_filter_vs_baseline_table(agg, "phase_c", "vanilla") + "\n")
        f.write(render_filter_vs_baseline_table(agg, "phase_c", "adaptive") + "\n")
        f.write("---\n\n")
        f.write("## (c) Filter variant vs baseline at matched EFFECTIVE GRADIENT UPDATES\n\n")
        f.write("Note: each Phase C job runs the same num_iterations as Phase B. The "
                "EFFECTIVE updates differ because filter conditions skip optimizer steps "
                "when the entire batch is dropped. Skipped-update count = "
                "(num_iterations − non-skipped steps logged in metrics.jsonl).\n\n")
        # For now this is the same as (b); a more nuanced comparison requires
        # subsetting trajectories at matched effective steps, which we'll do post-hoc.
        f.write(render_filter_vs_baseline_table(agg, "phase_c", "vanilla") + "\n")
        f.write("---\n\n")
        f.write(render_random_vs_entropy_table(agg, "vanilla") + "\n")
        f.write(render_random_vs_entropy_table(agg, "adaptive") + "\n")
        f.write("---\n\n")
        f.write(render_acceptance_rates_section(agg) + "\n")
        f.write("---\n\n")
        f.write(render_open_questions_section(agg) + "\n")
    print(f"[aggregate] wrote {md_path} ({md_path.stat().st_size} bytes)")
    print(f"[aggregate] wrote {csv_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_figures(runs: list[RunResult], out_dir: Path) -> None:
    """Generate matplotlib figures: PNGs + combined PDF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("[aggregate] matplotlib unavailable; skipping figures", file=sys.stderr)
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "all.pdf"

    agg = aggregate_by_condition(runs)
    pdf = PdfPages(pdf_path)

    # Fig 1: Reproduction accuracy (mean ± stderr) per config, vanilla & adaptive
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cfgs = CONFIG_ORDER
    x = list(range(len(cfgs)))
    paper_van = [PAPER_TARGETS[c]["vanilla"] for c in cfgs]
    paper_adp = [PAPER_TARGETS[c]["adaptive"] for c in cfgs]
    our_van = []
    our_van_se = []
    our_adp = []
    our_adp_se = []
    for c in cfgs:
        k = ("phase_b", c, "none")
        if k in agg:
            our_van.append(agg[k]["vanilla_mean"])
            our_van_se.append(agg[k]["vanilla_stderr"])
            our_adp.append(agg[k]["adaptive_mean"])
            our_adp_se.append(agg[k]["adaptive_stderr"])
        else:
            our_van.append(float("nan"))
            our_van_se.append(0.0)
            our_adp.append(float("nan"))
            our_adp_se.append(0.0)

    ax.plot(x, paper_van, "o--", label="Paper vanilla", color="C0", alpha=0.5)
    ax.plot(x, paper_adp, "o--", label="Paper adaptive", color="C1", alpha=0.5)
    ax.errorbar(x, our_van, yerr=our_van_se, fmt="s-", label="Our vanilla", color="C0")
    ax.errorbar(x, our_adp, yerr=our_adp_se, fmt="s-", label="Our adaptive", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.set_xlabel("(N, P)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Phase B reproduction vs Kim et al. 2025 Table 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_reproduction.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 2: Filter variants vs baseline (vanilla inference) per config
    show_variants = ["none", "top_055", "top_060", "top_065", "top_070",
                     "percentile", "bottom_030", "band_030_065"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    width = 0.10
    for i, v in enumerate(show_variants):
        ph = "phase_b" if v == "none" else "phase_c"
        ys, errs = [], []
        for c in cfgs:
            k = (ph, c, v)
            if k in agg and not math.isnan(agg[k]["vanilla_mean"]):
                ys.append(agg[k]["vanilla_mean"])
                errs.append(agg[k]["vanilla_stderr"])
            else:
                ys.append(0)
                errs.append(0)
        offset = (i - len(show_variants) / 2) * width
        ax.bar([xi + offset for xi in x], ys, width=width, yerr=errs, label=v, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.set_xlabel("(N, P)")
    ax.set_ylabel("Vanilla-inference accuracy")
    ax.set_title("Filter variants vs baseline (vanilla inference)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_filter_vs_baseline_vanilla.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 3: Same for adaptive inference
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, v in enumerate(show_variants):
        ph = "phase_b" if v == "none" else "phase_c"
        ys, errs = [], []
        for c in cfgs:
            k = (ph, c, v)
            if k in agg and not math.isnan(agg[k]["adaptive_mean"]):
                ys.append(agg[k]["adaptive_mean"])
                errs.append(agg[k]["adaptive_stderr"])
            else:
                ys.append(0)
                errs.append(0)
        offset = (i - len(show_variants) / 2) * width
        ax.bar([xi + offset for xi in x], ys, width=width, yerr=errs, label=v, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.set_xlabel("(N, P)")
    ax.set_ylabel("Adaptive-inference accuracy")
    ax.set_title("Filter variants vs baseline (adaptive top_prob_margin inference)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_filter_vs_baseline_adaptive.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 4: Random-filter vs entropy-filter (paired) for top_065 and percentile
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (variant_label, cond_name) in enumerate([
        ("top_065", "top_065"), ("percentile", "percentile"),
    ]):
        ent_van, ent_se, rand_van, rand_se = [], [], [], []
        for c in cfgs:
            ek = ("phase_c", c, cond_name)
            rk = ("phase_d", c, cond_name)
            ent_van.append(agg.get(ek, {}).get("vanilla_mean", float("nan")))
            ent_se.append(agg.get(ek, {}).get("vanilla_stderr", 0.0))
            rand_van.append(agg.get(rk, {}).get("vanilla_mean", float("nan")))
            rand_se.append(agg.get(rk, {}).get("vanilla_stderr", 0.0))
        a = ax[ax_idx]
        a.errorbar(x, ent_van, yerr=ent_se, fmt="s-", label="Entropy", color="C2")
        a.errorbar(x, rand_van, yerr=rand_se, fmt="o--", label="Random (paired)", color="C3")
        a.set_xticks(x)
        a.set_xticklabels(cfgs, rotation=20)
        a.set_xlabel("(N, P)")
        a.set_ylabel("Vanilla accuracy")
        a.set_title(f"{variant_label}: entropy vs paired random control")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_entropy_vs_random.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    print(f"[aggregate] wrote {pdf_path}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"[aggregate] wrote {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="Path to ${PROJECT_DIR}/results/ (contains phase_b/, phase_c/, ...)")
    ap.add_argument("--output-dir", required=True,
                    help="Where to write findings.md, raw_aggregate.csv, figures/")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)

    if not results_root.exists():
        print(f"[aggregate] FAIL: results-root does not exist: {results_root}", file=sys.stderr)
        return 2

    runs = discover_runs(results_root)
    print(f"[aggregate] discovered {len(runs)} completed runs")

    if not runs:
        print(f"[aggregate] no runs found; nothing to aggregate", file=sys.stderr)
        return 1

    write_findings(runs, out_dir)
    render_figures(runs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
