# `entropy_filtered/` — entropy-filtered MDM training (the modification)

Five-variant ablation of the project's research extension to Kim et al. 2025: filter MDM training masks by the model's own prediction entropy at masked positions, dropping subproblems that are too easy (wasted gradient) or too hard (likely intractable per Conjecture B.13).

## Status (Phase 5)

| Component | Status |
|---|---|
| Filter module (`src/filter.py`) — 5 modes, warmup, mean/median, abs+pct thresholds | done |
| Filtered training loop (`src/train_filtered.py`) | done |
| Five ablation configs at production scale | done |
| Smoke config (CPU, < 10 min) | done |
| 15 unit tests for the filter, all numerical | passing |
| Smoke test (CPU, end-to-end, filter fires after warmup) | passing in 2.6 s |

## What this package does

Imports from `baseline/src/*` (data, model, diffusion, inference, evaluation, utils) and only changes the **training-step inner loop**. The change:

1. Mask creation is identical to baseline: sample `n` ~ Uniform{1, ..., maskable_len}, mask `n` random positions per sample.
2. Run a `torch.no_grad()` forward pass on the masked batch.
3. Compute per-sample mean (or median) Shannon entropy at masked positions, excluding the mask token from the categorical.
4. Decide which samples to keep based on `mode`. During the first `warmup_steps` (default 500), all samples are kept regardless.
5. Run the loss-bearing forward pass on the kept samples, compute the standard MDM 1/n-weighted loss, step the optimizer.

The acceptance rate is logged in the JSONL metrics so we can audit how aggressively the filter is firing across training.

## The 5 variants

| Variant | Mode | What it tests |
|---|---|---|
| `none`             | unfiltered                                  | Control. Should reproduce baseline exactly. |
| `bottom`           | drop low-entropy (easy) masks               | "Remove wasted gradient on trivial subproblems." |
| `top`              | drop high-entropy (intractable) masks       | Direct test of Prop 3.3 — the paper's theoretical concern. |
| `band`             | drop both ends, ABSOLUTE thresholds         | Full proposal with fixed `H_low`, `H_high`. |
| `percentile_band`  | drop both ends, batch-RELATIVE percentiles  | Self-calibrating to the model's current state. |

User-confirmed defaults (saved 2026-04-26):
- `warmup_steps = 500`
- `pct_low = 0.25`, `pct_high = 0.75` (keep middle 50%)
- `reduction = "mean"`
- Both absolute and percentile flavors are first-class — chosen by the `mode` config flag.

## Layout

```
entropy_filtered/
├── README.md                       this file
├── configs/
│   ├── _base_25_275.yaml           shared base; runs as 'none' (control)
│   ├── lo_nae_sat_25_275_none.yaml
│   ├── lo_nae_sat_25_275_bottom.yaml
│   ├── lo_nae_sat_25_275_top.yaml
│   ├── lo_nae_sat_25_275_band.yaml
│   ├── lo_nae_sat_25_275_percentile.yaml
│   └── lo_nae_sat_smoke_band.yaml  CPU smoke (warmup=20, 100 steps)
├── src/
│   ├── filter.py                   the filter logic (5 modes + warmup)
│   └── train_filtered.py           filtered training loop (imports baseline/)
├── scripts/
│   ├── smoke_test.sh               gate: pytest (baseline + filter) + smoke
│   └── smoke_test.py
└── tests/
    ├── conftest.py
    └── test_filter.py              15 numerical assertions
```

## Quick start

```bash
cd <repo_root>
bash entropy_filtered/scripts/smoke_test.sh   # gate: 38 unit tests + end-to-end filtered smoke
```

To run any of the 5 production variants:

```bash
python -m entropy_filtered.src.train_filtered \
    --config entropy_filtered/configs/lo_nae_sat_25_275_band.yaml
```

## How wall-clock is matched (Phase 7 protocol)

Per `paper_notes.md` §13.2 / user spec, the experimental protocol matches **wall-clock**, not gradient-step count, between variants. The Phase 7 Slurm scripts will:

1. Set the same `--time=` walltime budget for all 5 variants and 5 (N, P) configs.
2. Each run terminates either at `num_iterations` *or* at the walltime — whichever comes first.
3. Final accuracy is reported at the matched-walltime checkpoint.

This ensures we are not biased by the filter's per-step cost (one extra no-grad forward) or by its acceptance rate (effective gradient steps per wall-clock are different across variants).

## Audit trail in metrics.jsonl

Every logged step records:
- `loss`, `grad_norm`, `mean_n`, `n_masked_total` — same as baseline.
- `filter_n_kept`, `filter_n_dropped` — per-batch acceptance counts.
- `filter_H_min`, `filter_H_max`, `filter_H_mean` — per-batch entropy stats.
- `skipped_optim_step` — 1 iff the filter dropped every sample in the batch (rare).

Phase 8 aggregation will use these to plot acceptance-rate vs step (the headline diagnostic for whether the filter is doing what we intended).

## What's deferred to Phase 7

- Per-(N, P) variants for the other four production configs (30, 270), (40, 260), (50, 250), (100, 200). Currently only the (25, 275) point has its 5 variants. Replicating across the other four is ~4 × 5 = 20 small YAML files; will be auto-generated when we are ready to schedule the Bouchet job array.
- Sudoku entropy-filter run. Phase 4's Sudoku data path is in place; the same `train_filtered.py` will accept a Sudoku data adapter once that's wired (and once the ARM-with-ordering baseline is vendored from upstream per `code_audit.md`).

## Hypotheses being tested (paper-friendly framing)

1. `band_filter` matches baseline accuracy at lower wall-clock → "training-time efficiency."
2. `band_filter` exceeds baseline at the same wall-clock → strongest result.
3. `top_filter` exceeds baseline but `bottom_filter` does not → confirms the paper's theoretical danger zone.
4. `bottom_filter` exceeds baseline but `top_filter` does not → counter to the theory; easy subproblems are the wasted compute.
5. Adaptive inference (Top-Probability-Margin) is more effective on entropy-filter-trained models than on baseline-trained models (interaction between training and inference fixes).

Negative results are publishable: even if no variant beats baseline, the methodology of entropy-as-learnability-proxy is itself a contribution and the filter design generalizes the paper's analytic Prop 3.3 to any data distribution.
