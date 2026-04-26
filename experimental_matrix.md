# Production experimental matrix — authorized 2026-04-26 (rev. 2026-04-26)

This document is the **master plan** for the production array. Generated from the post-design-review authorization. Job counts and resource estimates are deterministic given the inputs below; assumptions are listed at the end so any change can be re-costed.

> **Status.** Draft. Not yet authorized for sbatch. Per-config iteration budgets are placeholders until the calibration phase completes — see Phase 1 below. The size ablation and 5-seed bump are deferred to Phase 3, run only after the MV-core phase identifies an empirically-best `H_high`.

---

## 1. Phase structure (three production phases, three authorization gates)

| Phase | Slices | Jobs | Gate to next phase |
|---|---|---|---|
| **Phase 1 — Calibration** | Calibration | **5** | User reviews per-config plateau steps, professor approves iteration budgets |
| **Phase 2 — MV core + H_low sanity** | MV core (135) + H_low sanity (3) | **138** | User + professor review threshold-sweep results, identify empirically-best `H_high` |
| **Phase 3 — Scaling at best H_high** | Size ablation (24) + 5-seed bump (30) | **54** | User authorizes camera-ready re-eval at 10k samples |
| **Total production** |  | **192** |  |
| **Grand total incl. calibration** |  | **197** |  |

**Why three phases:** the size ablation and 5-seed bump both depend on knowing the empirically-best `H_high`. Locking that to 0.65 a priori would mean the scaling claim and the statistical-power claim are both made at a possibly-suboptimal threshold — a reviewer would correctly flag this as uncontrolled. Splitting into Phase 2 / Phase 3 adds one review gate but produces the right scientific answer.

---

## 2. Slice summary

| Slice | Phase | Conditions | Configs | Seeds | Jobs | Purpose |
|---|---|---|---|---|---|---|
| **Calibration** | 1 | 1 (none) | 5 | 1 | **5** | Per-config iteration budgets via plateau detection |
| **MV core** | 2 | 9 | 5 | 3 | **135** | Headline ablation + Q12 regime taxonomy + entropy-vs-random control |
| **H_low sanity** | 2 | 1 | 1 | 3 | **3** | Validate H_low dismissal at hard config (100, 200) |
| **Size ablation** | 3 | 3 | 1 | 2 | **24** | Scaling claim across {6M, 14M, 19M, 38M} on (25, 275) — uses Phase-2's best `H_high` |
| **5-seed bump** | 3 | 5 | 3 | 2 (extra) | **30** | Statistical power on headline conditions — uses Phase-2's best `H_high` |
| **Production array total** |  |  |  |  | **192** |  |
| **Grand total** |  |  |  |  | **197** |  |

---

## 2. MV core — 135 jobs

**Conditions per (config, seed)** (9 total):

| # | Variant | Mode | Threshold | Random control? |
|---|---|---|---|---|
| 1 | `none` | none | n/a | n/a (this IS the control) |
| 2 | `top_055` | top | H_high = 0.55 | yes (paired) |
| 3 | `top_065` | top | H_high = 0.65 | yes (paired) |
| 4 | `top_070` | top | H_high = 0.70 | yes (paired) |
| 5 | `percentile` | percentile_band | pct_low=0.25, pct_high=0.75 | yes (paired) |
| 6 | `random_top_055` | RANDOM (replays cond 2's per-step keep counts) | n/a | — |
| 7 | `random_top_065` | RANDOM (replays cond 3's per-step keep counts) | n/a | — |
| 8 | `random_top_070` | RANDOM (replays cond 4's per-step keep counts) | n/a | — |
| 9 | `random_percentile` | RANDOM (replays cond 5's per-step keep counts) | n/a | — |

**Configs:** (25, 275), (30, 270), (40, 260), (50, 250), (100, 200) — all at 14M params.
**Seeds:** {0, 1, 2}.

**Q12 regime coverage from this slice:**
- `top_055` lands inside the converged H cluster (~0.57) → **continuous-mode** test.
- `top_065` lands between clusters (~0.62 midpoint) → **gate-mode** test.
- `top_070` lands above the early-training transient (~0.67) → **no-op** test (should ≈ variant=none).

**Random-control execution order:** entropy condition runs first; its `metrics.jsonl` is read by the paired random condition. See `paired-replay implementation` memory note.

---

## 3. H_low sanity check — 3 jobs

**Single condition** added on the hardest config:

| Variant | Mode | Threshold | Config | Seeds | Jobs |
|---|---|---|---|---|---|
| `band_hlow040` | band | H_low=0.40, H_high=0.65 | (100, 200) only | {0, 1, 2} | 3 |

**Decision rule for what this measures:**
- If `band_hlow040` ≈ `top_065` on (100, 200): H_low has nothing to act on at hard configs — confirms smoke's dismissal generalizes. Future H_low sweeps unnecessary.
- If `band_hlow040` ≠ `top_065`: there IS mass below 0.40 at hard configs. Triggers a follow-up H_low sweep (out of scope for v1).

No paired random control for this slice — the comparison axis is `band_hlow040` vs. `top_065` (both already in the matrix), not entropy vs. random.

---

## 4. Model-size ablation — 24 jobs (Phase 3)

**Sizes:** {6M, 14M, 19M, 38M} — 4 points spanning a ~6× range. All architectures share `head_dim=64` and `ff/hidden=4` (the same scaling family as the project's 14M default). Depth scales with width; this is the simplest defensible scaling rule and avoids unprincipled per-size tuning.

| Label | hidden | n_layers | n_heads | ff | param count | within ±10%? |
|---|---|---|---|---|---|---|
| **6M**  | 256 | 8  | 4 | 1024 | ~6.3M  | ✓ (5.4–6.6M band) |
| **14M** | 384 | 8  | 6 | 1536 | ~14.2M | ✓ (project default) |
| **19M** | 384 | 11 | 6 | 1536 | ~19.5M | ✓ (17.1–20.9M band) — paper's stated size |
| **38M** | 512 | 12 | 8 | 2048 | ~37.7M | ✓ (34.2–41.8M band) |

**Sizing methodology** (documented for reviewers):
- Param count formula per layer (vocab_size and bias terms negligible): `4·hidden² + 2·hidden·ff`. Add `vocab_size·hidden` for the output head.
- 14M is fixed as the project default (matches the smoke run and the reproduction baseline).
- The other three are sized **width-then-depth** to land within ±10% of the target param count while keeping `head_dim=64` and `ff/hidden=4` constant. The user explicitly authorized this approach over hand-tuning.
- 19M is chosen to match the paper's stated 19M MDM, enabling direct comparison.
- 6M and 38M bracket 14M by ~0.4× and ~2.7×, providing two scaling-curve points on either side of the default.

**Conditions** (3 of the MV core, no random controls):
1. `none`
2. `top_<best>` — uses the empirically-best `H_high` from Phase 2's MV core results, NOT a pre-locked 0.65
3. `percentile`

**Config:** (25, 275) only.
**Seeds:** {0, 1} (2 seeds — exploratory scaling; statistical-power bump is in the next slice).

Total: 4 sizes × 3 conditions × 2 seeds = **24 jobs**.

**Reuse from MV core:** the 14M × (25, 275) cells for `none` and `percentile` are already in MV at 3 seeds. The size ablation re-runs them at 2 seeds for self-consistency (deterministic; output should match MV bit-exactly given the Task-3 determinism flags). If they don't match, that's a determinism bug to fix. **In job counting we count all 24 — but at array runtime, the 4 redundant 14M `none`+`percentile` jobs can be dropped if validated.** The `top_<best>` cell at 14M is genuinely new (the matching MV cell may use a different `H_high`).

**What this slice does NOT include** (intentional):
- No random controls. Entropy-vs-random is established at 14M in MV; we're not re-litigating it across sizes for v1.
- No (N, P) sweep at non-14M sizes. Scaling claim is "the filter effect grows/shrinks as we vary model size on the same data distribution."

---

## 5. 5-seed bump — 30 jobs (Phase 3)

**Conditions** (5 — the headline conditions only):
1. `none`
2. `top_<best>` — uses the empirically-best `H_high` from Phase 2's MV core results
3. `percentile`
4. `random_top_<best>` (paired with condition 2)
5. `random_percentile` (paired with condition 3)

**Strategic configs** (3 — span the difficulty axis):
- (25, 275) — easiest
- (50, 250) — middle
- (100, 200) — hardest

**Extra seeds:** {3, 4} (2 additional seeds beyond MV's {0, 1, 2}, totaling 5 seeds at these condition×config cells).

Total: 5 conditions × 3 configs × 2 extra seeds = **30 jobs**.

**Why these 3 configs:** detecting a 1% effect with σ ≈ 0.6% requires ~5 seeds. The other 2 configs ((30, 270) and (40, 260)) stay at 3 seeds — adequate for showing the trend, not for tight CIs.

---

## 6. Calibration (Phase 1) — 5 jobs

**Purpose:** determine per-config `num_iterations` to avoid wasting 90% of production compute on post-convergence noise (Q10 in methodology_notes).

| # | Config | Variant | Seed | Stop criterion |
|---|---|---|---|---|
| C1 | (25, 275) | none | 0 | rolling-mean plateau OR 50K cap |
| C2 | (30, 270) | none | 0 | same |
| C3 | (40, 260) | none | 0 | same |
| C4 | (50, 250) | none | 0 | same |
| C5 | (100, 200) | none | 0 | same |

**Plateau criterion (authorized 2026-04-26):**

```
stable[step] := |mean(loss[step-W+1 .. step]) - mean(loss[step-2W+1 .. step-W])|
                / max(mean(prev), 1e-10) < tolerance

with W=500, tolerance=0.005 (0.5% relative), check_every=500, min_step=1000.
```

Skipped-optim steps (where the filter dropped the entire batch — N/A for `variant=none` calibration runs but matters for the production array if reused) are excluded from the rolling means. The check fires every 500 training steps starting at step 1000; on first trigger, training exits cleanly and writes `plateau_step.txt`.

**Implementation details:**
- Plateau detection runs **DURING** training, not post-hoc — implemented in `entropy_filtered.src.train_filtered._is_plateau` and exercised by 10 unit tests in `entropy_filtered/tests/test_early_stop.py`. This means a job that converges at step 8K exits at step 8K, not after burning 5h of walltime.
- `plateau_step.txt` format contract: **single integer, no whitespace** (just `str(step)` written to disk). Either the detected plateau step (early-stop case) or `num_iterations` (cap case). Downstream parsers do `int(open(f).read())` — no `.strip()` needed.
- `plateau_step.txt` lands in **`${PROJECT_DIR}/results/calibration/<run>/`** (the BACKED-UP project area), not just scratch. The Slurm script asserts the file exists and parses correctly before exiting; failure surfaces as a non-zero exit code.

**Output (per task) under `${PROJECT_DIR}/results/calibration/<run>/`:**
- `plateau_step.txt` — the integer
- `metrics.jsonl`, `config.yaml`, `ckpt_step*.pt`
- `warnings.log` — all UserWarnings during training (always written, may be empty)
- `deterministic_warnings.log` — filtered to torch determinism warnings only (audit trail for eventually flipping `warn_only=True` → `False`)
- `jobstats.txt` — Slurm resource accounting

**Cost estimate:** plateau is likely much earlier than 50K (smoke showed loss plateau on (25, 275) by step ~110, though that's one seed and may not reflect a 1000-step rolling-mean criterion). Conservatively assume average plateau at ~25K → 25,000 / 3.92 = 6,378 s ≈ 1.77 h training per job. Total: 5 × ~2 h = **~10 GPU-hours**; user's 30 GPU-hour estimate has 3× buffer.

**Authorization:** `submit calibration` — this is the next thing the user will explicitly authorize.

---

## 7. Resource estimates

### Per-job compute (14M model, batch=128, H200)

| Phase | Cost |
|---|---|
| Training | 25,000 steps / 3.92 step/s ≈ **1.77 h** (placeholder; actual = calibration result) |
| Eval — 4 wall-clock checkpoints × (1 vanilla + 1 adaptive) × 2,000 samples | **~1.13 h** |
| Per-job total compute | **~2.90 h** |
| Per-job allocated walltime (1.5× buffer) | **~4.35 h** → round to `--time=05:00:00` |

### GPU-hours by slice (assuming 25K-step plateau)

| Slice | Jobs | Avg per-job | GPU-hours |
|---|---|---|---|
| MV core | 135 | 2.90 h | 391.5 |
| H_low sanity | 3 | 2.90 h | 8.7 |
| Size ablation | 24 | varies (~4.0 h avg) | 96.0 |
| 5-seed bump | 30 | 2.90 h | 87.0 |
| **Production array total** | **192** |  | **~583 GPU-hours** |
| Calibration | 5 | ~2.80 h | 14.0 |
| **GRAND TOTAL** | **197** |  | **~597 GPU-hours** |

### Wall-clock at concurrency

| Concurrency | Wall-clock for production array |
|---|---|
| 24 (current sacctmgr cap) | 583 / 24 = **24.3 h** ≈ 1 day |
| 72 (full gpu_h200 if PI authorizes) | 583 / 72 = **8.1 h** |

### Sensitivity to calibration outcome

| If plateau at... | MV-core per-job train | Total prod GPU-hours | Wall-clock @ 24 |
|---|---|---|---|
| 5K steps  | 0.35 h  | ~480 | ~20 h |
| 10K steps | 0.71 h  | ~530 | ~22 h |
| 25K steps | 1.77 h  | ~583 | ~24 h |
| 50K steps (no convergence) | 3.55 h | ~925 | ~38 h |

The matrix is feasible across the entire calibration-outcome envelope. **No replanning needed regardless of where the plateau lands.**

---

## 8. Resolved spec items (post-authorization)

All five open items from the previous revision have been resolved:

1. **`best_H_high` for size ablation and 5-seed bump.** **Resolved: revise post-MV.** Phase 3 uses the empirically-best `H_high` from Phase 2's threshold sweep, not a pre-locked 0.65. This is why Phase 3 exists as a separate stage. (See §1, §4, §5.)

2. **Model-size architecture table.** **Resolved: width-then-depth scaling at fixed `head_dim=64` and `ff/hidden=4`.** All four sizes within ±10% of target. (See §4 table.)

3. **Plateau criterion.** **Resolved: rolling-mean relative.** `|mean(last W) - mean(prev W)| / mean(prev) < 0.005` with `W=500`, checked every 500 steps from step 1000. Implemented in `train_filtered._is_plateau` with 10 unit tests. Plateau detection runs DURING training, not post-hoc. (See §6.)

4. **Per-step vs. per-10 logging for MV first run.** **Resolved: keep `log_every=100` (default) for all production runs.** The Q11 filter-dynamics observation (sharp regime crossing between logged steps 100 and 110) is interesting but the threshold sweep itself is the experimental answer to whether the regime shift is filter-driven or natural. If after Phase 2 the user wants per-step granularity to characterize a specific crossing, we can re-run a single MV condition with `log_every=1` as a one-off — but locking `log_every=1` for all 138 Phase-2 jobs would 50×-bloat `metrics.jsonl` (~250MB per run × 138 = ~34GB) for marginal scientific value.

5. **Eval test-set construction.** **Resolved: per-config test sets, shared across (variant, seed) cells.** Each (N, P) config generates its own 2,000-sample test set deterministically from `data.seed=42` (already in all configs). All 30 cells of (variant × seed) for a given config share the same test set — so within-config comparisons are paired and noise-free. Across configs the test sets necessarily differ (different N, P → different distribution). For the camera-ready re-eval at 10,000 samples, the larger test set is a superset of the 2,000-sample one (same seed, larger draw count), preserving comparability with screening numbers.

> **One new open item identified during this revision:** when `top_<best>` is determined post-Phase-2, an authorization gate is needed to confirm the chosen value before Phase 3 runs. Procedure: agent reports the per-condition test accuracy with confidence intervals, user picks the best `H_high` (or asks for a tie-breaker analysis), then types `submit phase3 H_high=<value>`.

---

## 9. Machine-readable matrix (YAML)

```yaml
# production_matrix.yaml — emit-ready spec for the array

defaults:
  model_size: "14M"
  model: { hidden: 384, n_layers: 8, n_heads: 6, ff: 1536 }
  batch_size: 128
  lr: 1.0e-3
  pad_to: 512
  filter_warmup_steps: 500
  eval_samples_screening: 2000
  eval_samples_final: 10000
  eval_checkpoints_wallclock_h: [1, 2, 4, 8]
  eval_inference_modes: ["vanilla", "adaptive_top_prob_margin"]

slices:
  mv_core:
    phase: 2
    conditions:
      - { name: "none",                mode: "none" }
      - { name: "top_055",             mode: "top",              H_high: 0.55 }
      - { name: "top_065",             mode: "top",              H_high: 0.65 }
      - { name: "top_070",             mode: "top",              H_high: 0.70 }
      - { name: "percentile",          mode: "percentile_band",  pct_low: 0.25, pct_high: 0.75 }
      - { name: "random_top_055",      mode: "random_replay",    paired_with: "top_055" }
      - { name: "random_top_065",      mode: "random_replay",    paired_with: "top_065" }
      - { name: "random_top_070",      mode: "random_replay",    paired_with: "top_070" }
      - { name: "random_percentile",   mode: "random_replay",    paired_with: "percentile" }
    configs: [[25, 275], [30, 270], [40, 260], [50, 250], [100, 200]]
    seeds: [0, 1, 2]
    jobs: 135

  hlow_sanity:
    phase: 2
    conditions:
      - { name: "band_hlow040",        mode: "band",             H_low: 0.40, H_high: 0.65 }
    configs: [[100, 200]]
    seeds: [0, 1, 2]
    jobs: 3

  size_ablation:
    phase: 3
    h_high_source: "phase2_winner"   # NOT a pre-locked literal — resolved post-MV
    conditions:
      - { name: "none",                mode: "none" }
      - { name: "top_best",            mode: "top",              H_high: "<phase2_best>" }
      - { name: "percentile",          mode: "percentile_band",  pct_low: 0.25, pct_high: 0.75 }
    sizes:
      - { label: "6M",  param_count: 6.3,  model: { hidden: 256, n_layers: 8,  n_heads: 4, ff: 1024 } }
      - { label: "14M", param_count: 14.2, model: { hidden: 384, n_layers: 8,  n_heads: 6, ff: 1536 } }
      - { label: "19M", param_count: 19.5, model: { hidden: 384, n_layers: 11, n_heads: 6, ff: 1536 } }
      - { label: "38M", param_count: 37.7, model: { hidden: 512, n_layers: 12, n_heads: 8, ff: 2048 } }
    configs: [[25, 275]]
    seeds: [0, 1]
    jobs: 24

  five_seed_bump:
    phase: 3
    h_high_source: "phase2_winner"
    conditions:
      - { name: "none",                mode: "none" }
      - { name: "top_best",            mode: "top",              H_high: "<phase2_best>" }
      - { name: "percentile",          mode: "percentile_band",  pct_low: 0.25, pct_high: 0.75 }
      - { name: "random_top_best",     mode: "random_replay",    paired_with: "top_best" }
      - { name: "random_percentile",   mode: "random_replay",    paired_with: "percentile" }
    configs: [[25, 275], [50, 250], [100, 200]]
    seeds: [3, 4]   # extras on top of MV's {0, 1, 2}
    jobs: 30

  calibration:
    phase: 1
    purpose: "Determine per-config num_iterations from convergence criterion"
    conditions:
      - { name: "none", mode: "none" }
    configs: [[25, 275], [30, 270], [40, 260], [50, 250], [100, 200]]
    seeds: [0]
    stop_criterion:
      type: "rolling_mean_relative"
      window_steps: 500
      tolerance: 0.005
      check_every: 500
      min_step: 1000
      cap_steps: 50000
    jobs: 5

totals:
  production_array_jobs: 192
  calibration_jobs: 5
  grand_total_jobs: 197
  estimated_gpu_hours_at_25k_plateau: 597
  wallclock_at_concurrency_24_h: 25
  wallclock_at_concurrency_72_h: 8.5
```

---

## 10. Assumptions (any change re-costs the matrix)

1. **Plateau at ~25K steps** for cost estimation. Actual will come from calibration; matrix is feasible across [5K, 50K] envelope.
2. **3.92 steps/s on H200** for 14M model (smoke-validated; see `pad_to=512` invariant memory).
3. **2,000 screening eval samples** and adaptive-inference cost ~17 min per checkpoint per config.
4. **Determinism flags from Task 3** are in place — required for size-ablation deduplication and resume bit-exactness.
5. **Random-replay implementation** uses paired-replay (option i) — see `random_filter_paired_replay` memory.
6. **Single conservative walltime** of `--time=05:00:00` for 14M jobs, `--time=08:00:00` for 38M. Other sizes interpolate.
7. **gpu_h200 has unlimited MaxTime**; no partition-cap concern.
