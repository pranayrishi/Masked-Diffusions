# Bug log

Issues discovered during the autonomous experimental run, with the fix and any
re-runs that were triggered.

---

## 2026-04-26 — Phase B undertraining: 10K iterations was a floor, not a cap

**Symptom.** Phase B array 9550134 task 0 (variant=none on (N=25, P=275)) reported
vanilla-inference accuracy of **61.66%** after a full 10K-iteration training run
that finished with loss 0.503. Paper Table 1 reports **78.06%** for the same
config and inference protocol. The 16-percentage-point gap is over the user's
5% suspicion threshold.

**Root cause.** The model was undertrained:

- Loss was still descending at step 10K (0.55 at ~step 5K → 0.50 at step 10K).
  Calibration's plateau detector with a 500-step window fired around step 1500
  on (25, 275) — but that was NOT a real plateau, it was the strict criterion
  satisfied early because of low per-step variance.
- Naive marginal-only baseline (predicting from P(NAE=1)=0.75 marginal,
  ignoring constraints) yields expected accuracy = 0.75² + 0.25² = **62.5%**.
  Our 61.66% is essentially at the marginal-prediction level — meaning the
  model has learned the obs marginal but **NOT** the constraint structure.
  The paper's 78.06% requires the model to actually use the NAE constraints
  to recover latents and propagate to obs values.
- The user's instruction was unambiguous: *"10,000 steps minimum or until loss
  plateaus for 2,000 consecutive steps under the rolling-mean criterion,
  whichever comes later. Do not under-train."* The 10K is the FLOOR. Phase B's
  configs did not enable the plateau detector, so jobs trained exactly 10K and
  stopped.

**Fix (commits below).**

1. `entropy_filtered/src/train_filtered.py`: the `--override <key>.<sub>=<val>`
   handler used to KeyError when the top-level key (e.g. `early_stop`) wasn't
   already in the YAML. Replaced `cfg_dict[head][tail] = ...` with
   `cfg_dict.setdefault(head, {})[tail] = ...` so missing top-level keys are
   autocreated. Required to pass the new early-stop overrides.

2. `slurm/04_phase_b_baseline.sh`, `05_phase_c_filter.sh`,
   `06_phase_d_random.sh`, `07_phase_e_seedbump.sh`: bumped
   `num_iterations` from 10000 → **50000** (matches paper's 5×10⁴ budget) and
   added `early_stop` overrides:

   ```
   early_stop.enabled=true
   early_stop.criterion=rolling_mean_relative
   early_stop.tolerance=0.005
   early_stop.window=2000        # the user's instruction
   early_stop.check_every=1000
   early_stop.min_step=10000     # the user's 10K floor
   ```

   So a job stops when the 2000-step rolling-mean is within 0.5% relative of the
   prior 2000 steps AND we've trained at least 10K steps. The 50K cap is the
   safety net for configs that never plateau.

3. `slurm/04_phase_b_baseline.sh`: walltime 03:00:00 → **06:00:00** to
   accommodate worst case (50K iterations × 1/3.4 step/s = 4h 5min training,
   plus eval).

4. Phase C / D / E: walltime 03:00:00 → **06:00:00**, partition switched from
   `gpu_h200` to **`scavenge_gpu`** (mixed GPU types, auto-requeue on
   preemption — our `test_resume.py` is bit-exact verified). Concurrency cap
   raised from %24 to %48 to match scavenge throughput.

5. `baseline/src/utils.py`: reverted `torch.use_deterministic_algorithms`
   `warn_only=False` → **`warn_only=True`**. The Phase 1 calibration audit
   was H200-only. scavenge_gpu has mixed GPU types where some cuBLAS ops are
   not guaranteed deterministic; warn_only=True surfaces them to warnings.log
   instead of aborting. Promote to False after a per-partition audit.

**Re-runs triggered.**

- Phase B array 9550134: tasks 4-24 cancelled before they started; tasks 0-3
  let to finish to provide a "10K baseline" data point for comparison.
- Phase C array 9550641: full cancellation (150 tasks, none had started).
- Resubmitting Phase B + C at 50K-iteration budget after this commit.

**Lesson.** "Plateau detection" with a too-narrow window can fire at apparent
plateaus that are actually still descending. The 500-step window in Phase 1
calibration was too strict; the 2000-step window the user specified is more
robust. Cross-check: the loss trajectory itself, not just the plateau detector,
should be inspected before committing to an iteration budget.

---

## 2026-04-26 — Adaptive inference missing Gumbel(0, 0.5) noise

**Symptom.** Phase B 50K results on (N=25, P=275): vanilla 60-64%, adaptive
(top_prob_margin) 63-66%. Paper Table 1 reports vanilla 78.06% / adaptive 93.76%.
The adaptive-vs-vanilla delta we measured (~5pp) is far below the paper's
~16pp delta. Possible cause: paper's adaptive inference adds Gumbel noise to
the oracle scores; our implementation defaulted to noise=none.

**Root cause.** paper_notes.md line 504 documents the Table 1 inference setup
as "50 reverse steps per sample, **Gumbel coeff 0.5**". Paper §3.4 specifies
Gumbel(0, 1) × 0.5 added to selection scores (NOT to logits used for sampling
the unmasked token). The `_add_score_noise` helper in `baseline/src/inference.py`
already implements this; we just had the noise parameters defaulting off.

**Fix.** Changed defaults in:
- `entropy_filtered/src/train_filtered.py` `FilteredTrainConfig`:
  `eval_noise="gumbel"`, `eval_noise_scale=0.5`.
- `baseline/src/run_eval_only.py` CLI defaults: same.

Vanilla inference is unaffected (`run_inference` ignores noise for
`strategy="vanilla"`).

**Re-runs triggered.**

- Phase B 9561941 tasks 7-24 (queued at the time of this commit): pick up the
  fix automatically when they start; they will eval with Gumbel noise.
- Phase C 9587034 (all 150 queued): same.
- Phase D, E, F: not yet submitted, will use the new defaults.

The 5 already-finished (25, 275) Phase B runs (9556517 tasks 0-4) plus tasks
5/6 of 9561941 already in eval do NOT have Gumbel-adaptive numbers. They need
a re-eval pass via `baseline/src/run_eval_only.py` once the rest of Phase B
completes; queueing that as a Phase F-like cleanup pass.

**Open question.** The vanilla-accuracy gap (60-64% measured vs 78.06% paper)
is independent of Gumbel noise (vanilla doesn't use noise). Cause TBD; possible
candidates from the user's prompt: data generator differences (we use
without-replacement triples; paper convention may differ), training-data size,
exact 19M-vs-14M architecture detail, mask-token handling under attention
mask. Investigating further while the experiment runs.
