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
