#!/usr/bin/env bash
#
# 01_smoke.sh — Bouchet smoke job: 1 H200 GPU on the gpu_devel partition,
#               30-minute walltime. Validates the GPU pipeline end-to-end
#               before committing any GPU-hours to the production array.
#
# What it does (in order):
#   1. Loads the YCRC PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 module — same exact
#      torch + numpy versions our local tests use, no pip install needed.
#   2. Runs all 52 unit tests (32 baseline + 20 entropy_filtered) on the GPU node.
#   3. Runs baseline/scripts/smoke_test.sh — a tiny end-to-end training on a
#      (5, 10) toy distribution, additional sanity beyond the unit tests.
#   4. GPU sanity print: torch.cuda.is_available() + device name.
#   5. Runs ONE production-shape filtered training: 19M MDM, RoPE,
#      (N, P) = (25, 275), mode = band, 250 steps. Overrides the production
#      log_every (100) → 10 and the production warmup_steps (500) → 50, so we
#      get rich logging AND the filter actually fires within 250 steps.
#   6. Post-processes the resulting metrics.jsonl into:
#        a. 20-bin histogram of filter_H_mean across all logged steps;
#        b. Trajectory of H_min / H_mean / H_max at steps {1, 50, 100, 150, 200, 250};
#        c. Acceptance-rate trajectory at the same checkpoints + 250-step average;
#        d. Total drop counts.
#      This is the diagnostic that drives any threshold retuning. The agent
#      hands the trajectory back; the user + professor decide thresholds.
#   7. rsyncs results from scratch into project (backed up) and runs jobstats
#      for right-sizing the production array's --time= and --mem=.
#
# Renders from this template via slurm/_render_scripts.sh from the .gitignored
# slurm/cluster_config.local.yaml. Do NOT edit the rendered output by hand.

#SBATCH --job-name=mdm-smoke
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_SMOKE}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${PROJECT_DIR}/logs/%x-%j.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${EMAIL}

set -euo pipefail
module purge
module load ${PYTORCH_MODULE}

cd $SLURM_SUBMIT_DIR

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
# DO NOT set PYTHONPATH — would clobber the YCRC PyTorch module's site-packages.
# `cd $SLURM_SUBMIT_DIR` + `python -m` makes our packages importable.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OUTPUT="${SCRATCH_DIR}/runs/smoke-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT}"
mkdir -p "${PROJECT_DIR}/logs"

echo "==> 01/06 baseline unit tests (32)"
( cd baseline && python -m pytest tests/ -q )

echo "==> 02/06 entropy_filtered unit tests (20)"
( cd entropy_filtered && python -m pytest tests/ -q )

echo "==> 03/06 baseline smoke (tiny end-to-end on (5,10) toy)"
bash baseline/scripts/smoke_test.sh

echo "==> 04/06 GPU sanity"
python - <<'PYEOF'
import torch
print(f"torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device 0 = {torch.cuda.get_device_name(0)}")
PYEOF

echo "==> 05/06 production-shape filtered run (250 steps, mode=band, N=25 P=275)"
echo "    Overrides: num_iterations=250, log_every=10, entropy_filter.warmup_steps=50"
RUN_DIR="${OUTPUT}/band_250steps"
python -m entropy_filtered.src.train_filtered \
    --config entropy_filtered/configs/lo_nae_sat_25_275_band.yaml \
    --override num_iterations=250 \
    --override log_every=10 \
    --override ckpt_every=250 \
    --override entropy_filter.warmup_steps=50 \
    --override output_dir="${RUN_DIR}"

echo "==> 06/06 post-process metrics.jsonl → entropy + acceptance diagnostics"
python - "${RUN_DIR}/metrics.jsonl" <<'PYEOF'
"""Diagnostics for threshold retuning. Reads the metrics.jsonl from the smoke
run and prints a histogram of filter_H_mean, the H trajectory, and the
acceptance-rate trajectory. No external dependencies."""
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if not rows:
    print("FAIL: empty metrics.jsonl"); sys.exit(1)

# Identify the warmup window from the config logged in the run dir
cfg_path = path.parent / "config.yaml"
warmup_steps = 50  # default smoke override
try:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    warmup_steps = int(cfg.get("entropy_filter", {}).get("warmup_steps", warmup_steps))
except Exception:
    pass
print(f"[diagnostics] warmup_steps from config: {warmup_steps}")
print(f"[diagnostics] logged rows: {len(rows)}")

# ------ (a) 20-bin histogram of filter_H_mean across all logged steps ------
H = [r["filter_H_mean"] for r in rows if "filter_H_mean" in r]
print()
print("=" * 72)
print(f"(a) Histogram of filter_H_mean across {len(H)} logged steps")
print("=" * 72)
if H:
    lo, hi = min(H), max(H)
    nbins = 20
    bw = (hi - lo) / nbins if hi > lo else 1.0
    counts = [0] * nbins
    for h in H:
        idx = nbins - 1 if h == hi else int((h - lo) / bw)
        counts[idx] += 1
    mx = max(counts) if counts else 0
    for i, c in enumerate(counts):
        a = lo + i * bw
        b = a + bw
        bar = "#" * int(50 * c / mx) if mx else ""
        print(f"  [{a:6.4f}, {b:6.4f}): {c:>4}  {bar}")

# ------ (b) Trajectory of H_min / H_mean / H_max at checkpoint steps ------
print()
print("=" * 72)
print("(b) H_min / H_mean / H_max trajectory at checkpoint steps")
print("=" * 72)
print(f"{'step':>5}  {'H_min':>7}  {'H_mean':>7}  {'H_max':>7}")
checkpoints = [1, 50, 100, 150, 200, 250]
for target in checkpoints:
    nearest = min(rows, key=lambda r: abs(r["step"] - target))
    print(f"  {nearest['step']:>5}  {nearest['filter_H_min']:>7.4f}  "
          f"{nearest['filter_H_mean']:>7.4f}  {nearest['filter_H_max']:>7.4f}")

# ------ (c) Acceptance-rate trajectory at checkpoints + 250-step average ------
print()
print("=" * 72)
print("(c) Filter acceptance rate (kept / batch_size) trajectory")
print("=" * 72)
print(f"{'step':>5}  {'kept':>5}  {'dropped':>7}  {'accept':>7}")
for target in checkpoints:
    nearest = min(rows, key=lambda r: abs(r["step"] - target))
    kept = nearest.get("filter_n_kept", 0)
    dropped = nearest.get("filter_n_dropped", 0)
    accept = kept / max(1, kept + dropped)
    print(f"  {nearest['step']:>5}  {kept:>5}  {dropped:>7}  {accept:>6.1%}")

# Average acceptance rate post-warmup. Pre-warmup filter is OFF so kept==batch.
post = [r for r in rows if r["step"] > warmup_steps]
total_kept = sum(r.get("filter_n_kept", 0) for r in post)
total_dropped = sum(r.get("filter_n_dropped", 0) for r in post)
avg_accept = total_kept / max(1, total_kept + total_dropped)
print()
print(f"  Post-warmup ({len(post)} steps logged): kept={total_kept}, dropped={total_dropped}, "
      f"avg acceptance = {avg_accept:.2%}")

# ------ (d) Total drop counts (already part of the above, surfaced explicitly) ------
print()
print("=" * 72)
print("(d) Total drop counts")
print("=" * 72)
all_kept = sum(r.get("filter_n_kept", 0) for r in rows)
all_dropped = sum(r.get("filter_n_dropped", 0) for r in rows)
print(f"  Across all {len(rows)} logged steps: kept={all_kept}, dropped={all_dropped}")
print(f"  Post-warmup only ({len(post)} steps): kept={total_kept}, dropped={total_dropped}")
PYEOF

# Move final results into the BACKED-UP project area
PROJECT_RESULTS="${PROJECT_DIR}/results/smoke-${SLURM_JOB_ID}"
mkdir -p "${PROJECT_RESULTS}"
rsync -av "${OUTPUT}/" "${PROJECT_RESULTS}/"

# Right-sizing input for the production array's --time= and --mem=
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

echo
echo "Smoke complete. Results: ${PROJECT_RESULTS}"
echo "Diagnostics for threshold retuning are above (sections a–d)."
