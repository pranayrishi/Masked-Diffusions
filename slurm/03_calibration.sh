#!/usr/bin/env bash
#
# 03_calibration.sh — Bouchet calibration array: 5 variant=none jobs, one per (N, P).
#
# Purpose: determine the per-config `num_iterations` for the production array.
# Each job trains the 14M MDM until plateau (detected DURING training every 500
# steps; train_filtered.py exits cleanly on detection) OR a 50K-step cap.
#
# Plateau criterion (in entropy_filtered.src.train_filtered._is_plateau):
#   |mean(loss[step-W+1..step]) - mean(loss[step-2W+1..step-W])| / mean(prev) < tol
# with W=500, tol=0.005 (0.5% relative), check_every=500, min_step=1000.
# Skipped-optim steps (filter dropped the entire batch) are excluded from the means.
#
# Output (per task) under ${PROJECT_DIR}/results/calibration/<run>/:
#   - plateau_step.txt   single integer, no whitespace; written by train_filtered.py
#   - metrics.jsonl, config.yaml, ckpt_step*.pt, warnings.log
#   - deterministic_warnings.log  filtered to torch determinism warnings only
#
# This script DOES NOT detect plateau itself — that is owned by train_filtered.py.
# This script ONLY: launches training, copies results to the BACKED-UP project area,
# extracts determinism warnings, and asserts the output file exists in the correct
# format before the job exits.
#
# Renders from this template via slurm/_render_scripts.sh from the .gitignored
# slurm/cluster_config.local.yaml. Do NOT edit the rendered output by hand.

#SBATCH --job-name=mdm-calib
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_PRODUCTION}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --array=0-4
#SBATCH --output=${PROJECT_DIR}/logs/%x-%A_%a.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${EMAIL}
#SBATCH --requeue

set -euo pipefail
module purge
module load ${PYTORCH_MODULE}

cd $SLURM_SUBMIT_DIR

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
# DO NOT set PYTHONPATH — would clobber the YCRC PyTorch module's site-packages.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Map array index → (N, P) config.
IDX=$SLURM_ARRAY_TASK_ID
CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$IDX]}

CFG_FILE="entropy_filtered/configs/calibration/lo_nae_sat_${CONFIG}_none.yaml"
RUN_NAME="calibration_${CONFIG}_none"
SCRATCH_OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SCRATCH_OUTPUT}"

# Final landing place, in the BACKED-UP project area. plateau_step.txt MUST end up here
# (and be a parseable single integer) before this script exits non-zero is OK; we want
# the assertion below to surface any failure to the user, not silently succeed.
PROJECT_RESULTS="${PROJECT_DIR}/results/calibration/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${PROJECT_RESULTS}"

echo "[task ${IDX}] config=${CONFIG}  cfg-file=${CFG_FILE}"
echo "[task ${IDX}] scratch_output=${SCRATCH_OUTPUT}"
echo "[task ${IDX}] project_results=${PROJECT_RESULTS}"

# Resume from latest checkpoint if a previous attempt was preempted (--requeue safe).
LATEST_CKPT=$(ls -t "${SCRATCH_OUTPUT}"/ckpt_step*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[task ${IDX}] resuming from ${LATEST_CKPT}"
    python -m entropy_filtered.src.train_filtered \
        --config "${CFG_FILE}" \
        --override output_dir="${SCRATCH_OUTPUT}" \
        --resume "${LATEST_CKPT}"
else
    echo "[task ${IDX}] starting fresh"
    python -m entropy_filtered.src.train_filtered \
        --config "${CFG_FILE}" \
        --override output_dir="${SCRATCH_OUTPUT}"
fi

# train_filtered.py wrote plateau_step.txt directly (single integer, no whitespace).
# Sync everything to the project (backed-up) area.
rsync -av "${SCRATCH_OUTPUT}/" "${PROJECT_RESULTS}/"

# Filter the captured warnings.log down to torch-determinism warnings for the audit
# trail. We grep case-insensitive on "deterministic" to catch both
# "Deterministic behavior" and "non-deterministic" wording.
if [[ -f "${PROJECT_RESULTS}/warnings.log" ]]; then
    grep -i deterministic "${PROJECT_RESULTS}/warnings.log" \
        > "${PROJECT_RESULTS}/deterministic_warnings.log" 2>/dev/null || true
    if [[ -s "${PROJECT_RESULTS}/deterministic_warnings.log" ]]; then
        echo "[task ${IDX}] determinism warnings captured:"
        cat "${PROJECT_RESULTS}/deterministic_warnings.log"
    else
        echo "[task ${IDX}] no torch-determinism warnings emitted (warn_only=True path clean)"
    fi
fi

# jobstats for right-sizing the production array's --time= and --mem=
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

# === Final-step assertion: plateau_step.txt MUST exist in the project area AND parse
#     as a single integer. Failure here exits non-zero so the user sees the failure
#     before the next stage tries to use it. ===
PLATEAU_FILE="${PROJECT_RESULTS}/plateau_step.txt"
if [[ ! -f "${PLATEAU_FILE}" ]]; then
    echo "[task ${IDX}] FAIL: ${PLATEAU_FILE} does not exist"
    exit 2
fi

# Validate the file contents are EXACTLY one integer with no extra whitespace.
PLATEAU_RAW=$(cat "${PLATEAU_FILE}")
if ! [[ "${PLATEAU_RAW}" =~ ^[0-9]+$ ]]; then
    echo "[task ${IDX}] FAIL: plateau_step.txt is not a single integer: ${PLATEAU_RAW@Q}"
    exit 3
fi

PLATEAU_STEP=$((10#${PLATEAU_RAW}))   # base-10 to avoid 0-prefixed-octal weirdness
echo "[task ${IDX}] DONE.  config=${CONFIG}  plateau_step=${PLATEAU_STEP}"
echo "[task ${IDX}] Results: ${PROJECT_RESULTS}"
