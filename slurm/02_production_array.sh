#!/usr/bin/env bash
#
# 02_production_array.sh — Bouchet production job array: 75 entropy-filtered runs.
#
# Each task is one of (5 (N, P) configs) × (5 filter variants) × (3 seeds) = 75 jobs.
# Each job trains a 19M MDM with one filter mode, one (N, P), one seed for the same
# WALL-CLOCK budget. Final model + JSONL metrics + ckpts go to project_pi_<labname>.
#
# Walltime budget per task: SET FROM MEASUREMENT after smoke job 01_smoke.sh
# reports per-step seconds. Honest walltime ≈ 1.5 × measured wall = better queue
# priority. Bouchet partitions cap at 24 or 48 h; we'll fit inside whichever
# applies to ${PARTITION_PRODUCTION}.
#
# Concurrency: %${ARRAY_CONCURRENCY} is the maximum the user is permitted on this
# association; rendered from cluster_config.local.yaml after Phase 0 reads it from
# `sacctmgr show user $USER --associations`. PI has authorized full usage so we
# do NOT throttle below the limit.
#
# Partition choice: default ${PARTITION_PRODUCTION} (likely gpu_h200). If queue
# times look long (jobs PD for >1 hour on the smoke), switch to scavenge_gpu — our
# checkpoint/resume is bit-exact verified (baseline/tests/test_resume.py) so
# preemption costs are bounded.

#SBATCH --job-name=mdm-prod
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_PRODUCTION}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=${PRODUCTION_WALLTIME}                 # set from smoke measurement
#SBATCH --array=0-74%${ARRAY_CONCURRENCY}             # 75 tasks; cap from sacctmgr
#SBATCH --output=${PROJECT_DIR}/logs/%x-%A_%a.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${EMAIL}
#SBATCH --requeue                                     # auto-requeue on preemption (scavenge_gpu safe)

set -euo pipefail
module purge
module load ${PYTORCH_MODULE}

cd $SLURM_SUBMIT_DIR

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
# DO NOT set PYTHONPATH — would clobber the YCRC PyTorch module's site-packages.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Map array index → (N, P) config × variant × seed.
# Index decomposition (75 = 5 × 5 × 3):
#     idx  = config_idx * 15 + variant_idx * 3 + seed_idx
#   ⇒ seed_idx    = idx % 3
#     variant_idx = (idx // 3) % 5
#     config_idx  = idx // 15
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX % 3))
VARIANT_IDX=$(((IDX / 3) % 5))
CONFIG_IDX=$((IDX / 15))

CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
VARIANTS=("none" "bottom" "top" "band" "percentile")
SEEDS=(0 1 2)

CONFIG=${CONFIGS[$CONFIG_IDX]}
VARIANT=${VARIANTS[$VARIANT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CFG_FILE="entropy_filtered/configs/lo_nae_sat_${CONFIG}_${VARIANT}.yaml"
RUN_NAME="lo_nae_sat_${CONFIG}_${VARIANT}_seed${SEED}"
# Use a STABLE run dir keyed only on the array job id (NOT the array task id),
# so a requeued task resumes into the same dir. test_resume.py guarantees
# bit-exact resume across processes.
OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${OUTPUT}"

echo "[task ${IDX}] config=${CONFIG} variant=${VARIANT} seed=${SEED}"
echo "[task ${IDX}] config-file=${CFG_FILE}"
echo "[task ${IDX}] output=${OUTPUT}"

# If a previous (preempted) attempt left a checkpoint, resume from the latest one.
LATEST_CKPT=$(ls -t "${OUTPUT}"/ckpt_step*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[task ${IDX}] resuming from ${LATEST_CKPT}"
    python -m entropy_filtered.src.train_filtered \
        --config "${CFG_FILE}" \
        --override seed=${SEED} \
        --override output_dir="${OUTPUT}" \
        --resume "${LATEST_CKPT}"
else
    echo "[task ${IDX}] starting fresh"
    python -m entropy_filtered.src.train_filtered \
        --config "${CFG_FILE}" \
        --override seed=${SEED} \
        --override output_dir="${OUTPUT}"
fi

# Move final results into the BACKED-UP project area
PROJECT_RESULTS="${PROJECT_DIR}/results/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${PROJECT_RESULTS}"
rsync -av "${OUTPUT}/" "${PROJECT_RESULTS}/"

# jobstats for right-sizing future runs
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

echo "[task ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
