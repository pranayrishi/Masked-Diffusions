#!/usr/bin/env bash
#
# 02_production_array.sh — Bouchet production job array: 75 entropy-filtered runs.
#
# Each task is one of (5 (N, P) configs) × (5 filter variants) × (3 seeds) = 75 jobs.
# Each job trains a 19M MDM with one filter mode, one (N, P), one seed for the same
# WALL-CLOCK budget. Final model + JSONL metrics + ckpts go to project_pi_<labname>.
#
# Walltime budget per task: 8 hours. Will be revised down once we have measured
# steps/sec from the smoke job (`01_smoke.sh`) on the actual partition.
#
# Total scheduled array = 75 tasks. With 4 GPUs available concurrently on
# pi_jks79 (assumed; revise after PI confirms partition), wall-clock to finish
# the array is ~75 × 8h ÷ 4 = 150 hours = ~6 days. With 8 GPUs concurrent: ~75 hours = ~3 days.
#
# Estimated GPU-hours: 75 tasks × 8 h walltime = 600 GPU-hours UPPER BOUND. The actual
# is likely 150-300 GPU-hours (each run completes well before walltime). The smoke job
# will give us the real per-step rate; we'll revise --time= for the production submission.

#SBATCH --job-name=mdm-prod
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_PRODUCTION}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-74%8                                # 75 tasks; 8 concurrent (revise per PI)
#SBATCH --output=${PROJECT_DIR}/logs/%x-%A_%a.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${EMAIL}

set -euo pipefail
module purge
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

cd $SLURM_SUBMIT_DIR

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
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
OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTPUT}"

echo "[task ${IDX}] config=${CONFIG} variant=${VARIANT} seed=${SEED}"
echo "[task ${IDX}] config-file=${CFG_FILE}"
echo "[task ${IDX}] output=${OUTPUT}"

python -m entropy_filtered.src.train_filtered \
    --config "${CFG_FILE}" \
    --override seed=${SEED} \
    --override output_dir="${OUTPUT}"

# Move final results into the BACKED-UP project area
PROJECT_RESULTS="${PROJECT_DIR}/results/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${PROJECT_RESULTS}"
rsync -av "${OUTPUT}/" "${PROJECT_RESULTS}/"

# jobstats for right-sizing future runs
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

echo "[task ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
