#!/usr/bin/env bash
#
# 04_phase_b_baseline.sh — Phase B reproduction baseline.
#
# 25 jobs = 5 (N, P) configs × 5 seeds, variant=none. Each job trains the 14M MDM
# for 10,000 iterations (or until the rolling-mean plateau detector fires) and runs
# held-out evaluation under both vanilla and adaptive (top_prob_margin) inference
# at 5,000 test samples. Targets: paper Table 1 (vanilla {78.06, 75.70, 74.60, 67.94,
# 62.84}, adaptive {93.76, 93.54, 92.21, 90.01, 88.91}).
#
# Output (per task) under ${PROJECT_DIR}/results/phase_b/<run>/:
#   - eval_results.json     {strategy, accuracy, num_samples, ...}
#   - metrics.jsonl         per-step training diagnostics
#   - filter_trace.jsonl    per-step keep counts (always written; trivial for variant=none)
#   - plateau_step.txt      single integer
#   - ckpt_step*.pt         checkpoints
#   - warnings.log          (should be empty: warn_only=False with audit-clean H200)
#   - jobstats.txt          Slurm resource accounting

#SBATCH --job-name=mdm-phaseB
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_PRODUCTION}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-24
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Map array index → (config, seed). 25 = 5 configs × 5 seeds.
IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / 5))
SEED=$((IDX % 5))
CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$CONFIG_IDX]}

CFG_FILE="entropy_filtered/configs/lo_nae_sat_${CONFIG}_none.yaml"
RUN_NAME="phase_b_${CONFIG}_none_seed${SEED}"
SCRATCH_OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
PROJECT_RESULTS="${PROJECT_DIR}/results/phase_b/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_OUTPUT}" "${PROJECT_RESULTS}"

echo "[task ${IDX}] config=${CONFIG} seed=${SEED}  cfg-file=${CFG_FILE}"

# Resume on requeue.
LATEST_CKPT=$(ls -t "${SCRATCH_OUTPUT}"/ckpt_step*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[task ${IDX}] resuming from ${LATEST_CKPT}"
    RESUME_ARG="--resume ${LATEST_CKPT}"
fi

# Phase B uses 10K iterations (the user-authorized minimum) with the same plateau
# detector enabled — early-stop fires if loss plateaus before 10K. Eval at 5K samples
# under both vanilla and adaptive inference.
python -m entropy_filtered.src.train_filtered \
    --config "${CFG_FILE}" \
    --override seed=${SEED} \
    --override num_iterations=50000 \
    --override output_dir="${SCRATCH_OUTPUT}" \
    --override eval_num_samples=5000 \
    --override eval_test_seed=99999 \
    --override eval_num_steps=50 \
    --override early_stop.enabled=true \
    --override early_stop.criterion=rolling_mean_relative \
    --override early_stop.tolerance=0.005 \
    --override early_stop.window=2000 \
    --override early_stop.check_every=1000 \
    --override early_stop.min_step=10000 \
    ${RESUME_ARG}

# Sync to backed-up project area.
rsync -av "${SCRATCH_OUTPUT}/" "${PROJECT_RESULTS}/"

# jobstats for cost accounting.
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

# Final-step assertion: eval_results.json MUST exist and parse as valid JSON.
EVAL_FILE="${PROJECT_RESULTS}/eval_results.json"
if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "[task ${IDX}] FAIL: ${EVAL_FILE} does not exist"
    exit 2
fi
python3 -c "import json; d=json.load(open('${EVAL_FILE}')); assert d['results'], 'no eval results'; pairs=['{}={:.4%}'.format(r['strategy'], r['obs_accuracy']) for r in d['results']]; print('[task ${IDX}] eval ok: ' + ', '.join(pairs))"

echo "[task ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
