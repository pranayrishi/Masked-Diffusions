#!/usr/bin/env bash
#
# 06_phase_d_random.sh — Phase D: paired random_replay controls.
#
# 150 jobs = 5 (N, P) configs × 10 entropy-filter conditions × 3 seeds. Each job
# trains the SAME model+seed+data as the paired Phase C entropy run, but with
# entropy_filter.mode="random_replay" — the filter drops the SAME number of
# samples per step as the paired entropy run did, but selects them uniformly at
# random instead of by H. The paired Phase C run's filter_trace.jsonl is
# auto-discovered from the project results dir.
#
# Submit AFTER Phase C array has completed (or with --dependency=afterok:<phase_c_jobid>).
#
# Same condition table as Phase C; identical index decomposition (config × condition × seed).

#SBATCH --job-name=mdm-phaseD
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-149%48
#SBATCH --output=${PROJECT_DIR}/logs/%x-%A_%a.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
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

IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / 30))
CONDITION_IDX=$(((IDX / 3) % 10))
SEED=$((IDX % 3))

CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$CONFIG_IDX]}

COND_NAMES=("top_055" "top_060" "top_065" "top_070" "bottom_030" \
            "band_030_055" "band_030_060" "band_030_065" "band_030_070" "percentile")
COND_NAME=${COND_NAMES[$CONDITION_IDX]}

CFG_FILE="entropy_filtered/configs/lo_nae_sat_${CONFIG}_none.yaml"

# Discover the paired Phase C run's filter_trace.jsonl. There may be multiple
# matching directories if Phase C was re-run; take the most recent.
PHASE_C_PATTERN="${PROJECT_DIR}/results/phase_c/phase_c_${CONFIG}_${COND_NAME}_seed${SEED}-*"
PHASE_C_DIR=$(ls -td ${PHASE_C_PATTERN} 2>/dev/null | head -1)
if [[ -z "${PHASE_C_DIR}" ]]; then
    echo "[task ${IDX}] FAIL: no Phase C results matching ${PHASE_C_PATTERN}"
    echo "[task ${IDX}]   Phase C must complete before Phase D for this (config, condition, seed)."
    exit 4
fi
TRACE_FILE="${PHASE_C_DIR}/filter_trace.jsonl"
if [[ ! -f "${TRACE_FILE}" ]]; then
    echo "[task ${IDX}] FAIL: paired filter_trace.jsonl missing at ${TRACE_FILE}"
    exit 5
fi

RUN_NAME="phase_d_${CONFIG}_${COND_NAME}_seed${SEED}"
SCRATCH_OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
PROJECT_RESULTS="${PROJECT_DIR}/results/phase_d/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_OUTPUT}" "${PROJECT_RESULTS}"

echo "[task ${IDX}] config=${CONFIG} condition=${COND_NAME} seed=${SEED}"
echo "[task ${IDX}] paired_trace=${TRACE_FILE}"

# Resume on requeue
LATEST_CKPT=$(ls -t "${SCRATCH_OUTPUT}"/ckpt_step*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[task ${IDX}] resuming from ${LATEST_CKPT}"
    RESUME_ARG="--resume ${LATEST_CKPT}"
fi

python -m entropy_filtered.src.train_filtered \
    --config "${CFG_FILE}" \
    --override seed=${SEED} \
    --override num_iterations=50000 \
    --override output_dir="${SCRATCH_OUTPUT}" \
    --override entropy_filter.mode=random_replay \
    --override entropy_filter.paired_trace_path="${TRACE_FILE}" \
    --override eval_num_samples=5000 \
    --override eval_test_seed=99999 \
    --override eval_num_steps=50 \
    --override eval_noise=gumbel \
    --override eval_noise_scale=0.5 \
    ${RESUME_ARG}

rsync -av "${SCRATCH_OUTPUT}/" "${PROJECT_RESULTS}/"
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

EVAL_FILE="${PROJECT_RESULTS}/eval_results.json"
if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "[task ${IDX}] FAIL: ${EVAL_FILE} does not exist"
    exit 2
fi
python3 -c "import json; d=json.load(open('${EVAL_FILE}')); assert d['results']; pairs=['{}={:.4%}'.format(r['strategy'], r['obs_accuracy']) for r in d['results']]; print('[task ${IDX}] eval ok: ' + ', '.join(pairs))"

echo "[task ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
