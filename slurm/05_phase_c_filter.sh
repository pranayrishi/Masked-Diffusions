#!/usr/bin/env bash
#
# 05_phase_c_filter.sh — Phase C: entropy-filter conditions sweep.
#
# 150 jobs = 5 (N, P) configs × 10 entropy-filter conditions × 3 seeds. Each job
# trains the 14M MDM with one filter config, runs held-out evaluation under
# vanilla and adaptive (top_prob_margin) inference, and writes filter_trace.jsonl
# (used by Phase D's paired random_replay controls).
#
# 10 entropy-filter conditions:
#   0: top, H_high=0.55, name="top_055"
#   1: top, H_high=0.60, name="top_060"
#   2: top, H_high=0.65, name="top_065"
#   3: top, H_high=0.70, name="top_070"
#   4: bottom, H_low=0.30, name="bottom_030"
#   5: band, H_low=0.30, H_high=0.55, name="band_030_055"
#   6: band, H_low=0.30, H_high=0.60, name="band_030_060"
#   7: band, H_low=0.30, H_high=0.65, name="band_030_065"
#   8: band, H_low=0.30, H_high=0.70, name="band_030_070"
#   9: percentile_band, pct_low=0.25, pct_high=0.75, name="percentile"
#
# Index mapping (0..149):
#   config_idx    = idx // 30
#   condition_idx = (idx // 3) % 10
#   seed_idx      = idx % 3

#SBATCH --job-name=mdm-phaseC
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

# Index decomposition
IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / 30))
CONDITION_IDX=$(((IDX / 3) % 10))
SEED=$((IDX % 3))

CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$CONFIG_IDX]}

# Condition table — order must match the comment block above.
COND_NAMES=("top_055" "top_060" "top_065" "top_070" "bottom_030" \
            "band_030_055" "band_030_060" "band_030_065" "band_030_070" "percentile")
COND_MODES=("top" "top" "top" "top" "bottom" \
            "band" "band" "band" "band" "percentile_band")
COND_HHIGH=("0.55" "0.60" "0.65" "0.70" "1000000.0" \
            "0.55" "0.60" "0.65" "0.70" "1000000.0")
COND_HLOW=("0.0" "0.0" "0.0" "0.0" "0.30" \
           "0.30" "0.30" "0.30" "0.30" "0.0")

COND_NAME=${COND_NAMES[$CONDITION_IDX]}
COND_MODE=${COND_MODES[$CONDITION_IDX]}
COND_HHIGH_VAL=${COND_HHIGH[$CONDITION_IDX]}
COND_HLOW_VAL=${COND_HLOW[$CONDITION_IDX]}

CFG_FILE="entropy_filtered/configs/lo_nae_sat_${CONFIG}_none.yaml"
RUN_NAME="phase_c_${CONFIG}_${COND_NAME}_seed${SEED}"
SCRATCH_OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
PROJECT_RESULTS="${PROJECT_DIR}/results/phase_c/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_OUTPUT}" "${PROJECT_RESULTS}"

echo "[task ${IDX}] config=${CONFIG} condition=${COND_NAME} (mode=${COND_MODE} H_high=${COND_HHIGH_VAL} H_low=${COND_HLOW_VAL}) seed=${SEED}"

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
    --override entropy_filter.mode=${COND_MODE} \
    --override entropy_filter.H_high=${COND_HHIGH_VAL} \
    --override entropy_filter.H_low=${COND_HLOW_VAL} \
    --override eval_num_samples=5000 \
    --override eval_test_seed=99999 \
    --override eval_num_steps=50 \
    --override eval_noise=gumbel \
    --override eval_noise_scale=0.5 \
    ${RESUME_ARG}

# Sync to backed-up project area.
rsync -av "${SCRATCH_OUTPUT}/" "${PROJECT_RESULTS}/"
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

# Final-step assertion: eval_results.json + filter_trace.jsonl must both exist.
EVAL_FILE="${PROJECT_RESULTS}/eval_results.json"
TRACE_FILE="${PROJECT_RESULTS}/filter_trace.jsonl"
for f in "${EVAL_FILE}" "${TRACE_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[task ${IDX}] FAIL: ${f} does not exist"
        exit 2
    fi
done
python3 -c "import json; d=json.load(open('${EVAL_FILE}')); assert d['results']; pairs=['{}={:.4%}'.format(r['strategy'], r['obs_accuracy']) for r in d['results']]; print('[task ${IDX}] eval ok: ' + ', '.join(pairs))"

echo "[task ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
