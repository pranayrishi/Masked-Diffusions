#!/usr/bin/env bash
#
# 07_phase_e_seedbump.sh — Phase E: 5-seed bump on headline conditions.
#
# Adds 2 extra seeds (3, 4) to 4 headline conditions × 5 (N, P) configs = 40 jobs.
# variant=none already has 5 seeds in Phase B (seeds 0-4), so Phase E does NOT
# include it. The 4 conditions covered here:
#   0: best_filter (entropy)         — env BEST_VARIANT (e.g. "top_065")
#   1: percentile (entropy)
#   2: random_<best>                 — paired random control of best_filter
#   3: random_percentile             — paired random control of percentile
#
# BEST_VARIANT must be set as an environment variable BEFORE submitting:
#   export BEST_VARIANT=top_065
#   sbatch slurm/_rendered/07_phase_e_seedbump.sh
#
# (Or sbatch --export=ALL,BEST_VARIANT=top_065 ...)
#
# Index decomposition (0..39):
#   config_idx    = idx // 8
#   condition_idx = (idx // 2) % 4
#   seed_idx      = idx % 2          → seed = 3 + seed_idx (i.e., 3 or 4)

#SBATCH --job-name=mdm-phaseE
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-39%24
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

# Required environment input.
if [[ -z "${BEST_VARIANT:-}" ]]; then
    BEST_FILE="${PROJECT_DIR}/findings/best_variant.txt"
    if [[ -f "${BEST_FILE}" ]]; then
        BEST_VARIANT=$(cat "${BEST_FILE}")
    else
        echo "FAIL: BEST_VARIANT not set and ${BEST_FILE} not found"
        exit 6
    fi
fi
echo "[phase_e] BEST_VARIANT=${BEST_VARIANT}"

IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / 8))
CONDITION_IDX=$(((IDX / 2) % 4))
SEED=$((3 + (IDX % 2)))   # seed in {3, 4}

CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$CONFIG_IDX]}

# Resolve condition-specific overrides.
case $CONDITION_IDX in
    0)  # best_filter (entropy variant identified by BEST_VARIANT)
        COND_NAME="${BEST_VARIANT}"
        ENTROPY_OVERRIDES=""    # filled below by best-variant decoder
        ;;
    1)  # percentile (entropy)
        COND_NAME="percentile"
        ;;
    2)  # random_<best> (paired control)
        COND_NAME="random_${BEST_VARIANT}"
        ;;
    3)  # random_percentile
        COND_NAME="random_percentile"
        ;;
esac

# Decode condition name into mode + thresholds.
# (Same table as Phase C/D, plus random_replay handling.)
case $COND_NAME in
    top_055)        MODE="top"; HHIGH=0.55; HLOW=0.0 ;;
    top_060)        MODE="top"; HHIGH=0.60; HLOW=0.0 ;;
    top_065)        MODE="top"; HHIGH=0.65; HLOW=0.0 ;;
    top_070)        MODE="top"; HHIGH=0.70; HLOW=0.0 ;;
    bottom_030)     MODE="bottom"; HHIGH=1000000.0; HLOW=0.30 ;;
    band_030_055)   MODE="band"; HHIGH=0.55; HLOW=0.30 ;;
    band_030_060)   MODE="band"; HHIGH=0.60; HLOW=0.30 ;;
    band_030_065)   MODE="band"; HHIGH=0.65; HLOW=0.30 ;;
    band_030_070)   MODE="band"; HHIGH=0.70; HLOW=0.30 ;;
    percentile)     MODE="percentile_band"; HHIGH=1000000.0; HLOW=0.0 ;;
    random_*)
        # For random_replay, we need the paired entropy run's filter_trace.jsonl.
        ENT_NAME=${COND_NAME#random_}
        # paired entropy run for THIS seed lives in phase_e (same-seed) OR
        # the seed{3,4} bump is itself the entropy run we want to pair.
        # Strategy: pair with the Phase E entropy seed_3/4 run we are about to write —
        # but that creates a chicken-and-egg. Instead, pair with the same condition's
        # Phase C run (which ran seeds 0-2). Use the seed%3 mapping.
        # Concretely: random seed 3 pairs with entropy seed 0; random seed 4 pairs with entropy seed 1.
        PAIRED_SEED=$((SEED % 3))
        PHASE_C_PATTERN="${PROJECT_DIR}/results/phase_c/phase_c_${CONFIG}_${ENT_NAME}_seed${PAIRED_SEED}-*"
        PHASE_C_DIR=$(ls -td ${PHASE_C_PATTERN} 2>/dev/null | head -1)
        if [[ -z "${PHASE_C_DIR}" ]]; then
            echo "[phase_e ${IDX}] FAIL: paired Phase C dir not found: ${PHASE_C_PATTERN}"
            exit 7
        fi
        TRACE_FILE="${PHASE_C_DIR}/filter_trace.jsonl"
        MODE="random_replay"
        HHIGH=1000000.0
        HLOW=0.0
        ;;
    *)
        echo "[phase_e ${IDX}] FAIL: unknown condition name ${COND_NAME}"
        exit 8
        ;;
esac

CFG_FILE="entropy_filtered/configs/lo_nae_sat_${CONFIG}_none.yaml"
RUN_NAME="phase_e_${CONFIG}_${COND_NAME}_seed${SEED}"
SCRATCH_OUTPUT="${SCRATCH_DIR}/runs/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}"
PROJECT_RESULTS="${PROJECT_DIR}/results/phase_e/${RUN_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_OUTPUT}" "${PROJECT_RESULTS}"

echo "[phase_e ${IDX}] config=${CONFIG} condition=${COND_NAME} seed=${SEED} mode=${MODE}"

LATEST_CKPT=$(ls -t "${SCRATCH_OUTPUT}"/ckpt_step*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[phase_e ${IDX}] resuming from ${LATEST_CKPT}"
    RESUME_ARG="--resume ${LATEST_CKPT}"
fi

EXTRA_OVERRIDES=""
if [[ "${MODE}" == "random_replay" ]]; then
    EXTRA_OVERRIDES="--override entropy_filter.paired_trace_path=${TRACE_FILE}"
fi

python -m entropy_filtered.src.train_filtered \
    --config "${CFG_FILE}" \
    --override seed=${SEED} \
    --override num_iterations=50000 \
    --override output_dir="${SCRATCH_OUTPUT}" \
    --override entropy_filter.mode=${MODE} \
    --override entropy_filter.H_high=${HHIGH} \
    --override entropy_filter.H_low=${HLOW} \
    --override eval_num_samples=5000 \
    --override eval_test_seed=99999 \
    --override eval_num_steps=50 \
    ${EXTRA_OVERRIDES} ${RESUME_ARG}

rsync -av "${SCRATCH_OUTPUT}/" "${PROJECT_RESULTS}/"
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

EVAL_FILE="${PROJECT_RESULTS}/eval_results.json"
if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "[phase_e ${IDX}] FAIL: ${EVAL_FILE} does not exist"
    exit 2
fi
python3 -c "import json; d=json.load(open('${EVAL_FILE}')); assert d['results']; pairs=['{}={:.4%}'.format(r['strategy'], r['obs_accuracy']) for r in d['results']]; print('[phase_e ${IDX}] eval ok: ' + ', '.join(pairs))"

echo "[phase_e ${IDX}] DONE. Results: ${PROJECT_RESULTS}"
