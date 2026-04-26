#!/usr/bin/env bash
#
# 08_phase_f_eval10k.sh — Phase F: 10,000-sample re-evaluation of headline conditions.
#
# Eval-only (no training). Loads checkpoints from Phase B/C/D/E and runs vanilla +
# adaptive (top_prob_margin) inference at 10,000 test samples for camera-ready
# precision. Each task writes eval_results_10k.json into the source run dir.
#
# 125 tasks = 5 (N, P) configs × 5 headline conditions × 5 seeds.
#
# Headline conditions:
#   0: none           (Phase B; seeds 0-4 all in Phase B)
#   1: BEST_VARIANT   (Phase C for seeds 0-2; Phase E for seeds 3-4)
#   2: percentile     (Phase C for seeds 0-2; Phase E for seeds 3-4)
#   3: random_BEST    (Phase D for seeds 0-2; Phase E for seeds 3-4)
#   4: random_pct     (Phase D for seeds 0-2; Phase E for seeds 3-4)
#
# BEST_VARIANT must be set as an env var at submission time (or via the auto-discovered
# ${PROJECT_DIR}/findings/best_variant.txt).
#
# Index decomposition (0..124):
#   config_idx    = idx // 25
#   condition_idx = (idx // 5) % 5
#   seed_idx      = idx % 5

#SBATCH --job-name=mdm-phaseF
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION_PRODUCTION}
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-124%24
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

# Best-variant resolution (matches Phase E)
if [[ -z "${BEST_VARIANT:-}" ]]; then
    BEST_FILE="${PROJECT_DIR}/findings/best_variant.txt"
    if [[ -f "${BEST_FILE}" ]]; then
        BEST_VARIANT=$(cat "${BEST_FILE}")
    else
        echo "FAIL: BEST_VARIANT not set and ${BEST_FILE} not found"
        exit 6
    fi
fi
echo "[phase_f] BEST_VARIANT=${BEST_VARIANT}"

IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / 25))
CONDITION_IDX=$(((IDX / 5) % 5))
SEED=$((IDX % 5))

CONFIGS=("25_275" "30_270" "40_260" "50_250" "100_200")
CONFIG=${CONFIGS[$CONFIG_IDX]}

# Map condition_idx → (variant_name, source_phase_for_seed_0_to_2, source_phase_for_seed_3_4)
case $CONDITION_IDX in
    0) VARIANT_NAME="none";          PHASE_LO="phase_b"; PHASE_HI="phase_b" ;;
    1) VARIANT_NAME="${BEST_VARIANT}"; PHASE_LO="phase_c"; PHASE_HI="phase_e" ;;
    2) VARIANT_NAME="percentile";    PHASE_LO="phase_c"; PHASE_HI="phase_e" ;;
    3) VARIANT_NAME="random_${BEST_VARIANT}"; PHASE_LO="phase_d"; PHASE_HI="phase_e" ;;
    4) VARIANT_NAME="random_percentile";      PHASE_LO="phase_d"; PHASE_HI="phase_e" ;;
esac

# Pick phase based on seed
if [[ $SEED -le 2 ]]; then
    SOURCE_PHASE=$PHASE_LO
else
    SOURCE_PHASE=$PHASE_HI
fi

# Locate the source run directory.
SOURCE_PATTERN="${PROJECT_DIR}/results/${SOURCE_PHASE}/${SOURCE_PHASE}_${CONFIG}_${VARIANT_NAME}_seed${SEED}-*"
SOURCE_DIR=$(ls -td ${SOURCE_PATTERN} 2>/dev/null | head -1)
if [[ -z "${SOURCE_DIR}" ]]; then
    echo "[phase_f ${IDX}] FAIL: source run dir not found: ${SOURCE_PATTERN}"
    exit 4
fi
echo "[phase_f ${IDX}] config=${CONFIG} variant=${VARIANT_NAME} seed=${SEED}"
echo "[phase_f ${IDX}] source=${SOURCE_DIR}"

python -m baseline.src.run_eval_only \
    --run-dir "${SOURCE_DIR}" \
    --num-samples 10000 \
    --strategies vanilla,top_prob_margin \
    --num-steps 50 \
    --eval-test-seed 99999 \
    --output-name eval_results_10k.json

# Validate and copy a thin summary into a Phase F results dir for aggregation.
PHASE_F_DIR="${PROJECT_DIR}/results/phase_f/${SOURCE_PHASE}_${CONFIG}_${VARIANT_NAME}_seed${SEED}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${PHASE_F_DIR}"
cp "${SOURCE_DIR}/eval_results_10k.json" "${PHASE_F_DIR}/eval_results.json"   # canonical name for aggregator
cp "${SOURCE_DIR}/config.yaml" "${PHASE_F_DIR}/config.yaml"
[[ -f "${SOURCE_DIR}/metrics.jsonl" ]] && cp "${SOURCE_DIR}/metrics.jsonl" "${PHASE_F_DIR}/metrics.jsonl" || true
jobstats $SLURM_JOB_ID > "${PHASE_F_DIR}/jobstats.txt" 2>&1 || true

EVAL_FILE="${PHASE_F_DIR}/eval_results.json"
if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "[phase_f ${IDX}] FAIL: ${EVAL_FILE} does not exist"
    exit 2
fi
python3 -c "import json; d=json.load(open('${EVAL_FILE}')); assert d['results']; print(f'[phase_f ${IDX}] eval10k ok: ' + ', '.join(f\"{r[\\\"strategy\\\"]}={r[\\\"obs_accuracy\\\"]:.4%}\" for r in d['results']))"

echo "[phase_f ${IDX}] DONE. Results: ${PHASE_F_DIR}"
