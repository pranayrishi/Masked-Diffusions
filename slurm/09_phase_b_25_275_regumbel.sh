#!/usr/bin/env bash
#
# 09_phase_b_25_275_regumbel.sh — Re-evaluate the 5 finished (25, 275) Phase B
# 50K checkpoints with Gumbel(0, 0.5) noise on adaptive scores. Diagnostic
# run to test whether adding the missing paper-prescribed Gumbel noise closes
# the adaptive-vs-vanilla gap (currently ~5pp; paper reports ~16pp).
#
# Loads the same checkpoints used in the original eval; writes a new
# eval_results_gumbel.json into each run directory. Eval-only, so much faster
# than training (~20 min/task).

#SBATCH --job-name=mdm-regumbel
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --output=${PROJECT_DIR}/logs/%x-%A_%a.out
#SBATCH --error=${PROJECT_DIR}/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${EMAIL}

set -euo pipefail
module purge
module load ${PYTORCH_MODULE}
cd $SLURM_SUBMIT_DIR

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SEED=$SLURM_ARRAY_TASK_ID

# Find the (25, 275) Phase B run dir for this seed. Tasks 0,2,4 came from
# 9556517 (50K trained). Tasks 1,3 came from 9556517 too but plateau-stopped at
# 10K — we'll re-eval them anyway to see Gumbel behavior at 10K.
RUN_DIR=$(ls -td ${PROJECT_DIR}/results/phase_b/phase_b_25_275_none_seed${SEED}-* 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
    echo "[regumbel ${SEED}] FAIL: no run dir for seed${SEED}"
    exit 4
fi
echo "[regumbel ${SEED}] run_dir=${RUN_DIR}"

python -m baseline.src.run_eval_only \
    --run-dir "${RUN_DIR}" \
    --num-samples 5000 \
    --strategies vanilla,top_prob_margin \
    --num-steps 50 \
    --noise gumbel \
    --noise-scale 0.5 \
    --eval-test-seed 99999 \
    --output-name eval_results_gumbel.json

echo "[regumbel ${SEED}] DONE"
