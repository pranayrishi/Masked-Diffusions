#!/usr/bin/env bash
#
# 01_smoke.sh — Bouchet smoke job: 1 GPU on the `devel` partition, 30 min walltime.
#
# Purpose: validate that everything that worked on the laptop also works on Bouchet
# (Conda env, GPU, data paths, checkpoint write to project_pi_..., JSONL log) before
# committing any GPU-hours to the 75-job production array.
#
# What it does:
#   1. Loads the project Conda env.
#   2. Runs the lightweight smoke (the same one we run locally; ~3 s on H200).
#   3. Runs ONE filtered training at production model size (19M MDM, RoPE) for
#      ~250 steps on (N, P) = (25, 275) — enough to confirm the GPU pipeline works
#      end-to-end without burning a long walltime slot.
#   4. Saves logs + a final checkpoint to ~/project_pi_<labname>/mdm_research/runs/smoke/.
#
# IMPORTANT — placeholders use ${ACCOUNT}, ${PARTITION_SMOKE}, ${GPU_TYPE},
# ${PROJECT_DIR}, ${SCRATCH_DIR}, ${CONDA_ENV}, ${EMAIL}. These are filled in by
# `slurm/_render_scripts.sh` from `cluster_config.local.yaml` (the .gitignored
# values file). DO NOT EDIT THE PLACEHOLDERS DIRECTLY in this file — it is the
# template, not the runtime script.

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

echo "==> 01/03 unit tests"
( cd baseline && python -m pytest tests/ -q ) || { echo "baseline tests failed"; exit 1; }
( cd entropy_filtered && python -m pytest tests/ -q ) || { echo "entropy_filtered tests failed"; exit 1; }

echo "==> 02/03 GPU sanity"
python - <<'PYEOF'
import torch
print(f"torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device 0 = {torch.cuda.get_device_name(0)}")
PYEOF

echo "==> 03/03 production-shape filtered run (250 steps, mode=band, N=25 P=275)"
python -m entropy_filtered.src.train_filtered \
    --config entropy_filtered/configs/lo_nae_sat_25_275_band.yaml \
    --override num_iterations=250 \
    --override output_dir="${OUTPUT}/band_250steps"

# Move final results into the BACKED-UP project area
PROJECT_RESULTS="${PROJECT_DIR}/results/smoke-${SLURM_JOB_ID}"
mkdir -p "${PROJECT_RESULTS}"
rsync -av "${OUTPUT}/" "${PROJECT_RESULTS}/"

# Optional: jobstats for right-sizing the production array
jobstats $SLURM_JOB_ID > "${PROJECT_RESULTS}/jobstats.txt" 2>&1 || true

echo "Smoke complete. Results: ${PROJECT_RESULTS}"
