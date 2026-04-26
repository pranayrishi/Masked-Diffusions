#!/usr/bin/env bash
#
# 00_setup.sh — one-time Conda env setup on Bouchet.
#
# Runs as a devel-partition batch job (NOT on the login node) to avoid
# congesting shared login resources. Creates `mdm` env, installs project
# dependencies, prints torch version on completion.
#
# Re-runnable: `conda create` will fail fast if the env exists; we tolerate
# that and proceed to (re)install project dependencies.
#
# Renders from this template via slurm/_render_scripts.sh.

#SBATCH --job-name=mdm-setup
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=devel
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=${PROJECT_DIR}/setup_%j.out
#SBATCH --error=${PROJECT_DIR}/setup_%j.err

set -euo pipefail
module reset
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh

# Idempotent env creation
if conda env list | awk '{print $1}' | grep -q "^${CONDA_ENV}$"; then
    echo "[setup] conda env '${CONDA_ENV}' already exists, reusing"
else
    echo "[setup] creating conda env '${CONDA_ENV}' (python=3.10)"
    conda create -n ${CONDA_ENV} python=3.10 -y
fi
conda activate ${CONDA_ENV}

# Install project + test extras (pinned in baseline/pyproject.toml)
cd ${PROJECT_DIR}/baseline
pip install --upgrade pip
pip install -e ".[test]"

# Sanity import
python - <<'PYEOF'
import torch
print(f"mdm env OK  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
PYEOF

# Run the unit-test suites once on this allocation to make sure the env
# matches the codebase. baseline tests must pass before we trust GPU jobs.
cd ${PROJECT_DIR}
( cd baseline && python -m pytest tests/ -q )
( cd entropy_filtered && python -m pytest tests/ -q )

echo "Environment setup complete"
