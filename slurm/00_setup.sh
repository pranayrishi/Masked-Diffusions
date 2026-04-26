#!/usr/bin/env bash
#
# 00_setup.sh — environment setup on Bouchet using the YCRC PyTorch module.
#
# Strategy: instead of `conda env + pip install torch==2.1.2` (which would download
# ~2 GB at compute-node bandwidth = 30+ min), we load the YCRC-provided
# PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 module — same exact torch version, already
# compiled, instantly available. We then `pip install --user` the three small
# Python-only deps we still need (pyyaml, tqdm, pytest), which is a few hundred KB.
#
# Verified 2026-04-26 that numpy 1.24.2 (bundled with the YCRC module) reproduces
# our seed-42 test ground truth identically to numpy 1.26.3 on the laptop.
#
# Re-runnable: pip install --user is idempotent.
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
module purge
module load ${PYTORCH_MODULE}

cd ${PROJECT_DIR}

# Verify the module gives us what we need
python - <<'PYEOF'
import sys, torch, numpy
print(f"python   = {sys.version.split()[0]}")
print(f"torch    = {torch.__version__}  cuda_compiled={torch.version.cuda}")
print(f"numpy    = {numpy.__version__}")
PYEOF

# Verified what's already in the YCRC PyTorch module (sys.path entries):
#   torch 2.1.2, numpy 1.24.2 (via SciPy-bundle), pyyaml 6.0, pytest 7.2.0,
#   networkx, sympy, expecttest, Pillow, protobuf, ...
# Only `tqdm` is missing — installing into the per-user site, which is shared
# across all allocations and does not need to be reinstalled.
#
# CRITICAL: do NOT set PYTHONPATH explicitly — the YCRC module manages sys.path
# dynamically. Setting PYTHONPATH from scratch clobbers the module's torch /
# numpy / pyyaml entries (verified 2026-04-26 in setup_9530796.out). Instead,
# we rely on `cd $SLURM_SUBMIT_DIR` putting the project root on sys.path[0]
# via Python's -m flag and on the conftest.py self-location for tests.
echo "==> installing tqdm (only missing dep) into user site"
pip install --user tqdm 2>&1 | tail -5

# Run the test suites on this CPU-only allocation. Both packages must pass.
echo "==> baseline tests"
( cd baseline && python -m pytest tests/ -v )

echo "==> entropy_filtered tests"
( cd entropy_filtered && python -m pytest tests/ -v )

echo "Environment setup complete"
