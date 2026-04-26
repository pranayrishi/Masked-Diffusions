#!/usr/bin/env bash
#
# _render_scripts.sh — render Slurm script templates with values from
# cluster_config.local.yaml (gitignored). Produces ready-to-sbatch scripts in
# slurm/_rendered/ which are also gitignored.
#
# Usage:
#   bash slurm/_render_scripts.sh
# This renders every *.sh template in slurm/ (except this one) into slurm/_rendered/.
#
# This script DOES NOT run sbatch. The user must explicitly submit:
#   sbatch slurm/_rendered/01_smoke.sh
# (And the project rule: no sbatch without the user's explicit "submit" instruction.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CFG="${SCRIPT_DIR}/cluster_config.local.yaml"

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: ${CFG} not found."
    echo "Copy slurm/cluster_config.example.yaml to slurm/cluster_config.local.yaml and fill it in."
    exit 1
fi

# Read YAML values via Python (yaml is a hard dependency of the project)
read_yaml() {
    python3 -c "import yaml,sys; d=yaml.safe_load(open('$CFG')); print(d['$1'])"
}

ACCOUNT=$(read_yaml account)
PARTITION_SMOKE=$(read_yaml partition_smoke)
PARTITION_PRODUCTION=$(read_yaml partition_production)
GPU_TYPE=$(read_yaml gpu_type)
PROJECT_DIR=$(read_yaml project_dir)
SCRATCH_DIR=$(read_yaml scratch_dir)
CONDA_ENV=$(read_yaml conda_env)
EMAIL=$(read_yaml email)

OUT_DIR="${SCRIPT_DIR}/_rendered"
mkdir -p "${OUT_DIR}"

for tpl in "${SCRIPT_DIR}"/*.sh; do
    name=$(basename "$tpl")
    case "$name" in
        _render_scripts.sh) continue ;;
    esac
    out="${OUT_DIR}/${name}"
    sed -e "s|\${ACCOUNT}|${ACCOUNT}|g" \
        -e "s|\${PARTITION_SMOKE}|${PARTITION_SMOKE}|g" \
        -e "s|\${PARTITION_PRODUCTION}|${PARTITION_PRODUCTION}|g" \
        -e "s|\${GPU_TYPE}|${GPU_TYPE}|g" \
        -e "s|\${PROJECT_DIR}|${PROJECT_DIR}|g" \
        -e "s|\${SCRATCH_DIR}|${SCRATCH_DIR}|g" \
        -e "s|\${CONDA_ENV}|${CONDA_ENV}|g" \
        -e "s|\${EMAIL}|${EMAIL}|g" \
        "$tpl" > "$out"
    chmod +x "$out"
    echo "rendered ${name} -> ${out}"
done

echo
echo "Rendered scripts are in slurm/_rendered/."
echo "DO NOT sbatch yet — user must explicitly approve before submission."
