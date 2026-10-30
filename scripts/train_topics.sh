#!/usr/bin/env bash
#SBATCH --job-name=topic-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ROOT_DIR="${CELLLUCID_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "${PYTHON_BIN}" -u -m celllucid.trainer.topic --config "${ROOT_DIR}/configs/topic.yaml" "$@"
