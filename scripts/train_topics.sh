#!/usr/bin/env bash
#SBATCH --job-name=celllucid-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
# Train the topic model on a completed cache (single process, one GPU).
# Usage: bash scripts/train_topics.sh --cache /path/to/cache --output-dir /path/to/run [--resume]
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
exec "${PYTHON_BIN}" -u -m celllucid.trainer.topic --config "${ROOT_DIR}/configs/topic.yaml" "$@"
