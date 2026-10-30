#!/usr/bin/env bash
#SBATCH --job-name=hca-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Slurm executes a spool copy; resolve the checkout from the submission directory.
if [[ -n "${CELLLUCID_ROOT:-}" ]]; then
  ROOT_DIR="${CELLLUCID_ROOT}"
elif [[ -f "${SCRIPT_DIR}/../data/download_hca.py" ]]; then
  ROOT_DIR="${SCRIPT_DIR}/.."
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "Cannot locate CellLucid. Submit from the repository or set CELLLUCID_ROOT." >&2
  exit 1
fi
if [[ ! -f "${ROOT_DIR}/data/download_hca.py" ]]; then
  echo "Invalid CellLucid repository: ${ROOT_DIR}" >&2
  exit 1
fi
cd "${ROOT_DIR}"
ROOT_DIR="${PWD}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
HCA_CONFIG="${HCA_CONFIG:-${ROOT_DIR}/data/hca.yaml}"
exec "${PYTHON_BIN}" -u -m data.download_hca --config "${HCA_CONFIG}" "$@"
