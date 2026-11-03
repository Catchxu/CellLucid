#!/usr/bin/env bash
#SBATCH --job-name=celllucid-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ROOT_DIR="${CELLLUCID_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# Single-node workers use loopback independently of hostname resolution.
exec "${PYTHON_BIN}" -u -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
  --nproc_per_node="${NPROC_PER_NODE}" -m celllucid.trainer.topic \
  --config "${ROOT_DIR}/configs/topic.yaml" "$@"
