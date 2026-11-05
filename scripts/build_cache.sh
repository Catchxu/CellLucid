#!/usr/bin/env bash
#SBATCH --job-name=celllucid-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
# Encode the training cache (every eligible cell once) with one frozen backbone replica per GPU.
# Usage: NPROC_PER_NODE=4 bash scripts/build_cache.sh --manifest /path/to/atlas/manifest.json --output /path/to/cache
# An interrupted build resumes when rerun with the same arguments.
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Single-node workers use loopback independently of hostname resolution.
exec "${PYTHON_BIN}" -u -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
  --nproc_per_node="${NPROC_PER_NODE}" -m celllucid.data.cache \
  --config "${ROOT_DIR}/configs/topic.yaml" "$@"
