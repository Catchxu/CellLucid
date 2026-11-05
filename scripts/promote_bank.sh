#!/usr/bin/env bash
# Copy an exported topic bank into assets/topic_banks/<name>.
# Usage: bash scripts/promote_bank.sh --bank /path/to/run/topic_bank --name default
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u -m celllucid.tasks.publish "$@"
