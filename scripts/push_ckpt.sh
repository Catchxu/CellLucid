#!/usr/bin/env bash
# Promote the completed topic bank; upload only when --push-hf 1 is explicit.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u -m celllucid.tasks.publish "$@"
