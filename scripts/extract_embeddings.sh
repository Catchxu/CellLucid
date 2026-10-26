#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
exec "${PYTHON_BIN}" -m celllucid.tasks.embeddings --config "${ROOT_DIR}/configs/embeddings.yaml" "$@"
