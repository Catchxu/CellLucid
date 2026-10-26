#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
exec "${PYTHON_BIN}" -m celllucid.tasks.download_hca --config "${ROOT_DIR}/configs/hca.yaml" "$@"
