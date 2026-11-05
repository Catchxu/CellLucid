#!/usr/bin/env bash
# Upload the release assets (TF-Sapiens weights, topic banks, gene symbols) and the model card
# to the Hugging Face Hub. Needs `hf auth login` and network access to huggingface.co.
# Usage: bash scripts/push_assets.sh [repo-id]   (default: kaichenxu/CellLucid)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${1:-kaichenxu/CellLucid}"
hf upload "${REPO_ID}" "${ROOT_DIR}/assets" . --repo-type model \
  --include "tf_sapiens/*" --include "topic_banks/*" --include "gene_symbols.json*" \
  --commit-message "Upload CellLucid assets"
hf upload "${REPO_ID}" "${ROOT_DIR}/docs/model_card.md" README.md --repo-type model \
  --commit-message "Update model card"
