#!/usr/bin/env bash
# Run HCA acquisition, count preparation, deduplication, cache building and topic pretraining in sequence.
# Usage: PYTHON_BIN=/path/to/python NPROC_PER_NODE=4 bash scripts/pretrain_atlas.sh /path/to/HCA /path/to/run
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
HCA_ROOT="$(realpath -m "$1")"
OUTPUT_DIR="$(realpath -m "$2")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
mkdir -p "${HCA_ROOT}" "${OUTPUT_DIR}"
trap 'printf "failed: line %s, exit %s\n" "$LINENO" "$?" > "${OUTPUT_DIR}/pipeline_status.txt"' ERR
printf 'download\n' > "${OUTPUT_DIR}/pipeline_status.txt"
bash data/download_hca.sh --output-dir "${HCA_ROOT}"
printf 'annotation\n' > "${OUTPUT_DIR}/pipeline_status.txt"
if [[ ! -f assets/gene_symbols.json ]]; then
  "${PYTHON_BIN}" -B -c 'from data.download import download_file; download_file("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz", "assets/gencode.v49.annotation.gtf.gz", max_bytes=200000000)'
  "${PYTHON_BIN}" -B data/build_gene_mapping.py --gtf assets/gencode.v49.annotation.gtf.gz --output assets/gene_symbols.json
fi
printf 'preprocess\n' > "${OUTPUT_DIR}/pipeline_status.txt"
bash data/prepare_topics.sh --manifest "${HCA_ROOT}/manifest.json" --output-dir "${HCA_ROOT}/pretraining"
printf 'deduplicate\n' > "${OUTPUT_DIR}/pipeline_status.txt"
if [[ ! -f runs/corpus_dedup/duplicates.json ]]; then
  mkdir -p runs/corpus_dedup
  "${PYTHON_BIN}" -m celllucid.data.dedup --manifest "${HCA_ROOT}/pretraining/manifest.json" \
    --output runs/corpus_dedup/duplicates.json
fi
printf 'cache\n' > "${OUTPUT_DIR}/pipeline_status.txt"
bash scripts/build_cache.sh --manifest "${HCA_ROOT}/pretraining/manifest.json" --output "${OUTPUT_DIR}/cache"
printf 'pretrain\n' > "${OUTPUT_DIR}/pipeline_status.txt"
RESUME_ARGS=()
if [[ -f "${OUTPUT_DIR}/checkpoints/latest/checkpoint.json" ]]; then
  RESUME_ARGS=(--resume)
fi
bash scripts/train_topics.sh --cache "${OUTPUT_DIR}/cache" --output-dir "${OUTPUT_DIR}" "${RESUME_ARGS[@]}"
printf 'export\n' > "${OUTPUT_DIR}/pipeline_status.txt"
if [[ ! -d "${OUTPUT_DIR}/topic_bank" ]]; then
  "${PYTHON_BIN}" -m celllucid.tasks.topic_bank --checkpoint "${OUTPUT_DIR}/checkpoints/latest" \
    --cache "${OUTPUT_DIR}/cache" --output "${OUTPUT_DIR}/topic_bank"
fi
printf 'promote\n' > "${OUTPUT_DIR}/pipeline_status.txt"
bash scripts/push_ckpt.sh --checkpoint-dir "${OUTPUT_DIR}/topic_bank" --assets-dir "${ROOT_DIR}/assets" --push-hf 0
printf 'completed\n' > "${OUTPUT_DIR}/pipeline_status.txt"
