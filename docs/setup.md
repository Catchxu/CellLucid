# CellLucid setup and small-data example

This implementation covers HCA acquisition, count preprocessing, a **frozen
TranscriptFormer** adapter, OT topic modeling, masked training, topic-bank export
and inference. Pathway mapping and the query agent are future independent modules.

## Environment

On Sunlab, the isolated project environment is `.venv/`. It inherits the existing
`py312` scientific stack; installation does not modify that shared environment.

```bash
/data1021/xukaichen/miniconda3/envs/py312/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install omegaconf==2.3.0 hydra-core==1.3.2
.venv/bin/python -m pip install --no-deps -e assets/transcriptformer_source
.venv/bin/python -m pip install --no-deps -e .
```

The upstream source is unmodified, revision
`c943a89a4de9511a8ab1010715bf1cfa8827cc99` (TranscriptFormer 0.6.1):
https://github.com/czi-ai/transcriptformer/tree/c943a89a4de9511a8ab1010715bf1cfa8827cc99

Obtain that revision in `assets/transcriptformer_source/` when setting up a fresh
clone. Its MIT license remains with the upstream source. CellLucid follows the
existing repository's GPL-3.0 license.

**Compatibility:** upstream pins PyTorch 2.5.1 and older scientific packages.
Our isolated Sunlab environment intentionally uses the installed PyTorch 2.11
stack for RTX 5090 support. Upstream dependency-pin conflicts are therefore
expected from `pip check`; do not downgrade the shared environment to remove them.
The adapter uses the upstream model, tokenizer, vocabulary construction,
projection and pooling without editing their implementation or learned weights.
See `docs/validation.md` for actual integration-test results and limits.

## Backbone assets

TF-Sapiens is the initial human-only backbone. Obtain its public artifact from:

https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz

Extract it under `assets/`, producing `assets/tf_sapiens/config.json`,
`model_weights.pt` and `vocabs/`. The archive is approximately 1.82 GB.
Backbone identities hash the actual configuration, weights and vocabulary files.
Training checkpoints never duplicate those frozen weights.

## HCA: inventory before download

```bash
.venv/bin/celllucid-download-hca --config configs/hca.yaml --inventory-only
bash scripts/download_hca.sh
```

The supplied configuration selects **one** public human H5AD count matrix under
10 MB. It does not download the full HCA collection. The selected file/version,
original project metadata, size and SHA256 are retained in
`data/hca_smoke/manifest.json`. Resume uses that frozen manifest rather than
silently choosing a new file. Use another output directory for another selection.

The current acquisition adapter can select H5AD or Loom files; the count
preprocessor initially accepts H5AD. Other formats need explicit conversion.
Project summaries do not establish per-cell donor identity. Metadata missing from
the matrix is retained as missing; project aggregates are not assigned to cells.
Checksums deduplicate files, not biological cells or alternate analyses of them.
Review project/library/contributor matrix overlap before building a large corpus.

## PBMC smoke workflow

The example input is `/data1021/xukaichen/data/CTA/pbmc.h5ad`.
The source file is opened read-only. The default configuration examines 256 cells;
feature extraction uses 64 of them, with two reproducible masked views.

```bash
bash scripts/preprocess.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_embeddings.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_topics.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_topics.sh
```

These commands target fresh output paths. The prepared Sunlab checkout already
contains these example outputs; choose new paths to repeat them, or use
`--resume` for topic training.

The example uses a 127-gene-token context to keep validation short. For research
training, configure the intended context length (up to the checkpoint maximum),
more cells, suitable masks and study/donor-held-out evaluation. These choices
change the recorded feature/bank identity; do not reuse incompatible caches.

The preprocessing stage preserves integer counts and gene IDs and never guesses
how to invert a normalized matrix. Select `X`, `raw`, or `layers/<name>` explicitly
when necessary. `auto` tries `layers/counts`, then `raw.X`, then `X`.
Missing genes in a dataset are excluded from its masked targets.

If an HCA matrix supplies symbols only, provide a curated JSON symbol-to-Ensembl
mapping using `gene_mapping` in the preprocessing configuration. Its checksum is
recorded. Unmapped identifiers are preserved; backbone vocabulary alignment
reports no overlap rather than manufacturing identifiers. Ambiguous duplicate
mapped IDs are rejected and require an explicit collapse policy upstream.

## Objective and freezing

Only topic prototypes T enter the optimizer. Cell-topic weights are direct cosine
softmax assignments. The gene dictionary B is row-normalized differentiable
entropy-regularized OT with fixed uniform topic and gene marginals.
There is one outer loss: cross-entropy between predicted and observed
proportions over hidden, available genes. Cells without positive hidden target
mass or sufficient visible expression are excluded during feature extraction.

Masking happens **before** upstream sorting, normalization, clipping and
tokenization. Training rejects full-input feature caches. Cached masked views
are fixed views, not freshly randomized masks each epoch; regenerate them or
increase `n_views` for another masking design.

Input gene embeddings are taken **after** the pretrained projection.
Cell embeddings follow upstream pooling. Equal dimensions do not guarantee
aligned input/output geometry or meaningful topics; those are empirical
assumptions for subsequent experiments.

## Outputs and extensions

Training writes logs, resolved configuration, split/cache/backbone manifests,
topic-only resume checkpoints and a final-state `topic_bank/`. A resumed run
exports a new epoch-specific bank if the first bank already exists. The best
validation checkpoint is recorded separately; the default exported bank is the
final state, not implicitly the best checkpoint.

The topic bank contains T, B, gene priors, ordered gene IDs and checksums.
Inference writes HDF5 cell IDs and topic usage; B is unchanged across datasets.
HCA metadata and inference records are extension points for future pathway
interpretation, schema indexing and agents.

Large files live in ignored `assets/`, `data/`, `cache/`, `runs/` and
`topic_banks/`. Keep code and small configurations in Git; publish the selected
data manifest with a future model release.

## Tests

```bash
.venv/bin/python -m pytest -q -m "not integration"
CELLLUCID_TF_CHECKPOINT=assets/tf_sapiens CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest -q -m integration
.venv/bin/python -m ruff check src tests
```

The ordinary suite uses tiny generated count matrices for numerical and
data-contract tests. The marked test loads the real frozen backbone.

## Reproduce the HCA symbol-mapping test

The included `hca_preprocess.yaml` and `hca_inference.yaml` describe the tested
HCA matrix. On a fresh set of output paths, prepare its explicit annotation:

```bash
.venv/bin/python scripts/build_gene_mapping.py \
  --annotation /data1021/xukaichen/data/CTA/pbmc.h5ad \
  --gene-id-column gene_ids --output data/resources/symbol_to_ensembl.json
.venv/bin/celllucid-preprocess --config configs/hca_preprocess.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/celllucid-infer --config configs/hca_inference.yaml
```

Use a versioned, comprehensive species-matched annotation for a research corpus;
this example lookup is limited to identifiers in the supplied annotation table.
