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

HCA acquisition uses one configuration, `configs/hca.yaml`, and one script,
`scripts/download_hca.sh`. Run from the repository root with Python >=3.11 and
the package dependencies available. The script uses `.venv/bin/python` if it
exists, otherwise `python3` from the active environment; `PYTHON_BIN` can override
that interpreter. It imports the checkout through `PYTHONPATH`.

Inspect the source inventory without downloading matrix contents, then download:

```bash
bash scripts/download_hca.sh --inventory-only
bash scripts/download_hca.sh
```

The same script contains Slurm resource directives. They are comments during
ordinary Bash execution. To submit a CPU download job, supply your cluster's
account and partition on the command line:

```bash
sbatch --account=<account> --partition=<cpu_partition> scripts/download_hca.sh
```

Default Slurm resources are one node, four CPUs, 16 GB RAM and 48 hours.
Scheduler options can override these defaults. Logs use Slurm's normal
`slurm-<job_id>.out` destination unless `--output`/`--error` are supplied to
`sbatch`; any chosen log directory must exist before submission. Slurm's spool
copy locates the repository through `SLURM_SUBMIT_DIR`, so submit from the
repository root or export `CELLLUCID_ROOT` with its absolute path.

The configuration selects public, human-associated H5AD matrices across all
source categories in HCA catalog `dcp60`, requiring SHA256. Default limits are
10,000 files, 100 GB per file and 1 TB total; these are ceilings, not the expected
size. Four concurrent transfers are used, with a preflight check requiring the
remaining bytes plus 100 GB free. Shared filesystem usage can change afterwards.
This is an acquisition set; assess biological overlap, modality, species and raw
count availability before selecting training data. Human-associated source files
can also contain other species. A matrix label does not establish raw counts in X.

The default directory is `data/HCA`. Override it without a separate config:

```bash
bash scripts/download_hca.sh --output-dir /path/to/data/HCA --inventory-only
sbatch --account=<account> --partition=<cpu_partition> \
  scripts/download_hca.sh --output-dir /path/to/data/HCA
```

The manifest records file UUID/version, source metadata, bytes, SHA256,
API/catalog and discovery stopping condition. Resume uses this frozen inventory;
rerun the same command after interruption. Completed files are verified locally,
partial files resume where HTTP Range is supported, and failed files are recorded
without preventing unrelated transfers. Any remaining failure gives a nonzero
exit status. A process lock prevents concurrent jobs from writing the same
collection. Changing selection or catalog requires a fresh output directory.

For a small smoke example using the same configuration and script:

```bash
bash scripts/download_hca.sh --output-dir data/hca_smoke \
  --max-files 1 --max-file-bytes 10000000 --max-total-bytes 10000000
```

The acquisition adapter also supports Loom discovery; preprocessing currently
accepts H5AD. Other formats need explicit conversion. Project summaries do not
establish per-cell donor identity. Metadata missing from a matrix stays missing;
project aggregates are not assigned to cells. File identity does not deduplicate
biological cells or alternate analyses. Preserve source counts and review raw
layers, mixed species, spatial assays and study overlap before training.

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
HCA matrix. First download the small HCA example above to a fresh output path
and confirm that the selected source matches the preprocessing input path.
Prepare its explicit annotation:

```bash
.venv/bin/python scripts/build_gene_mapping.py \
  --annotation /data1021/xukaichen/data/CTA/pbmc.h5ad \
  --gene-id-column gene_ids --output data/resources/symbol_to_ensembl.json
.venv/bin/celllucid-preprocess --config configs/hca_preprocess.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/celllucid-infer --config configs/hca_inference.yaml
```

Use a versioned, comprehensive species-matched annotation for a research corpus;
this example lookup is limited to identifiers in the supplied annotation table.
