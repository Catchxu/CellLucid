# CellLucid

CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.

The current implementation covers HCA acquisition, count preprocessing, a frozen
TranscriptFormer backbone and topic modeling. Pathway mapping and the query agent
will be added later.

## Installation and backbone

Use Python >=3.11 with a PyTorch build appropriate for your accelerator. From the
repository root, install CellLucid into your active environment:

```bash
python -m pip install -e .
```

Install the unmodified TranscriptFormer source at revision
`c943a89a4de9511a8ab1010715bf1cfa8827cc99` (package version 0.6.1):

```bash
mkdir -p assets
git clone https://github.com/czi-ai/transcriptformer.git assets/transcriptformer_source
git -C assets/transcriptformer_source checkout c943a89a4de9511a8ab1010715bf1cfa8827cc99
python -m pip install -e assets/transcriptformer_source
```

Upstream pins PyTorch 2.5.1 and older scientific packages. If retaining a newer
accelerator-compatible environment, install its required dependencies explicitly
and use `--no-deps` for the upstream installation; validate that environment before
training. The previous integration run used PyTorch 2.11 with unmodified upstream
source. This does not establish compatibility with every environment.

Download [TF-Sapiens](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz)
and extract it under `assets/tf_sapiens/`, containing `config.json`,
`model_weights.pt` and `vocabs/`.

All backbone parameters stay frozen. Only topic embeddings are trained.

## Repository layout

- `data/`: HCA acquisition code, `hca.yaml`, the shell entrypoint and download guide.
- `configs/preprocess.yaml`: count source, gene identifiers and preprocessing settings.
- `configs/topic.yaml`: shared backbone, features, topic model, training and inference settings.
- `scripts/`: modeling command entrypoints and the gene-mapping utility.
- `src/celllucid/`: implementation; `tests/`: existing automated tests.
- `docs/`: future figures and tutorials, stored directly without subfolders.

Dataset paths, output destinations and training duration are explicit command
arguments. Downloaded assets, datasets, caches and experiment outputs stay outside
Git. No private server paths are required.

## Download HCA

Dataset acquisition utilities and their usage guide are in [`data/`](data/README.md).
Use `bash data/download_hca.sh --output-dir /path/to/HCA` for direct execution or
submit the same script through Slurm as described in the guide.

## Prepare counts and extract features

Select a raw count source and the column containing gene identifiers for each
input. Override the preprocessing configuration where the source differs:

```bash
bash scripts/preprocess.sh \
  --input /path/to/source.h5ad --output /path/to/processed.h5ad \
  --count-source layers/counts --gene-key feature_id
bash scripts/extract_embeddings.sh \
  --input /path/to/processed.h5ad --output /path/to/masked_features.h5
```

Count sources are `X`, `raw` or `layers/<name>`. Input must contain finite,
nonnegative integer counts; normalized values are not inverted or treated as
counts. If needed, pass `--gene-mapping /path/to/symbol_to_ensembl.json`, using a
versioned annotation appropriate to the species. `scripts/build_gene_mapping.py`
can build an explicit lookup from an annotated H5AD var table.

Standardized output stores counts in `X`, identifiers in `var.feature_id`, symbols
in `var.gene_symbol`, original cell metadata in `obs` and provenance in
`uns.celllucid`. Missing gene measurements are distinguished from observed zeros.

Extraction uses `backbone` and `features` from `topic.yaml`; inference uses the
same backbone section. `max_length: null` uses the checkpoint's sequence length.
Review context length, precision, batch sizes and attention compilation for your
hardware before extraction; full-context performance has not yet been validated.
Masking happens before backbone tokenization. Cached views retain source, mask,
gene order and backbone identities. Training requires masked feature caches.

## Train and infer

Edit `topic.yaml` for the experiment. Its model/optimizer values are starting
settings, not established atlas-scale hyperparameters. The default split groups
by `obs.study_id`: supply consistent study labels across files from the same
study, with at least two studies. For a different experimental design, explicitly
set the group or split column in `training.split`. Project aggregates are not
fabricated as donor or cell annotations.

```bash
bash scripts/train_topics.sh \
  --caches /path/to/study_a.h5 /path/to/study_b.h5 \
  --output-dir /path/to/run --epochs 50
bash scripts/infer_topics.sh \
  --input /path/to/new_processed.h5ad \
  --topic-bank /path/to/run/topic_bank --output /path/to/topic_usage.h5
```

The epoch count above is illustrative; choose it for the experiment. Stage scripts
also accept `--config` for an explicit experiment configuration. Preprocessing,
extraction and inference process all cells by default; `--max-cells` limits a
small trial without adding a separate smoke configuration.

Resume training with the same inputs and output directory, adding
`--resume /path/to/run/checkpoints/latest` and a larger total `--epochs`.
The run saves its resolved training settings, split/cache/backbone identities,
metrics, topic checkpoints and exported topic bank. Existing output files are
protected from accidental overwriting.

Cell-topic mixtures use direct cosine softmax assignment. The shared topic-gene
dictionary uses entropy-regularized OT with fixed marginals. The training loss
compares predicted and observed proportions on hidden, available genes. Exported
banks contain fixed topics, the dictionary, ordered genes and provenance;
inference writes cell IDs and topic usage while retaining the bank identity.

## Tests

Use the existing tests; remove disposable outputs after validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q -m 'not integration'
```

The marked integration test additionally requires real TranscriptFormer assets
through `CELLLUCID_TF_CHECKPOINT` and a suitable accelerator. Pytest temporary
outputs are under `runs/pytest-tmp`; clean that directory after testing while
preserving scientific run outputs.
