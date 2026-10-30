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
submit the same script through Slurm as described in the guide. The same guide
describes explicit cohort manifests and `data/prepare_topics.sh` for preparing
reviewed source matrices with study labels.