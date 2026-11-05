# CellLucid pretraining and development

CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.

The current implementation pretrains a shared topic model across an HCA atlas in two
stages. Every eligible cell is masked and encoded once into a cache of frozen TranscriptFormer
embeddings; the topic model is then trained in one pass over that cache.
Inference on new datasets encodes unmasked counts directly and needs no cache. Pathway
interpretation and the query agent will be added later.

## Install

Use Python >=3.11 with a CUDA-enabled PyTorch build for your hardware. Install
CellLucid and its frozen backbone dependency in the active environment:

```bash
python -m pip install -e ".[backbone]"
mkdir -p assets
git clone https://github.com/czi-ai/transcriptformer.git assets/transcriptformer_source
git -C assets/transcriptformer_source checkout c943a89a4de9511a8ab1010715bf1cfa8827cc99
python -m pip install --no-deps -e assets/transcriptformer_source
```

The pinned upstream version is 0.6.1. It declares older scientific-package pins;
the `backbone` extra installs the Lightning, Scanpy and Hydra runtime dependencies
needed by the adapter without imposing the upstream's older PyTorch pins. Core
CellLucid requirements provide OmegaConf and the count-matrix libraries. Install
the source with `--no-deps` as shown above. Current validation uses PyTorch 2.11
and AnnData 0.12. Upstream
source and backbone weights are not modified.

Download [TF-Sapiens](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz)
and extract under `assets/tf_sapiens/`, containing `config.json`,
`model_weights.pt` and `vocabs/`. Obtain the pinned gene annotation and mapping
as described in [the data guide](data.md).

## Repository

| Path | Purpose |
| --- | --- |
| `data/` | HCA acquisition, annotation mapping and full-inventory corpus preparation |
| `configs/preprocess.yaml` | Count-source policy, QC, sharding and study split |
| `configs/topic.yaml` | Frozen backbone, cache, topic model, training and evaluation settings |
| `scripts/build_cache.sh` | Multi-GPU, resumable encoding of the training cache |
| `scripts/train_topics.sh` | Single-GPU topic training on a completed cache |
| `src/celllucid/data/dedup.py` | Cells stored in more than one source file of a study |
| `src/celllucid/data/cache.py` | Cache plan, build and in-memory reader |
| `src/celllucid/models/topic.py` | Topic model on frozen cells with an OT topic-gene dictionary |
| `src/celllucid/trainer/topic.py` | Training loop, monitoring, evaluation and checkpoints |
| `src/celllucid/evaluation.py` | Fixed evaluation pools and their metrics |
| `tests/` | Numerical and pipeline tests |
| `docs/` | Flat data, pretraining and future tutorial/figure documentation |

Large assets, datasets and runs are ignored by Git. Cluster account, partition
and data paths are supplied at launch rather than embedded in project files.

## Prepare the atlas

Use the complete acquisition manifest, not a hand-selected small cohort:

```bash
bash data/download_hca.sh --output-dir /path/to/HCA
bash data/prepare_topics.sh --manifest /path/to/HCA/manifest.json \
  --output-dir /path/to/atlas
```

Preparation examines every inventory record, validates full count matrices,
normalizes gene identifiers, sums duplicate gene features, filters cells using
explicit QC and writes sparse row shards. Each file receives a coverage record,
including exclusion reasons. An unavailable raw count matrix is reported rather
than reconstructed from normalized expression. Acquisition coverage is separate
from the HCA portal's headline cell count.

Exact duplicate cells with the same study/donor/cell identity, aligned counts and
measurement availability are removed. This does not claim complete biological
deduplication when independent submissions rename cells or change counts.
Study identities keep related source files in the same training/validation split.
The source index supports restart after completed files; partial work for the
interrupted file is rebuilt. Preserve the original download manifest and matrices.

## Pretrain

Pretraining runs in three steps after count preparation:

```bash
python -m celllucid.data.dedup --manifest /path/to/atlas/manifest.json \
  --output runs/corpus_dedup/duplicates.json
NPROC_PER_NODE=4 bash scripts/build_cache.sh --manifest /path/to/atlas/manifest.json \
  --output /path/to/run/cache
bash scripts/train_topics.sh --cache /path/to/run/cache --output-dir /path/to/run
```

For a workstation, `NPROC_PER_NODE=4 bash scripts/pretrain_atlas.sh /path/to/HCA /path/to/run`
runs acquisition, preprocessing, deduplication, cache building, training and export in
sequence. Each step resumes: completed downloads, preprocessing sources, the duplicate list,
finished cache parts and the latest training checkpoint are reused. Progress is recorded in
`<run>/pipeline_status.txt`.

### Training cache

The cache holds every eligible cell of the training split exactly once, masked with view 2.
Eligible cells exclude dropped duplicates, the evaluation-pool cells and every duplicate copy
of an evaluation cell. Studies are neither upsampled nor downsampled, so each enters training in
proportion to its size. A part stores FP16 embeddings, the hidden-gene set as packed bits and
the nonzero hidden counts; training holds the whole cache in memory (about 10 KB per cell).
`identity.json` records the corpus, backbone, duplicate list, evaluation cells and plan;
training and export check it.

The cache also holds a study-balanced initialization sample (mask view 1, plus an unmasked
encoding of the same cells). It gives the k-means topics, the training mean of masked
embeddings, the inference mean of unmasked embeddings and the empirical gene marginal.

The backbone compiles flex attention and the whole model (`compile_attention`,
`compile_model`). Inputs are padded to a fixed sequence length and the last microbatch is
padded to `encoder_batch_size`, so shapes stay static. On one GPU this encodes about 106
cells per second against 11 with eager attention, with embeddings within cosine 0.99999.

### Model and objective

Cells keep their frozen geometry: an embedding minus the training mean is scored against unit
topic vectors by cosine, and a softmax at temperature 0.1 gives the assignment. There is no
cell projection. A gene tower maps the frozen gene vectors through an orthogonal linear map
(identity at initialization), which preserves every cosine between genes. Entropic OT
(epsilon 0.03, 200 iterations) with a uniform topic marginal and the empirical gene marginal
turns cos(topic, mapped gene) into the dictionary. The topic-averaged dictionary therefore
equals the gene marginal, so balanced topic usage reproduces the population expression
profile.

The only loss is masked compositional CE on hidden genes. The OT plan is a constrained
normalization of the cosines, not an objective. Row-normalized, it is a softmax over genes of
cos(topic, mapped gene) / epsilon plus a gene bias shared by all topics, which Sinkhorn sets so
that the topics average to the gene marginal. CE reaches the topics through both the
assignment and the dictionary, and reaches the gene map through the unrolled Sinkhorn
iterations. One iteration is compiled (`compile_sinkhorn`).

The OT value is deliberately not minimized. The dictionary depends only on how a topic's
cosines differ across genes, while the OT value rewards their absolute level, so it pulls
topics out of the cell space toward their genes. In a 1000-step pilot with weight 3, two thirds
of each topic's norm left the cell subspace, assignments flattened (held-out cell-topic MI
0.35 against 0.81 without it), and held-out CE gain and pair-ranking AUC were lower.

Training uses one device and is one pass over the cache: the cells are read in a fixed random
permutation (seeded by `training.seed`) in batches of `training.batch_size`, so the number of
steps is the cache size divided by the batch size, and the last partial batch is dropped.
Because the permutation is global, a batch's study composition differs from the corpus
proportions only by sampling noise, and no study arrives in long runs. Checkpoints in
`<run>/checkpoints/latest` hold weights, optimizer and RNG state and the step; `--resume`
continues from them and reproduces an uninterrupted run. `--max-steps` stops early without
changing the schedule.

### Monitoring

`<run>/metrics.jsonl` records, on training batches: CE, its gain over predicting the gene
marginal for every cell, the mean cell effective number of topics, the mean cosine between a
cell and its best topic, and the largest topic mass. Falling cell-topic cosine and rising
cell effective topics indicate topics drifting away from cells and flattening assignments.
Every `eval_every_steps`, the fixed evaluation pools (`evaluation.pools`) are scored:
held-out CE gain over the gene marginal, within- and cross-study pair-ranking AUC and cell-topic
mutual information, with full per-pool results and the unmasked inference readout.

The pools are a fixed local artifact (`runs/eval_pools`, described by its README), not built by
this repository. The cache excludes their cells (`cache.evaluation_pools`) so they never enter
training. Set both `cache.evaluation_pools` and `evaluation.pools` to `null` to train without
them.

## Promote and publish a completed bank

The standard local run directory is `runs/hca_pretraining`. Its checkpoints and
logs remain there. After training completes, the pipeline exports the latest checkpoint and promotes the
topic bank to `assets/topic_bank` automatically. Promotion validates the
package and copies only `topics.safetensors`, `genes.json` and `manifest.json`.
A repeated promotion replaces those release files without copying training state.

```text
runs/hca_pretraining/
  cache/                      # Training cache and initialization sample
  checkpoints/latest/
  topic_bank/                 # Exported from the final checkpoint
assets/
  tf_sapiens/                 # Separate frozen backbone
  topic_bank/                 # User-facing topic bank
```

This follows scCAFM's checkpoint-package -> local-assets -> explicit Hub upload
workflow. To promote an already completed bank manually:

```bash
bash scripts/push_ckpt.sh --push-hf 0
```

Create the dedicated Hugging Face Space later and authenticate with `hf auth login`.
Then explicitly upload the bank (replace the placeholder repository ID):

```bash
bash scripts/push_ckpt.sh --push-hf 1 \
  --hf-repo-id your-org/CellLucid --hf-repo-type space
```

The Space must already exist. No repository is created and no upload is triggered
by training. For a model repository, use `--hf-repo-type model`. `CKPT_DIR`,
`ASSETS_DIR`, `HF_REPO_ID` and `HF_REPO_TYPE` can also be supplied as environment
variables. `CKPT_DIR` refers to the exported bank package, defaulting to
`runs/hca_pretraining/topic_bank`; `ASSETS_DIR` defaults to `assets`.

Uploads place only the three bank files under `topic_bank/` in the remote
repository. They exclude backbone weights, annotations, source code, optimizer
state and logs. The matching `TopicBank.from_pretrained` API or CLI
`--hub-repo-id` downloads the same subfolder into local assets. Pin
`revision` / `--hub-revision` to a tag or commit for reproducible inference.

## Inference and tests

For new datasets, preprocess raw counts, then apply an exported bank:

```bash
bash scripts/preprocess.sh --input /path/to/input.h5ad --output /path/to/counts.h5ad \
  --count-source layers/counts --gene-key feature_id
bash scripts/infer_topics.sh --input /path/to/counts.h5ad \
  --topic-bank assets/topic_bank --output /path/to/usage.h5
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q -m 'not integration'
```

Remove `runs/pytest-tmp` and other disposable test outputs after testing while
preserving scientific results. The marked integration test requires real backbone
assets and an accelerator. Full GPU throughput should be benchmarked for the
chosen context, precision and batch sizes before a long pretraining run.

Checkpoints use format 5 and topic banks format 4. A bank stores normalized topics, the
dictionary, the gene marginal, both cell means and the gene map; `TopicBank.transform` centers
unmasked embeddings by the inference mean. Earlier checkpoints and banks cannot be loaded.
