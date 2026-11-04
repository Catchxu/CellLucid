# CellLucid pretraining and development

CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.

The current implementation pretrains a shared topic model across an HCA atlas.
Each training batch contains raw counts: fresh masking, a frozen TranscriptFormer
forward, and a topic update happen in one loop. No offline embedding cache is
required. Pathway interpretation and the query agent will be added later.

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
| `configs/topic.yaml` | Frozen backbone, topic model and online pretraining settings |
| `scripts/train_topics.sh` | Single-process or multi-GPU pretraining entrypoint |
| `src/celllucid/data/atlas.py` | Streaming raw-count batches partitioned across ranks/workers |
| `src/celllucid/trainer/topic.py` | Frozen forwards and topic updates in one training loop |
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

Activate the environment before submission. After count preparation, submit one
GPU job for the complete online training workflow. The training and preparation
scripts request email on completion or failure; supply the recipient at submission:

```bash
sbatch --account=<account> --partition=<gpu_partition> --gpus=8 \
  --mail-user=<email> \
  scripts/train_topics.sh --manifest /path/to/atlas/manifest.json \
  --output-dir /path/to/run --epochs 3 \
  --batch-size 128 --encoder-batch-size 16
```

For a workstation, run acquisition, preprocessing and pretraining sequentially:

```bash
NPROC_PER_NODE=4 bash scripts/pretrain_atlas.sh /path/to/HCA /path/to/run
```

Set `PYTHON_BIN` to the active environment's interpreter when needed. The pipeline
uses three epochs and the settings in `configs/topic.yaml`. It resumes downloads,
completed preprocessing sources and an existing latest training checkpoint.
Progress is recorded in `<run>/pipeline_status.txt`; keep the process running in a
persistent session or launch it detached with its output redirected to a log.

The epoch count is an explicit experiment choice. `--batch-size` is the number
of cells per optimizer update per rank; `--encoder-batch-size` controls the frozen
forward microbatch within that update. Values are used as supplied without
memory-based reduction. `NPROC_PER_NODE` defaults to the number of allocated
Slurm GPUs; it can be set explicitly for direct `torchrun` use. Single-node
multi-GPU training uses DDP for topic and projection parameters and one frozen backbone
replica per GPU. Submit from the repository root or set `CELLLUCID_ROOT`.

All training shards contribute to one shared topic bank. Shards and cell rows
are shuffled each epoch without building an atlas-wide cell index in RAM. Masks
are fresh per cell and epoch; validation uses fixed masks on held-out studies.
Measured zero expression and unavailable genes remain distinct. Distributed
training repeats initial batches only to fill its final synchronized step; all
validation cells are evaluated once globally.

Trainable parameters are topic embeddings and a residual cell projection
(2048 -> 512 -> 2048 with GELU). Its final layer starts at zero, making the
projection initially the identity. Cell-topic assignments use cosine softmax on
projected cells. Gene vectors and TranscriptFormer remain frozen.

The objective is `masked_ce + ot_weight * ot_clustering`. The second term is the
regularized OT value `<P,C> + epsilon * KL(P || a*b)` with cosine cost, uniform
topic marginal and the fixed gene marginal. The same transport plan defines the
shared gene dictionary. Gradients from reconstruction reach both the topic bank
and cell projection; OT clustering updates the topic bank. DDP averages the
shared OT term once and weights reconstruction by each rank's usable cell count.

`configs/topic.yaml` sets 128 topics, residual hidden dimension 512, OT weight
10.0, epsilon 0.03 and 200 Sinkhorn iterations for the revised experiment. These
are starting settings, not a guarantee against collapse. Check marginal errors,
held-out reconstruction and diversity during short validation before a long run.

Checkpoints save topic/projection/optimizer state and the next batch cursor, not the frozen
backbone. Resume with the same corpus/settings/world size and
`--resume /path/to/run/checkpoints/latest`. `--max-steps` sets an explicit total
optimizer-step limit. After all epochs finish, the best validation-CE checkpoint is exported to
`<run>/topic_bank/`. Explicit `--max-steps` stops save resumable checkpoints
without creating a release bank. Completed banks contain
topics, the residual projection, a fixed gene dictionary, ordered genes and provenance. No cell embeddings
are written during pretraining.

Training records reconstruction loss and detached topic diagnostics in
`<run>/metrics.jsonl` at initialization, after the first forward, every logging
interval and at epoch boundaries. Forward diagnostics are labeled `before_update`;
checkpoints contain the updated weights. Cell-usage statistics are summed across
all DDP ranks and cover the cells in the logging window or epoch segment since
resume. Existing running processes require restart to load the instrumentation.

- `topic_cosine_mean`, `topic_cosine_p95` and
  `topic_duplicate_pair_fraction_099` measure redundancy of normalized topic vectors.
- `topic_effective_dimension` is the exponential entropy of the normalized
  eigenvalues of the topic Gram matrix; identical directions give one.
- `gene_dictionary_jsd` is the Jensen-Shannon divergence across equally weighted
  topic-gene distributions in nats; identical programs give zero.
- `usage_effective_topics` is exp(H(mean cell-topic usage));
  `cell_effective_topics_mean` is the mean of exp(H(usage)) across cells.
- `usage_max_probability`, `usage_low_mass_fraction` (mean usage below 0.1/K),
  `cell_topic_mutual_information` and `topic_mean_usage` describe assignment
  concentration, low-usage topics and variation across cells.

These are monitoring statistics and do not contribute gradients or penalties.
High diversity alone does not establish coherent biological programs, and uniform
cell assignments can inflate the effective topic count. Interpret geometry,
gene programs and cell usage together with held-out reconstruction.

## Promote and publish a completed bank

The standard local run directory is `runs/hca_pretraining`. Its checkpoints and
logs remain there. After the three-epoch pipeline completes, it promotes the
selected topic bank to `assets/topic_bank` automatically. Promotion validates the
package and copies only `topics.safetensors`, `genes.json` and `manifest.json`.
A repeated promotion replaces those release files without copying training state.

```text
runs/hca_pretraining/
  checkpoints/latest/
  checkpoints/best/
  topic_bank/                 # Best validation-CE checkpoint exported at completion
assets/
  tf_sapiens/                 # Separate frozen backbone
  topic_bank/                 # User-facing topic bank and residual projection
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

Revised checkpoints use format 4 and topic banks use format 3 to include the
projection. Earlier architecture checkpoints cannot resume the revised model. Select the best bank using held-out reconstruction,
with diversity diagnostics used to identify degenerate solutions. The initial
OT weight was chosen from a real masked-batch gradient check: unweighted OT
prototype gradients were about 40 times smaller than reconstruction gradients.
Weight 10 makes the OT contribution about one quarter of that reconstruction
prototype gradient norm on this batch. This calibration is not evidence of an
optimal weight. At epsilon 0.03, 200 Sinkhorn iterations reduced the initial
maximum topic-marginal error to about 5.4e-8. The dictionary gene-marginal error is
also recorded during training as `dictionary_gene_marginal_max_error`.
