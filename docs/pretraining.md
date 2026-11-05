# CellLucid pretraining and development

CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.

The current implementation pretrains a shared topic model across an HCA atlas in two
stages. Every eligible cell is masked and encoded once into a cache of frozen TranscriptFormer
embeddings; the topic model is then trained in one pass over that cache.
Inference on new datasets encodes unmasked counts directly and needs no cache. Pathway
interpretation and the query agent will be added later.

## Install

Install as in the [README](../README.md):

```bash
python -m pip install -e ".[backbone]"
python -m pip install --no-deps transcriptformer==0.6.1
hf download kaichenxu/CellLucid --local-dir assets
```

TranscriptFormer 0.6.1 on PyPI matches upstream commit `c943a89` in every module the backbone
fingerprint covers. It pins older scientific packages, so it is installed with `--no-deps`. The
`backbone` extra supplies the Lightning, Scanpy and Hydra runtime it needs, without the upstream
PyTorch pin. Current validation uses PyTorch 2.11 and AnnData 0.12. Upstream code and backbone
weights are not modified.

The backbone fingerprint, which a topic bank must match, covers the TF-Sapiens files, the
upstream modules and the input settings. It records the PyTorch version without fingerprinting
it, so a bank works across PyTorch builds.

`hf download` fetches the TF-Sapiens weights, both topic banks and the gene-symbol mapping in
one step. The weights are byte-identical to CZI's
[tf_sapiens.tar.gz](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz),
which is needed only on a machine that cannot reach the Hub. Building the gene annotation from
GENCODE is described in [the data guide](data.md).

## Repository

| Path | Purpose |
| --- | --- |
| `data/` | HCA acquisition, annotation mapping and full-inventory corpus preparation |
| `configs/preprocess.yaml` | Count-source policy, QC, sharding and study split |
| `configs/topic.yaml` | Frozen backbone, cache, topic model, training and evaluation settings |
| `scripts/build_cache.sh` | Multi-GPU, resumable encoding of the training cache |
| `scripts/train_topics.sh` | Single-GPU topic training on a completed cache |
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

Pretraining runs in two steps after count preparation:

```bash
NPROC_PER_NODE=4 bash scripts/build_cache.sh --manifest /path/to/atlas/manifest.json \
  --output /path/to/run/cache
bash scripts/train_topics.sh --cache /path/to/run/cache --output-dir /path/to/run
```

For a workstation, `NPROC_PER_NODE=4 bash scripts/pretrain_atlas.sh /path/to/HCA /path/to/run`
runs acquisition, preprocessing, cache building, training and export in sequence. Each step
resumes: completed downloads, preprocessing sources, finished cache parts and the latest training checkpoint are reused. Progress is recorded in
`<run>/pipeline_status.txt`.

### Training cache

The cache holds every eligible cell of the training split exactly once, masked with view 2.
Eligible cells exclude the evaluation-pool cells. Studies are neither upsampled nor downsampled, so each enters training in
proportion to its size. A part stores FP16 embeddings, the hidden-gene set as packed bits and
the nonzero hidden counts; training holds the whole cache in memory (about 10 KB per cell).
`identity.json` records the corpus, backbone, evaluation cells and plan;
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
topic vectors by cosine, and a softmax at temperature 0.05 gives the assignment. There is no
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

`model.excluded_genes` leaves technical genes out of the loss: matching genes are not scored
and keep only the uniform share of the gene marginal, so no topic is defined by them. The
default rule matches gene symbols from `model.gene_symbols`:
- the 13 mtDNA protein genes (`MT-`), which track cell stress and lysis;
- 79 canonical cytosolic ribosomal protein genes (`RPL`/`RPS` with a number, their X/Y
  isoforms, `RPLP0-2`, `RPSA`), which track cell size and sequencing depth;
- 250 TCR and BCR V segments (`TRAV`, `TRBV`, `TRDV`, `TRGV`, `IGHV`, `IGKV`, `IGLV`), whose use
  marks a clone rather than a cell type.

The rule keeps constant-region genes (`TRAC`, `IGHM`, `IGKC` and the like), J and D segments
(not expressed in this corpus; an `IGHD` prefix would also catch the IgD constant gene), and the
tissue-restricted or functionally distinct ribosomal paralogs (`RPL3L`, `RPL10L`, `RPL39L`,
`RPL22L1`, `RPL26L1`, `RPL7L1`, `RPS27L`). Cell assignment is unaffected, since the frozen
embeddings still see every gene. Setting `excluded_genes: null` trains the `all_genes` bank.

The OT value is deliberately not minimized. The dictionary depends only on how a topic's
cosines differ across genes, while the OT value rewards their absolute level, so it pulls
topics out of the cell space toward their genes. In a 1000-step pilot with weight 3, two thirds
of each topic's norm left the cell subspace, assignments flattened (held-out cell-topic MI
0.35 against 0.81 without it), and held-out CE gain and pair-ranking AUC were lower.

Training uses one device and is one pass over the cache: the cells are read in a fixed random
permutation (seeded by `training.seed`) in batches of `training.batch_size`, so the number of
steps is the cache size divided by the batch size, and the last partial batch is dropped.
Because the permutation is global, a batch's study composition differs from the corpus
proportions only by sampling noise, and no study arrives in long runs.

The AdamW learning rate rises linearly to `training.optimizer.lr` over `training.warmup_steps`
and then follows a cosine decay to `training.min_lr` at the last step; `warmup_steps: 0` with
`min_lr` equal to the peak gives a constant rate. Warmup keeps the first updates from pulling
the k-means topics away from the cells before the gene map has aligned; the decay lets the
topics settle instead of drifting at a constant rate. Checkpoints in
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

## Topic banks and assets

Two banks are released, trained identically except for `model.excluded_genes`:

| Bank | Genes in the loss | Use |
| --- | --- | --- |
| `default` | technical genes excluded | interpretation, querying, comparing cell types across studies |
| `all_genes` | every gene | quality control, technical and sample-level analysis, full-profile reconstruction |

After training, `celllucid.tasks.topic_bank` exports the checkpoint and
`scripts/promote_bank.sh` copies it into `assets/topic_banks/<name>`:

```bash
python -m celllucid.tasks.topic_bank --checkpoint /path/to/run/checkpoints/latest \
  --cache /path/to/run/cache --output /path/to/run/topic_bank
bash scripts/promote_bank.sh --bank /path/to/run/topic_bank --name default
```

The published assets folder holds the backbone and both banks:

```text
assets/
  tf_sapiens/                 # TF-Sapiens weights and vocabularies (MIT, CZI)
  topic_banks/default/        # topics.safetensors, genes.json, symbols.json, manifest.json
  topic_banks/all_genes/
  gene_symbols.json           # GENCODE v49 symbol-to-Ensembl mapping
```

`scripts/push_assets.sh` uploads these files and the model card (`docs/model_card.md`) to
[kaichenxu/CellLucid](https://huggingface.co/kaichenxu/CellLucid). It needs `hf auth login`
and a network route to huggingface.co. Sunlab has none, since its `HF_ENDPOINT` mirror is
download-only, so run it from a machine that has a route. The repository must already exist:
`hf repos create kaichenxu/CellLucid --repo-type model`.

```bash
bash scripts/push_assets.sh
```

## Design decisions

Each choice below was tested on the full corpus or a pilot. Each row compares one setting
change with all other settings fixed. AUCs are held-out within-/cross-study pair ranking.

| Decision | Alternative tested | Result |
| --- | --- | --- |
| No OT value term in the loss | OT weight 3 | Flattened assignments; MI 0.35 vs 0.81, cross AUC 0.83 vs 0.88 |
| One pass, every cell once, natural study proportions | study-balanced upsampling | Chosen for simplicity; batch composition differs from the corpus only by sampling noise |
| Warmup 200 steps, cosine decay from 1e-3 | constant 1e-3; peak 3e-4 | Small but consistent gains; drift saturates by step 1200 |
| τ 0.05 | τ 0.1, 0.03 | AUC 0.853 / 0.911 vs 0.826 / 0.890 at τ 0.1; τ 0.03 adds little and separates same-type cells across studies |
| 128 topics | 256 topics | 256 adds reconstruction, not cross-study ranking, and leaves 39 topics nearly unused |
| Exclude technical genes (`default` bank) | all genes | Within AUC 0.863 vs 0.853, cell-type MI 0.60 vs 0.56; no ribosomal, mitochondrial, V-segment or flat topics (about 45% of usage otherwise); reference cross-study AUC 0.917 vs 0.929 |

Cardiac topics illustrate the resolution. Cardiomyocytes load on atrial, excitation–contraction
and maturation programs. TBX5, GATA4 and TBX20 rank among the top genes of these topics, and the
CollecTRI targets of NKX2-5, TBX5, MEF2C and GATA4 are enriched most in them.

## Inference and tests

For new datasets, preprocess raw counts, then apply an exported bank:

```bash
bash scripts/preprocess.sh --input /path/to/input.h5ad --output /path/to/counts.h5ad \
  --count-source layers/counts --gene-key feature_id
bash scripts/infer_topics.sh --input /path/to/counts.h5ad \
  --topic-bank assets/topic_banks/default --output /path/to/usage.h5
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q -m 'not integration'
```

Remove `runs/pytest-tmp` and other disposable test outputs after testing while
preserving scientific results. The marked integration test requires real backbone
assets and an accelerator. Full GPU throughput should be benchmarked for the
chosen context, precision and batch sizes before a long pretraining run.

Checkpoints use format 5 and topic banks format 5. A bank stores normalized topics, the
dictionary, the gene marginal, both cell means and the gene map, with Ensembl IDs and gene
symbols; `TopicBank.transform` centers unmasked embeddings by the inference mean. Earlier
checkpoints and banks cannot be loaded.
