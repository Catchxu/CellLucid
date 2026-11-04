# CellLucid

**CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.**

Use a shared topic bank to describe cells in your own single-cell dataset,
compare their gene programs, and inspect the genes associated with each topic.
The current release supports topic-bank inference and inspection. Natural-language
querying and pathway interpretation are under development.

## Installation

Use Python >=3.11 and a CUDA-enabled PyTorch installation compatible with your GPU:

```bash
python -m pip install -e ".[backbone]"
```

Inference requires an exported CellLucid topic bank and its matching frozen
[TranscriptFormer TF-Sapiens](https://virtualcellmodels.cziscience.com/model/transcriptformer)
backbone. Keep the bank's files together, including `manifest.json`, `genes.json`
and `topics.safetensors`. Backbone dependency and asset setup are described in
[the setup guide](docs/pretraining.md#install). A public CellLucid topic-bank
release is not yet available.

## Topic-bank assets

The local release package lives in `assets/topic_bank/`. Once the dedicated
Hugging Face repository is available, download and load the same package with:

```python
from celllucid.tasks.topic_bank import TopicBank

bank = TopicBank.from_pretrained(
    "your-org/CellLucid", repo_type="space", local_dir="assets"
)
```

`your-org/CellLucid` is a placeholder for the future repository. A model repository
is also supported by setting `repo_type="model"`. Downloads select only the three
files under `topic_bank/`; the frozen backbone remains a separate dependency.

## Apply a topic bank to your data

Start with an H5AD containing raw integer counts and Ensembl gene identifiers.
The example below reads counts from `layers/counts` and gene IDs from
`adata.var["feature_id"]`:

```bash
bash scripts/preprocess.sh \
  --input /path/to/input.h5ad --output /path/to/counts.h5ad \
  --count-source layers/counts --gene-key feature_id

bash scripts/infer_topics.sh \
  --input /path/to/counts.h5ad \
  --topic-bank assets/topic_bank \
  --output /path/to/topic_usage.h5
```

Choose `--count-source X` when raw counts are stored in `adata.X`. Gene-symbol
inputs require an explicit mapping; see [the data guide](docs/data.md).
`--topic-bank` defaults to `assets/topic_bank`. To download the bank during CLI
inference, use `--hub-repo-id your-org/CellLucid --hub-repo-type space` instead.
Set the backbone asset path and inference batch size in `configs/topic.yaml` to
match the supplied bank. Preprocessing writes a new file and preserves the input.

## Read cell-topic assignments

The output stores one row per retained cell and one column per topic:

```python
import h5py
import pandas as pd

with h5py.File("/path/to/topic_usage.h5", "r") as handle:
    cell_ids = handle["cell_ids"].asstr()[:]
    usage = pd.DataFrame(handle["theta"][:], index=cell_ids)

usage.columns = [f"topic_{i}" for i in usage.columns]
print(usage.head())
```

Each row sums to one and represents a mixture of gene programs. Align cells to
your annotations using the saved cell IDs. `gene_available` records which bank
genes were measured in the input dataset.

## Inspect topic-associated genes

```python
from celllucid.tasks.topic_bank import TopicBank

bank = TopicBank("assets/topic_bank")
for topic_id, genes in enumerate(bank.top_genes(n=20)):
    print(topic_id, genes)
```

Gene weights describe each shared program; they are not pathway annotations or
statistical enrichment results. Topic usage can be summarized by sample, cell
type or another annotation in your dataset.

## Further documentation

- [Data sources and gene mapping](docs/data.md)
- [Pretraining and development](docs/pretraining.md)

Large datasets, model assets and experiment outputs are excluded from Git.
