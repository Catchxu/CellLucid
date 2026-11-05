# CellLucid

**CellLucid enables natural-language querying of single-cell atlases through interpretable topic modeling.**

CellLucid describes each cell as a mixture of 128 gene programs (topics) learned across the
Human Cell Atlas. A frozen [TranscriptFormer](https://github.com/czi-ai/transcriptformer)
(TF-Sapiens) encodes the cells, and a pretrained topic bank turns each embedding into topic
usage. Each topic is defined by a distribution over genes. The natural-language query agent is
under development.

## Installation

Use Python ≥ 3.11 with a CUDA build of PyTorch for your GPU. Install CellLucid, then the
TranscriptFormer code (`--no-deps`, because it pins an older PyTorch), then download the model
assets:

```bash
git clone https://github.com/Catchxu/CellLucid.git && cd CellLucid
python -m pip install -e ".[backbone]"
python -m pip install --no-deps transcriptformer==0.6.1
hf download kaichenxu/CellLucid --local-dir assets
```

The [Hugging Face repository](https://huggingface.co/kaichenxu/CellLucid) holds everything
inference needs: the TF-Sapiens weights, redistributed unmodified from TranscriptFormer, and
both CellLucid topic banks. Do not download TF-Sapiens separately.

## Usage

Start from an H5AD with raw integer counts and Ensembl gene IDs. Preprocess it, then assign
topics:

```bash
bash scripts/preprocess.sh --input data.h5ad --output counts.h5ad \
  --count-source layers/counts --gene-key feature_id
bash scripts/infer_topics.sh --input counts.h5ad --output usage.h5 \
  --topic-bank assets/topic_banks/default
```

Use `--count-source X` when the counts are in `adata.X`. Gene-symbol inputs need an explicit
mapping; see the [data guide](docs/data.md).

Read the usage table and the genes of each topic:

```python
import h5py, pandas as pd
from celllucid import TopicBank

with h5py.File("usage.h5") as f:
    usage = pd.DataFrame(f["theta"][:], index=f["cell_ids"].asstr()[:])  # cells x topics, rows sum to 1

bank = TopicBank("assets/topic_banks/default")
for topic, genes in enumerate(bank.top_genes(n=10)):
    print(topic, [g["gene"] for g in genes])
```

## Choosing a topic bank

| Bank | Topics are defined without | Best for |
| --- | --- | --- |
| `default` | mitochondrial, ribosomal-protein and TCR/BCR V-segment genes | interpreting programs, comparing cell types across studies, querying |
| `all_genes` | (nothing excluded) | quality control, technical or sample-level analysis, full-profile reconstruction |

Both banks assign topics from the same embeddings, and each takes about 30 MB.

## More

- [Data sources and gene mapping](docs/data.md)
- [Pretraining, design decisions and publishing assets](docs/pretraining.md)

CellLucid code is released under GPL-3.0. TF-Sapiens is released by the Chan Zuckerberg
Initiative under the MIT license, and its use follows CZI's
[Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy).
