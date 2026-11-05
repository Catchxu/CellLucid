---
license: gpl-3.0
library_name: celllucid
tags:
  - single-cell
  - topic-modeling
  - transcriptomics
  - human-cell-atlas
  - transcriptformer
---

# CellLucid model assets

**CellLucid enables natural-language querying of single-cell atlases through interpretable topic
modeling.**

These are the model assets for [CellLucid](https://github.com/Catchxu/CellLucid). Each cell is
described as a mixture of 128 gene programs (topics).

## Contents

| Path | What it is | Origin |
| --- | --- | --- |
| `tf_sapiens/` | TF-Sapiens backbone: `config.json`, `model_weights.pt`, `vocabs/` | **Not trained by CellLucid.** TranscriptFormer weights released by the Chan Zuckerberg Initiative, redistributed unmodified |
| `topic_banks/default/` | Topic bank whose topics exclude mitochondrial, ribosomal-protein and TCR/BCR V-segment genes | Trained by CellLucid |
| `topic_banks/all_genes/` | Topic bank trained on every gene | Trained by CellLucid |
| `gene_symbols.json` | GENCODE v49 gene symbol to Ensembl ID mapping | Built from GENCODE by CellLucid |

## Backbone: TranscriptFormer (TF-Sapiens)

The files in `tf_sapiens/` come from
[TranscriptFormer](https://github.com/czi-ai/transcriptformer) by the Chan Zuckerberg Initiative.
They were downloaded from
[CZI](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz) and are
redistributed here unmodified, so that CellLucid installs from one place. CellLucid keeps the
backbone frozen and does not train or fine-tune it. TranscriptFormer is released under the
[MIT license](https://github.com/czi-ai/transcriptformer/blob/main/LICENSE.md), and its use
follows CZI's [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy).
If you use these weights, please cite TranscriptFormer:

> Pearce et al., *A cross-species generative cell atlas across 1.5 billion years of evolution:
> The TranscriptFormer single-cell model*, bioRxiv (2025).

## Topic banks

CellLucid trained both banks on frozen TF-Sapiens embeddings of 7.5 million cells from 22 Human
Cell Atlas studies, each cell used once; five further studies were held out for evaluation.
Each bank holds 128 topics: unit vectors in the backbone's embedding space, which assign a cell
by a temperature-0.05 softmax, plus a topic-by-gene dictionary. Banks store Ensembl IDs
(`genes.json`) and gene symbols (`symbols.json`).

| Bank | Best for |
| --- | --- |
| `default` | interpreting programs, comparing cell types across studies, querying |
| `all_genes` | quality control, technical or sample-level analysis, full-profile reconstruction |

## Usage

```bash
hf download kaichenxu/CellLucid --local-dir assets
```

Then follow the [CellLucid README](https://github.com/Catchxu/CellLucid#usage) to preprocess
counts and assign topics. The topic banks and CellLucid code are released under GPL-3.0.
