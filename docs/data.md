# HCA acquisition and atlas preprocessing

The `data/` directory contains the HCA downloader, its configuration, gene annotation
mapping and full-inventory preprocessing. Pretraining then deduplicates cells stored in
several source files, encodes a study-balanced embedding cache and trains on that cache; see
[pretraining](pretraining.md).

## Acquire data and annotation

Run from the repository root in the active Python environment:

```bash
bash data/download_hca.sh --output-dir /path/to/HCA --inventory-only
bash data/download_hca.sh --output-dir /path/to/HCA
```

The single download script also supports `sbatch`. Supply your account,
partition, resource overrides and log paths at launch. `data/hca.yaml` sets the
source format and file/byte budgets. The default selection is public,
human-associated H5AD matrices in HCA catalog `dcp60`; it is not all files or all
cells advertised by HCA. Extend acquisition coverage explicitly when adding
other supported source formats. Original matrices and source metadata are retained
under `raw/<file_id>/` and `metadata/`, with file versions and transfer status in
`manifest.json`. Rerun the same command to resume an interrupted download.

For symbol-only gene annotations, use the pinned
[GENCODE human release 49](https://www.gencodegenes.org/human/release_49.html):

```bash
curl -L https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz \
  -o assets/gencode.v49.annotation.gtf.gz
python data/build_gene_mapping.py --gtf assets/gencode.v49.annotation.gtf.gz \
  --output assets/gene_symbols.json
```

Only unambiguous symbol-to-gene mappings are written. Explicit Ensembl identifiers
are preferred; unknown or ambiguous symbols are not invented. The TF-Sapiens
vocabulary defines the ordered human gene space. Preparation accepts modern and
legacy H5AD tables and CSR, CSC or dense expression storage. CSC matrices are
converted once to CSR in memory for each source; provision RAM for both sparse
representations during conversion. CSR and dense inputs remain streamed in row
batches. Shared-filesystem memory maps are avoided because random conversion
writes cause severe I/O amplification.

## Prepare every source in the inventory

```bash
sbatch --account=<account> --partition=<cpu_partition> --mail-user=<email> \
  --output=/path/to/logs/prepare-%j.out data/prepare_topics.sh \
  --manifest /path/to/HCA/manifest.json --output-dir /path/to/atlas
```

Preparation and training request email on completion or failure. Supply the
recipient with `--mail-user`. Create the log directory before submission. The input is the complete acquisition
manifest. No per-dataset selection YAML or small-cohort manifest is required.
`configs/preprocess.yaml` specifies:

- Count sources to examine in order: `layers/counts`, `layers/cellranger`, `raw`, `X`.
- Full-matrix checks for finite, nonnegative integer expression.
- RNA modality exclusions for spatial/ATAC assays and primary human rows where
  those annotations are available.
- A minimum of 200 expressed backbone-aligned genes per retained cell.
- Sparse shards of up to 10,000 cells, with gene availability preserved.
- A deterministic 20% study-held-out validation split.

Every source has an included, empty or excluded outcome. Excluded inputs and the
reason are recorded; unreadable files and unexpected I/O errors stop preparation.
Duplicate features mapping to the same gene are summed. Exact duplicates matching
study/donor/cell identity, measured-gene availability and aligned counts are
removed across files. Changed counts or renamed cells require additional overlap
review; the duplicate index does not establish biological independence.

```text
<atlas>/
  manifest.json             # Corpus coverage, shared gene order, shards and splits
  genes.json                # One vocabulary for the entire corpus
  preparation.json          # Explicit preprocessing inputs/settings
  coverage.json             # Outcome and counts for every acquisition record
  sources.sqlite            # Completed source records and exact-duplicate index
  files/<file_id>/
    source.json
    shard_00000.h5          # Sparse counts, cell IDs, assay and gene availability
    ...
```

Completed source records are reused on restart with the same inputs/settings.
The interrupted file's `.part` directory is rebuilt. A new corpus configuration
requires a new output directory. Original downloads and metadata are read-only.
Sparse payloads are streamed in bounded batches. Cell metadata is held for the
current source file rather than materializing metadata for the whole atlas.

## Deduplicate, cache and pretrain

After preparation finishes:

```bash
python -m celllucid.data.dedup --manifest /path/to/atlas/manifest.json \
  --output runs/corpus_dedup/duplicates.json
NPROC_PER_NODE=4 bash scripts/build_cache.sh --manifest /path/to/atlas/manifest.json \
  --output /path/to/run/cache
bash scripts/train_topics.sh --cache /path/to/run/cache --output-dir /path/to/run
```

Deduplication finds cells deposited in more than one file of a study (the same barcode with
matching counts on shared genes) and keeps the copy with the most measured genes. Prepared
shards are not modified. The cache builder reads raw-count shards, masks each occurrence of a
cell with its own view and encodes it once with the frozen TranscriptFormer; training then
reads only the cache. See [pretraining](pretraining.md) for checkpoint/resume semantics and the [root README](../README.md) for inference.
