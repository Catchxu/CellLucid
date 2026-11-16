# HCA acquisition and atlas preprocessing

The `data/` directory contains the HCA downloader, its configuration, gene annotation
mapping, full-inventory preprocessing and the builder of the Reactome pathway table. Pretraining then encodes every prepared training
cell once into an embedding cache and trains on that cache; see [pretraining](pretraining.md).

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
are preferred; unknown or ambiguous symbols are not invented. An Ensembl gene ID is
read wherever it appears in the identifier, so version suffixes (`ENSG….12`) and
symbol prefixes (`TNMD_ENSG…`) are dropped. The TF-Sapiens
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

- Count sources to examine in order: `layers/counts`, `layers/raw`, `raw`, `raw.X`
  (the raw matrix of anndata files written before 0.7), `X`.
- Full-matrix checks: the first integer source is used; else the first source
  holding log1p(counts), restored to integers with expm1; else the first finite,
  nonnegative one with a warning. A file with none of these is excluded.
- ATAC exclusion for files whose library methods are all ATAC and for cells whose
  `assay` is ATAC. Visium files are prepared as expression; spatial coordinates are
  not used. Primary human rows are selected where those annotations are available.
- Exclusion of files that measure fewer than 1,000 backbone genes. Targeted panels
  such as MERFISH measure about 300; whole-transcriptome HCA files measure over
  11,000.
- Files excluded by name, with the reason (`exclude_files`). `decidua.h5ad` holds
  the cells of the two vento18 files of its study under other names; it is excluded
  and the vento18 files are kept.
- A minimum of 200 expressed backbone-aligned genes per retained cell.
- Removal of cells that an earlier file already holds (below).
- Sparse shards of up to 10,000 cells, with gene availability preserved.
- A fixed study-held-out validation split (`validation_studies`): the five
  studies of the held-out evaluation pool, so the pool stays held out when
  studies are added. Every other study is used for training.

Every source has an included, empty or excluded outcome. Excluded inputs and the
reason are recorded; unreadable files and unexpected I/O errors stop preparation.
Duplicate features mapping to the same gene are summed.

Count sources are validated for every file first. Files are then prepared in three
groups: stored integer counts, counts restored from log1p, and normalized
expression, so a cell held several ways is kept as stored counts. A cell is dropped as a copy when a QC-passing cell of
an earlier file has the same obs name, or within one study the same 16-nt barcode
(unique within each file), and the two cells' expressed genes, on the genes both
files measure, have a Jaccard index of at least 0.9. Each cell is confirmed on its
own, because plain barcodes and row numbers recur across samples: in one HTAN
pair, half of the barcode matches are different cells. Matched and removed counts
per earlier file are recorded under `duplicate_matches` in the file's coverage
report. A copy is not detected when it shares neither its name nor a unique
barcode, as in a stitched Visium file that holds three slides.

```text
<atlas>/
  manifest.json             # Corpus coverage, shared gene order, shards and splits
  genes.json                # One vocabulary for the entire corpus
  preparation.json          # Explicit preprocessing inputs/settings
  coverage.json             # Outcome and counts for every acquisition record
  sources.sqlite            # Count-source choices and completed source records
  files/<file_id>/
    source.json
    signature.npz           # Name and barcode keys of QC-passing cells for later files
    genes.npy               # Their expressed genes, read per duplicate candidate
    shard_00000.h5          # Sparse counts, cell IDs, assay and gene availability
    ...
```

Completed source records are reused on restart with the same inputs/settings.
The interrupted file's `.part` directory is rebuilt. A new corpus configuration
requires a new output directory. Original downloads and metadata are read-only.
Sparse payloads are streamed in bounded batches. Cell metadata is held for the
current source file. Across the atlas, hashed cell keys are kept in memory and
expressed genes are read from each file's `genes.npy` per duplicate candidate.

## Cache and pretrain

After preparation finishes:

```bash
NPROC_PER_NODE=4 bash scripts/build_cache.sh --manifest /path/to/atlas/manifest.json \
  --output /path/to/run/cache
bash scripts/train_topics.sh --cache /path/to/run/cache --output-dir /path/to/run
```

The cache builder reads raw-count shards, masks each cell once and encodes it with the frozen
TranscriptFormer; training then reads only the cache. Prepared shards are not modified. See [pretraining](pretraining.md) for checkpoint/resume semantics and the [root README](../README.md) for inference.

## Reactome pathways

Topic mapping reads `assets/reactome/human_pathways.csv`: one row per human pathway of
Reactome release 97, with its hierarchy, gene symbols and Reactome's `isInDisease` flag. It is
built from four files of the same release. `ReactomePathways.txt`,
`ReactomePathwaysRelation.txt` and `ReactomePathways.gmt` come from the
[Reactome downloads](https://reactome.org/download-data). `nodes_pathway.csv` holds the flag:
it is exported from the Reactome graph database (Neo4j) with the Python `neo4j` driver and
`csv` module, so the flag reads `True` or `False`, under the columns `pathway_id` and
`is_in_disease`:

```cypher
MATCH (p:Pathway {speciesName: "Homo sapiens"}) RETURN p.stId, p.isInDisease
```

```bash
python data/reactome.py --input-dir /path/to/reactome_release97 --release 97 \
  --output assets/reactome/human_pathways.csv
```

Reactome offers only its current release for download, so the release 97 table is kept in the
Hugging Face repository. Reactome data are released under CC0.
