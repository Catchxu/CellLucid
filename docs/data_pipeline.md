# Data contracts

## Acquisition manifest
HCA file UUID + version identifies a source file. The manifest retains its
project IDs, file role/source, URL, checksum, size and complete original API
record. API discovery never downloads expression contents. Downloading has
explicit file and total-byte budgets and atomically promotes verified partial
files. Signed/resolved URLs are not a replacement for stable source identity.

## Counts
A standard H5AD stores counts in X, Ensembl IDs in var.feature_id, original symbols
in var.gene_symbol, original cell metadata in obs and processing provenance in
uns.celllucid. Only small cell batches become dense. Sparse X is written
incrementally; preprocessing never loads the entire atlas expression matrix.
Full dataset metadata and gene lists may still be resident in memory.

Input and output paths must differ. Fractional or negative expression is rejected.
Duplicate normalized IDs are rejected. Zero-expression cells are excluded.
Feature filtering and QC settings must be explicit. Availability currently means
the gene panel present in a dataset; within-file mixed panels need upstream
availability-aware separation rather than treating unavailable genes as zeros.

## Frozen features
A feature shard stores cell/view identity, ordered genes, gene availability,
fixed projected gene vectors, contextual cell vectors and a dataset/backbone/
preprocessing/masking fingerprint. Counts remain in the referenced source H5AD.
All views of a cell stay in the same split. A modified source file invalidates
the feature cache.

## Topic bank
The dictionary has shape topics by genes, with each topic row summing to one.
The per-cell mixture has shape cells by topics and sums to one per cell.
Exported banks include checksums and immutable source identities. They do not
include pretrained encoder weights or imply cell-type/pathway labels.
