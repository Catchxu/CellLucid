# Validation record

Validated on Sunlab on 2026-10-26 (America/New_York).

## Automated checks

- **20 tests passed**, including the real TF-Sapiens GPU freeze/determinism test.
- Ruff checks passed; the Python wheel built successfully.
- Tests cover OT marginals and gradients, masked target isolation, source/gene
  alignment, count validation, grouped splits, cache invalidation, bank integrity,
  resumed HTTP transfers, and exact optimizer/pipeline resume behavior.
- Upstream weights and source code were not edited. The checkpoint SHA256 is
  `eff027e7393f92a32cadcbb2e13a917e24b62b71341571bf40f38ff6ec6d3021`.
- The provided PBMC source remains unchanged: SHA256 `e649aacfd25f51c62f89e5834bb09e45ee18cd5054468a3efc771b42499723e4`.

## Real PBMC smoke run

- Input: `/data1021/xukaichen/data/CTA/pbmc.h5ad`, 68,579 cells, 32,738 genes.
- Preprocessed 256 cells; extracted 64 cells with two fixed masked views each.
- Backbone: frozen TF-Sapiens, 2,048-dimensional representations, 127-gene-token
  context, FP32 and uncompiled upstream attention for this small test.
- Learned 16 topic vectors; dictionary shape: 16 x 23823.
- 51 training cells and 13 validation cells; all views of a cell share its split.
- Validation cross-entropy: **8.486 initially -> 6.550 after three epochs**.
- Dictionary row-sum error: 1.19e-07; fixed gene-marginal
  error: 3.64e-11.
- Exported bank: `runs/pbmc_smoke/topic_bank/`.
- Full-input inference succeeded on 16 PBMC cells.

## Small HCA test

- Project: `4c8e9d75-d85a-47de-9598-06549cf44b91`, cell-type mapping in
  subcortical multiple-sclerosis lesions.
- File: `ctype_BC.h5ad`, UUID `1e3ceb1d-9c82-4164-915c-3c1445d964a5`.
- Version: `2024-10-28T11:36:35.640000Z`.
- Downloaded **4,792,583 bytes** and verified the published SHA256:
  `07fa5a6d218d15ed1441db5408441f455bb71ec0cbccd2f8054514c07a781d51`.
- Matrix: 275 cells and 14,156 genes with integer-valued expression.
- This author matrix provides symbols. An explicit lookup from the supplied
  PBMC annotation mapped 11,723 symbols to Ensembl IDs. Unmapped names are
  retained, not guessed; 11,216 features overlap the TF-Sapiens bank vocabulary.
  This source-specific lookup is not a comprehensive reference annotation.
- The same PBMC-trained bank inferred topic usage for 16 HCA cells, retaining
  the same bank ID and dictionary. Both inference outputs have finite,
  nonnegative mixtures summing to one.

## Practical limits

This is an integration smoke test, not atlas-scale training or a biological
validation of the topics. Its cell-level split is not held-out-study evaluation.
The small context length and fixed cached mask views are explicit development
choices. Larger research runs should configure the intended context, masks,
cohort overlap policy and study/donor-held-out splits.

The initial count preprocessor supports H5AD. Other downloaded formats need
conversion first. Acquisition budgets are deliberately small by default.
Upstream older package pins differ from the tested PyTorch 2.11 environment;
see setup.md. The smoke setting `compile_attention: false` triggers an upstream
warning about unfused attention and materialized score matrices. Use and validate
compiled attention before large-scale extraction. Other warnings in the test
log are Matplotlib/Pyparsing deprecations imported by upstream dependencies.

Full logs, configurations, checkpoints, manifests and machine-readable checks
remain in the ignored `runs/`, `data/` and `cache/` directories.
