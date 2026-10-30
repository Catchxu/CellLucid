# HCA data download

The acquisition code, configuration and shell entrypoint live together in this
directory, following the repository's data-utility layout. Modeling data loaders,
count processing, feature caches and splits remain in `src/celllucid/data/`.
Run the commands below from the repository root with the project dependencies
available in the active Python environment. The shell entrypoint imports both
this directory and `src/` from the checkout; no TranscriptFormer weights are
needed for downloading.

## Files

| File | Purpose |
| --- | --- |
| `download_hca.sh` | Single Bash and Slurm entrypoint |
| `hca.yaml` | Source selection, budgets, concurrency and space reserve |
| `download_hca.py` | Inventory and transfer orchestration |
| `hca.py` | HCA Azul metadata discovery |
| `download.py` | Resumable HTTP transfer and file-size validation |
| `manifest.py` | Source-file inventory serialization |
| `metadata.py` | Project summaries and original source metadata |

## Download HCA

From the repository root:

```bash
bash data/download_hca.sh --output-dir /path/to/HCA --inventory-only
bash data/download_hca.sh --output-dir /path/to/HCA
```

The same download script supports Slurm:

```bash
sbatch --account=<account> --partition=<cpu_partition> \
  data/download_hca.sh --output-dir /path/to/HCA
```

Choose scheduler resources and log paths at submission as needed. Submit from the
repository root so the Slurm spool copy can use `SLURM_SUBMIT_DIR`. The download
script accepts `PYTHON_BIN`; set `PYTHON_BIN=python3` to select the active environment
explicitly if a project `.venv` also exists. The other stage scripts use `python3`
from the active environment by default and also accept `PYTHON_BIN`.

`manifest.json` records source UUID/version, project, URL, byte size, local
path and transfer status. Original metadata is retained in `metadata/`; matrices
are in `raw/<file_id>/`. Rerun the same command to resume. Changing the catalog or
selection requires a new output directory. `--max-files`, `--max-file-bytes` and
`--max-total-bytes` can limit a trial using the same configuration.

This inventory contains public human-associated H5AD matrices, not a curated RNA
training cohort. Review modality, species, overlapping studies and count layers
before preprocessing. The default file/byte budgets are explicit in `data/hca.yaml`.

## Configuration and outputs

`data/hca.yaml` selects accessible public human-associated H5AD matrices from HCA
catalog `dcp60`, across source categories.
It sets explicit file, per-file-byte and total-byte budgets, four download workers
and a 100 GB free-space reserve. Discovery records whether the catalog was
exhausted or a limit was reached. These limits do not imply biological coverage.

The default output is `data/HCA/`; a separate dataset location can be selected
with `--output-dir`. The acquisition code and guide are tracked in Git; generated
subdirectories and large datasets are ignored.

```text
<output_dir>/
  manifest.json
  raw/<file_id>/<original_file_name>
  metadata/<file_id>.json
```

The manifest's `selection` and `api` describe the frozen inventory; each record
includes its source version, expected bytes, local path and status.
Verified files are checked locally when resuming. Interrupted transfers retain
`.part` files and use HTTP Range where supported. Failed transfers retain their
errors in the manifest and make the final command exit nonzero. A process lock
prevents two jobs from modifying the same output directory concurrently.

The shell file retains Slurm resource headers, which Bash treats as comments.
Supply cluster account, partition and any resource overrides at submission;
choose `sbatch --output` and `--error` paths if needed and create their parent
directory before submission. Submit from the repository root, or set
`CELLLUCID_ROOT` explicitly, so Slurm's spool copy can find the checkout.

This utility runs from a source checkout; acquisition is not a console command
in the installed modeling package. The Python equivalent of the shell entrypoint
is `PYTHONPATH=.:src python -m data.download_hca --config data/hca.yaml`.
