# Command-line interface

## Denoise one dataset

```bash
ambidose denoise --input INPUT --output cleaned.h5ad
```

`INPUT` may be a raw `.h5`, `.h5ad`, raw MTX directory, or Cell Ranger `outs/` directory. The matching Cell Ranger filtered barcodes are the default: when a sibling filtered matrix is found, its barcode list is used automatically.

All `denoise` options:

| Option | Purpose |
|---|---|
| `--input`, `--root`, `--manifest` | Select exactly one single input, sample root, or library manifest |
| `--output` | Required output path |
| `--output-format` | `h5ad` (default) or `10x-mtx`. MTX defaults to Cell Ranger v3 (`matrix.mtx.gz`, `barcodes.tsv.gz`, three-column `features.tsv.gz`). |
| `--mtx-version` | `3` (default) or `2` (uncompressed `genes.tsv`). Only used with `--output-format 10x-mtx`. |
| `--report`, `--summary-json` | HTML and machine-readable QC outputs |
| `--cells-only` | Emit called cells with denoised counts in `X` |
| `--cell-barcodes` | External whitelist for single-input mode |
| `--cell-calling` | `diem`, `chi`, `emptydrops`, `ordmag`, `force`, or `off` |
| `--empty-umi-max` | Upper UMI bound for the empty-droplet pool |
| `--expected-cells` | Cell Ranger OrdMag size hint; required by `force` |
| `--max-cells` | Hard cap after cell calling |
| `--sample-key` | Independent library column for χ and per-library typing |
| `--type-key` | Trusted broad grouping from outside AmbiDose; otherwise coarse Leiden per library. Not a cell-type atlas. |
| `--full-typing` | Disable the large-library fast graph |
| `--n-jobs` | Worker limit for typing and structure regression |
| `--mark-doublets` | Scrublet and type-residual diagnostics on `raw_counts` |

Example:

```bash
ambidose denoise \
  --input raw_feature_bc_matrix.h5 \
  --cell-barcodes filtered_barcodes.tsv.gz \
  --type-key cell_type \
  --output cleaned.h5ad
```

If the Cell Ranger whitelist is unavailable or known to be unreliable, `--cell-barcodes` accepts a barcode TSV exported by a trusted external caller such as CellBender or EmptyDrops. `--input` must still point to the matching raw droplet matrix, not a cell-only corrected matrix.

For OCM/CMO biological samples multiplexed in one GEM/library, do not use the within-GEM sample assignment as `--sample-key`; those cells share one ambient profile. For combined independent GEMs/libraries, use a GEM or library column as `--sample-key`. AmbiDose does not correct OCM tags or demultiplex samples.

## Inspect inputs and generate QC outputs

```bash
ambidose inspect --input /data/sample/outs --output-json inspection.json

ambidose denoise \
  --input /data/sample/outs \
  --report report.html \
  --summary-json summary.json \
  --cells-only \
  --output cleaned.h5ad
```

See {doc}`../tutorials/reports_and_plots` for report contents, plotting, and analysis-ready output semantics.

## Denoise a sample root

```bash
ambidose denoise --root /data/runs --output cleaned.h5ad
```

The root mode discovers Cell Ranger sample folders (v3+ `outs/` first; v2 layouts still match), estimates each library's ambient profile separately, then concatenates samples over their shared genes. `--cell-barcodes` is intentionally rejected in root mode because barcodes must be paired per sample.

For per-library external whitelists, use a TSV manifest instead:

```bash
ambidose denoise --manifest libraries.tsv --output cleaned.h5ad
```

## Export χ only

```bash
ambidose estimate-chi \
  --input /data/sample/outs \
  --output ambient_profile.csv \
  --top 20
```

For multi-sample AnnData, the CSV rows retain sample names.

Cell calling and depth controls include `--cell-calling`, `--empty-umi-max`, `--expected-cells`, and `--max-cells`. Output controls include `--output-format`, `--cells-only`, and `--mark-doublets`; doublet diagnostics use the preserved raw-count layer.

Run `ambidose COMMAND --help` for the authoritative option list.
