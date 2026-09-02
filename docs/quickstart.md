# Quickstart

This page runs the supported workflow from raw droplets to a denoised integer-count layer.

## 1. Prepare the input

The default input is a Cell Ranger raw matrix plus its matching
filtered cell barcodes. Keep raw and filtered outputs from the same library.
If that Cell Ranger whitelist is unavailable or known to be unreliable,
`cell_barcodes=` can instead receive an external whitelist from
CellBender, EmptyDrops, or another caller.

| Input | Python | CLI |
|---|---|---|
| Raw feature matrix HDF5 | `read_10x_h5(...)` | `--input raw_feature_bc_matrix.h5` |
| Raw matrix directory | `read_10x_mtx(...)` | `--input raw_feature_bc_matrix/` |
| Cell Ranger `outs/` | Use `ambidose.io.sniff_input` | `--input outs/` |
| Existing AnnData | `scanpy.read_h5ad(...)` | `--input counts.h5ad` |

For a standalone raw HDF5 or MTX directory, pass `cell_barcodes=` explicitly
unless a matching filtered output is beside it. Do not interpret every
barcode above the empty-droplet UMI threshold as a trusted cell call. An
external caller supplies barcode names only; always run AmbiDose on the
matching raw count matrix, not on CellBender-corrected counts.

## 2. Run the standard workflow

If you already load your own filtered, cells-only object the usual scanpy way, pass the matching raw matrix as `raw=` -- `adata`'s own barcodes become the cell whitelist:

```python
import scanpy as sc
import ambidose as amdose

adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
adata = amdose.denoise(adata, raw="raw_feature_bc_matrix.h5", sample_key=None)
```

This returns a new, barcode/gene-aligned copy of `adata` with the results attached (any `obs` columns you already had are preserved) -- it does not mutate the object you passed in. `X` on the returned object is the denoised counts, ready for `sc.pp.normalize_total` etc. with no extra step; the original input moves to `layers["raw_counts"]` (same convention `analysis_ready()` uses, and the same convention the whitelist-first form below also follows).

`write_report()`/`summarize()` work on this returned object, but only see cells: the empty droplets used to estimate χ were never part of `adata` and are not carried into the copy, so `n_empty`/`n_other` read 0 and the droplet-class and barcode-rank panels show cells only. If you want the full report with those panels, use the lower-level form below and call `write_report()` on its (mutated) raw object instead.

The lower-level, whitelist-first form still works and is what `raw=` calls internally -- reach for it when you want the full report, want to load the raw matrix yourself (e.g. to also inspect empty droplets), or already have a barcode list from an external caller:

```python
import ambidose as amdose

adata = amdose.read_10x_h5("raw_feature_bc_matrix.h5")
barcodes = amdose.read_10x_barcodes("filtered_feature_bc_matrix.h5")

amdose.denoise(
    adata,
    cell_barcodes=barcodes,
)
```

In this form, `denoise()` mutates the AnnData in place and also returns the same object. Passing `cell_barcodes=` a second time on the same object raises: drop `obs["ambidose_droplet"]` and stored χ, or copy the raw matrix, if you need a new whitelist. Omit `cell_barcodes` to reuse existing labels and χ. `raw=` and `cell_barcodes=` are mutually exclusive.

`summarize()` / `write_report()` are post-`denoise()` only: they read `uns["ambidose"]`. To inspect a file before denoising, use `inspect_input`.

## 3. Inspect the outputs

```python
print(adata.obs["ambidose_droplet"].value_counts())
print(adata.obs[["ambidose_rho", "ambidose_d"]].describe())
print(adata.layers["ambidose_denoised"])
```

| Location | Meaning |
|---|---|
| `obs["ambidose_droplet"]` | `empty`, `cell`, or `other` |
| `obs["ambidose_doublet"]` | Optional Scrublet prediction written by `mark_doublets()`; diagnostic only |
| `obs["ambidose_rho_trust"]` | `ok`, `low_evidence`, `ceiling_risk`, `type_structure_risk`, `under_execution`, `over_removal`, or `not_cell` |
| `var["ambidose_chi"]` | Single-sample ambient profile |
| `uns["ambidose_chi"]` | Multi-sample ambient profiles, samples × genes |
| `obs["ambidose_rho"]` | Per-cell ambient fraction |
| `obs["ambidose_d"]` | Per-cell absolute ambient UMI dose |
| `obs["ambidose_dose_execution_ratio"]` | Actual removed UMI divided by estimated dose; `NaN` for zero-dose rows |
| `obs["ambidose_removed_fraction"]` | Fraction of each barcode's input UMI removed |
| `X` | Set to denoised counts only when counts were read from `X`; `denoise(layer=...)` leaves caller-owned `X` unchanged |
| `layers["ambidose_denoised"]` | Corrected non-negative integer UMI counts for `cell` barcodes; empty / other / doublet rows are passed through unchanged |
| `layers["raw_counts"]` | The count matrix selected by `layer` (or `X` when `layer=None`), preserved |
| `obs["ambidose_cluster"]` | Automatic coarse Leiden groups when no `type_key` is given; not cell-type names |

Calling `subtract()` directly (bypassing `denoise()`) does not do this -- `X` stays whatever it was, and only `layers["ambidose_denoised"]` is written; see {doc}`user_guide/workflow`.

## 4. Use your own coarse labels

A trusted **broad** annotation is preferable when available. Compute it outside AmbiDose (your usual annotator, not this package) and pass the column:

```python
amdose.denoise(
    adata,
    cell_barcodes=barcodes,
    type_key="cell_type",
)
```

Labels should describe major populations with enough cells to estimate type-level expression. They protect native genes and identify ambient-only genes. Fine atlases and per-cell marker overrides are the wrong grain; annotate cell types on the denoised counts after this step. See {doc}`faq`.

## 5. Multi-sample data

Each sample must retain its own empty droplets and its own ambient profile. For a directory containing Cell Ranger sample folders:

```bash
ambidose denoise \
  --root /data/cellranger_runs \
  --output cleaned.h5ad
```

The CLI estimates $\chi_s$ before concatenation, intersects the gene sets, and keeps the sample-indexed profiles in `uns["ambidose_chi"]`.

### OCM/CMO multiplexing and external cell whitelists

OCM does not by itself replace the default Cell Ranger whitelist. Use the
matching Cell Ranger filtered barcodes when they are reliable. For an unusual
high-background library where Cell Ranger cell calling is unreliable, extract
only the barcode names from an alternative caller and pass them explicitly:

```python
import scanpy as sc
import ambidose as amdose

raw = amdose.read_10x_h5("raw_feature_bc_matrix.h5")
cellbender = sc.read_10x_h5("cellbender_filtered.h5")

amdose.denoise(
    raw,
    cell_barcodes=set(cellbender.obs_names),
    sample_key=None,
)
```

This uses CellBender only for cell calling; AmbiDose still operates on raw
counts. Biological OCM assignments remain ordinary `obs` metadata. Cells
multiplexed in one GEM well share one ambient profile, and empty droplets
cannot be assigned reliably to the biological samples, so do not use the OCM
assignment as `sample_key`. For multiple independent GEM wells, use the
GEM/library identifier as `sample_key` or process each well independently.
OCM/CMO tag correction and singlet/multiplet demultiplexing are external to
AmbiDose.

## 6. Continue with scanpy

```python
cells = amdose.analysis_ready(adata)
# X is denoised; raw UMIs are in layers["raw_counts"]
# summarize(cells) still uses raw_counts, not the overwritten X

import scanpy as sc
sc.pp.filter_genes(cells, min_cells=3)
sc.pp.normalize_total(cells)
sc.pp.log1p(cells)
```

Subset cells only after ambient estimation unless the object already contains a valid stored $\chi$.
