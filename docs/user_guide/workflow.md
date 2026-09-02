# Workflow and data contract

## Use `denoise()`

The product path is `denoise()`, then `write_report()` (or CLI `--report`). `classify_droplets`, `estimate_chi`, `estimate_dose`, and `subtract` are steps inside that path.

The completed product output keeps a small public schema: `obs` contains `ambidose_droplet`, the final `ambidose_d` and `ambidose_rho`, `ambidose_removed_umi`, and `ambidose_rho_trust`; automatic typing also adds `ambidose_cluster`. Estimator-specific working columns are removed before `denoise()` returns. `var` stores the ambient profile and per-gene removal, while `uns["ambidose"]` stores run-level summaries.

Two calling conventions: `denoise(raw_adata, cell_barcodes=...)` mutates and returns the raw+empty-droplet object you loaded yourself (the form the rest of this page describes); `denoise(filtered_adata, raw=...)` instead takes your own already-filtered, cells-only object as `adata`, uses its `obs_names` as the whitelist, and returns a denoised, barcode/gene-aligned copy of that same object (see {doc}`../quickstart`). The two are mutually exclusive (`raw=` and `cell_barcodes=` cannot both be given) and produce identical results for the same underlying data.

## Raw counts are required

`adata.X`, or the layer passed with `layer=`, must contain raw UMI counts. In `raw=` mode, the raw pool's `X` is authoritative and `layer=` is rejected; `layers["raw_counts"]` in the result contains the matched raw-pool cell counts, not the filtered object's original `X`. Classify, χ, `denoise()`, and default subtraction reject non-finite, negative, or non-integer values. Normalized or log-transformed data are rejected because rounding them would violate the no-count-inflation guarantee.

AnnData views are rejected by mutating functions. Use `.copy()` before calling AmbiDose on a slice.

## Droplet classification

`classify_droplets()` writes `obs["ambidose_droplet"]` and `obs["n_umi"]`.

Default mode:

```python
amdose.classify_droplets(
    adata,
    cell_barcodes=filtered_barcodes,
    empty_umi_max=100,
)
```

The matching Cell Ranger filtered barcodes are the default whitelist. If they are unavailable or known to be unreliable, the same `cell_barcodes` argument accepts barcode names exported by a trusted external caller such as CellBender or EmptyDrops. In either case, the AnnData count input must remain the matching Cell Ranger **raw** matrix; a cell-only corrected matrix cannot supply the empty-droplet pool.

Whitelisted barcodes are labeled `cell`. Non-cell barcodes with `0 < n_umi <= 100` are labeled `empty`; higher-count uncalled barcodes remain `other`. Only a trailing `-1` barcode suffix is normalized.

Additional cell-calling modes are available when no trusted whitelist is used:

- `empty_umi_max=100`: barcodes above the threshold are called cells. Use for smoke tests, not as a replacement for a real filtered list.
- `expected_cells=N`: the top-N UMI barcodes are cells and lower-count remaining droplets provide the empty pool.
- `call_cells()` / `denoise(cell_calling="diem")`: 3-component empty/debris/cell mixture; first inflection only if the mixture is inflated (`n_mix > 2.5 × n_inflection`). Below the cliff, keep barcodes closer to high-UMI cells than to χ. `cell_calling="off"` uses an external whitelist.

`mark_doublets()` optionally writes Scrublet predictions to
`obs["ambidose_doublet"]`. This diagnostic does not alter droplet labels,
dose estimates, or the cells retained by `analysis_ready()`.

## Ambient profile χ

```python
amdose.estimate_chi(adata, sample_key="sample")
```

For a single sample, $\chi$ is stored in `adata.var["ambidose_chi"]`. For multiple samples it is a pandas DataFrame in `adata.uns["ambidose_chi"]`, indexed by sample and aligned to `var_names`.

Every sample needs at least `min_empty` empty droplets. AmbiDose does not borrow or pool $\chi$ across samples. If genes are subset later, the retained profile is aligned and renormalized before dose estimation or subtraction.

### OCM/CMO and ambient-profile boundaries

OCM/CMO biological assignments from cells multiplexed in one GEM/library do not define separate ambient soups. Keep those assignments as ordinary `obs` metadata and use `sample_key=None` for that library. If several independently loaded GEMs/libraries are combined, use a GEM or library identifier as `sample_key`; each level must retain its own empty droplets. Do not use the within-GEM OCM biological sample label as `sample_key`.

AmbiDose operates on gene-expression counts. It does not correct OCM tags or perform OCM demultiplexing.

## Cell types

Grouping is an operational identity for dose and subtraction, not a cell-type atlas. AmbiDose does not name lineages and does not call annotation or language-model APIs.

The default estimator uses a broad `type_key`. `denoise()` resolves one in this order:

1. use the caller's `type_key` column, which must be independently established **broad** labels (major lineages from your own annotation pipeline);
2. otherwise run label-free Leiden clustering (resolution 0.08 / 0.2 / 0.35; 0.35 if scFair is unavailable) and store `obs["ambidose_cluster"]`. Libraries with at least 20,000 cells use a cheaper graph. Pass `typing_fast=False` or CLI `--full-typing` for the full-cell graph.

Do not pass a high-resolution atlas or a per-cell reassignment of cluster labels from ambient-contaminated marker scores. Fine fragments skip extra-clear below 10 cells and weaken exclusive gene ownership. Annotate cell types on the denoised counts after `denoise()`, or supply broad labels computed outside this package.

Dose is estimated per Leiden fragment. Dominant-gene ownership is computed **per sample** from that sample's groups: fragments whose whole profiles are indistinguishable from split noise form a meta-group and share identity-gene ownership. Concatenating libraries does not share an owner catalog across samples. Each sample still uses its own $\chi_s$.

## Dose estimation

```python
type_key = amdose.resolve_type_key(adata, type_key="cell_type")
dose = amdose.estimate_dose(adata, type_key=type_key, sample_key="sample")
```

The function writes `ambidose_rho`, `ambidose_d`, raw estimates, evidence counts, fallback flags, and shrinkage diagnostics. Empty and other droplets retain zero estimated dose. Their rows in `ambidose_denoised` are the original counts (passed through, not corrected). Doublet predictions are diagnostic and do not remove a cell from dose estimation.

Calling `estimate_dose()` without `type_key` intentionally selects the label-free quantile-floor ablation. It is not the default type-aware estimator.

## Subtraction

```python
amdose.subtract(adata, type_key=type_key, sample_key="sample")
```

The type-aware operator clears identified ambient-only genes at the type aggregate, protects native genes, and expands removal back to cells without exceeding observed counts. By default, cross-type protection requires both cross-cell expression structure and the cross-type anchor-ratio criterion; this protection affects subtraction but not dose estimation. Corrected counts are written to `layers["ambidose_denoised"]`; calling `subtract()` on its own (as shown above) leaves `X` unchanged. `denoise()` itself (the actual product entry point) takes one further step at the end: it sets `X` to these denoised counts and moves the original input into `layers["raw_counts"]`, for both calling conventions.

If a dose was estimated with a type key, pass the same key to `subtract()`. The function detects and rejects accidental degradation to the label-free operator.

## Re-running and stored state

`denoise()` skips classification when the droplet column already exists and skips ambient estimation when a valid $\chi$ is already stored. Remove stale AmbiDose columns yourself before intentionally recomputing from changed raw data; there is no compatibility or migration layer.

## Output checks

```python
import numpy as np
from scipy import sparse

raw = sparse.csr_matrix(adata.layers["raw_counts"])
den = sparse.csr_matrix(adata.layers["ambidose_denoised"])  # == adata.X after denoise()
assert np.issubdtype(den.dtype, np.integer)
assert den.min() >= 0
assert (den > raw).nnz == 0
```
