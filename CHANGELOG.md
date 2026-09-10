# Changelog

All notable changes to this project are documented in this file.

## [0.5.1] - 2026-09-10


### Changed

- Product dose path is `estimate_dose_adaptive`. Executed ρ uses a
  sample-level unlabeled scale `s(q)`, `q = median(ρ̂) n̄ / λ_e` from
  empty-droplet mean UMI (replaces a global 0.704). The map shrinks at
  low q (floor 0.50) and expands at high q (cap 1.25). Shrink is blended
  out as median selected ρ̂ approaches 0 so clean libraries are not
  treated like fat-empty high-contamination runs. `estimate_chi` stores
  `λ_e` in `uns["ambidose"]["empty_umi"]`; missing empties raise.
- Default subtraction is dual-channel: rank-1 along χ under type
  budgets, plus soupOnly extra-clear of unexpressed unowned genes that
  may exceed `d_c`. Unsaturated rank-1 is spent inside cells; saturated
  gene columns stay type-level; protected genes are allocated by library
  size. High-χ U extra-clear uses an 80% χ-mass prefix. Pearson
  reweighting reallocates unowned rank-1 without shrinking the type
  budget.
- Ownership: gap-cascade with noise-aware folds; ambient-ceiling
  exceptions require cross-type specificity; high-χ U is revoked when a
  minority of cells in the type exceeds the type ceiling.
- `ambidose_rho_trust` is a run-state label, not a calibrated
  probability that a cell was correctly corrected. Count monotonicity
  (`0 ≤ corrected ≤ raw`) does not bound native-molecule loss.

## [0.3.3] - 2026-09-07

### Changed

- Ambient dose estimation for samples with more than one cell type now
  estimates each type's own contribution to the empty-droplet profile
  directly, instead of comparing it against a pooled profile of every
  other type. This removes spurious cross-type differences in estimated
  contamination, improves native signal retention, and improves
  specificity/precision on barnyard validation data with no loss of
  sensitivity.
- Gene-ownership decisions between cell types now account for sampling
  noise: a fold-change gap that only clears the ownership threshold by
  less than its own measurement uncertainty is no longer treated as a
  confident decision.
- High-χ genes (the smallest set of ambient genes covering 15% of the
  empty-droplet profile) use single-owner assignment; other genes keep
  shared ownership across related clusters. Prevents highly ambient genes
  such as hemoglobin from being co-owned by every fragment of a merged
  cluster.

### Fixed

- `denoise()`'s documentation still described the previous ambient
  dose-estimation method; updated to match the change above.

## [0.3.2] - 2026-09-04

### Changed

- Dominant-gene ownership now allows a marker gene to be shared by several
  related sub-population clusters ("gap cascade") instead of being awarded
  to a single global-argmax winner, better matching biology where markers
  are routinely shared across related sub-populations rather than owned
  exclusively by one.
- Leftover pooled-dose reallocation is now capped against what the
  previous single-winner ownership rule would have redistributed for the
  same type/gene, and down-weights genes whose observed level already
  exceeds the ambient ceiling. Both are defensive safeguards against
  over-correction on small or fragmented clusters; no change to must-win
  barnyard benchmarks.
- `write_10x_mtx()` and CLI `--output-format 10x-mtx` default to Cell Ranger
  v3 (`features.tsv.gz`). Pass `version=2` or `--mtx-version 2` for v2
  (`genes.tsv`). Input discovery prefers v3+ `raw_feature_bc_matrix` (and
  `.gz` barcodes) when both layouts are present.

## [0.3.1] - 2026-09-03

### Documentation

- Default Python call is `denoise(adata, raw=...)` after loading the filtered matrix.
- R tutorial leads with `reticulate`.
- Bioconda install, PyPI download badge, and Zenodo DOI `10.5281/zenodo.22278199`.

## [0.3.0] - 2026-09-02

First public release.

### Added

- `denoise()` as the product entry: empty-droplet χ, coarse Leiden types
  (scFair population-count routing at 0.08 / 0.2 / 0.35), frozen operational
  ρ, and type-aware subtraction to non-negative integer counts.
- Rank-1 leftover reallocation along χ onto genes that are neither
  native-protected nor soupOnly. soupOnly extra-clear of unowned unexpressed
  genes is unchanged.
- `denoise(adata, raw=...)` for a filtered cells-only AnnData plus the
  matching raw matrix. CLI with Cell Ranger `outs/`, `--root`, and
  `--manifest`; HTML report and 10x MTX export.
- Diagnostic plots (`ambidose.pl`) and QC report (`write_report`).
- Tutorials: Cell Ranger to denoised counts, reports on GSE147203, external
  cell whitelists, and using the Python API from R via `reticulate` or the CLI.

### Notes

- `rho` is an operational dose, not a calibrated contamination fraction.
  Inspect `obs["ambidose_rho_trust"]` and the QC report before treating it as
  a quantitative rate. Automatic Leiden groups are a coarse identity for
  subtraction, not cell-type names.
