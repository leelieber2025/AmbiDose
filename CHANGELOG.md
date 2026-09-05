# Changelog

All notable changes to this project are documented in this file.

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
