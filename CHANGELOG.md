# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
