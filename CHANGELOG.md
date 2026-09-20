# Changelog

All notable changes to this project are documented in this file.

## [0.5.9] - 2026-09-20

### Changed

- For types whose unexpressed-unowned UMI mass, or estimated soup per
  cell, matches empty droplets (Poisson), extra-clear and leftover
  realloc are skipped and rank-1 is capped at empty U-gene soup instead
  of the full estimated dose. Removal stays low, not zero. Affected cells
  are counted in `uns["ambidose"]["n_empty_consistent_skip_cells"]`.

## [0.5.8] - 2026-09-20

### Fixed

- SoupOnly extra-clear no longer splits unexpressed-unowned (U) genes into
  a "high-χ" and "low-χ" group before capping them at the remaining
  `d_c`. That split was left over from 0.5.6's change to cover the full χ
  mass (`SOUP_ONLY_CHI_MASS = 1.0`): every U gene with nonzero χ already
  fell in the "high" group, so only a gene with exactly zero χ (an edge
  case) ever reached the "low" group, where it additionally received a
  dose-correlation reweighting no other U gene gets. All U genes now go
  through the same single capping path. No change for any gene with
  nonzero χ.

## [0.5.7] - 2026-09-19

### Changed

- `denoise()` and the CLI now use a Cell Ranger filtered barcode list as
  cells as-is by default (explicit `--cell-barcodes`, auto-detected
  `filtered_*`, or a manifest/root library). Previously that list was
  refined against ambient χ by default. Pass `cell_calling='chi'`
  (`--cell-calling chi`) to trim it against soup instead;
  `cell_calling='off'` remains equivalent to the new default.
  `'diem'` / `'emptydrops'` / `expect_cells` are unchanged and only apply
  when no filtered list is available.
- Error messages for missing empty droplets, an unmatched barcode list,
  and dose/χ or subtract/dose provenance mismatches now state the problem
  and what to pass next, instead of a single terse sentence. Exact
  message text changed; code matching on the old wording should match on
  the new wording instead.

## [0.5.6] - 2026-09-18

Several dose, ownership, and soupOnly thresholds that were fixed constants
fitted on a small number of evaluation panels are now computed from each
sample's own data, or dropped where the fixed value added nothing beyond
an existing bound.

### Changed

- `estimate_dose_adaptive()` no longer rescales the selected ρ by a fixed
  piecewise curve `s(q)`. Executed dose is the selected estimator directly.
  Per-sample `q = median(ρ̂) n̄ / λ_e` is still recorded as a diagnostic.
- The adaptive dose estimator switches from the fixed to the mixture
  estimate when their disagreement exceeds the Poisson sampling error of
  the two estimates, instead of a fixed fold/gap threshold.
- SoupOnly extra-clear now covers the full χ mass of unexpressed-unowned
  (U) genes, not a fixed 80% prefix, and is capped by default at the
  remaining `d_c` after rank-1 with no soft allowance beyond it
  (`high_u_remaining_multiplier` default is now `1.0`, was `1.10`).
- SoupOnly extra-clear eligibility is now a per-gene test (is this type's
  mean compatible with that type's own ρ, under a Poisson sampling
  margin) instead of a single fixed cross-type ratio cutoff.
- Exclusive gene ownership is now decided by comparing each candidate's
  conservative mean (point estimate minus its sampling error) against the
  runner-up, instead of requiring a fixed fold-change margin. Housekeeping
  ties stay unowned.
- High-χ unique-argmax ownership uses the Lorenz-curve knee of the
  sample's own χ to decide single-winner eligibility, instead of a fixed
  χ-mass threshold.
- A merged fragment now inherits its meta-group's ownership only when it
  is the sole member of that group or its own mean still exceeds every
  other meta-group's mean, instead of a fixed minimum-share threshold.
- Several dose defaults are now derived from the sample instead of fixed:
  the χ prefix used for evidence genes is the sample's own Lorenz knee,
  the ratio floor is calibrated so empty droplets land at a median ρ̂ of
  about 1, and the unexpressed-gene mean floor is the sample's own
  Poisson-expected value rather than a fixed UMI count.
- Dose log-ρ shrinkage now uses each sample's own median evidence
  (valid-gene count or exposure) as the shrinkage weight, instead of a
  fixed constant.
- The opt-in `relax_hk_when_soup_like` test now uses a Poisson ceiling on
  leftover native mass versus the empty-droplet χ, run whenever expected
  soup on that gene is at least 1 UMI, instead of two fixed thresholds.

### Removed

- The global soupOnly ρ floor: extra-clear eligibility is already bounded
  by the remaining `d_c` and by the per-gene type-ρ-compatibility test
  above, so the separate floor was redundant.
- Unused `REALLOC_CAP_FOLD`, `_topk_owner_masks`, and `OWNER_TOP_K` (dead
  code, not reachable from the product path).

## [0.5.5] - 2026-09-17

### Changed

- High-χ soupOnly extra-clear (the χ-mass-prefix-0.8 unexpressed-unowned
  genes) is capped by default at 1.10× the remaining `d_c` after rank-1,
  instead of unbounded. Configurable via `subtract()`/`denoise()`'s new
  `cap_high_u_to_remaining` (default `True`) and `high_u_remaining_multiplier`
  (default `1.10`). Low-χ U is unchanged (already capped at remaining `d_c`).
- `estimate_dose_adaptive()` (the `denoise()` default dose path) shrinks
  per-cell dose toward the sample median using each cell's ambient exposure
  (library size × ambient χ mass on unexpressed-unowned genes) rather than
  the count of positive-evidence genes. New `evidence_mode` parameter on
  `estimate_dose()`, `estimate_dose_adaptive()`, `subtract()`, and
  `denoise()`; `estimate_dose()`'s own default is unchanged
  (`"positive_genes"`), `estimate_dose_adaptive()`'s default is now
  `"exposure"`.

### Fixed

- A sample with no cells carrying a valid selected dose (for example, every
  droplet fell back to empty after refinement) no longer passes NaN into the
  unlabeled-scale calculation; it now uses the neutral scale (1.0).
- `estimate_dose_adaptive()` now rolls back `obs`/`uns` on any exception,
  matching `estimate_dose()`'s existing atomicity.
- `subtract()` called on its own (not via `denoise()`) with the default
  `dose="ambidose_dose"` and no explicit `droplet_key` now reuses the
  droplet grouping recorded when that dose was estimated, instead of
  resolving it independently.
- Cells scored through the mixture-dose fallback path now report QC
  fallback status consistent with `ambidose_mixture_status`, instead of
  carrying over the typed-MLE path's fallback flag.

### Added

- `subtract()` warns when `type_key` is given together with
  `clip_negative=False`, since that combination silently ignores `type_key`
  and uses the untyped continuous χ-direction path.

### Documentation

- `simulate_barnyard()`'s docstring notes it is a compact workflow/edge-case
  fixture, not a calibrated performance benchmark.

## [0.5.3] - 2026-09-14

### Changed

- Leftover rank-1 budget is evaluated jointly across protected and unprotected non-soupOnly genes within each library and type. Targets above the physical r_t = 1 ambient ceiling remain ineligible.
- Within-type gene response to cell-specific rho now limits both the priority and capacity of leftover allocation. Protected targets also require this response and are scaled by one minus native confidence; unsupported residual budget remains unspent instead of being forced onto weak evidence. No fixed protected/unprotected split or cross-library calibration is used.
- Protected-gene takes are allocated to cells by a Poisson noise-corrected blend of dose rank and the within-type soupOnly-anchor rate, falling back to dose rank when the anchor has no resolved signal.

## [0.5.2] - 2026-09-12

### Changed

- After high-χ single-winner ownership, a type no longer owns a gene
  when its `r_t = mean/(n̄χ)` is at most 1, if some other type in the
  same sample has `r_t > 1`. Types that still look ambient-level on a
  gene do not inherit protection when a clear expressor is present.
- SoupOnly extra-clear is not applied to a gene in the type that is the
  exclusive `r_t` argmax (fold over the runner-up). A type that uniquely
  leads on a gene is not extra-cleared on that gene. Libraries with a
  single usable type are unchanged.
- Unused rank-1 budget is not reallocated onto genes already above the
  ρ=1 ambient ceiling (`r_t > 1`).
- `estimate_chi` records empty-droplet NB2 overdispersion φ on
  `uns["ambidose"]["empty_umi"]`. It is not used at subtraction.

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
