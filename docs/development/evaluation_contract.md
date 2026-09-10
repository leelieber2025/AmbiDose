# Evaluation contract

## Product protocol (0.5.1)

- Dose step is always `estimate_dose_adaptive`. Do not score
  `estimate_dose` as the product path.
- Fetal and kidney scores use the full cell population (no subsample).
  Fetal still excludes `F35_liver_CD45pos_FCAImmP7462238`.
- Bind every reported number to the package version and the `src/ambidose`
  hashes written by the eval job.
- Must-win barnyards first (Mixture, hgmm12k, GSE147203). If those fail,
  do not treat tissue leak/retention as a promotion signal.
- Runner: `scripts/manuscript/eval_0p36_datasets.py` (`--job barnyard`,
  `kidney`, `fetal`, plus `gse218853` / `synthetic` when needed).
  Parallel: `--job all --workers 4`. Release wrapper:
  `scripts/release_check.py --eval --workers 4`.
  Output under `data/processed/eval_<version>_<YYYYMMDD>/` (override with
  `AMBIDOSE_EVAL_OUT`).

## Fetal-liver comparator cohort

All fetal-liver analyses use the same 39-sample cohort, including AmbiDose-only summaries, matched-subset comparisons, Table 4, absolute-retention metrics, and paired per-sample tests. The fixed
excluded sample is `F35_liver_CD45pos_FCAImmP7462238`.

The excluded sample must not be included in any formal full-population or comparator result.
