# `uns['ambidose']`

`denoise()` writes this dictionary on the AnnData it returns. Older fields stay. The ones below say where χ came from and which subtraction path ran.

| Field | Written by | Meaning |
|---|---|---|
| `chi_provenance` | `estimate_chi` | `sample_key`, `layer`, `droplet_key`, `empty_label` used to build χ. |
| `empty_umi[<sample>]` | `estimate_chi`, refreshed by `subtract` | Per library: `lam_e` (mean empty UMI), `knee_umi` (empty-cloud barcode-rank knee), `n_empty`, `phi`. |
| `n_empty_consistent_skip_cells` | `subtract` | Cells whose type matched empty droplets. Extra-clear is off and rank-1 is capped at empty U-gene soup. |
| `n_low_soup_full_chi_cells` | `subtract` | Cells that were not empty-consistent, but soup per cell was still within 3× the empty-cloud knee. Rank-1 is a full χ take (`protect_scale=0`). |
| `n_tiny_protected_cells` | `subtract` | Cells in groups smaller than the minimum type size. Extra-clear is skipped. |
| `removal` | `subtract` | How much of the stored dose was actually removed, including execution-ratio percentiles. |
| `dose` | `estimate_dose_adaptive` | Which dose estimator ran, and per-sample `lam_e`, `q`, `rho_median`. |
| `cell_calling` | `denoise` | Method and how many barcodes were called, when AmbiDose called cells itself. |
| `postprocess_completed` | `denoise` | Summary and trust columns were written. |

If `n_low_soup_full_chi_cells` is 0, the library did not enter the low-soup full-χ path. A cells-only matrix uses `knee_umi` stored at `estimate_chi` time; it does not recompute the knee from rows that are no longer in the object.

Log lines go to the `ambidose` logger (stderr). Command output that is the result of the CLI (JSON, file paths, χ tables) stays on stdout.
