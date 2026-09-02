# API Reference

The recommended namespace is:

```python
import ambidose as amdose
```

Only names exported by `ambidose.__all__` are the top-level public API. Evaluation helpers remain under `ambidose.metrics`; optional comparison wrappers remain under `ambidose.baselines`.

## Product workflow

```{eval-rst}
.. automodule:: ambidose
   :members: denoise, analysis_ready, summarize, inspect_input, write_report, classify_droplets, mark_doublets, estimate_chi, resolve_type_key, estimate_dose, subtract
   :member-order: bysource
```

## Input and output

```{eval-rst}
.. automodule:: ambidose.io
   :members: ResolvedInput, ManifestRow, read_manifest, sniff_input, list_10x_genomes, read_10x_h5, read_10x_mtx, read_10x_barcodes, find_filtered_barcodes, find_10x_mtx_samples, write_h5ad, write_10x_mtx
   :member-order: bysource
```

## Diagnostic plots

```{eval-rst}
.. automodule:: ambidose.pl
   :members: barcode_rank, ambient_profile, dose_distribution, gene_change, doublet_diagnostic, summary
   :member-order: bysource
```

## Synthetic datasets

```{eval-rst}
.. automodule:: ambidose.datasets
   :members: make_toy, make_barnyard_toy, scenario_housekeeping, scenario_zero_ambient
   :member-order: bysource
```

## Evaluation helpers

```{eval-rst}
.. automodule:: ambidose.metrics
   :members: umi_by_genome, assign_majority_genome, leakage_by_species, marker_leakage_table, summarize_marker_leakage, overcorrection_report, n_inflated, barnyard_kill_row, shannon_entropy, cross_batch_entropy, knn_indices, knn_batch_entropy
   :member-order: bysource
```

## Optional baselines

These functions support method comparison and are not used by the default AmbiDose estimator.

```{eval-rst}
.. automodule:: ambidose.baselines
   :members: MissingBaseline, run_cellbender, read_cellbender_h5, cluster_cells, estimate_dose_global_rho, run_scar, run_scvi
   :member-order: bysource
```
