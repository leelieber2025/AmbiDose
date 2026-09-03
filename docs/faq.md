# FAQ

## Is there an R package?

No. Call the Python package from R with `reticulate`, or run `ambidose denoise` and read MTX or H5AD into Seurat. See {doc}`tutorials/from_r`. Estimating $\chi$ requires the matching raw matrix; a filtered Seurat object is not sufficient.

## Should I pass filtered barcodes?

Yes. In Python, load the Cell Ranger **filtered** matrix and pass the matching **raw** matrix as `raw=`. Barcodes on `adata` are the whitelist:

```python
adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
adata = amdose.denoise(adata, raw="raw_feature_bc_matrix.h5", sample_key=None)
```

On the CLI, a Cell Ranger `outs/` input pairs raw and filtered automatically. `cell_barcodes=` / `--cell-barcodes` is the lower-level form when you load the raw matrix yourself.

If that whitelist is unavailable or known to be unreliable, `cell_barcodes=` / `--cell-barcodes` accept barcode names from CellBender, EmptyDrops, or another caller. Pass names only and keep the matching Cell Ranger **raw** matrix as the count input so empty droplets remain available for soup estimation.

## How should I handle OCM/CMO multiplexing?

OCM/CMO does not by itself require a different cell whitelist. Use the matching Cell Ranger filtered barcodes by default and an external whitelist only when cell calling is unreliable.

For biological samples multiplexed in one GEM/library, their OCM assignments are ordinary cell metadata: use `sample_key=None` while estimating the shared ambient profile. When combining multiple independently loaded GEMs/libraries, use a GEM or library column as `sample_key`; do not use the within-GEM OCM sample assignment. AmbiDose neither corrects OCM tags nor performs demultiplexing.

## Can I use a filtered matrix alone?

Not to estimate $\chi$: filtered matrices normally discard the empty-droplet pool. You may denoise a cell-only AnnData only when it already carries a valid `ambidose_chi` estimated from the matching raw library.

## Which matrix should I use?

Raw integer UMI counts. Do not normalize, log-transform, scale, or select HVGs before denoising. The corrected matrix is written to a layer; perform standard Scanpy preprocessing afterward.

## Should I provide `type_key`?

Use a trusted **broad** annotation when you have one (major lineages, not a high-resolution atlas). Pass that column as `type_key=` / `--type-key`. Otherwise `denoise()` builds label-free Leiden groups (`obs["ambidose_cluster"]`) at a resolution chosen from scFair's structural population count. Those groups are an identity for dose estimation and gene protection, not cell-type names.

`estimate_dose()` by itself does not create labels: omitting `type_key` selects the quantile-floor ablation.

## Does AmbiDose annotate cell types?

No. It does not assign lineage names, does not run marker panels, and does not call external annotation or language-model APIs. Automatic Leiden clusters may split one biological type into several fragments; that is expected. Finer clustering is not a better setting for this method: small groups skip extra-clear, and ubiquitous genes lose a unique owner, which weakens protection.

Do cell-type annotation **after** denoising, on the corrected counts, with whatever tool you already use. If you already have independent broad labels, pass them in as `type_key` instead of (or on a second run after) the automatic groups. Do not feed raw, ambient-contaminated marker scores back as a per-cell override of those groups.

## Why does a sample need at least ten empty droplets?

A sample-specific simplex estimated from too few droplets is unstable. AmbiDose raises instead of silently borrowing soup from another library. Retain more raw droplets or estimate a valid matching profile upstream.

## Why was my AnnData view rejected?

Mutating an AnnData view can silently materialize a detached object. Call `.copy()` on the slice, then run AmbiDose.

## Does AmbiDose change `adata.X`?

Yes, by default: `denoise()` sets `X` to the denoised counts (also written to `layers["ambidose_denoised"]`) and moves the original input into `layers["raw_counts"]`. Calling `subtract()` on its own, without going through `denoise()`, does not touch `X` -- corrected counts land only in `layers["ambidose_denoised"]` in that case. Metadata and diagnostics are added to `obs`, `var`, and `uns` either way.

## Can corrected counts exceed raw counts?

No. The subtraction validates raw integer input, limits removal to observed counts, and writes nonnegative integers. `ambidose.metrics.n_inflated(raw, den)` is available for external result checks.

## Is this batch correction?

No. Every sample uses its own ambient profile, but AmbiDose does not align latent spaces or remove biological/technical batch structure beyond ambient RNA.

## Why are some `rho` estimates high?

Inspect `obs["ambidose_rho_trust"]` and the QC report (`write_report` / `--report`). A non-`ok` label means `ambidose_rho` should be interpreted cautiously; it is not a recommendation to discard the cell. `under_execution` means less than half of the $\chi$-direction dose $d_c$ was removed; `over_removal` means total removal exceeded $d_c$ by more than 5% (soupOnly extra-clear is allowed to do this) or removed more than half of the cell UMI total. $d_c$ bounds the rank-1 take along $\chi$, not the cell's total UMI removal. Sample-level counts and explanations are stored in `uns["ambidose"]["trust"]`. Broadly expressed genes remain a calibration limitation; see {doc}`user_guide/method`.

## Must I pass cell barcodes to `denoise()`?

Not in the usual Python call: `denoise(adata, raw=...)` uses `adata.obs_names` as the whitelist. You only need `cell_barcodes=` when you load the raw matrix yourself.

Without `raw=`, `cell_barcodes`, an existing whitelist, or an explicit cell-calling mode, Python `denoise()` raises. The CLI resolves that case to `cell_calling="diem"`. Pass `cell_calling="diem"` in Python to obtain the CLI behavior. DIEM builds a three-component empty/debris/cell mixture whitelist and applies the first inflection only when the mixture call is inflated relative to the rank-curve cliff. To retain a Cell Ranger or external whitelist without refinement:

```python
amdose.denoise(adata, cell_barcodes="filtered_barcodes.tsv", cell_calling="off")
```

On the CLI, use `--cell-calling off --cell-barcodes ...`, or point `--input` to a Cell Ranger `outs/` directory and add `--cell-calling off`. The `chi` mode refines a provided list against empty-droplet χ. Do not truncate the list to chip capacity unless that cap is part of the intended cell-calling protocol.

`ordmag` / `force` remain as Cell Ranger step 1 / `--force-cells`. Empty droplets for χ stay in the SoupX UMI≤100 band; barcodes that fail the caller are `other`, not soup.

A UMI cutoff without a whitelist is smoke-only and only runs when you pass `empty_umi_max` explicitly.

## How do I report the method?

State how cells were called, the empty-droplet range, whether labels were supplied or inferred, and that corrected integer counts came from `ambidose_denoised`. Cite AmbiDose using `CITATION.cff` and cite the external tools actually used.
