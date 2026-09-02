# Method

## Scope

AmbiDose is a count-removal operator for droplet scRNA-seq. Empty droplets identify ambient composition; broad cell populations identify genes unlikely to contain native expression; cell depth converts a contamination fraction into an absolute UMI dose.

Cell calling is an input-preparation step rather than the inferential target. AmbiDose is not a batch integration model or a posterior generative denoiser.

## Ambient profile

For empty droplet $e$ in sample $s$,

$$y_e \sim \operatorname{Poisson}(n_e\chi_s), \qquad \sum_g \chi_{s,g}=1.$$

The implementation estimates $\chi_s$ by summing counts over that sample's empty droplets and normalizing the gene totals to a simplex. Profiles are never shared across samples.

## Per-cell dose

For cell $c$ in sample $s$,

$$y_c \approx \text{native}_c + d_c\chi_s, \qquad d_c=\rho_c n_c.$$

Here $n_c$ is the cell's observed library size and $\rho_c\in[0,1]$ an operational scale. $d_c$ is the $\chi$-direction rank-1 budget: the mass allocated along $\chi_s$. It is not a calibrated contamination rate and not a cap on the cell's total UMI removal.

For a broad type $t$, AmbiDose compares the observed type mean with the ambient-only ceiling $\bar n_t\chi_s$, including a Poisson noise margin. Genes consistent with ambient-only expression form the evidence pool. Genes that look native in every type — mean above that ceiling everywhere, or a MALAT1-like collision ($r_t>0.85$ and a large library fraction in every type) — are dropped. True soup sits at or below the ceiling in every type and is kept; a cross-type fold filter is not used, because rank-1 ambient is uniform by construction. The fixed estimator uses a Poisson maximum-likelihood estimate when sufficient type-specific evidence is available and otherwise uses a conservative quantile estimate over high-$\chi$ genes. Raw log-$\rho$ values are shrunk toward the within-sample median according to the number of observed evidence genes.

The standard `denoise()` workflow also fits a two-component native-plus-ambient mixture. It retains the fixed estimate when the two estimators agree and selects the mixture estimate when their disagreement is large. For a sample with only one usable cell type, it uses the untyped $\chi$ quantile floor instead of a two-component fit. `estimate_dose()` exposes the fixed estimator used within this workflow; calling it directly is not an alternative product path.

## Type-aware subtraction

The elementary rank-one removal is

$$\tilde y_{c,g}=\max(y_{c,g}-d_c\chi_{s,g},0).$$

The default subtraction algorithm aggregates by type. Rank-1 take is `dose·χ·(1 − 0.9·native_confidence)` with `dose` equal to type-median $\rho$ times the group's library sum, never exceeding that χ-direction budget on those genes. Unused budget after clipping on protected genes is reallocated along χ to genes that are neither native-protected nor soupOnly. Per-cell $d_c$ is that budget's cell-level counterpart, not a per-gene weight. Unexpressed unowned genes are extra-cleared (`soupOnly`) only when the type's dose-weighted $\rho$ is at least 0.01 and `mean/(n̄χ)` is well below the ambient ceiling (below 0.4); that extra-clear is not scaled back to $d_c$. Genes with a unique owner, mitochondrial genes, genes that look native in every type, and genes sitting on the $\rho=1$ ambient ceiling (`mean/(n̄χ)>0.85`) are excluded from extra-clear.

The default cross-type protection is applied only during subtraction. A candidate gene must first show cross-cell expression structure associated with latent programs derived from confident-owner genes; it must then exceed the cross-type anchor ratio. This two-stage rule protects broadly distributed native signal without allowing a uniformly distributed contaminant to serve as its own evidence. Dose estimation is unchanged. Set `cross_type_anchor=False` only to reproduce the reference ablation used in method comparisons.

The final matrix is rounded to nonnegative `int32` counts. Removal only acts on existing sparse entries, so it cannot introduce expression at a previously zero gene and cannot write a count above raw.

## Why broad types matter

A high-$\chi$ gene can be native in one lineage and ambient in another. A single global gene list cannot distinguish those cases. Broad types supply the minimum biological structure needed for that distinction; fine annotation is not the objective and is not produced by AmbiDose. Automatic typing is label-free Leiden within each library. It does not name cell types and does not call external annotators. Groups with fewer than `MIN_TYPE_CELLS=10` cells skip type-level extra-clear and use the conservative rank-1 protected path, so results can change discontinuously at that threshold; the QC label `type_structure_risk` marks these small groups. A higher Leiden resolution is not a more accurate correction: it makes unique gene ownership harder and can increase over-removal of housekeeping-shaped genes. Pass `type_key` only for independently established broad labels.

## Known limitation

Genes with similar expression across all broad types, including many ribosomal, mitochondrial, and housekeeping genes, may lie near the $\rho=1$ ambient ceiling even when contamination is low. Their inclusion in the MLE evidence pool can produce an upward bias in $\rho$ at low true contamination. At high true contamination the same estimator can *under*-estimate $\rho$ and subtract far less than the true ambient mass; on toys the mapping from true $\rho$ to $\hat\rho$ is not monotone. An iterative estimator corrected the low-$\rho$ bias in a synthetic calibration experiment but reduced performance on real benchmark datasets and is therefore not used by `denoise()`. Later diagnostics (uniform-gene $r$ bimodality, alternative evidence-pool ranking, two-component kick of high-$r$ genes) either failed to separate soup from housekeeping on tissue or under-estimated $\rho$ the same way the iterative estimator did. There is currently no promoted fix.

Treat `ambidose_rho` as an operational scale and `ambidose_d` as the $\chi$-direction rank-1 budget. `denoise()` writes `obs["ambidose_rho_trust"]` (`ok`, `low_evidence`, `ceiling_risk`, `type_structure_risk`, `under_execution`, `over_removal`) without changing the estimator. `under_execution` means less than half of $d_c$ was removed. `over_removal` means total removal exceeded $d_c$ by more than 5% (expected when soupOnly extra-clear is active) or exceeded half of the cell total. Inspect the QC report and validate retention of biologically important genes for each new tissue.

## Assumptions

- Input values are raw integer UMIs.
- Empty droplets and cells come from the same library.
- Each multi-sample library has enough retained empty droplets.
- Broad labels describe populations with reasonably comparable biology.
- Ambient composition is rank-one within a sample.
