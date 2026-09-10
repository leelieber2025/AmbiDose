# AmbiDose algorithm (draft notes — not the product API)

Product is **0.5.1** (`src/ambidose/`). 0.3.7 is archived at
`archive/ambidose-0.3.7-pre-rewrite.zip` and `archive/ambidose-0.3.7-src/`.
0.4.0–0.4.3 was a simplification experiment and was reverted.
This document is a model sketch. Historical Pareto numbers are not constraints.

## Model

For cell \(c\) of type \(z_c\), library size \(n_c\):

\[
\mathbb{E}[y_{cg}] = (1-\rho_c)\, n_c\, \Phi_{z_c,g} + \rho_c\, n_c\, \chi_g,
\qquad d_c = \rho_c n_c.
\]

- \(\chi\): empty-droplet profile, one per sample.
- \(\rho_c\): χ-deconv mixture. Φ **updates on expressed genes** and is
  frozen at 0 on unowned / unexpressed genes (off-species on a barnyard).
- Unexpressed unowned genes: extra-clear, but only remaining \(d_c\)
  after rank-1. Without that channel, barnyard off-species is not covered
  by \(d\chi\) alone.

No PBMC scale factor. No second contamination spectrum.

## Subtract

Spend the mixture posterior, not \(\min(y, d\chi)\):

\[
r_{cg}=\frac{\rho_c\chi_g}{\rho_c\chi_g+(1-\rho_c)\Phi_{z,g}},
\qquad \mathrm{take}_{g}=\sum_c y_{cg}\, r_{cg}.
\]

Unexpressed genes have \(\Phi_g=0\), so \(r=1\) and off-species is
cleared without inflating \(\rho\). Type-level integer allocation uses
weights \(y\cdot r\). No soupOnly channel.

## Not in the model

Fixed quantile dose as the default (kept only for `type_key is None`
ablation and single-type libraries). Adaptive fixed-vs-mixture selection.
Global `0.704` ρ scale (replaced by sample-level `s(q)`). Gap-cascade ownership as a subtraction input. Cross-type
anchor. Row scaling back to \(d_c\).
