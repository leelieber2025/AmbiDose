# AmbiDose Documentation

[![PyPI version](https://img.shields.io/pypi/v/ambidose.svg)](https://pypi.org/project/ambidose/)
[![PyPI downloads](https://img.shields.io/pepy/dt/ambidose.svg)](https://pepy.tech/project/ambidose)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/ambidose.svg)](https://anaconda.org/bioconda/ambidose)
[![Conda downloads](https://img.shields.io/conda/dn/bioconda/ambidose.svg)](https://anaconda.org/bioconda/ambidose)
[![Python versions](https://img.shields.io/pypi/pyversions/ambidose.svg)](https://pypi.org/project/ambidose/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/AmbiDose/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22278199.svg)](https://doi.org/10.5281/zenodo.22278199)

## What AmbiDose does

AmbiDose removes ambient RNA from droplet single-cell RNA-seq while keeping the count matrix suitable for ordinary count-based downstream analysis.

```text
raw droplets + filtered cell barcodes
                 │
                 ▼
       classify empty droplets
                 │
                 ▼
       estimate sample soup χ
                 │
                 ▼
   coarse types → per-cell ρ and d
                 │
                 ▼
     subtract rank-1 d_c χ_s
     (soupOnly extra-clear
      may exceed d_c)
```

Empty droplets determine the ambient composition $\chi_s$ for each sample. Each cell receives an operational scale $\rho_c$ and $\chi$-direction dose $d_c=\rho_c n_c$ (the rank-1 budget along $\chi_s$, not a cap on total UMI removal). The standard workflow (`denoise()`) writes non-negative integer counts to `adata.layers["ambidose_denoised"]` and, as its last step, also sets them as `adata.X` -- the original input moves to `adata.layers["raw_counts"]`.

AmbiDose estimates ambient RNA and subtracts it. It does not fit a count posterior, correct batch effects, or assign cell-type names. Automatic Leiden groups are an operational identity for dose and subtraction. The usual whitelist is the matching Cell Ranger filtered barcodes, refined against χ. An external barcode list is accepted when that call is missing or unreliable. The CLI default without a whitelist is DIEM. Python callers may request DIEM, EmptyDrops, OrdMag, or a fixed cell count. χ is always estimated from the matching raw droplet matrix.

## Where to go

| Goal | Page |
|---|---|
| Install and verify the package | {doc}`installation` |
| Run one complete analysis | {doc}`quickstart` |
| Understand inputs, outputs, and multi-sample behavior | {doc}`user_guide/workflow` |
| Understand the estimator and its limits | {doc}`user_guide/method` |
| Look up functions and parameters | {doc}`api/index` |
| Diagnose an error | {doc}`faq` |

### Default Python call

```python
import scanpy as sc
import ambidose as amdose

adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
adata = amdose.denoise(adata, raw="raw_feature_bc_matrix.h5", sample_key=None)

counts = adata.X  # denoised; original input is in layers["raw_counts"]
```

Barcodes on `adata` are the cell whitelist. `raw=` is the matching unfiltered matrix. The call returns a new object.

To load the raw matrix yourself and pass `cell_barcodes=`, see {doc}`quickstart`.

### Default CLI call

```bash
ambidose denoise \
  --input /path/to/cellranger/outs \
  --output cleaned.h5ad
```

Pointing `--input` at a Cell Ranger `outs/` directory automatically pairs the raw matrix with filtered barcodes.

:::{note}
The entry point is `denoise()`. Treat `rho` as an operational scale; see {doc}`user_guide/method`.
:::

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: installation
:link-type: doc

Requirements and optional extras.
:::

:::{grid-item-card} Quickstart {octicon}`rocket;1em;`
:link: quickstart
:link-type: doc

A first end-to-end run.
:::

:::{grid-item-card} Tutorial {octicon}`play;1em;`
:link: tutorials/raw_to_denoised
:link-type: doc

Raw Cell Ranger output to analysis-ready counts.
:::

:::{grid-item-card} User Guide {octicon}`book;1em;`
:link: user_guide/index
:link-type: doc

Inputs, labels, outputs, and CLI.
:::

:::{grid-item-card} Method {octicon}`beaker;1em;`
:link: user_guide/method
:link-type: doc

Model, estimator, and limitations.
:::

:::{grid-item-card} API Reference {octicon}`code;1em;`
:link: api/index
:link-type: doc

Public Python interface.
:::

:::{grid-item-card} FAQ {octicon}`question;1em;`
:link: faq
:link-type: doc

Common decisions and failures.
:::
::::

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

installation
quickstart
tutorials/index
user_guide/index
api/index
faq
citation
references
changelog
license
```
