# AmbiDose

[![PyPI version](https://img.shields.io/pypi/v/ambidose.svg)](https://pypi.org/project/ambidose/)
[![PyPI downloads](https://img.shields.io/pepy/dt/ambidose.svg)](https://pepy.tech/project/ambidose)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/ambidose.svg)](https://anaconda.org/bioconda/ambidose)
[![Conda downloads](https://img.shields.io/conda/dn/bioconda/ambidose.svg)](https://anaconda.org/bioconda/ambidose)
[![Python versions](https://img.shields.io/pypi/pyversions/ambidose.svg)](https://pypi.org/project/ambidose/)
[![Documentation](https://readthedocs.org/projects/ambidose/badge/?version=latest)](https://ambidose.readthedocs.io/en/latest/)
[![CI](https://github.com/leelieber2025/AmbiDose/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/AmbiDose/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22278199.svg)](https://doi.org/10.5281/zenodo.22278199)

**AmbiDose** removes ambient RNA from droplet scRNA-seq. Empty droplets
estimate the soup profile χ; each cell gets an operational dose
`d_c = ρ_c · n_c`. Type-aware subtraction writes non-negative integer
counts. It does not integrate batches or annotate cell types.

Docs: [Read the Docs](https://ambidose.readthedocs.io/en/latest/).

## Install

```bash
pip install ambidose
# or: conda install -c conda-forge -c bioconda ambidose
```

Python 3.10–3.13. A GPU is not required. Details:
[installation](https://ambidose.readthedocs.io/en/latest/installation.html).

## First run

```python
import scanpy as sc
import ambidose as amdose

adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
adata = amdose.denoise(adata, raw="raw_feature_bc_matrix.h5", sample_key=None)
# adata.X is denoised integer UMI; original counts in layers["raw_counts"]
```

```bash
ambidose denoise --input /path/to/outs --output cleaned.h5ad
```

`rho` is an operational dose, not a calibrated contamination rate. Inspect
`obs["ambidose_rho_trust"]` and the QC report before using it quantitatively.

## Status

**0.3.0 (Alpha).** Import as `import ambidose as amdose`. The entry point is
`denoise()`. See the
[API reference](https://ambidose.readthedocs.io/en/latest/api/index.html).

## Next steps

1. [Installation](https://ambidose.readthedocs.io/en/latest/installation.html)
2. [Quickstart](https://ambidose.readthedocs.io/en/latest/quickstart.html)
3. [Tutorials](https://ambidose.readthedocs.io/en/latest/tutorials/index.html)
4. [FAQ](https://ambidose.readthedocs.io/en/latest/faq.html) if something looks off
5. [From R](https://ambidose.readthedocs.io/en/latest/tutorials/from_r.html) (`reticulate` or CLI)

## Citation

Pin the package version used in the analysis, for example `ambidose==0.3.0`.
Software record: [10.5281/zenodo.22278199](https://doi.org/10.5281/zenodo.22278199).
See `CITATION.cff` and the [citation page](https://ambidose.readthedocs.io/en/latest/citation.html).

```bibtex
@software{li2026ambidose,
  title   = {AmbiDose: per-cell ambient dose removal for droplet scRNA-seq},
  author  = {Li, Zhao},
  year    = {2026},
  version = {0.3.0},
  doi     = {10.5281/zenodo.22278199},
  url     = {https://doi.org/10.5281/zenodo.22278199},
}
```

## License

Software: [Apache License 2.0](LICENSE).

## Author

**Zhao Li (李钊)**
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)
