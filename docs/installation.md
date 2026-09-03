# Installation

## Requirements

- Python 3.10–3.13
- NumPy, SciPy, pandas, AnnData, h5py, scanpy 1.10+, igraph, and scFair
- Raw integer UMI counts for the standard denoising workflow

A GPU is not required.

## Install

```bash
pip install ambidose
# or
conda install -c conda-forge -c bioconda ambidose
```

`pip` installs scFair with AmbiDose. Optional comparison baselines that use scVI-tools are separate:

```bash
pip install "ambidose[baselines]"
```

R users: there is no AmbiDose R package. Install this Python package, then follow {doc}`tutorials/from_r` (`reticulate`, or the CLI plus Seurat).

## Check the install

```bash
ambidose --version
```

```python
import ambidose as amdose

print(amdose.__version__)
toy = amdose.datasets.make_toy(n_samples=1, n_empty=80, n_cells=30)
amdose.denoise(toy, sample_key=None, empty_umi_max=80, type_key="cell_type")
print(toy.layers["ambidose_denoised"].shape)
```

## Development install

```bash
git clone https://github.com/leelieber2025/AmbiDose.git
cd AmbiDose
pip install -e ".[dev]"
pytest
```

## Build the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```
