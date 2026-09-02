# Using AmbiDose from R

There is no AmbiDose R package. The Python package is the product. This page is the R counterpart of {doc}`raw_to_denoised`: same workflow, called from R the way [scvi-tools does in R](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/r/api_overview_in_R.html) — `reticulate` imports the Python API.

Two paths:

1. **CLI, then Seurat.** Run `ambidose denoise` in a shell, write a 10x MTX directory, `Read10X` in R. No `reticulate`.
2. **`reticulate`.** Call `ambidose.denoise` from R, then convert the AnnData to a Seurat object.

AmbiDose still needs the **raw** Cell Ranger matrix (empty droplets included). A Seurat object that only holds filtered cells is not enough to estimate $\chi$.

## 0. One-time setup

Install AmbiDose in a Python 3.10–3.13 environment (conda is convenient next to R):

```bash
conda create -n ambidose python=3.12
conda activate ambidose
pip install ambidose
ambidose --version
```

In R, install `reticulate` (and Seurat if you continue there). `anndataR` is optional; it is what the scvi-tools R tutorial uses to inspect AnnData.

```r
install.packages("reticulate")
# install.packages("Seurat")
# BiocManager::install("anndataR")  # optional
```

Point `reticulate` at that env **before** importing Python modules:

```r
library(reticulate)
use_condaenv("ambidose", required = TRUE)
# or: use_python("/path/to/envs/ambidose/bin/python", required = TRUE)
```

Check:

```r
amdose <- import("ambidose")
amdose$`__version__`
```

`reticulate` maps R `NULL` to Python `None`. Integer arguments that Python treats as ints need the `L` suffix (`4L`, not `4`). Use `$` for attributes and methods.

## 1. CLI path (no reticulate)

This matches a Cell Ranger `outs/` directory. `--cells-only` and `--output-format 10x-mtx` write denoised integer counts as a standard MTX folder that Seurat already knows.

```bash
ambidose denoise \
  --input /data/project/sample_01/outs \
  --cells-only \
  --output-format 10x-mtx \
  --output /data/project/sample_01/ambidose_mtx \
  --report /data/project/sample_01/ambidose_report.html
```

If `outs/` is not a single directory, pass the raw matrix and the filtered barcode list explicitly:

```bash
ambidose denoise \
  --input /data/project/sample_01/outs/raw_feature_bc_matrix.h5 \
  --cell-barcodes /data/project/sample_01/outs/filtered_feature_bc_matrix/barcodes.tsv.gz \
  --cells-only \
  --output-format 10x-mtx \
  --output /data/project/sample_01/ambidose_mtx
```

Then in R:

```r
library(Seurat)

counts <- Read10X("/data/project/sample_01/ambidose_mtx")
seu <- CreateSeuratObject(counts, assay = "RNA")
```

The MTX export is cells-only denoised `X`. It cannot store `layers["raw_counts"]` or per-cell `ambidose_rho`. Keep the HTML `--report` (and/or `--output` as `h5ad` in a second run) if you need those columns.

To keep ρ, dose, and trust labels, write H5AD as well:

```bash
ambidose denoise \
  --input /data/project/sample_01/outs \
  --cells-only \
  --output /data/project/sample_01/cleaned.h5ad \
  --report /data/project/sample_01/ambidose_report.html
```

Load that file in R with [zellkonverter](https://bioconductor.org/packages/zellkonverter) (Bioconductor) and convert to Seurat:

```r
library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)

sce <- readH5AD("/data/project/sample_01/cleaned.h5ad")
seu <- as.Seurat(sce, counts = "X", data = NULL)
```

`X` on a `--cells-only` H5AD is already denoised integer UMI. Do not log-normalize before AmbiDose; do it after, in Seurat, as usual.

## 2. reticulate path (same API as the Python tutorial)

This is the scvi-tools pattern: import Scanpy and AmbiDose, run the Python functions, convert results when you need them in R.

```r
library(reticulate)
use_condaenv("ambidose", required = TRUE)

sc <- import("scanpy")
amdose <- import("ambidose")

FILTERED <- "/data/project/sample_01/outs/filtered_feature_bc_matrix"
RAW <- "/data/project/sample_01/outs/raw_feature_bc_matrix.h5"

adata <- sc$read_10x_mtx(FILTERED)
adata <- amdose$denoise(adata, raw = RAW, sample_key = NULL)
adata
```

`raw=` is the matching empty-droplet matrix. `adata`'s barcodes are the cell whitelist. The call returns a **new** AnnData (the R variable is replaced); it does not mutate a Seurat object you already had.

If you already have a broad type column in Python `obs`, pass it:

```r
adata <- amdose$denoise(
  adata,
  raw = RAW,
  sample_key = NULL,
  type_key = "cell_type"
)
```

`type_key` must exist on the AnnData, not on a Seurat object AmbiDose cannot see. Fine atlases are the wrong grain; see {doc}`../faq`.

Inspect from R:

```r
obs <- py_to_r(adata$obs)
head(obs[, c("ambidose_rho", "ambidose_d", "ambidose_rho_trust")])
table(obs$ambidose_rho_trust)
```

Write H5AD and continue in Seurat (same conversion as the CLI path):

```r
adata$write_h5ad("/data/project/sample_01/cleaned.h5ad")
```

```r
library(zellkonverter)
library(Seurat)

sce <- readH5AD("/data/project/sample_01/cleaned.h5ad")
seu <- as.Seurat(sce, counts = "X", data = NULL)
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu)
seu <- ScaleData(seu)
seu <- RunPCA(seu)
```

Optional: keep the AnnData in Python and only pull tables into R (`py_to_r(adata$obs)`, `py_to_r(adata$var)`), the way the scvi-tools R tutorial pulls `model$history` and DE frames. Do not round-trip the sparse count matrix through `py_to_r` unless you have to; write H5AD or MTX instead.

## 3. You already have a Seurat object

Export the **filtered** barcodes if that object is your cell call, but still point AmbiDose at the **raw** library.

```r
write.table(
  colnames(seu),
  file = "filtered_barcodes.tsv",
  quote = FALSE, row.names = FALSE, col.names = FALSE
)
```

```bash
ambidose denoise \
  --input /data/project/sample_01/outs/raw_feature_bc_matrix.h5 \
  --cell-barcodes filtered_barcodes.tsv \
  --cells-only \
  --output-format 10x-mtx \
  --output ambidose_mtx
```

Do not pass log-normalized, scaled, or HVG-subset counts. Do not run AmbiDose on CellBender-corrected counts; an external caller may supply barcode names only.

If the Seurat object already carries a coarse lineage column you trust, write it onto the AnnData `obs` in Python (or join it after conversion). Passing a 50-cluster Seurat `Idents` as `type_key` is the wrong grain.

## 4. What lands where

| Python / H5AD | In Seurat after `as.Seurat` / `Read10X` |
|---|---|
| `X` (denoised integer UMI) | Assay counts |
| `layers["raw_counts"]` | Present in H5AD; dropped by 10x MTX |
| `obs["ambidose_rho"]` | Cell metadata if you used H5AD |
| `obs["ambidose_d"]` | Cell metadata if you used H5AD |
| `obs["ambidose_rho_trust"]` | Cell metadata if you used H5AD |
| `obs["ambidose_cluster"]` | Automatic coarse Leiden groups when no `type_key`; not cell-type names |
| HTML `--report` | Not inside the Seurat object; open in a browser |

Annotation, DE, and trajectory belong **after** this step, on the denoised counts, with the R tools you already use.

## 5. Multi-sample and CLI details

Several Cell Ranger runs under one root:

```bash
ambidose denoise --root /data/cellranger_runs --cells-only --output cleaned.h5ad
```

Each library keeps its own empty droplets and its own $\chi$. See {doc}`../user_guide/cli` and {doc}`../user_guide/workflow`.

OCM/CMO assignments are ordinary cell metadata. Do not use the within-GEM sample tag as `--sample-key`.
