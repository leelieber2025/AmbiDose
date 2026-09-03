# Using AmbiDose from R

AmbiDose is distributed as a Python package. This tutorial runs the same workflow as {doc}`raw_to_denoised` from R.

Call the Python API through `reticulate`, or run `ambidose denoise` on the command line and read the output into Seurat. In either case the matching Cell Ranger **raw** matrix is required. Empty droplets are used to estimate $\chi$. A Seurat object of filtered cells is not a substitute.

## 0. Setup

Install AmbiDose in Python 3.10–3.13:

```bash
conda create -n ambidose python=3.12
conda activate ambidose
pip install ambidose
# or: conda install -c conda-forge -c bioconda ambidose
ambidose --version
```

In R, install `reticulate`. Install Seurat if you will continue analysis there. `anndataR` is optional.

```r
install.packages("reticulate")
# install.packages("Seurat")
# BiocManager::install("anndataR")
```

Select the Python environment before importing modules:

```r
library(reticulate)
use_condaenv("ambidose", required = TRUE)
# or: use_python("/path/to/envs/ambidose/bin/python", required = TRUE)
```

```r
amdose <- import("ambidose")
amdose$`__version__`
```

R `NULL` maps to Python `None`. Integer arguments that Python treats as `int` need the `L` suffix (`4L`). Use `$` for attributes and methods.

## 1. `reticulate`

Import Scanpy and AmbiDose, run `denoise`, then convert when you need an R object.

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

`raw=` is the matching unfiltered matrix. Barcodes on `adata` are the cell whitelist. The call returns a new AnnData; it does not modify a Seurat object.

A broad type column already stored in Python `obs` can be passed as `type_key`:

```r
adata <- amdose$denoise(
  adata,
  raw = RAW,
  sample_key = NULL,
  type_key = "cell_type"
)
```

`type_key` must be a column on the AnnData. Fine-grained atlases are unsuitable; see {doc}`../faq`.

```r
obs <- py_to_r(adata$obs)
head(obs[, c("ambidose_rho", "ambidose_d", "ambidose_rho_trust")])
table(obs$ambidose_rho_trust)
```

Write H5AD and continue in Seurat:

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

Tables can be pulled with `py_to_r(adata$obs)` and `py_to_r(adata$var)`. Prefer H5AD or MTX for the count matrix.

## 2. Command line

Point `--input` at a Cell Ranger `outs/` directory. `--cells-only` and `--output-format 10x-mtx` write denoised integer counts as an MTX folder.

```bash
ambidose denoise \
  --input /data/project/sample_01/outs \
  --cells-only \
  --output-format 10x-mtx \
  --output /data/project/sample_01/ambidose_mtx \
  --report /data/project/sample_01/ambidose_report.html
```

If `outs/` is not a single directory, pass the raw matrix and the filtered barcode list:

```bash
ambidose denoise \
  --input /data/project/sample_01/outs/raw_feature_bc_matrix.h5 \
  --cell-barcodes /data/project/sample_01/outs/filtered_feature_bc_matrix/barcodes.tsv.gz \
  --cells-only \
  --output-format 10x-mtx \
  --output /data/project/sample_01/ambidose_mtx
```

```r
library(Seurat)

counts <- Read10X("/data/project/sample_01/ambidose_mtx")
seu <- CreateSeuratObject(counts, assay = "RNA")
```

MTX holds cells-only denoised `X`. It does not store `layers["raw_counts"]` or per-cell `ambidose_rho`. Keep `--report`, or write H5AD, if those fields are needed.

```bash
ambidose denoise \
  --input /data/project/sample_01/outs \
  --cells-only \
  --output /data/project/sample_01/cleaned.h5ad \
  --report /data/project/sample_01/ambidose_report.html
```

```r
library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)

sce <- readH5AD("/data/project/sample_01/cleaned.h5ad")
seu <- as.Seurat(sce, counts = "X", data = NULL)
```

On a `--cells-only` H5AD, `X` is already denoised integer UMI. Normalize in Seurat after this step.

## 3. Existing Seurat object

Export filtered barcodes if that object is your cell call. AmbiDose still needs the raw library.

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

Use raw integer UMI. CellBender (or another caller) may supply barcode names; do not run AmbiDose on a corrected count matrix.

Broad lineage labels can be written onto AnnData `obs` in Python or joined after conversion. High-resolution `Idents` are unsuitable as `type_key`.

## 4. Outputs

| Python / H5AD | In Seurat after `as.Seurat` / `Read10X` |
|---|---|
| `X` (denoised integer UMI) | Assay counts |
| `layers["raw_counts"]` | Present in H5AD; omitted by 10x MTX |
| `obs["ambidose_rho"]` | Cell metadata if you used H5AD |
| `obs["ambidose_d"]` | Cell metadata if you used H5AD |
| `obs["ambidose_rho_trust"]` | Cell metadata if you used H5AD |
| `obs["ambidose_cluster"]` | Coarse Leiden groups when `type_key` is omitted; not cell-type names |
| HTML `--report` | Separate file |

Annotate, test differential expression, and run trajectories on the denoised counts.

## 5. Multiple libraries

```bash
ambidose denoise --root /data/cellranger_runs --cells-only --output cleaned.h5ad
```

Each library keeps its own empty droplets and its own $\chi$. See {doc}`../user_guide/cli` and {doc}`../user_guide/workflow`.

OCM/CMO assignments are cell metadata. Do not use the within-GEM sample tag as `--sample-key`.
