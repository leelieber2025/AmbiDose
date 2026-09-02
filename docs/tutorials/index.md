# Tutorials

These tutorials are task-oriented, complete workflows. Start with the raw-to-denoised walkthrough; it covers input discovery, validation, execution, quality control, and downstream use.

Use the external-whitelist tutorial only when the matching Cell Ranger filtered barcodes are unavailable or known to be unreliable.

There is no R package. {doc}`from_r` is the R counterpart of the Python walkthrough: CLI export to Seurat, or `reticulate` calling the same Python API.

```{toctree}
:maxdepth: 2

raw_to_denoised
from_r
external_cell_whitelists
reports_and_plots
```
