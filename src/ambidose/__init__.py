"""AmbiDose public API.

Recommended usage::

    import scanpy as sc
    import ambidose as amdose

    adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
    adata = amdose.denoise(adata, raw="raw_feature_bc_matrix.h5", sample_key=None)

``denoise()`` is the product entry. ``classify_droplets``, ``estimate_chi``,
``estimate_dose``, and ``subtract`` are steps inside that path, not alternate
workflows.

``ambidose.metrics`` (ground-truth benchmarking: barnyard leakage, marker
leakage, entropy) is a separate evaluation module, not part of this product
surface -- import it directly (``from ambidose.metrics import ...``).
"""

from __future__ import annotations

from . import datasets, pl
from ._version import __version__
from .io import read_10x_barcodes, read_10x_h5, read_10x_mtx, write_h5ad
from .pp import (
    analysis_ready,
    call_cells,
    classify_droplets,
    denoise,
    estimate_chi,
    estimate_dose,
    mark_doublets,
    resolve_type_key,
    subtract,
)
from .reporting import inspect_input, summarize, write_report

__all__ = [
    "__version__",
    "analysis_ready",
    "call_cells",
    "classify_droplets",
    "datasets",
    "denoise",
    "estimate_chi",
    "estimate_dose",
    "inspect_input",
    "mark_doublets",
    "pl",
    "resolve_type_key",
    "read_10x_barcodes",
    "read_10x_h5",
    "read_10x_mtx",
    "subtract",
    "summarize",
    "write_report",
    "write_h5ad",
]
