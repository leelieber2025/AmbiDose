"""Focused plots for AmbiDose diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _axis(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


def barcode_rank(adata, *, ax=None):
    """Plot descending raw UMI totals by barcode rank."""
    ax = _axis(ax)
    from .pp import raw_count_matrix

    totals = np.sort(np.asarray(raw_count_matrix(adata).sum(axis=1)).ravel())[::-1]
    ax.loglog(np.arange(1, len(totals) + 1), np.maximum(totals, 1))
    ax.set(xlabel="Barcode rank", ylabel="Raw UMI", title="Barcode rank")
    return ax


def ambient_profile(adata, *, top: int = 20, ax=None):
    """Plot the top single-library ambient genes."""
    ax = _axis(ax)
    if "ambidose_chi" not in adata.var:
        raise ValueError("single-library ambidose_chi missing")
    chi = np.asarray(adata.var["ambidose_chi"], dtype=float)
    names_arr = np.asarray(adata.var_names, dtype=str)
    idx = np.lexsort((names_arr, -chi))[:top][::-1]
    ax.barh(adata.var_names[idx].astype(str), chi[idx])
    ax.set(xlabel="Ambient fraction", title="Top ambient genes")
    return ax


def dose_distribution(adata, *, ax=None):
    """Plot estimated ambient fractions across cells."""
    ax = _axis(ax)
    if "ambidose_rho" not in adata.obs:
        raise ValueError("ambidose_rho missing")
    from .pp import require_run_keys

    key, _layer_out = require_run_keys(adata)
    mask = adata.obs[key].astype(str).to_numpy() == "cell"
    rho = np.asarray(adata.obs["ambidose_rho"], dtype=float)[mask]
    bins = np.linspace(0, 1, 41)
    ax.hist(rho, bins=bins)
    ax.set(xlabel="Contamination fraction (rho)", ylabel="Cells", title="Dose distribution")
    return ax


def gene_change(adata, gene: str, *, layer: str | None = None, ax=None):
    """Compare raw and corrected counts for one gene across cells."""
    ax = _axis(ax)
    if gene not in adata.var_names:
        raise KeyError(gene)
    if layer is None:
        from .pp import require_run_keys

        _key, layer = require_run_keys(adata)
    pos = int(adata.var_names.get_loc(gene))
    from .pp import raw_count_matrix

    raw_m = raw_count_matrix(adata)
    raw = np.asarray(raw_m[:, pos].toarray() if sparse.issparse(raw_m) else raw_m[:, pos]).ravel()
    den_x = adata.layers[layer]
    den = np.asarray(den_x[:, pos].toarray() if sparse.issparse(den_x) else den_x[:, pos]).ravel()
    ax.scatter(raw, den, s=5, alpha=0.4)
    limit = max(float(raw.max(initial=0)), 1.0)
    ax.plot([0, limit], [0, limit], color="black", linewidth=1)
    ax.set(xlabel="Raw count", ylabel="Corrected count", title=f"Change: {gene}")
    return ax


def doublet_diagnostic(adata, *, ax=None):
    """Scatter of UMI count vs rho, colored by heterotypic type-residual.

    Positive color leans "excess mass looks like another cell type's
    native program" (heterotypic doublet); negative leans "excess mass
    looks like ambient chi" (soup). Diagnostic only: mark_doublets() does
    not relabel droplets or change rho/dose from this score -- the reader
    decides.
    """
    ax = _axis(ax)
    if "ambidose_type_residual" not in adata.obs or "ambidose_rho" not in adata.obs:
        raise ValueError(
            "ambidose_type_residual/ambidose_rho missing; run "
            "mark_doublets(type_key=...) after denoise() first"
        )
    from .pp import require_run_keys

    key, _layer_out = require_run_keys(adata)
    residual = adata.obs["ambidose_type_residual"].to_numpy(dtype=float)
    mask = (adata.obs[key].astype(str).to_numpy() == "cell") & np.isfinite(residual)
    n_umi = np.asarray(adata.obs["n_umi"], dtype=float)[mask]
    rho = np.asarray(adata.obs["ambidose_rho"], dtype=float)[mask]
    res = residual[mask]
    lim = float(np.max(np.abs(res))) if res.size else 1.0
    lim = lim if lim > 0 else 1.0
    points = ax.scatter(
        np.maximum(n_umi, 1), rho, c=res, s=6, alpha=0.6, cmap="coolwarm", vmin=-lim, vmax=lim
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="UMI (n_umi)",
        ylabel="Contamination fraction (rho)",
        title="Doublet vs soup (type residual)",
    )
    import matplotlib.pyplot as plt

    plt.colorbar(points, ax=ax, label="type residual (other-type LLR - chi LLR)")
    return ax


def summary(adata):
    """Return a compact run summary figure (five panels if doublet QC was run)."""
    import matplotlib.pyplot as plt

    from .pp import require_run_keys

    has_doublet_qc = "ambidose_type_residual" in adata.obs
    ncols = 3 if has_doublet_qc else 2
    fig, axes = plt.subplots(2, ncols, figsize=(16 if has_doublet_qc else 11, 8))

    barcode_rank(adata, ax=axes[0, 0])
    dose_distribution(adata, ax=axes[0, 1])
    key, _layer_out = require_run_keys(adata)
    labels = adata.obs[key]
    counts = labels.astype(str).value_counts()
    axes[1, 0].bar([str(x) for x in counts.index], counts.to_numpy())
    axes[1, 0].set(title="Droplet classes", ylabel="Barcodes")
    if "ambidose_chi" in adata.var:
        ambient_profile(adata, top=15, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "Per-library ambient profiles stored in uns", ha="center")
        axes[1, 1].set_axis_off()
    if has_doublet_qc:
        doublet_diagnostic(adata, ax=axes[0, 2])
        if "ambidose_doublet_score" in adata.obs:
            axes[1, 2].hist(np.asarray(adata.obs["ambidose_doublet_score"], dtype=float), bins=40)
            axes[1, 2].set(xlabel="Scrublet doublet score", ylabel="Cells", title="Doublet score")
        else:
            axes[1, 2].set_axis_off()
    fig.tight_layout()
    return fig
