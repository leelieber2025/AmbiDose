"""Eval helpers (CellBender/scVI/SoupX-style dose). Not the product path.

``cluster_cells`` is only used by ``denoise()`` when coarse-clustering prep
fails (``prepared is None``). The product path clusters on the already
prepared PCA/neighbors object. Other callables belong to ``scripts/``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._dose import _dose_quantile_from_chi
from ._shared import (
    CHI_KEY,
    DOSE_KEY,
    DROPLET_KEY,
    LEIDEN_RESOLUTION_FINE,
    RHO_KEY,
    _reject_view,
    _sample_names,
)
from .io import write_10x_mtx
from .pp import _has_variable_gene


class MissingBaseline(RuntimeError):
    """A baseline tool is not installed or its output is missing."""


def find_cellbender() -> Path:
    env = os.environ.get("CELLBENDER")
    if env:
        p = Path(env)
        if p.exists():
            return p
    which = shutil.which("cellbender")
    if which:
        return Path(which)
    raise MissingBaseline(
        "cellbender not found. Install CellBender or set CELLBENDER=path/to/cellbender"
    )


def run_cellbender(
    mtx_or_h5: str | Path,
    output_h5: str | Path,
    *,
    expected_cells: int = 12000,
    total_droplets: int = 25000,
    epochs: int = 50,
    fpr: float = 0.01,
    cuda: bool = True,
    posterior_batch_size: int = 512,
    estimator: str = "map",
    cpu_threads: int = 8,
    estimator_multiple_cpu: bool = False,
) -> Path:
    """Run ``cellbender remove-background``. Returns the output ``.h5`` path.

    ``estimator_multiple_cpu``: pass ``--estimator-multiple-cpu`` (CellBender's
    own flag for parallelizing the MCKP posterior estimator across CPU
    threads) -- relevant for ``estimator="mckp"``, which is markedly slower
    than the default ``"map"`` and, on an 8GB card, OOMs well below
    ``posterior_batch_size=512`` (see CHANGELOG's "CellBender baseline
    fairness" entry; ``posterior_batch_size~32`` plus this flag is the
    known-working combination on this project's GPU).
    """
    binary = find_cellbender()
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "remove-background",
        "--input",
        str(mtx_or_h5),
        "--output",
        str(output_h5),
        "--expected-cells",
        str(expected_cells),
        "--total-droplets-included",
        str(total_droplets),
        "--epochs",
        str(epochs),
        "--fpr",
        str(fpr),
        "--posterior-batch-size",
        str(posterior_batch_size),
        "--estimator",
        estimator,
        "--cpu-threads",
        str(cpu_threads),
    ]
    if cuda:
        cmd.append("--cuda")
    if estimator_multiple_cpu:
        cmd.append("--estimator-multiple-cpu")
    # CellBender indexes `counts[total_droplets]`; that must be < n_barcodes.
    print("cellbender:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return output_h5


def read_cellbender_h5(path: str | Path) -> AnnData:
    """Load CellBender ``remove-background`` output without importing cellbender."""
    import h5py
    from scipy import sparse as sp

    with h5py.File(path, "r") as f:
        # Read only the `matrix/` group by explicit path, not a
        # visititems() basename flatten over the whole file: CellBender h5s
        # also have metadata/, global_latents/, droplet_latents/ groups, and
        # a flatten keyed by basename ("data", "shape", "barcodes", ...)
        # silently lets whichever group's dataset gets visited *last*
        # (h5py visits in alphabetical group order) win -- e.g.
        # metadata/... sorts after matrix/..., so a same-named dataset
        # there would silently overwrite the real matrix with no error.
        m = f["matrix"]
        d: dict[str, np.ndarray] = {
            "data": m["data"][()],
            "indices": m["indices"][()],
            "indptr": m["indptr"][()],
            "shape": m["shape"][()],
            "barcodes": m["barcodes"][()],
        }
        if "gene_names" in m:
            d["gene_names"] = m["gene_names"][()]
        elif "features" in m and "name" in m["features"]:
            d["gene_names"] = m["features"]["name"][()]
        else:
            d["gene_names"] = m["name"][()]
        latents_key = None
        if "barcode_indices_for_latents" in m:
            latents_key = "barcode_indices_for_latents"
        elif "barcodes_analyzed_inds" in m:
            latents_key = "barcodes_analyzed_inds"
        if latents_key is not None:
            d[latents_key] = m[latents_key][()]

    x = sp.csc_matrix((d.pop("data"), d.pop("indices"), d.pop("indptr")), shape=d.pop("shape"))
    x = x.T.tocsr()
    barcodes = d.pop("barcodes")
    if np.issubdtype(barcodes.dtype, np.bytes_):
        barcodes = barcodes.astype(str)
    gene_names = d.pop("gene_names")
    if np.issubdtype(gene_names.dtype, np.bytes_):
        gene_names = gene_names.astype(str)
    adata = AnnData(x)
    adata.obs_names = barcodes.astype(str)
    adata.var_names = gene_names.astype(str)
    for key in ("barcode_indices_for_latents", "barcodes_analyzed_inds"):
        if key not in d:
            continue
        idx = np.asarray(d[key])
        valid = (
            idx.size
            and np.issubdtype(idx.dtype, np.integer)
            and idx.min() >= 0
            and idx.max() < adata.n_obs
        )
        if valid:
            adata = adata[idx].copy()
        break
    return adata


def cluster_cells(
    adata: AnnData,
    *,
    n_top_genes: int = 2000,
    resolution: float = LEIDEN_RESOLUTION_FINE,
    key_added: str = "ambidose_cluster",
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
) -> AnnData:
    """Leiden clusters on cell-labeled droplets; labels written to ``adata.obs``."""
    import scanpy as sc

    _reject_view(adata, "cluster_cells")
    if droplet_key in adata.obs:
        mask = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
        ad = adata[mask].copy()
    else:
        ad = adata.copy()
        mask = np.ones(adata.n_obs, dtype=bool)
    if layer is not None:
        ad.X = ad.layers[layer]
    if ad.n_vars < 3 or ad.n_obs < 3:
        labels = np.full(adata.n_obs, None, dtype=object)
        labels[np.flatnonzero(mask)] = "0"
        adata.obs[key_added] = labels
        return adata
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    if not _has_variable_gene(ad.X):
        # No gene varies across any cell (a degenerate/constant input) --
        # same "nothing to cluster on" family as the n_vars/n_obs < 3
        # guard above. Scaling to unit variance would divide by zero
        # (NaN), and scanpy's HVG dispersion ranking raises on an empty
        # finite-dispersion subset before that -- skip straight to one
        # cluster rather than let either crash.
        labels = np.full(adata.n_obs, None, dtype=object)
        labels[np.flatnonzero(mask)] = "0"
        adata.obs[key_added] = labels
        return adata
    n_hvg = min(n_top_genes, max(ad.n_vars - 1, 1))
    try:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg)
        hv = ad.var["highly_variable"].to_numpy()
    except IndexError:
        # Belt-and-suspenders: the all-zero-variance case is already
        # short-circuited above, but scanpy's dispersion ranking can hit
        # the same empty-finite-dispersion-subset bug for other reasons
        # (e.g. NaN dispersions) -- fall back to "nothing is variable"
        # rather than propagate a raw IndexError from a plotting/QC
        # helper deep inside scanpy.
        hv = np.zeros(ad.n_vars, dtype=bool)
    ad_hvg = ad[:, hv].copy() if hv.any() else ad
    sc.pp.scale(ad_hvg, max_value=10)
    n_comps_ceiling = min(30, ad_hvg.n_vars - 1, ad_hvg.n_obs - 1)
    if n_comps_ceiling < 2:
        # PCA needs n_comps <= min(n_obs, n_vars) - 1; on a handful of
        # cells that ceiling can be 0 or 1, below the floor this used to
        # apply unconditionally (max(n_comps, 2) could push n_comps *above*
        # its own ceiling, e.g. n_obs=2 -> ceiling=1 -> floored to 2 ->
        # sklearn PCA raises). Too few cells for a meaningful embedding at
        # all -- same family as MIN_TYPE_CELLS/_dominant_owner_masks's
        # single-type guard -- so skip straight to one cluster.
        labels = np.full(adata.n_obs, None, dtype=object)
        labels[np.flatnonzero(mask)] = "0"
        adata.obs[key_added] = labels
        return adata
    n_comps = n_comps_ceiling
    sc.tl.pca(ad_hvg, n_comps=n_comps)
    sc.pp.neighbors(ad_hvg, n_neighbors=min(15, ad_hvg.n_obs - 1))
    sc.tl.leiden(ad_hvg, resolution=resolution, flavor="igraph", n_iterations=2, random_state=0)
    labels = np.full(adata.n_obs, None, dtype=object)
    labels[np.flatnonzero(mask)] = ad_hvg.obs["leiden"].astype(str).to_numpy()
    adata.obs[key_added] = labels
    return adata


def estimate_dose_global_rho(
    adata: AnnData,
    *,
    sample_key: str | None = None,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    **quantile_kwargs,
) -> np.ndarray:
    """One ρ per sample = median of the quantile-floor per-cell ρ; d_c = ρ_s · n_c."""
    _dose_quantile_from_chi(
        adata,
        sample_key=sample_key,
        droplet_key=droplet_key,
        cell_label=cell_label,
        **quantile_kwargs,
    )
    rho = np.asarray(adata.obs[RHO_KEY], dtype=np.float64)
    n = np.asarray(adata.obs["n_umi"], dtype=np.float64)
    is_cell = (
        adata.obs[droplet_key].astype(str).to_numpy() == cell_label
        if droplet_key in adata.obs
        else np.ones(adata.n_obs, dtype=bool)
    )
    samples = _sample_names(adata, sample_key)
    rho_s = np.zeros_like(rho)
    if samples is None:
        med = float(np.median(rho[is_cell])) if is_cell.any() else 0.0
        rho_s[is_cell] = med
    else:
        for s in pd.unique(samples):
            idx = is_cell & (samples == s)
            med = float(np.median(rho[idx])) if idx.any() else 0.0
            rho_s[idx] = med
    adata.obs[RHO_KEY] = rho_s
    adata.obs[DOSE_KEY] = rho_s * n
    return adata.obs[DOSE_KEY].to_numpy(dtype=np.float64)


def run_scar(
    adata: AnnData,
    *,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    max_epochs: int = 100,
    batch_size: int = 128,
) -> AnnData:
    """Denoise droplets with scAR (scvi-tools). UMI data uses NB, not ZINB.

    Expects ``var['ambidose_chi']``. Train on cell-labeled droplets only.
    """
    try:
        import torch
        from scvi.external import SCAR
    except ImportError as exc:
        raise MissingBaseline(
            "scAR needs scvi-tools (pip install scvi-tools[torch] or the [baselines] extra)"
        ) from exc
    if CHI_KEY not in adata.var.columns:
        raise KeyError(f"{CHI_KEY} missing; run estimate_chi first")
    if droplet_key not in adata.obs.columns:
        raise KeyError(f"{droplet_key!r} missing; run classify_droplets first")
    chi = np.asarray(adata.var[CHI_KEY], dtype=np.float32)
    is_cell = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
    cells = adata[is_cell].copy()
    SCAR.setup_anndata(cells)
    model = SCAR(cells, ambient_profile=chi, gene_likelihood="nb")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model.train(max_epochs=max_epochs, accelerator=accelerator, batch_size=batch_size)
    den = sparse.csr_matrix(np.asarray(model.get_denoised_counts(), dtype=np.float32))
    out = sparse.lil_matrix(adata.shape, dtype=np.float32)
    out[np.flatnonzero(is_cell)] = den
    adata.layers["scar_denoised"] = out.tocsr()
    return adata


def run_scvi(
    adata: AnnData,
    *,
    layer: str | None = None,
    batch_key: str | None = None,
    continuous_covariate_keys: list[str] | None = None,
    n_latent: int = 10,
    max_epochs: int = 50,
    batch_size: int = 128,
    key_added: str = "X_scvi",
) -> AnnData:
    """Train scVI on counts. Optional baseline, not part of AmbiDose."""
    try:
        import torch
        from scvi.model import SCVI
    except ImportError as exc:
        raise MissingBaseline(
            "scVI needs scvi-tools (pip install scvi-tools[torch] or the [baselines] extra)"
        ) from exc
    SCVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
        continuous_covariate_keys=continuous_covariate_keys,
    )
    model = SCVI(adata, n_latent=n_latent, gene_likelihood="nb")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model.train(max_epochs=max_epochs, accelerator=accelerator, batch_size=batch_size)
    adata.obsm[key_added] = model.get_latent_representation()
    return adata


__all__ = [
    "MissingBaseline",
    "cluster_cells",
    "estimate_dose_global_rho",
    "find_cellbender",
    "read_cellbender_h5",
    "run_cellbender",
    "run_scar",
    "run_scvi",
    "write_10x_mtx",
]
