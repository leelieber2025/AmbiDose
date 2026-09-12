"""Coarse Leiden typing used by denoise when type_key is omitted."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from anndata import AnnData

from ._budget import (
    _alloc_budget as _alloc_budget,
)
from ._budget import (
    _alloc_integer_budget as _alloc_integer_budget,
)
from ._budget import (
    _subtract_row as _subtract_row,
)
from ._dose import (
    _rho_from_chi as _rho_from_chi,
)
from ._dose import (
    _top_chi_indices as _top_chi_indices,
)
from ._dose import (
    _two_component_mixture_em as _two_component_mixture_em,
)
from ._dose import (
    diagnose_dose_disagreement as diagnose_dose_disagreement,
)
from ._dose import (
    estimate_dose as estimate_dose,
)
from ._dose import (
    estimate_dose_mixture as estimate_dose_mixture,
)
from ._dose import (
    q_abs_scale as q_abs_scale,
)
from ._droplets import _has_variable_gene
from ._ownership import (
    _complete_linkage_labels as _complete_linkage_labels,
)
from ._ownership import (
    _cross_cell_structure_mask as _cross_cell_structure_mask,
)
from ._ownership import (
    _type_means as _type_means,
)
from ._shared import (
    CLUSTER_KEY,
    DROPLET_KEY,
    EMPTY_TYPES,
    FINE_N_DENSITY_POPS,
    LEIDEN_RESOLUTION_COARSE,
    LEIDEN_RESOLUTION_FINE,
    LEIDEN_RESOLUTION_MEDIUM,
    MEDIUM_N_DENSITY_POPS,
    TYPING_FAST_N_CELLS,
    _record_cluster_diag,
    _sample_storage_id,
    _stable_subsample_indices,
    _validated_sample_values,
    _validated_type_values,
)
from ._shared import (
    SHRINK_K as SHRINK_K,
)
from ._shared import (
    TYPE_KEY as TYPE_KEY,
)
from ._shared import (
    _dense_chunk_columns as _dense_chunk_columns,
)
from ._shared import (
    _require_ram as _require_ram,
)
from ._shared import (
    _thread_worker_count as _thread_worker_count,
)


def _embed_coarse_hvg(sub: AnnData, *, typing_fast: bool) -> AnnData:
    """Normalize, HVG, PCA, neighbors on a cell-only copy. Mutates ``sub``."""
    import scanpy as sc

    if sub.n_vars < 3 or sub.n_obs < 3:
        raise ValueError("coarse embedding needs at least 3 cells and 3 genes")

    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    if not _has_variable_gene(sub.X):
        # No gene varies across any cell -- same "nothing to embed" family
        # as the n_vars/n_obs < 3 guard above. Scaling to unit variance
        # would divide by zero (NaN -> PCA raises "Input X contains NaN"),
        # and scanpy's HVG dispersion ranking raises on an empty finite-
        # dispersion subset before that. Raise the same error as the
        # too-few guard so callers (_resolve_coarse_resolution) fall back
        # to cluster_cells the same way.
        raise ValueError("coarse embedding needs at least 3 cells and 3 genes")
    n_hvg = min(2000, max(sub.n_vars - 1, 1))
    try:
        sc.pp.highly_variable_genes(sub, n_top_genes=n_hvg)
        hv = sub.var["highly_variable"].to_numpy()
    except IndexError:
        hv = np.zeros(sub.n_vars, dtype=bool)
    sub_hvg = sub[:, hv].copy() if hv.any() else sub
    if typing_fast:
        sc.pp.scale(sub_hvg, max_value=10, zero_center=False)
    else:
        sc.pp.scale(sub_hvg, max_value=10)
    n_comps = max(2, min(30, sub_hvg.n_vars - 1, sub_hvg.n_obs - 1))
    pca_kw = {"n_comps": n_comps}
    if typing_fast:
        pca_kw["zero_center"] = False
    sc.tl.pca(sub_hvg, **pca_kw)
    n_neighbors = min(15, sub_hvg.n_obs - 1)
    if typing_fast:
        try:
            sc.pp.neighbors(sub_hvg, n_neighbors=n_neighbors, transformer="pynndescent")
        except TypeError:
            sc.pp.neighbors(sub_hvg, n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(sub_hvg, n_neighbors=n_neighbors)
    return sub_hvg


def _resolve_coarse_resolution(
    adata: AnnData,
    *,
    droplet_key: str,
    cell_label: str,
    layer: str | None = None,
    return_prepared: bool = False,
    typing_fast: bool = True,
):
    """Choose 0.08, 0.2, or 0.35 from scFair's label-free structure count."""
    fallback = LEIDEN_RESOLUTION_FINE

    def done(resolution: float, prepared: AnnData | None = None):
        return (resolution, prepared) if return_prepared else resolution

    mask = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
    if int(mask.sum()) < 50:
        return done(fallback)
    try:
        sub = adata[mask].copy()
        if layer is not None:
            sub.X = sub.layers[layer]
        n_pops_from = sub.n_obs
        use_fast = bool(typing_fast) and sub.n_obs >= TYPING_FAST_N_CELLS
        if use_fast and sub.n_obs > TYPING_FAST_N_CELLS:
            pick = _stable_subsample_indices(sub.obs_names, TYPING_FAST_N_CELLS, seed=0)
            estimate_from = _embed_coarse_hvg(sub[pick].copy(), typing_fast=True)
            n_pops_from = TYPING_FAST_N_CELLS
        else:
            estimate_from = None
        prepared = _embed_coarse_hvg(sub, typing_fast=use_fast)
        if estimate_from is None:
            estimate_from = prepared
    except MemoryError as orig:
        raise MemoryError(
            "ambidose: label-free clustering ran out of memory. Use Cell Ranger "
            "filtered barcodes so empty droplets are not treated as cells, "
            "or run with more memory"
        ) from orig
    except (ValueError, np.linalg.LinAlgError) as orig:
        print(f"clustering prep skipped ({orig}); using cluster_cells", file=sys.stderr)
        _record_cluster_diag(adata, prep_fallback=True, prep_error=f"{type(orig).__name__}: {orig}")
        return done(fallback)

    import importlib.util

    if importlib.util.find_spec("scfair") is None:
        _record_cluster_diag(adata, scfair_fallback=True, scfair_error="scfair not installed")
        return done(fallback, prepared)
    n_pops = None
    try:
        import scfair

        n_pops = scfair.pp.estimate_n_populations(estimate_from).n_populations
        if n_pops is None:
            resolution = fallback
        elif n_pops >= FINE_N_DENSITY_POPS:
            resolution = LEIDEN_RESOLUTION_FINE
        elif n_pops >= MEDIUM_N_DENSITY_POPS:
            resolution = LEIDEN_RESOLUTION_MEDIUM
        else:
            resolution = LEIDEN_RESOLUTION_COARSE
    except MemoryError as orig:
        raise MemoryError("ambidose: structural population estimation ran out of memory") from orig
    except (ValueError, np.linalg.LinAlgError) as orig:
        _record_cluster_diag(
            adata, scfair_fallback=True, scfair_error=f"{type(orig).__name__}: {orig}"
        )
        resolution = fallback
    _record_cluster_diag(
        adata,
        typing_fast=bool(use_fast),
        n_pops_from=int(n_pops_from),
        n_pops=None if n_pops is None else int(n_pops),
        resolution=float(resolution),
    )
    return done(resolution, prepared)


def _annotate_coarse_types_single(
    adata: AnnData,
    *,
    droplet_key: str,
    cell_label: str,
    layer: str | None = None,
    typing_fast: bool = True,
) -> str:
    """Create label-free Leiden groups for dose estimation and subtraction."""
    import scanpy as sc

    from .baselines import cluster_cells

    resolution, prepared = _resolve_coarse_resolution(
        adata,
        droplet_key=droplet_key,
        cell_label=cell_label,
        layer=layer,
        return_prepared=True,
        typing_fast=typing_fast,
    )
    if prepared is not None:
        # Reuse the normalize/log1p/HVG/scale/PCA/neighbors work
        # _resolve_coarse_resolution already did to pick this resolution,
        # instead of cluster_cells() redoing all of it from scratch on the
        # same cells.
        sc.tl.leiden(
            prepared, resolution=resolution, flavor="igraph", n_iterations=2, random_state=0
        )
        is_cell_mask = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
        cluster_labels = np.full(adata.n_obs, None, dtype=object)
        cluster_labels[np.flatnonzero(is_cell_mask)] = prepared.obs["leiden"].astype(str).to_numpy()
        adata.obs[CLUSTER_KEY] = cluster_labels
    else:
        cluster_cells(
            adata,
            resolution=resolution,
            key_added=CLUSTER_KEY,
            droplet_key=droplet_key,
            cell_label=cell_label,
            n_top_genes=min(2000, max(adata.n_vars - 1, 2)),
            layer=layer,
        )

    return CLUSTER_KEY


def _type_sample_dependence(
    adata: AnnData,
    *,
    type_key: str,
    sample_key: str | None,
    droplet_key: str,
    cell_label: str,
    scope: str,
) -> None:
    """Record descriptive type-by-library dependence without changing results."""
    diag: dict[str, object] = {"scope": scope, "sample_key": sample_key}
    if sample_key is None:
        diag.update(n_samples=1, n_types=0, cramers_v=0.0, max_type_sample_fraction=1.0)
    else:
        is_cell = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
        samples = _validated_sample_values(adata, sample_key).astype(str)
        types = _validated_type_values(adata, type_key)
        valid_type = np.array([t not in EMPTY_TYPES for t in types], dtype=bool)
        included = is_cell & valid_type
        table = pd.crosstab(types[included], samples[included], dropna=False)
        values = table.to_numpy(dtype=np.float64)
        n_total = float(values.sum())
        expected = (
            values.sum(axis=1, keepdims=True) @ values.sum(axis=0, keepdims=True) / n_total
            if n_total
            else np.zeros_like(values)
        )
        chi2 = float(
            np.divide(
                (values - expected) ** 2,
                expected,
                out=np.zeros_like(values),
                where=expected > 0,
            ).sum()
        )
        dim = min(max(values.shape[0] - 1, 0), max(values.shape[1] - 1, 0))
        cramers_v = float(np.sqrt(chi2 / (n_total * dim))) if n_total and dim else 0.0
        row_totals = values.sum(axis=1)
        concentration = np.divide(
            values.max(axis=1),
            row_totals,
            out=np.ones_like(row_totals),
            where=row_totals > 0,
        )
        diag.update(
            n_samples=int(values.shape[1]),
            n_types=int(values.shape[0]),
            cramers_v=cramers_v,
            median_type_sample_fraction=float(np.median(concentration))
            if concentration.size
            else 1.0,
            max_type_sample_fraction=float(concentration.max()) if concentration.size else 1.0,
            contingency=table,
        )
    uns = dict(adata.uns.get("ambidose", {}))
    uns["type_sample_dependence"] = diag
    adata.uns["ambidose"] = uns


def _annotate_coarse_types(
    adata: AnnData,
    *,
    droplet_key: str,
    cell_label: str,
    sample_key: str | None,
    layer: str | None = None,
    typing_fast: bool = True,
) -> str:
    """Cluster each independent library separately and merge unique labels."""
    if sample_key is None:
        return _annotate_coarse_types_single(
            adata,
            droplet_key=droplet_key,
            cell_label=cell_label,
            layer=layer,
            typing_fast=typing_fast,
        )
    samples = _validated_sample_values(adata, sample_key).astype(str).to_numpy()
    unique_samples = list(pd.unique(samples))
    if len(unique_samples) == 1:
        return _annotate_coarse_types_single(
            adata,
            droplet_key=droplet_key,
            cell_label=cell_label,
            layer=layer,
            typing_fast=typing_fast,
        )
    labels = np.full(adata.n_obs, None, dtype=object)
    next_label = 0
    per_sample: dict[str, dict] = {}
    for sample in unique_samples:
        sample_mask = samples == sample
        sub = adata[sample_mask].copy()
        _annotate_coarse_types_single(
            sub,
            droplet_key=droplet_key,
            cell_label=cell_label,
            layer=layer,
            typing_fast=typing_fast,
        )
        sub_cells = sub.obs[droplet_key].astype(str).to_numpy() == cell_label
        local = sub.obs[CLUSTER_KEY].astype(str).to_numpy()
        mapped: dict[str, str] = {}
        for value in pd.unique(local[sub_cells]):
            mapped[str(value)] = str(next_label)
            next_label += 1
        global_idx = np.flatnonzero(sample_mask)
        for value, global_label in mapped.items():
            labels[global_idx[sub_cells & (local == value)]] = global_label
        per_sample[_sample_storage_id(sample)] = {
            "sample": sample,
            "n_cells": int(sub_cells.sum()),
            "n_clusters": int(len(mapped)),
            "clustering": dict(sub.uns.get("ambidose", {}).get("clustering", {})),
        }
    adata.obs[CLUSTER_KEY] = labels
    _record_cluster_diag(
        adata,
        scope="per_sample",
        n_samples=len(unique_samples),
        n_clusters=int(next_label),
        per_sample=per_sample,
    )
    return CLUSTER_KEY


def resolve_type_key(
    adata: AnnData,
    *,
    type_key: str | None = None,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    typing_fast: bool = True,
) -> str:
    """Return coarse types; automatic groups are resolved per library."""
    if type_key is not None:
        if type_key not in adata.obs.columns:
            raise KeyError(f"{type_key!r} missing on obs")
        resolved = type_key
        scope = "provided"
    else:
        resolved = _annotate_coarse_types(
            adata,
            sample_key=sample_key,
            droplet_key=droplet_key,
            cell_label=cell_label,
            layer=layer,
            typing_fast=typing_fast,
        )
        scope = (
            "per_sample"
            if sample_key is not None
            and _validated_sample_values(adata, sample_key).astype(str).nunique() > 1
            else "global"
        )
    _type_sample_dependence(
        adata,
        type_key=resolved,
        sample_key=sample_key,
        droplet_key=droplet_key,
        cell_label=cell_label,
        scope=scope,
    )
    return resolved


def _default_type_key(adata: AnnData) -> str | None:
    """Grouping column for a follow-up subtract/trust/doublet call."""
    stored = adata.uns.get("ambidose", {})
    if isinstance(stored, dict):
        used = stored.get("dose_type_key")
        if used in adata.obs.columns:
            return str(used)
    return None
