"""Evaluation helpers. Barnyard leakage is the empty-droplet gold standard."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .pp import DROPLET_KEY, _reject_view, require_run_keys

MIN_MARKERS_PRESENT = 3
COARSE_MARKERS = {
    "Tcell": ["CD3D", "CD3E", "CD3G", "CD2", "CD8A", "CD8B", "IL7R", "TRAC"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRB1"],
    "Bcell": ["MS4A1", "CD79A", "CD79B", "CD19", "IGHM", "IGHD"],
    "Myeloid": [
        "LYZ",
        "CD14",
        "CD68",
        "CST3",
        "AIF1",
        "FCGR3A",
        "C1QA",
        "C1QB",
        "C1QC",
        "S100A8",
        "S100A9",
    ],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7"],
    "Stromal": ["COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRB", "ACTA2"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "Erythroid": ["HBA1", "HBA2", "HBB", "GYPA", "ALAS2"],
}
MOUSE_COARSE_MARKERS = {
    "Tcell": ["Cd3d", "Cd3e", "Cd3g", "Cd8a"],
    "NK": ["Nkg7", "Klrd1", "Ncr1", "Gzma"],
    "Bcell": ["Ms4a1", "Cd79a", "Cd79b", "Cd19"],
    "Myeloid": ["Lyz2", "Cd68", "Csf1r", "C1qa", "C1qb", "Aif1"],
    "Endothelial": ["Pecam1", "Cdh5", "Vwf", "Cldn5"],
    "Stromal": ["Col1a1", "Col1a2", "Pdgfrb", "Acta2", "Dcn"],
    "Epithelial": ["Epcam", "Cdh1", "Krt8", "Krt18"],
    "Erythroid": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt"],
}


def _as_csr(x):
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(x)


def umi_by_genome(
    adata: AnnData,
    *,
    genome_key: str = "genome",
    layer: str | None = None,
) -> pd.DataFrame:
    if genome_key not in adata.var.columns:
        raise KeyError(f"{genome_key!r} missing on var")
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    g = adata.var[genome_key].astype(str).to_numpy()
    cols = {}
    for name in sorted(set(g)):
        cols[name] = np.asarray(x[:, g == name].sum(axis=1)).ravel().astype(np.float64)
    return pd.DataFrame(cols, index=adata.obs_names)


def assign_majority_genome(
    adata: AnnData,
    *,
    genome_key: str = "genome",
    layer: str | None = None,
    key_added: str = "ambidose_species",
) -> pd.DataFrame:
    """Label each barcode by the genome with the most UMIs."""
    _reject_view(adata, "assign_majority_genome")
    umi = umi_by_genome(adata, genome_key=genome_key, layer=layer)
    total = umi.sum(axis=1)
    species = umi.idxmax(axis=1).astype(object)
    species.loc[total == 0] = "unassigned"
    adata.obs[key_added] = species.astype(str)
    adata.obs[f"{key_added}_frac"] = (umi.max(axis=1) / total.clip(lower=1.0)).to_numpy()
    return umi


def leakage_by_species(
    adata: AnnData,
    *,
    genome_key: str = "genome",
    species_key: str = "ambidose_species",
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
) -> pd.DataFrame:
    """Off-genome UMI fraction among cell-labeled droplets of each species."""
    umi = umi_by_genome(adata, genome_key=genome_key, layer=layer)
    species = adata.obs[species_key].astype(str)
    key = droplet_key
    if key == DROPLET_KEY:
        with contextlib.suppress(KeyError):
            key, _ = require_run_keys(adata)
    if key in adata.obs:
        cells = adata.obs[key].astype(str).to_numpy() == cell_label
    elif isinstance(adata.uns.get("ambidose"), dict) and "droplet_key" in adata.uns["ambidose"]:
        raise KeyError(
            f"obs[{key!r}] missing after denoise(); leakage_by_species "
            "refuses to treat every barcode as a cell"
        )
    else:
        cells = np.ones(adata.n_obs, dtype=bool)
    rows = []
    for sp in sorted(umi.columns.astype(str)):
        mask = cells & (species.to_numpy() == sp)
        if not mask.any():
            continue
        tot = umi.loc[mask].sum(axis=1)
        own = umi.loc[mask, sp]
        leak = (tot - own) / tot.clip(lower=1.0)
        rows.append(
            {
                "species": sp,
                "n_cells": int(mask.sum()),
                "leakage_mean": float(leak.mean()),
                "leakage_median": float(leak.median()),
                "off_umi_mean": float((tot - own).mean()),
                "total_umi_mean": float(tot.mean()),
            }
        )
    return pd.DataFrame(rows)


def marker_leakage_table(
    x,
    gene_index: pd.Index,
    coarse_type: np.ndarray,
    types: list[str],
    markers: dict[str, list[str]],
) -> pd.DataFrame:
    """Mean marker-set CP10K of row-type markers among column-type cells.

    Values are ``count / library_size * 1e4``, not CPM (1e6). Diagonal is
    on-target. ``x`` is the full cell×gene matrix so library size uses the
    whole transcriptome; only marker columns are densified.
    """
    x = _as_csr(x)
    gene_index = pd.Index(gene_index.astype(str))
    if not gene_index.is_unique:
        raise ValueError("gene_index must be unique for marker leakage evaluation")
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    n[n == 0] = 1.0
    gene_pos = {g: i for i, g in enumerate(gene_index.astype(str))}
    coarse_type = np.asarray(coarse_type).astype(str)
    rows = []
    for src in types:
        present = [g for g in markers.get(src, []) if g in gene_pos]
        idx = [gene_pos[g] for g in present]
        if idx:
            marker_cp10k = np.asarray(x[:, idx].todense()) / n[:, None] * 1e4
            marker_mean_per_cell = marker_cp10k.mean(axis=1)
        else:
            marker_mean_per_cell = np.zeros(x.shape[0])
        row = {}
        for tgt in types:
            mask = coarse_type == tgt
            row[tgt] = float(marker_mean_per_cell[mask].mean()) if mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, index=types)


def summarize_marker_leakage(table: pd.DataFrame) -> dict:
    types = list(table.index)
    diag = np.array([table.loc[t, t] for t in types], dtype=np.float64)
    on_mean = float(np.mean(diag)) if diag.size else float("nan")
    if len(types) < 2:
        # A single type has no off-diagonal entries at all; np.fill_diagonal
        # on a 1x1 (or empty) matrix leaves nothing non-NaN, and
        # np.nanmean over an all-NaN array raises RuntimeWarning "Mean of
        # empty slice" for a value that's already known to be undefined.
        return {
            "on_target_mean": on_mean,
            "off_target_mean": float("nan"),
            "leak_ratio": float("nan"),
        }
    off = table.to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(off, np.nan)
    off_mean = float(np.nanmean(off)) if off.size else float("nan")
    return {
        "on_target_mean": on_mean,
        "off_target_mean": off_mean,
        "leak_ratio": float(off_mean / on_mean) if on_mean > 0 else float("nan"),
    }


def overcorrection_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-type on-target (leakage-table diagonal) for each method."""
    if not tables:
        return pd.DataFrame(columns=["type"])
    types = list(next(iter(tables.values())).index)
    rows = []
    for t in types:
        row: dict = {"type": t}
        for method, tab in tables.items():
            row[method] = (
                float(tab.loc[t, t]) if t in tab.index and t in tab.columns else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def n_inflated(
    raw: AnnData,
    den: AnnData,
    *,
    layer: str | None = None,
    eps: float = 1e-6,
) -> int:
    """Count gene×cell entries where corrected exceeds raw (count-integrity fail).

    Aligns by ``obs_names``/``var_names``, not position: a matching shape is
    not evidence of matching row order (e.g. CellBender's own output can be
    subset/reordered by ``barcode_indices_for_latents``), and a positional
    comparison over reordered-but-otherwise-identical data silently reports
    nonzero inflation that isn't real.
    """
    obs_r = raw.obs_names.astype(str)
    obs_d = den.obs_names.astype(str)
    if not obs_r.is_unique or not obs_d.is_unique:
        # .equals() compares positionally and returns True for two
        # identical-looking runs of duplicates (e.g. both all "x"),
        # silently falling through to the positional comparison this
        # function exists to avoid -- a missed obs_names_make_unique() or a
        # multi-sample concat should be a loud error here, not a coin flip.
        raise ValueError(
            "raw and/or den have duplicate obs_names; call "
            "obs_names_make_unique() before computing n_inflated"
        )
    if not obs_r.equals(obs_d):
        if set(obs_r) != set(obs_d):
            raise ValueError(
                "raw and den have different barcode sets; align them "
                "(e.g. via a shared `.obs_names.intersection(...)`) before "
                "computing n_inflated"
            )
        den = den[obs_r].copy()
    var_r = raw.var_names.astype(str)
    var_d = den.var_names.astype(str)
    if not var_r.is_unique or not var_d.is_unique:
        raise ValueError(
            "raw and/or den have duplicate var_names; use unique gene IDs "
            "before computing n_inflated"
        )
    if not var_r.equals(var_d):
        if set(var_r) != set(var_d):
            raise ValueError(
                "raw and den have different gene sets; align them before computing n_inflated"
            )
        den = den[:, var_r].copy()
    a = _as_csr(raw.X)
    b = _as_csr(den.layers[layer] if layer is not None else den.X)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = b.astype(np.float64) - a.astype(np.float64)
    d.data = np.maximum(d.data - eps, 0.0)
    d.eliminate_zeros()
    return int(d.nnz)


def barnyard_kill_row(
    raw: AnnData,
    den: AnnData,
    *,
    method: str,
    genome_key: str = "genome",
    species_key: str = "ambidose_species",
    layer: str | None = None,
) -> dict:
    """Ambient-removal / endogenous-retain / inflation for a barnyard pair."""
    umi_r = umi_by_genome(raw, genome_key=genome_key)
    umi_d = umi_by_genome(den, genome_key=genome_key, layer=layer)
    umi_d = umi_d.reindex(index=umi_r.index, columns=umi_r.columns).fillna(0.0)
    species = raw.obs[species_key].astype(str)
    off_raw = on_raw = off_den = on_den = 0.0
    for sp in umi_r.columns.astype(str):
        mask = species.to_numpy() == sp
        if not mask.any():
            continue
        tot_r = umi_r.loc[mask].sum(axis=1)
        own_r = umi_r.loc[mask, sp]
        tot_d = umi_d.loc[mask].sum(axis=1)
        own_d = umi_d.loc[mask, sp]
        off_raw += float((tot_r - own_r).sum())
        on_raw += float(own_r.sum())
        off_den += float((tot_d - own_d).sum())
        on_den += float(own_d.sum())
    # Degenerate denominators (off_raw/on_raw/removed <= 0) return nan, not
    # a best-possible score -- a species with 0 off-target UMI to begin
    # with tells you nothing about sensitivity, and returning 0.0/1.0 there
    # let a pure count-inflation method (every nonzero +1, den > raw
    # everywhere) score sensitivity/specificity/precision that looked
    # plausible instead of visibly undefined; downstream all()/comparisons
    # over nan fail loudly (as they should) instead of silently passing.
    sens = (off_raw - off_den) / off_raw if off_raw > 0 else float("nan")
    spec = on_den / on_raw if on_raw > 0 else float("nan")
    removed = (off_raw - off_den) + (on_raw - on_den)
    prec = (off_raw - off_den) / removed if removed > 0 else float("nan")
    # sens/spec/prec are proportions of real UMI moved/kept -- only
    # meaningful in [0,1]. A method that invents counts (writes above raw)
    # can push these outside that range (confirmed: a "+1 to every nonzero"
    # method scored sensitivity=-0.404, specificity=1.057, precision=1.000
    # before this clip); n_inflated is the right signal for "this method
    # invents counts," not a wildly out-of-range value on axes that are
    # supposed to read as percentages.
    if np.isfinite(sens):
        sens = float(np.clip(sens, 0.0, 1.0))
    if np.isfinite(spec):
        spec = float(np.clip(spec, 0.0, 1.0))
    if np.isfinite(prec):
        prec = float(np.clip(prec, 0.0, 1.0))
    return {
        "method": method,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ars": float(sens),
        "ers": float(spec),
        "precision": float(prec),
        "n_inflated": n_inflated(raw, den, layer=layer),
        "off_umi_mean": float(off_den / max(len(raw), 1)),
        "on_umi_mean": float(on_den / max(len(raw), 1)),
    }


def shannon_entropy(labels: np.ndarray, *, n_classes: int | None = None) -> float:
    """Shannon entropy; if ``n_classes`` is set, divide by ``log(n_classes)`` (0–1)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    if n_classes is None:
        return h
    # Normalize by the larger of the claimed and observed class counts: if
    # more classes actually appear in `labels` than `n_classes` claims (e.g.
    # an upstream NaN/"nan"-string category, or a subset that doesn't cover
    # every declared batch), log(n_classes) understates the max possible
    # entropy and h/log(n_classes) can exceed 1 -- silently breaking the
    # documented 0-1 range for every caller that averages this (e.g.
    # knn_batch_entropy).
    denom = max(n_classes, len(np.unique(labels)))
    if denom <= 1:
        return 0.0
    return float(h / np.log(denom))


def cross_batch_entropy(
    batch: np.ndarray,
    group: np.ndarray,
    *,
    skip: str = "-1",
) -> dict:
    """SoupX-style: size-weighted mean entropy of ``batch`` within each ``group``.

    Normalized to ``[0, 1]`` by ``log(n_batches)``. Higher means more mixing.
    """
    batch = np.asarray(batch).astype(str)
    group = np.asarray(group).astype(str)
    n_batches = len(set(batch))
    weights = []
    values = []
    per: dict[str, float] = {}
    for g in sorted(set(group)):
        if g == skip:
            continue
        mask = group == g
        h = shannon_entropy(batch[mask], n_classes=n_batches)
        per[g] = h
        values.append(h)
        weights.append(int(mask.sum()))
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mean = float(v @ w / w.sum()) if w.sum() > 0 else 0.0
    return {"mean": mean, "n_groups": len(per), "n_batches": n_batches, "per_group": per}


def knn_indices(distances, *, k: int) -> np.ndarray:
    """``(n, k)`` neighbor indices from a scanpy ``obsp['distances']`` matrix."""
    mat = distances.tocsr() if sparse.issparse(distances) else sparse.csr_matrix(distances)
    n = mat.shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        cols = mat.indices[start:end]
        data = mat.data[start:end]
        keep = cols != i
        cols = cols[keep]
        data = data[keep]
        if cols.size == 0:
            continue
        take = cols[np.argsort(data)[:k]]
        out[i, : take.size] = take
    return out


def knn_batch_entropy(batch: np.ndarray, knn: np.ndarray) -> float:
    """Mean normalized entropy of ``batch`` labels among each cell's k neighbors."""
    batch = np.asarray(batch).astype(str)
    knn = np.asarray(knn)
    n_batches = len(set(batch))
    hs = []
    for nbrs in knn:
        nbrs = np.asarray(nbrs)
        nbrs = nbrs[nbrs >= 0]
        if nbrs.size == 0:
            continue
        hs.append(shannon_entropy(batch[nbrs], n_classes=n_batches))
    return float(np.mean(hs)) if hs else 0.0
