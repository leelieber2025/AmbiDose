"""Empty-droplet calling, ambient profile, and per-cell dose subtraction."""

from __future__ import annotations

import copy
import json
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._budget import (
    _alloc_budget as _alloc_budget,
)
from ._budget import (
    _alloc_integer_budget as _alloc_integer_budget,
)
from ._budget import (
    _confidence_weighted_take,
    _expand_take_to_cells,
    _integerize_corrected,
    _realloc_unspent_rank1,
    _selected_data_positions,
)
from ._budget import (
    _subtract_row as _subtract_row,
)
from ._dose import (
    SOUP_ONLY_RHO_FLOOR,
    _native_everywhere_mask,
    _type_residual_score,
    _unexpressed_mask,
    estimate_dose_adaptive,
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
from ._ownership import (
    _complete_linkage_labels as _complete_linkage_labels,
)
from ._ownership import (
    _cross_cell_structure_mask as _cross_cell_structure_mask,
)
from ._ownership import (
    _cross_type_anchor_mask,
    _dominant_owner_masks,
    _mt_gene_mask,
    _p_set_is_soup_like,
    _type_masks,
)
from ._ownership import (
    _type_means as _type_means,
)
from ._shared import (
    CHI_KEY,
    CLUSTER_KEY,
    DOSE_KEY,
    DROPLET_KEY,
    EMPTY_TYPE,
    EMPTY_TYPES,
    FINE_N_DENSITY_POPS,
    LAYER_OUT,
    LEIDEN_RESOLUTION_COARSE,
    LEIDEN_RESOLUTION_FINE,
    LEIDEN_RESOLUTION_MEDIUM,
    MEDIUM_N_DENSITY_POPS,
    MIN_GENES,
    MIN_TYPE_CELLS,
    MIX_INFLATION_RATIO,
    OVER_EXECUTION_RATIO,
    OVER_REMOVAL_FRACTION,
    RHO_KEY,
    RHO_TRUST_KEY,
    SAMPLE_KEY_DEFAULT,
    TRUST_CEILING,
    TRUST_LOW_EVIDENCE,
    TRUST_NOT_CELL,
    TRUST_OK,
    TRUST_OVER_REMOVAL,
    TRUST_TYPE,
    TRUST_UNDER_EXECUTION,
    TYPING_FAST_N_CELLS,
    UNDER_EXECUTION_RATIO,
    _as_csr,
    _chi_for_obs,
    _chi_vector,
    _configure_scanpy_n_jobs,
    _mc_worker_count,
    _profile,
    _record_cluster_diag,
    _reject_view,
    _require_raw_integer_counts,
    _resolve_cell_mask,
    _same_matrix,
    _sample_names,
    _sample_storage_id,
    _stable_count_order,
    _stable_subsample_indices,
    _StageProgress,
    _usable_cpu_count,
    _validate_chi_frame,
    _validate_output_layer,
    _validated_sample_values,
    _validated_type_values,
    raw_count_matrix,
    require_run_keys,
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


def _has_variable_gene(x, *, atol: float = 1e-10) -> bool:
    """Return whether any column varies without densifying sparse matrices."""
    if sparse.issparse(x):
        values = x.astype(np.float64, copy=False)
        mean = np.asarray(values.mean(axis=0)).ravel()
        squared = values.copy()
        squared.data **= 2
        variance = np.asarray(squared.mean(axis=0)).ravel() - mean**2
    else:
        variance = np.asarray(x, dtype=np.float64).var(axis=0)
    return bool(np.any(variance > atol))


def classify_droplets(
    adata: AnnData,
    *,
    empty_umi_max: int | None = None,
    empty_umi_min: int = 0,
    expected_cells: int | None = None,
    cell_barcodes: str | Path | list[str] | set[str] | None = None,
    other_barcodes: str | Path | list[str] | set[str] | None = None,
    layer: str | None = None,
    key_added: str = DROPLET_KEY,
) -> AnnData:
    """Label barcodes as ``empty``, ``cell``, or ``other``.

    SoupX-style: ``empty_umi_min < n_umi <= empty_umi_max`` are empty
    (default range matches SoupX ``soupRange = c(0, 100)`` when max is 100).
    Barcodes below ``empty_umi_min`` are ``other`` (likely junk).

    If ``cell_barcodes`` is set (Cell Ranger filtered list), those barcodes are
    cells; the empty UMI range applies only to the remainder. Do not use
    ``UMI > empty_umi_max`` as cells — that pulls in debris.

    If ``expected_cells`` is set instead, the top ``expected_cells`` barcodes
    by UMI are cells; barcodes below the 10th percentile of those cell UMIs
    among the remainder are empty (further capped at ``empty_umi_max`` if
    given, and floored at ``empty_umi_min`` -- without a cap, that 10th
    percentile can be in the thousands on a real dataset and pull debris
    into "empty"); the band in between is ``other``.

    ``other_barcodes`` identifies rejected caller candidates which must never
    be reused as empty droplets for estimating χ.
    """
    _reject_view(adata, "classify_droplets")
    if cell_barcodes is not None and expected_cells is not None:
        raise ValueError("pass only one of cell_barcodes or expected_cells")
    if expected_cells is not None and expected_cells < 1:
        raise ValueError("expected_cells must be at least 1")
    if empty_umi_min < 0:
        raise ValueError("empty_umi_min must be nonnegative")
    if empty_umi_max is not None and empty_umi_max < 0:
        raise ValueError("empty_umi_max must be nonnegative")
    if empty_umi_max is not None and empty_umi_min >= empty_umi_max:
        raise ValueError("empty_umi_min must be smaller than empty_umi_max")
    # empty_umi_max IS allowed alongside expected_cells (and always was
    # alongside cell_barcodes, see the cap= below): it's an optional extra
    # bound on what counts as "empty," not a competing cell-calling mode --
    # expected_cells alone still decides which barcodes are cells.
    if empty_umi_max is None and expected_cells is None and cell_barcodes is None:
        raise ValueError("pass empty_umi_max, expected_cells, or cell_barcodes")
    _require_raw_integer_counts(adata, layer=layer, fname="classify_droplets")
    x = adata.layers[layer] if layer is not None else adata.X
    totals = np.asarray(_as_csr(x).sum(axis=1)).ravel()
    label = np.array(["other"] * adata.n_obs, dtype=object)
    if isinstance(cell_barcodes, (str, Path)):
        from .io import read_10x_barcodes

        cell_barcodes = read_10x_barcodes(cell_barcodes)
    if cell_barcodes is not None:
        from .io import normalize_barcode

        _normalized_barcode_map(adata.obs_names, where="adata.obs_names")
        normalized_whitelist = _normalized_barcode_map(cell_barcodes, where="cell_barcodes")
        wanted = set(normalized_whitelist)
        is_cell = np.array([normalize_barcode(n) in wanted for n in adata.obs_names.astype(str)])
        n_matched = int(is_cell.sum())
        n_wanted = len(wanted)
        if n_matched != n_wanted:
            raise ValueError(
                f"{n_matched}/{n_wanted} cell_barcodes matched adata.obs_names; "
                "every whitelist barcode must be present after normalization"
            )
        label[is_cell] = "cell"
        cap = 100 if empty_umi_max is None else empty_umi_max
        empty = (~is_cell) & (totals > empty_umi_min) & (totals <= cap)
        if other_barcodes is not None:
            is_other = _cell_barcode_mask(adata, other_barcodes)
            if bool(np.any(is_cell & is_other)):
                raise ValueError("cell_barcodes and other_barcodes must be disjoint")
            empty &= ~is_other
        label[empty] = "empty"
    elif expected_cells is not None:
        if adata.n_obs < 2:
            raise ValueError("cannot infer an ambient profile from fewer than two barcodes")
        if expected_cells >= adata.n_obs:
            expected_cells = adata.n_obs - 1
        order = _stable_count_order(adata, totals)
        cell_idx = order[:expected_cells]
        label[cell_idx] = "cell"
        if cell_idx.size == 0:
            rest = order
            eligible = totals[rest] > empty_umi_min
            if empty_umi_max is not None:
                eligible &= totals[rest] <= empty_umi_max
            label[rest[eligible]] = "empty"
            label[rest[~eligible]] = "other"
        elif expected_cells < adata.n_obs:
            cell_floor = np.percentile(totals[cell_idx], 10)
            # cell_floor (10th percentile of the *called cells'* UMI) can be
            # in the thousands on a real dataset, in which case "< cell_floor"
            # alone pulls debris/low-quality droplets into "empty" -- the
            # same failure the docstring already warns against for treating
            # UMI > empty_umi_max as cells, just on the other end. Bound the
            # empty range with the same empty_umi_max/empty_umi_min knobs
            # the other two classification modes already use, when given.
            if empty_umi_max is not None:
                cell_floor = min(cell_floor, float(empty_umi_max))
            rest = order[expected_cells:]
            empty_mask = (totals[rest] < cell_floor) & (totals[rest] > empty_umi_min)
            label[rest[empty_mask]] = "empty"
            label[rest[~empty_mask]] = "other"
    else:
        label[(totals > empty_umi_min) & (totals <= empty_umi_max)] = "empty"
        label[totals > empty_umi_max] = "cell"

    # Rejected caller candidates can never become ambient references. Keep a
    # barcode called as a cell by the active mode, but force every other
    # rejected barcode out of the empty-droplet pool in all calling modes.
    if other_barcodes is not None:
        is_other = _cell_barcode_mask(adata, other_barcodes)
        label[is_other & (label != "cell")] = "other"

    adata.obs["n_umi"] = totals
    adata.obs[key_added] = label
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata


def _good_turing_proportions(counts: np.ndarray) -> np.ndarray:
    """Simple Good-Turing proportions for a pooled ambient count vector."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    n_genes = int(counts.size)
    if total <= 0 or n_genes == 0:
        return np.full(n_genes, 1.0 / max(n_genes, 1))
    p = counts / total
    n0 = int((counts == 0).sum())
    if n0 == 0:
        return p
    n1 = int((counts == 1).sum())
    p0 = (n1 / total) if n1 else 1.0 / (total + n_genes)
    p0 = min(max(p0, 0.0), 1.0 - 1e-12)
    pos = counts > 0
    mass_pos = float(p[pos].sum())
    out = np.zeros(n_genes, dtype=np.float64)
    if mass_pos > 0:
        out[pos] = p[pos] * (1.0 - p0) / mass_pos
    out[~pos] = p0 / n0
    out = np.clip(out, 0.0, None)
    s = float(out.sum())
    return out / s if s > 0 else np.full(n_genes, 1.0 / n_genes)


def _barcode_rank_curve(
    totals: np.ndarray, *, lower: float, exclude_from: int = 50, window: float = 1.0
) -> dict[str, float]:
    """Knee and inflection on the log-log barcode-rank curve.

    DropletUtils ``barcodeRanks`` (Lun): unique totals only, skip the top
    ``exclude_from`` ranks, slide a window of arc length ``window`` (log10
    units). Knee = strongest curvature (midpoint above the chord, shortest
    chord). Inflection = steepest (most negative) slope. Inflection is the
    cell/debris cut on a cliff-and-knee library; knee sits higher and is
    EmptyDrops' always-retain line, not a debris filter.
    """
    y_all = np.sort(totals[totals > lower])[::-1]
    if y_all.size < 50:
        inf = float(np.inf)
        return {"knee_umi": inf, "inflection_umi": inf, "n_knee": 0, "n_inflection": 0}
    y_all = y_all[: min(int(y_all.size), 80_000)]
    vals, lens = np.unique(y_all, return_counts=True)
    order = np.argsort(vals)[::-1]
    vals = vals[order].astype(np.float64)
    lens = lens[order].astype(np.float64)
    ranks = np.cumsum(lens) - (lens - 1.0) / 2.0
    keep = ranks > exclude_from
    if int(keep.sum()) < 4:
        knee = float(y_all[min(exclude_from, y_all.size - 1)])
        n_k = int((y_all >= knee).sum())
        return {
            "knee_umi": knee,
            "inflection_umi": knee,
            "n_knee": n_k,
            "n_inflection": n_k,
        }
    x = np.log10(ranks[keep])
    y = np.log10(np.clip(vals[keep], 1.0, None))
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    rhs = cum + window
    left = np.flatnonzero(rhs <= cum[-1])
    if left.size == 0:
        knee = float(vals[keep][-1])
        n_k = int((y_all >= knee).sum())
        return {
            "knee_umi": knee,
            "inflection_umi": knee,
            "n_knee": n_k,
            "n_inflection": n_k,
        }

    def _at(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(cum, target, side="right") - 1
        idx = np.clip(idx, 0, cum.size - 2)
        span = np.clip(step[idx], 1e-12, None)
        t = (target - cum[idx]) / span
        return x[idx] + t * (x[idx + 1] - x[idx]), y[idx] + t * (y[idx + 1] - y[idx])

    lx, ly = x[left], y[left]
    rx, ry = _at(rhs[left])
    mx, my = _at(cum[left] + window / 2.0)
    dx = rx - lx
    grad = (ry - ly) / np.clip(dx, 1e-12, None)
    intercept = ry - grad * rx
    above = my > grad * mx + intercept
    gap = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
    elbow = np.flatnonzero((~above) & (grad < -1.0))
    if elbow.size:
        first = int(elbow[0])
        maybe_knee = np.flatnonzero(above[: first + 1])
        before = int(np.searchsorted(lx, mx[first], side="right"))
        infl_i = int(np.argmin(grad[: max(before, 1)]))
    else:
        maybe_knee = np.flatnonzero(above)
        infl_i = int(np.argmin(grad))
    knee_i = int(maybe_knee[np.argmin(gap[maybe_knee])]) if maybe_knee.size else infl_i
    knee_umi = float(10.0 ** my[knee_i])
    infl_umi = float(10.0 ** my[infl_i])
    return {
        "knee_umi": knee_umi,
        "inflection_umi": infl_umi,
        "n_knee": int((y_all >= knee_umi).sum()),
        "n_inflection": int((y_all >= infl_umi).sum()),
    }


def _maybe_cut_inflated_mixture(
    keep_mask: np.ndarray, totals: np.ndarray, curve: dict[str, float]
) -> tuple[np.ndarray, bool]:
    """Intersect mixture with first inflection only if mixture looks inflated."""
    infl = curve["inflection_umi"]
    n_mix = int(keep_mask.sum())
    n_inf = int(curve["n_inflection"])
    if np.isfinite(infl) and n_inf >= 50 and n_mix > MIX_INFLATION_RATIO * n_inf:
        return keep_mask & (totals >= infl), True
    return keep_mask, False


def _closer_to_cells_than_chi(
    x,
    totals: np.ndarray,
    *,
    high: np.ndarray,
    below: np.ndarray,
    lower: int,
) -> np.ndarray:
    """True for ``below`` barcodes closer to the high-cell mean than to empty χ."""
    x = _as_csr(x)
    totals = np.asarray(totals, dtype=np.float64)
    high = np.asarray(high, dtype=bool)
    below = np.asarray(below, dtype=bool)
    empty = (totals > 0) & (totals <= lower) & (~high) & (~below)
    out = np.zeros(totals.size, dtype=bool)
    if int(empty.sum()) < 10 or int(high.sum()) < 10 or int(below.sum()) == 0:
        return out
    chi = np.asarray(x[empty].sum(axis=0)).ravel().astype(np.float64)
    hi = np.asarray(x[high].sum(axis=0)).ravel().astype(np.float64)
    chi_n = chi / (np.linalg.norm(chi) + 1e-12)
    hi_n = hi / (np.linalg.norm(hi) + 1e-12)
    xb = x[below].astype(np.float64)
    nrm = np.sqrt(np.asarray(xb.multiply(xb).sum(axis=1)).ravel())
    nrm = np.maximum(nrm, 1e-12)
    c_chi = np.asarray(xb.dot(chi_n)).ravel() / nrm
    c_hi = np.asarray(xb.dot(hi_n)).ravel() / nrm
    out[np.flatnonzero(below)[c_hi > c_chi]] = True
    return out


def _barcode_knee_umi(totals: np.ndarray, *, lower: float) -> float:
    """UMI at the barcode-rank knee (EmptyDrops always-retain)."""
    return _barcode_rank_curve(totals, lower=lower)["knee_umi"]


def _ordmag_count(totals: np.ndarray, n_expect: int) -> tuple[int, float]:
    """Cell Ranger OrdMag: n barcodes with UMI ≥ p99(top n_expect)/10."""
    y = np.sort(totals)[::-1]
    n_expect = int(max(1, min(n_expect, y.size)))
    m = float(np.percentile(y[:n_expect], 99)) if n_expect else 0.0
    thr = m / 10.0
    return int((y >= thr).sum()), thr


def _ordmag_auto_expect(totals: np.ndarray, *, hi: int = 45_000) -> dict[str, float]:
    """Cell Ranger ≥7 expect-cells: min_x (OrdMag(x)−x)²/x on x∈[2, hi]."""
    y = np.sort(totals)[::-1]
    hi = int(min(hi, max(y.size, 2)))
    grid = np.unique(np.clip(np.round(np.geomspace(2, hi, 80)).astype(np.int64), 2, hi))
    best_x, best_loss, best_n, best_thr = 2, float("inf"), 0, 0.0
    for x in grid:
        n, thr = _ordmag_count(y, int(x))
        loss = (n - x) ** 2 / max(x, 1)
        if loss < best_loss:
            best_x, best_loss, best_n, best_thr = int(x), float(loss), n, thr
    return {
        "expect_cells": float(best_x),
        "n_ordmag": float(best_n),
        "umi_threshold": float(best_thr),
        "loss": float(best_loss),
    }


def _csr_multinomial_logpmf(x, n: np.ndarray, logp: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    x = x.tocsr()
    n = np.asarray(n, dtype=np.float64)
    out = gammaln(n + 1.0)
    if x.data.size == 0:
        return out
    contrib = x.data * logp[x.indices] - gammaln(x.data + 1.0)
    starts = x.indptr[:-1]
    widths = np.diff(x.indptr)
    nz = widths > 0
    reduced = np.add.reduceat(contrib, starts[nz])
    out[np.flatnonzero(nz)] += reduced
    return out


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _split_cell_profiles(xc_hi, *, min_group: int = 10) -> list[np.ndarray]:
    """One Good-Turing profile for the high-UMI cell seed, or two if justified.

    A single pooled multinomial averages two genuinely different real cell
    types into a profile that fits neither well. When that happens, the
    minority type never wins the cell/debris likelihood comparison for its
    own barcodes outside the high-UMI band and gets absorbed into the
    debris component (see ``_diem_keep``'s docstring). Try a 2-way cosine
    k-means split of the high-UMI rows and keep it only when both groups
    clear ``min_group`` and the split fits the data materially better than
    the pooled profile (log-likelihood minus a BIC-style penalty for the
    extra free profile) -- otherwise collapse back to one profile, so a
    genuinely homogeneous high-UMI band is untouched.
    """
    n = xc_hi.shape[0]
    n_genes = xc_hi.shape[1]
    pooled = _good_turing_proportions(np.asarray(xc_hi.sum(axis=0)).ravel())
    if n < 2 * min_group or n * n_genes > 5_000_000:
        return [pooled]
    dense = np.asarray(xc_hi.todense(), dtype=np.float64)
    row_sum = dense.sum(axis=1)
    row_sum_safe = np.where(row_sum > 0, row_sum, 1.0)
    normed = dense / row_sum_safe[:, None]
    from scipy.cluster.vq import kmeans2

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _centroids, labels = kmeans2(normed, 2, seed=0, minit="++")
    except (ValueError, np.linalg.LinAlgError):
        return [pooled]
    counts = np.bincount(labels, minlength=2)
    if counts.min() < min_group:
        return [pooled]
    profiles = [
        _good_turing_proportions(np.asarray(xc_hi[labels == k].sum(axis=0)).ravel())
        for k in range(2)
    ]
    logp_pooled = np.log(np.clip(pooled, 1e-300, None))
    ll_pooled = float(_csr_multinomial_logpmf(xc_hi, row_sum, logp_pooled).sum())
    ll_split = 0.0
    for k in range(2):
        idx = labels == k
        logp_k = np.log(np.clip(profiles[k], 1e-300, None))
        ll_split += float(_csr_multinomial_logpmf(xc_hi[idx], row_sum[idx], logp_k).sum())
    penalty = 0.5 * n_genes * np.log(max(n, 2))
    if ll_split - penalty <= ll_pooled:
        return [pooled]
    return profiles


def _diem_keep(
    x,
    totals: np.ndarray,
    *,
    lower: int = 100,
    n_hvg: int = 2000,
    max_iter: int = 20,
) -> np.ndarray:
    """True for barcodes a mixture of empty / debris / cell(s) calls as cells.

    Empty / debris / cell multinomials. UMI ≤ ``lower`` fix the empty
    profile and are never cells. High-UMI candidates seed cells (as up to
    two sub-profiles, see :func:`_split_cell_profiles` -- a single pooled
    cell profile can't represent two markedly different real cell types,
    and whichever type it under-weights loses every likelihood comparison
    to the debris component and gets dropped); mid-UMI candidates seed
    debris. Empty χ stays pooled from the empty band so debris can differ
    from ambient. Keep if the best-fitting cell sub-profile's posterior is
    the largest and ≥ 0.5. CPU-only.
    """
    if n_hvg < 1:
        raise ValueError("n_hvg must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    x = _as_csr(x)
    totals = np.asarray(totals, dtype=np.float64)
    empty = totals <= lower
    cand = totals > lower
    if int(empty.sum()) < 10:
        raise ValueError(f"DIEM needs at least 10 barcodes with UMI <= {lower} for the empty pool")
    if int(cand.sum()) < 10:
        return cand.copy()
    gsum = np.asarray(x.sum(axis=0)).ravel()
    if gsum.size <= n_hvg:
        gidx = np.arange(gsum.size)
    else:
        cutoff = np.partition(gsum, gsum.size - n_hvg)[gsum.size - n_hvg]
        gidx = np.flatnonzero(gsum >= cutoff)
    xe = x[empty][:, gidx]
    xc = x[cand][:, gidx]
    n_c = np.asarray(xc.sum(axis=1)).ravel().astype(np.float64)
    ct = totals[cand]
    p90 = float(np.percentile(ct, 90))
    hi = ct >= p90
    infl = _barcode_rank_curve(totals, lower=lower)["inflection_umi"]
    mid = (~hi) & np.isfinite(infl) & (ct < infl)
    p_e = _good_turing_proportions(np.asarray(xe.sum(axis=0)).ravel())
    cell_profiles = _split_cell_profiles(xc[hi])
    if int(mid.sum()) >= 10:
        p_d = _good_turing_proportions(np.asarray(xc[mid].sum(axis=0)).ravel())
    else:
        p_d = p_e.copy()
    logp_e = np.log(np.clip(p_e, 1e-300, None))
    logp_d = np.log(np.clip(p_d, 1e-300, None))
    logp_c_list = [np.log(np.clip(p, 1e-300, None)) for p in cell_profiles]
    n_cand = int(cand.sum())
    post_c = np.zeros(n_cand, dtype=np.float64)
    post_d = np.zeros(n_cand, dtype=np.float64)
    post_e = np.zeros(n_cand, dtype=np.float64)
    post_c[hi] = 1.0
    for _ in range(max_iter):
        ll_e = _csr_multinomial_logpmf(xc, n_c, logp_e)
        ll_d = _csr_multinomial_logpmf(xc, n_c, logp_d)
        ll_c_stack = np.vstack([_csr_multinomial_logpmf(xc, n_c, logp) for logp in logp_c_list])
        ll_c = ll_c_stack.max(axis=0)
        which_c = ll_c_stack.argmax(axis=0)
        m = np.maximum(np.maximum(ll_e, ll_d), ll_c)
        w_e = np.exp(ll_e - m)
        w_d = np.exp(ll_d - m)
        w_c = np.exp(ll_c - m)
        z = w_e + w_d + w_c + 1e-300
        post_e, post_d, post_c = w_e / z, w_d / z, w_c / z
        post_c[hi] = 1.0
        post_d[hi] = 0.0
        post_e[hi] = 0.0
        num_d = np.asarray(xc.T.dot(post_d)).ravel()
        p_d = _good_turing_proportions(num_d)
        logp_d = np.log(np.clip(p_d, 1e-300, None))
        for k in range(len(cell_profiles)):
            weight_k = post_c * (which_c == k)
            if weight_k.sum() > 0:
                num_c_k = np.asarray(xc.T.dot(weight_k)).ravel()
                cell_profiles[k] = _good_turing_proportions(num_c_k)
                logp_c_list[k] = np.log(np.clip(cell_profiles[k], 1e-300, None))
    keep = np.zeros(totals.size, dtype=bool)
    is_cell = (post_c >= 0.5) & (post_c >= post_d) & (post_c >= post_e)
    keep[np.flatnonzero(cand)[is_cell]] = True
    return keep


def _mc_bin_pvals(
    niters: int,
    n_b: int,
    n_genes: int,
    cdf: np.ndarray,
    logp: np.ndarray,
    obs_unit: np.ndarray,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    """One UMI-magnitude bin's Monte Carlo p-values -- the parallel unit.

    Module-level (not a closure) so :class:`ProcessPoolExecutor` can pickle
    it. Bins are independent (disjoint output slices, no shared state), so
    this is safe to run concurrently with one call per bin.
    """
    from scipy.special import gammaln

    rng = np.random.default_rng(seed)
    p = np.diff(np.concatenate([[0.0], cdf]))
    max_chunk_bytes = 128 * 1024**2
    chunk_size = max(1, min(niters, max_chunk_bytes // max(n_genes * 8, 1)))
    extreme = np.zeros(obs_unit.size, dtype=np.int64)
    completed = 0
    while completed < niters:
        size = min(chunk_size, niters - completed)
        bc = rng.multinomial(n_b, p, size=size).astype(np.float64, copy=False)
        sim_logp = gammaln(n_b + 1.0) + (bc * logp).sum(axis=1) - gammaln(bc + 1.0).sum(axis=1)
        extreme += np.sum((sim_logp / n_b)[None, :] <= obs_unit[:, None], axis=1)
        completed += size
    return (1.0 + extreme) / (1.0 + niters)


def _multinomial_mc_pvals(
    x,
    totals: np.ndarray,
    test: np.ndarray,
    p_amb: np.ndarray,
    logp: np.ndarray,
    *,
    lower: int,
    niters: int,
    random_state: int,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Monte Carlo p-values vs ambient for ``test`` barcodes (EmptyDrops-style).

    Each UMI-magnitude bin is independent (disjoint output slices), so this
    dispatches one bin per worker via :class:`ProcessPoolExecutor` when
    there's more than one bin and the job is big enough to be worth the
    process-pool overhead; the worker count is capped by both usable CPUs
    and free RAM (:func:`_mc_worker_count`). Each bin gets its own
    reproducibly-spawned RNG stream (:class:`numpy.random.SeedSequence`),
    so the result for a given ``random_state`` is identical whether run
    serially or in parallel, and independent of worker count.
    """
    n_test = int(test.sum())
    if n_test == 0:
        return np.zeros(0, dtype=np.float64)
    obs = _csr_multinomial_logpmf(x[test], totals[test], logp)
    n_vals = np.clip(np.rint(totals[test]), 1, None).astype(np.int64)
    edges = np.unique(
        np.clip(
            np.round(np.geomspace(max(lower + 1, 1), max(int(n_vals.max()), lower + 2), 48)),
            1,
            None,
        ).astype(np.int64)
    )
    bin_id = np.digitize(n_vals, edges, right=True)
    pvals = np.ones(n_test, dtype=np.float64)
    cdf = np.cumsum(p_amb)
    cdf /= cdf[-1]
    n_genes = int(logp.size)
    obs_unit_full = obs / np.maximum(n_vals, 1)

    bins = np.unique(bin_id)
    seeds = np.random.SeedSequence(random_state).spawn(len(bins))
    jobs = []
    for b, seed in zip(bins, seeds, strict=False):
        in_b = bin_id == b
        n_b = int(np.median(n_vals[in_b])) if in_b.any() else 1
        n_b = max(n_b, 1)
        jobs.append((in_b, n_b, obs_unit_full[in_b], seed))

    # Below this, per-worker fork/pickle overhead would exceed the time
    # saved -- niters*n_genes*n_jobs is the aggregate simulated-count-matrix
    # size across bins, a proxy for total MC work.
    MIN_PARALLEL_WORK = 5_000_000
    n_workers = (
        _mc_worker_count(
            min(len(jobs), _usable_cpu_count() if n_jobs is None else max(1, n_jobs)),
        )
        if niters * n_genes * len(jobs) >= MIN_PARALLEL_WORK
        else 1
    )
    if n_workers <= 1:
        for in_b, n_b, obs_unit, seed in jobs:
            pvals[in_b] = _mc_bin_pvals(niters, n_b, n_genes, cdf, logp, obs_unit, seed)
        return pvals

    # spawn, not the platform-default fork: OpenBLAS's own thread pool is
    # live in this process (see _mc_worker_count's docstring), and forking
    # a multi-threaded process risks a deadlocked child if a non-forking
    # thread held a lock at fork time.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(_mc_bin_pvals, niters, n_b, n_genes, cdf, logp, obs_unit, seed): in_b
            for in_b, n_b, obs_unit, seed in jobs
        }
        for fut in as_completed(futures):
            pvals[futures[fut]] = fut.result()
    return pvals


def _empty_drops_keep(
    x,
    totals: np.ndarray,
    *,
    lower: int = 100,
    fdr: float = 0.001,
    niters: int = 10000,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> np.ndarray:
    """True for barcodes EmptyDrops (Lun et al. 2019) would call as non-empty.

    Ambient profile from barcodes with UMI ≤ ``lower``; barcodes above the
    barcode-rank knee are always retained; remaining barcodes are tested for
    a worse-than-null multinomial fit to ambient (Monte Carlo + BH).
    """
    x = _as_csr(x)
    totals = np.asarray(totals, dtype=np.float64)
    empty = totals <= lower
    if int(empty.sum()) < 10:
        raise ValueError(
            f"EmptyDrops needs at least 10 barcodes with UMI <= {lower} for the ambient pool"
        )
    ambient = np.asarray(x[empty].sum(axis=0)).ravel().astype(np.float64)
    p_amb = _good_turing_proportions(ambient)
    logp = np.log(np.clip(p_amb, 1e-300, None))
    retain = _barcode_knee_umi(totals, lower=lower)
    always = totals >= retain
    test = (totals > lower) & (~always)
    keep = always.copy()
    if int(test.sum()):
        pvals = _multinomial_mc_pvals(
            x,
            totals,
            test,
            p_amb,
            logp,
            lower=lower,
            niters=niters,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        q = _bh_fdr(pvals)
        keep[np.flatnonzero(test)[q <= fdr]] = True
    return keep


def _whitelist_vs_chi_keep(
    x,
    totals: np.ndarray,
    in_white: np.ndarray,
    *,
    lower: int = 100,
    fdr: float = 0.001,
    niters: int = 10000,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, float]:
    """Keep whitelist barcodes that reject ambient χ; always keep first inflection.

    χ is pooled from UMI ≤ ``lower`` barcodes *not* on the whitelist (SoupX
    empty band). Barcodes at or above the rank-curve inflection stay even if
    they look like soup (they *are* the soup's parent types). Below
    inflection, BH FDR vs multinomial χ; failures are debris, not empty.
    """
    x = _as_csr(x)
    totals = np.asarray(totals, dtype=np.float64)
    in_white = np.asarray(in_white, dtype=bool)
    empty = (~in_white) & (totals <= lower)
    if int(empty.sum()) < 10:
        raise ValueError(f"chi test needs at least 10 non-whitelist barcodes with UMI <= {lower}")
    ambient = np.asarray(x[empty].sum(axis=0)).ravel().astype(np.float64)
    p_amb = _good_turing_proportions(ambient)
    logp = np.log(np.clip(p_amb, 1e-300, None))
    curve = _barcode_rank_curve(totals, lower=lower)
    infl = curve["inflection_umi"]
    always = in_white.copy()
    if np.isfinite(infl):
        always &= totals >= infl
    else:
        always[:] = False
    test = in_white & (~always)
    keep = always.copy()
    if int(test.sum()):
        pvals = _multinomial_mc_pvals(
            x,
            totals,
            test,
            p_amb,
            logp,
            lower=lower,
            niters=niters,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        q = _bh_fdr(pvals)
        keep[np.flatnonzero(test)[q <= fdr]] = True
    return keep, float(infl)


def _cell_barcode_mask(adata: AnnData, cell_barcodes) -> np.ndarray:
    from .io import normalize_barcode

    if isinstance(cell_barcodes, (str, Path)):
        from .io import read_10x_barcodes

        cell_barcodes = read_10x_barcodes(cell_barcodes)
    _normalized_barcode_map(adata.obs_names, where="adata.obs_names")
    normalized_whitelist = _normalized_barcode_map(cell_barcodes, where="cell_barcodes")
    wanted = set(normalized_whitelist)
    is_cell = np.array([normalize_barcode(n) in wanted for n in adata.obs_names.astype(str)])
    n_matched = int(is_cell.sum())
    if n_matched != len(wanted):
        raise ValueError(
            f"{n_matched}/{len(wanted)} cell_barcodes matched adata.obs_names; "
            "every whitelist barcode must be present after normalization"
        )
    return is_cell


def call_cells(
    adata: AnnData,
    *,
    method: str = "diem",
    expect_cells: int | None = None,
    cell_barcodes: str | Path | list[str] | set[str] | None = None,
    max_cells: int | None = None,
    lower: int = 100,
    fdr: float = 0.001,
    niters: int = 10000,
    random_state: int = 0,
    n_jobs: int | None = None,
    min_empty: int = 10,
    layer: str | None = None,
    path: str | Path | None = None,
) -> list[str]:
    """Return cell barcodes when a filtered whitelist is missing or unusable.

    ``diem`` (default) is a 3-component multinomial mixture (empty / debris
    / cell): UMI ≤ ``lower`` fixes empty χ; high-UMI seeds cells; mid-UMI
    seeds debris. The barcode-rank first inflection is applied only when
    the mixture count exceeds ``MIX_INFLATION_RATIO`` times the inflection
    count (debris-heavy libraries). Barcodes below that cliff are kept if
    they are closer to the high-cell mean than to empty χ. Otherwise the
    mixture's second mode is kept. CPU-only.

    ``chi`` starts from an existing whitelist (Cell Ranger filtered list)
    and drops barcodes whose profiles cannot reject empty-droplet χ
    (multinomial Monte Carlo, BH FDR). Barcodes above the first inflection
    are kept even if they resemble χ. Failures stay ``other``, not empty.

    ``emptydrops`` is Lun et al. 2019: test vs ambient, always keep the
    barcode-rank knee, BH FDR ``fdr``. It calls non-empty droplets, not
    intact cells, and over-calls debris-heavy libraries.

    ``ordmag`` is Cell Ranger step 1 (Zheng et al. 2017): UMI ≥ 99th
    percentile of the top ``expect_cells`` barcodes, divided by 10.
    ``force`` is Cell Ranger ``--force-cells`` (exactly the top N by UMI).

    Empty droplets for χ remain the SoupX UMI≤100 band among barcodes not
    in this list. ``max_cells`` is an optional hard cap, not the method.

    ``niters`` sets the Monte Carlo p-value resolution for ``chi`` and
    ``emptydrops`` (min resolvable p-value is ``1/(niters+1)``). At the
    default ``fdr=0.001``, ``niters=2000`` sits close enough to that floor
    to both bias the call toward over-rejecting and make individual calls
    sensitive to ``random_state`` on whitelists with many thousands of
    borderline (near-``lower``) barcodes; 10000 was validated to bring both
    down substantially on real 16-43k-cell whitelists.
    """
    if not (0 < fdr <= 1):
        raise ValueError("fdr must satisfy 0 < fdr <= 1")
    if niters < 1:
        raise ValueError("niters must be at least 1")
    if lower < 0:
        raise ValueError("lower must be nonnegative")
    if min_empty < 1:
        raise ValueError("min_empty must be at least 1")
    _require_raw_integer_counts(adata, layer=layer, fname="call_cells")
    method = str(method).strip().lower()
    if method not in ("diem", "chi", "emptydrops", "ordmag", "force"):
        raise ValueError(
            f"method must be 'diem', 'chi', 'emptydrops', 'ordmag', or 'force', got {method!r}"
        )
    if expect_cells is not None and method not in ("ordmag", "force"):
        raise ValueError(f"expect_cells is not used by cell-calling method {method!r}")
    if method == "chi" and cell_barcodes is None:
        raise ValueError("method='chi' requires cell_barcodes (e.g. Cell Ranger filtered list)")
    if method in ("ordmag", "force") and (expect_cells is None or expect_cells < 1):
        raise ValueError(f"{method} requires expect_cells >= 1")
    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be at least 1")
    x = adata.layers[layer] if layer is not None else adata.X
    x = _as_csr(x)
    totals = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    extra = ""
    if method == "chi":
        in_white = _cell_barcode_mask(adata, cell_barcodes)
        keep_mask, infl = _whitelist_vs_chi_keep(
            x,
            totals,
            in_white,
            lower=lower,
            fdr=fdr,
            niters=niters,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        keep = np.flatnonzero(keep_mask)
        keep = _stable_count_order(adata, totals, keep)
        extra = (
            f", lower={lower}, fdr={fdr}, inflection_umi={infl:.0f}, "
            f"n_whitelist={int(in_white.sum())}"
        )
    elif method == "diem":
        mix_mask = _diem_keep(x, totals, lower=lower)
        n_mix = int(mix_mask.sum())
        curve = _barcode_rank_curve(totals, lower=lower)
        infl = curve["inflection_umi"]
        keep_mask, cut = _maybe_cut_inflated_mixture(mix_mask, totals, curve)
        n_rescue = 0
        if cut:
            rescued = _closer_to_cells_than_chi(
                x,
                totals,
                high=keep_mask,
                below=mix_mask & ~keep_mask,
                lower=lower,
            )
            n_rescue = int(rescued.sum())
            keep_mask = keep_mask | rescued
        keep = np.flatnonzero(keep_mask)
        keep = _stable_count_order(adata, totals, keep)
        extra = (
            f", lower={lower}, n_mixture={n_mix}, inflection_umi={infl:.0f}, "
            f"n_inflection={curve['n_inflection']}, "
            f"inflection_cut={'on' if cut else 'off'}, n_rescued={n_rescue}"
        )
    elif method == "emptydrops":
        keep_mask = _empty_drops_keep(
            x, totals, lower=lower, fdr=fdr, niters=niters, random_state=random_state, n_jobs=n_jobs
        )
        keep = np.flatnonzero(keep_mask)
        keep = _stable_count_order(adata, totals, keep)
        extra = f", lower={lower}, fdr={fdr}, knee_umi={_barcode_knee_umi(totals, lower=lower):.0f}"
    else:
        order = _stable_count_order(adata, totals)
        n_obs = int(order.size)
        n_expect = min(int(expect_cells), max(n_obs, 1))
        if method == "force":
            keep = order[:n_expect]
            threshold = float(totals[keep[-1]]) if keep.size else 0.0
        else:
            top = totals[order[:n_expect]]
            m = float(np.percentile(top, 99)) if top.size else 0.0
            threshold = m / 10.0
            keep = np.flatnonzero(totals >= threshold)
            keep = _stable_count_order(adata, totals, keep)
        extra = f", expect_cells={expect_cells}, umi_threshold={threshold:.1f}"
    if max_cells is not None and keep.size > max_cells:
        keep = keep[: int(max_cells)]
        extra += f", max_cells={max_cells}"
    n_noncell = adata.n_obs - int(keep.size)
    if n_noncell < min_empty:
        raise ValueError(
            f"cell calling must leave at least {min_empty} non-cell candidates; got {n_noncell}"
        )
    names = adata.obs_names.astype(str).to_numpy()[keep].tolist()
    print(
        f"ambidose: called {len(names)} cells (method={method}{extra})",
        file=sys.stderr,
    )
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(names) + ("\n" if names else ""))
    adata.obs["n_umi"] = totals
    return names


def mark_doublets(
    adata: AnnData,
    *,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    threshold: float | None = None,
    type_key: str | None = None,
    sample_key: str | None = SAMPLE_KEY_DEFAULT,
    layer: str | None = None,
    random_state: int = 0,
) -> AnnData:
    """Flag predicted doublets among cells without changing droplet labels."""
    _reject_view(adata, "mark_doublets")
    if droplet_key not in adata.obs:
        raise KeyError(f"{droplet_key!r} missing; run classify_droplets first")
    if type_key is not None and type_key not in adata.obs.columns:
        raise KeyError(f"{type_key!r} missing on obs")
    if sample_key is not None:
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
        _validated_sample_values(adata, sample_key)
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"layer={layer!r} missing from adata.layers")
    if CHI_KEY in adata.var.columns and CHI_KEY in adata.uns:
        raise ValueError("ambiguous χ state: both var and uns representations are present")
    _require_raw_integer_counts(adata, layer=layer, fname="mark_doublets")
    import scanpy as sc

    is_cell = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
    scores = np.full(adata.n_obs, np.nan, dtype=np.float64)
    predicted = pd.array([pd.NA] * adata.n_obs, dtype="boolean")
    n_cell = int(is_cell.sum())
    status = "skipped_too_few_cells"
    if n_cell >= 30:
        source = adata.layers[layer] if layer is not None else adata.X
        sub = adata[is_cell].copy()
        sub.X = source[is_cell].copy()
        n_pcs = max(2, min(30, n_cell - 2, max(sub.n_vars // 4, 2)))
        kw = {"random_state": random_state, "n_prin_comps": n_pcs}
        if threshold is not None:
            kw["threshold"] = threshold
        try:
            sc.pp.scrublet(sub, **kw)
        except (ValueError, RuntimeError) as exc:
            status = "scrublet_error"
            print(f"scrublet skipped ({n_cell} cells, {sub.n_vars} genes): {exc}")
        else:
            status = "ok"
            scores[is_cell] = np.asarray(sub.obs["doublet_score"], dtype=np.float64)
            predicted[is_cell] = np.asarray(sub.obs["predicted_doublet"], dtype=bool)
    adata.obs["ambidose_doublet_score"] = scores
    adata.obs["ambidose_doublet"] = predicted
    run = dict(adata.uns.get("ambidose", {}))
    run["doublets"] = {
        "layer": layer,
        "threshold": threshold,
        "random_state": int(random_state),
        "n_evaluated": n_cell if status == "ok" else 0,
        "status": status,
    }
    adata.uns["ambidose"] = run
    if type_key is not None and type_key != CLUSTER_KEY:
        adata.obs["ambidose_type_residual"] = _type_residual_score(
            adata,
            is_cell=is_cell,
            type_key=type_key,
            sample_key=sample_key,
            layer=layer,
        )
    elif "ambidose_type_residual" in adata.obs:
        del adata.obs["ambidose_type_residual"]
    return adata


def estimate_chi(
    adata: AnnData,
    *,
    droplet_key: str = DROPLET_KEY,
    empty_label: str = "empty",
    sample_key: str | None = SAMPLE_KEY_DEFAULT,
    layer: str | None = None,
    min_empty: int = 10,
) -> np.ndarray:
    """Estimate a simplex ambient profile from empty droplets.

    ``sample_key=None`` stores one library profile in ``adata.var[CHI_KEY]``.
    An explicit ``sample_key`` requires that obs column and stores a
    sample-by-gene DataFrame in ``adata.uns[CHI_KEY]``, including when the
    column contains only one sample label.
    """
    _reject_view(adata, "estimate_chi")
    if min_empty < 1:
        raise ValueError("min_empty must be at least 1")
    _require_raw_integer_counts(adata, layer=layer, fname="estimate_chi")
    x = adata.layers[layer] if layer is not None else adata.X
    x = _as_csr(x)

    if droplet_key not in adata.obs:
        raise KeyError(f"{droplet_key!r} missing; run classify_droplets first")

    empty = adata.obs[droplet_key].astype(str).to_numpy() == empty_label
    if int(empty.sum()) < min_empty:
        raise ValueError(f"need at least {min_empty} empty droplets, got {int(empty.sum())}")

    if sample_key is not None and sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
    use_sample = sample_key is not None
    if not use_sample:
        chi = _profile(x[empty])
        adata.uns.pop(CHI_KEY, None)
        adata.var[CHI_KEY] = chi
        run = dict(adata.uns.get("ambidose", {}))
        run["chi_provenance"] = {
            "sample_key": sample_key,
            "layer": layer,
            "droplet_key": droplet_key,
            "empty_label": empty_label,
        }
        adata.uns["ambidose"] = run
        return chi

    if not adata.var_names.is_unique:
        raise ValueError("multi-sample estimate_chi requires unique var_names for gene alignment")
    samples = _validated_sample_values(adata, sample_key)
    names: list[str] = []
    rows: list[np.ndarray] = []
    for name in samples.unique():
        mask = empty & (samples.to_numpy() == name)
        if int(mask.sum()) < min_empty:
            raise ValueError(
                f"sample {name!r}: need {min_empty} empty droplets, got {int(mask.sum())}"
            )
        names.append(str(name))
        rows.append(_profile(x[mask], sample=str(name)))
    chi = np.vstack(rows)
    if CHI_KEY in adata.var.columns:
        del adata.var[CHI_KEY]
    adata.uns[CHI_KEY] = pd.DataFrame(chi, index=names, columns=adata.var_names.astype(str))
    run = dict(adata.uns.get("ambidose", {}))
    run["chi_provenance"] = {
        "sample_key": sample_key,
        "layer": layer,
        "droplet_key": droplet_key,
        "empty_label": empty_label,
    }
    adata.uns["ambidose"] = run
    return chi


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


def subtract(
    adata: AnnData,
    *,
    dose: str | np.ndarray = DOSE_KEY,
    layer: str | None = None,
    layer_out: str = LAYER_OUT,
    sample_key: str | None = None,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    type_key: str | None = None,
    clip_negative: bool = True,
    max_type_mean: float = 0.05,
    min_chi: float = 1e-6,
    top_n: int = 100,
    empirical_margin: bool = True,
    cross_type_anchor: bool = True,
    relax_hk_when_soup_like: bool = False,
    n_jobs: int | None = None,
) -> AnnData:
    """Subtract a rank-1 take along χ, plus soupOnly extra-clear.

    ``d_c = ρ_c n_c`` is the per-cell χ-direction budget. Rank-1 take uses
    the type-median ρ and library-size weights so that scalar dose noise is
    not written onto gene residuals. Unused rank-1 budget after clipping
    on protected genes is reallocated to non-protected, non-soupOnly genes
    along χ. Unexpressed unowned genes are extra-cleared at the type's
    observed total when the type's dose-weighted ρ is at least
    ``SOUP_ONLY_RHO_FLOOR``; that extra-clear is outside the rank-1 budget
    and is not scaled back to ``d_c``. Type-aware subtraction preserves
    native programs; continuous residual mode is available only through
    ``clip_negative=False``.
    """
    _validate_output_layer(layer=layer, layer_out=layer_out)
    _reject_view(adata, "subtract")
    if clip_negative:
        _require_raw_integer_counts(adata, layer=layer, fname="subtract")
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X).copy().astype(np.float64)
    x.sum_duplicates()
    x.sort_indices()
    raw_x = x.copy()
    obs_keys = adata.obs_names.astype(str)
    if not obs_keys.is_unique:
        raise ValueError("subtract requires unique obs_names for stable integer allocation")
    gene_keys = np.asarray(_feature_keys(adata, where="subtract"), dtype=str)
    raw_gene_totals = np.asarray(x.sum(axis=0)).ravel()
    if isinstance(dose, str):
        if dose not in adata.obs:
            raise KeyError(f"{dose!r} missing; run estimate_dose first")
        d_v = np.asarray(adata.obs[dose], dtype=np.float64)
    else:
        d_v = np.asarray(dose, dtype=np.float64)
        if d_v.shape != (adata.n_obs,):
            raise ValueError("dose array must have length n_obs")
    if not np.isfinite(d_v).all():
        raise ValueError("dose contains non-finite values")
    if (d_v < 0).any():
        raise ValueError(
            "dose contains negative values -- d_c must be >= 0 (a negative "
            "dose would ADD counts above raw during subtraction, breaking "
            "the n_inflated=0 guarantee)"
        )
    n_v = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    over_n = d_v > n_v
    if over_n.any():
        raise ValueError("dose exceeds n_umi; stored and executed dose must be identical")

    samples = _sample_names(adata, sample_key)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    if (d_v[~is_cell] > 0).any():
        raise ValueError("positive dose found on non-cell droplets")
    use_sample = samples is not None
    if type_key is None:
        type_key = _default_type_key(adata)
    if type_key is not None and type_key not in adata.obs.columns:
        # Reject invalid labels instead of silently switching to untyped subtraction.
        cols = sorted(adata.obs.columns.astype(str))
        shown = cols[:20]
        more = f", and {len(cols) - 20} more" if len(cols) > 20 else ""
        raise KeyError(f"type_key={type_key!r} not in adata.obs (available: {shown}{more})")
    if isinstance(dose, str) and dose == DOSE_KEY:
        run = adata.uns.get("ambidose", {})
        provenance = run.get("dose_provenance")
        has_estimator_state = "dose_type_key" in run or "dose_sample_key" in run
        if has_estimator_state and provenance is None:
            raise ValueError("stored dose lacks provenance; recompute estimate_dose")
        if provenance is not None:
            requested = {
                "type_key": type_key,
                "sample_key": sample_key,
                "layer": layer,
                "droplet_key": droplet_key,
                "cell_label": cell_label,
            }
            if provenance != requested:
                raise ValueError(
                    "subtract parameters do not match dose provenance: "
                    f"stored={provenance}, requested={requested}"
                )

    use_mask = type_key is not None and type_key in adata.obs.columns and clip_negative

    if not use_mask:
        # x.nonzero() filters on *value*, not structure: a CSR matrix can
        # hold explicit stored zeros (left over from prior arithmetic, or
        # any op that doesn't call eliminate_zeros()), so len(nonzero()[0])
        # can be < len(x.data) -- then `x.data -= d_v[rows] * chi[cols]`
        # either raises a shape mismatch or, worse, silently misaligns which
        # gene each x.data entry gets debited against. Read the structural
        # indices directly instead, matching x.data 1:1 regardless of value.
        x.sort_indices()
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        cols = x.indices
        if use_sample:
            sample_names = _validated_sample_values(adata, sample_key).to_numpy()
            chi_by_sample = {
                name: _chi_for_obs(adata, sample_key=sample_key, sample_name=name)
                for name in pd.unique(sample_names)
            }
            for name, chi in chi_by_sample.items():
                row_mask = sample_names[rows] == name
                x.data[row_mask] -= d_v[rows[row_mask]] * chi[cols[row_mask]]
        else:
            chi = _chi_vector(adata)
            x.data -= d_v[rows] * chi[cols]
    else:
        x.sort_indices()
        data_positions = _selected_data_positions(x, np.flatnonzero(is_cell))
        n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
        types = _validated_type_values(adata, type_key).to_numpy()
        types = np.where(is_cell, types, EMPTY_TYPE)
        n_tiny_protected = 0
        n_meta_groups = 0
        mt_mask = _mt_gene_mask(adata.var_names)
        samples = _sample_names(adata, sample_key)
        if samples is None:
            groups = [None]
            sample_of = np.array([None] * adata.n_obs, dtype=object)
        else:
            groups = list(pd.unique(samples))
            sample_of = samples

        def process_sample(s):
            sample_native: dict[str, list[int]] = {}
            sample_tiny = 0
            chi = _chi_for_obs(
                adata,
                sample_key=sample_key if s is not None else None,
                sample_name=s,
            )
            in_s = sample_of == s
            types_s_all = np.where(in_s, types, EMPTY_TYPE)
            type_names_s = list(pd.unique(types[in_s]))
            type_indices_s = {t: np.flatnonzero(in_s & (types == t)) for t in type_names_s}
            type_means_s = {
                t: np.asarray(x[idx_t].mean(axis=0)).ravel()
                for t, idx_t in type_indices_s.items()
                if t not in EMPTY_TYPES and idx_t.size > 0
            }
            type_means = {
                t: type_means_s[t]
                for t, idx_t in type_indices_s.items()
                if t in type_means_s and idx_t.size >= MIN_TYPE_CELLS
            }
            dominant_masks, n_meta_s = _dominant_owner_masks(
                x,
                n,
                types_s_all,
                type_means,
                type_indices=type_indices_s,
                cell_keys=obs_keys,
                max_type_mean=max_type_mean,
            )
            sample_n_meta = int(n_meta_s)
            native_everywhere = _native_everywhere_mask(
                x,
                n,
                chi,
                types_s_all,
                in_s,
                max_type_mean=max_type_mean,
            )
            extra_protect_s: dict[str, np.ndarray] = {}
            if cross_type_anchor:
                u_masks_s: dict[str, np.ndarray] = {}
                for t, mean_t in type_means_s.items():
                    idx_t = type_indices_s[t]
                    expected_t = float(n[idx_t].mean()) * chi
                    u_masks_s[t] = _unexpressed_mask(
                        mean_t, expected_t, idx_t.size, max_type_mean=max_type_mean
                    )
                extra_protect_s = _cross_type_anchor_mask(
                    x,
                    n,
                    chi,
                    types_s_all,
                    type_means_s,
                    u_masks_s,
                    type_indices=type_indices_s,
                    max_type_mean=max_type_mean,
                    min_chi=min_chi,
                    n_jobs=n_jobs,
                )
            for t in type_names_s:
                idx = type_indices_s[t]
                if idx.size == 0:
                    continue
                d_sum = float(d_v[idx].sum())
                if d_sum <= 0:
                    continue
                if t in EMPTY_TYPES:
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                    is_p = np.zeros(adata.n_vars, dtype=bool)
                    native_confidence = np.zeros(adata.n_vars)
                elif idx.size < MIN_TYPE_CELLS:
                    # Fragments smaller than MIN_TYPE_CELLS skip extra-clear
                    # and take only the protected rank-1 slice.
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                    is_p = np.ones(adata.n_vars, dtype=bool)
                    native_confidence = np.ones(adata.n_vars)
                    sample_tiny += int(idx.size)
                else:
                    extra_mask = extra_protect_s.get(t, np.zeros(adata.n_vars, dtype=bool))
                    exclude = (
                        dominant_masks.get(t, np.zeros(adata.n_vars, dtype=bool))
                        | mt_mask
                        | extra_mask
                        | native_everywhere
                    )
                    mask_kw = {
                        "max_type_mean": max_type_mean,
                        "min_chi": min_chi,
                        "top_n": top_n,
                        "exclude": exclude,
                        "empirical_margin": empirical_margin,
                    }
                    is_u, is_p, native_confidence = _type_masks(x, n, chi, idx, **mask_kw)
                    if relax_hk_when_soup_like and _p_set_is_soup_like(x, n, chi, idx, is_p):
                        is_u, is_p, native_confidence = _type_masks(
                            x, n, chi, idx, **mask_kw, collision_exception=False
                        )
                y_cl = np.asarray(x[idx].sum(axis=0)).ravel().astype(np.float64)
                n_idx = n[idx]
                d_idx = d_v[idx]
                n_sum = float(n_idx.sum())
                rho_t = float(d_sum / n_sum) if n_sum > 0 else 0.0
                if rho_t < SOUP_ONLY_RHO_FLOOR:
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                # Pool cell-specific doses without replacing their heterogeneous
                # rho values by one type-level statistic. Ambient-like genes are
                # weighted by d_c; native genes stay library-weighted so noisy dose
                # estimates do not create artificial within-type expression structure.
                # soupOnly zeros unexpressed unowned genes at the type total
                # when rho_t is above the floor.
                take_rank1 = _confidence_weighted_take(y_cl, chi, d_sum, native_confidence, None)
                take_rank1 = np.where(is_u, 0.0, take_rank1)
                take_rank1 = _realloc_unspent_rank1(take_rank1, y_cl, chi, d_sum, is_p, is_u)
                take_rank1_native = np.where(is_p, take_rank1, 0.0)
                take_rank1_ambient = np.where(is_p, 0.0, take_rank1)
                take_u = np.where(is_u, y_cl, 0.0)
                _expand_take_to_cells(
                    x,
                    idx,
                    take_rank1_ambient,
                    d_idx,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                _expand_take_to_cells(
                    x,
                    idx,
                    take_rank1_native,
                    n_idx,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                _expand_take_to_cells(
                    x,
                    idx,
                    take_u,
                    d_idx,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                group_key = json.dumps(
                    [None if s is None else str(s), str(t)],
                    separators=(",", ":"),
                )
                sample_native[group_key] = np.flatnonzero(is_p).tolist()
            return sample_native, sample_n_meta, sample_tiny

        sample_results = list(map(process_sample, groups))
        native_internal: dict[str, list[int]] = {}
        meta_groups_by_sample: dict[str, dict] = {}
        total_meta_groups = 0
        for sample, (sample_native, sample_n_meta, sample_tiny) in zip(
            groups, sample_results, strict=True
        ):
            native_internal.update(sample_native)
            sample_id = _sample_storage_id(None if sample is None else str(sample))
            meta_groups_by_sample[sample_id] = {
                "sample": "" if sample is None else str(sample),
                "sample_is_global": sample is None,
                "n_meta_groups": int(sample_n_meta),
            }
            n_meta_groups = max(n_meta_groups, sample_n_meta)
            total_meta_groups += sample_n_meta
            n_tiny_protected += sample_tiny
        native_idx_by_type: dict[str, list[int]] = {}
        native_group_identity: dict[str, dict] = {}
        for i, identity_key in enumerate(sorted(native_internal)):
            sample_name, type_name = json.loads(identity_key)
            group_id = f"group_{i:06d}"
            native_idx_by_type[group_id] = native_internal[identity_key]
            native_group_identity[group_id] = {
                "sample": "" if sample_name is None else str(sample_name),
                "sample_is_global": sample_name is None,
                "type": str(type_name),
            }

        uns = dict(adata.uns.get("ambidose", {}))
        uns["native_genes_by_type"] = native_idx_by_type
        uns["native_group_identity"] = native_group_identity
        uns["meta_groups_by_sample"] = meta_groups_by_sample
        uns["total_meta_groups"] = int(total_meta_groups)
        uns["max_meta_groups_per_sample"] = int(n_meta_groups)
        uns["n_tiny_protected_cells"] = int(n_tiny_protected)
        adata.uns["ambidose"] = uns
        if n_tiny_protected:
            print(
                f"ambidose: {n_tiny_protected} cells in groups "
                f"<{MIN_TYPE_CELLS} cells; extra-clear skipped "
                "(rank-1 protected take only)",
                file=sys.stderr,
            )

    if clip_negative:
        np.maximum(x.data, 0.0, out=x.data)
        # soupOnly extra-clear (is_u) takes the group's observed count on
        # those genes and is outside the rank-1 d·χ budget. Do not scale
        # the row back to d_c: that put soup UMIs back after they were
        # cleared. Rank-1 take is already bounded by d_c. Cells that lose
        # more than d_c are flagged over_removal in QC.
        x = _integerize_corrected(raw_x, x, gene_keys=gene_keys)
        from .io import cast_int32_counts

        x.data = cast_int32_counts(x.data)
    adata.var["ambidose_removed_umi"] = raw_gene_totals - np.asarray(x.sum(axis=0)).ravel()
    actual_removed = np.asarray(raw_x.sum(axis=1) - x.sum(axis=1)).ravel()
    adata.obs["ambidose_removed_umi"] = actual_removed
    execution_ratio = np.full(actual_removed.shape, np.nan, dtype=float)
    positive_dose = d_v > 0
    execution_ratio[positive_dose] = actual_removed[positive_dose] / d_v[positive_dose]
    finite_ratio = execution_ratio[np.isfinite(execution_ratio)]
    uns = dict(adata.uns.get("ambidose", {}))
    uns["removal"] = {
        "n_zero_dose_removed": int(((d_v == 0) & (actual_removed > 0)).sum()),
        "execution_ratio_percentiles": (
            dict(
                zip(
                    ("p1", "p5", "median", "p95", "p99"),
                    np.percentile(finite_ratio, [1, 5, 50, 95, 99]).tolist(),
                    strict=True,
                )
            )
            if finite_ratio.size
            else {}
        ),
    }
    adata.uns["ambidose"] = uns
    adata.layers[layer_out] = x
    return adata


def _write_rho_trust(
    adata: AnnData, *, droplet_key: str, cell_label: str, layer_out: str = LAYER_OUT
) -> dict:
    """Flag cells where rho needs cautious quantitative interpretation."""
    n = adata.n_obs
    labels = np.full(n, TRUST_NOT_CELL, dtype=object)
    is_cell = np.ones(n, dtype=bool)
    if droplet_key in adata.obs:
        is_cell = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
    rho = np.asarray(adata.obs[RHO_KEY], dtype=np.float64) if RHO_KEY in adata.obs else np.zeros(n)
    fallback = (
        adata.obs["ambidose_dose_fallback"].to_numpy(dtype=bool)
        if "ambidose_dose_fallback" in adata.obs
        else np.zeros(n, dtype=bool)
    )
    has_fixed_genes = "ambidose_n_dose_genes" in adata.obs
    has_mixture_genes = "ambidose_mixture_n_genes" in adata.obs
    n_genes = (
        np.asarray(adata.obs["ambidose_n_dose_genes"], dtype=np.int64)
        if has_fixed_genes
        else np.zeros(n, dtype=np.int64)
    )
    if has_mixture_genes:
        # A cell's active dose can come from either estimator (adaptive
        # selection, ambidose_dose_selected) -- reading only the fixed
        # count for a mixture-selected cell is a phantom zero, not real
        # "no evidence" (estimate_dose_mixture() never wrote n_dose_genes
        # at all, so it defaulted to 0 for every mixture-derived cell
        # before ambidose_mixture_n_genes existed).
        mixture_genes = np.asarray(adata.obs["ambidose_mixture_n_genes"], dtype=np.int64)
        if "ambidose_dose_disagreement" in adata.obs:
            used_mixture = adata.obs["ambidose_dose_disagreement"].to_numpy(dtype=bool)
        elif not has_fixed_genes:
            # A standalone estimate_dose_mixture() run with no fixed pass
            # at all: every is_cell row's dose is the mixture one.
            used_mixture = np.ones(n, dtype=bool)
        else:
            used_mixture = np.zeros(n, dtype=bool)
        n_genes = np.where(used_mixture, mixture_genes, n_genes)
    one_type = (
        adata.obs["ambidose_one_type"].to_numpy(dtype=bool)
        if "ambidose_one_type" in adata.obs
        else np.zeros(n, dtype=bool)
    )
    tiny_type = np.zeros(n, dtype=bool)
    type_col = _default_type_key(adata)
    if type_col is not None:
        types = _validated_type_values(adata, type_col).to_numpy()
        for t in pd.unique(types[is_cell]):
            if t in EMPTY_TYPES:
                continue
            mask = is_cell & (types == t)
            if int(mask.sum()) < MIN_TYPE_CELLS:
                tiny_type[mask] = True
    structure = one_type | tiny_type
    low = fallback | (n_genes < MIN_GENES)
    ceiling = rho >= 0.95
    under_execution = np.zeros(n, dtype=bool)
    over_removal = np.zeros(n, dtype=bool)
    execution_ratio = np.full(n, np.nan, dtype=np.float64)
    removed_fraction = np.zeros(n, dtype=np.float64)
    pred_total = 0.0
    removed_total = 0.0
    sample_unspent = False
    if layer_out in adata.layers and DOSE_KEY in adata.obs:
        raw = raw_count_matrix(adata)
        den = _as_csr(adata.layers[layer_out])
        n_raw = np.asarray(raw.sum(axis=1)).ravel().astype(np.float64)
        n_den = np.asarray(den.sum(axis=1)).ravel().astype(np.float64)
        removed = np.maximum(n_raw - n_den, 0.0)
        pred = np.asarray(adata.obs[DOSE_KEY], dtype=np.float64)
        positive_dose = pred > 0
        execution_ratio[positive_dose] = removed[positive_dose] / pred[positive_dose]
        positive_raw = n_raw > 0
        removed_fraction[positive_raw] = removed[positive_raw] / n_raw[positive_raw]
        under_execution = is_cell & positive_dose & (execution_ratio < UNDER_EXECUTION_RATIO)
        over_removal = is_cell & (
            (positive_dose & (execution_ratio > OVER_EXECUTION_RATIO))
            | (removed_fraction > OVER_REMOVAL_FRACTION)
        )
        pred_total = float(pred[is_cell].sum())
        removed_total = float(removed[is_cell].sum())
        sample_unspent = pred_total > 0 and removed_total < UNDER_EXECUTION_RATIO * pred_total
    adata.obs["ambidose_dose_execution_ratio"] = execution_ratio
    adata.obs["ambidose_removed_fraction"] = removed_fraction
    adata.obs["ambidose_trust_under_execution"] = under_execution
    adata.obs["ambidose_trust_over_removal"] = over_removal
    adata.obs["ambidose_trust_ceiling"] = is_cell & ceiling
    adata.obs["ambidose_trust_low_evidence"] = is_cell & low
    adata.obs["ambidose_trust_type_structure"] = is_cell & structure
    labels[is_cell] = TRUST_OK
    labels[is_cell & structure] = TRUST_TYPE
    labels[is_cell & low] = TRUST_LOW_EVIDENCE
    labels[is_cell & ceiling] = TRUST_CEILING
    labels[under_execution] = TRUST_UNDER_EXECUTION
    labels[over_removal] = TRUST_OVER_REMOVAL
    adata.obs[RHO_TRUST_KEY] = pd.Categorical(
        labels,
        categories=[
            TRUST_OK,
            TRUST_LOW_EVIDENCE,
            TRUST_CEILING,
            TRUST_TYPE,
            TRUST_UNDER_EXECUTION,
            TRUST_OVER_REMOVAL,
            TRUST_NOT_CELL,
        ],
    )
    cell_labels = labels[is_cell]
    counts = {
        TRUST_OK: int((cell_labels == TRUST_OK).sum()),
        TRUST_LOW_EVIDENCE: int((cell_labels == TRUST_LOW_EVIDENCE).sum()),
        TRUST_CEILING: int((cell_labels == TRUST_CEILING).sum()),
        TRUST_TYPE: int((cell_labels == TRUST_TYPE).sum()),
        TRUST_UNDER_EXECUTION: int((cell_labels == TRUST_UNDER_EXECUTION).sum()),
        TRUST_OVER_REMOVAL: int((cell_labels == TRUST_OVER_REMOVAL).sum()),
    }
    uns = dict(adata.uns.get("ambidose", {}))
    uns["droplet_key"] = droplet_key
    uns["layer_out"] = layer_out
    uns["trust"] = {
        "n_cells": int(is_cell.sum()),
        "counts": counts,
        "predicted_removed_umi": pred_total,
        "actual_removed_umi": removed_total,
        "sample_dose_unspent": sample_unspent,
        "note": (
            "QC flags apply to quantitative interpretation of ambidose_rho, "
            "not to cell filtering. d_c is the χ-direction rank-1 budget, not a "
            "cap on total UMI removal. under_execution means less than half of "
            "that budget was removed; over_removal means total removal exceeded "
            "d_c by more than 5% (soupOnly extra-clear is allowed to do this) "
            "or exceeded half of the cell total."
        ),
    }
    adata.uns["ambidose"] = uns
    return counts


_INTERNAL_DENOISE_OBS = (
    "ambidose_d_raw",
    "ambidose_rho_raw",
    "ambidose_n_dose_genes",
    "ambidose_dose_fallback",
    "ambidose_one_type",
    "ambidose_shrink_w",
    "ambidose_dose_mixture",
    "ambidose_rho_mixture",
    "ambidose_dose_mixture_cell",
    "ambidose_rho_mixture_cell",
    "ambidose_dose_mixture_empty",
    "ambidose_rho_mixture_empty",
    "ambidose_mixture_profile",
    "ambidose_mixture_status",
    "ambidose_mixture_n_genes",
    "ambidose_mixture_iterations",
    "ambidose_mixture_converged",
    "ambidose_mixture_profile_tv",
    "ambidose_mixture_empty_tv",
    "ambidose_dose_log2_ratio",
    "ambidose_dose_rho_gap",
    "ambidose_dose_disagreement",
    "ambidose_dose_diagnosis",
    "ambidose_dose_selected",
    "ambidose_zero_dose_removed",
    "ambidose_trust_under_execution",
    "ambidose_trust_over_removal",
    "ambidose_trust_ceiling",
    "ambidose_trust_low_evidence",
    "ambidose_trust_type_structure",
)


def _clear_internal_denoise_obs(adata: AnnData) -> None:
    for column in _INTERNAL_DENOISE_OBS:
        if column in adata.obs:
            del adata.obs[column]


def _print_denoise_summary(
    counts: dict,
    *,
    n_cell: int,
    n_empty: int,
    median_dose: float,
    median_rho: float,
    median_execution_ratio: float | None,
    layer_out: str,
) -> None:
    """Print the result and explain rho-specific QC flags."""
    n_ok = counts.get(TRUST_OK, 0)
    n_flagged = n_cell - n_ok
    ok_pct = 100.0 * n_ok / n_cell if n_cell else 0.0
    flagged_pct = 100.0 * n_flagged / n_cell if n_cell else 0.0

    print("ambidose: denoise completed", file=sys.stderr)
    print(f"  cells corrected:             {n_cell:,}", file=sys.stderr)
    print(f"  reference empty droplets:    {n_empty:,}", file=sys.stderr)
    print(f"  median estimated ambient:    {100.0 * median_rho:.1f}%", file=sys.stderr)
    print(f"  median ambient UMI per cell: {median_dose:.1f}", file=sys.stderr)
    if median_execution_ratio is not None:
        print(
            f"  median dose actually removed: {100.0 * median_execution_ratio:.1f}%",
            file=sys.stderr,
        )
    print(f"  corrected counts:            layers[{layer_out!r}]", file=sys.stderr)
    print("ambidose: QC for estimated ambient fractions", file=sys.stderr)
    print(
        f"  suitable for interpretation: {n_ok:,} cells ({ok_pct:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  interpret with caution:      {n_flagged:,} cells ({flagged_pct:.1f}%)",
        file=sys.stderr,
    )
    reasons = (
        (TRUST_LOW_EVIDENCE, "too few informative genes"),
        (TRUST_CEILING, "estimate near the upper limit"),
        (TRUST_TYPE, "cell group too small or insufficiently resolved"),
        (TRUST_UNDER_EXECUTION, "less than half of estimated dose removed"),
        (TRUST_OVER_REMOVAL, "removal exceeded dose or half of cell UMIs"),
    )
    for key, explanation in reasons:
        count = counts.get(key, 0)
        if count:
            print(f"    {count:,}: {explanation}", file=sys.stderr)
    if n_flagged:
        print(
            "ambidose: QC flags concern interpretation of obs['ambidose_rho']; "
            "they are not a recommendation to remove those cells. See the QC "
            "report or obs['ambidose_rho_trust'] for cell-level details.",
            file=sys.stderr,
        )


def _normalized_barcode_map(names, *, where: str) -> dict[str, str]:
    from .io import normalize_barcode

    values = [str(x) for x in names]
    normalized = [normalize_barcode(x) for x in values]
    duplicated = pd.Index(normalized).duplicated(keep=False)
    if bool(duplicated.any()):
        # Two distinct causes give the same symptom (a repeated normalized
        # ID) but need different messages: an exact duplicate input row
        # (same raw string twice) is not a normalization problem at all,
        # while two genuinely different raw barcodes normalizing to the
        # same ID is. Distinguishing them avoids pointing a user with a
        # merely-duplicated barcode file at the wrong bug.
        norm_arr = np.asarray(normalized, dtype=object)
        val_arr = np.asarray(values, dtype=object)
        by_norm: dict[str, set[str]] = {}
        for n, v in zip(norm_arr[duplicated], val_arr[duplicated], strict=True):
            by_norm.setdefault(n, set()).add(v)
        true_collisions = sorted(n for n, raws in by_norm.items() if len(raws) > 1)
        if true_collisions:
            examples = true_collisions[:5]
            raise ValueError(
                f"{where}: barcode normalization is non-injective; normalized IDs "
                f"collide for {examples}"
            )
        dup_values = sorted({v for v in val_arr[duplicated]})[:5]
        raise ValueError(f"{where}: duplicate barcodes in input: {dup_values}")
    return dict(zip(normalized, values, strict=True))


def _feature_keys(adata: AnnData, *, where: str) -> list[str]:
    if "gene_ids" in adata.var.columns:
        ids = adata.var["gene_ids"]
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise ValueError(f"{where}: var['gene_ids'] must be complete and unique")
        return ids.astype(str).tolist()
    if not adata.var_names.is_unique:
        raise ValueError(
            f"{where}: duplicated var_names cannot be aligned safely without a "
            "complete, unique var['gene_ids'] column"
        )
    return adata.var_names.astype(str).tolist()


def _load_raw_pool(raw: str | Path | AnnData) -> AnnData:
    """Load ``raw`` (a path or an already-loaded AnnData) into one raw
    droplet-pool AnnData for :func:`_denoise_onto_filtered`.

    Feature identity is validated before a working copy is deduplicated.
    """
    if isinstance(raw, AnnData):
        raw_adata = raw.copy()
    else:
        from .io import read_10x_h5, read_10x_mtx, sniff_input

        resolved = sniff_input(raw)
        if resolved.kind == "root":
            raise ValueError(
                f"{raw!r} resolves to a multi-library root directory; pass one "
                "library's own raw path (or an already-loaded raw AnnData)"
            )
        if resolved.kind == "h5ad":
            import scanpy as sc

            raw_adata = sc.read_h5ad(resolved.raw)
        elif resolved.kind == "mtx":
            raw_adata = read_10x_mtx(resolved.raw)
        else:
            raw_adata = read_10x_h5(resolved.raw)
    _feature_keys(raw_adata, where="raw pool")
    raw_adata.var_names_make_unique()
    return raw_adata


def _denoise_onto_filtered(adata: AnnData, raw: str | Path | AnnData, **kwargs) -> AnnData:
    """Implements ``denoise(adata, raw=..., ...)``; see :func:`denoise`.

    Runs the ordinary raw-pool ``denoise()`` path with ``adata``'s own
    barcodes as the whitelist, then copies the results back onto a
    barcode/gene-aligned copy of the ORIGINAL ``adata`` (not the bigger
    raw+empty-droplet object) so a filtered, already-scanpy-loaded object
    is what the caller gets back -- with ``X`` set to the denoised counts
    and the original input preserved in ``layers["raw_counts"]``
    (``analysis_ready()``'s own convention), not left as the input counts.

    ``report=`` (forwarded via ``**kwargs``) is written from the raw pool
    BEFORE it is reduced to ``adata``'s own cells, so the report still gets
    the full empty-droplet panels. Calling ``write_report()``/``summarize()``
    yourself afterward on the object this function returns sees cells only
    (``n_empty``/``n_other`` read 0, droplet-class/barcode-rank panels show
    cells only) since the empty droplets are not carried into the returned
    copy.
    """
    if kwargs.get("cell_barcodes") is not None:
        raise ValueError(
            "pass only one of `raw` (adata's own obs_names become the whitelist) or cell_barcodes"
        )
    _reject_view(adata, "denoise")

    raw_adata = _load_raw_pool(raw)
    sample_key = kwargs.get("sample_key")
    if sample_key is not None and sample_key not in raw_adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in raw adata.obs")
    whitelist = list(adata.obs_names.astype(str))
    filtered_barcodes = _normalized_barcode_map(whitelist, where="filtered adata")
    raw_barcodes = _normalized_barcode_map(raw_adata.obs_names, where="raw pool")
    missing_barcodes = [key for key in filtered_barcodes if key not in raw_barcodes]
    if missing_barcodes:
        raise ValueError(
            f"raw= requires every filtered barcode to map to the raw pool; "
            f"{len(missing_barcodes)}/{len(whitelist)} are missing"
        )
    if sample_key is not None:
        raw_samples = _validated_sample_values(raw_adata, sample_key)
        matched_raw_samples = np.asarray(
            [raw_samples.loc[raw_barcodes[key]] for key in filtered_barcodes],
            dtype=str,
        )
        if sample_key in adata.obs.columns:
            filtered_samples = _validated_sample_values(adata, sample_key).to_numpy()
            disagree = filtered_samples != matched_raw_samples
            if bool(disagree.any()):
                raise ValueError(
                    "filtered and raw sample assignments disagree for "
                    f"{int(disagree.sum())} matched barcodes"
                )

    requested_calling = kwargs.get("cell_calling")
    if requested_calling is not None and str(requested_calling).strip().lower() not in (
        "off",
        "none",
        "external",
    ):
        raise ValueError(
            "raw= treats adata.obs_names as the complete filtered-cell set; "
            "cell_calling must be 'off' (or omitted) so cells are not dropped"
        )
    type_key = kwargs.get("type_key")
    if type_key is not None:
        if type_key not in adata.obs.columns:
            raise KeyError(f"{type_key!r} missing on filtered adata.obs")
        raw_adata.obs[type_key] = pd.Series(
            [EMPTY_TYPE] * raw_adata.n_obs, index=raw_adata.obs_names, dtype=object
        )
        for normalized, filtered_name in filtered_barcodes.items():
            raw_name = raw_barcodes[normalized]
            raw_adata.obs.loc[raw_name, type_key] = adata.obs.loc[filtered_name, type_key]

    kwargs["cell_calling"] = "off"
    kwargs["cell_barcodes"] = whitelist
    denoise(raw_adata, **kwargs)
    cells = analysis_ready(raw_adata)

    cells_barcodes = _normalized_barcode_map(cells.obs_names, where="denoised raw pool")
    cells_names = [cells_barcodes[key] for key in filtered_barcodes]

    filtered_features = _feature_keys(adata, where="filtered adata")
    cells_features = _feature_keys(cells, where="denoised raw pool")
    feature_to_cells_name = dict(zip(cells_features, cells.var_names.astype(str), strict=True))
    missing_features = [key for key in filtered_features if key not in feature_to_cells_name]
    if missing_features:
        raise ValueError(
            f"raw= requires every filtered feature to map to the raw pool; "
            f"{len(missing_features)}/{adata.n_vars} are missing"
        )
    cells_var_names = [feature_to_cells_name[key] for key in filtered_features]

    matched = cells[cells_names, cells_var_names]
    layer_out = kwargs["layer_out"]
    droplet_key = kwargs["droplet_key"]
    out = adata.copy()
    for col in list(out.obs.columns):
        if col.startswith("ambidose"):
            del out.obs[col]
    for col in list(out.var.columns):
        if col == CHI_KEY or col.startswith("ambidose"):
            del out.var[col]
    for name in list(out.layers):
        if str(name).startswith("ambidose"):
            del out.layers[name]
    for key in list(out.uns):
        if str(key).startswith("ambidose"):
            del out.uns[key]
    out.layers[layer_out] = matched.layers[layer_out].copy()
    out.layers["raw_counts"] = matched.layers["raw_counts"].copy()
    for col in matched.obs.columns:
        if col == droplet_key or col.startswith("ambidose"):
            out.obs[col] = matched.obs[col].to_numpy()
    for col in matched.var.columns:
        if col == CHI_KEY or col.startswith("ambidose"):
            out.var[col] = matched.var[col].to_numpy()
    if sample_key is not None:
        out.obs[sample_key] = matched.obs[sample_key].to_numpy()
    out.uns["ambidose"] = copy.deepcopy(matched.uns.get("ambidose", {}))
    out.uns["ambidose"]["input_layer"] = None
    if CHI_KEY in cells.uns:
        out.uns[CHI_KEY] = cells.uns[CHI_KEY].copy()
    has_var_chi = CHI_KEY in out.var.columns
    has_uns_chi = CHI_KEY in out.uns
    if has_var_chi == has_uns_chi:
        raise RuntimeError("completed raw= result must contain exactly one χ representation")
    out.X = out.layers[layer_out].copy()
    return out


def denoise(
    adata: AnnData,
    *,
    raw: str | Path | AnnData | None = None,
    sample_key: str | None = SAMPLE_KEY_DEFAULT,
    type_key: str | None = None,
    empty_umi_max: int | None = None,
    cell_barcodes: str | Path | list[str] | set[str] | None = None,
    expect_cells: int | None = None,
    cell_calling: str | None = None,
    max_cells: int | None = None,
    droplet_key: str = DROPLET_KEY,
    layer: str | None = None,
    layer_out: str = LAYER_OUT,
    clip_negative: bool = True,
    cross_type_anchor: bool = True,
    relax_hk_when_soup_like: bool = False,
    typing_fast: bool = True,
    n_jobs: int | None = None,
    report: bool | str | Path = False,
) -> AnnData:
    """Recommended entry: classify (if needed), estimate χ, dose, subtract.

    ``report``: write a QC HTML report (:func:`write_report`) once denoising
    finishes. ``True`` writes ``ambidose_report.html`` in the current
    working directory; a path writes there instead. The path actually used
    is printed to stderr. Default ``False`` (no report).

    ``raw``: pass the matching raw (unfiltered) droplet matrix -- a path
    to a 10x ``.h5``/mtx directory (auto-sniffed the same way the CLI's
    ``--input`` is) or an already-loaded raw ``AnnData`` -- to run
    AmbiDose the more common scanpy way: ``adata`` is your own already-
    filtered, cells-only object (e.g. from ``sc.read_10x_mtx``), its own
    ``obs_names`` become the cell whitelist, and a denoised copy of THIS
    ``adata`` (not the bigger raw+empty-droplet object) is returned --
    instead of loading the raw pool yourself and calling ``analysis_ready()``
    afterward. As with the traditional ``cell_barcodes=`` form below, the
    returned object already has ``X`` set to the denoised counts and the matched cells from the raw pool preserved in
    ``layers["raw_counts"]`` (same convention ``analysis_ready()`` uses) -- ready for ``sc.pp.normalize_total`` etc.
    with no extra step. Mutually exclusive with ``cell_barcodes`` (the whitelist is
    ``adata`` itself). ``layer=`` is not accepted: raw mode always reads
    counts from the raw pool's ``X``. Other keywords, including ``report=``,
    apply to the underlying run -- pass it here to get
    the full report (with empty-droplet panels), written from the raw pool
    before it is reduced to ``adata``'s own cells. Calling
    ``write_report()``/``summarize()`` yourself on the returned object
    still works, but sees cells only (no empty-droplet panels), since the
    returned object never carries empty droplets. See
    :func:`_denoise_onto_filtered`.

    Default groups are label-free coarse Leiden clusters, an operational
    identity for dose and subtraction, not cell-type names. Pass
    ``type_key`` for independently established broad labels computed
    outside this package. AmbiDose does not annotate lineages or call
    external APIs. Fine clusters are not an upgrade.
    Dose selection keeps the fixed estimate when it agrees with the internal
    mixture estimate and uses the mixture estimate on large disagreement.
    The mixture estimator itself chooses leave-one-type ambient unless that
    profile is a single other cell type, in which case it uses empty-droplet χ.
    A sample with one cell type uses the untyped χ quantile floor instead of
    a two-component fit.
    ``estimate_dose()`` remains the fixed quantile-floor estimator used inside
    this path; it is not an alternative product entry.
    ``cross_type_anchor`` enables subtraction-only cross-cell structure and
    cross-type-anchor protection; dose estimation remains unchanged.
    ``typing_fast`` (default True) uses a cheaper graph when the library has
    at least ``TYPING_FAST_N_CELLS`` cells. Pass ``typing_fast=False`` for the
    full-cell scale/PCA/neighbors graph used to pick Leiden resolution.

    Cell calling: whenever ``cell_barcodes`` resolves to a whitelist (given
    explicitly, auto-detected from a Cell Ranger ``outs/`` directory, or via
    a manifest/root library), it is refined against ambient χ by default --
    ``cell_calling`` values other than ``'off'``/``'none'``/``'external'``
    (including the ``'diem'`` default) all mean "refine this whitelist."
    Only ``cell_calling='off'`` trusts the list as-is and skips refinement.
    Without any whitelist, ``cell_calling='diem'`` builds the empty/debris/
    cell whitelist from scratch, ``emptydrops`` / ``expect_cells`` are as
    documented, and ``empty_umi_max`` alone is smoke-only.

    ``cell_barcodes`` is only read when droplet labels and χ are not already
    on the object. Passing a new whitelist after a previous ``denoise()``
    raises; drop the droplet column and stored χ, or copy the raw AnnData.

    ``n_jobs`` controls scanpy KNN/Leiden and structure-regression blocks
    (default: auto-detect from CPU affinity and available RAM). It also limits the Monte Carlo p-value step used by cell calling.
    """
    stored_run = adata.uns.get("ambidose", {})
    if isinstance(stored_run, dict) and (
        stored_run.get("core_completed") is True or stored_run.get("completed") is True
    ):
        required = ["raw_counts", str(stored_run.get("layer_out", LAYER_OUT))]
        missing = [name for name in required if name not in adata.layers]
        if missing:
            raise RuntimeError(
                "This AnnData records a completed AmbiDose run but required "
                f"layers are missing: {missing}"
            )
        raise ValueError(
            "denoise() has already completed on this AnnData; rerunning would "
            "subtract ambient counts a second time"
        )
    if raw is not None and layer is not None:
        raise ValueError(
            "denoise(raw=...) always reads counts from the raw pool X; "
            "layer= is not supported in raw mode"
        )
    _validate_output_layer(layer=layer, layer_out=layer_out)
    if expect_cells is not None and cell_calling is not None:
        calling_method = str(cell_calling).strip().lower()
        if calling_method not in ("ordmag", "force"):
            raise ValueError(f"expect_cells is not used by cell_calling={calling_method!r}")
    if layer_out in adata.layers:
        raise ValueError(
            f"layer_out={layer_out!r} already exists in adata.layers; "
            "refusing to overwrite user data"
        )
    if raw is None:
        if sample_key is not None and sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
        if type_key is not None and type_key not in adata.obs.columns:
            raise KeyError(f"type_key={type_key!r} not in adata.obs")
    if raw is not None:
        return _denoise_onto_filtered(
            adata,
            raw,
            sample_key=sample_key,
            type_key=type_key,
            empty_umi_max=empty_umi_max,
            cell_barcodes=cell_barcodes,
            expect_cells=expect_cells,
            cell_calling=cell_calling,
            max_cells=max_cells,
            droplet_key=droplet_key,
            layer=layer,
            layer_out=layer_out,
            clip_negative=clip_negative,
            cross_type_anchor=cross_type_anchor,
            relax_hk_when_soup_like=relax_hk_when_soup_like,
            typing_fast=typing_fast,
            n_jobs=n_jobs,
            report=report,
        )
    input_matrix = adata.layers[layer] if layer is not None else adata.X
    if "raw_counts" in adata.layers and not _same_matrix(adata.layers["raw_counts"], input_matrix):
        source = f"layers[{layer!r}]" if layer is not None else "X"
        raise ValueError(
            "layers['raw_counts'] already exists but differs from the requested "
            f"input {source}; rename or remove the conflicting layer before denoise()"
        )
    has_var_chi = CHI_KEY in adata.var.columns
    has_uns_chi = CHI_KEY in adata.uns
    if has_var_chi and has_uns_chi:
        raise ValueError("ambiguous χ state: both var and uns representations are present")
    chi_ready = has_var_chi or has_uns_chi
    if cell_barcodes is not None and expect_cells is not None:
        raise ValueError("pass only one of cell_barcodes or expect_cells")
    if cell_barcodes is not None and (droplet_key in adata.obs or chi_ready):
        stale = []
        if droplet_key in adata.obs:
            stale.append(f"obs[{droplet_key!r}]")
        if chi_ready:
            stale.append("stored χ")
        raise ValueError(
            "denoise(cell_barcodes=...) does not re-call cells or re-estimate "
            f"χ on an object that already has {' and '.join(stale)}"
        )
    resolved_n_jobs = _configure_scanpy_n_jobs(n_jobs)
    _reject_view(adata, "denoise")
    _require_raw_integer_counts(adata, layer=layer, fname="denoise")
    uns = dict(adata.uns.get("ambidose", {}))
    uns["droplet_key"] = droplet_key
    uns["layer_out"] = layer_out
    uns["input_layer"] = layer
    adata.uns["ambidose"] = uns
    if droplet_key not in adata.obs:
        # Cells-only object that already carries χ: do not invent empties
        # from a UMI cutoff. Low-UMI cells in a filtered matrix are still
        # cells; treating them as soup would both drop them from dose and
        # (if χ were missing) estimate soup from cells.
        if chi_ready and cell_barcodes is None:
            adata.obs[droplet_key] = pd.Categorical(["cell"] * adata.n_obs)
        elif cell_barcodes is not None:
            # A whitelist is refined against ambient chi by default -- the
            # only way to trust it as-is is an explicit cell_calling='off'.
            off = cell_calling is not None and str(cell_calling).strip().lower() in (
                "off",
                "none",
                "external",
            )
            if off:
                if isinstance(cell_barcodes, (str, Path)):
                    from .io import read_10x_barcodes

                    cell_barcodes = read_10x_barcodes(cell_barcodes)
                n_whitelist = len(cell_barcodes)
                classify_droplets(
                    adata,
                    empty_umi_max=empty_umi_max,
                    cell_barcodes=cell_barcodes,
                    layer=layer,
                    key_added=droplet_key,
                )
                n_called = int((adata.obs[droplet_key].astype(str) == "cell").sum())
                uns = dict(adata.uns.get("ambidose", {}))
                uns["cell_calling"] = {
                    "method": "off",
                    "n_whitelist": n_whitelist,
                    "n_called": n_called,
                    "n_dropped": n_whitelist - n_called,
                }
                adata.uns["ambidose"] = uns
            else:
                if isinstance(cell_barcodes, (str, Path)):
                    from .io import read_10x_barcodes

                    cell_barcodes = read_10x_barcodes(cell_barcodes)
                n_whitelist = len(cell_barcodes)
                called = call_cells(
                    adata,
                    method="chi",
                    cell_barcodes=cell_barcodes,
                    max_cells=max_cells,
                    lower=empty_umi_max if empty_umi_max is not None else 100,
                    layer=layer,
                    n_jobs=resolved_n_jobs,
                )
                uns = dict(adata.uns.get("ambidose", {}))
                uns["cell_calling"] = {
                    "method": "chi",
                    "max_cells": None if max_cells is None else int(max_cells),
                    "n_whitelist": n_whitelist,
                    "n_called": len(called),
                    "n_dropped": n_whitelist - len(called),
                }
                adata.uns["ambidose"] = uns
                from .io import normalize_barcode

                called_ids = {normalize_barcode(x) for x in called}
                rejected = [x for x in cell_barcodes if normalize_barcode(x) not in called_ids]
                classify_droplets(
                    adata,
                    empty_umi_max=empty_umi_max,
                    cell_barcodes=called,
                    other_barcodes=rejected,
                    layer=layer,
                    key_added=droplet_key,
                )
        elif cell_calling is not None and str(cell_calling).strip().lower() in (
            "off",
            "none",
            "external",
        ):
            raise ValueError(
                "cell_calling='off' needs cell_barcodes (Cell Ranger filtered "
                "list or another whitelist)"
            )
        elif (
            cell_calling is not None and str(cell_calling).strip().lower() in ("diem", "emptydrops")
        ) or expect_cells is not None:
            method = str(cell_calling).strip().lower() if cell_calling is not None else "ordmag"
            called = call_cells(
                adata,
                method=method,
                expect_cells=expect_cells,
                max_cells=max_cells,
                lower=empty_umi_max if empty_umi_max is not None else 100,
                layer=layer,
                n_jobs=resolved_n_jobs,
            )
            uns = dict(adata.uns.get("ambidose", {}))
            uns["cell_calling"] = {
                "method": method,
                "expect_cells": None if expect_cells is None else int(expect_cells),
                "max_cells": None if max_cells is None else int(max_cells),
                "n_called": len(called),
            }
            adata.uns["ambidose"] = uns
            classify_droplets(
                adata,
                empty_umi_max=empty_umi_max,
                cell_barcodes=called,
                layer=layer,
                key_added=droplet_key,
            )
        elif empty_umi_max is not None:
            print(
                "ambidose: empty_umi_max without cell_barcodes is smoke-only",
                file=sys.stderr,
            )
            classify_droplets(
                adata,
                empty_umi_max=empty_umi_max,
                layer=layer,
                key_added=droplet_key,
            )
        else:
            raise ValueError(
                "denoise() needs cell_barcodes, cell_calling='diem' or "
                "'emptydrops', expect_cells (OrdMag/force-cells), existing "
                "obs['ambidose_droplet'], or a stored χ. UMI-threshold "
                "cell calling is smoke-only: pass empty_umi_max explicitly"
            )
    if sample_key is not None and sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
    sk = sample_key
    if sk is not None:
        _validated_sample_values(adata, sk)
        if chi_ready:
            if CHI_KEY not in adata.uns:
                raise ValueError(
                    f"explicit sample_key requires sample-specific profiles in uns[{CHI_KEY!r}]"
                )
            _validate_chi_frame(adata, sk)
    uns = dict(adata.uns.get("ambidose", {}))
    uns["sample_key"] = sk
    adata.uns["ambidose"] = uns
    if not chi_ready:
        print(
            "ambidose: estimating ambient profile (χ) from empty droplets...",
            file=sys.stderr,
            flush=True,
        )
        estimate_chi(adata, droplet_key=droplet_key, sample_key=sk, layer=layer)
    if type_key is None:
        print(
            "ambidose: resolving coarse cell types (Leiden clustering)...",
            file=sys.stderr,
            flush=True,
        )
    import scanpy as sc

    previous_scanpy_n_jobs = sc.settings.n_jobs
    sc.settings.n_jobs = resolved_n_jobs
    try:
        resolved = resolve_type_key(
            adata,
            type_key=type_key,
            droplet_key=droplet_key,
            layer=layer,
            sample_key=sk,
            typing_fast=typing_fast,
        )
    finally:
        sc.settings.n_jobs = previous_scanpy_n_jobs
    with _StageProgress("estimating per-cell ambient dose"):
        estimate_dose_adaptive(
            adata,
            type_key=resolved,
            droplet_key=droplet_key,
            sample_key=sk,
            layer=layer,
        )
    with _StageProgress("subtracting ambient counts"):
        subtract(
            adata,
            sample_key=sk,
            droplet_key=droplet_key,
            type_key=resolved,
            layer=layer,
            layer_out=layer_out,
            clip_negative=clip_negative,
            cross_type_anchor=cross_type_anchor,
            relax_hk_when_soup_like=relax_hk_when_soup_like,
            n_jobs=resolved_n_jobs,
        )
    adata.layers["raw_counts"] = raw_count_matrix(adata).copy()
    # An explicit layer leaves caller-owned normalized/log-transformed X intact.
    if layer is None:
        adata.X = adata.layers[layer_out].copy()
    uns = dict(adata.uns.get("ambidose", {}))
    uns["subtraction_completed"] = True
    uns["core_completed"] = True
    uns["completed"] = True
    uns["postprocess_completed"] = False
    uns["report_completed"] = not bool(report)
    adata.uns["ambidose"] = uns
    if droplet_key in adata.obs:
        is_cell_diag = adata.obs[droplet_key].astype(str).to_numpy() == "cell"
    else:
        is_cell_diag = np.ones(adata.n_obs, dtype=bool)
    n_empty = (
        int((adata.obs[droplet_key].astype(str) == "empty").sum())
        if droplet_key in adata.obs
        else 0
    )
    n_cell = int(is_cell_diag.sum())
    # Dose and rho are zero on non-cell droplets; summarize cells only.
    d = np.asarray(adata.obs[DOSE_KEY], dtype=np.float64)[is_cell_diag]
    rho = np.asarray(adata.obs[RHO_KEY], dtype=np.float64)[is_cell_diag]
    trust_counts = _write_rho_trust(
        adata, droplet_key=droplet_key, cell_label="cell", layer_out=layer_out
    )
    removal = adata.uns.get("ambidose", {}).get("removal", {})
    execution_percentiles = removal.get("execution_ratio_percentiles", {})
    median_execution_ratio = execution_percentiles.get("median")
    _print_denoise_summary(
        trust_counts,
        n_cell=n_cell,
        n_empty=n_empty,
        median_dose=float(np.median(d)) if n_cell else 0.0,
        median_rho=float(np.median(rho)) if n_cell else 0.0,
        median_execution_ratio=median_execution_ratio,
        layer_out=layer_out,
    )
    _clear_internal_denoise_obs(adata)
    uns = dict(adata.uns.get("ambidose", {}))
    uns["postprocess_completed"] = True
    adata.uns["ambidose"] = uns
    if report:
        from .reporting import write_report

        report_path = Path("ambidose_report.html") if report is True else Path(report)
        write_report(adata, report_path)
        print(f"ambidose: wrote QC report to {report_path.resolve()}", file=sys.stderr)
        uns = dict(adata.uns.get("ambidose", {}))
        uns["report_completed"] = True
        adata.uns["ambidose"] = uns
    return adata


def analysis_ready(
    adata: AnnData,
    *,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    denoised_layer: str | None = None,
    raw_layer: str = "raw_counts",
) -> AnnData:
    """Return cell-only AnnData with corrected X and preserved raw counts.

    Droplet column and denoised layer default to the keys ``denoise()`` stored
    in ``uns['ambidose']``.
    """
    stored_key, stored_layer = require_run_keys(adata)
    if droplet_key is None:
        droplet_key = stored_key
    if denoised_layer is None:
        denoised_layer = stored_layer
    if raw_layer == denoised_layer:
        raise ValueError("raw_layer must differ from denoised_layer")
    if raw_layer != "raw_counts" and raw_layer in adata.layers:
        raise ValueError(
            f"raw_layer={raw_layer!r} already exists in adata.layers; "
            "refusing to overwrite user data"
        )
    if droplet_key not in adata.obs:
        raise KeyError(f"{droplet_key!r} missing; run denoise first")
    if denoised_layer not in adata.layers:
        raise KeyError(f"{denoised_layer!r} missing; run denoise first")
    keep = adata.obs[droplet_key].astype(str).to_numpy() == cell_label
    out = adata[keep].copy()
    # Raw counts may live in a non-X layer (denoise(layer=...)), not X itself;
    # reuse the same resolution order raw_count_matrix()/summarize() use
    # (layers['raw_counts'] -> uns input_layer -> X) instead of assuming X.
    out.layers[raw_layer] = raw_count_matrix(out).copy()
    out.X = out.layers[denoised_layer].copy()
    return out
