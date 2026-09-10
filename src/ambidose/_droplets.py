"""Droplet calling: classify, EmptyDrops/DIEM, call_cells, doublets."""

from __future__ import annotations

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
    _type_residual_score,
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
    CHI_KEY,
    CLUSTER_KEY,
    DROPLET_KEY,
    MIX_INFLATION_RATIO,
    SAMPLE_KEY_DEFAULT,
    _as_csr,
    _mc_worker_count,
    _reject_view,
    _require_raw_integer_counts,
    _stable_count_order,
    _usable_cpu_count,
    _validated_sample_values,
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


def _normalized_barcode_map(names, *, where: str) -> dict[str, str]:
    from .io import normalize_barcode

    values = [str(x) for x in names]
    normalized = [normalize_barcode(x) for x in values]
    duplicated = pd.Index(normalized).duplicated(keep=False)
    if bool(duplicated.any()):
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
