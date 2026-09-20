"""Ambient-dose estimation and adaptive estimator selection."""

from __future__ import annotations

import copy
import sys
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._shared import (
    DOSE_KEY,
    DROPLET_KEY,
    EMPTY_TYPES,
    MIN_GENES,
    MIN_VALID,
    RHO_KEY,
    SHRINK_K,
    _as_csr,
    _chi_for_obs,
    _chi_vector,
    _need,
    _reject_view,
    _require_raw_integer_counts,
    _resolve_cell_mask,
    _sample_names,
    _sample_storage_id,
    _validated_type_values,
)


def _dose_provenance(*, type_key, sample_key, layer, droplet_key, cell_label):
    return {
        "type_key": type_key,
        "sample_key": sample_key,
        "layer": layer,
        "droplet_key": droplet_key,
        "cell_label": cell_label,
    }


def _validate_chi_provenance(adata: AnnData, *, sample_key, layer) -> None:
    provenance = adata.uns.get("ambidose", {}).get("chi_provenance")
    if provenance is None:
        return
    requested = {"sample_key": sample_key, "layer": layer}
    stored = {key: provenance.get(key) for key in requested}
    if stored != requested:
        raise ValueError(
            _need(
                f"Dose settings do not match how χ was estimated (stored={stored}, "
                f"requested={requested}).",
                "Re-run estimate_chi with the same sample_key and layer, or call denoise().",
            )
        )


def _quantile_rho_on_genes(
    x, n, chi, cell_idx, gidx, *, quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rho, n_valid) for ``cell_idx`` using gene columns ``gidx``."""
    n_cells = cell_idx.size
    rho = np.full(n_cells, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_cells, dtype=np.int64)
    if gidx.size == 0 or n_cells == 0:
        return rho, n_valid
    sub = x[cell_idx][:, gidx].toarray().astype(np.float64)
    expected = n[cell_idx, None] * chi[gidx][None, :]
    ratios = np.divide(sub, expected, out=np.full_like(sub, np.nan), where=expected > 0)
    ratios[sub <= 0] = np.nan
    n_valid = np.sum(~np.isnan(ratios), axis=1)
    ok = n_valid >= 1
    if ok.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            q = np.nanquantile(ratios[ok], quantile, axis=1)
        rho[ok] = np.clip(q, 0.0, 1.0)
    return rho, n_valid


def _mle_rho_on_genes(x, n, chi, cell_idx, gidx) -> tuple[np.ndarray, np.ndarray]:
    """Poisson MLE ``ρ = Σ y / Σ(n χ)`` on ``gidx``. Zeros are data."""
    n_cells = cell_idx.size
    rho = np.full(n_cells, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_cells, dtype=np.int64)
    if gidx.size == 0 or n_cells == 0:
        return rho, n_valid
    sub = x[cell_idx][:, gidx].toarray().astype(np.float64)
    mass = n[cell_idx, None] * chi[gidx][None, :]
    mass = np.where(mass > 0, mass, 0.0)
    denom = mass.sum(axis=1)
    # Shrinkage depends on genes observed in this cell, not the size of gidx.
    n_valid = np.sum(sub > 0, axis=1)
    ok = denom > 0
    rho[ok] = np.clip(sub.sum(axis=1)[ok] / denom[ok], 0.0, 1.0)
    return rho, n_valid


NOISE_K = 2.0
U_MAX_LIBRARY_FRAC = 0.003
# y/(nχ) quantile when MLE is not identified and empty calibration is unavailable.
DOSE_RATIO_QUANTILE = 0.5


def _expression_floor(n_cells: int, max_type_mean: float | None) -> float:
    """Mean floor for the unexpressed test. ``None`` is 0 (Poisson only)."""
    if max_type_mean is not None:
        return float(max_type_mean)
    return 0.0


def _chi_prefix_n(chi: np.ndarray, *, min_n: int = MIN_GENES) -> int:
    """Gene count at this sample's χ Lorenz knee, at least ``min_n``."""
    chi = np.asarray(chi, dtype=np.float64)
    if chi.size == 0:
        return max(int(min_n), 1)
    order = np.argsort(-chi)
    total = float(chi[order].sum())
    if total <= 0:
        return max(int(min_n), 1)
    c = np.cumsum(chi[order]) / total
    x = np.arange(1, c.size + 1, dtype=np.float64) / c.size
    n = int(np.argmax(c - x)) + 1
    return max(n, int(min_n), 1)


def q_abs_scale(q: float, *, hat_rho: float) -> float:
    """Executed scale is 1. Raises if ``q`` or ``hat_rho`` is not finite."""
    if not np.isfinite(hat_rho):
        raise ValueError(f"hat_rho must be finite, got {hat_rho!r}")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q!r}")
    return 1.0


def _unexpressed_mask(
    mean: np.ndarray,
    expected: np.ndarray,
    n_cells_t: int,
    *,
    max_type_mean: float | None,
    noise_k: float = NOISE_K,
    margin_sd: np.ndarray | None = None,
    dominant_exclusion_applied: bool = False,
) -> np.ndarray:
    """True where a type's mean is compatible with a pure-ambient ceiling.

    The default margin uses Poisson sampling variance around ``n_bar * chi``.
    ``margin_sd`` uses empirical variance and is allowed only when the caller
    also excludes dominant native genes; otherwise variable native genes can
    be misclassified as ambient. ``max_type_mean`` bounds behavior when the
    expected ambient count is near zero.
    """
    if margin_sd is not None and not dominant_exclusion_applied:
        raise ValueError(
            "margin_sd requires dominant_exclusion_applied=True; empirical "
            "variance is unsafe without excluding dominant native genes"
        )
    floor = _expression_floor(n_cells_t, max_type_mean)
    if n_cells_t <= 0:
        return mean <= floor
    if margin_sd is not None:
        margin = noise_k * margin_sd / np.sqrt(n_cells_t)
    else:
        margin = noise_k * np.sqrt(np.maximum(expected, 0.0) / n_cells_t)
    ceiling = np.maximum(floor, expected + margin)
    return mean <= ceiling


def _mean_compatible_with_type_rho(
    mean: np.ndarray,
    expected_rho1: np.ndarray,
    n_cells: int,
    rho_t: float,
    *,
    noise_k: float = NOISE_K,
) -> np.ndarray:
    """True where the type mean is compatible with soup at this type's ρ.

    ``expected_rho1`` is n̄χ (the ρ=1 ceiling). True soup sits at ρ_t · n̄χ;
    the Poisson margin matches ``_unexpressed_mask``.
    """
    expected_soup = float(max(rho_t, 0.0)) * np.asarray(expected_rho1, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if n_cells <= 0:
        return mean <= 0.0
    margin = noise_k * np.sqrt(np.maximum(expected_soup, 0.0) / n_cells)
    return mean <= expected_soup + margin


def _u_mass_fits_empty_droplets(
    x,
    cell_idx: np.ndarray,
    empty_idx: np.ndarray,
    is_u: np.ndarray,
    *,
    noise_k: float = NOISE_K,
    lam_e: float | None = None,
    chi: np.ndarray | None = None,
) -> bool:
    """True if this type's U-gene UMI total is compatible with empty droplets.

    Empty mean on U genes × n_cells is the soup-per-droplet null. When
    empty rows are absent, ``lam_e * χ_U * n_cells`` is the same null.
    """
    u = np.asarray(is_u, dtype=bool)
    if not u.any() or cell_idx.size == 0:
        return False
    cols = np.flatnonzero(u)
    observed = float(np.asarray(x[cell_idx][:, cols].sum()))
    if empty_idx.size:
        empty_u = float(np.asarray(x[empty_idx][:, cols].sum()))
        expected = empty_u / empty_idx.size * cell_idx.size
    elif lam_e is not None and chi is not None and lam_e > 0:
        expected = float(lam_e) * float(np.asarray(chi)[u].sum()) * cell_idx.size
    else:
        return False
    return observed <= expected + noise_k * np.sqrt(max(expected, 0.0))


def _soup_per_cell_fits_empty(
    rho_t: float, n_bar: float, lam_e: float, *, noise_k: float = NOISE_K
) -> bool:
    """True if estimated soup UMIs per cell are not above one empty droplet."""
    if not np.isfinite(lam_e) or lam_e <= 0 or n_bar <= 0:
        return False
    soup = float(rho_t) * float(n_bar)
    return soup <= lam_e + noise_k * np.sqrt(lam_e)


def _soup_u_mask(
    mean: np.ndarray,
    chi: np.ndarray,
    n_cells: int,
    n_bar: float,
    *,
    max_type_mean: float | None,
    min_chi: float,
    native_everywhere: np.ndarray,
    apply_rt_and_collision: bool,
    rho_t: float | None = None,
) -> np.ndarray:
    """Unexpressed ∩ χ-supported genes. Same ceiling as dose U evidence.

    ``apply_rt_and_collision`` adds ceiling/abundance guards and a
    type-ρ compatibility test. Dose MLE omits those (``False``).
    """
    expected = np.asarray(chi, dtype=np.float64) * float(n_bar)
    u = _unexpressed_mask(mean, expected, n_cells, max_type_mean=max_type_mean)
    u = u & ~np.asarray(native_everywhere, dtype=bool) & (chi >= min_chi)
    if apply_rt_and_collision and n_bar > 0:
        if rho_t is None:
            raise ValueError("rho_t is required when apply_rt_and_collision=True")
        r_t = np.divide(mean, expected, out=np.zeros_like(mean), where=expected > 0)
        u = u & _mean_compatible_with_type_rho(mean, expected, n_cells, rho_t)
        u = u & ~((mean > U_MAX_LIBRARY_FRAC * n_bar) & (r_t > 0.5))
        u = u & ~(r_t > 0.85)
    return u


def _top_chi_indices(chi: np.ndarray, candidates: np.ndarray, top_n: int | None) -> np.ndarray:
    """Return top χ candidates, including every tie at the boundary.

    ``top_n is None`` uses this sample's χ Lorenz-knee count.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    n = _chi_prefix_n(chi) if top_n is None else int(top_n)
    if n < 1:
        raise ValueError("top_n must be at least 1")
    if candidates.size <= n:
        return candidates
    values = chi[candidates]
    cutoff = np.partition(values, values.size - n)[values.size - n]
    return candidates[values >= cutoff]


def _calibrate_ratio_quantile(x, n, chi, empty_idx: np.ndarray, gidx: np.ndarray) -> float:
    """Choose the y/(nχ) quantile so empty droplets have median ρ̂ ≈ 1."""
    if empty_idx.size < MIN_VALID or gidx.size == 0:
        return DOSE_RATIO_QUANTILE

    def med_rho(q: float) -> float:
        rho, nv = _quantile_rho_on_genes(x, n, chi, empty_idx, gidx, quantile=q)
        ok = np.isfinite(rho) & (nv >= 1)
        if not ok.any():
            return np.nan
        return float(np.median(rho[ok]))

    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        m = med_rho(mid)
        if not np.isfinite(m) or m < 1.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _rho_from_chi(
    x,
    n,
    chi,
    cell_idx,
    *,
    top_n: int | None,
    min_chi: float,
    quantile: float | None,
    min_valid: int = MIN_VALID,
    empty_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile floor and its per-cell number of valid soup genes."""
    candidates = np.flatnonzero(chi >= min_chi)
    gidx = _top_chi_indices(chi, candidates, top_n)
    q = quantile
    if q is None:
        q = _calibrate_ratio_quantile(
            x, n, chi, np.asarray([] if empty_idx is None else empty_idx), gidx
        )
    rho, n_valid = _quantile_rho_on_genes(x, n, chi, cell_idx, gidx, quantile=q)
    rho[n_valid < min_valid] = 0.0
    return np.nan_to_num(rho, nan=0.0), n_valid


def _dose_quantile_from_chi(
    adata: AnnData,
    *,
    quantile: float | None = None,
    top_n: int | None = None,
    min_chi: float = 1e-6,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    min_valid: int = MIN_VALID,
) -> np.ndarray:
    _reject_view(adata, "_dose_quantile_from_chi")
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["n_umi"] = n
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    rho = np.zeros(adata.n_obs, dtype=np.float64)
    samples = _sample_names(adata, sample_key)
    if samples is not None:
        for s in pd.unique(samples):
            chi = _chi_for_obs(adata, sample_key=sample_key, sample_name=s)
            idx = np.flatnonzero(is_cell & (samples == s))
            empty_idx = np.flatnonzero(~is_cell & (samples == s))
            rho[idx], _ = _rho_from_chi(
                x,
                n,
                chi,
                idx,
                top_n=top_n,
                min_chi=min_chi,
                quantile=quantile,
                min_valid=min_valid,
                empty_idx=empty_idx,
            )
    else:
        chi = _chi_vector(adata)
        idx = np.flatnonzero(is_cell)
        empty_idx = np.flatnonzero(~is_cell)
        rho[idx], _ = _rho_from_chi(
            x,
            n,
            chi,
            idx,
            top_n=top_n,
            min_chi=min_chi,
            quantile=quantile,
            min_valid=min_valid,
            empty_idx=empty_idx,
        )
    dose = rho * n
    adata.obs[DOSE_KEY] = dose
    adata.obs[RHO_KEY] = rho
    uns = dict(adata.uns.get("ambidose", {}))
    uns["dose_type_key"] = None
    uns["dose_sample_key"] = sample_key
    uns["dose_provenance"] = _dose_provenance(
        type_key=None,
        sample_key=sample_key,
        layer=layer,
        droplet_key=droplet_key,
        cell_label=cell_label,
    )
    uns["dose"] = {"method": "quantile"}
    adata.uns["ambidose"] = uns
    return dose


def _native_everywhere_mask(
    x,
    n: np.ndarray,
    chi: np.ndarray,
    types: np.ndarray,
    mask: np.ndarray,
    *,
    max_type_mean: float | None,
) -> np.ndarray:
    """Drop genes that look native in every type, not true soup.

    Empty-droplet soup sits at or below ``n̄ χ`` in every type and is kept.
    Housekeeping exceeds that ceiling in every type. Genes on the ceiling
    in every type that also take a large library fraction are dropped.
    """
    above = []
    collide = []
    for t in pd.unique(types[mask]):
        if t in EMPTY_TYPES:
            continue
        idx = np.flatnonzero(mask & (types == t))
        if idx.size == 0:
            continue
        mean = np.asarray(x[idx].mean(axis=0)).ravel()
        n_bar = float(n[idx].mean())
        expected = n_bar * chi
        unexp = _unexpressed_mask(mean, expected, idx.size, max_type_mean=max_type_mean)
        above.append(~unexp)
        r_t = np.divide(mean, expected, out=np.zeros_like(mean), where=expected > 0)
        collide.append((n_bar > 0) & (mean > U_MAX_LIBRARY_FRAC * n_bar) & (r_t > 0.85))
    if len(above) < 2:
        return np.zeros(x.shape[1], dtype=bool)
    stacked_above = np.vstack(above)
    stacked_collide = np.vstack(collide)
    return stacked_above.all(axis=0) | stacked_collide.all(axis=0)


def _atomic_obs_uns(func):
    """Restore result metadata if an estimator fails before commit."""
    from functools import wraps

    @wraps(func)
    def wrapped(adata, *args, **kwargs):
        obs_before = adata.obs.copy(deep=True)
        uns_before = copy.deepcopy(adata.uns)
        try:
            return func(adata, *args, **kwargs)
        except Exception:
            adata.obs = obs_before
            adata.uns = uns_before
            raise

    return wrapped


@_atomic_obs_uns
def estimate_dose(
    adata: AnnData,
    *,
    type_key: str | None = None,
    quantile: float | None = None,
    top_n: int | None = None,
    min_chi: float = 1e-6,
    max_type_mean: float | None = None,
    soup_quantile: float | None = None,
    min_genes: int = MIN_GENES,
    min_valid: int = MIN_VALID,
    evidence_mode: str = "positive_genes",
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
) -> np.ndarray:
    """Per-cell absolute ambient dose ``d_c = ρ_c · n_c``.

    ``type_key is None``: quantile floor on top soup genes.
    ``type_key`` set: unexpressed soup genes, dropping genes that look
    native in every type, then shrink log ρ within the sample. Fewer than
    two types uses the untyped quantile floor.

    Dose does not use cross-type gene protection.
    ``evidence_mode="exposure"`` shrinks with ``n_c · sum(χ_U)`` instead of
    the count of positive U genes.
    """
    _reject_view(adata, "estimate_dose")
    _require_raw_integer_counts(adata, layer=layer, fname="estimate_dose")
    _validate_chi_provenance(adata, sample_key=sample_key, layer=layer)
    if quantile is not None and not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if soup_quantile is not None and not 0 <= soup_quantile <= 1:
        raise ValueError("soup_quantile must be between 0 and 1")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1")
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1")
    if min_valid < 1:
        raise ValueError("min_valid must be at least 1")
    if evidence_mode not in {"positive_genes", "exposure"}:
        raise ValueError("evidence_mode must be 'positive_genes' or 'exposure'")
    if min_chi < 0:
        raise ValueError("min_chi must be nonnegative")
    if max_type_mean is not None and max_type_mean < 0:
        raise ValueError("max_type_mean must be nonnegative")
    if type_key is not None and type_key not in adata.obs.columns:
        raise KeyError(f"{type_key!r} missing on obs")
    if type_key is not None:
        _validated_type_values(adata, type_key)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    samples = _sample_names(adata, sample_key)
    if samples is None:
        _chi_vector(adata)
    else:
        for sample in pd.unique(samples):
            _chi_for_obs(adata, sample_key=sample_key, sample_name=sample)
    stale_prefixes = ("ambidose_mixture_", "ambidose_dose_mixture")
    stale_exact = {
        "ambidose_rho_mixture",
        "ambidose_rho_mixture_cell",
        "ambidose_rho_mixture_empty",
        "ambidose_dose_log2_ratio",
        "ambidose_dose_rho_gap",
        "ambidose_dose_disagreement",
        "ambidose_dose_diagnosis",
        "ambidose_dose_selected",
        "ambidose_d_raw",
        "ambidose_rho_raw",
        "ambidose_n_dose_genes",
        "ambidose_dose_fallback",
        "ambidose_one_type",
        "ambidose_shrink_w",
    }
    for col in list(adata.obs.columns):
        if col in stale_exact or col.startswith(stale_prefixes):
            del adata.obs[col]
    run = dict(adata.uns.get("ambidose", {}))
    run.pop("dose", None)
    adata.uns["ambidose"] = run
    if type_key is None:
        return _dose_quantile_from_chi(
            adata,
            quantile=quantile,
            top_n=top_n,
            min_chi=min_chi,
            droplet_key=droplet_key,
            cell_label=cell_label,
            layer=layer,
            sample_key=sample_key,
            min_valid=min_valid,
        )

    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["n_umi"] = n
    types = _validated_type_values(adata, type_key).to_numpy()

    rho_raw = np.full(adata.n_obs, np.nan, dtype=np.float64)
    n_valid = np.zeros(adata.n_obs, dtype=np.int64)
    evidence_exposure = np.zeros(adata.n_obs, dtype=np.float64)
    fallback = np.zeros(adata.n_obs, dtype=bool)
    one_type = np.zeros(adata.n_obs, dtype=bool)

    if samples is None:
        # None, not the string "_all": a real sample can be literally named
        # "_all" (pd.unique(samples) would then legitimately produce it),
        # and a string-sentinel comparison can't tell that case apart from
        # "no sample_key given" -- picking the wrong branch below and
        # silently using the global (not per-sample) chi for that sample.
        # None can never come from adata.obs[...].astype(str), so it can't
        # collide.
        groups: list = [None]
        sample_of = np.array([None] * adata.n_obs, dtype=object)
    else:
        groups = list(pd.unique(samples))
        sample_of = samples

    for s in groups:
        chi = _chi_for_obs(
            adata,
            sample_key=sample_key if s is not None else None,
            sample_name=s,
        )
        in_s = is_cell & (sample_of == s)
        sample_types = [t for t in pd.unique(types[in_s]) if t not in EMPTY_TYPES]
        # One type: U is "genes this library doesn't express," which is soup
        # plus noise. The type mean already contains the contamination, so the
        # type-aware floor goes to ρ≈1. Same estimator as type_key=None.
        empty_idx = np.flatnonzero((sample_of == s) & ~is_cell)
        if len(sample_types) < 2:
            idx = np.flatnonzero(in_s)
            if idx.size:
                rho_raw[idx], n_valid[idx] = _rho_from_chi(
                    x,
                    n,
                    chi,
                    idx,
                    top_n=top_n,
                    min_chi=min_chi,
                    quantile=quantile,
                    min_valid=min_valid,
                    empty_idx=empty_idx,
                )
                fallback[idx] = True
                one_type[idx] = True
            continue
        # Drop genes that look native in every type (above the empty-droplet
        # ceiling, or MALAT1-like collision). True soup sits at/below that
        # ceiling in every type and stays in the pool. See
        # _native_everywhere_mask.
        native_everywhere = _native_everywhere_mask(
            x, n, chi, types, in_s, max_type_mean=max_type_mean
        )
        for t in sample_types:
            idx = np.flatnonzero(in_s & (types == t))
            if idx.size == 0:
                continue
            mean = np.asarray(x[idx].mean(axis=0)).ravel()
            n_bar = float(n[idx].mean()) if idx.size else 0.0
            u_t = _soup_u_mask(
                mean,
                chi,
                idx.size,
                n_bar,
                max_type_mean=max_type_mean,
                min_chi=min_chi,
                native_everywhere=native_everywhere,
                apply_rt_and_collision=False,
            )
            cand = np.flatnonzero(u_t)
            fb = cand.size < min_genes
            if fb and soup_quantile is not None:
                soup = (
                    (chi >= np.quantile(chi, soup_quantile)) & (chi >= min_chi) & ~native_everywhere
                )
                gidx = _top_chi_indices(chi, np.flatnonzero(soup), top_n)
            else:
                gidx = _top_chi_indices(chi, cand, top_n)
            if fb:
                q = quantile
                if q is None:
                    q = _calibrate_ratio_quantile(x, n, chi, empty_idx, gidx)
                rho_c, nv = _quantile_rho_on_genes(
                    x,
                    n,
                    chi,
                    idx,
                    gidx,
                    quantile=q,
                )
            else:
                rho_c, nv = _mle_rho_on_genes(x, n, chi, idx, gidx)
                evidence_exposure[idx] = n[idx] * float(chi[gidx].sum())
            if evidence_mode == "positive_genes":
                rho_c[nv < min_valid] = np.nan
            rho_raw[idx] = rho_c
            n_valid[idx] = nv
            fallback[idx] = fb

    rho = np.zeros(adata.n_obs, dtype=np.float64)
    shrink_w = np.ones(adata.n_obs, dtype=np.float64)
    diag_samples: dict[str, dict] = {}
    for s in groups:
        s_label = "global library" if s is None else str(s)
        sample_id = _sample_storage_id(None if s is None else str(s))
        idx_all = np.flatnonzero(is_cell & (sample_of == s))
        fin = idx_all[np.isfinite(rho_raw[idx_all])]
        nan = idx_all[~np.isfinite(rho_raw[idx_all])]
        if fin.size == 0:
            rho[idx_all] = 0.0
            shrink_w[idx_all] = 0.0
            print(
                f"ambidose: no finite ρ_raw in sample {s_label!r}; dose set to 0", file=sys.stderr
            )
            diag_samples[sample_id] = {
                "n_cell": int(idx_all.size),
                "mu_log_rho": float("nan"),
                "tau2": float("nan"),
            }
            continue
        if one_type[idx_all].any():
            rho[idx_all] = np.clip(np.nan_to_num(rho_raw[idx_all], nan=0.0), 0.0, 1.0)
            shrink_w[idx_all] = 1.0
            d_s = rho[idx_all] * n[idx_all]
            diag_samples[sample_id] = {
                "n_cell": int(idx_all.size),
                "mu_log_rho": float("nan"),
                "tau2": float("nan"),
                "d_median": float(np.median(d_s)) if idx_all.size else 0.0,
                "rho_median": float(np.median(rho[idx_all])) if idx_all.size else 0.0,
                "frac_fallback": 1.0,
                "mean_shrink_w": 1.0,
                "mean_n_dose_genes": 0.0,
                "frac_rho_gt_095_small_n": 0.0,
            }
            continue
        theta = np.log(rho_raw[fin] + 1e-8)
        mu = float(np.median(theta))
        if evidence_mode == "exposure":
            evidence = evidence_exposure[fin]
        else:
            evidence = n_valid[fin].astype(np.float64)
        pos = evidence > 0
        k_shrink = float(np.median(evidence[pos])) if pos.any() else float(SHRINK_K)
        if k_shrink <= 0:
            k_shrink = float(SHRINK_K)
        w = evidence / (evidence + k_shrink)
        shrink_w[fin] = w
        rho[fin] = np.clip(np.exp(w * theta + (1.0 - w) * mu), 0.0, 1.0)
        if nan.size:
            rho[nan] = np.clip(np.exp(mu), 0.0, 1.0)
            shrink_w[nan] = 0.0
        mad = float(np.median(np.abs(theta - mu)))
        tau2 = (1.4826 * mad) ** 2
        d_s = rho[idx_all] * n[idx_all]
        small = (
            n[idx_all] <= np.quantile(n[idx_all], 0.1)
            if idx_all.size >= 10
            else np.zeros(idx_all.size, dtype=bool)
        )
        frac_hi = float(np.mean(rho[idx_all][small] > 0.95)) if small.any() else 0.0
        diag_samples[sample_id] = {
            "n_cell": int(idx_all.size),
            "mu_log_rho": mu,
            "tau2": tau2,
            "d_median": float(np.median(d_s)),
            "rho_median": float(np.median(rho[idx_all])),
            "frac_fallback": float(fallback[idx_all].mean()),
            "mean_shrink_w": float(shrink_w[idx_all].mean()),
            "shrink_k": k_shrink,
            "mean_n_dose_genes": float(n_valid[idx_all].mean()),
            "frac_rho_gt_095_small_n": frac_hi,
        }

    dose = rho * n
    d_raw = np.full(adata.n_obs, np.nan, dtype=np.float64)
    finite_raw = np.isfinite(rho_raw)
    d_raw[finite_raw] = rho_raw[finite_raw] * n[finite_raw]
    adata.obs[DOSE_KEY] = dose
    adata.obs[RHO_KEY] = rho
    adata.obs["ambidose_d_raw"] = d_raw
    adata.obs["ambidose_rho_raw"] = rho_raw
    adata.obs["ambidose_n_dose_genes"] = n_valid
    adata.obs["ambidose_dose_fallback"] = fallback
    adata.obs["ambidose_one_type"] = one_type
    adata.obs["ambidose_shrink_w"] = shrink_w
    for s in groups:
        sample_id = _sample_storage_id(None if s is None else str(s))
        diag_samples[sample_id]["sample"] = "" if s is None else str(s)
        diag_samples[sample_id]["sample_is_global"] = s is None
    uns = dict(adata.uns.get("ambidose", {}))
    uns["dose"] = {
        "method": "typed",
        "evidence_mode": evidence_mode,
        "samples": diag_samples,
    }
    uns["dose_type_key"] = type_key
    uns["dose_sample_key"] = sample_key
    uns["dose_provenance"] = _dose_provenance(
        type_key=type_key,
        sample_key=sample_key,
        layer=layer,
        droplet_key=droplet_key,
        cell_label=cell_label,
    )
    adata.uns["ambidose"] = uns
    return dose


def diagnose_dose_disagreement(
    adata: AnnData,
    *,
    fixed_dose: np.ndarray | None = None,
    mixture_dose: np.ndarray | None = None,
    fold_threshold: float | None = None,
    rho_gap_threshold: float | None = None,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
) -> dict[str, float | int]:
    """Use fixed dose on agreement and mixture dose on large disagreement.

    Default: mixture wins when |ρ gap| exceeds Poisson sampling error of
    the two estimates (``NOISE_K * sqrt(se_fixed² + se_mix²)``, se² = ρ/n).
    Explicit ``fold_threshold`` / ``rho_gap_threshold`` remain for tests.
    """
    _reject_view(adata, "diagnose_dose_disagreement")
    if fold_threshold is not None and fold_threshold <= 1:
        raise ValueError("fold_threshold must be > 1")
    if rho_gap_threshold is not None and not 0 <= rho_gap_threshold <= 1:
        raise ValueError("rho_gap_threshold must be between 0 and 1")
    standalone = fixed_dose is not None or mixture_dose is not None
    if not standalone:
        run = adata.uns.get("ambidose", {})
        fixed_provenance = run.get("dose_provenance")
        mixture_provenance = run.get("mixture_provenance")
        if fixed_provenance is None or mixture_provenance is None:
            raise ValueError("stored dose estimators lack provenance; recompute both estimators")
        if fixed_provenance != mixture_provenance:
            raise ValueError(
                "fixed and mixture doses were computed from different inputs; "
                "recompute both with identical type_key, sample_key, layer, "
                "droplet_key, and cell_label"
            )
        requested = {"layer": layer, "droplet_key": droplet_key, "cell_label": cell_label}
        mismatched = {
            key: (fixed_provenance.get(key), value)
            for key, value in requested.items()
            if fixed_provenance.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"dose diagnosis parameters do not match estimator provenance: {mismatched}"
            )
    if fixed_dose is None:
        if DOSE_KEY not in adata.obs:
            raise KeyError(f"{DOSE_KEY!r} missing; run estimate_dose first")
        fixed_dose = adata.obs[DOSE_KEY].to_numpy(dtype=np.float64)
    if mixture_dose is None:
        key = "ambidose_dose_mixture"
        if key not in adata.obs:
            raise KeyError(f"{key!r} missing; run estimate_dose_mixture first")
        mixture_dose = adata.obs[key].to_numpy(dtype=np.float64)
    fixed = np.asarray(fixed_dose, dtype=np.float64)
    mixture = np.asarray(mixture_dose, dtype=np.float64)
    if fixed.shape != (adata.n_obs,) or mixture.shape != (adata.n_obs,):
        raise ValueError("dose arrays must have length adata.n_obs")
    n = (
        np.asarray(_as_csr(adata.layers[layer] if layer is not None else adata.X).sum(axis=1))
        .ravel()
        .astype(np.float64)
    )
    for name, values in (("fixed_dose", fixed), ("mixture_dose", mixture)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"{name} must be nonnegative")
        if (values > n).any():
            raise ValueError(f"{name} must not exceed n_umi")
    fixed_rho = np.divide(fixed, n, out=np.zeros_like(fixed), where=n > 0)
    mixture_rho = np.divide(mixture, n, out=np.zeros_like(mixture), where=n > 0)
    eps = 1e-8
    log2_ratio = np.log2((fixed_rho + eps) / (mixture_rho + eps))
    rho_gap = np.abs(fixed_rho - mixture_rho)
    eligible = _resolve_cell_mask(adata, droplet_key, cell_label)
    abs_l2 = np.abs(log2_ratio)
    if fold_threshold is None and rho_gap_threshold is None:
        se_f = np.sqrt(np.maximum(fixed_rho, 0.0) / np.maximum(n, 1.0))
        se_m = np.sqrt(np.maximum(mixture_rho, 0.0) / np.maximum(n, 1.0))
        gap_cut_arr = NOISE_K * np.sqrt(se_f**2 + se_m**2)
        large = eligible & (rho_gap >= gap_cut_arr)
        fold_cut = float("nan")
        gap_cut = float(np.median(gap_cut_arr[eligible])) if eligible.any() else 0.0
    else:
        fold_cut = float(np.log2(fold_threshold)) if fold_threshold is not None else 0.0
        gap_cut = 0.0 if rho_gap_threshold is None else float(rho_gap_threshold)
        large = eligible & (abs_l2 >= fold_cut) & (rho_gap >= gap_cut)
    n_unconverged_blocked = 0
    if not standalone and "ambidose_mixture_status" in adata.obs:
        mixture_status = adata.obs["ambidose_mixture_status"].astype(str).to_numpy()
        usable = np.isin(mixture_status, ["fitted_converged", "quantile_fallback"])
        n_unconverged_blocked = int((large & ~usable).sum())
        large = large & usable
    direction = np.full(adata.n_obs, "agreement", dtype=object)
    direction[large & (fixed_rho > mixture_rho)] = "fixed_high"
    direction[large & (mixture_rho > fixed_rho)] = "mixture_high"
    adata.obs["ambidose_dose_log2_ratio"] = log2_ratio
    adata.obs["ambidose_dose_rho_gap"] = rho_gap
    adata.obs["ambidose_dose_disagreement"] = large
    adata.obs["ambidose_dose_diagnosis"] = direction
    adata.obs["ambidose_dose_selected"] = np.where(large, mixture, fixed)
    n_eligible = int(eligible.sum())
    summary: dict[str, float | int] = {
        "n_cells": n_eligible,
        "n_disagreement": int(large.sum()),
        "fraction_disagreement": float(large.sum() / n_eligible) if n_eligible else 0.0,
        "median_fixed_rho": float(np.median(fixed_rho[eligible])) if n_eligible else 0.0,
        "median_mixture_rho": float(np.median(mixture_rho[eligible])) if n_eligible else 0.0,
        "median_abs_rho_gap": float(np.median(rho_gap[eligible])) if n_eligible else 0.0,
        "n_fixed_high": int(np.sum(direction == "fixed_high")),
        "n_mixture_high": int(np.sum(direction == "mixture_high")),
        "n_disagreement_unconverged_kept_fixed": n_unconverged_blocked,
        "fold_cut_log2": float(fold_cut),
        "rho_gap_cut": float(gap_cut),
    }
    diag_columns = {
        "median_fixed_n_genes_disagreement": "ambidose_n_dose_genes",
        "median_fixed_shrink_w_disagreement": "ambidose_shrink_w",
        "mixture_converged_fraction_disagreement": "ambidose_mixture_converged",
        "median_mixture_profile_tv_disagreement": "ambidose_mixture_profile_tv",
        "median_mixture_empty_tv_disagreement": "ambidose_mixture_empty_tv",
    }
    for output_key, obs_key in diag_columns.items():
        if not standalone and obs_key in adata.obs:
            values = adata.obs[obs_key].to_numpy(dtype=np.float64)
            summary[output_key] = float(np.median(values[large])) if large.any() else 0.0
    if not standalone and "ambidose_dose_fallback" in adata.obs:
        fallback = adata.obs["ambidose_dose_fallback"].to_numpy(dtype=bool)
        summary["fixed_fallback_fraction_disagreement"] = (
            float(fallback[large].mean()) if large.any() else 0.0
        )
    return summary


@_atomic_obs_uns
def estimate_dose_adaptive(
    adata: AnnData,
    *,
    type_key: str,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    fold_threshold: float | None = None,
    rho_gap_threshold: float | None = None,
    quantile: float | None = None,
    top_n: int | None = None,
    min_chi: float = 1e-6,
    min_valid: int = MIN_VALID,
    evidence_mode: str = "exposure",
) -> np.ndarray:
    """Select fixed dose on agreement and mixture dose on large disagreement.

    Executed ``ambidose_d`` / ``ambidose_rho`` are the selected values
    (scale 1). ``q = median(ρ̂) n̄ / λ_e`` is still recorded per sample.
    ``ambidose_dose_selected`` is the same as executed dose.
    """
    estimate_dose(
        adata,
        type_key=type_key,
        droplet_key=droplet_key,
        cell_label=cell_label,
        layer=layer,
        sample_key=sample_key,
        quantile=quantile,
        top_n=top_n,
        min_chi=min_chi,
        min_valid=min_valid,
        evidence_mode=evidence_mode,
    )
    fixed_dose_meta = copy.deepcopy(adata.uns.get("ambidose", {}).get("dose", {}))
    estimate_dose_mixture(
        adata,
        type_key=type_key,
        droplet_key=droplet_key,
        cell_label=cell_label,
        layer=layer,
        sample_key=sample_key,
        fallback_quantile=quantile,
        fallback_top_n=top_n,
        fallback_min_chi=min_chi,
        fallback_min_valid=min_valid,
    )
    selection_summary = diagnose_dose_disagreement(
        adata,
        droplet_key=droplet_key,
        cell_label=cell_label,
        fold_threshold=fold_threshold,
        rho_gap_threshold=rho_gap_threshold,
        layer=layer,
    )
    selected = adata.obs["ambidose_dose_selected"].to_numpy(dtype=np.float64)
    n = np.asarray(
        _as_csr(adata.layers[layer] if layer is not None else adata.X).sum(axis=1)
    ).ravel()
    selected_rho = np.divide(selected, n, out=np.zeros_like(selected), where=n > 0)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    samples = _sample_names(adata, sample_key)
    sample_of = np.array([None] * adata.n_obs, dtype=object) if samples is None else samples
    groups = [None] if samples is None else list(pd.unique(samples))
    executed_rho = np.zeros_like(selected_rho)
    selected_samples = {}
    drop = (
        adata.obs[droplet_key].astype(str).to_numpy()
        if droplet_key is not None and droplet_key in adata.obs
        else None
    )
    stored_empty = dict(adata.uns.get("ambidose", {})).get("empty_umi", {})
    for sample in groups:
        mask = is_cell & (sample_of == sample)
        sample_id = _sample_storage_id(None if sample is None else str(sample))
        lam_e = float("nan")
        if drop is not None:
            empty = drop == "empty"
            if sample is not None:
                empty = empty & (sample_of == sample)
            if empty.any():
                lam_e = float(n[empty].mean())
        if not np.isfinite(lam_e) or lam_e <= 0:
            rec = stored_empty.get(sample_id)
            if rec is not None and rec.get("lam_e") is not None:
                lam_e = float(rec["lam_e"])
        if not np.isfinite(lam_e) or lam_e <= 0:
            raise ValueError(
                "empty-droplet mean UMI is required for q-scale; "
                "run estimate_chi with empty droplets present or set "
                "uns['ambidose']['empty_umi']"
            )
        hat = float(np.median(selected_rho[mask])) if mask.any() else 0.0
        n_bar = float(n[mask].mean()) if mask.any() else 0.0
        q = hat * n_bar / lam_e if n_bar > 0 else float("nan")
        # A valid library can contain only empty droplets after cell calling
        # (or after refinement). There is no cell-level q to record.
        scale = q_abs_scale(q, hat_rho=hat) if mask.any() else 1.0
        executed_rho[mask] = np.clip(selected_rho[mask] * scale, 0.0, 1.0)
        selected_samples[sample_id] = {
            "sample": "" if sample is None else str(sample),
            "sample_is_global": sample is None,
            "n_cell": int(mask.sum()),
            "d_median": float(np.median(executed_rho[mask] * n[mask])) if mask.any() else 0.0,
            "rho_median": float(np.median(executed_rho[mask])) if mask.any() else 0.0,
            "hat_rho": float(hat),
            "q": float(q),
            "q_scale": float(scale),
            "lam_e": float(lam_e),
        }
    executed = executed_rho * n
    adata.obs[DOSE_KEY] = executed
    adata.obs[RHO_KEY] = executed_rho
    uns = dict(adata.uns.get("ambidose", {}))
    uns["dose"] = {
        "evidence_mode": evidence_mode,
        "fixed": fixed_dose_meta,
        "selected": {"samples": selected_samples},
        "selection": selection_summary,
        "samples": selected_samples,
        "q_scale": {
            "method": "identity",
        },
    }
    adata.uns["ambidose"] = uns
    return executed


def estimate_dose_mixture(*args, **kwargs):
    """Wrapper so tests can patch ``ambidose._dose.estimate_dose_mixture``."""
    from ._mixture import estimate_dose_mixture as _impl

    return _impl(*args, **kwargs)
