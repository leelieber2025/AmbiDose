"""Ambient-dose estimation and adaptive estimator selection."""

from __future__ import annotations

import copy
import sys
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.optimize import nnls

from ._shared import (
    CHI_KEY,
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
            f"dose parameters do not match chi provenance: stored={stored}, requested={requested}"
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


NOISE_K = 2.0  # module constant; see _unexpressed_mask
# Housekeeping sits above this library fraction; true per-gene soup stays below.
U_MAX_LIBRARY_FRAC = 0.003
# soupOnly extra-clear only genes well below the ρ=1 ceiling. True soup
# has mean/(n̄χ) ≈ ρ (typically 0.01–0.3). Mid-ceiling leftover genes on
# zero-ambient libraries sit at 0.4–0.85 and must not be wiped.
SOUP_ONLY_MAX_RT = 0.4
# Type-level ρ below this: no soupOnly extra-clear, rank-1 only.
# Σd_c/Σn_c is the dose-weighted type ρ. Almost-uncontaminated types must
# not wipe U genes the same way heavily contaminated types do.
SOUP_ONLY_RHO_FLOOR = 0.01
# Sample-level executed scale s(q), q = median(ρ̂) n̄ / λ_e.
# Piecewise log-linear on natural-depth PBMC inject + realistic_gt;
# Cargnelli held out. Expand cap 1.25, shrink floor 0.50.
# docs/research/piecewise_caps_research_20260909.md
Q_SCALE_T = 316.59502562631263
Q_SCALE_A_LO = -1.4412479601313954
Q_SCALE_B_LO = 0.26081257417233045
Q_SCALE_A_HI = -1.5088953130300833
Q_SCALE_B_HI = 0.27637025546757915
Q_SCALE_EXPAND_CAP = 1.25
Q_SCALE_SHRINK_FLOOR = 0.50
# Blend shrink toward identity as median selected ρ̂ → 0.
# w = clip(ρ̂ / LOW_RHO, 0, 1); executed = (1-w)*1 + w*s(q) when s(q)<1.
# Hard cut at 0.10 jumped GSE (ρ̂=0.091) to s=1. Curve a,b,T unchanged.
Q_SCALE_LOW_RHO = 0.10


def _q_curve_scale(q: float) -> float:
    """Piecewise s(q) without the low-ρ̂ blend."""
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q!r}")
    if q <= 0:
        return 1.0
    if q < Q_SCALE_T:
        s = float(np.exp(Q_SCALE_A_LO + Q_SCALE_B_LO * np.log(q)))
        s = min(s, 1.0)
        s = max(s, Q_SCALE_SHRINK_FLOOR)
    else:
        s = float(np.exp(Q_SCALE_A_HI + Q_SCALE_B_HI * np.log(q)))
        s = max(s, 1.0)
        s = min(s, Q_SCALE_EXPAND_CAP)
    return float(np.clip(s, Q_SCALE_SHRINK_FLOOR, Q_SCALE_EXPAND_CAP))


def q_abs_scale(q: float, *, hat_rho: float) -> float:
    """Map unlabeled q = median(ρ̂) n̄ / λ_e to a sample executed scale.

    ``q<=0`` is a legitimate zero-ambient sample (median ρ̂ is exactly 0
    across the group), not an error: no ambient signal means there is
    nothing to expand or shrink, so the scale is the identity, 1.0 -- the
    executed dose stays 0 either way (``executed_rho = selected_rho *
    scale`` with ``selected_rho`` already 0). Only non-finite ``q`` (NaN/
    inf, an upstream data problem) still raises.

    Shrink is blended out as ``hat_rho → 0``: ``w = clip(hat_rho /
    Q_SCALE_LOW_RHO, 0, 1)``, executed ``(1-w) + w s(q)``. Expand
    (``s(q) >= 1``) is unchanged. High ρ̂ (Cargnelli) keeps the curve.
    """
    if not np.isfinite(hat_rho):
        raise ValueError(f"hat_rho must be finite, got {hat_rho!r}")
    s = _q_curve_scale(q)
    if s >= 1.0:
        return s
    w = float(np.clip(hat_rho / Q_SCALE_LOW_RHO, 0.0, 1.0))
    return float((1.0 - w) + w * s)


def _unexpressed_mask(
    mean: np.ndarray,
    expected: np.ndarray,
    n_cells_t: int,
    *,
    max_type_mean: float,
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
    if n_cells_t <= 0:
        return mean <= max_type_mean
    if margin_sd is not None:
        margin = noise_k * margin_sd / np.sqrt(n_cells_t)
    else:
        margin = noise_k * np.sqrt(np.maximum(expected, 0.0) / n_cells_t)
    ceiling = np.maximum(max_type_mean, expected + margin)
    return mean <= ceiling


def _soup_u_mask(
    mean: np.ndarray,
    chi: np.ndarray,
    n_cells: int,
    n_bar: float,
    *,
    max_type_mean: float,
    min_chi: float,
    native_everywhere: np.ndarray,
    apply_rt_and_collision: bool,
) -> np.ndarray:
    """Unexpressed ∩ χ-supported genes. Same ceiling as dose U evidence.

    ``apply_rt_and_collision`` adds subtract's mid-ceiling / abundance
    guards so extra-clear does not wipe housekeeping. Dose MLE evidence
    omits those guards (``False``).
    """
    expected = np.asarray(chi, dtype=np.float64) * float(n_bar)
    u = _unexpressed_mask(mean, expected, n_cells, max_type_mean=max_type_mean)
    u = u & ~np.asarray(native_everywhere, dtype=bool) & (chi >= min_chi)
    if apply_rt_and_collision and n_bar > 0:
        r_t = np.divide(mean, expected, out=np.zeros_like(mean), where=expected > 0)
        u = u & (r_t < SOUP_ONLY_MAX_RT)
        u = u & ~((mean > U_MAX_LIBRARY_FRAC * n_bar) & (r_t > 0.5))
        u = u & ~(r_t > 0.85)
    return u


def _top_chi_indices(chi: np.ndarray, candidates: np.ndarray, top_n: int) -> np.ndarray:
    """Return top χ candidates, including every tie at the boundary."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size <= top_n:
        return candidates
    values = chi[candidates]
    cutoff = np.partition(values, values.size - top_n)[values.size - top_n]
    return candidates[values >= cutoff]


def _rho_from_chi(
    x, n, chi, cell_idx, *, top_n: int, min_chi: float, quantile: float, min_valid: int = MIN_VALID
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile floor and its per-cell number of valid soup genes."""
    candidates = np.flatnonzero(chi >= min_chi)
    gidx = _top_chi_indices(chi, candidates, top_n)
    rho, n_valid = _quantile_rho_on_genes(x, n, chi, cell_idx, gidx, quantile=quantile)
    rho[n_valid < min_valid] = 0.0
    return np.nan_to_num(rho, nan=0.0), n_valid


def _dose_quantile_from_chi(
    adata: AnnData,
    *,
    quantile: float = 0.15,
    top_n: int = 100,
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
            rho[idx], _ = _rho_from_chi(
                x,
                n,
                chi,
                idx,
                top_n=top_n,
                min_chi=min_chi,
                quantile=quantile,
                min_valid=min_valid,
            )
    else:
        chi = _chi_vector(adata)
        idx = np.flatnonzero(is_cell)
        rho[idx], _ = _rho_from_chi(
            x,
            n,
            chi,
            idx,
            top_n=top_n,
            min_chi=min_chi,
            quantile=quantile,
            min_valid=min_valid,
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
    max_type_mean: float,
) -> np.ndarray:
    """Drop genes that look native in every type, not true soup.

    Empty-droplet soup sits at or below ``n̄ χ`` in every type and is kept.
    Housekeeping exceeds that ceiling in every type. Genes that sit on the
    ceiling in every type *and* take a large library fraction (MALAT1-like
    collision) stay unidentifiable and are also dropped. Cross-type fold
    is not used: true soup is uniform by construction.
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
        # r_t near 1 is the ρ=1 ceiling (housekeeping / MALAT1). True soup
        # has r_t ≈ ρ, which is below this even when contamination is high.
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
    quantile: float = 0.15,
    top_n: int = 100,
    min_chi: float = 1e-6,
    max_type_mean: float = 0.05,
    soup_quantile: float = 0.75,
    min_genes: int = MIN_GENES,
    min_valid: int = MIN_VALID,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
) -> np.ndarray:
    """Per-cell absolute ambient dose ``d_c = ρ_c · n_c``.

    ``type_key is None``: quantile floor on top soup genes (ablation).
    ``type_key`` set: unexpressed ∩ soup genes, dropping genes that look
    native (or ceiling-collision) in every type, then shrink log ρ within
    sample. A sample with fewer than two types uses the untyped quantile
    floor: the type-aware U set is not identified from one contaminated
    mean (see ``_native_everywhere_mask``).

    Dose estimation deliberately does not use cross-type gene protection;
    local protection decisions must not perturb the shared per-cell dose.
    """
    _reject_view(adata, "estimate_dose")
    _require_raw_integer_counts(adata, layer=layer, fname="estimate_dose")
    _validate_chi_provenance(adata, sample_key=sample_key, layer=layer)
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if not 0 <= soup_quantile <= 1:
        raise ValueError("soup_quantile must be between 0 and 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1")
    if min_valid < 1:
        raise ValueError("min_valid must be at least 1")
    if min_chi < 0:
        raise ValueError("min_chi must be nonnegative")
    if max_type_mean < 0:
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
            if fb:
                soup = (
                    (chi >= np.quantile(chi, soup_quantile)) & (chi >= min_chi) & ~native_everywhere
                )
                gidx = np.flatnonzero(soup)
            else:
                gidx = cand
            gidx = _top_chi_indices(chi, gidx, top_n)
            # 15% quantile is SoupX protection for a contaminated gene set.
            # On a clean U_t it sits well below E[y/(nχ)]=ρ. Poisson MLE
            # uses zeros and matches true d·χ on barnyard.
            if fb:
                rho_c, nv = _quantile_rho_on_genes(
                    x,
                    n,
                    chi,
                    idx,
                    gidx,
                    quantile=quantile,
                )
            else:
                rho_c, nv = _mle_rho_on_genes(x, n, chi, idx, gidx)
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
        w = n_valid[fin] / (n_valid[fin] + SHRINK_K)
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
    uns["dose"] = {"method": "typed", "samples": diag_samples}
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


def _tv_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _simplex(values: np.ndarray, pseudocount: float) -> np.ndarray:
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    profile = np.asarray(values, dtype=np.float64) + pseudocount
    if not np.isfinite(profile).all() or np.any(profile < 0):
        raise ValueError("mixture profile must be finite and nonnegative")
    total = profile.sum()
    if total <= 0:
        raise ValueError("mixture profile has non-positive mass")
    return profile / total


def _mixture_responsibility(coo, local_rho, native, ambient) -> np.ndarray:
    r = local_rho[coo.row]
    ambient_mass = r * ambient[coo.col]
    native_mass = (1.0 - r) * native[coo.col]
    denom = ambient_mass + native_mass
    return np.divide(ambient_mass, denom, out=np.zeros_like(ambient_mass), where=denom > 0)


def _two_component_mixture_em(
    *,
    coo,
    n_cells: np.ndarray,
    n_vars: int,
    native: np.ndarray,
    ambient: np.ndarray,
    max_iter: int,
    convergence: float,
    initial_rho: float,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray]:
    """Fit per-cell ρ with a locked ambient profile and an updating native profile."""
    local_rho = np.full(n_cells.size, initial_rho, dtype=np.float64)
    native = np.asarray(native, dtype=np.float64).copy()
    ambient = np.asarray(ambient, dtype=np.float64)
    did_converge = False
    n_iter = 0
    responsibility = np.zeros(coo.data.shape[0], dtype=np.float64)
    for step in range(1, max_iter + 1):
        n_iter = step
        responsibility = _mixture_responsibility(coo, local_rho, native, ambient)
        ambient_by_cell = np.bincount(
            coo.row, weights=coo.data * responsibility, minlength=n_cells.size
        )
        updated_rho = np.clip(
            np.divide(
                ambient_by_cell, n_cells, out=np.zeros_like(ambient_by_cell), where=n_cells > 0
            ),
            0.0,
            1.0,
        )
        native_counts = np.bincount(
            coo.col, weights=coo.data * (1.0 - responsibility), minlength=n_vars
        )
        updated_native = _simplex(native_counts, pseudocount)
        rho_delta = float(np.max(np.abs(updated_rho - local_rho)))
        native_delta = float(0.5 * np.abs(updated_native - native).sum())
        local_rho = updated_rho
        native = updated_native
        if max(rho_delta, native_delta) < convergence:
            did_converge = True
            break
    responsibility = _mixture_responsibility(coo, local_rho, native, ambient)
    inferred = np.zeros(n_vars, dtype=np.float64)
    ambient_gene_counts = np.bincount(coo.col, weights=coo.data * responsibility, minlength=n_vars)
    ambient_total = ambient_gene_counts.sum()
    if ambient_total > 0:
        inferred = ambient_gene_counts / ambient_total
    return local_rho, native, n_iter, did_converge, inferred


def _mixture_loglik(
    coo, n_cells: np.ndarray, local_rho: np.ndarray, native: np.ndarray, ambient: np.ndarray
) -> np.ndarray:
    """Per-cell log-likelihood of a fitted two-component mixture's own data."""
    r = local_rho[coo.row]
    mix = r * ambient[coo.col] + (1.0 - r) * native[coo.col]
    return np.bincount(
        coo.row,
        weights=coo.data * np.log(np.maximum(mix, 1e-300)),
        minlength=n_cells.size,
    )


def _type_residual_score(
    adata: AnnData,
    *,
    is_cell: np.ndarray,
    type_key: str,
    sample_key: str | None,
    layer: str | None,
    max_iter: int = 500,
    convergence: float = 1e-3,
    pseudocount: float = 1e-8,
) -> np.ndarray:
    """Per-cell heterotypic-doublet-vs-soup log-likelihood-ratio diagnostic.

    Fits two locked-ambient two-component mixtures per cell: the native
    profile is always this cell's own coarse type; the ambient side is
    either the leave-one-type pooled profile of every *other* coarse type
    in the same sample (doublet hypothesis: excess mass is a second cell
    program), or empty-droplet chi (soup hypothesis: excess mass is
    ambient-shaped). The returned score is the per-cell log-likelihood
    difference, other-type fit minus chi fit: positive leans doublet-
    shaped, negative leans soup-shaped. It is diagnostic only -- it does
    not feed rho/dose, and nothing here relabels droplets. NaN where chi
    or a second type is unavailable for that cell's sample (fewer than two
    coarse types, or no empty-droplet chi estimated).
    """
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    types = _validated_type_values(adata, type_key).to_numpy()
    samples = _sample_names(adata, sample_key)
    groups = [None] if samples is None else list(pd.unique(samples))
    sample_of = np.array([None] * adata.n_obs, dtype=object) if samples is None else samples
    score = np.full(adata.n_obs, np.nan, dtype=np.float64)
    for sample in groups:
        chi = None
        if CHI_KEY in adata.var.columns or CHI_KEY in adata.uns:
            chi = _chi_for_obs(
                adata, sample_key=sample_key if sample is not None else None, sample_name=sample
            )
        if chi is None:
            continue
        in_sample = is_cell & (sample_of == sample)
        sample_types = [t for t in pd.unique(types[in_sample]) if t not in EMPTY_TYPES]
        if len(sample_types) < 2:
            continue
        type_sums = {
            t: np.asarray(x[in_sample & (types == t)].sum(axis=0)).ravel().astype(np.float64)
            for t in sample_types
        }
        type_profiles = {t: _simplex(type_sums[t], pseudocount) for t in sample_types}
        total_profile = np.sum(list(type_sums.values()), axis=0)
        chi_ambient = _simplex(chi, pseudocount)
        for cell_type in sample_types:
            idx = np.flatnonzero(in_sample & (types == cell_type))
            idx = idx[n[idx] > 0]
            if idx.size == 0:
                continue
            coo = x[idx].tocoo()
            native_init = type_profiles[cell_type]
            other_ambient = _simplex(total_profile - type_sums[cell_type], pseudocount)
            kw = {
                "coo": coo,
                "n_cells": n[idx],
                "n_vars": adata.n_vars,
                "native": native_init,
                "max_iter": max_iter,
                "convergence": convergence,
                "initial_rho": 0.5,
                "pseudocount": pseudocount,
            }
            other_fit = _two_component_mixture_em(ambient=other_ambient, **kw)
            chi_fit = _two_component_mixture_em(ambient=chi_ambient, **kw)
            ll_other = _mixture_loglik(coo, n[idx], other_fit[0], other_fit[1], other_ambient)
            ll_chi = _mixture_loglik(coo, n[idx], chi_fit[0], chi_fit[1], chi_ambient)
            score[idx] = ll_other - ll_chi
    return score


def estimate_dose_mixture(
    adata: AnnData,
    *,
    type_key: str,
    max_iter: int = 500,
    convergence: float = 1e-3,
    initial_rho: float = 0.5,
    pseudocount: float = 1e-8,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    fallback_quantile: float = 0.15,
    fallback_top_n: int = 100,
    fallback_min_chi: float = 1e-6,
    fallback_min_valid: int = MIN_VALID,
) -> np.ndarray:
    """Estimate dose from native and contamination profiles.

    One mixture fit per type, on an ambient reference derived from
    empty-droplet χ whenever χ is available:

    - NNLS-regress χ against the sample's type-profile matrix
      (χ ≈ Σ_t π_t · profile_t, π ≥ 0, renormalized to sum to 1) to
      estimate each type's own share of what actually shows up in empty
      droplets.
    - That type's ambient reference is χ with its own estimated share
      subtracted back out: ``(χ - π_t·profile_t) / (1 - π_t)``, clipped to
      stay nonnegative (π_t capped at 0.95).

    This replaces the leave-one-type ("cell", DecontX-style) ambient this
    function used before 2026-09-07: giving every type in a sample the
    same complement-of-everyone-else profile makes each type's fit see a
    structurally different contamination hypothesis, which introduces
    type-linked bias into ρ that has no counterpart in true injected
    contamination. Estimating each type's actual self-contamination share
    from χ directly, instead of assuming it, removes most of that bias
    while improving native retention and, on both barnyard datasets,
    specificity/precision at essentially unchanged sensitivity. Full
    rationale, the leave-one-type counterfactual, and the before/after
    evaluation across all manuscript datasets are in
    docs/fig1_design_gap_audit_20260907.md and the 2026-09-07 DEVLOG
    entries ("gap-3 fix merged" and the counterfactual confirmation above
    it).

    A sample with fewer than two types has no type-profile matrix to
    regress against: the χ mixture EM then fits a contaminated type mean
    against χ and reports ρ≈1. Those samples use the untyped
    quantile-floor on χ instead (``fallback_*`` controls that floor --
    same knobs and defaults as ``estimate_dose()``'s own untyped path,
    exposed explicitly here rather than hardcoded, so the two don't
    silently drift apart). Empty droplets remain the independent
    composition used by ``subtract()``.
    """
    _reject_view(adata, "estimate_dose_mixture")
    _require_raw_integer_counts(adata, layer=layer, fname="estimate_dose_mixture")
    _validate_chi_provenance(adata, sample_key=sample_key, layer=layer)
    if type_key not in adata.obs.columns:
        raise KeyError(f"{type_key!r} missing on obs")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if convergence <= 0:
        raise ValueError("convergence must be positive")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    if not 0 < initial_rho < 1:
        raise ValueError("initial_rho must be strictly between 0 and 1")
    if not 0 <= fallback_quantile <= 1:
        raise ValueError("fallback_quantile must be between 0 and 1")
    if fallback_top_n < 1:
        raise ValueError("fallback_top_n must be at least 1")
    if fallback_min_valid < 1:
        raise ValueError("fallback_min_valid must be at least 1")
    if fallback_min_chi < 0:
        raise ValueError("fallback_min_chi must be nonnegative")
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    nnz_per_cell = np.diff(x.indptr)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    types = _validated_type_values(adata, type_key).to_numpy()
    samples = _sample_names(adata, sample_key)
    groups = [None] if samples is None else list(pd.unique(samples))
    sample_of = np.array([None] * adata.n_obs, dtype=object) if samples is None else samples
    rho = np.zeros(adata.n_obs, dtype=np.float64)
    rho_cell = np.zeros(adata.n_obs, dtype=np.float64)
    rho_empty = np.full(adata.n_obs, np.nan, dtype=np.float64)
    iterations = np.zeros(adata.n_obs, dtype=np.int64)
    converged = pd.array([pd.NA] * adata.n_obs, dtype="boolean")
    profile_tv = np.zeros(adata.n_obs, dtype=np.float64)
    empty_tv = np.full(adata.n_obs, np.nan, dtype=np.float64)
    profile = np.full(adata.n_obs, "not_evaluated", dtype=object)
    status = np.full(adata.n_obs, "not_evaluated", dtype=object)
    n_genes_ev = np.zeros(adata.n_obs, dtype=np.int64)
    for sample in groups:
        chi = None
        if CHI_KEY in adata.var.columns or CHI_KEY in adata.uns:
            chi = _chi_for_obs(
                adata, sample_key=sample_key if sample is not None else None, sample_name=sample
            )
        in_sample = is_cell & (sample_of == sample)
        sample_types = [t for t in pd.unique(types[in_sample]) if t not in EMPTY_TYPES]
        type_sums = {
            t: np.asarray(x[in_sample & (types == t)].sum(axis=0)).ravel().astype(np.float64)
            for t in sample_types
        }
        type_profiles = {t: _simplex(type_sums[t], pseudocount) for t in sample_types}
        total_profile = (
            np.sum(list(type_sums.values()), axis=0)
            if sample_types
            else np.zeros(adata.n_vars, dtype=np.float64)
        )
        empty_ambient = _simplex(chi, pseudocount) if chi is not None else None
        if len(sample_types) < 2:
            idx = np.flatnonzero(in_sample)
            idx = idx[n[idx] > 0]
            if idx.size == 0:
                continue
            if chi is None:
                raise ValueError(
                    "empty-droplet χ is required for a single-type library "
                    "(leave-one-type ambient is empty); run estimate_chi first"
                )
            rho_q, n_valid_q = _rho_from_chi(
                x,
                n,
                chi,
                idx,
                top_n=fallback_top_n,
                min_chi=fallback_min_chi,
                quantile=fallback_quantile,
                min_valid=fallback_min_valid,
            )
            rho[idx] = rho_q
            rho_empty[idx] = rho_q
            profile[idx] = "empty"
            status[idx] = "quantile_fallback"
            n_genes_ev[idx] = n_valid_q
            continue
        pi: dict[object, float] = {}
        if empty_ambient is not None:
            basis = np.column_stack([type_profiles[t] for t in sample_types])
            coef, _resid = nnls(basis, empty_ambient)
            total_coef = float(coef.sum())
            if total_coef > 0:
                coef = coef / total_coef
            pi = {t: float(coef[j]) for j, t in enumerate(sample_types)}
        for cell_type in sample_types:
            idx = np.flatnonzero(in_sample & (types == cell_type))
            idx = idx[n[idx] > 0]
            if idx.size == 0:
                continue
            coo = x[idx].tocoo()
            native_init = type_profiles[cell_type]
            if empty_ambient is not None:
                pi_t = min(pi.get(cell_type, 0.0), 0.95)
                self_removed = empty_ambient - pi_t * type_profiles[cell_type]
                ambient_t = _simplex(np.clip(self_removed, 0.0, None), pseudocount)
                selected_profile = "chi_deconv"
            else:
                ambient_t = _simplex(total_profile - type_sums[cell_type], pseudocount)
                pi_t = float("nan")
                selected_profile = "cell"
            fit = _two_component_mixture_em(
                coo=coo,
                n_cells=n[idx],
                n_vars=adata.n_vars,
                native=native_init,
                ambient=ambient_t,
                max_iter=max_iter,
                convergence=convergence,
                initial_rho=initial_rho,
                pseudocount=pseudocount,
            )
            selected_rho, selected_native, n_iter, did_converge, _inferred = fit
            rho_cell[idx] = selected_rho
            rho_empty[idx] = selected_rho
            rho[idx] = selected_rho
            if empty_ambient is not None:
                empty_tv[idx] = _tv_distance(selected_native, empty_ambient)
            iterations[idx] = n_iter
            converged[idx] = did_converge
            status[idx] = "fitted_converged" if did_converge else "fitted_unconverged"
            profile_tv[idx] = _tv_distance(selected_native, ambient_t)
            profile[idx] = selected_profile
            n_genes_ev[idx] = nnz_per_cell[idx]
    dose = rho * n
    adata.obs["ambidose_dose_mixture"] = dose
    adata.obs["ambidose_rho_mixture"] = rho
    adata.obs["ambidose_dose_mixture_cell"] = rho_cell * n
    adata.obs["ambidose_rho_mixture_cell"] = rho_cell
    adata.obs["ambidose_dose_mixture_empty"] = rho_empty * n
    adata.obs["ambidose_rho_mixture_empty"] = rho_empty
    adata.obs["ambidose_mixture_profile"] = pd.Categorical(profile)
    adata.obs["ambidose_mixture_status"] = pd.Categorical(status)
    adata.obs["ambidose_mixture_n_genes"] = n_genes_ev
    adata.obs["ambidose_mixture_iterations"] = iterations
    adata.obs["ambidose_mixture_converged"] = converged
    adata.obs["ambidose_mixture_profile_tv"] = profile_tv
    adata.obs["ambidose_mixture_empty_tv"] = empty_tv
    uns = dict(adata.uns.get("ambidose", {}))
    uns["mixture_provenance"] = _dose_provenance(
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
    fold_threshold: float = 2.0,
    rho_gap_threshold: float = 0.10,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
) -> dict[str, float | int]:
    """Use fixed dose on agreement and mixture dose on large disagreement."""
    _reject_view(adata, "diagnose_dose_disagreement")
    if fold_threshold <= 1:
        raise ValueError("fold_threshold must be > 1")
    if not 0 <= rho_gap_threshold <= 1:
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
    large = (
        eligible & (np.abs(log2_ratio) >= np.log2(fold_threshold)) & (rho_gap >= rho_gap_threshold)
    )
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


def estimate_dose_adaptive(
    adata: AnnData,
    *,
    type_key: str,
    droplet_key: str = DROPLET_KEY,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    fold_threshold: float = 2.0,
    rho_gap_threshold: float = 0.10,
    quantile: float = 0.15,
    top_n: int = 100,
    min_chi: float = 1e-6,
    min_valid: int = MIN_VALID,
) -> np.ndarray:
    """Select fixed dose on agreement and mixture dose on large disagreement.

    Executed ``ambidose_d`` / ``ambidose_rho`` are the selected values
    multiplied by the sample-level scale ``s(q)``, ``q = median(ρ̂) n̄ / λ_e``.
    Shrink is blended toward 1 as median selected ρ̂ approaches 0.
    ``ambidose_dose_selected`` remains the unscaled estimator output.
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
        scale = q_abs_scale(q, hat_rho=hat)
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
        "fixed": fixed_dose_meta,
        "selected": {"samples": selected_samples},
        "selection": selection_summary,
        "samples": selected_samples,
        "q_scale": {
            "T": Q_SCALE_T,
            "expand_cap": Q_SCALE_EXPAND_CAP,
            "shrink_floor": Q_SCALE_SHRINK_FLOOR,
            "low_rho": Q_SCALE_LOW_RHO,
            "low_rho_blend": True,
        },
    }
    adata.uns["ambidose"] = uns
    return executed
