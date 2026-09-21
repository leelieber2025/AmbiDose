"""Two-component mixture dose estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.optimize import nnls

from ._dose import (
    _dose_provenance,
    _rho_from_chi,
    _validate_chi_provenance,
)
from ._shared import (
    CHI_KEY,
    EMPTY_TYPES,
    MIN_VALID,
    _as_csr,
    _chi_for_obs,
    _reject_view,
    _require_raw_integer_counts,
    _resolve_cell_mask,
    _sample_names,
    _validated_type_values,
)


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
    fallback_quantile: float | None = None,
    fallback_top_n: int | None = None,
    fallback_min_chi: float = 1e-6,
    fallback_min_valid: int = MIN_VALID,
) -> np.ndarray:
    """Estimate dose from native and contamination profiles.

    One mixture fit per type. Ambient is empty-droplet χ with that type's
    NNLS share removed: ``(χ - π_t profile_t) / (1 - π_t)``, π_t ≤ 0.95.
    Samples with fewer than two types use the untyped χ quantile floor
    (``fallback_*``, same defaults as ``estimate_dose()``).
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
    if fallback_quantile is not None and not 0 <= fallback_quantile <= 1:
        raise ValueError("fallback_quantile must be between 0 and 1")
    if fallback_top_n is not None and fallback_top_n < 1:
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
                empty_idx=np.flatnonzero((sample_of == sample) & ~is_cell),
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
