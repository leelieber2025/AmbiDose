"""Empty-droplet calling, ambient profile, and per-cell dose subtraction."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ._budget import (
    CEILING_CROSS_TYPE_GATE,
    ENRICH_STRENGTH,
    _apply_dose_enrichment,
    _cap_take_to_remaining,
    _confidence_weighted_take,
    _expand_take_to_cells,
    _high_chi_u_mask,
    _integerize_corrected,
    _pre_enrich_sat_mask,
    _realloc_unspent_rank1,
    _revoke_u_with_expressing_subset,
    _selected_data_positions,
)
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
    SOUP_ONLY_RHO_FLOOR,
    _native_everywhere_mask,
    _soup_u_mask,
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
    _type_residual_score as _type_residual_score,
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
from ._droplets import (
    _barcode_knee_umi as _barcode_knee_umi,
)
from ._droplets import (
    _barcode_rank_curve as _barcode_rank_curve,
)
from ._droplets import (
    _bh_fdr as _bh_fdr,
)
from ._droplets import (
    _closer_to_cells_than_chi as _closer_to_cells_than_chi,
)
from ._droplets import (
    _diem_keep as _diem_keep,
)
from ._droplets import (
    _empty_drops_keep as _empty_drops_keep,
)
from ._droplets import (
    _good_turing_proportions as _good_turing_proportions,
)
from ._droplets import (
    _has_variable_gene as _has_variable_gene,
)
from ._droplets import (
    _maybe_cut_inflated_mixture as _maybe_cut_inflated_mixture,
)
from ._droplets import (
    _multinomial_mc_pvals as _multinomial_mc_pvals,
)
from ._droplets import (
    _normalized_barcode_map,
    call_cells,
    classify_droplets,
)
from ._droplets import (
    _whitelist_vs_chi_keep as _whitelist_vs_chi_keep,
)
from ._droplets import mark_doublets as mark_doublets
from ._ownership import (
    _ceiling_cross_type_gate,
    _cross_type_anchor_mask,
    _dominant_owner_masks,
    _mt_gene_mask,
    _p_set_is_soup_like,
    _restrict_high_chi_to_single_winner,
    _type_masks,
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
    DOSE_KEY,
    DROPLET_KEY,
    EMPTY_TYPE,
    EMPTY_TYPES,
    LAYER_OUT,
    MIN_GENES,
    MIN_TYPE_CELLS,
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
    UNDER_EXECUTION_RATIO,
    _as_csr,
    _chi_for_obs,
    _chi_vector,
    _configure_scanpy_n_jobs,
    _profile,
    _reject_view,
    _require_raw_integer_counts,
    _resolve_cell_mask,
    _same_matrix,
    _sample_names,
    _sample_storage_id,
    _StageProgress,
    _validate_chi_frame,
    _validate_output_layer,
    _validated_sample_values,
    _validated_type_values,
    raw_count_matrix,
    require_run_keys,
)
from ._shared import (
    CLUSTER_KEY as CLUSTER_KEY,
)
from ._shared import (
    LEIDEN_RESOLUTION_COARSE as LEIDEN_RESOLUTION_COARSE,
)
from ._shared import (
    LEIDEN_RESOLUTION_FINE as LEIDEN_RESOLUTION_FINE,
)
from ._shared import (
    LEIDEN_RESOLUTION_MEDIUM as LEIDEN_RESOLUTION_MEDIUM,
)
from ._shared import (
    MIX_INFLATION_RATIO as MIX_INFLATION_RATIO,
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
    _mc_worker_count as _mc_worker_count,
)
from ._shared import (
    _require_ram as _require_ram,
)
from ._shared import (
    _thread_worker_count as _thread_worker_count,
)
from ._typing import _annotate_coarse_types_single as _annotate_coarse_types_single
from ._typing import (
    _default_type_key,
    resolve_type_key,
)
from ._typing import _embed_coarse_hvg as _embed_coarse_hvg
from ._typing import _resolve_coarse_resolution as _resolve_coarse_resolution
from ._typing import _type_sample_dependence as _type_sample_dependence


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
        n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
        run["chi_provenance"] = {
            "sample_key": sample_key,
            "layer": layer,
            "droplet_key": droplet_key,
            "empty_label": empty_label,
        }
        run["empty_umi"] = {
            _sample_storage_id(None): {
                "sample": "",
                "lam_e": float(n[empty].mean()),
                "n_empty": int(empty.sum()),
            }
        }
        adata.uns["ambidose"] = run
        return chi

    if not adata.var_names.is_unique:
        raise ValueError("multi-sample estimate_chi requires unique var_names for gene alignment")
    samples = _validated_sample_values(adata, sample_key)
    names: list[str] = []
    rows: list[np.ndarray] = []
    empty_umi: dict[str, dict] = {}
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    sample_arr = samples.to_numpy()
    for name in samples.unique():
        mask = empty & (sample_arr == name)
        if int(mask.sum()) < min_empty:
            raise ValueError(
                f"sample {name!r}: need {min_empty} empty droplets, got {int(mask.sum())}"
            )
        names.append(str(name))
        rows.append(_profile(x[mask], sample=str(name)))
        empty_umi[_sample_storage_id(str(name))] = {
            "sample": str(name),
            "lam_e": float(n[mask].mean()),
            "n_empty": int(mask.sum()),
        }
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
    run["empty_umi"] = empty_umi
    adata.uns["ambidose"] = run
    return chi


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
    along χ. Unowned rank-1 and soupOnly takes are then scaled toward
    ``take × max(Pearson(y/n, ρ), 0)`` with blend ``ENRICH_STRENGTH``
    (default 0.1), so a type-level budget that has already saturated
    observed capacity cannot fully wipe cells whose counts are independent
    of dose. Unowned rank-1 genes that already meet observed capacity
    before that blend stay on the type-level integer allocator; the rest
    of the rank-1 take is spent inside each cell so fractional χ-budget is
    not taken from other cells of the same type. Unexpressed unowned genes are extra-cleared at the type's
    observed total when the type's dose-weighted ρ is at least
    ``SOUP_ONLY_RHO_FLOOR``, but only up to each cell's remaining
    ``d_c`` after rank-1. Type-aware subtraction preserves
    native programs; continuous residual mode is available only through
    ``clip_negative=False``.
    """
    _validate_output_layer(layer=layer, layer_out=layer_out)
    _reject_view(adata, "subtract")
    if clip_negative:
        _require_raw_integer_counts(adata, layer=layer, fname="subtract")
    src = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    x = src.astype(np.float64, copy=True)
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
            dominant_masks, dominant_masks_sw, n_meta_s = _dominant_owner_masks(
                x,
                n,
                types_s_all,
                type_means,
                type_indices=type_indices_s,
                cell_keys=obs_keys,
                max_type_mean=max_type_mean,
                also_single_winner=True,
            )
            dominant_masks = _restrict_high_chi_to_single_winner(
                dominant_masks, dominant_masks_sw, chi
            )
            sample_n_meta = int(n_meta_s)
            ceiling_gate_s: dict[str, np.ndarray] | None = None
            if CEILING_CROSS_TYPE_GATE:
                ceiling_gate_s = _ceiling_cross_type_gate(n, chi, type_means_s, type_indices_s)
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
                y_cl = np.asarray(x[idx].sum(axis=0)).ravel().astype(np.float64)
                leftover_cap = None
                if t in EMPTY_TYPES:
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                    is_p = np.zeros(adata.n_vars, dtype=bool)
                    native_confidence = np.zeros(adata.n_vars)
                    r_t = np.ones(adata.n_vars)
                elif idx.size < MIN_TYPE_CELLS:
                    # Fragments smaller than MIN_TYPE_CELLS skip extra-clear
                    # and take only the protected rank-1 slice.
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                    is_p = np.ones(adata.n_vars, dtype=bool)
                    native_confidence = np.ones(adata.n_vars)
                    r_t = np.ones(adata.n_vars)
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
                    if ceiling_gate_s is not None:
                        mask_kw["ceiling_gate"] = ceiling_gate_s.get(
                            t, np.ones(adata.n_vars, dtype=bool)
                        )
                    is_u, is_p, native_confidence, r_t = _type_masks(x, n, chi, idx, **mask_kw)
                    if relax_hk_when_soup_like and _p_set_is_soup_like(x, n, chi, idx, is_p):
                        is_u, is_p, native_confidence, r_t = _type_masks(
                            x, n, chi, idx, **mask_kw, collision_exception=False
                        )
                    mean_t = np.asarray(x[idx].mean(axis=0)).ravel()
                    n_bar_t = float(n[idx].mean()) if idx.size else 0.0
                    is_u = is_u & _soup_u_mask(
                        mean_t,
                        chi,
                        idx.size,
                        n_bar_t,
                        max_type_mean=max_type_mean,
                        min_chi=min_chi,
                        native_everywhere=native_everywhere,
                        apply_rt_and_collision=True,
                    )
                    is_u = _revoke_u_with_expressing_subset(x, idx, n, chi, is_u)
                    # The gap-cascade's wider ownership (dominant_masks) can
                    # protect more genes than the frozen single-winner rule
                    # (dominant_masks_sw) would have, which frees up more of
                    # d_sum as "unspent" -- _realloc_unspent_rank1 then
                    # concentrates that extra leftover onto whatever else is
                    # still unprotected, over-correcting it past what the
                    # already-validated single-winner baseline ever did.
                    # Compute what leftover the single-winner rule would have
                    # produced and use it as a hard cap below, so the wider
                    # ownership's benefit is never paid for by pushing
                    # realloc's total footprint past the frozen baseline.
                    exclude_sw = (
                        dominant_masks_sw.get(t, np.zeros(adata.n_vars, dtype=bool))
                        | mt_mask
                        | extra_mask
                        | native_everywhere
                    )
                    is_u_sw, _is_p_sw, conf_sw, _r_t_sw = _type_masks(
                        x, n, chi, idx, **{**mask_kw, "exclude": exclude_sw}
                    )
                    take_sw = _confidence_weighted_take(y_cl, chi, d_sum, conf_sw, None)
                    take_sw = np.where(is_u_sw, 0.0, take_sw)
                    leftover_cap = max(0.0, d_sum - float(np.sum(take_sw)))
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
                # soupOnly extra-clears unexpressed unowned genes when rho_t
                # is above the floor. High-χ U (χ-mass prefix 0.8) is
                # unbounded; low-χ U is capped at remaining d_c after rank-1
                # and high-χ extra-clear.
                take_rank1 = _confidence_weighted_take(y_cl, chi, d_sum, native_confidence, None)
                take_rank1 = np.where(is_u, 0.0, take_rank1)
                take_rank1 = _realloc_unspent_rank1(
                    take_rank1, y_cl, chi, d_sum, is_p, is_u, r_t, leftover_cap=leftover_cap
                )
                take_rank1_native = np.where(is_p, take_rank1, 0.0)
                take_rank1_ambient = np.where(is_p, 0.0, take_rank1)
                take_u = np.where(is_u, y_cl, 0.0)
                sat_ambient = _pre_enrich_sat_mask(take_rank1_ambient, y_cl)
                take_rank1_ambient = _apply_dose_enrichment(
                    x,
                    idx,
                    take_rank1_ambient,
                    d_v,
                    n,
                    strength=ENRICH_STRENGTH,
                    renormalize=True,
                    observed=y_cl,
                )
                high_u = _high_chi_u_mask(is_u, chi)
                take_u_high = np.where(high_u, take_u, 0.0)
                take_u_low = np.where(is_u & ~high_u, take_u, 0.0)
                take_u_low = _apply_dose_enrichment(
                    x, idx, take_u_low, d_v, n, strength=ENRICH_STRENGTH
                )
                row_before = np.asarray(x[idx].sum(axis=1)).ravel()
                _expand_take_to_cells(
                    x,
                    idx,
                    take_rank1_ambient,
                    d_idx,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                    sat_mask=sat_ambient,
                    gene_names=gene_keys,
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
                    take_u_high,
                    d_idx,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                lost = row_before - np.asarray(x[idx].sum(axis=1)).ravel()
                remaining = np.clip(d_idx - lost, 0.0, None)
                take_u_low = _cap_take_to_remaining(take_u_low, remaining)
                _expand_take_to_cells(
                    x,
                    idx,
                    take_u_low,
                    remaining,
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
        # soupOnly extra-clear is capped at remaining d_c after rank-1.
        # Do not scale the whole row back to d_c: that put soup UMIs back
        # after they were cleared. Rank-1 take is already bounded by d_c.
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
    The mixture estimator itself fits each type's own χ-deconvolved ambient
    reference (χ with that type's NNLS-estimated self-contamination share
    subtracted back out, not a leave-one-type profile of the other types).
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
