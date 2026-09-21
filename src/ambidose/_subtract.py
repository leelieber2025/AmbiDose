"""Type-aware rank-1 subtraction along chi plus extra-clear."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._budget import (
    CEILING_CROSS_TYPE_GATE,
    ENRICH_STRENGTH,
    _ambient_slope_support,
    _analytic_anchor_cell_weights,
    _apply_dose_enrichment,
    _cap_take_to_remaining,
    _confidence_weighted_take,
    _expand_take_to_cells,
    _integerize_corrected,
    _migrate_unspent_rank1,
    _pre_enrich_sat_mask,
    _revoke_u_with_expressing_subset,
    _selected_data_positions,
    _soup_first_chi,
)
from ._dose import (
    _empty_cloud_knee_umi,
    _empty_consistent_rank1_budget,
    _native_everywhere_mask,
    _soup_just_above_empty,
    _soup_per_cell_fits_empty,
    _soup_u_mask,
    _u_mass_fits_empty_droplets,
    _unexpressed_mask,
)
from ._ownership import (
    _ceiling_cross_type_gate,
    _cross_type_anchor_mask,
    _dominant_owner_masks,
    _mt_gene_mask,
    _p_set_is_soup_like,
    _restrict_high_chi_to_single_winner,
    _revoke_u_if_rt_winner,
    _strip_ambient_level_owners,
    _type_masks,
)
from ._shared import (
    DOSE_KEY,
    EMPTY_TYPE,
    EMPTY_TYPES,
    LAYER_OUT,
    MIN_TYPE_CELLS,
    _as_csr,
    _chi_for_obs,
    _chi_vector,
    _feature_keys,
    _need,
    _reject_view,
    _require_raw_integer_counts,
    _resolve_cell_mask,
    _sample_names,
    _sample_storage_id,
    _validate_output_layer,
    _validated_sample_values,
    _validated_type_values,
    get_logger,
)
from ._typing import _default_type_key


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
    max_type_mean: float | None = None,
    min_chi: float = 1e-6,
    top_n: int | None = None,
    empirical_margin: bool = True,
    cross_type_anchor: bool = True,
    relax_hk_when_soup_like: bool = False,
    cap_high_u_to_remaining: bool = True,
    high_u_remaining_multiplier: float = 1.0,
    n_jobs: int | None = None,
) -> AnnData:
    """Subtract a rank-1 take along χ, plus extra-clear of unexpressed unowned genes.

    ``d_c = ρ_c n_c`` is the per-cell budget. Unused rank-1 after clipping
    on protected genes is reallocated along χ. Unowned takes are blended
    toward ``take × max(Pearson(y/n, ρ), 0)``. Extra-clear is limited to
    remaining ``d_c``. When unexpressed-unowned UMIs match empty droplets,
    or estimated soup per cell is not above the empty mean, extra-clear
    and leftover realloc are skipped and rank-1 is capped at empty U-gene
    soup. ``clip_negative=False`` is continuous residual mode.
    """
    _validate_output_layer(layer=layer, layer_out=layer_out)
    if high_u_remaining_multiplier < 0 or not np.isfinite(high_u_remaining_multiplier):
        raise ValueError("high_u_remaining_multiplier must be finite and nonnegative")
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

    # Inherit droplet labels stored with the dose when droplet_key is None.
    if isinstance(dose, str) and dose == DOSE_KEY and droplet_key is None:
        stored_run = adata.uns.get("ambidose", {})
        stored_provenance = stored_run.get("dose_provenance")
        if isinstance(stored_provenance, dict):
            droplet_key = stored_provenance.get("droplet_key")
        elif isinstance(stored_run.get("droplet_key"), str):
            droplet_key = stored_run["droplet_key"]
    samples = _sample_names(adata, sample_key)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    if (d_v[~is_cell] > 0).any():
        raise ValueError("positive dose found on non-cell droplets")
    use_sample = samples is not None
    type_key_explicit = type_key is not None
    if type_key is None:
        type_key = _default_type_key(adata)
    if type_key is not None and type_key not in adata.obs.columns:
        # Invalid type_key must not fall through to untyped subtraction.
        cols = sorted(adata.obs.columns.astype(str))
        shown = cols[:20]
        more = f", and {len(cols) - 20} more" if len(cols) > 20 else ""
        raise KeyError(f"type_key={type_key!r} not in adata.obs (available: {shown}{more})")
    if type_key_explicit and not clip_negative:
        warnings.warn(
            "type_key is ignored when clip_negative=False; subtraction uses the "
            "continuous untyped χ-direction path and does not write native_genes_by_type",
            UserWarning,
            stacklevel=2,
        )
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
                    _need(
                        "subtract() arguments do not match how dose was estimated "
                        f"(stored={provenance}, requested={requested}).",
                        "Re-run estimate_dose with these arguments, or call denoise() "
                        "instead of subtract() alone.",
                    )
                )

    use_mask = type_key is not None and type_key in adata.obs.columns and clip_negative

    if not use_mask:
        # Use CSR structure, not nonzero(): stored zeros would misalign x.data.
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
        n_empty_skip = 0
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
            sample_empty_skip = 0
            sample_layer2 = 0
            chi = _chi_for_obs(
                adata,
                sample_key=sample_key if s is not None else None,
                sample_name=s,
            )

            in_s = sample_of == s
            empty_idx_s = np.flatnonzero(in_s & ~is_cell)
            if empty_idx_s.size:
                lam_e = float(n[empty_idx_s].mean())
                knee_e = _empty_cloud_knee_umi(n, empty_idx_s)
            else:
                recs = adata.uns.get("ambidose", {}).get("empty_umi", {})
                rec = recs.get(_sample_storage_id(None if s is None else str(s)), {})
                if not rec and recs:
                    rec = next(iter(recs.values()))
                lam_e = float(rec["lam_e"]) if rec.get("lam_e") is not None else 0.0
                stored_knee = rec.get("knee_umi") if rec else None
                knee_e = float(stored_knee) if stored_knee is not None else lam_e
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
            if max_type_mean is None:
                sizes = [int(v.size) for v in type_indices_s.values() if v.size >= MIN_TYPE_CELLS]
                type_mean_floor = 1.0 / float(np.median(sizes)) if sizes else 0.0
            else:
                type_mean_floor = float(max_type_mean)
            dominant_masks, dominant_masks_sw, n_meta_s = _dominant_owner_masks(
                x,
                n,
                types_s_all,
                type_means,
                type_indices=type_indices_s,
                cell_keys=obs_keys,
                max_type_mean=type_mean_floor,
                also_single_winner=True,
            )
            dominant_masks = _restrict_high_chi_to_single_winner(
                dominant_masks, dominant_masks_sw, chi
            )
            n_bar_s = {
                t: float(n[type_indices_s[t]].mean())
                for t in type_means
                if type_indices_s.get(t) is not None and type_indices_s[t].size
            }
            dominant_masks = _strip_ambient_level_owners(dominant_masks, type_means, n_bar_s, chi)
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
                    max_type_mean=type_mean_floor,
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
                n_idx = n[idx]
                n_sum = float(n_idx.sum())
                rho_t = float(d_sum / n_sum) if n_sum > 0 else 0.0
                y_cl = np.asarray(x[idx].sum(axis=0)).ravel().astype(np.float64)
                leftover_cap = None
                rank1_budget = d_sum
                anchor_u = np.zeros(adata.n_vars, dtype=bool)
                fits_empty = False
                n_bar_t = float(n_idx.mean()) if idx.size else 0.0
                if t in EMPTY_TYPES:
                    is_u = np.zeros(adata.n_vars, dtype=bool)
                    is_p = np.zeros(adata.n_vars, dtype=bool)
                    native_confidence = np.zeros(adata.n_vars)
                    r_t = np.ones(adata.n_vars)
                elif idx.size < MIN_TYPE_CELLS:
                    # Too few cells: rank-1 only, no extra-clear.
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
                        "rho_t": rho_t,
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
                    anchor_u = is_u.copy()
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
                        rho_t=rho_t,
                    )
                    is_u = _revoke_u_with_expressing_subset(x, idx, n, chi, is_u)
                    is_u = _revoke_u_if_rt_winner(is_u, t, type_means, n_bar_s, chi)
                    # leftover_cap is unused rank-1 under unique-argmax ownership.
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
                    other_u = np.zeros(adata.n_vars, dtype=bool)
                    for t2, mask_t2 in dominant_masks.items():
                        if t2 != t:
                            other_u |= is_u & np.asarray(mask_t2, dtype=bool)
                    # Other-type-owned U is real cross-type soup when it exceeds
                    # empty droplets. If it matches empty, estimated dose is χ
                    # overlapping native (clean cells, no extra soup).
                    cross_fits_empty = other_u.any() and _u_mass_fits_empty_droplets(
                        x, idx, empty_idx_s, other_u, lam_e=lam_e, chi=chi
                    )
                    fits_empty = (
                        _u_mass_fits_empty_droplets(x, idx, empty_idx_s, is_u, lam_e=lam_e, chi=chi)
                        or _soup_per_cell_fits_empty(rho_t, n_bar_t, lam_e)
                        or cross_fits_empty
                    )
                    if fits_empty:
                        rank1_budget = _empty_consistent_rank1_budget(
                            d_sum,
                            idx.size,
                            empty_idx_s,
                            x,
                            is_u,
                            lam_e=lam_e,
                            chi=chi,
                        )
                        is_u = np.zeros(adata.n_vars, dtype=bool)
                        leftover_cap = 0.0
                        sample_empty_skip += int(idx.size)
                d_idx = d_v[idx]
                # Rank-1 along χ; extra-clear of unexpressed unowned genes follows.
                # Layer 1 (fits_empty): skip extra-clear, capped rank-1, U wiped.
                # Layer 2 (soup just above the empty-cloud knee, ≤3× ceiling):
                # SoupX-like χ take. Layer 3 (more soup): full native-χ protection.
                # Skip still uses mean empty; the knee is only this gate.
                protect_scale = (
                    0.0
                    if (not fits_empty) and _soup_just_above_empty(rho_t, n_bar_t, knee_e)
                    else 1.0
                )
                if protect_scale == 0.0 and not fits_empty:
                    sample_layer2 += int(idx.size)
                chi_take = (
                    _soup_first_chi(chi, native_confidence, is_u)
                    if protect_scale == 0.0 and not fits_empty
                    else chi
                )
                take_rank1 = _confidence_weighted_take(
                    y_cl,
                    chi_take,
                    rank1_budget,
                    native_confidence,
                    None,
                    protect_scale=protect_scale,
                )
                take_rank1 = np.where(is_u, 0.0, take_rank1)
                if (not fits_empty) and is_u.any() and protect_scale < 1.0:
                    u_chi = np.minimum(y_cl, rank1_budget * chi_take)
                    take_rank1 = np.where(is_u, (1.0 - protect_scale) * u_chi, take_rank1)
                ambient_support = _ambient_slope_support(x, idx, n, chi, d_idx)
                take_rank1 = _migrate_unspent_rank1(
                    take_rank1,
                    y_cl,
                    chi,
                    d_sum,
                    is_p,
                    is_u,
                    r_t,
                    leftover_cap=leftover_cap,
                    native_confidence=native_confidence,
                    ambient_support=ambient_support,
                )
                native_cell_weights, anchor_weight_meta = _analytic_anchor_cell_weights(
                    x, idx, n, chi, anchor_u, d_idx
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
                    native_cell_weights,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                if cap_high_u_to_remaining:
                    rank1_lost = row_before - np.asarray(x[idx].sum(axis=1)).ravel()
                    u_remaining = np.clip(d_idx - rank1_lost, 0.0, None) * float(
                        high_u_remaining_multiplier
                    )
                    take_u = _cap_take_to_remaining(take_u, u_remaining)
                else:
                    u_remaining = d_idx
                _expand_take_to_cells(
                    x,
                    idx,
                    take_u,
                    u_remaining,
                    data_positions=data_positions,
                    cell_keys=obs_keys[idx],
                )
                group_key = json.dumps(
                    [None if s is None else str(s), str(t)],
                    separators=(",", ":"),
                )
                sample_native[group_key] = np.flatnonzero(is_p).tolist()
            return (
                sample_native,
                sample_n_meta,
                sample_tiny,
                sample_empty_skip,
                sample_layer2,
                knee_e,
            )

        sample_results = list(map(process_sample, groups))
        native_internal: dict[str, list[int]] = {}
        meta_groups_by_sample: dict[str, dict] = {}
        total_meta_groups = 0
        n_layer2 = 0
        empty_umi = dict(adata.uns.get("ambidose", {}).get("empty_umi", {}))
        for sample, (
            sample_native,
            sample_n_meta,
            sample_tiny,
            sample_empty_skip,
            sample_layer2,
            knee_e,
        ) in zip(groups, sample_results, strict=True):
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
            n_empty_skip += sample_empty_skip
            n_layer2 += sample_layer2
            if np.isfinite(knee_e) and knee_e > 0:
                rec = dict(empty_umi.get(sample_id, {}))
                rec.setdefault("sample", "" if sample is None else str(sample))
                rec["knee_umi"] = float(knee_e)
                empty_umi[sample_id] = rec
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
        uns["n_empty_consistent_skip_cells"] = int(n_empty_skip)
        uns["n_low_soup_full_chi_cells"] = int(n_layer2)
        uns["empty_umi"] = empty_umi
        adata.uns["ambidose"] = uns
        if n_tiny_protected:
            get_logger().info(
                "ambidose: %s cells in groups <%s cells; extra-clear skipped "
                "(rank-1 protected take only)",
                n_tiny_protected,
                MIN_TYPE_CELLS,
            )

    if clip_negative:
        np.maximum(x.data, 0.0, out=x.data)
        # soupOnly extra-clear is capped at remaining d_c after rank-1 for
        # low-χ U and at the configured soft allowance for high-χ U.
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
