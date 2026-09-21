"""Product denoise entry and analysis-ready reduction."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ._chi import estimate_chi
from ._dose import estimate_dose_adaptive
from ._droplets import (
    _normalized_barcode_map,
    call_cells,
    classify_droplets,
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
    _configure_scanpy_n_jobs,
    _feature_keys,
    _need,
    _reject_view,
    _require_raw_integer_counts,
    _same_matrix,
    _StageProgress,
    _validate_chi_frame,
    _validate_output_layer,
    _validated_sample_values,
    _validated_type_values,
    raw_count_matrix,
    require_run_keys,
)
from ._subtract import subtract
from ._typing import _default_type_key, resolve_type_key


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
        if "ambidose_mixture_status" in adata.obs:
            mixture_fallback = (
                adata.obs["ambidose_mixture_status"].astype(str).to_numpy() == "quantile_fallback"
            )
            fallback = np.where(used_mixture, mixture_fallback, fallback)
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
            "cap on total UMI removal; high-χ U has a bounded soft allowance. "
            "under_execution means less than half of that budget was removed; "
            "over_removal means total removal exceeded d_c by more than 5% "
            "(usually high-χ soupOnly extra-clear) "
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
    evidence_mode: str = "exposure",
    cross_type_anchor: bool = True,
    relax_hk_when_soup_like: bool = False,
    cap_high_u_to_remaining: bool = True,
    high_u_remaining_multiplier: float = 1.0,
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

    Default groups are label-free coarse Leiden clusters. Pass ``type_key``
    for labels computed outside this package. Dose keeps the typed estimate
    when it agrees with the mixture estimate. ``typing_fast`` uses a cheaper
    graph on libraries with at least ``TYPING_FAST_N_CELLS`` cells.

    Cells come from the Cell Ranger filtered list when one is available
    (``cell_barcodes``, auto-detected ``filtered_*`` next to raw, or
    ``raw=`` filtered ``obs_names``). That list is used as-is. Pass
    ``cell_calling='chi'`` only if that list is known to be over-called.
    Without a filtered list, ``cell_calling='diem'`` / ``'emptydrops'`` /
    ``expect_cells`` build a cell list from the raw matrix.

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
            evidence_mode=evidence_mode,
            cross_type_anchor=cross_type_anchor,
            relax_hk_when_soup_like=relax_hk_when_soup_like,
            cap_high_u_to_remaining=cap_high_u_to_remaining,
            high_u_remaining_multiplier=high_u_remaining_multiplier,
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
            calling = None if cell_calling is None else str(cell_calling).strip().lower()
            trust = calling is None or calling in ("off", "none", "external")
            if trust:
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
                if calling != "chi":
                    raise ValueError(
                        _need(
                            f"cell_calling={cell_calling!r} does not use the filtered barcode list.",
                            "Omit cell_calling to use the 10x filtered barcodes as cells. "
                            "Pass cell_calling='chi' only to trim that list against soup.",
                        )
                    )
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
                _need(
                    "cell_calling='off' has no filtered barcode list to use.",
                    "Pass cell_barcodes= the Cell Ranger filtered barcodes, "
                    "or a Cell Ranger outs/ folder so they are found automatically.",
                )
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
                _need(
                    "denoise() has no cell list.",
                    "Pass a Cell Ranger outs/ folder or cell_barcodes= the "
                    "filtered barcodes TSV. Only if that list is missing, use "
                    "cell_calling='diem' or 'emptydrops'.",
                )
            )
    if sample_key is not None and sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
    sk = sample_key
    if sk is not None:
        _validated_sample_values(adata, sk)
        if chi_ready:
            if CHI_KEY not in adata.uns:
                raise ValueError(
                    _need(
                        f"sample_key is set but uns[{CHI_KEY!r}] has no per-library χ.",
                        "Re-run estimate_chi with the same sample_key, or omit sample_key "
                        "for a single library.",
                    )
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
            evidence_mode=evidence_mode,
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
            cap_high_u_to_remaining=cap_high_u_to_remaining,
            high_u_remaining_multiplier=high_u_remaining_multiplier,
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
