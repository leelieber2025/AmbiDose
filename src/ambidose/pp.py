"""Empty-droplet calling, ambient profile, and per-cell dose subtraction."""

from __future__ import annotations

from ._budget import (
    _alloc_budget as _alloc_budget,
)
from ._budget import (
    _alloc_integer_budget as _alloc_integer_budget,
)
from ._budget import (
    _confidence_weighted_take as _confidence_weighted_take,
)
from ._budget import (
    _expand_take_to_cells as _expand_take_to_cells,
)
from ._budget import (
    _subtract_row as _subtract_row,
)
from ._chi import estimate_chi
from ._denoise import (
    _print_denoise_summary as _print_denoise_summary,
)
from ._denoise import (
    _write_rho_trust as _write_rho_trust,
)
from ._denoise import (
    analysis_ready,
    denoise,
)
from ._dose import (
    _rho_from_chi as _rho_from_chi,
)
from ._dose import (
    _top_chi_indices as _top_chi_indices,
)
from ._dose import (
    diagnose_dose_disagreement as diagnose_dose_disagreement,
)
from ._dose import (
    estimate_dose as estimate_dose,
)
from ._dose import (
    estimate_dose_adaptive as estimate_dose_adaptive,
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
    _normalized_barcode_map as _normalized_barcode_map,
)
from ._droplets import (
    _whitelist_vs_chi_keep as _whitelist_vs_chi_keep,
)
from ._droplets import (
    call_cells,
    classify_droplets,
)
from ._droplets import mark_doublets as mark_doublets
from ._mixture import (
    _two_component_mixture_em as _two_component_mixture_em,
)
from ._mixture import (
    _type_residual_score as _type_residual_score,
)
from ._mixture import (
    estimate_dose_mixture as estimate_dose_mixture,
)
from ._ownership import (
    _complete_linkage_labels as _complete_linkage_labels,
)
from ._ownership import (
    _cross_cell_structure_mask as _cross_cell_structure_mask,
)
from ._ownership import (
    _dominant_owner_masks as _dominant_owner_masks,
)
from ._ownership import (
    _type_masks as _type_masks,
)
from ._ownership import (
    _type_means as _type_means,
)
from ._shared import (
    CHI_KEY as CHI_KEY,
)
from ._shared import (
    CLUSTER_KEY as CLUSTER_KEY,
)
from ._shared import (
    DOSE_KEY as DOSE_KEY,
)
from ._shared import (
    DROPLET_KEY as DROPLET_KEY,
)
from ._shared import (
    EMPTY_TYPE as EMPTY_TYPE,
)
from ._shared import (
    LAYER_OUT as LAYER_OUT,
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
    MIN_GENES as MIN_GENES,
)
from ._shared import (
    MIN_TYPE_CELLS as MIN_TYPE_CELLS,
)
from ._shared import (
    MIX_INFLATION_RATIO as MIX_INFLATION_RATIO,
)
from ._shared import (
    RHO_KEY as RHO_KEY,
)
from ._shared import (
    SAMPLE_KEY_DEFAULT as SAMPLE_KEY_DEFAULT,
)
from ._shared import (
    SHRINK_K as SHRINK_K,
)
from ._shared import (
    TYPE_KEY as TYPE_KEY,
)
from ._shared import (
    _as_csr as _as_csr,
)
from ._shared import (
    _configure_scanpy_n_jobs as _configure_scanpy_n_jobs,
)
from ._shared import (
    _dense_chunk_columns as _dense_chunk_columns,
)
from ._shared import (
    _mc_worker_count as _mc_worker_count,
)
from ._shared import (
    _reject_view as _reject_view,
)
from ._shared import (
    _require_ram as _require_ram,
)
from ._shared import (
    _thread_worker_count as _thread_worker_count,
)
from ._shared import (
    _validate_chi_frame as _validate_chi_frame,
)
from ._shared import (
    _validated_type_values as _validated_type_values,
)
from ._shared import (
    raw_count_matrix as raw_count_matrix,
)
from ._shared import (
    require_run_keys as require_run_keys,
)
from ._subtract import subtract
from ._typing import _annotate_coarse_types_single as _annotate_coarse_types_single
from ._typing import (
    _default_type_key as _default_type_key,
)
from ._typing import _embed_coarse_hvg as _embed_coarse_hvg
from ._typing import _resolve_coarse_resolution as _resolve_coarse_resolution
from ._typing import _type_sample_dependence as _type_sample_dependence
from ._typing import (
    resolve_type_key,
)

__all__ = [
    "analysis_ready",
    "call_cells",
    "classify_droplets",
    "denoise",
    "diagnose_dose_disagreement",
    "estimate_chi",
    "estimate_dose",
    "estimate_dose_adaptive",
    "estimate_dose_mixture",
    "mark_doublets",
    "q_abs_scale",
    "resolve_type_key",
    "subtract",
]
