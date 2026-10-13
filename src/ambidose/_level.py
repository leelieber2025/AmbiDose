"""Cross-type check of each type's ambient level (diagnostic only).

The level of ``rho`` within a type is weakly identified: the mixture
likelihood is flat along ``p' = (1 - s) p + s a`` and the empty-droplet ``chi``
differs from the in-cell ambient gene by gene. This module gives an
independent, count-based reading of each type's level from genes that other
types express and this type should not. It never changes ``rho`` or the
dose; it only reports where the executed level is far above that reading.

For a reference type ``u`` take the genes most enriched in ``u`` relative to
``chi``. In another type ``t`` those genes are, to first order, ambient, so
the chi-mass weighted median of ``mean_tg / (n_bar_t chi_g)`` over them reads
the ambient level of ``t``. The minimum over reference types is used because
native expression shared with a related type can only raise a reading. A
median over many genes is used rather than an extreme: ``chi`` and the real
ambient disagree gene by gene by a factor near 2.7 on kidney libraries, which
makes minimum-type statistics collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._shared import (
    DOSE_KEY,
    EMPTY_TYPES,
    _as_csr,
    _chi_for_obs,
    _resolve_cell_mask,
    _sample_names,
    _validated_type_values,
)

LEVEL_TOP_GENES = 4000
LEVEL_MARKER_GENES = 100
LEVEL_MIN_MARKER_COUNT = 100.0
LEVEL_MIN_TYPE_CELLS = 60
LEVEL_MIN_REFERENCES = 4
LEVEL_RISK_RATIO = 2.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cum, 0.5)])


def _type_readings(x, n, chi, in_sample, types):
    """Cross-type readings of one sample: ``{type: (per-reference readings, stats)}``.

    Types are evaluated only when at least ``LEVEL_MIN_REFERENCES`` other types
    with ``LEVEL_MIN_TYPE_CELLS`` cells exist; a type needs that many readings.
    """
    stats: dict[object, dict] = {}
    for t in pd.unique(types[in_sample]):
        if t in EMPTY_TYPES:
            continue
        idx = np.flatnonzero(in_sample & (types == t))
        if idx.size < LEVEL_MIN_TYPE_CELLS:
            continue
        stats[t] = {
            "idx": idx,
            "n_cells": idx.size,
            "n_bar": float(n[idx].mean()),
            "mean": np.asarray(x[idx].mean(axis=0)).ravel(),
        }
    out: dict[object, tuple] = {}
    if len(stats) < LEVEL_MIN_REFERENCES + 1:
        return stats, out
    top = np.argsort(-chi)[:LEVEL_TOP_GENES]
    markers: dict[object, np.ndarray] = {}
    for u, d in stats.items():
        profile = d["mean"] / max(float(d["mean"].sum()), 1e-12)
        enrichment = profile[top] / np.maximum(chi[top], 1e-12)
        counted = d["mean"][top] * d["n_cells"] >= LEVEL_MIN_MARKER_COUNT
        pick = top[counted][np.argsort(-enrichment[counted])[:LEVEL_MARKER_GENES]]
        markers[u] = pick
    for t, d in stats.items():
        readings = []
        for u, genes in markers.items():
            if u == t or genes.size < 20:
                continue
            expected = d["n_bar"] * chi[genes]
            keep = expected > 0
            if keep.sum() < 20:
                continue
            readings.append(
                _weighted_median(
                    d["mean"][genes][keep] / expected[keep],
                    expected[keep] * d["n_cells"],
                )
            )
        if len(readings) >= LEVEL_MIN_REFERENCES:
            out[t] = (readings, d)
    return stats, out


def cross_type_level_check(
    adata: AnnData,
    *,
    type_key: str,
    droplet_key: str | None = None,
    cell_label: str = "cell",
    layer: str | None = None,
    sample_key: str | None = None,
    ratio: float = LEVEL_RISK_RATIO,
) -> tuple[np.ndarray, list[dict]]:
    """Return a per-observation ``level_risk`` mask and a per-(sample, type) table.

    A type is at risk when its UMI-weighted executed ``rho`` exceeds ``ratio``
    times the cross-type reading. Types are checked only when at least
    ``LEVEL_MIN_REFERENCES`` other types with ``LEVEL_MIN_TYPE_CELLS`` cells
    exist in the same sample; otherwise they are left unevaluated and never
    flagged. Returns an all-False mask when ``chi`` or the dose is missing.
    """
    n_obs = adata.n_obs
    risk = np.zeros(n_obs, dtype=bool)
    table: list[dict] = []
    if DOSE_KEY not in adata.obs or type_key not in adata.obs.columns:
        return risk, table
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    n = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    dose = np.asarray(adata.obs[DOSE_KEY], dtype=np.float64)
    is_cell = _resolve_cell_mask(adata, droplet_key, cell_label)
    types = _validated_type_values(adata, type_key).to_numpy()
    samples = _sample_names(adata, sample_key)
    groups = [None] if samples is None else list(pd.unique(samples))
    sample_of = np.array([None] * n_obs, dtype=object) if samples is None else samples
    for sample in groups:
        try:
            chi = np.asarray(
                _chi_for_obs(
                    adata,
                    sample_key=sample_key if sample is not None else None,
                    sample_name=sample,
                ),
                dtype=np.float64,
            )
        except (KeyError, ValueError):
            continue
        in_sample = is_cell & (sample_of == sample)
        stats, readings_by_type = _type_readings(x, n, chi, in_sample, types)
        for t, (readings, d) in readings_by_type.items():
            idx = d["idx"]
            reading = float(min(readings))
            level = float(dose[idx].sum() / max(float(n[idx].sum()), 1e-12))
            at_risk = bool(reading > 0 and level > ratio * reading)
            if at_risk:
                risk[idx] = True
            table.append(
                {
                    "sample": "" if sample is None else str(sample),
                    "type": str(t),
                    "n_cells": int(idx.size),
                    "n_references": len(readings),
                    "executed_rho": level,
                    "cross_type_reading": reading,
                    "ratio": level / reading if reading > 0 else float("nan"),
                    "level_risk": at_risk,
                }
            )
    return risk, table
