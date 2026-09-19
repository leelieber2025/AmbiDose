"""Ambient profile chi from empty droplets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._shared import (
    CHI_KEY,
    DROPLET_KEY,
    SAMPLE_KEY_DEFAULT,
    _as_csr,
    _nb2_phi_from_empty,
    _profile,
    _reject_view,
    _require_raw_integer_counts,
    _sample_storage_id,
    _validated_sample_values,
)


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
        x_empty = x[empty]
        chi = _profile(x_empty)
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
                "phi": _nb2_phi_from_empty(x_empty),
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
        x_empty = x[mask]
        rows.append(_profile(x_empty, sample=str(name)))
        empty_umi[_sample_storage_id(str(name))] = {
            "sample": str(name),
            "lam_e": float(n[mask].mean()),
            "n_empty": int(mask.sum()),
            "phi": _nb2_phi_from_empty(x_empty),
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
