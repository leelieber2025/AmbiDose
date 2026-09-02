"""Synthetic droplet matrices with a known ambient profile. No download.

``make_toy`` / ``make_barnyard_toy`` are primitives. Named scenarios
(``scenario_housekeeping``, ``scenario_zero_ambient``) and label transforms
(``split_type_labels``, ``merge_type_labels``) pin operator failure modes.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData, concat
from scipy import sparse


def make_toy(
    *,
    n_genes: int = 40,
    n_empty: int = 200,
    n_cells: int = 80,
    n_samples: int = 2,
    empty_umi: int = 40,
    cell_umi: int = 400,
    contamination: float = 0.2,
    rho_sd: float = 0.0,
    umi_log_sd: float = 0.0,
    n_tech_genes: int = 0,
    lysis: tuple[float, float] | None = None,
    seed: int = 0,
) -> AnnData:
    """Two cell types plus empty droplets; ambient is lysis-weighted soup.

    ``adata.uns['true_chi']`` is samples × genes (simplex). ``obs['droplet']``
    is ``empty`` or ``cell``. Counts are Poisson draws, stored as CSR.
    """
    rng = np.random.default_rng(seed)
    n_bio = n_genes
    n_total = n_bio + n_tech_genes
    genes = np.array([f"g{i}" for i in range(n_total)])
    lam = np.ones(2) if lysis is None else np.asarray(lysis, dtype=float)

    # Two type means: first / second block of biological genes are markers.
    half = n_bio // 2
    mu_bio = np.stack(
        [
            np.concatenate([np.full(half, 4.0), np.full(n_bio - half, 0.2)]),
            np.concatenate([np.full(half, 0.2), np.full(n_bio - half, 4.0)]),
        ]
    )
    mu_bio = mu_bio / mu_bio.sum(axis=1, keepdims=True)
    if n_tech_genes:
        mu = np.hstack([mu_bio, np.zeros((2, n_tech_genes))])
        mu = mu / mu.sum(axis=1, keepdims=True)
    else:
        mu = mu_bio

    blocks: list[AnnData] = []
    true_chi = []
    true_p = []
    for s in range(n_samples):
        p0 = 0.5 if n_samples == 1 else 0.2 + 0.6 * s / (n_samples - 1)
        p = np.array([p0, 1.0 - p0])
        w = p * lam
        chi = (w @ mu) / w.sum()
        chi = chi / chi.sum()
        if n_tech_genes:
            tech = np.zeros(n_total)
            tech[n_bio:] = 1.0 / n_tech_genes
            chi = 0.6 * chi + 0.4 * tech
        else:
            chi[0] += 0.03 * (s + 1)
        chi = np.clip(chi, 0, None)
        chi = chi / chi.sum()
        true_chi.append(chi)
        true_p.append(p)

        empty = rng.poisson(empty_umi * chi, size=(n_empty, n_total))
        n0 = int(round(p0 * n_cells))
        types = np.array([0] * n0 + [1] * (n_cells - n0))
        rng.shuffle(types)
        native = mu[types]
        if rho_sd > 0:
            rho_c = np.clip(
                rng.lognormal(np.log(max(contamination, 1e-6)), rho_sd, size=n_cells), 0.0, 0.95
            )
        else:
            rho_c = np.full(n_cells, float(contamination))
        if umi_log_sd > 0:
            n_c = rng.lognormal(np.log(cell_umi), umi_log_sd, size=n_cells)
        else:
            n_c = np.full(n_cells, float(cell_umi))
        mean = (1.0 - rho_c)[:, None] * n_c[:, None] * native + rho_c[:, None] * n_c[:, None] * chi
        cells = rng.poisson(mean)

        x = np.vstack([empty, cells]).astype(np.float32)
        ad = AnnData(sparse.csr_matrix(x))
        ad.var_names = genes
        ad.obs_names = [f"s{s}_{i}" for i in range(ad.n_obs)]
        ad.obs["sample"] = f"s{s}"
        ad.obs["droplet"] = ["empty"] * n_empty + ["cell"] * n_cells
        ad.obs["cell_type"] = ["none"] * n_empty + [f"t{t}" for t in types]
        ad.obs["true_rho"] = np.concatenate([np.zeros(n_empty), rho_c])
        ad.obs["true_d"] = np.concatenate([np.zeros(n_empty), rho_c * n_c])
        blocks.append(ad)

    adata = concat(blocks, axis=0, merge="same")
    adata.obs_names_make_unique()
    adata.uns["true_chi"] = np.vstack(true_chi)
    adata.uns["true_p"] = np.vstack(true_p)
    adata.uns["true_mu"] = mu
    adata.uns["true_lambda"] = lam
    adata.uns["true_contamination"] = contamination
    return adata


def make_barnyard_toy(
    *,
    n_genes_per_species: int = 20,
    n_empty: int = 200,
    n_human: int = 40,
    n_mouse: int = 40,
    empty_umi: int = 40,
    cell_umi: int = 400,
    contamination: float = 0.2,
    rho_sd: float = 0.0,
    umi_log_sd: float = 0.0,
    seed: int = 0,
) -> AnnData:
    """One library, human+mouse cells, soup is a mix of both (barnyard)."""
    rng = np.random.default_rng(seed)
    g_h = n_genes_per_species
    genes = [f"hg19_g{i}" for i in range(g_h)] + [f"mm10_g{i}" for i in range(g_h)]
    genome = ["hg19"] * g_h + ["mm10"] * g_h
    mu_h = np.concatenate([np.full(g_h, 3.0), np.full(g_h, 0.05)])
    mu_m = np.concatenate([np.full(g_h, 0.05), np.full(g_h, 3.0)])
    mu_h = mu_h / mu_h.sum()
    mu_m = mu_m / mu_m.sum()
    chi = 0.55 * mu_h + 0.45 * mu_m
    chi = chi / chi.sum()

    empty = rng.poisson(empty_umi * chi, size=(n_empty, 2 * g_h))
    n_cells = n_human + n_mouse
    if rho_sd > 0:
        rho_c = np.clip(
            rng.lognormal(np.log(max(contamination, 1e-6)), rho_sd, size=n_cells), 0.0, 0.95
        )
    else:
        rho_c = np.full(n_cells, float(contamination))
    if umi_log_sd > 0:
        n_c = rng.lognormal(np.log(cell_umi), umi_log_sd, size=n_cells)
    else:
        n_c = np.full(n_cells, float(cell_umi))
    native = np.vstack(
        [np.repeat(mu_h[None, :], n_human, axis=0), np.repeat(mu_m[None, :], n_mouse, axis=0)]
    )
    mean = (1.0 - rho_c)[:, None] * n_c[:, None] * native + rho_c[:, None] * n_c[:, None] * chi
    cells = rng.poisson(mean)
    x = np.vstack([empty, cells]).astype(np.float32)
    adata = AnnData(sparse.csr_matrix(x))
    adata.var_names = genes
    adata.var["genome"] = genome
    n = adata.n_obs
    adata.obs_names = [f"bc{i}" for i in range(n)]
    adata.obs["droplet"] = ["empty"] * n_empty + ["cell"] * n_cells
    adata.obs["true_species"] = ["none"] * n_empty + ["hg19"] * n_human + ["mm10"] * n_mouse
    adata.obs["true_rho"] = np.concatenate([np.zeros(n_empty), rho_c])
    adata.obs["true_d"] = np.concatenate([np.zeros(n_empty), rho_c * n_c])
    adata.uns["true_chi"] = chi
    adata.uns["true_contamination"] = contamination
    return adata


_EMPTY_TYPES = frozenset({"none", "-1", ""})


def _cell_mask(adata: AnnData) -> np.ndarray:
    if "droplet" in adata.obs:
        return adata.obs["droplet"].astype(str).to_numpy() == "cell"
    if "ambidose_droplet" in adata.obs:
        return adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    return np.ones(adata.n_obs, dtype=bool)


def scenario_housekeeping(
    *,
    true_rho: float = 0.1,
    n_hk: int = 40,
    n_types: int = 4,
    genes_per_type: int = 20,
    n_cells_per_type: int = 200,
    n_empty: int = 200,
    empty_umi: int = 40,
    mu_own: float = 8.0,
    mu_hk: float = 5.0,
    seed: int = 0,
) -> AnnData:
    """Type-exclusive markers plus genes at the same level in every type.

    Shared genes sit at the ρ=1 ambient ceiling in every type. This is the
    documented housekeeping bias, not a new estimator. ``n_empty=0`` is the
    cells-only calibration object (χ stored on ``var``).
    """
    rng = np.random.default_rng(seed)
    n_genes = n_types * genes_per_type + n_hk
    native = np.zeros((n_types, n_genes))
    for t in range(n_types):
        native[t, t * genes_per_type : (t + 1) * genes_per_type] = mu_own
    if n_hk:
        native[:, n_types * genes_per_type :] = mu_hk
    pop_mean = native.mean(axis=0)
    chi = pop_mean / pop_mean.sum()

    empty = rng.poisson(empty_umi * chi, size=(n_empty, n_genes))
    X_cells = []
    types = []
    rho_c = []
    d_c = []
    for t in range(n_types):
        for _ in range(n_cells_per_type):
            nat = rng.poisson(native[t])
            n_cell = float(nat.sum())
            amb = rng.poisson(true_rho * n_cell * chi)
            X_cells.append(nat + amb)
            types.append(f"t{t}")
            rho_c.append(float(true_rho))
            d_c.append(float(true_rho * n_cell))
    X_cells = np.asarray(X_cells, dtype=np.float64)
    x = np.vstack([empty.astype(np.float64), X_cells]) if n_empty else X_cells
    ad = AnnData(sparse.csr_matrix(x))
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    n_cells = n_types * n_cells_per_type
    ad.obs_names = [f"bc{i}" for i in range(ad.n_obs)]
    ad.obs["droplet"] = ["empty"] * n_empty + ["cell"] * n_cells
    ad.obs["cell_type"] = ["none"] * n_empty + types
    ad.obs["true_rho"] = np.concatenate([np.zeros(n_empty), np.asarray(rho_c)])
    ad.obs["true_d"] = np.concatenate([np.zeros(n_empty), np.asarray(d_c)])
    ad.obs["sample"] = "s0"
    ad.uns["true_chi"] = chi
    ad.uns["true_mu"] = native
    ad.uns["true_contamination"] = float(true_rho)
    ad.uns["scenario"] = "housekeeping"
    if n_empty == 0:
        ad.obs["ambidose_droplet"] = "cell"
        ad.var["ambidose_chi"] = chi
    return ad


def scenario_zero_ambient(**kwargs) -> AnnData:
    """``make_toy`` with true ρ = 0. Same counts model, named failure check."""
    kwargs.pop("contamination", None)
    ad = make_toy(contamination=0.0, **kwargs)
    ad.uns["scenario"] = "zero_ambient"
    return ad


def split_type_labels(
    adata: AnnData,
    *,
    type_key: str = "cell_type",
    key_added: str = "label_split",
    n_splits: int = 2,
    seed: int = 0,
) -> AnnData:
    """Split each true cell type into ``n_splits`` fake labels. Counts unchanged."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if type_key not in adata.obs:
        raise KeyError(type_key)
    rng = np.random.default_rng(seed)
    src = adata.obs[type_key].astype(str).to_numpy()
    out = src.copy()
    is_cell = _cell_mask(adata)
    for t in np.unique(src[is_cell]):
        if t in _EMPTY_TYPES:
            continue
        idx = np.flatnonzero(is_cell & (src == t))
        parts = rng.integers(0, n_splits, size=idx.size)
        out[idx] = np.array([f"{t}_s{int(k)}" for k in parts], dtype=object)
    adata.obs[key_added] = out
    adata.uns["label_transform"] = "split"
    return adata


def merge_type_labels(
    adata: AnnData,
    *,
    type_key: str = "cell_type",
    key_added: str = "label_merged",
    merged_name: str = "merged",
) -> AnnData:
    """Collapse every cell type into one label. Counts unchanged."""
    if type_key not in adata.obs:
        raise KeyError(type_key)
    src = adata.obs[type_key].astype(str).to_numpy()
    out = src.copy()
    is_cell = _cell_mask(adata)
    cell_types = is_cell & ~np.isin(src, list(_EMPTY_TYPES))
    out[cell_types] = merged_name
    adata.obs[key_added] = out
    adata.uns["label_transform"] = "merge"
    return adata
