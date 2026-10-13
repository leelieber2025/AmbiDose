"""Cross-type level check: diagnostic flag only, never changes the dose."""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ambidose._level import (
    LEVEL_MIN_REFERENCES,
    LEVEL_RISK_RATIO,
    cross_type_level_check,
)
from ambidose._shared import CHI_KEY, DOSE_KEY, RHO_KEY


def _library(n_types=6, cells_per_type=80, genes_per_type=120, rho=0.15, seed=0):
    """Each type expresses its own block of genes; ambient follows chi (all types pooled)."""
    rng = np.random.default_rng(seed)
    n_genes = n_types * genes_per_type + 400
    block = {t: np.arange(t * genes_per_type, (t + 1) * genes_per_type) for t in range(n_types)}
    native_profile = {}
    for t in range(n_types):
        p = np.full(n_genes, 1e-6)
        p[block[t]] = rng.gamma(2.0, 1.0, genes_per_type)
        native_profile[t] = p / p.sum()
    chi = sum(native_profile.values()) / n_types
    chi = chi / chi.sum()
    rows, labels = [], []
    for t in range(n_types):
        for _ in range(cells_per_type):
            n_umi = int(rng.integers(4000, 6000))
            amb = rng.poisson(rho * n_umi * chi)
            nat = rng.poisson((1 - rho) * n_umi * native_profile[t])
            rows.append(amb + nat)
            labels.append(f"T{t}")
    x = sparse.csr_matrix(np.vstack(rows).astype(np.float64))
    adata = AnnData(x)
    adata.obs["ambidose_type"] = pd.Categorical(labels)
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = chi
    n = np.asarray(x.sum(axis=1)).ravel()
    adata.obs[RHO_KEY] = rho
    adata.obs[DOSE_KEY] = rho * n
    return adata, n


def test_correct_level_is_not_flagged():
    adata, _ = _library()
    risk, table = cross_type_level_check(
        adata, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert len(table) == 6
    assert not risk.any()
    ratios = [r["ratio"] for r in table]
    assert all(0.5 < r < LEVEL_RISK_RATIO for r in ratios)


def test_inflated_type_is_flagged_and_dose_is_untouched():
    adata, n = _library()
    inflated = (adata.obs["ambidose_type"] == "T2").to_numpy()
    dose = adata.obs[DOSE_KEY].to_numpy().copy()
    dose[inflated] *= 3.5
    adata.obs[DOSE_KEY] = dose
    before = adata.obs[DOSE_KEY].to_numpy().copy()
    risk, table = cross_type_level_check(
        adata, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert risk[inflated].all()
    assert not risk[~inflated].any()
    flagged = [r["type"] for r in table if r["level_risk"]]
    assert flagged == ["T2"]
    np.testing.assert_array_equal(adata.obs[DOSE_KEY].to_numpy(), before)


def test_too_few_types_is_not_evaluated():
    adata, _ = _library(n_types=LEVEL_MIN_REFERENCES)  # one short of references + 1
    risk, table = cross_type_level_check(
        adata, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert table == [] and not risk.any()


def test_small_types_are_skipped():
    adata, _ = _library(cells_per_type=30)
    risk, table = cross_type_level_check(
        adata, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert table == [] and not risk.any()


def test_missing_chi_or_dose_does_not_raise():
    adata, _ = _library()
    del adata.var[CHI_KEY]
    risk, table = cross_type_level_check(
        adata, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert table == [] and not risk.any()
    adata2, _ = _library()
    del adata2.obs[DOSE_KEY]
    risk2, table2 = cross_type_level_check(
        adata2, type_key="ambidose_type", droplet_key="ambidose_droplet"
    )
    assert table2 == [] and not risk2.any()
