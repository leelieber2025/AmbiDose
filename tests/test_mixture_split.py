import numpy as np
from anndata import AnnData
from scipy import sparse

import ambidose._mixture as mixture
from ambidose.pp import CHI_KEY


def _library(seed=0):
    """Dominant type A (900 cells) and a minor type B (320 cells) made of two distinct subtypes."""
    rng = np.random.default_rng(seed)
    g = 300
    chi = rng.dirichlet(np.ones(g) * 0.5)
    profiles = {}
    for name, block in (("A", slice(0, 100)), ("B1", slice(100, 160)), ("B2", slice(160, 220))):
        p = np.full(g, 1e-4)
        p[block] = rng.dirichlet(np.ones(block.stop - block.start))
        profiles[name] = p / p.sum()
    labels = ["A"] * 900 + ["B1"] * 160 + ["B2"] * 160
    rows = []
    for lab in labels:
        rho = rng.uniform(0.05, 0.3)
        depth = int(rng.integers(1500, 3000))
        mu = depth * ((1 - rho) * profiles[lab] + rho * chi)
        rows.append(rng.poisson(mu))
    adata = AnnData(sparse.csr_matrix(np.asarray(rows, dtype=float)))
    adata.obs["cell_type"] = ["A"] * 900 + ["B"] * 320
    adata.obs["subtype"] = labels
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = chi
    return adata


def test_split_labels_recover_subtypes():
    adata = _library()
    b = adata[adata.obs["cell_type"] == "B"]
    labels = mixture._split_labels(b.X)
    assert labels is not None
    truth = b.obs["subtype"].to_numpy()
    agree = max(
        np.mean((labels == labels[0]) == (truth == truth[0])),
        1 - np.mean((labels == labels[0]) == (truth == truth[0])),
    )
    assert agree > 0.9


def _flags(monkeypatch, reading, **kw):
    adata = _library()
    monkeypatch.setattr(
        mixture,
        "_type_readings",
        lambda x, n, chi, in_sample, types: ({}, {"B": ([reading] * 4, {})}),
    )
    mixture.estimate_dose_mixture(adata, type_key="cell_type", **kw)
    return adata.obs["ambidose_mixture_split"].to_numpy(), (
        adata.obs["cell_type"] == "B"
    ).to_numpy()


def test_split_follows_the_cross_type_reading(monkeypatch):
    low, is_b = _flags(monkeypatch, 1e-3)
    high, _ = _flags(monkeypatch, 0.99)
    # whichever of the whole-type and sub-group levels is closer to the reading wins,
    # so the two extreme readings pick different fits for B; the largest type never splits
    assert low[is_b].all() != high[is_b].all()
    assert not low[~is_b].any() and not high[~is_b].any()


def test_split_needs_a_reading():
    adata = _library()  # two types: too few reference types for a reading
    mixture.estimate_dose_mixture(adata, type_key="cell_type")
    assert not adata.obs["ambidose_mixture_split"].any()


def test_split_can_be_disabled(monkeypatch):
    low, _ = _flags(monkeypatch, 1e-3, split_heterogeneous=False)
    high, _ = _flags(monkeypatch, 0.99, split_heterogeneous=False)
    assert not low.any() and not high.any()
