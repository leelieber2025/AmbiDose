import numpy as np
import pytest
from scipy import sparse

from ambidose.datasets import make_toy
from ambidose.pp import (
    DROPLET_KEY,
    _empty_drops_keep,
    _whitelist_vs_chi_keep,
    call_cells,
    classify_droplets,
    estimate_chi,
    mark_doublets,
)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_emptydrops_bh_excludes_always_retained_barcodes(monkeypatch):
    x = sparse.csr_matrix([[3, 2]] * 15 + [[6, 4], [7, 4], [12, 8], [13, 8]])
    totals = np.asarray(x.sum(axis=1)).ravel()
    seen = []
    monkeypatch.setattr("ambidose._droplets._barcode_knee_umi", lambda *args, **kwargs: 20.0)
    monkeypatch.setattr(
        "ambidose._droplets._multinomial_mc_pvals",
        lambda *_args, **_kwargs: np.array([0.01, 0.01]),
    )

    def bh(pvals):
        seen.append(len(pvals))
        return np.asarray(pvals)

    monkeypatch.setattr("ambidose._droplets._bh_fdr", bh)
    keep = _empty_drops_keep(x, totals, lower=5, fdr=0.05)
    assert seen == [2]
    assert keep[-4:].all()


def test_whitelist_chi_tests_low_umi_candidates(monkeypatch):
    x = sparse.csr_matrix([[3, 2]] * 17 + [[30, 20], [45, 30], [60, 40]])
    totals = np.asarray(x.sum(axis=1)).ravel()
    in_white = np.zeros(20, dtype=bool)
    in_white[-3:] = True
    seen = []

    monkeypatch.setattr(
        "ambidose._droplets._barcode_rank_curve",
        lambda *args, **kwargs: {"inflection_umi": 101.0},
    )

    def accept_all(_x, _totals, test, *_args, **_kwargs):
        seen.extend(_totals[test].tolist())
        return np.zeros(int(test.sum()))

    monkeypatch.setattr("ambidose._droplets._multinomial_mc_pvals", accept_all)
    keep, _ = _whitelist_vs_chi_keep(x, totals, in_white, lower=100)
    assert seen == [50, 75, 100]
    assert keep[in_white].all()


def test_classify_cell_barcodes_does_not_promote_debris():
    adata = make_toy(n_empty=50, n_cells=20, n_samples=1, seed=1)
    true_cells = adata.obs_names[adata.obs["droplet"].to_numpy() == "cell"][:10]
    classify_droplets(adata, empty_umi_max=80, cell_barcodes=list(true_cells))
    lab = adata.obs["ambidose_droplet"].astype(str)
    assert (lab.loc[true_cells] == "cell").all()
    leftover_cells = adata.obs_names[
        (adata.obs["droplet"].to_numpy() == "cell") & (lab.to_numpy() != "cell")
    ]
    if len(leftover_cells):
        assert (lab.loc[leftover_cells] == "other").all()
    empty = (adata.obs["droplet"].to_numpy() == "empty") & (adata.obs["n_umi"].to_numpy() > 0)
    if empty.any():
        assert set(lab[empty].unique()) <= {"empty", "other"}


def test_classify_by_umi_threshold():
    adata = make_toy(n_empty=50, n_cells=20, n_samples=1, seed=1)
    classify_droplets(adata, empty_umi_max=80)
    # Toy empties have mean UMI ~40; cells ~400.
    empty = adata.obs["ambidose_droplet"] == "empty"
    assert empty.sum() >= 40
    assert (~empty).sum() >= 15
    # UMI==0 is junk, not soup (SoupX soupRange lower bound).
    zeros = adata.obs["n_umi"] == 0
    if zeros.any():
        assert (adata.obs.loc[zeros, "ambidose_droplet"] == "other").all()


def test_mixture_inflection_cut_only_when_inflated():
    from ambidose.pp import MIX_INFLATION_RATIO, _maybe_cut_inflated_mixture

    totals = np.concatenate([np.full(80, 2000.0), np.full(150, 400.0), np.full(400, 20.0)])
    mix = totals > 100
    curve = {"inflection_umi": 2000.0, "n_inflection": 80}
    out, cut = _maybe_cut_inflated_mixture(mix.copy(), totals, curve)
    assert MIX_INFLATION_RATIO == 2.5
    assert cut
    assert int(out.sum()) == 80
    curve_ok = {"inflection_umi": 400.0, "n_inflection": 150}
    out2, cut2 = _maybe_cut_inflated_mixture(mix.copy(), totals, curve_ok)
    assert not cut2
    assert int(out2.sum()) == int(mix.sum())


def test_rescue_keeps_cell_like_below_inflection():
    from scipy import sparse

    from ambidose.pp import _closer_to_cells_than_chi

    rng = np.random.default_rng(4)
    n_genes = 30
    chi = rng.random(n_genes)
    chi /= chi.sum()
    empty = np.vstack([rng.multinomial(25, chi) for _ in range(40)])
    high = np.zeros((20, n_genes))
    high[:, :6] = rng.integers(40, 80, size=(20, 6))
    soupish = np.vstack([rng.multinomial(200, chi) for _ in range(15)])
    cellish = np.zeros((15, n_genes))
    cellish[:, :6] = rng.integers(15, 35, size=(15, 6))
    x = sparse.csr_matrix(np.vstack([empty, high, soupish, cellish]).astype(np.float64))
    totals = np.asarray(x.sum(axis=1)).ravel()
    high_m = np.zeros(totals.size, dtype=bool)
    high_m[40:60] = True
    below = np.zeros(totals.size, dtype=bool)
    below[60:] = True
    rescued = _closer_to_cells_than_chi(x, totals, high=high_m, below=below, lower=40)
    assert int(rescued[75:].sum()) >= 10
    assert int(rescued[60:75].sum()) <= 5


def test_barcode_rank_inflection_below_knee_on_cliff():
    from ambidose.pp import _barcode_rank_curve

    rng = np.random.default_rng(0)
    cells = rng.integers(800, 2000, size=80)
    empty = rng.integers(5, 40, size=400)
    c = _barcode_rank_curve(np.concatenate([cells, empty]).astype(np.float64), lower=50)
    assert c["knee_umi"] >= c["inflection_umi"]
    assert 50 < c["n_inflection"] < 200


def test_call_cells_diem_keeps_typed_cells():
    from anndata import AnnData
    from scipy import sparse

    rng = np.random.default_rng(1)
    n_genes = 40
    chi = rng.random(n_genes)
    chi /= chi.sum()
    empty = rng.multinomial(30, chi, size=80)
    native = np.zeros((50, n_genes))
    native[:, :8] = rng.integers(80, 200, size=(50, 8))
    native += rng.multinomial(80, chi, size=50)
    x = sparse.csr_matrix(np.vstack([empty, native]).astype(np.float64))
    adata = AnnData(x)
    adata.obs_names = [f"e{i}" for i in range(80)] + [f"c{i}" for i in range(50)]
    names = call_cells(adata, method="diem")
    called = set(names)
    assert len(called) >= 20
    assert sum(n.startswith("c") for n in names) >= 40
    assert sum(n.startswith("e") for n in names) == 0


def test_call_cells_diem_drops_mid_umi_debris():
    from scipy import sparse

    from ambidose.pp import _diem_keep

    rng = np.random.default_rng(3)
    n_genes = 40
    chi = rng.random(n_genes)
    chi /= chi.sum()
    empty = rng.multinomial(30, chi, size=80)
    native = np.zeros((40, n_genes))
    native[:, :8] = rng.integers(80, 200, size=(40, 8))
    native += rng.multinomial(40, chi, size=40)
    debris = np.zeros((30, n_genes))
    debris[:, 8:16] = rng.integers(20, 60, size=(30, 8))
    debris += rng.multinomial(40, chi, size=30)
    x = sparse.csr_matrix(np.vstack([empty, native, debris]).astype(np.float64))
    totals = np.asarray(x.sum(axis=1)).ravel()
    keep = _diem_keep(x, totals, lower=50, n_hvg=40)
    names = np.array(
        [f"e{i}" for i in range(80)] + [f"c{i}" for i in range(40)] + [f"d{i}" for i in range(30)]
    )
    called = names[keep]
    assert sum(n.startswith("c") for n in called) >= 30
    assert sum(n.startswith("d") for n in called) <= 10
    assert sum(n.startswith("e") for n in called) == 0


def test_call_cells_chi_drops_soup_like_whitelist():
    from anndata import AnnData
    from scipy import sparse

    rng = np.random.default_rng(2)
    n_genes = 40
    chi = rng.random(n_genes)
    chi /= chi.sum()
    empty = np.vstack([rng.multinomial(int(n), chi) for n in rng.integers(15, 30, 80)])
    native = np.zeros((50, n_genes))
    native[:, :8] = rng.integers(150, 400, size=(50, 8))
    native += np.vstack([rng.multinomial(80, chi) for _ in range(50)])
    fake = np.vstack([rng.multinomial(int(n), chi) for n in rng.integers(80, 120, 20)])
    other = np.vstack([rng.multinomial(int(n), chi) for n in np.logspace(2.1, 2.9, 80).astype(int)])
    x = sparse.csr_matrix(np.vstack([empty, native, fake, other]).astype(np.float64))
    adata = AnnData(x)
    adata.obs_names = (
        [f"e{i}" for i in range(80)]
        + [f"c{i}" for i in range(50)]
        + [f"f{i}" for i in range(20)]
        + [f"o{i}" for i in range(80)]
    )
    white = [f"c{i}" for i in range(50)] + [f"f{i}" for i in range(20)]
    names = call_cells(
        adata, method="chi", cell_barcodes=white, lower=40, niters=2000, random_state=2
    )
    called = set(names)
    assert sum(n.startswith("c") for n in names) >= 40
    assert sum(n.startswith("f") for n in names) <= 5
    assert sum(n.startswith("e") for n in names) == 0
    assert called <= set(white)


def test_call_cells_emptydrops_keeps_typed_cells():
    from anndata import AnnData
    from scipy import sparse

    rng = np.random.default_rng(1)
    n_genes = 40
    chi = rng.random(n_genes)
    chi /= chi.sum()
    empty = rng.multinomial(30, chi, size=80)
    native = np.zeros((50, n_genes))
    native[:, :8] = rng.integers(80, 200, size=(50, 8))
    native += rng.multinomial(80, chi, size=50)
    x = sparse.csr_matrix(np.vstack([empty, native]).astype(np.float64))
    adata = AnnData(x)
    adata.obs_names = [f"e{i}" for i in range(80)] + [f"c{i}" for i in range(50)]
    names = call_cells(adata, method="emptydrops", niters=2000, random_state=1)
    called = set(names)
    assert len(called) >= 20
    assert sum(n.startswith("c") for n in names) >= sum(n.startswith("e") for n in names)


def test_call_cells_ordmag_separates_high_umi():
    from anndata import AnnData
    from scipy import sparse

    rng = np.random.default_rng(0)
    n_cell, n_empty, n_genes = 40, 80, 20
    cells = rng.integers(800, 2000, size=(n_cell, n_genes))
    empty = rng.integers(0, 40, size=(n_empty, n_genes))
    x = sparse.csr_matrix(np.vstack([empty, cells]).astype(np.float64))
    adata = AnnData(x)
    adata.obs_names = [f"e{i}" for i in range(n_empty)] + [f"c{i}" for i in range(n_cell)]
    names = call_cells(adata, expect_cells=n_cell, method="ordmag")
    called = set(names)
    assert len(called) >= 35
    assert called <= {f"c{i}" for i in range(n_cell)}


def test_call_cells_force_is_top_n():
    adata = make_toy(n_empty=50, n_cells=30, n_samples=1, seed=3)
    names = call_cells(adata, expect_cells=25, method="force")
    assert len(names) == 25
    n = adata.obs["n_umi"].to_numpy()
    cutoff = np.sort(n)[::-1][24]
    assert min(n[adata.obs_names.get_indexer(names)]) >= cutoff


def test_call_cells_max_cells_caps_ordmag():
    adata = make_toy(n_empty=40, n_cells=80, n_samples=1, seed=4)
    names = call_cells(adata, expect_cells=80, method="ordmag", max_cells=20)
    assert len(names) == 20


def test_classify_rejects_nonpositive_expected_cells():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=1)
    with pytest.raises(ValueError, match="at least 1"):
        classify_droplets(adata, expected_cells=0)


def test_expected_cells_takes_priority_over_empty_umi_cap():
    adata = make_toy(n_empty=20, n_cells=20, n_samples=1, seed=22)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    expected = 7
    top = np.argsort(totals)[::-1][:expected]

    classify_droplets(adata, expected_cells=expected, empty_umi_max=80)

    labels = adata.obs[DROPLET_KEY].astype(str).to_numpy()
    assert int((labels == "cell").sum()) == expected
    assert set(np.flatnonzero(labels == "cell")) == set(top)
    assert np.all(totals[labels == "empty"] <= 80)


def test_mark_doublets_writes_scores():
    adata = make_toy(n_empty=30, n_cells=80, n_samples=1, seed=9)
    classify_droplets(adata, empty_umi_max=80)
    mark_doublets(adata)
    assert "ambidose_doublet_score" in adata.obs
    assert str(adata.obs["ambidose_doublet"].dtype) == "boolean"
    # No type_key -> no heterotypic residual, and rho/dose are untouched.
    assert "ambidose_type_residual" not in adata.obs


def test_mark_doublets_clears_stale_type_residual():
    adata = make_toy(n_empty=30, n_cells=80, n_samples=1, seed=40)
    classify_droplets(adata, empty_umi_max=80)
    adata.obs["ambidose_type_residual"] = 1.0

    mark_doublets(adata, type_key=None, sample_key=None)

    assert "ambidose_type_residual" not in adata.obs


def test_mark_doublets_does_not_relabel_or_drop_cells(monkeypatch):
    adata = make_toy(n_empty=30, n_cells=80, n_samples=1, seed=9)
    classify_droplets(adata, empty_umi_max=80)
    before = adata.obs[DROPLET_KEY].astype(str).to_numpy().copy()

    def fake_scrublet(sub, **kwargs):
        sub.obs["doublet_score"] = 0.5
        predicted = np.zeros(sub.n_obs, dtype=bool)
        predicted[:5] = True
        sub.obs["predicted_doublet"] = predicted

    monkeypatch.setattr("scanpy.pp.scrublet", fake_scrublet)
    mark_doublets(adata)

    np.testing.assert_array_equal(adata.obs[DROPLET_KEY].astype(str), before)
    assert int(adata.obs["ambidose_doublet"].sum()) == 5


def test_mark_doublets_type_residual_needs_chi_and_two_types():
    adata = make_toy(n_empty=30, n_cells=80, n_samples=1, seed=9)
    classify_droplets(adata, empty_umi_max=80)
    adata.obs["one_type"] = "t0"
    # No chi estimated yet, and only one type -> nothing scoreable.
    mark_doublets(adata, type_key="one_type", sample_key=None)
    assert "ambidose_type_residual" in adata.obs
    assert adata.obs["ambidose_type_residual"].isna().all()


def test_mark_doublets_type_residual_scores_two_types_with_chi():
    from ambidose.pp import denoise

    adata = make_toy(n_empty=60, n_cells=80, n_samples=1, seed=11)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    before_rho = adata.obs["ambidose_rho"].to_numpy(dtype=float).copy()
    mark_doublets(adata, type_key="cell_type", sample_key=None)
    assert "ambidose_type_residual" in adata.obs
    residual = adata.obs["ambidose_type_residual"].to_numpy(dtype=float)
    is_cell = adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    assert np.isfinite(residual[is_cell]).any()
    # Diagnostic only: rho is untouched by mark_doublets.
    np.testing.assert_array_equal(adata.obs["ambidose_rho"].to_numpy(dtype=float), before_rho)


def test_type_residual_separates_synthetic_heterotypic_doublets_from_soup():
    """hgmm-style synthetic gold standard: cells built as (own type + other
    type's program) must score higher (more doublet-leaning) than cells
    built as (own type + chi-shaped ambient) of the same excess magnitude.
    """
    from anndata import AnnData
    from scipy import sparse as sp

    from ambidose.pp import CHI_KEY, DROPLET_KEY, _type_residual_score

    rng = np.random.default_rng(5)
    n_genes = 60
    chi = rng.random(n_genes)
    chi /= chi.sum()
    profile_a = np.zeros(n_genes)
    profile_a[:15] = rng.integers(20, 40, size=15)
    profile_a = profile_a / profile_a.sum()
    profile_b = np.zeros(n_genes)
    profile_b[30:45] = rng.integers(20, 40, size=15)
    profile_b = profile_b / profile_b.sum()

    n_pure = 60
    n_doub = 25
    n_soup = 25
    pure_a = rng.multinomial(300, profile_a, size=n_pure)
    pure_b = rng.multinomial(300, profile_b, size=n_pure)
    doublets = (
        np.vstack(
            [
                rng.multinomial(300, profile_a, size=n_doub),
                rng.multinomial(150, profile_b, size=n_doub),
            ]
        )
        .reshape(n_doub, 2, n_genes)
        .sum(axis=1)
    )
    soup = rng.multinomial(300, profile_a, size=n_soup) + rng.multinomial(150, chi, size=n_soup)

    x = np.vstack([pure_a, pure_b, doublets, soup]).astype(np.float64)
    adata = AnnData(sp.csr_matrix(x))
    adata.obs["cell_type"] = ["a"] * n_pure + ["b"] * n_pure + ["a"] * n_doub + ["a"] * n_soup
    adata.obs[DROPLET_KEY] = "cell"
    adata.var[CHI_KEY] = chi
    is_cell = np.ones(adata.n_obs, dtype=bool)
    score = _type_residual_score(
        adata, is_cell=is_cell, type_key="cell_type", sample_key=None, layer=None
    )
    doub_idx = np.arange(2 * n_pure, 2 * n_pure + n_doub)
    soup_idx = np.arange(2 * n_pure + n_doub, 2 * n_pure + n_doub + n_soup)
    assert np.isfinite(score[doub_idx]).all()
    assert np.isfinite(score[soup_idx]).all()
    assert float(np.median(score[doub_idx])) > float(np.median(score[soup_idx]))


def test_estimate_chi_recovers_true_chi():
    adata = make_toy(n_empty=300, n_cells=40, n_samples=2, seed=2)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    chi = estimate_chi(adata, sample_key="sample")
    true = adata.uns["true_chi"]
    # uns index follows sample unique order in appearance: s0, s1
    names = list(adata.uns["ambidose_chi"].index)
    for i, name in enumerate(names):
        s_idx = int(name[1:])
        assert _cosine(chi[i], true[s_idx]) > 0.95


def test_mc_worker_count_capped_by_cpu_and_memory(monkeypatch):
    from ambidose.pp import _mc_worker_count

    monkeypatch.setattr("ambidose._shared._usable_cpu_count", lambda: 8)
    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: None)
    assert _mc_worker_count(20) == 8

    # a tight memory budget must cap workers below the CPU count
    per_worker = 256 * 1024**2
    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: per_worker * 2 * 2)
    assert _mc_worker_count(20) == 2

    # a single bin/job never needs a worker pool
    assert _mc_worker_count(1) == 1


def test_configure_scanpy_n_jobs_auto_detects_and_overrides(monkeypatch):
    import scanpy as sc

    from ambidose.pp import _configure_scanpy_n_jobs

    monkeypatch.setattr("ambidose._shared._usable_cpu_count", lambda: 6)
    previous = sc.settings.n_jobs
    assert _configure_scanpy_n_jobs(None) == 6
    assert sc.settings.n_jobs == previous
    assert _configure_scanpy_n_jobs(-1) == 6
    assert _configure_scanpy_n_jobs(3) == 3
    assert sc.settings.n_jobs == previous
    with pytest.raises(ValueError, match="n_jobs"):
        _configure_scanpy_n_jobs(0)
    with pytest.raises(ValueError, match="n_jobs"):
        _configure_scanpy_n_jobs(-2)


def test_multinomial_mc_pvals_parallel_matches_serial(monkeypatch):
    from ambidose.pp import _good_turing_proportions, _multinomial_mc_pvals

    rng = np.random.default_rng(11)
    n_genes = 800
    n_test = 300
    chi = rng.random(n_genes)
    chi /= chi.sum()
    p_amb = _good_turing_proportions(rng.multinomial(50000, chi))
    logp = np.log(np.clip(p_amb, 1e-300, None))
    totals = np.geomspace(150, 3000, n_test).astype(np.int64)
    x = sparse.csr_matrix(
        np.vstack([rng.multinomial(int(n), chi) for n in totals]).astype(np.float64)
    )
    test = np.ones(n_test, dtype=bool)
    kwargs = {"lower": 100, "niters": 2000, "random_state": 3}

    pvals_parallel = _multinomial_mc_pvals(
        x, totals.astype(np.float64), test, p_amb, logp, **kwargs
    )

    monkeypatch.setattr("ambidose.pp._mc_worker_count", lambda *a, **k: 1)
    pvals_serial = _multinomial_mc_pvals(x, totals.astype(np.float64), test, p_amb, logp, **kwargs)

    np.testing.assert_array_equal(pvals_parallel, pvals_serial)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empty_umi_max": -1},
        {"empty_umi_min": -1, "empty_umi_max": 100},
        {"empty_umi_min": 100, "empty_umi_max": 100},
        {"empty_umi_min": 200, "empty_umi_max": 100},
    ],
)
def test_classify_rejects_invalid_empty_umi_range_atomically(kwargs):
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=34)
    before = adata.obs.copy(deep=True)
    with pytest.raises(ValueError):
        classify_droplets(adata, **kwargs)
    assert adata.obs.equals(before)


def test_classify_whitelist_failure_is_atomic():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=38)
    before = adata.obs.copy(deep=True)
    with pytest.raises(ValueError, match="matched"):
        classify_droplets(adata, cell_barcodes=["missing-barcode"])
    assert adata.obs.equals(before)


def test_call_cells_rejects_unused_expect_cells_atomically():
    adata = make_toy(n_empty=20, n_cells=20, n_samples=1, seed=48)
    before = adata.obs.copy(deep=True)

    with pytest.raises(ValueError, match="expect_cells is not used"):
        call_cells(adata, method="diem", expect_cells=10)

    assert adata.obs.equals(before)


def test_call_cells_requires_noncell_candidates_atomically():
    adata = make_toy(n_empty=10, n_cells=10, n_samples=1, seed=49)
    before = adata.obs.copy(deep=True)

    with pytest.raises(ValueError, match="non-cell candidates"):
        call_cells(adata, method="force", expect_cells=adata.n_obs)

    assert adata.obs.equals(before)


def test_call_cells_failure_is_atomic():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=39)
    before = adata.obs.copy(deep=True)
    with pytest.raises(ValueError, match="matched"):
        call_cells(adata, method="chi", cell_barcodes=["missing-barcode"], niters=1)
    assert adata.obs.equals(before)


def test_classify_keeps_rejected_whitelist_barcodes_out_of_empty_pool():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=36)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    classify_droplets(
        adata,
        empty_umi_max=1000,
        cell_barcodes=cells[:2],
        other_barcodes=cells[2:],
    )
    labels = adata.obs["ambidose_droplet"].astype(str)
    assert (labels.loc[cells[:2]] == "cell").all()
    assert (labels.loc[cells[2:]] == "other").all()


def test_expected_cells_keeps_other_barcodes_out_of_empty_pool():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=37)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    rejected = [str(adata.obs_names[int(np.argmin(totals))])]
    classify_droplets(
        adata,
        expected_cells=2,
        empty_umi_max=1000,
        other_barcodes=rejected,
    )
    assert str(adata.obs.loc[rejected[0], "ambidose_droplet"]) == "other"


def test_classify_mutually_exclusive_inputs_fail_atomically():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=35)
    before = adata.obs.copy(deep=True)
    with pytest.raises(ValueError):
        classify_droplets(adata, cell_barcodes=[adata.obs_names[0]], expected_cells=1)
    assert adata.obs.equals(before)


def test_rho_trust_skips_missing_type_sentinel():
    from anndata import AnnData

    from ambidose.pp import _write_rho_trust

    adata = AnnData(sparse.csr_matrix(np.ones((12, 2), dtype=int)))
    adata.obs[DROPLET_KEY] = "cell"
    adata.obs["cell_type"] = ["A"] * 10 + [np.nan, np.nan]
    adata.obs["ambidose_rho"] = 0.1
    adata.obs["ambidose_n_dose_genes"] = 10
    adata.obs["ambidose_dose_fallback"] = False
    adata.obs["ambidose_one_type"] = False
    adata.uns["ambidose"] = {"dose_type_key": "cell_type"}
    _write_rho_trust(adata, droplet_key=DROPLET_KEY, cell_label="cell")
    assert not adata.obs["ambidose_trust_type_structure"].to_numpy(dtype=bool).any()
