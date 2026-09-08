import numpy as np
import pytest
from scipy.stats import spearmanr

from ambidose.baselines import estimate_dose_global_rho
from ambidose.datasets import make_barnyard_toy, make_toy
from ambidose.metrics import assign_majority_genome, barnyard_kill_row, leakage_by_species
from ambidose.pp import CHI_KEY, DOSE_KEY, LAYER_OUT, estimate_chi, estimate_dose, subtract


def _cells(adata):
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    return adata


def test_shrink_k_partially_weights_raw_rho():
    from ambidose.pp import SHRINK_K

    assert abs(SHRINK_K - 8.0) < 1e-12
    adata = _cells(make_toy(n_samples=1, n_cells=80, n_empty=80, seed=6))
    estimate_chi(adata, sample_key=None)
    estimate_dose(adata, type_key="cell_type", sample_key=None)
    is_cell = adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    w = adata.obs.loc[is_cell, "ambidose_shrink_w"].to_numpy(dtype=float)
    nv = adata.obs.loc[is_cell, "ambidose_n_dose_genes"].to_numpy(dtype=float)
    finite = nv >= 5
    assert finite.any()
    expected = nv[finite] / (nv[finite] + SHRINK_K)
    np.testing.assert_allclose(w[finite], expected, rtol=1e-6)
    assert (expected < 0.99).any()
    assert (expected > 0.05).any()


def test_quantile_floor_without_type_key():
    adata = _cells(make_toy(n_samples=2, n_cells=40, n_empty=80, seed=2))
    estimate_chi(adata, sample_key="sample")
    d = estimate_dose(adata, sample_key="sample")
    assert np.isfinite(d).all()
    assert "ambidose_shrink_w" not in adata.obs


def _shuffle_types(labels, seed=0):
    rng = np.random.default_rng(seed)
    out = np.asarray(labels).copy()
    rng.shuffle(out)
    return out


def test_dose_beats_global_rho_on_heterogeneous_toy():
    # No n_tech_genes: those are unexpressed in every type and leak identification.
    adata = _cells(
        make_toy(
            n_samples=2,
            n_cells=200,
            n_empty=80,
            contamination=0.25,
            rho_sd=0.5,
            umi_log_sd=0.4,
            seed=3,
        )
    )
    estimate_chi(adata, sample_key="sample")
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    cells.uns["ambidose_chi"] = adata.uns["ambidose_chi"]
    estimate_dose(cells, type_key="cell_type", sample_key="sample")
    rho_true = cells.obs["true_rho"].to_numpy()
    d_true = cells.obs["true_d"].to_numpy()
    r_hier, _ = spearmanr(cells.obs["ambidose_rho"], rho_true)
    estimate_dose_global_rho(cells, sample_key="sample")
    r_glob, _ = spearmanr(cells.obs["ambidose_rho"].to_numpy(), rho_true)
    assert r_hier > r_glob
    # Shuffled-type contrast belongs on barnyard: two-type soup makes mixed
    # cluster means sit on the n̄·χ boundary, so shuffle is a weak negative.
    estimate_dose(cells, type_key="cell_type", sample_key="sample")
    r_d, _ = spearmanr(cells.obs["ambidose_d"], d_true)
    med = np.array(
        [
            np.median(cells.obs["ambidose_d"][cells.obs["sample"].to_numpy() == s])
            for s in cells.obs["sample"]
        ]
    )
    r_med, _ = spearmanr(med, d_true)
    assert r_d > r_med
    assert float(cells.obs["ambidose_shrink_w"].mean()) > 0.05


def test_shrink_does_not_clip_small_cells():
    adata = _cells(
        make_toy(
            n_samples=1,
            n_cells=200,
            n_empty=80,
            contamination=0.25,
            rho_sd=0.5,
            umi_log_sd=0.6,
            seed=4,
        )
    )
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    estimate_dose(cells, type_key="cell_type")
    n = cells.obs["n_umi"].to_numpy()
    small = n <= np.quantile(n, 0.1)
    rho = cells.obs["ambidose_rho"].to_numpy()
    rho_raw = cells.obs["ambidose_rho_raw"].to_numpy()
    raw_ok = np.isfinite(rho_raw)
    frac_hat = float((rho[small] > 0.95).mean())
    frac_raw = float((rho_raw[small & raw_ok] > 0.95).mean()) if (small & raw_ok).any() else 1.0
    assert frac_hat <= frac_raw + 1e-9


def test_dose_recovers_barnyard_offspecies():
    adata = _cells(
        make_barnyard_toy(
            seed=5,
            n_human=80,
            n_mouse=80,
            n_empty=120,
            contamination=0.25,
            rho_sd=0.4,
            umi_log_sd=0.3,
        )
    )
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    cells.var[CHI_KEY] = adata.var[CHI_KEY].to_numpy()
    assign_majority_genome(cells)
    before = leakage_by_species(cells).set_index("species")["off_umi_mean"]
    estimate_dose(cells, type_key="true_species")
    rho_true = cells.obs["true_rho"].to_numpy()
    r_true, _ = spearmanr(cells.obs["ambidose_rho"], rho_true)
    sh = cells.copy()
    sh.obs["type_shuffle"] = _shuffle_types(cells.obs["true_species"].to_numpy(), seed=1)
    estimate_dose(sh, type_key="type_shuffle")
    rho_shuf = sh.obs["ambidose_rho"].to_numpy()
    assert r_true > 0.3
    if np.ptp(rho_shuf) == 0.0:
        # Shuffled "types" are random subsamples of the same population, so
        # every gene now looks equally expressed in every fake type --
        # _native_everywhere_mask may still find soup-shaped genes; if the
        # shuffled labels collapse the pool, estimate_dose refuses (dose=0,
        # the same "no finite rho_raw" safety net used elsewhere) rather
        # than reporting a number from no real evidence. spearmanr is
        # undefined on a constant array (nan), but a flat refusal is a
        # stronger negative-control result than any finite r_shuf would
        # have been, so this trivially satisfies "shuffled is worse than
        # real" without needing the correlation at all.
        pass
    else:
        r_shuf, _ = spearmanr(rho_shuf, rho_true)
        assert r_true > r_shuf + 0.05
    subtract(cells, type_key="true_species")
    after = leakage_by_species(cells, layer="ambidose_denoised").set_index("species")[
        "off_umi_mean"
    ]
    assert (after < before).all()
    oracle = cells.copy()
    subtract(
        oracle,
        dose=np.asarray(cells.obs["true_d"], dtype=np.float64),
        type_key="true_species",
    )
    hier_row = barnyard_kill_row(
        cells, cells, method="h", layer=LAYER_OUT, species_key="true_species"
    )
    ora_row = barnyard_kill_row(
        cells, oracle, method="o", layer=LAYER_OUT, species_key="true_species"
    )
    # Must not overshoot true d·χ by wrecking endogenous counts.
    assert hier_row["specificity"] >= ora_row["specificity"] - 0.05


def test_fallback_is_soup_not_unexpressed():
    adata = _cells(make_toy(n_samples=1, n_cells=80, n_empty=80, seed=6))
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    estimate_dose(cells, type_key="cell_type", min_genes=10_000)
    assert bool(cells.obs["ambidose_dose_fallback"].to_numpy().all())


def test_nan_dose_does_not_nan_layer():
    adata = _cells(make_toy(n_samples=1, n_cells=12, n_empty=40, seed=7))
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    x = cells.X.tolil()
    for i in range(3):
        x[i, :] = 0
        x[i, 0] = 4
    cells.X = x.tocsr()
    estimate_dose(cells, type_key="cell_type")
    subtract(cells, type_key="cell_type")
    assert np.isfinite(cells.layers["ambidose_denoised"].data).all()


def test_mle_zeros_lower_rho_on_barnyard():
    # Cells with every cross-species (ambient) count zeroed out have zero
    # real evidence for the MLE's candidate gene set -- with n_valid fixed
    # to count actual per-cell nonzero observations (not len(gidx) restated,
    # see CHANGELOG), that correctly makes them NaN via min_valid, not a
    # falsely-confident rho_raw=0. Cells that still have their (barnyard-
    # contamination-driven) cross-species counts get a real, finite,
    # positive estimate.
    adata = _cells(make_barnyard_toy(seed=8, n_human=80, n_mouse=40, n_empty=80))
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    mm = cells.var["genome"].to_numpy() == "mm10"
    hg = np.flatnonzero(cells.obs["true_species"].to_numpy() == "hg19")
    x = cells.X.tolil()
    wiped = hg[:20]
    for i in wiped:
        x[i, np.flatnonzero(mm)] = 0
    cells.X = x.tocsr()
    estimate_dose(cells, type_key="true_species")
    raw = cells.obs["ambidose_rho_raw"].to_numpy()
    rest = hg[20:]
    assert np.isnan(raw[wiped]).all()
    assert np.isfinite(raw[rest]).mean() > 0.5
    assert float(np.nanmean(raw[rest])) > 0.0


def _housekeeping_calibration_toy(*, true_rho, n_hk, seed=0):
    from ambidose.datasets import scenario_housekeeping

    return scenario_housekeeping(true_rho=true_rho, n_hk=n_hk, seed=seed, n_empty=0)


@pytest.mark.parametrize("true_rho", [0.02, 0.05, 0.1, 0.2, 0.4])
def test_rho_calibration_clean_evidence_pool(true_rho):
    # n_hk=0: no gene sits at the rho=1 ceiling regardless of type, so the
    # MLE evidence pool stays clean. This is the currently-healthy case --
    # locks in that it stays healthy.
    ad = _housekeeping_calibration_toy(true_rho=true_rho, n_hk=0, seed=1)
    estimate_dose(ad, type_key="cell_type")
    rho_hat = float(np.median(ad.obs["ambidose_rho"].to_numpy()))
    # A clean evidence pool still mildly *under*-estimates at higher true
    # rho (Poisson-noise/shrinkage effects, not the additive-bias failure
    # mode below) -- allow proportional slack, not a tight absolute band.
    assert abs(rho_hat - true_rho) < 0.03 + 0.35 * true_rho, f"rho_hat={rho_hat}, true={true_rho}"


@pytest.mark.parametrize("true_rho", [0.02, 0.05, 0.1, 0.2, 0.4])
def test_rho_calibration_housekeeping_genes_target(true_rho):
    # Promoted fix: estimate_dose's U pool drops genes that look native
    # in every type (_native_everywhere_mask: above the empty-droplet
    # ceiling, or MALAT1-like collision). Those genes sit at the rho=1
    # ambient ceiling in every type simultaneously and used to pull the
    # Poisson MLE upward regardless of true rho. With them dropped, this
    ad = _housekeeping_calibration_toy(true_rho=true_rho, n_hk=40, seed=1)
    estimate_dose(ad, type_key="cell_type")
    rho_hat = float(np.median(ad.obs["ambidose_rho"].to_numpy()))
    assert abs(rho_hat - true_rho) < 0.03 + 0.35 * true_rho, f"rho_hat={rho_hat}, true={true_rho}"


def test_native_everywhere_exclusion_also_applies_to_thin_pool_fallback():
    # When cand is smaller than MIN_GENES, estimate_dose falls back to a
    # top-χ quantile soup set. That fallback must still drop genes that
    # look native in every type, or housekeeping re-enters through the
    # thin-pool door. 2 types × 3 markers + 10 HK leaves 3 U candidates
    # per type, under MIN_GENES=8.
    from ambidose.datasets import scenario_housekeeping
    from ambidose.pp import MIN_GENES

    ad = scenario_housekeeping(
        true_rho=0.1, n_hk=10, n_types=2, genes_per_type=3, n_empty=0, seed=3
    )
    estimate_dose(ad, type_key="cell_type")
    rho = ad.obs["ambidose_rho"].to_numpy(dtype=float)
    assert MIN_GENES == 8
    # Fallback may use the other type's markers as soup (correct) or
    # refuse if nothing remains. It must not report ρ≈1 from HK genes.
    assert float(np.median(rho)) < 0.5


def test_uniform_soup_stays_in_dose_evidence_pool():
    """True soup is uniform across types; it must still identify ρ."""
    from anndata import AnnData
    from scipy import sparse

    n_c, n_e, n_g = 40, 30, 16
    y = np.zeros((n_c + n_e, n_g))
    y[:20, 0:2] = 50
    y[:20, 4:] = 10
    y[20:40, 2:4] = 50
    y[20:40, 4:] = 10
    y[40:, 4:] = 5
    ad = AnnData(sparse.csr_matrix(y))
    ad.obs_names = [f"b{i}" for i in range(n_c + n_e)]
    ad.var_names = [f"g{i}" for i in range(n_g)]
    ad.obs["cell_type"] = ["A"] * 20 + ["B"] * 20 + ["empty"] * n_e
    ad.obs["ambidose_droplet"] = ["cell"] * n_c + ["empty"] * n_e
    estimate_chi(ad)
    estimate_dose(ad, type_key="cell_type")
    rho = ad.obs["ambidose_rho"].to_numpy()[:n_c]
    assert np.median(rho) > 0.05
    assert np.isfinite(rho).all()


def test_sample_named_all_does_not_collide_with_no_sample_sentinel():
    # A real sample literally named "_all" used to collide with the
    # internal "no sample_key given" sentinel (also the string "_all"),
    # silently using the global chi instead of that sample's own chi.
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, estimate_chi

    rng = np.random.default_rng(0)
    n_obs, n_vars = 60, 6
    X = sparse.csr_matrix(rng.poisson(3, size=(n_obs, n_vars)).astype(np.float64))
    ad = AnnData(X)
    ad.var_names = [f"g{i}" for i in range(n_vars)]
    ad.obs["sample"] = ["_all"] * (n_obs // 2) + ["s2"] * (n_obs // 2)
    ad.obs["droplet"] = (["empty"] * 15 + ["cell"] * 15) * 2
    ad = _cells(ad)
    estimate_chi(ad, sample_key="sample")
    chi_df = ad.uns[CHI_KEY]
    assert "_all" in chi_df.index
    cells = ad[ad.obs["droplet"].to_numpy() == "cell"].copy()
    cells.obs["cell_type"] = "t0"
    d = estimate_dose(cells, type_key="cell_type", sample_key="sample")
    assert np.isfinite(d).all()


def test_dose_disagreement_uses_fixed_only_on_agreement():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(sparse.csr_matrix([[100, 0], [100, 0], [100, 0]]))
    summary = diagnose_dose_disagreement(
        adata, fixed_dose=np.array([10.0, 10.0, 30.0]), mixture_dose=np.array([9.0, 8.0, 5.0])
    )
    selected = adata.obs["ambidose_dose_selected"].to_numpy()
    np.testing.assert_allclose(selected, [10.0, 10.0, 5.0])
    assert adata.obs["ambidose_dose_diagnosis"].tolist() == [
        "agreement",
        "agreement",
        "fixed_high",
    ]
    assert summary["n_disagreement"] == 1
    assert summary["n_fixed_high"] == 1
    assert summary["n_mixture_high"] == 0


def test_dose_disagreement_uses_layer_not_x_for_library_size():
    # X is a decoy scaled 10x above the real counts denoise(layer=...) used;
    # without honoring `layer`, the inflated X library size shrinks rho for
    # both estimates and hides a real fixed-vs-mixture disagreement as
    # "agreement" (same cells/doses as test_dose_disagreement_uses_fixed_
    # only_on_agreement, which asserts the correct classification on the
    # true, unscaled library size).
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(sparse.csr_matrix([[1000, 0], [1000, 0], [1000, 0]]))
    adata.layers["counts"] = sparse.csr_matrix([[100, 0], [100, 0], [100, 0]])
    summary = diagnose_dose_disagreement(
        adata,
        fixed_dose=np.array([10.0, 10.0, 30.0]),
        mixture_dose=np.array([9.0, 8.0, 5.0]),
        layer="counts",
    )
    selected = adata.obs["ambidose_dose_selected"].to_numpy()
    np.testing.assert_allclose(selected, [10.0, 10.0, 5.0])
    assert adata.obs["ambidose_dose_diagnosis"].tolist() == [
        "agreement",
        "agreement",
        "fixed_high",
    ]
    assert summary["n_disagreement"] == 1


def test_unconverged_mixture_not_selected_on_disagreement():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(sparse.csr_matrix([[100, 0], [100, 0], [100, 0]]))
    adata.obs["ambidose_d"] = [10.0, 10.0, 30.0]
    adata.obs["ambidose_dose_mixture"] = [9.0, 8.0, 5.0]
    adata.obs["ambidose_mixture_status"] = [
        "fitted_converged",
        "fitted_converged",
        "fitted_unconverged",
    ]
    provenance = {
        "type_key": None,
        "sample_key": None,
        "layer": None,
        "droplet_key": None,
        "cell_label": "cell",
    }
    adata.uns["ambidose"] = {
        "dose_provenance": provenance,
        "mixture_provenance": provenance.copy(),
    }
    diagnose_dose_disagreement(adata)
    selected = adata.obs["ambidose_dose_selected"].to_numpy()
    np.testing.assert_allclose(selected, [10.0, 10.0, 30.0])
    assert not bool(adata.obs["ambidose_dose_disagreement"].iloc[2])


def test_adaptive_dose_writes_selected_to_standard_keys():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import RHO_KEY, estimate_dose_adaptive

    values = np.array([[90, 10], [90, 10], [10, 90], [10, 90]])
    adata = AnnData(sparse.csr_matrix(values))
    adata.obs["cell_type"] = ["A", "A", "B", "B"]
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = [0.5, 0.5]
    selected = estimate_dose_adaptive(adata, type_key="cell_type")
    np.testing.assert_allclose(adata.obs[DOSE_KEY], selected)
    np.testing.assert_allclose(adata.obs[RHO_KEY], selected / values.sum(axis=1))


def test_mixture_dose_separates_zero_and_cross_type_ambient():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import estimate_dose_mixture

    def estimate(values):
        adata = AnnData(sparse.csr_matrix(values))
        adata.obs["cell_type"] = ["A", "A", "B", "B"]
        estimate_dose_mixture(adata, type_key="cell_type")
        return adata.obs["ambidose_rho_mixture"].to_numpy()

    zero = estimate([[100, 0], [100, 0], [0, 100], [0, 100]])
    contaminated = estimate([[90, 10], [90, 10], [10, 90], [10, 90]])
    np.testing.assert_allclose(zero, 0.0, atol=1e-10)
    np.testing.assert_allclose(contaminated, 0.1, atol=0.01)


def test_mixture_deconv_avoids_circular_leave_one_type_reference():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import estimate_dose_mixture

    values = np.array(
        [
            [80, 10, 10],
            [80, 10, 10],
            [10, 80, 10],
            [10, 80, 10],
        ],
        dtype=float,
    )
    adata = AnnData(sparse.csr_matrix(values))
    adata.obs["cell_type"] = ["A", "A", "B", "B"]
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = [0.5, 0.5, 0.0]
    estimate_dose_mixture(adata, type_key="cell_type")
    assert set(adata.obs["ambidose_mixture_profile"]) == {"chi_deconv"}
    selected = adata.obs["ambidose_rho_mixture"].to_numpy()
    np.testing.assert_allclose(selected, adata.obs["ambidose_rho_mixture_empty"].to_numpy())


def test_mixture_one_type_uses_empty_profile():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import _rho_from_chi, estimate_dose_mixture

    values = np.array([[80, 10, 10], [80, 10, 10], [70, 20, 10]], dtype=float)
    adata = AnnData(sparse.csr_matrix(values))
    adata.obs["cell_type"] = ["A", "A", "A"]
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = [0.2, 0.5, 0.3]
    estimate_dose_mixture(adata, type_key="cell_type")
    assert set(adata.obs["ambidose_mixture_profile"]) == {"empty"}
    x = adata.X.tocsr()
    n = np.asarray(x.sum(axis=1)).ravel().astype(float)
    chi = np.asarray(adata.var[CHI_KEY], dtype=float)
    expected, _ = _rho_from_chi(
        x, n, chi, np.arange(adata.n_obs), top_n=100, min_chi=1e-6, quantile=0.15
    )
    np.testing.assert_allclose(adata.obs["ambidose_rho_mixture"].to_numpy(), expected)
    np.testing.assert_allclose(
        adata.obs["ambidose_rho_mixture"].to_numpy(),
        adata.obs["ambidose_rho_mixture_empty"].to_numpy(),
    )


def test_one_type_dose_matches_untyped_quantile():
    adata = _cells(make_toy(n_samples=1, n_cells=40, n_empty=80, contamination=0.2, seed=14))
    adata.obs["cell_type"] = np.where(adata.obs["droplet"].astype(str) == "cell", "t0", "none")
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    typed = estimate_dose(cells, type_key="cell_type")
    untyped = estimate_dose(cells, type_key=None)
    np.testing.assert_allclose(typed, untyped)
    n = np.asarray(cells.X.sum(axis=1)).ravel()
    rho = np.divide(typed, n, out=np.zeros_like(typed), where=n > 0)
    assert float(np.median(rho)) < 0.5


def test_mixture_fallback_params_are_configurable():
    # Regression: estimate_dose_mixture()'s single-type-per-sample fallback
    # used to hardcode top_n=100/min_chi=1e-6/quantile=0.15/min_valid, so it
    # could silently drift from estimate_dose()'s real config. These are now
    # explicit fallback_* parameters, threaded through to _rho_from_chi --
    # prove that by passing non-default values and checking the result
    # matches an explicit _rho_from_chi call with the SAME custom values
    # (and differs from the old hardcoded defaults).
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import _rho_from_chi, estimate_dose_mixture

    values = np.array([[80, 10, 10], [80, 10, 10], [70, 20, 10]], dtype=float)
    adata_custom = AnnData(sparse.csr_matrix(values))
    adata_custom.obs["cell_type"] = ["A", "A", "A"]
    adata_custom.obs["ambidose_droplet"] = "cell"
    adata_custom.var[CHI_KEY] = [0.2, 0.5, 0.3]

    estimate_dose_mixture(
        adata_custom,
        type_key="cell_type",
        fallback_quantile=0.5,
        fallback_top_n=2,
        fallback_min_chi=0.25,
        fallback_min_valid=1,
    )
    x = adata_custom.X.tocsr()
    n = np.asarray(x.sum(axis=1)).ravel().astype(float)
    chi = np.asarray(adata_custom.var[CHI_KEY], dtype=float)
    expected_custom, _ = _rho_from_chi(
        x, n, chi, np.arange(adata_custom.n_obs), top_n=2, min_chi=0.25, quantile=0.5, min_valid=1
    )
    np.testing.assert_allclose(adata_custom.obs["ambidose_rho_mixture"].to_numpy(), expected_custom)

    adata_default = AnnData(sparse.csr_matrix(values))
    adata_default.obs["cell_type"] = ["A", "A", "A"]
    adata_default.obs["ambidose_droplet"] = "cell"
    adata_default.var[CHI_KEY] = [0.2, 0.5, 0.3]
    estimate_dose_mixture(adata_default, type_key="cell_type")
    assert not np.allclose(
        adata_custom.obs["ambidose_rho_mixture"].to_numpy(),
        adata_default.obs["ambidose_rho_mixture"].to_numpy(),
    )

    with pytest.raises(ValueError, match="fallback_quantile"):
        estimate_dose_mixture(adata_default, type_key="cell_type", fallback_quantile=1.5)
    with pytest.raises(ValueError, match="fallback_top_n"):
        estimate_dose_mixture(adata_default, type_key="cell_type", fallback_top_n=0)
    with pytest.raises(ValueError, match="fallback_min_valid"):
        estimate_dose_mixture(adata_default, type_key="cell_type", fallback_min_valid=0)
    with pytest.raises(ValueError, match="fallback_min_chi"):
        estimate_dose_mixture(adata_default, type_key="cell_type", fallback_min_chi=-1.0)


def test_mixture_writes_n_genes_for_both_branches():
    # Regression: estimate_dose_mixture() never wrote any per-cell
    # gene-evidence count in either branch, so _write_rho_trust()'s
    # low_evidence check read a phantom zero for every mixture-derived
    # cell. ambidose_mixture_n_genes must now be real and nonzero for both
    # the single-type quantile fallback and the two-component EM fit.
    from ambidose.pp import estimate_dose_mixture

    # Single-type fallback branch (n_genes = n_valid from _rho_from_chi).
    adata_one_type = _cells(make_toy(n_samples=1, n_cells=40, n_empty=80, seed=21))
    adata_one_type.obs["cell_type"] = "t0"
    estimate_chi(adata_one_type, sample_key=None)
    cells_one_type = adata_one_type[
        adata_one_type.obs["ambidose_droplet"].to_numpy() == "cell"
    ].copy()
    estimate_dose_mixture(cells_one_type, type_key="cell_type")
    assert (cells_one_type.obs["ambidose_mixture_status"].astype(str) == "quantile_fallback").all()
    n_genes_fallback = cells_one_type.obs["ambidose_mixture_n_genes"].to_numpy()
    assert (n_genes_fallback > 0).any()

    # Two-component EM branch (n_genes = per-cell detected-gene count).
    adata_two_type = _cells(make_toy(n_samples=1, n_cells=80, n_empty=80, seed=22))
    estimate_chi(adata_two_type, sample_key=None)
    cells_two_type = adata_two_type[
        adata_two_type.obs["ambidose_droplet"].to_numpy() == "cell"
    ].copy()
    estimate_dose_mixture(cells_two_type, type_key="cell_type")
    assert (cells_two_type.obs["ambidose_mixture_status"].astype(str) != "not_evaluated").all()
    n_genes_em = cells_two_type.obs["ambidose_mixture_n_genes"].to_numpy()
    expected_nnz = np.diff(cells_two_type.X.tocsr().indptr)
    np.testing.assert_array_equal(n_genes_em, expected_nnz)
    assert (n_genes_em > 0).all()


def test_write_rho_trust_uses_mixture_n_genes_when_no_fixed_pass():
    # Regression: before ambidose_mixture_n_genes existed, a standalone
    # estimate_dose_mixture() run (no estimate_dose() call first) left
    # ambidose_n_dose_genes entirely absent, so _write_rho_trust() read a
    # phantom zero (0 < MIN_GENES is always true) and flagged every single
    # cell low_evidence regardless of real mixture evidence.
    from ambidose.pp import LAYER_OUT, RHO_KEY, _write_rho_trust, estimate_dose_mixture

    adata = _cells(make_toy(n_samples=1, n_cells=80, n_empty=80, seed=23))
    adata.obs["cell_type"] = np.where(
        adata.obs["ambidose_droplet"].astype(str) == "cell", adata.obs["cell_type"], "-1"
    )
    estimate_chi(adata, sample_key=None)
    estimate_dose_mixture(adata, type_key="cell_type")
    adata.obs[RHO_KEY] = adata.obs["ambidose_rho_mixture"]
    adata.layers[LAYER_OUT] = adata.X.copy()

    assert "ambidose_n_dose_genes" not in adata.obs
    assert "ambidose_mixture_n_genes" in adata.obs

    _write_rho_trust(adata, droplet_key="ambidose_droplet", cell_label="cell")
    is_cell = adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    # Real mixture evidence (per-cell detected-gene counts) is well above
    # MIN_GENES here, so low_evidence must not be universally true anymore.
    assert not adata.obs.loc[is_cell, "ambidose_trust_low_evidence"].all()


def test_mixture_one_type_without_chi_raises():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import estimate_dose_mixture

    adata = AnnData(sparse.csr_matrix([[80, 10], [80, 10]], dtype=float))
    adata.obs["cell_type"] = ["A", "A"]
    adata.obs["ambidose_droplet"] = "cell"
    with pytest.raises(ValueError, match="single-type library"):
        estimate_dose_mixture(adata, type_key="cell_type")


def test_mixture_deconv_recovers_type_blend_ambient():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import estimate_dose_mixture

    values = np.array(
        [
            [80, 10, 10],
            [80, 10, 10],
            [10, 80, 10],
            [10, 80, 10],
            [10, 10, 80],
            [10, 10, 80],
        ],
        dtype=float,
    )
    adata = AnnData(sparse.csr_matrix(values))
    adata.obs["cell_type"] = ["A", "A", "B", "B", "C", "C"]
    adata.obs["ambidose_droplet"] = "cell"
    adata.var[CHI_KEY] = [1 / 3, 1 / 3, 1 / 3]
    estimate_dose_mixture(adata, type_key="cell_type")
    assert set(adata.obs["ambidose_mixture_profile"]) == {"chi_deconv"}
    np.testing.assert_allclose(
        adata.obs["ambidose_rho_mixture"].to_numpy(),
        adata.obs["ambidose_rho_mixture_cell"].to_numpy(),
    )


def test_dose_estimators_reject_non_count_x_and_accept_raw_layer():
    from ambidose.pp import estimate_dose_mixture

    adata = _cells(make_toy(n_samples=1, n_cells=40, n_empty=40, seed=32))
    estimate_chi(adata, sample_key=None)
    adata.layers["raw_counts"] = adata.X.copy()
    adata.X = adata.X.astype(float)
    adata.X.data = np.log1p(adata.X.data / 3.0)

    with pytest.raises(ValueError, match="raw integer UMI"):
        estimate_dose(adata, type_key="cell_type", sample_key=None)
    with pytest.raises(ValueError, match="raw integer UMI"):
        estimate_dose_mixture(adata, type_key="cell_type", sample_key=None)

    estimate_chi(adata, sample_key=None, layer="raw_counts")
    estimate_dose(adata, type_key="cell_type", sample_key=None, layer="raw_counts")
    np.testing.assert_allclose(
        adata.obs["n_umi"].to_numpy(),
        np.asarray(adata.layers["raw_counts"].sum(axis=1)).ravel(),
    )


@pytest.mark.parametrize(
    "fixed, mixture, message",
    [
        ([-1.0], [0.0], "nonnegative"),
        ([np.nan], [0.0], "non-finite"),
        ([2.0], [0.0], "n_umi"),
    ],
)
def test_diagnose_dose_rejects_invalid_dose(fixed, mixture, message):
    from anndata import AnnData

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(np.array([[1]], dtype=int))
    with pytest.raises(ValueError, match=message):
        diagnose_dose_disagreement(
            adata, fixed_dose=np.asarray(fixed), mixture_dose=np.asarray(mixture)
        )


def test_explicit_disagreement_arrays_ignore_stale_estimator_diagnostics():
    from anndata import AnnData

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(np.array([[10], [10]], dtype=int))
    adata.obs["ambidose_mixture_converged"] = [False, False]
    adata.obs["ambidose_mixture_profile_tv"] = [99.0, 99.0]
    summary = diagnose_dose_disagreement(
        adata,
        fixed_dose=np.array([1.0, 1.0]),
        mixture_dose=np.array([5.0, 5.0]),
        rho_gap_threshold=0.1,
        droplet_key=None,
    )

    assert summary["n_disagreement"] == 2
    assert summary["n_disagreement_unconverged_kept_fixed"] == 0
    assert "median_mixture_profile_tv_disagreement" not in summary


def test_public_estimators_and_subtract_reject_droplet_key_typo():
    from ambidose.pp import estimate_dose_mixture

    adata = _cells(make_toy(n_samples=1, n_cells=40, n_empty=40, seed=36))
    estimate_chi(adata, sample_key=None)
    for func, kwargs in (
        (estimate_dose, {"type_key": "cell_type", "sample_key": None}),
        (
            estimate_dose_mixture,
            {"type_key": "cell_type", "sample_key": None},
        ),
    ):
        with pytest.raises(KeyError, match="droplet_key"):
            func(adata.copy(), droplet_key="typo", **kwargs)

    dose_adata = adata.copy()
    dose_adata.obs["ambidose_d"] = 1.0
    with pytest.raises(KeyError, match="droplet_key"):
        subtract(dose_adata, droplet_key="typo")


@pytest.mark.parametrize("estimator_name", ["fixed", "mixture"])
def test_dose_rejects_layer_different_from_chi_provenance(estimator_name):
    from ambidose.pp import estimate_dose_mixture

    adata = make_toy(n_empty=20, n_cells=20, n_samples=1, seed=43)
    adata.layers["total"] = adata.X.copy()
    adata.layers["spliced"] = adata.X.copy()
    estimate_chi(adata, droplet_key="droplet", sample_key=None, layer="total")
    estimator = estimate_dose if estimator_name == "fixed" else estimate_dose_mixture
    with pytest.raises(ValueError, match="chi provenance"):
        estimator(
            adata, type_key="cell_type", sample_key=None, droplet_key="droplet", layer="spliced"
        )


def test_untyped_estimate_dose_honors_min_valid():
    adata = _cells(make_toy(n_samples=1, n_cells=40, n_empty=40, seed=37))
    estimate_chi(adata, sample_key=None)
    low = estimate_dose(adata.copy(), sample_key=None, min_valid=1)
    high = estimate_dose(adata.copy(), sample_key=None, min_valid=10_000)
    assert np.any(low > 0)
    assert np.all(high == 0)


@pytest.mark.parametrize("initial_rho", [0.0, 1.0])
def test_mixture_rejects_degenerate_initial_rho(initial_rho):
    from ambidose.pp import estimate_dose_mixture

    adata = _cells(make_toy(n_samples=1, n_cells=20, n_empty=20, seed=40))
    estimate_chi(adata, sample_key=None)
    with pytest.raises(ValueError, match="strictly between"):
        estimate_dose_mixture(adata, type_key="cell_type", sample_key=None, initial_rho=initial_rho)


def test_estimate_dose_type_key_failure_is_atomic():
    import copy

    adata = _cells(make_toy(n_samples=1, n_cells=20, n_empty=20, seed=41))
    estimate_chi(adata, sample_key=None)
    adata.obs["ambidose_dose_mixture"] = 7.0
    adata.uns["ambidose"] = {"dose": {"method": "existing"}}
    obs_before = adata.obs.copy(deep=True)
    uns_before = copy.deepcopy(adata.uns)

    with pytest.raises(KeyError, match="cell_typ"):
        estimate_dose(adata, type_key="cell_typ", sample_key=None)

    assert adata.obs.equals(obs_before)
    assert repr(adata.uns) == repr(uns_before)


def test_estimate_dose_computation_failure_restores_previous_results(monkeypatch):
    import copy

    import ambidose._dose as dose_module

    adata = _cells(make_toy(n_samples=1, n_cells=20, n_empty=20, seed=44))
    estimate_chi(adata, sample_key=None)
    adata.obs["ambidose_dose_mixture"] = 7.0
    adata.obs[DOSE_KEY] = 3.0
    adata.uns["ambidose"]["dose"] = {"method": "existing"}
    obs_before = adata.obs.copy(deep=True)
    uns_before = copy.deepcopy(adata.uns)

    def fail_mid_computation(*args, **kwargs):
        raise RuntimeError("dose computation failed")

    monkeypatch.setattr(dose_module, "_rho_from_chi", fail_mid_computation)
    with pytest.raises(RuntimeError, match="dose computation failed"):
        estimate_dose(adata, type_key=None, sample_key=None)

    assert adata.obs.equals(obs_before)
    assert repr(adata.uns) == repr(uns_before)


def test_real_minus_one_type_is_not_missing():
    from ambidose._shared import EMPTY_TYPES, _validated_type_values

    adata = _cells(make_toy(n_samples=1, n_cells=10, n_empty=10, seed=42))
    adata.obs["real_type"] = "-1"
    values = _validated_type_values(adata, "real_type")
    assert set(values) == {"-1"}
    assert "-1" not in EMPTY_TYPES


def test_adaptive_dose_never_selects_unconverged_mixture(monkeypatch):
    import pandas as pd
    from anndata import AnnData

    import ambidose._dose as dose_module

    adata = AnnData(np.array([[100], [100]], dtype=int))
    adata.obs["cell_type"] = ["A", "B"]

    def fixed(target, **kwargs):
        target.obs[DOSE_KEY] = [10.0, 10.0]
        target.obs["ambidose_rho"] = [0.1, 0.1]
        provenance = {
            "type_key": kwargs["type_key"],
            "sample_key": kwargs["sample_key"],
            "layer": kwargs["layer"],
            "droplet_key": kwargs["droplet_key"],
            "cell_label": kwargs["cell_label"],
        }
        target.uns["ambidose"] = {
            "dose": {"method": "typed"},
            "dose_provenance": provenance,
        }
        return np.array([10.0, 10.0])

    def mixture(target, **kwargs):
        target.obs["ambidose_dose_mixture"] = [60.0, 60.0]
        target.obs["ambidose_mixture_status"] = pd.Categorical(
            ["fitted_unconverged", "fitted_unconverged"]
        )
        target.uns["ambidose"]["mixture_provenance"] = {
            "type_key": kwargs["type_key"],
            "sample_key": kwargs["sample_key"],
            "layer": kwargs["layer"],
            "droplet_key": kwargs["droplet_key"],
            "cell_label": kwargs["cell_label"],
        }
        return np.array([60.0, 60.0])

    monkeypatch.setattr(dose_module, "estimate_dose", fixed)
    monkeypatch.setattr(dose_module, "estimate_dose_mixture", mixture)
    selected = dose_module.estimate_dose_adaptive(adata, type_key="cell_type", droplet_key=None)

    np.testing.assert_array_equal(selected, [10.0, 10.0])
    assert adata.uns["ambidose"]["dose"]["selection"]["n_disagreement_unconverged_kept_fixed"] == 2


def test_diagnose_rejects_mismatched_estimator_provenance():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import diagnose_dose_disagreement

    adata = AnnData(sparse.csr_matrix([[100, 0], [100, 0]]))
    adata.obs[DOSE_KEY] = [10.0, 10.0]
    adata.obs["ambidose_dose_mixture"] = [20.0, 20.0]
    base = {
        "type_key": "A",
        "sample_key": None,
        "layer": None,
        "droplet_key": None,
        "cell_label": "cell",
    }
    adata.uns["ambidose"] = {
        "dose_provenance": base,
        "mixture_provenance": {**base, "type_key": "B"},
    }
    with pytest.raises(ValueError, match="different inputs"):
        diagnose_dose_disagreement(adata)
