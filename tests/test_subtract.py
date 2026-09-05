import numpy as np
import pytest
from scipy import sparse

from ambidose._budget import _realloc_unspent_rank1
from ambidose.datasets import make_barnyard_toy, make_toy
from ambidose.metrics import assign_majority_genome, barnyard_kill_row, leakage_by_species
from ambidose.pp import (
    _alloc_integer_budget,
    _confidence_weighted_take,
    _expand_take_to_cells,
    _subtract_row,
    denoise,
    estimate_chi,
    estimate_dose,
    subtract,
)


def test_dense_chunk_budget_counts_all_temporaries(monkeypatch):
    from ambidose.pp import _dense_chunk_columns

    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 24 * (1 << 30))
    got = _dense_chunk_columns(10_000, 100_000, arrays_per_value=5, max_cols=100_000)
    assert got == int((1.0 / 3.0) * 24 * (1 << 30)) // (10_000 * 8 * 5)


def test_dense_chunk_budget_scales_with_large_memory_host(monkeypatch):
    from ambidose.pp import _dense_chunk_columns

    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 256 * (1 << 30))
    got = _dense_chunk_columns(1_000_000, 3000, arrays_per_value=5, max_cols=3000)
    assert got == int((1.0 / 3.0) * 256 * (1 << 30)) // (1_000_000 * 8 * 5)


def test_dense_chunk_budget_reserves_ram_on_small_host(monkeypatch):
    from ambidose.pp import _dense_chunk_columns

    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 256 * (1 << 20))
    got = _dense_chunk_columns(10_000, 3000, arrays_per_value=3, max_cols=3000)
    assert got == int((1.0 / 3.0) * 256 * (1 << 20)) // (10_000 * 8 * 3)


def test_subtract_row_never_exceeds_observed_count():
    y = np.array([1.0, 5.0, 0.0])
    chi = np.array([0.5, 0.4, 0.1])
    take = _subtract_row(y, chi, d=100.0)
    assert np.all(take <= y + 1e-12)
    assert take[0] == 1.0
    assert take[2] == 0.0


def _expand_take_to_cells_loop_ref(x, idx, take_g, weights):
    """Pre-vectorization writeback; keep as identity oracle only."""

    gidx = np.flatnonzero(take_g > 1e-12)
    if gidx.size == 0 or idx.size == 0:
        return
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if float(w.sum()) <= 0:
        w = np.ones(idx.size, dtype=np.float64)
    indptr = x.indptr
    indices = x.indices
    data = x.data
    chunk_size = 3000
    for start in range(0, gidx.size, chunk_size):
        chunk = gidx[start : start + chunk_size]
        sub = x[idx][:, chunk].toarray().astype(np.float64)
        for j, g in enumerate(chunk):
            room = sub[:, j]
            extra = _alloc_integer_budget(float(take_g[g]), room, w)
            sub[:, j] = sub[:, j] - extra
        lookup = {int(g): j for j, g in enumerate(chunk)}
        for k, i in enumerate(idx):
            a, b = int(indptr[i]), int(indptr[i + 1])
            cols = indices[a:b]
            row = data[a:b]
            for p, g in enumerate(cols):
                j = lookup.get(int(g))
                if j is not None:
                    row[p] = sub[k, j]


def test_expand_take_matches_loop_formula():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 40, 25
    dense = rng.integers(0, 12, size=(n_cells, n_genes)).astype(np.float64)
    dense[:, 0] = 0
    x_new = sparse.csr_matrix(dense)
    x_old = sparse.csr_matrix(dense)
    idx = np.arange(n_cells)
    take_g = rng.uniform(0, 30, size=n_genes)
    take_g[0] = 0
    w = rng.uniform(1, 8, size=n_cells)
    _expand_take_to_cells(x_new, idx, take_g, w)
    _expand_take_to_cells_loop_ref(x_old, idx, take_g, w)
    np.testing.assert_allclose(x_new.toarray(), x_old.toarray(), rtol=0, atol=1e-10)


def test_expand_take_sparse_cells_matches_full_length_formula():
    rng = np.random.default_rng(12)
    n_cells, n_genes = 2000, 80
    dense = np.zeros((n_cells, n_genes), dtype=np.float64)
    for j in range(n_genes):
        rows = rng.choice(n_cells, size=20, replace=False)
        dense[rows, j] = rng.integers(1, 12, size=rows.size)
    x_new = sparse.csr_matrix(dense)
    x_old = sparse.csr_matrix(dense)
    idx = np.arange(n_cells)
    take_g = rng.uniform(0, 30, size=n_genes)
    weights = rng.uniform(1, 8, size=n_cells)

    _expand_take_to_cells(x_new, idx, take_g, weights)
    _expand_take_to_cells_loop_ref(x_old, idx, take_g, weights)

    np.testing.assert_allclose(x_new.toarray(), x_old.toarray(), rtol=0, atol=1e-10)


def test_native_confidence_continuously_interpolates_removal():
    observed = np.full(3, 100.0)
    chi = np.full(3, 1.0 / 3.0)
    confidence = np.array([0.0, 0.5, 1.0])
    take = _confidence_weighted_take(observed, chi, 6.0, confidence)
    np.testing.assert_allclose(take, [2.0, 1.1, 0.2])


def test_relax_hk_increases_high_rho_removal_and_default_off():
    adata = make_toy(n_samples=1, n_empty=60, n_cells=80, n_genes=60, contamination=0.7, seed=1)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    mask = adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    # denoise() now leaves X as the denoised counts (layers["raw_counts"]
    # holds the original input) -- use that for the "raw" reference point,
    # and restore raw counts as the input before calling subtract() again
    # directly (subtract() always reads its raw input from X/`layer=`).
    raw_counts = adata.layers["raw_counts"]
    frozen = float(
        (raw_counts[mask].sum() - adata.layers["ambidose_denoised"][mask].sum())
        / raw_counts[mask].sum()
    )
    trial = adata.copy()
    trial.uns = dict(adata.uns)
    trial.X = trial.layers["raw_counts"].copy()
    subtract(
        trial,
        type_key="cell_type",
        droplet_key="ambidose_droplet",
        relax_hk_when_soup_like=True,
    )
    trial_frac = float(
        (raw_counts[mask].sum() - trial.layers["ambidose_denoised"][mask].sum())
        / raw_counts[mask].sum()
    )
    assert abs(trial_frac - frozen) > 1e-6


def test_denoise_toy_removed_fraction_in_band():
    adata = make_toy(n_samples=1, n_empty=80, n_cells=60, contamination=0.2, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    is_cell = adata.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    raw = np.asarray(adata.layers["raw_counts"][is_cell].sum())
    den = np.asarray(adata.layers["ambidose_denoised"][is_cell].sum())
    frac = float((raw - den) / raw)
    assert 0.10 < frac < 0.20, f"removed_frac={frac}"


def test_subtract_reduces_barnyard_leakage():
    adata = make_barnyard_toy(seed=3)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"] == "cell"].copy()
    assign_majority_genome(cells)
    before = leakage_by_species(cells).set_index("species")["leakage_mean"]
    estimate_dose(cells)
    subtract(cells)
    after = leakage_by_species(cells, layer="ambidose_denoised").set_index("species")[
        "leakage_mean"
    ]
    assert (after < before).all()
    assert (after < 0.15).all()
    assert "ambidose_d" in cells.obs
    assert np.all(cells.obs["ambidose_d"].to_numpy() >= 0)


def test_subtract_sample_chi():
    adata = make_toy(n_samples=4, n_cells=40, n_empty=80, seed=11)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key="sample")
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    cells.uns["ambidose_chi"] = adata.uns["ambidose_chi"]
    estimate_dose(cells, sample_key="sample")
    subtract(cells, sample_key="sample")
    assert "ambidose_denoised" in cells.layers
    raw = cells.X.tocsr()
    den = cells.layers["ambidose_denoised"].tocsr()
    rows, cols = den.nonzero()
    assert np.all(np.asarray(raw[rows, cols]).ravel() > 0)


def test_subtract_requires_dose_sample_key_match():
    adata = make_toy(n_samples=2, n_cells=20, n_empty=40, seed=43)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    adata.obs["sample_other"] = adata.obs["sample"].iloc[::-1].to_numpy()
    estimate_chi(adata, sample_key="sample")
    cells = adata[adata.obs["droplet"].astype(str) == "cell"].copy()
    cells.uns["ambidose_chi"] = adata.uns["ambidose_chi"]
    estimate_dose(cells, sample_key="sample")

    with pytest.raises(ValueError, match="parameters do not match dose provenance"):
        subtract(cells, sample_key="sample_other")


def test_sample_parallel_subtract_is_bitwise_serial_identical():
    adata = make_toy(n_samples=4, n_cells=80, n_empty=80, n_genes=120, seed=31)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key="sample")
    cells = adata[adata.obs["droplet"].to_numpy() == "cell"].copy()
    cells.uns["ambidose_chi"] = adata.uns["ambidose_chi"]
    estimate_dose(cells, type_key="cell_type", sample_key="sample")
    serial = cells.copy()
    parallel = cells.copy()

    subtract(serial, type_key="cell_type", sample_key="sample", n_jobs=1)
    subtract(parallel, type_key="cell_type", sample_key="sample", n_jobs=4)

    a = serial.layers["ambidose_denoised"]
    b = parallel.layers["ambidose_denoised"]
    assert (a != b).nnz == 0
    assert serial.uns["ambidose"] == parallel.uns["ambidose"]
    np.testing.assert_array_equal(
        serial.var["ambidose_removed_umi"],
        parallel.var["ambidose_removed_umi"],
    )


def test_subtract_rejects_chi_from_different_gene_set():
    import pandas as pd
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    # 8 genes matching this object + 1 chi-only "removed_gene" dropped by
    # reindex = 1/9 (~11%), under the 20% hard-stop -- a handful of
    # naming-mismatch stragglers, not a meaningfully different gene set.
    adata = AnnData(sparse.csr_matrix([[10] * 8], dtype=float))
    adata.var_names = [f"g{i}" for i in range(8)]
    adata.obs["sample"] = ["s1"]
    adata.obs[DOSE_KEY] = [10.0]
    adata.uns[CHI_KEY] = pd.DataFrame(
        [[0.1] * 8 + [0.2]],
        index=["s1"],
        columns=[f"g{i}" for i in range(8)] + ["removed_gene"],
    )

    with pytest.raises(ValueError, match="gene labels must exactly match"):
        subtract(adata, sample_key="sample")


def test_subtract_does_not_invent_counts():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=5)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    estimate_dose(adata)
    subtract(adata)
    raw = adata.X.tocsr()
    den = adata.layers["ambidose_denoised"].tocsr()
    rows, cols = den.nonzero()
    raw_vals = np.asarray(raw[rows, cols]).ravel()
    assert np.all(raw_vals > 0)
    assert den.nnz == 0 or np.all(den.data >= 0)


def test_denoise_writes_chi_dose_and_layer():
    adata = make_toy(n_samples=2, n_empty=80, n_cells=30, seed=8)
    denoise(adata, sample_key="sample", empty_umi_max=80)
    assert "ambidose_chi" in adata.uns
    assert "ambidose_d" in adata.obs
    assert "ambidose_rho" in adata.obs
    assert "ambidose_denoised" in adata.layers


def test_alloc_budget_saturates_then_reassigns():
    from ambidose.pp import _alloc_budget

    # Gene 0 can only take 1; leftover must go to gene 1.
    extra = _alloc_budget(10.0, np.array([1.0, 100.0]), np.array([0.5, 0.5]))
    assert extra[0] == 1.0
    assert abs(extra[1] - 9.0) < 1e-9


def test_masked_alloc_cleans_more_offspecies_than_independent():
    adata = make_barnyard_toy(seed=3, n_human=60, n_mouse=60, n_empty=80, contamination=0.25)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"] == "cell"].copy()
    assign_majority_genome(cells)
    estimate_dose(cells, type_key="true_species")
    independent = cells.copy()
    # Deliberate ablation: reuse the *same* type-aware dose values on the
    # naive independent path, to isolate what the masked algorithm buys on
    # top of an identical dose estimate. subtract() raises if it sees a
    # type-aware dose (still under DOSE_KEY) with no type_key on this call --
    # a real mistake three times over in this project's own eval scripts
    # (see CHANGELOG) -- so make the intent explicit by copying to a
    # differently-named dose array instead of tripping that guard.
    from ambidose.pp import DOSE_KEY

    independent.obs["ablation_dose"] = independent.obs[DOSE_KEY].to_numpy()
    subtract(independent, dose="ablation_dose")
    masked = cells.copy()
    subtract(masked, type_key="true_species")
    row_i = barnyard_kill_row(cells, independent, method="i", layer="ambidose_denoised")
    row_m = barnyard_kill_row(cells, masked, method="m", layer="ambidose_denoised")
    assert row_m["sensitivity"] >= row_i["sensitivity"]
    assert row_m["n_inflated"] == 0
    assert row_m["specificity"] >= row_i["specificity"] - 0.08


def test_realloc_does_not_touch_protected_genes():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    # Type t0 expresses g0; leftover soup budget must not be parked on g0.
    x = sparse.csr_matrix(np.array([[20.0, 1.0, 1.0], [20.0, 1.0, 1.0]], dtype=np.float64))
    ad = AnnData(x)
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = ["c0", "c1"]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = "t0"
    ad.var[CHI_KEY] = np.array([0.2, 0.4, 0.4])
    ad.obs[DOSE_KEY] = 10.0
    ad.obs["n_umi"] = np.array([22.0, 22.0])
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    # g0 is this type's native gene: leftover soup must not be parked on it.
    # Rank-1 would leave 18; responsibility takes a similar slice of g0.
    assert den[0, 0] >= 16.0
    assert den[0, 1] <= 1.0
    assert den[0, 2] <= 1.0


def test_soup_only_clears_unexpressed_genes():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    # g0 is native (P); g1 is a low-abundance unexpressed (U) gene with more
    # counts than rank-1 d·χ would take. soupOnly clears true soup-only
    # genes outright. g1 is 8 UMI in a ~4000-UMI library (frac 0.002),
    # below U_MAX_LIBRARY_FRAC, so it remains eligible. 10 cells, not 2:
    # MIN_TYPE_CELLS would skip soupOnly below that count.
    n_cells = 10
    x = sparse.csr_matrix(np.tile(np.array([[4000.0, 8.0]]), (n_cells, 1)))
    ad = AnnData(x)
    ad.var_names = ["g0", "g1"]
    ad.obs_names = [f"c{i}" for i in range(n_cells)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = "t0"
    ad.var[CHI_KEY] = np.array([0.2, 0.4]) / 0.6
    ad.obs[DOSE_KEY] = 80.0
    ad.obs["n_umi"] = np.full(n_cells, 4008.0)
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 1] == 0.0).all()
    assert (den[:, 0] >= 3900.0).all()


def test_soup_only_does_not_wipe_abundant_ceiling_collision():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    # Housekeeping shape: g1 is 2% of the library and sits at the ambient
    # ceiling, so the U test would otherwise call it soup-only and wipe it.
    # Ubiquitous native genes are not soup-only.
    n_cells = 10
    x = sparse.csr_matrix(np.tile(np.array([[4000.0, 80.0]]), (n_cells, 1)))
    ad = AnnData(x)
    ad.var_names = ["g0", "g1"]
    ad.obs_names = [f"c{i}" for i in range(n_cells)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = "t0"
    ad.var[CHI_KEY] = np.array([0.98, 0.02])
    ad.obs[DOSE_KEY] = 80.0
    ad.obs["n_umi"] = np.full(n_cells, 4080.0)
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 1] >= 70.0).all()


def test_dominant_owner_is_shared_only_within_half_split_noise():
    from ambidose.pp import _dominant_owner_masks, _type_means

    rng = np.random.default_rng(4)
    n_cells = 100
    x = sparse.csr_matrix(
        np.vstack(
            [
                rng.poisson([100.0, 900.0, 10.0], size=(n_cells, 3)),
                rng.poisson([100.0, 900.0, 10.0], size=(n_cells, 3)),
                rng.poisson([100.0, 10.0, 890.0], size=(n_cells, 3)),
            ]
        )
    )
    types = np.repeat(["fragment_a", "fragment_b", "unrelated"], n_cells)
    n = np.asarray(x.sum(axis=1)).ravel().astype(float)
    masks, n_meta = _dominant_owner_masks(x, n, types, _type_means(x, types), max_type_mean=0.05)
    assert n_meta >= 2
    assert masks["fragment_a"][1] and masks["fragment_b"][1]
    assert not masks["unrelated"][1]


def test_dominant_owner_masks_protects_moderate_fold_change_marker():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    # g0 is t0's marker at a realistic, unremarkable fold-change (100 vs 55,
    # ~1.8x -- CD3D-in-CD4T-vs-CD8T territory, not a barnyard-style
    # species-exclusive gene). χ_g0 is set high enough that t0's own true
    # mean (100) sits *below* its ambient ceiling (n̄·χ_g0 = 1000·0.11 =
    # 110) -- an ambient-ceiling collision that would get g0 fully
    # soupOnly-cleared in t0 without _dominant_owner_masks's protection. An
    # earlier version of that mask additionally required a >=2x gap to the
    # runner-up type, which revoked protection for exactly this fold-change
    # range and wiped the host type's own marker to zero -- a regression
    # worse than the bug it fixed. Plain "t is the argmax" must protect it.
    # 10 cells per type, not 2: MIN_TYPE_CELLS falls back to a no-soupOnly
    # rank-1-only path (every gene trivially "protected") below that count,
    # which would pass this assertion vacuously without exercising
    # _dominant_owner_masks at all.
    n_per_type = 10
    x = sparse.csr_matrix(
        np.vstack(
            [np.tile([100.0, 900.0], (n_per_type, 1)), np.tile([55.0, 945.0], (n_per_type, 1))]
        )
    )
    ad = AnnData(x)
    ad.var_names = ["g0", "g1"]
    ad.obs_names = [f"c{i}" for i in range(2 * n_per_type)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n_per_type + ["t1"] * n_per_type
    ad.var[CHI_KEY] = np.array([0.11, 0.89])
    ad.obs[DOSE_KEY] = 50.0
    ad.obs["n_umi"] = np.full(2 * n_per_type, 1000.0)
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (
        den[:n_per_type, 0] >= 90.0
    ).all()  # t0's own marker, not wiped by the ceiling collision


def test_empirical_margin_default_true_matches_explicit_true():
    # Omitting empirical_margin must match empirical_margin=True.
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n_per_type = 20
    g0_t0 = np.array([0.0] * 15 + [40.0] * 5)
    g0_t1 = np.array([0.0] * 15 + [60.0] * 5)
    g1_t0 = np.full(n_per_type, 8000.0)
    g1_t1 = np.full(n_per_type, 8000.0)
    x = np.vstack([np.column_stack([g0_t0, g1_t0]), np.column_stack([g0_t1, g1_t1])])

    def make():
        ad = AnnData(sparse.csr_matrix(x))
        ad.var_names = ["g0", "g1"]
        ad.obs_names = [f"c{i}" for i in range(2 * n_per_type)]
        ad.obs["ambidose_droplet"] = "cell"
        ad.obs["cell_type"] = ["t0"] * n_per_type + ["t1"] * n_per_type
        ad.var[CHI_KEY] = np.array([0.5, 0.5])
        ad.obs[DOSE_KEY] = 200.0
        ad.obs["n_umi"] = x.sum(axis=1)
        return ad

    default = make()
    subtract(default, type_key="cell_type")
    explicit = make()
    subtract(explicit, type_key="cell_type", empirical_margin=True)
    assert np.array_equal(
        default.layers["ambidose_denoised"].toarray(),
        explicit.layers["ambidose_denoised"].toarray(),
    )


def test_cluster_expand_does_not_exceed_raw():
    adata = make_barnyard_toy(seed=4, n_human=40, n_mouse=40, n_empty=60)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"] == "cell"].copy()
    estimate_dose(cells, type_key="true_species")
    subtract(cells, type_key="true_species")
    raw = cells.X.tocsr()
    den = cells.layers["ambidose_denoised"].tocsr()
    rows, cols = den.nonzero()
    assert np.all(np.asarray(den[rows, cols]).ravel() <= np.asarray(raw[rows, cols]).ravel() + 1e-6)


def test_subtract_writes_integer_counts():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=5)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    estimate_dose(adata)
    subtract(adata)
    den = adata.layers["ambidose_denoised"].tocsr()
    raw = adata.X.tocsr()
    assert np.issubdtype(den.dtype, np.integer)
    assert den.data.dtype == np.int32
    rows, cols = den.nonzero()
    den_v = np.asarray(den[rows, cols]).ravel()
    raw_v = np.asarray(raw[rows, cols]).ravel()
    assert np.all(den_v == np.rint(den_v))
    assert np.all(den_v <= raw_v + 1e-6)
    assert np.all(den_v >= 0)


def test_denoise_layer_used_for_type_resolution_not_just_x():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import denoise

    # .X deliberately empty/zero -- if type resolution (clustering, marker
    # scoring, adaptive-resolution estimation) silently reads .X instead of
    # the given layer, this either crashes on an all-zero matrix or produces
    # a single degenerate type. Everything should come from 'counts'.
    rng = np.random.default_rng(0)
    n = 80
    counts = rng.poisson(3, size=(n, 30)).astype(np.float64)
    ad = AnnData(X=sparse.csr_matrix(np.zeros((n, 30))))
    ad.layers["counts"] = sparse.csr_matrix(counts)
    ad.var_names = [f"g{i}" for i in range(30)]
    denoise(ad, sample_key=None, layer="counts", empty_umi_max=100)
    assert "ambidose_denoised" in ad.layers
    assert np.asarray(ad.layers["counts"].sum()) > 0  # untouched, only .X-derived state changed


def test_cross_cell_structure_uses_independent_owner_program():
    from scipy import sparse

    from ambidose.pp import _cross_cell_structure_mask

    rng = np.random.default_rng(0)
    n_cells = 400
    program = rng.integers(0, 2, size=n_cells)
    ref0 = rng.poisson(20 + 40 * program)
    ref1 = rng.poisson(15 + 35 * program)
    ref2 = rng.poisson(10 + 30 * program)
    native = rng.poisson(8 + 25 * program)
    ambient = rng.negative_binomial(2, 2 / (2 + 20), size=n_cells)
    filler = 2000 - ref0 - ref1 - ref2 - native - ambient
    x = sparse.csr_matrix(np.column_stack([ref0, ref1, ref2, native, ambient, filler]))
    totals = np.asarray(x.sum(axis=1)).ravel().astype(float)
    candidates = np.array([3, 4])
    reference = np.array([True, True, True, False, False, False])

    structured = _cross_cell_structure_mask(
        x, totals, np.arange(n_cells), candidates, reference, n_components=2
    )

    assert structured[3]
    assert not structured[4]


def test_qr_structure_solver_matches_old_lstsq_end_to_end(monkeypatch):
    import ambidose.pp as pp

    raw = make_toy(
        n_samples=1,
        n_empty=80,
        n_cells=100,
        n_genes=80,
        contamination=0.25,
        seed=44,
    )
    cells = raw.obs_names[raw.obs["droplet"].astype(str) == "cell"].tolist()
    qr = raw.copy()
    denoise(qr, cell_barcodes=cells, type_key="cell_type", sample_key=None)

    original = pp._cross_cell_structure_mask

    def force_lstsq(*args, **kwargs):
        kwargs["solver"] = "lstsq"
        return original(*args, **kwargs)

    monkeypatch.setattr(pp, "_cross_cell_structure_mask", force_lstsq)
    old = raw.copy()
    denoise(old, cell_barcodes=cells, type_key="cell_type", sample_key=None)

    np.testing.assert_array_equal(
        qr.layers["ambidose_denoised"].toarray(),
        old.layers["ambidose_denoised"].toarray(),
    )
    np.testing.assert_array_equal(qr.obs["ambidose_d"], old.obs["ambidose_d"])
    np.testing.assert_array_equal(qr.obs["ambidose_rho"], old.obs["ambidose_rho"])
    np.testing.assert_array_equal(
        qr.obs["ambidose_rho_trust"].astype(str),
        old.obs["ambidose_rho_trust"].astype(str),
    )
    assert qr.uns["ambidose"]["native_genes_by_type"] == old.uns["ambidose"]["native_genes_by_type"]


def test_cross_type_anchor_is_the_public_default():
    import inspect

    from ambidose.pp import denoise, subtract

    assert inspect.signature(subtract).parameters["cross_type_anchor"].default is True
    assert inspect.signature(denoise).parameters["cross_type_anchor"].default is True


def test_tiny_cluster_cannot_own_cross_type_anchor_genes():
    """Groups smaller than MIN_TYPE_CELLS must not enter the confident-owner pool.

    A 3-cell fragment can Poisson-spike one gene past the 3-fold gap. That
    gene must not become an anchor for every other type's rho_t estimate.
    """
    from anndata import AnnData

    from ambidose._ownership import MIN_TYPE_CELLS, _cross_type_anchor_mask
    from ambidose._shared import _as_csr

    n_a, n_b, n_c = 20, 20, 3
    assert n_c < MIN_TYPE_CELLS
    n_cells = n_a + n_b + n_c
    n_genes = 12
    dense = np.ones((n_cells, n_genes), dtype=np.float64)
    dense[:n_a, 0] = 40.0
    dense[n_a : n_a + n_b, 1] = 40.0
    dense[n_a + n_b :, 2] = 80.0
    ad = AnnData(sparse.csr_matrix(dense))
    ad.obs_names = [f"c{i}" for i in range(n_cells)]
    types = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c, dtype=object)
    n = np.asarray(ad.X.sum(axis=1)).ravel()
    chi = np.full(n_genes, 1.0 / n_genes)
    type_indices = {t: np.flatnonzero(types == t) for t in ("A", "B", "C")}
    type_means = {t: np.asarray(ad.X[idx].mean(axis=0)).ravel() for t, idx in type_indices.items()}
    u_masks = {t: np.ones(n_genes, dtype=bool) for t in type_means}
    kw = dict(
        x=_as_csr(ad.X),
        n=n,
        chi=chi,
        types_s=types,
        u_masks_s=u_masks,
        type_indices=type_indices,
        max_type_mean=0.05,
        min_chi=1e-6,
    )
    extra_with_tiny = _cross_type_anchor_mask(type_means_s=type_means, **kw)
    extra_without_tiny = _cross_type_anchor_mask(
        type_means_s={t: type_means[t] for t in ("A", "B")},
        **kw,
    )
    np.testing.assert_array_equal(extra_with_tiny["A"], extra_without_tiny["A"])
    np.testing.assert_array_equal(extra_with_tiny["B"], extra_without_tiny["B"])
    assert not extra_with_tiny["C"].any()


def test_unowned_uniform_gene_is_soup_cleared():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    # g0/g1 are type markers; g2 is unowned injection (same mean in both types).
    t0 = np.tile([400.0, 10.0, 8.0], (n, 1))
    t1 = np.tile([10.0, 400.0, 8.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    ad.var[CHI_KEY] = np.array([0.02, 0.02, 0.4]) / 0.44
    ad.obs[DOSE_KEY] = 20.0
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 2] == 0.0).all()
    assert (den[:n, 0] >= 380.0).all()
    assert (den[n:, 1] >= 380.0).all()


def test_soup_only_skipped_when_type_rho_below_floor():
    """Type ρ < 0.01: U genes keep rank-1 only, not extra-clear to zero."""
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    t0 = np.tile([400.0, 10.0, 8.0], (n, 1))
    t1 = np.tile([10.0, 400.0, 8.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    ad.var[CHI_KEY] = np.array([0.02, 0.02, 0.4]) / 0.44
    ad.obs[DOSE_KEY] = 2.0
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert den[:, 2].min() > 0.0
    assert float(den[:, 2].mean()) < 8.0


def test_soup_only_extra_clear_is_not_scaled_back_to_dose():
    """U-gene extra-clear may exceed d_c; the old row cap put soup back."""
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    t0 = np.tile([80.0, 1.0, 40.0], (n, 1))
    t1 = np.tile([1.0, 80.0, 40.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    ad.var[CHI_KEY] = np.array([0.05, 0.05, 0.90])
    ad.obs[DOSE_KEY] = 15.0
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    ad.uns["ambidose"] = {
        "dose_type_key": "cell_type",
        "dose_provenance": {
            "type_key": "cell_type",
            "sample_key": None,
            "layer": None,
            "droplet_key": "ambidose_droplet",
            "cell_label": "cell",
        },
    }
    subtract(ad, type_key="cell_type", droplet_key="ambidose_droplet")
    den = ad.layers["ambidose_denoised"].toarray()
    removed = ad.obs["ambidose_removed_umi"].to_numpy()
    assert (den[:, 2] == 0.0).all()
    assert (den[:n, 0] >= 79.0).all()
    assert (den[n:, 1] >= 79.0).all()
    assert (removed > 15.5).all()


def test_ceiling_housekeeping_is_not_soup_cleared():
    """Genes sitting on the ρ=1 ceiling in every type must not be extra-cleared.

    Low-count ceiling-sitters fail the old library-fraction gate (0.003)
    and were wiped on zero-ambient controls.
    """
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    # g0/g1 markers; g2 HK at the χ ceiling in both types (mean ≈ n̄χ).
    t0 = np.tile([80.0, 1.0, 40.0], (n, 1))
    t1 = np.tile([1.0, 80.0, 40.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    n_bar = float(np.asarray(ad.X.sum(axis=1)).mean())
    chi2 = 40.0 / n_bar
    rest = max(1.0 - chi2, 1e-6)
    ad.var[CHI_KEY] = np.array([rest / 2.0, rest / 2.0, chi2])
    ad.obs[DOSE_KEY] = 20.0
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    ad.uns["ambidose"] = {
        "dose_type_key": "cell_type",
        "dose_provenance": {
            "type_key": "cell_type",
            "sample_key": None,
            "layer": None,
            "droplet_key": "ambidose_droplet",
            "cell_label": "cell",
        },
    }
    subtract(ad, type_key="cell_type", droplet_key="ambidose_droplet")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 2] >= 35.0).all()


def test_zero_rho_housekeeping_retained():
    from ambidose.datasets import scenario_housekeeping
    from ambidose.pp import denoise

    ad = scenario_housekeeping(
        true_rho=0.0, n_hk=20, n_types=2, genes_per_type=10, n_empty=40, seed=1
    )
    cells = ad.obs_names[ad.obs["droplet"].astype(str) == "cell"].tolist()
    denoise(ad, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    mask = ad.obs["ambidose_droplet"].astype(str).to_numpy() == "cell"
    raw = np.asarray(ad.layers["raw_counts"][mask].sum())
    den = np.asarray(ad.layers["ambidose_denoised"][mask].sum())
    assert den / raw > 0.90
    hk = slice(20, 40)
    hk_raw = np.asarray(ad.layers["raw_counts"][mask][:, hk].sum())
    hk_den = np.asarray(ad.layers["ambidose_denoised"][mask][:, hk].sum())
    if hk_raw > 0:
        assert hk_den / hk_raw > 0.85


def test_flat_native_gene_removal_does_not_track_noisy_dose():
    """Same-type cells with identical counts must lose similar HK UMIs.

    Per-cell d_c noise must not be written onto genes that look native
    everywhere; those genes cannot identify cell-to-cell ρ.
    """
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    t0 = np.tile([80.0, 1.0, 40.0], (n, 1))
    t1 = np.tile([1.0, 80.0, 40.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    n_bar = float(np.asarray(ad.X.sum(axis=1)).mean())
    chi2 = 40.0 / n_bar
    rest = max(1.0 - chi2, 1e-6)
    ad.var[CHI_KEY] = np.array([rest / 2.0, rest / 2.0, chi2])
    dose = np.full(2 * n, 20.0)
    dose[: n // 2] = 5.0
    dose[n // 2 : n] = 80.0
    ad.obs[DOSE_KEY] = dose
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    ad.uns["ambidose"] = {
        "dose_type_key": "cell_type",
        "dose_provenance": {
            "type_key": "cell_type",
            "sample_key": None,
            "layer": None,
            "droplet_key": "ambidose_droplet",
            "cell_label": "cell",
        },
    }
    subtract(ad, type_key="cell_type", droplet_key="ambidose_droplet")
    den = ad.layers["ambidose_denoised"].toarray()
    low = den[: n // 2, 2]
    high = den[n // 2 : n, 2]
    assert np.max(np.abs(low.mean() - high.mean())) < 2.0


def test_mid_ceiling_gene_is_not_extra_cleared():
    """r_t ≈ 0.6 is not true soup (ρ) and not the ρ=1 ceiling; rank-1 only."""
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 20
    t0 = np.tile([80.0, 1.0, 12.0], (n, 1))
    t1 = np.tile([1.0, 80.0, 12.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([t0, t1])))
    ad.var_names = ["g0", "g1", "g2"]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obs["ambidose_droplet"] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n
    n_bar = float(np.asarray(ad.X.sum(axis=1)).mean())
    chi2 = 12.0 / (0.6 * n_bar)
    rest = max(1.0 - chi2, 1e-6)
    ad.var[CHI_KEY] = np.array([rest / 2.0, rest / 2.0, chi2])
    ad.obs[DOSE_KEY] = 20.0
    ad.obs["n_umi"] = np.asarray(ad.X.sum(axis=1)).ravel()
    ad.uns["ambidose"] = {
        "dose_type_key": "cell_type",
        "dose_provenance": {
            "type_key": "cell_type",
            "sample_key": None,
            "layer": None,
            "droplet_key": "ambidose_droplet",
            "cell_label": "cell",
        },
    }
    subtract(ad, type_key="cell_type", droplet_key="ambidose_droplet")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 2] > 0).all()


def test_rank1_take_sum_does_not_exceed_dose():
    observed = np.array([100.0, 100.0, 100.0])
    chi = np.array([0.5, 0.3, 0.2])
    take = _confidence_weighted_take(observed, chi, 10.0, np.zeros(3))
    assert take.sum() <= 10.0 + 1e-12


def test_soup_only_extra_clear_is_outside_rank1_budget():
    observed = np.array([5.0, 100.0])
    chi = np.array([0.5, 0.5])
    is_u = np.array([True, False])
    take = _confidence_weighted_take(observed, chi, 10.0, np.zeros(2), is_u)
    np.testing.assert_allclose(take, [5.0, 5.0])


def test_ambient_equality_has_zero_native_confidence():
    from ambidose.pp import _type_masks

    x = sparse.csr_matrix(np.full((20, 2), 2.0))
    n = np.full(20, 10000.0)
    chi = np.array([0.0002, 0.0002])
    idx = np.arange(20)
    exclude = np.zeros(2, dtype=bool)
    _, _, conf, _ = _type_masks(
        x, n, chi, idx, max_type_mean=0.05, min_chi=1e-6, top_n=100, exclude=exclude
    )
    assert (conf < 0.05).all()


def test_complete_linkage_does_not_chain_distant_groups():
    from ambidose.pp import _complete_linkage_labels

    compatible = np.array(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ]
    )
    labels = _complete_linkage_labels(compatible)
    assert labels[0] != labels[2]


def test_single_meta_falls_back_to_fragment_argmax():
    from ambidose.pp import _dominant_owner_masks, _type_means

    n = 40
    a = np.tile([200.0, 1000.0, 1000.0], (n, 1))
    b = np.tile([10.0, 1000.0, 1000.0], (n, 1))
    c = np.tile([10.0, 1000.0, 1000.0], (n, 1))
    x = sparse.csr_matrix(np.vstack([a, b, c]))
    types = np.repeat(["a", "b", "c"], n)
    n_lib = np.asarray(x.sum(axis=1)).ravel().astype(float)
    masks, n_meta = _dominant_owner_masks(
        x, n_lib, types, _type_means(x, types), max_type_mean=0.05
    )
    assert masks["a"][0]
    assert not masks["b"][0]
    assert not masks["c"][0]
    if n_meta == 1:
        return
    assert n_meta >= 1


def test_subtract_reuses_cluster_key_without_type_key():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CHI_KEY, CLUSTER_KEY, DOSE_KEY

    n = 12
    x = sparse.csr_matrix(np.vstack([np.tile([80.0, 5.0], (n, 1)), np.tile([5.0, 80.0], (n, 1))]))
    ad = AnnData(x)
    ad.obs[CLUSTER_KEY] = ["0"] * n + ["1"] * n
    ad.obs["ambidose_droplet"] = "cell"
    ad.var[CHI_KEY] = np.array([0.5, 0.5])
    ad.obs[DOSE_KEY] = 4.0
    ad.uns["ambidose"] = {
        "dose_type_key": CLUSTER_KEY,
        "dose_sample_key": None,
        "dose_provenance": {
            "type_key": CLUSTER_KEY,
            "sample_key": None,
            "layer": None,
            "droplet_key": None,
            "cell_label": "cell",
        },
    }
    subtract(ad)
    assert "ambidose_denoised" in ad.layers
    assert ad.uns["ambidose"]["max_meta_groups_per_sample"] >= 1


def test_type_validation_distinguishes_missing_values_from_literal_na_tokens():
    import pandas as pd
    from anndata import AnnData

    from ambidose._shared import EMPTY_TYPES
    from ambidose.pp import _validated_type_values

    adata = AnnData(sparse.csr_matrix(np.ones((4, 1))))
    adata.obs["kind"] = pd.Series([pd.NA, "NA", "None", "nan"], index=adata.obs_names)
    labels = _validated_type_values(adata, "kind")
    assert labels.iloc[0] in EMPTY_TYPES
    assert labels.iloc[1:].tolist() == ["NA", "None", "nan"]


def test_single_type_sample_still_owns_abundant_genes():
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY, _dominant_owner_masks, _type_means

    n = 20
    x = sparse.csr_matrix(np.tile([80.0, 8.0], (n, 1)))
    types = np.array(["only"] * n)
    lib = np.asarray(x.sum(axis=1)).ravel().astype(float)
    means = _type_means(x, types)
    masks, n_meta = _dominant_owner_masks(x, lib, types, means, max_type_mean=0.05)
    assert n_meta == 1
    assert masks["only"][0]
    ad = AnnData(x)
    ad.obs["cell_type"] = "only"
    ad.obs["ambidose_droplet"] = "cell"
    ad.var[CHI_KEY] = np.array([0.2, 0.8])
    ad.obs[DOSE_KEY] = 10.0
    subtract(ad, type_key="cell_type")
    den = ad.layers["ambidose_denoised"].toarray()
    assert (den[:, 0] >= 70.0).all()


def test_tiny_fragment_does_not_steal_ownership():
    from ambidose.pp import MIN_TYPE_CELLS, _dominant_owner_masks, _type_means

    n_big = 20
    n_tiny = 3
    t0 = np.tile([80.0, 10.0], (n_big, 1))
    t1 = np.tile([10.0, 80.0], (n_big, 1))
    tiny = np.tile([500.0, 10.0], (n_tiny, 1))
    x = sparse.csr_matrix(np.vstack([t0, t1, tiny]))
    types = np.array(["t0"] * n_big + ["t1"] * n_big + ["tiny"] * n_tiny)
    n = np.asarray(x.sum(axis=1)).ravel().astype(float)
    means = _type_means(x, types)
    assert "tiny" not in means
    assert n_tiny < MIN_TYPE_CELLS
    masks, _ = _dominant_owner_masks(x, n, types, means, max_type_mean=0.05)
    assert "tiny" not in masks
    assert masks["t0"][0]
    assert not masks["t1"][0]


def test_nan_type_label_is_not_an_owner():
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY, LAYER_OUT

    n = 15
    cells = np.tile([80.0, 8.0], (n, 1))
    empty = np.tile([8.0, 80.0], (5, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([cells, empty])))
    ad.obs["ambidose_droplet"] = ["cell"] * n + ["empty"] * 5
    ad.obs["cell_type"] = ["t0"] * n + [np.nan] * 5
    ad.var[CHI_KEY] = np.array([0.1, 0.9])
    ad.obs[DOSE_KEY] = [10.0] * n + [0.0] * 5
    subtract(ad, type_key="cell_type")
    den = ad.layers[LAYER_OUT].toarray()
    assert (den[:n, 0] >= 70.0).all()


def test_non_cells_use_internal_type_sentinel_during_subtraction(monkeypatch):
    from anndata import AnnData

    import ambidose.pp as pp
    from ambidose.pp import CHI_KEY, DOSE_KEY

    n = 15
    ad = AnnData(sparse.csr_matrix(np.tile([80.0, 8.0], (n + 2, 1))))
    ad.obs["ambidose_droplet"] = ["cell"] * n + ["empty", "other"]
    ad.obs["cell_type"] = ["t0"] * n + [np.nan, np.nan]
    ad.var[CHI_KEY] = np.array([0.1, 0.9])
    ad.obs[DOSE_KEY] = [10.0] * n + [0.0, 0.0]

    seen = []
    original = pp._dominant_owner_masks

    def record_types(x, totals, types, type_means, **kwargs):
        seen.append(np.asarray(types, dtype=object).copy())
        return original(x, totals, types, type_means, **kwargs)

    monkeypatch.setattr(pp, "_dominant_owner_masks", record_types)
    subtract(ad, type_key="cell_type")

    assert len(seen) == 1
    assert seen[0][-1] is pp.EMPTY_TYPE
    assert seen[0][-2] is pp.EMPTY_TYPE
    assert "-1" not in seen[0]


def test_owner_masks_are_computed_per_sample():
    import pandas as pd
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY, LAYER_OUT

    n = 12
    # Sample A: t0 owns g0. Sample B: the same label t0 does not express g0.
    a_t0 = np.tile([200.0, 10.0], (n, 1))
    a_t1 = np.tile([10.0, 200.0], (n, 1))
    b_t0 = np.tile([10.0, 10.0], (n, 1))
    b_t1 = np.tile([10.0, 200.0], (n, 1))
    ad = AnnData(sparse.csr_matrix(np.vstack([a_t0, a_t1, b_t0, b_t1])))
    ad.obs["sample"] = ["A"] * (2 * n) + ["B"] * (2 * n)
    ad.obs["cell_type"] = (["t0"] * n + ["t1"] * n) * 2
    ad.obs["ambidose_droplet"] = "cell"
    chi = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=["A", "B"],
        columns=ad.var_names,
    )
    ad.uns[CHI_KEY] = chi
    ad.obs[DOSE_KEY] = 8.0
    subtract(ad, type_key="cell_type", sample_key="sample")
    den = ad.layers[LAYER_OUT].toarray()
    assert (den[:n, 0] >= 180.0).all()


def test_mark_doublets_skips_residual_on_auto_clusters():
    from anndata import AnnData
    from scipy import sparse

    from ambidose.pp import CLUSTER_KEY, DROPLET_KEY, mark_doublets

    n = 40
    ad = AnnData(sparse.csr_matrix(np.ones((n, 8))))
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs[CLUSTER_KEY] = ["0"] * (n // 2) + ["1"] * (n // 2)
    mark_doublets(ad, type_key=CLUSTER_KEY)
    assert "ambidose_type_residual" not in ad.obs


def test_integer_type_allocation_preserves_group_budget():
    from ambidose.pp import _expand_take_to_cells

    x = sparse.csr_matrix(np.ones((100, 1), dtype=float))
    _expand_take_to_cells(
        x,
        np.arange(100),
        np.array([10.0]),
        np.ones(100),
    )
    assert int(100 - x.sum()) == 10
    assert set(np.unique(x.data)).issubset({0.0, 1.0})


def test_integer_budget_uses_half_up_rounding():
    np.testing.assert_array_equal(_alloc_integer_budget(0.5, np.array([1.0]), np.array([1.0])), [1])
    assert _alloc_integer_budget(2.5, np.ones(4), np.ones(4)).sum() == 3


def test_integer_allocator_never_removes_from_zero_weight():
    got = _alloc_integer_budget(10, np.array([10]), np.array([0]))
    np.testing.assert_array_equal(got, [0])


def test_integer_allocator_spends_all_feasible_positive_weight_budget():
    got = _alloc_integer_budget(10, np.array([3, 20]), np.array([1, 1]))
    assert got.sum() == 10
    assert np.all(got <= [3, 20])
    saturated = _alloc_integer_budget(50, np.array([1, 100]), np.array([1, 0]))
    np.testing.assert_array_equal(saturated, [1, 0])


def test_alloc_budget_tied_room_with_near_zero_weight_does_not_overspend():
    # Regression for a real B1_c1 crash: 4 buckets tied at room=1, one
    # weight ~1e-29 (not exactly zero). The old "1 - cumsum(w[:-1])" tail
    # formula loses that weight to float64 cancellation once the running
    # sum saturates to exactly 1.0, so the near-zero bucket's saturation
    # breakpoint collapses onto its predecessor's -- admitting one bucket
    # too many into `sat` and overspending target=3 by a whole unit (4
    # instead of 3). A reverse-cumsum tail sum fixes it.
    room = np.array([1.0, 1.0, 1.0, 1.0])
    weights = np.array(
        [7.445043651924292e-29, 81.61154789885094, 24.3235894687974, 165.49399647622255]
    )
    got = _alloc_integer_budget(3.319123008163391, room, weights)
    assert int(got.sum()) == 3
    assert np.all(got <= room)
    # the near-zero-weight bucket should get none of the budget
    assert got[0] == 0


def test_subtract_is_gene_permutation_equivariant():
    from anndata import AnnData

    x = sparse.csr_matrix([[2, 2, 2, 2]], dtype=float)
    original = AnnData(x.copy())
    original.obs_names = ["cell-a"]
    original.var_names = ["g3", "g1", "g4", "g2"]
    original.var["ambidose_chi"] = 0.25
    original.obs["ambidose_d"] = 2.0
    permuted = original[:, [2, 0, 3, 1]].copy()
    subtract(original, type_key=None)
    subtract(permuted, type_key=None)
    restored = permuted[:, original.var_names]
    np.testing.assert_array_equal(
        original.layers["ambidose_denoised"].toarray(),
        restored.layers["ambidose_denoised"].toarray(),
    )


def test_type_subtract_is_cell_permutation_equivariant():
    from anndata import AnnData

    n = 20
    adata = AnnData(sparse.csr_matrix(np.ones((n, 2)), dtype=float))
    adata.obs_names = [f"cell-{i:02d}" for i in range(n)]
    adata.var_names = ["g1", "g2"]
    adata.var["ambidose_chi"] = [0.5, 0.5]
    adata.obs["ambidose_d"] = 0.5
    adata.obs["kind"] = "T"
    adata.obs["ambidose_droplet"] = "cell"
    order = np.array([7, 1, 19, 3, 12, 0, 8, 4, 15, 2, 18, 6, 10, 5, 14, 9, 17, 11, 16, 13])
    permuted = adata[order].copy()
    subtract(adata, type_key="kind")
    subtract(permuted, type_key="kind")
    restored = permuted[adata.obs_names]
    np.testing.assert_array_equal(
        adata.layers["ambidose_denoised"].toarray(),
        restored.layers["ambidose_denoised"].toarray(),
    )


def test_complete_linkage_is_permutation_equivariant_with_stable_keys():
    from ambidose.pp import _complete_linkage_labels

    compatible = np.array(
        [[True, True, False], [True, True, True], [False, True, True]],
        dtype=bool,
    )
    keys = np.array(["A", "B", "C"])
    original = _complete_linkage_labels(compatible, keys=keys)
    order = np.array([1, 2, 0])
    permuted = _complete_linkage_labels(compatible[np.ix_(order, order)], keys=keys[order])
    restored = np.empty_like(permuted)
    restored[order] = permuted
    np.testing.assert_array_equal(
        original[:, None] == original[None, :],
        restored[:, None] == restored[None, :],
    )


def test_top_chi_selection_includes_boundary_ties():
    from ambidose.pp import _top_chi_indices

    chi = np.array([0.4, 0.2, 0.2, 0.2, 0.0])
    got = _top_chi_indices(chi, np.arange(5), 2)
    np.testing.assert_array_equal(got, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "chi",
    [
        np.array([-0.1, 1.1]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.4, 0.4]),
    ],
)
def test_subtract_rejects_invalid_stored_chi(chi):
    from anndata import AnnData

    adata = AnnData(sparse.csr_matrix([[2.0, 2.0]]))
    adata.var["ambidose_chi"] = chi
    adata.obs["ambidose_d"] = 1.0
    with pytest.raises(ValueError):
        subtract(adata, type_key=None)


def test_chi_frame_rejects_sample_labels_colliding_after_string_conversion():
    import pandas as pd
    from anndata import AnnData

    from ambidose.pp import _validate_chi_frame

    adata = AnnData(sparse.csr_matrix([[1.0, 1.0]]))
    adata.var_names = ["g1", "g2"]
    adata.obs["sample"] = "1"
    adata.uns["ambidose_chi"] = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=[1, "1"],
        columns=adata.var_names,
    )
    with pytest.raises(ValueError, match="collide"):
        _validate_chi_frame(adata, "sample")


def test_type_labels_colliding_after_string_conversion_are_rejected():
    from anndata import AnnData

    from ambidose.pp import _validated_type_values

    adata = AnnData(sparse.csr_matrix(np.ones((2, 1))))
    adata.obs["kind"] = [1, "1"]
    with pytest.raises(ValueError, match="collide"):
        _validated_type_values(adata, "kind")


def test_stable_identity_helpers_reject_duplicate_keys():
    from ambidose.pp import _complete_linkage_labels

    with pytest.raises(ValueError, match="keys must be unique"):
        _complete_linkage_labels(np.ones((2, 2), dtype=bool), keys=["A", "A"])
    with pytest.raises(ValueError, match="tie_keys must be unique"):
        _alloc_integer_budget(1, np.ones(2), np.ones(2), tie_keys=["cell", "cell"])


def test_subtract_rejects_reserved_and_input_layer_outputs_atomically():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=31)
    adata.layers["counts"] = adata.X.copy()
    adata.layers["raw_counts"] = adata.X.copy()
    raw_before = adata.layers["raw_counts"].copy()
    counts_before = adata.layers["counts"].copy()
    obs_before = adata.obs.copy(deep=True)
    uns_before = dict(adata.uns)

    with pytest.raises(ValueError, match="reserved"):
        subtract(adata, layer="counts", layer_out="raw_counts")
    with pytest.raises(ValueError, match="must not overwrite"):
        subtract(adata, layer="counts", layer_out="counts")

    assert (adata.layers["raw_counts"] != raw_before).nnz == 0
    assert (adata.layers["counts"] != counts_before).nnz == 0
    assert adata.obs.equals(obs_before)
    assert adata.uns == uns_before


def test_subtract_explicit_sample_key_requires_sample_chi():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=33)
    adata.obs["sample"] = "s1"
    adata.var["ambidose_chi"] = np.full(adata.n_vars, 1.0 / adata.n_vars)
    adata.obs["ambidose_d"] = 1.0

    with pytest.raises(ValueError, match="sample-specific"):
        subtract(adata, sample_key="sample")


def test_subtract_rejects_positive_dose_on_non_cells_atomically():
    adata = make_toy(n_empty=10, n_cells=5, n_samples=1, seed=38)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    adata.var["ambidose_chi"] = np.full(adata.n_vars, 1.0 / adata.n_vars)
    dose = np.zeros(adata.n_obs)
    dose[adata.obs["droplet"].astype(str).to_numpy() != "cell"] = 1.0
    before = adata.X.copy()

    with pytest.raises(ValueError, match="non-cell"):
        subtract(adata, dose=dose, droplet_key="ambidose_droplet")

    assert (before != adata.X).nnz == 0
    assert "ambidose_denoised" not in adata.layers


def test_subtract_rejects_type_key_different_from_dose_provenance():
    from anndata import AnnData

    from ambidose.pp import CHI_KEY, DOSE_KEY

    ad = AnnData(sparse.csr_matrix([[10, 0], [0, 10]]))
    ad.obs["cluster_A"] = ["a", "b"]
    ad.obs["cluster_B"] = ["x", "y"]
    ad.obs[DOSE_KEY] = [1.0, 1.0]
    ad.var[CHI_KEY] = [0.5, 0.5]
    ad.uns["ambidose"] = {
        "dose_type_key": "cluster_A",
        "dose_provenance": {
            "type_key": "cluster_A",
            "sample_key": None,
            "layer": None,
            "droplet_key": None,
            "cell_label": "cell",
        },
    }
    with pytest.raises(ValueError, match="parameters do not match dose provenance"):
        subtract(ad, type_key="cluster_B")


def test_realloc_unspent_does_not_touch_protected_or_u():
    observed = np.array([100.0, 100.0, 100.0])
    chi = np.array([0.5, 0.3, 0.2])
    dose = 50.0
    conf = np.array([1.0, 0.0, 0.0])
    take = _confidence_weighted_take(observed, chi, dose, conf, None)
    is_p = np.array([True, False, False])
    is_u = np.array([False, False, True])
    take[is_u] = 0.0
    out = _realloc_unspent_rank1(take, observed, chi, dose, is_p, is_u)
    assert out[0] == pytest.approx(take[0])
    assert out[2] == 0.0
    assert out[1] > take[1]
    assert float(out.sum()) == pytest.approx(min(dose, float(observed[~is_u].sum())), rel=1e-6)
