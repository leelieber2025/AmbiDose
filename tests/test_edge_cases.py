"""Synthetic edges that instantiate a known failure mode. Soup is injected
on top of native counts, not baked into the native template.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from ambidose.datasets import make_toy
from ambidose.pp import (
    CHI_KEY,
    DOSE_KEY,
    DROPLET_KEY,
    LAYER_OUT,
    MIN_TYPE_CELLS,
    classify_droplets,
    denoise,
    estimate_chi,
    estimate_dose_mixture,
    subtract,
)


def _csr(values):
    return sparse.csr_matrix(np.asarray(values, dtype=np.float64))


def test_chi_gene_mismatch_mixed_extra_and_n_missing_raises():
    # Mixed extra+dropped: 1 chi-only gene (n_dropped=1/10 <= 20%) plus 3
    # object-only genes (n_missing=3/10 > 20%). n_dropped is checked first
    # in _chi_for_obs, so extras must stay under the cutoff or this never
    # reaches the n_missing / extra-object-genes direction.
    ad = AnnData(_csr([[10] * 12]))
    ad.var_names = [f"g{i}" for i in range(9)] + [f"obj{i}" for i in range(3)]
    ad.obs["sample"] = "s0"
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs[DOSE_KEY] = 5.0
    ad.uns[CHI_KEY] = pd.DataFrame(
        [np.full(10, 0.1)],
        index=["s0"],
        columns=[f"g{i}" for i in range(9)] + ["extra0"],
    )
    with pytest.raises(ValueError, match="gene labels must exactly match"):
        subtract(ad, sample_key="sample")


def test_chi_gene_mismatch_under_20_percent_is_still_rejected():
    ad = AnnData(_csr([[10] * 9]))
    ad.var_names = [f"g{i}" for i in range(9)]
    ad.obs["sample"] = "s0"
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs[DOSE_KEY] = 5.0
    # 10-gene χ, 9 names overlap (this object's full gene set) → 1/10 of
    # χ's genes get dropped by reindex, under the 20% hard-stop.
    cols = [f"g{i}" for i in range(9)] + ["extra0"]
    row = np.full(10, 0.1)
    ad.uns[CHI_KEY] = pd.DataFrame([row], index=["s0"], columns=cols)
    with pytest.raises(ValueError, match="gene labels must exactly match"):
        subtract(ad, sample_key="sample")


def test_chi_gene_mismatch_over_20_percent_raises():
    # Inverse of the above: dropping half of chi's genes (not merely a
    # handful of naming stragglers) must hard-stop, not silently
    # renormalize -- this is the direction that was previously unchecked
    # (reindex drops unmatched chi columns without producing NaN, so a
    # NaN-only count never saw it).
    ad = AnnData(_csr([[10, 10, 10, 10, 10]]))
    ad.var_names = [f"g{i}" for i in range(5)]
    ad.obs["sample"] = "s0"
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs[DOSE_KEY] = 5.0
    cols = [f"g{i}" for i in range(5)] + [f"extra{i}" for i in range(5)]
    row = np.full(10, 0.1)
    ad.uns[CHI_KEY] = pd.DataFrame([row], index=["s0"], columns=cols)
    with pytest.raises(ValueError, match="gene labels must exactly match"):
        subtract(ad, sample_key="sample")


def test_multi_sample_empty_zero_umi_names_the_sample():
    adata = make_toy(n_samples=2, n_empty=40, n_cells=20, seed=21)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    s1_empty = (adata.obs["sample"].astype(str) == "s1") & (
        adata.obs[DROPLET_KEY].astype(str) == "empty"
    )
    x = adata.X.tolil()
    x[np.flatnonzero(s1_empty), :] = 0
    adata.X = x.tocsr()
    with pytest.raises(ValueError, match="sample 's1'"):
        estimate_chi(adata, sample_key="sample")


def test_subtract_rejects_int32_overflow():
    ad = AnnData(_csr([[3e9, 1.0]]))
    ad.var[CHI_KEY] = [0.5, 0.5]
    ad.obs[DOSE_KEY] = [0.0]
    ad.obs[DROPLET_KEY] = "cell"
    with pytest.raises(ValueError, match="int32"):
        subtract(ad)


def test_multi_sample_missing_empties_raises_for_that_sample():
    adata = make_toy(n_samples=2, n_empty=40, n_cells=20, seed=21)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    # Wipe empties in s1 only.
    s1_empty = (adata.obs["sample"].astype(str) == "s1") & (
        adata.obs[DROPLET_KEY].astype(str) == "empty"
    )
    adata.obs.loc[s1_empty, DROPLET_KEY] = "other"
    with pytest.raises(ValueError, match="sample 's1'"):
        estimate_chi(adata, sample_key="sample")


def test_tiny_type_does_not_crash_or_inflate():
    n_big, n_tiny = 20, MIN_TYPE_CELLS - 1
    native = np.vstack([np.tile([80.0, 5.0], (n_big, 1)), np.tile([5.0, 80.0], (n_tiny, 1))])
    ad = AnnData(_csr(native))
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs["cell_type"] = ["t0"] * n_big + ["t1"] * n_tiny
    ad.var[CHI_KEY] = [0.5, 0.5]
    ad.obs[DOSE_KEY] = 8.0
    subtract(ad, type_key="cell_type")
    raw = ad.X.tocsr()
    den = ad.layers[LAYER_OUT].tocsr()
    assert den.nnz == 0 or np.isfinite(den.data).all()
    rows, cols = den.nonzero()
    if rows.size:
        assert np.all(
            np.asarray(den[rows, cols]).ravel() <= np.asarray(raw[rows, cols]).ravel() + 1e-6
        )


def test_dose_above_n_umi_is_rejected():
    ad = AnnData(_csr([[10.0, 10.0]]))
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs[DOSE_KEY] = 100.0
    ad.var[CHI_KEY] = [0.5, 0.5]
    with pytest.raises(ValueError, match="dose exceeds n_umi"):
        subtract(ad)


def test_classify_matches_trailing_minus_one_suffix():
    ad = AnnData(_csr([[200.0, 0.0], [40.0, 0.0]]))
    ad.obs_names = ["AAACCTG-1", "EMPTYAAA-1"]
    classify_droplets(ad, cell_barcodes=["AAACCTG"], empty_umi_max=100)
    lab = ad.obs[DROPLET_KEY].astype(str)
    assert lab["AAACCTG-1"] == "cell"
    assert lab["EMPTYAAA-1"] == "empty"


def test_classify_rejects_unmatched_whitelist():
    ad = AnnData(_csr([[200.0]]))
    ad.obs_names = ["CELL"]
    with pytest.raises(ValueError, match="0/1 cell_barcodes matched"):
        classify_droplets(ad, cell_barcodes=["TOTALLY_OTHER"])


def test_two_type_unbalanced_library_still_uses_empty_profile():
    n0, n1 = 40, 2
    values = np.vstack([np.tile([90.0, 10.0], (n0, 1)), np.tile([10.0, 90.0], (n1, 1))])
    ad = AnnData(_csr(values))
    ad.obs["cell_type"] = ["A"] * n0 + ["B"] * n1
    ad.obs[DROPLET_KEY] = "cell"
    ad.var[CHI_KEY] = [0.5, 0.5]
    estimate_dose_mixture(ad, type_key="cell_type")
    assert set(ad.obs["ambidose_mixture_profile"].astype(str)) == {"empty"}


def test_chi_concentrated_on_one_gene_does_not_nan():
    ad = AnnData(_csr(np.tile([100.0, 5.0, 5.0], (15, 1))))
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs["cell_type"] = "t0"
    ad.var[CHI_KEY] = [0.98, 0.01, 0.01]
    ad.obs[DOSE_KEY] = 10.0
    subtract(ad, type_key="cell_type")
    den = ad.layers[LAYER_OUT]
    assert np.isfinite(den.data).all() if den.nnz else True
    assert (np.asarray(den.sum(axis=1)).ravel() <= 110.0 + 1e-6).all()


def test_expected_cells_at_least_leaves_one_empty_slot():
    ad = make_toy(n_samples=1, n_empty=5, n_cells=10, seed=3)
    classify_droplets(ad, expected_cells=ad.n_obs)
    lab = ad.obs[DROPLET_KEY].astype(str)
    assert (lab == "cell").sum() == ad.n_obs - 1
    assert (lab == "empty").sum() >= 1


def test_expected_cells_on_single_barcode_raises():
    from anndata import AnnData

    ad = AnnData(sparse.csr_matrix([[3, 0]], dtype=np.int32))
    with pytest.raises(ValueError, match="fewer than two barcodes"):
        classify_droplets(ad, expected_cells=1)


def test_automatic_typing_on_one_gene_uses_single_cluster():
    from anndata import AnnData

    from ambidose.pp import CLUSTER_KEY, denoise

    x = sparse.csr_matrix(np.r_[np.ones(20), np.full(40, 10)][:, None], dtype=np.int32)
    ad = AnnData(x)
    ad.obs_names = [f"b{i}" for i in range(ad.n_obs)]
    cells = ad.obs_names[20:].tolist()
    denoise(ad, cell_barcodes=cells, cell_calling="off", sample_key=None)
    assert set(ad.obs.loc[cells, CLUSTER_KEY].astype(str)) == {"0"}
    assert "ambidose_denoised" in ad.layers


def test_high_contamination_leaves_native_block():
    adata = make_toy(
        n_samples=1,
        n_cells=40,
        n_empty=80,
        contamination=0.6,
        seed=22,
    )
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    is_cell = adata.obs[DROPLET_KEY].astype(str).to_numpy() == "cell"
    den = adata.layers[LAYER_OUT][is_cell]
    raw = adata.layers["raw_counts"][is_cell]
    assert float(den.sum()) > 0.2 * float(raw.sum())
    rho = adata.obs.loc[is_cell, "ambidose_rho"].to_numpy(dtype=float)
    assert np.isfinite(rho).all()
    assert float(np.median(rho)) < 0.95


def test_require_ram_raises_before_allocation(monkeypatch):
    from ambidose.pp import _require_ram

    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 1024)
    with pytest.raises(MemoryError, match="needs 1.0 GiB free RAM"):
        _require_ram(1024**3, where="cross-type structure protection")


def test_structure_mask_chunks_match():
    from scipy import sparse as sp

    from ambidose.pp import _cross_cell_structure_mask

    rng = np.random.default_rng(0)
    n_cells, n_genes = 90, 50
    x = sp.csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float64))
    n = np.asarray(x.sum(axis=1)).ravel() + 1.0
    idx = np.arange(n_cells)
    cand = np.arange(12, 40)
    ref = np.zeros(n_genes, dtype=bool)
    ref[:20] = True
    a = _cross_cell_structure_mask(x, n, idx, cand, ref, chunk_size=5)
    b = _cross_cell_structure_mask(x, n, idx, cand, ref, chunk_size=40)
    np.testing.assert_array_equal(a, b)


def test_structure_mask_qr_matches_lstsq_decisions():
    from scipy import sparse as sp

    from ambidose.pp import _cross_cell_structure_mask

    rng = np.random.default_rng(14)
    n_cells, n_genes = 240, 90
    x = sp.csr_matrix(rng.poisson(1.5, size=(n_cells, n_genes)).astype(np.float64))
    n = np.asarray(x.sum(axis=1)).ravel() + 1.0
    idx = np.arange(n_cells)
    cand = np.arange(25, 80)
    ref = np.zeros(n_genes, dtype=bool)
    ref[:35] = True

    qr = _cross_cell_structure_mask(x, n, idx, cand, ref, chunk_size=7, solver="qr")
    old = _cross_cell_structure_mask(x, n, idx, cand, ref, chunk_size=7, solver="lstsq")

    np.testing.assert_array_equal(qr, old)


def test_structure_mask_one_and_four_threads_are_identical():
    from scipy import sparse as sp

    from ambidose.pp import _cross_cell_structure_mask

    rng = np.random.default_rng(19)
    x = sp.csr_matrix(rng.poisson(1.2, size=(500, 140)).astype(np.float64))
    n = np.asarray(x.sum(axis=1)).ravel() + 1.0
    idx = np.arange(x.shape[0])
    cand = np.arange(30, 130)
    ref = np.zeros(x.shape[1], dtype=bool)
    ref[:45] = True

    serial = _cross_cell_structure_mask(x, n, idx, cand, ref, n_jobs=1)
    parallel = _cross_cell_structure_mask(x, n, idx, cand, ref, n_jobs=4)

    np.testing.assert_array_equal(serial, parallel)


def test_thread_workers_are_bounded_by_cpu_memory_and_tasks(monkeypatch):
    from ambidose.pp import _thread_worker_count

    monkeypatch.setattr("ambidose._shared._usable_cpu_count", lambda: 16)
    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 12 * (1 << 30))
    assert _thread_worker_count(None, n_tasks=10, per_worker_bytes=1 << 30) == 4
    assert _thread_worker_count(2, n_tasks=10, per_worker_bytes=1 << 30) == 2
    assert _thread_worker_count(8, n_tasks=1, per_worker_bytes=1) == 1


def test_require_ram_allows_fit(monkeypatch):
    from ambidose.pp import _require_ram

    monkeypatch.setattr("ambidose._shared._available_ram_bytes", lambda: 2 * 1024**3)
    _require_ram(1024**3, where="cross-type structure protection")


def test_empties_with_zero_umi_raise_on_chi():
    ad = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=4)
    ad.obs[DROPLET_KEY] = ad.obs["droplet"].astype(str)
    empty = ad.obs[DROPLET_KEY].to_numpy() == "empty"
    x = ad.X.tolil()
    for i in np.flatnonzero(empty):
        x[i, :] = 0
    ad.X = x.tocsr()
    with pytest.raises(ValueError, match="zero total UMI"):
        estimate_chi(ad, sample_key=None)


def test_other_label_has_no_special_protection():
    n = 20
    native = np.vstack(
        [
            np.tile([80.0, 5.0, 5.0], (n, 1)),
            np.tile([5.0, 80.0, 5.0], (n, 1)),
            np.tile([5.0, 5.0, 80.0], (n, 1)),
        ]
    )
    ad = AnnData(_csr(native))
    ad.obs[DROPLET_KEY] = "cell"
    ad.obs["cell_type"] = ["t0"] * n + ["t1"] * n + ["Other"] * n
    ad.var[CHI_KEY] = [1 / 3, 1 / 3, 1 / 3]
    ad.obs[DOSE_KEY] = 8.0
    subtract(ad, type_key="cell_type")
    den = np.asarray(ad.layers[LAYER_OUT].todense())

    assert float(den[:n, 1].mean()) == float(den[2 * n :, 0].mean())
