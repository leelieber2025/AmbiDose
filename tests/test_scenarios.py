"""Named ambient-failure scenarios. Truth fields plus one operator check each."""

import numpy as np

from ambidose.datasets import (
    make_toy,
    merge_type_labels,
    scenario_housekeeping,
    scenario_zero_ambient,
    split_type_labels,
)
from ambidose.pp import denoise, estimate_dose


def _cells(adata):
    return adata.obs["droplet"].astype(str).to_numpy() == "cell"


def test_housekeeping_truth_contract():
    ad = scenario_housekeeping(true_rho=0.1, n_hk=40, n_empty=80, n_cells_per_type=30, seed=1)
    assert ad.uns["scenario"] == "housekeeping"
    assert "true_chi" in ad.uns
    assert {"true_rho", "true_d", "cell_type", "droplet"} <= set(ad.obs.columns)
    assert _cells(ad).sum() == 4 * 30
    assert (~_cells(ad)).sum() == 80


def test_housekeeping_clean_pool_tracks_rho():
    ad = scenario_housekeeping(true_rho=0.1, n_hk=0, n_empty=0, seed=1)
    estimate_dose(ad, type_key="cell_type")
    rho_hat = float(np.median(ad.obs["ambidose_rho"].to_numpy()))
    assert abs(rho_hat - 0.1) < 0.03 + 0.35 * 0.1


def test_housekeeping_shared_genes_bias_rho():
    # Promoted fix (see test_dose.py::test_rho_calibration_housekeeping_genes_target):
    # estimate_dose drops genes that look native in every type
    # (_native_everywhere_mask), removing the ρ=1 ceiling bias those genes
    # used to introduce.
    ad = scenario_housekeeping(true_rho=0.1, n_hk=40, n_empty=0, seed=1)
    estimate_dose(ad, type_key="cell_type")
    rho_hat = float(np.median(ad.obs["ambidose_rho"].to_numpy()))
    assert abs(rho_hat - 0.1) < 0.05


def test_split_type_labels_does_not_change_counts():
    ad = make_toy(n_samples=1, n_empty=40, n_cells=40, seed=2)
    x0 = ad.X.copy()
    true = ad.obs["cell_type"].astype(str).to_numpy()
    split_type_labels(ad, n_splits=2, seed=0)
    assert ad.uns["label_transform"] == "split"
    assert (ad.obs["cell_type"].astype(str).to_numpy() == true).all()
    cells = _cells(ad)
    n_true = len(set(true[cells]) - {"none"})
    n_split = len(set(ad.obs["label_split"].astype(str)[cells]) - {"none"})
    assert n_split >= 2 * n_true
    assert (ad.X - x0).nnz == 0


def test_split_labels_denoise_still_n_inflated_zero():
    ad = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=3)
    split_type_labels(ad, n_splits=2, seed=0)
    cells = ad.obs_names[_cells(ad)].tolist()
    raw = ad.X.tocsr().copy()
    denoise(ad, cell_barcodes=cells, type_key="label_split", sample_key=None)
    den = ad.layers["ambidose_denoised"].tocsr()
    assert (den > raw).nnz == 0
    assert "ambidose_rho_trust" in ad.obs


def test_merge_type_labels_one_type_trust():
    ad = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=4)
    merge_type_labels(ad)
    cells = ad.obs_names[_cells(ad)].tolist()
    assert set(ad.obs.loc[cells, "label_merged"].astype(str)) == {"merged"}
    raw = ad.X.tocsr().copy()
    denoise(ad, cell_barcodes=cells, type_key="label_merged", sample_key=None)
    den = ad.layers["ambidose_denoised"].tocsr()
    assert (den > raw).nnz == 0
    trust = ad.obs.loc[cells, "ambidose_rho_trust"].astype(str)
    # One merged type on the toy has empty UMI comparable to per-cell soup,
    # so the 3× empty-knee band uses a SoupX-like χ take and most cells
    # over-execute relative to protected dose. Trust labels stay in this set.
    assert set(trust) <= {"under_execution", "low_evidence", "over_removal"}


def test_zero_ambient_truth_and_no_inflation():
    ad = scenario_zero_ambient(n_samples=1, n_empty=50, n_cells=40, seed=5)
    assert ad.uns["scenario"] == "zero_ambient"
    cells = _cells(ad)
    assert np.allclose(ad.obs.loc[cells, "true_rho"].to_numpy(dtype=float), 0.0)
    names = ad.obs_names[cells].tolist()
    raw = ad.X.tocsr().copy()
    denoise(ad, cell_barcodes=names, type_key="cell_type", sample_key=None)
    den = ad.layers["ambidose_denoised"].tocsr()
    assert (den > raw).nnz == 0
    # Extra-clear/leftover off; rank-1 capped at empty U soup: some removal, not ~5%.
    loss = float((raw[cells] - den[cells]).sum()) / float(raw[cells].sum())
    assert 0 < loss < 0.04
    assert ad.uns["ambidose"].get("n_empty_consistent_skip_cells", 0) > 0
    rho = ad.obs.loc[cells, "ambidose_rho"].to_numpy(dtype=float)
    # A tighter rho<0.05 check used to be xfailed below this test; it was
    # a wrong expectation, not a bug (see
    # test_zero_ambient_reports_off_block_native_leak_not_zero), and is
    # now a passing assertion, not an xfail.
    assert np.isfinite(rho).all()
    # Dose is still estimated; removal is skipped, so trust is under_execution.


def test_zero_ambient_reports_off_block_native_leak_not_zero():
    # The toy has a 4.76% native off-block floor, so exact zero is not
    # identifiable from expression alone. Lock the operational-dose behavior without treating exact zero as the target.
    # Lower bound was 0.05 under the old fixed 0.704 multiplier; 0.3.14's
    # q-scale shrink floor (0.50, lower than 0.704) pulls this low-q
    # regime's executed ρ down further, median now ~0.0495 -- still
    # clearly nonzero, just past the old threshold's margin.
    ad = scenario_zero_ambient(n_samples=1, n_empty=50, n_cells=40, seed=5)
    cells = _cells(ad)
    denoise(
        ad,
        cell_barcodes=ad.obs_names[cells].tolist(),
        type_key="cell_type",
        sample_key=None,
    )
    rho = ad.obs.loc[cells, "ambidose_rho"].to_numpy(dtype=float)
    assert 0.03 < float(np.median(rho)) < 0.25
