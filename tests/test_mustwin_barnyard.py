"""Must-win barnyard floor: product denoise on a compact two-genome toy.

Catches executed-scale regressions that unit toys without soup+empties miss
(0.3.14 Mixture sensitivity drop). Real Mixture/hgmm stay in the eval job.
"""

from __future__ import annotations

import numpy as np

from ambidose._dose import Q_SCALE_LOW_RHO, q_abs_scale
from ambidose.datasets import make_barnyard_toy
from ambidose.metrics import assign_majority_genome, barnyard_kill_row
from ambidose.pp import DROPLET_KEY, LAYER_OUT, denoise, estimate_dose_adaptive


def test_q_scale_skips_shrink_on_clean_library():
    from ambidose._dose import _q_curve_scale

    curve = _q_curve_scale(24.84)
    clean = q_abs_scale(24.84, hat_rho=0.017)
    dirty = q_abs_scale(17.58, hat_rho=0.81)
    assert curve < clean < 1.0
    assert dirty < 1.0
    assert q_abs_scale(24.84, hat_rho=0.0) == 1.0
    assert 0.017 < Q_SCALE_LOW_RHO <= 0.10


def test_mustwin_barnyard_toy_sensitivity_floor():
    adata = make_barnyard_toy(
        n_genes_per_species=24,
        n_empty=80,
        n_human=40,
        n_mouse=40,
        empty_umi=30,
        cell_umi=500,
        contamination=0.15,
        seed=0,
    )
    adata.X = adata.X.astype(np.int32)
    denoise(adata, type_key="true_species", sample_key=None, droplet_key="droplet")
    drop = adata.obs[DROPLET_KEY] if DROPLET_KEY in adata.obs else adata.obs["droplet"]
    cells = adata[drop.astype(str) == "cell"].copy()
    assign_majority_genome(cells)
    row = barnyard_kill_row(cells, cells, method="product", layer=LAYER_OUT)
    assert row["n_inflated"] == 0
    assert 0.0 <= row["specificity"] <= 1.0
    rec = next(iter(adata.uns["ambidose"]["dose"]["samples"].values()))
    if rec["hat_rho"] < Q_SCALE_LOW_RHO:
        curve = q_abs_scale(rec["q"], hat_rho=Q_SCALE_LOW_RHO)
        assert curve <= rec["q_scale"] <= 1.0


def test_mustwin_clean_library_does_not_shrink():
    adata = make_barnyard_toy(
        n_empty=80,
        n_human=30,
        n_mouse=30,
        contamination=0.05,
        seed=1,
    )
    adata.X = adata.X.astype(np.int32)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    from ambidose.pp import estimate_chi

    estimate_chi(adata, sample_key=None, droplet_key=DROPLET_KEY)
    estimate_dose_adaptive(adata, type_key="true_species", sample_key=None, droplet_key=DROPLET_KEY)
    rec = next(iter(adata.uns["ambidose"]["dose"]["samples"].values()))
    if rec["hat_rho"] < Q_SCALE_LOW_RHO:
        curve = q_abs_scale(rec["q"], hat_rho=Q_SCALE_LOW_RHO)
        assert curve <= rec["q_scale"] <= 1.0
