import numpy as np
from anndata import AnnData
from scipy import sparse

from ambidose.baselines import (
    MissingBaseline,
    cluster_cells,
    find_cellbender,
    run_scar,
)
from ambidose.datasets import make_barnyard_toy
from ambidose.io import read_10x_mtx, write_10x_mtx
from ambidose.pp import estimate_chi


def test_find_cellbender_or_missing():
    try:
        p = find_cellbender()
        assert p.exists()
    except MissingBaseline:
        pass


def test_run_scar_missing_scvi():
    adata = make_barnyard_toy(n_empty=20, n_human=10, n_mouse=10, seed=1)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    estimate_chi(adata, sample_key=None)
    try:
        run_scar(adata, max_epochs=1)
    except MissingBaseline as exc:
        assert "scvi-tools" in str(exc)
    else:
        # scvi is installed in this environment
        assert "scar_denoised" in adata.layers


def test_cluster_cells_labels_cells():
    adata = make_barnyard_toy(n_empty=30, n_human=25, n_mouse=25, seed=7)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    cluster_cells(adata, n_top_genes=20)
    cells = adata.obs["ambidose_droplet"].astype(str) == "cell"
    assert (adata.obs.loc[cells, "ambidose_cluster"] != "-1").all()


def test_cluster_cells_tiny_input_does_not_crash():
    # n_obs=2 -> the PCA n_comps ceiling (min(n_obs, n_vars) - 1) is 1, below
    # the floor of 2 this used to apply unconditionally (sklearn's PCA then
    # raises: n_comps=2 > min(n_obs, n_vars)-1=1). Too few cells for a
    # meaningful embedding at all -- should fall back to one cluster, not
    # crash.
    x = sparse.csr_matrix(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    adata = AnnData(x)
    adata.var_names = ["g0", "g1", "g2"]
    cluster_cells(adata)
    assert list(adata.obs["ambidose_cluster"].astype(str)) == ["0", "0"]


def test_write_10x_mtx(tmp_path):
    adata = make_barnyard_toy(n_empty=5, n_human=5, n_mouse=5, seed=8)
    write_10x_mtx(adata, tmp_path / "mtx")
    assert (tmp_path / "mtx" / "matrix.mtx.gz").exists()
    assert (tmp_path / "mtx" / "barcodes.tsv.gz").exists()
    assert (tmp_path / "mtx" / "features.tsv.gz").exists()
    assert sparse.issparse(adata.X)
    back = read_10x_mtx(tmp_path / "mtx")
    assert back.n_obs == adata.n_obs
    assert back.n_vars == adata.n_vars
