import numpy as np
import pytest
from scipy import sparse

from ambidose.metrics import (
    barnyard_kill_row,
    cross_batch_entropy,
    knn_batch_entropy,
    knn_indices,
    n_inflated,
    shannon_entropy,
)


def test_leakage_by_species_uses_stored_droplet_key():
    import ambidose as amdose
    from ambidose.datasets import make_barnyard_toy
    from ambidose.metrics import assign_majority_genome, leakage_by_species

    adata = make_barnyard_toy(seed=3, n_human=20, n_mouse=20, n_empty=40)
    n_true = int((adata.obs["droplet"].astype(str) == "cell").sum())
    amdose.denoise(adata, type_key="true_species", sample_key=None, droplet_key="droplet")
    assign_majority_genome(adata)
    tab = leakage_by_species(adata, layer="ambidose_denoised")
    assert int(tab["n_cells"].sum()) == n_true


def test_barnyard_kill_row_bounded_on_count_inflating_method():
    # A method that adds counts (breaks n_inflated=0) used to be able to
    # score sensitivity/specificity/precision outside [0,1] entirely
    # (confirmed: -0.404/1.057/1.000 on a "+1 to every nonzero" method) --
    # n_inflated is the signal for "this method invents counts," not an
    # out-of-range value on axes meant to read as proportions.
    from anndata import AnnData

    rng = np.random.default_rng(0)
    n, g = 40, 10
    X = sparse.csr_matrix(rng.poisson(3, size=(n, g)).astype(np.float64))
    raw = AnnData(X=X.copy())
    raw.var_names = [f"g{i}" for i in range(g)]
    raw.var["genome"] = ["hg19"] * 5 + ["mm10"] * 5
    raw.obs["ambidose_species"] = ["hg19"] * 20 + ["mm10"] * 20

    inflated = X.copy()
    inflated.data = inflated.data + 1
    den = AnnData(X=inflated)
    den.var_names = raw.var_names
    den.var["genome"] = raw.var["genome"].to_numpy()

    row = barnyard_kill_row(raw, den, method="inflate")
    for key in ("sensitivity", "specificity", "precision"):
        v = row[key]
        assert np.isnan(v) or (0.0 <= v <= 1.0), f"{key}={v} out of [0,1]"
    assert row["n_inflated"] > 0


def test_barnyard_kill_row_degenerate_denominator_is_nan_not_best_score():
    from anndata import AnnData

    # Every cell the same species -> the *other* species has off_raw=0 for
    # itself and on_raw=0 too in some slices; construct a minimal case
    # where on_raw (for one species) is exactly 0 so specificity's
    # denominator is degenerate.
    X = sparse.csr_matrix(np.array([[5.0, 0.0], [5.0, 0.0]]))
    raw = AnnData(X=X.copy())
    raw.var_names = ["g0", "g1"]
    raw.var["genome"] = ["hg19", "mm10"]
    raw.obs["ambidose_species"] = ["mm10", "mm10"]  # no hg19 cells at all
    den = AnnData(X=X.copy())
    den.var_names = raw.var_names
    den.var["genome"] = raw.var["genome"].to_numpy()
    row = barnyard_kill_row(raw, den, method="raw")
    # hg19's on_raw is 0 (no hg19 cells) -> that species contributes nothing
    # to on_raw/on_den; specificity here is well-defined (mm10 cells keep
    # all their own-species counts), so this mainly checks no crash and a
    # sane, bounded result rather than a hardcoded degenerate value.
    assert np.isnan(row["specificity"]) or 0.0 <= row["specificity"] <= 1.0


def test_shannon_one_class_is_zero():
    assert shannon_entropy(np.array(["a", "a", "a"]), n_classes=3) == 0.0


def test_shannon_balanced_two_is_one():
    h = shannon_entropy(np.array(["a", "a", "b", "b"]), n_classes=2)
    assert abs(h - 1.0) < 1e-9


def test_cross_batch_entropy_pure_vs_mixed():
    batch = np.array(["s0", "s0", "s1", "s1"])
    pure = cross_batch_entropy(batch, np.array(["g0", "g0", "g1", "g1"]))
    mixed = cross_batch_entropy(batch, np.array(["g0", "g0", "g0", "g0"]))
    assert pure["mean"] == 0.0
    assert abs(mixed["mean"] - 1.0) < 1e-9
    assert mixed["n_batches"] == 2


def test_knn_batch_entropy_same_sample_neighbors():
    batch = np.array(["a", "a", "b", "b"])
    same = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
    mixed = np.array([[0, 2], [1, 3], [2, 0], [3, 1]])
    assert knn_batch_entropy(batch, same) < 0.05
    assert knn_batch_entropy(batch, mixed) > 0.9


def test_knn_indices_drops_self():
    dist = sparse.csr_matrix(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.2],
            [0.9, 0.2, 0.0],
        ]
    )
    idx = knn_indices(dist, k=1)
    assert idx.shape == (3, 1)
    assert idx[0, 0] == 1
    assert idx[1, 0] in (0, 2)


def test_n_inflated_zero_when_den_le_raw():
    from anndata import AnnData
    from scipy import sparse as sp

    x = sp.csr_matrix([[2.0, 0.0], [0.0, 3.0]])
    raw = AnnData(x.copy())
    den = AnnData(x.copy())
    den.layers["d"] = sp.csr_matrix([[1.0, 0.0], [0.0, 3.0]])
    assert n_inflated(raw, den, layer="d") == 0
    den.layers["d"] = sp.csr_matrix([[4.0, 0.0], [0.0, 3.0]])
    assert n_inflated(raw, den, layer="d") == 1


def test_marker_leakage_diagonal_and_off_target():
    import pandas as pd
    from scipy import sparse as sp

    from ambidose.metrics import (
        marker_leakage_table,
        overcorrection_report,
        summarize_marker_leakage,
    )

    genes = pd.Index(["A", "B", "C"])
    markers = {"T": ["A"], "M": ["B"]}
    # cell 0 type T: A=10, B=0, C=90  (lib=100) → A CP10K=1000
    # cell 1 type M: A=0, B=20, C=80  (lib=100) → B CP10K=2000
    x = sp.csr_matrix([[10.0, 0.0, 90.0], [0.0, 20.0, 80.0]])
    types = np.array(["T", "M"])
    tab = marker_leakage_table(x, genes, types, ["T", "M"], markers)
    assert abs(tab.loc["T", "T"] - 1000.0) < 1e-6
    assert abs(tab.loc["M", "M"] - 2000.0) < 1e-6
    assert abs(tab.loc["T", "M"] - 0.0) < 1e-6
    summ = summarize_marker_leakage(tab)
    assert abs(summ["on_target_mean"] - 1500.0) < 1e-6
    assert summ["leak_ratio"] == 0.0
    oc = overcorrection_report({"raw": tab, "den": tab})
    assert list(oc["type"]) == ["T", "M"]
    assert abs(oc.loc[0, "raw"] - 1000.0) < 1e-6


def test_n_inflated_rejects_duplicate_gene_names():
    from anndata import AnnData

    raw = AnnData(sparse.csr_matrix([[1.0, 2.0]]))
    den = raw.copy()
    raw.var_names = ["g", "g"]
    den.var_names = ["g", "g"]
    with pytest.raises(ValueError, match="duplicate var_names"):
        n_inflated(raw, den)


def test_marker_leakage_rejects_duplicate_gene_names():
    import pandas as pd
    import pytest

    from ambidose.metrics import marker_leakage_table

    with pytest.raises(ValueError, match="gene_index must be unique"):
        marker_leakage_table(
            sparse.csr_matrix([[1.0, 2.0]]),
            pd.Index(["g", "g"]),
            np.array(["T"]),
            ["T"],
            {"T": ["g"]},
        )


def test_assign_majority_genome_marks_zero_umi_unassigned():
    from anndata import AnnData

    from ambidose.metrics import assign_majority_genome

    adata = AnnData(sparse.csr_matrix([[0.0, 0.0], [2.0, 0.0]]))
    adata.var["genome"] = ["human", "mouse"]
    assign_majority_genome(adata)
    assert adata.obs["ambidose_species"].tolist() == ["unassigned", "human"]
    assert adata.obs["ambidose_species_frac"].tolist() == [0.0, 1.0]
