import h5py
import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from ambidose.io import _as_int_csr, _resolve_maybe_gz, read_10x_barcodes, write_10x_mtx


def test_read_10x_barcodes_from_v3_h5(tmp_path):
    path = tmp_path / "filtered_feature_bc_matrix.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/barcodes", data=[b"AA-1", b"BB-1"])

    assert read_10x_barcodes(path) == ["AA-1", "BB-1"]


def test_read_10x_barcodes_from_v2_multigenome_h5_deduplicates(tmp_path):
    path = tmp_path / "filtered_gene_bc_matrices.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("hg19/barcodes", data=[b"AA-1", b"BB-1"])
        f.create_dataset("mm10/barcodes", data=[b"AA-1", b"BB-1"])

    assert read_10x_barcodes(path) == ["AA-1", "BB-1"]


def test_read_10x_barcodes_h5_requires_barcode_dataset(tmp_path):
    path = tmp_path / "not_10x.h5"
    with h5py.File(path, "w"):
        pass

    try:
        read_10x_barcodes(path)
    except ValueError as exc:
        assert "no Cell Ranger barcode dataset" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_maybe_gz_prefers_compressed_when_both_exist(tmp_path):
    bare = tmp_path / "barcodes.tsv"
    gz = tmp_path / "barcodes.tsv.gz"
    bare.write_text("OLD\n")
    gz.write_bytes(b"\x1f\x8b")  # exists; contents unused
    assert _resolve_maybe_gz(bare) == gz
    assert _resolve_maybe_gz(gz) == gz


def test_as_int_csr_rejects_int32_overflow():
    with pytest.raises(ValueError, match="int32"):
        _as_int_csr(sparse.csr_matrix([[3e9]]))


def test_write_10x_mtx_rejects_missing_sample_key_before_writing(tmp_path):
    ad = AnnData(sparse.csr_matrix([[1]]))
    out = tmp_path / "mtx"
    with pytest.raises(KeyError, match="sample_key"):
        write_10x_mtx(ad, out, sample_key="typo")
    assert not out.exists()


def test_write_10x_mtx_default_is_v3_features_tsv_gz(tmp_path):
    import gzip

    ad = AnnData(sparse.csr_matrix([[1.0, 2.0]]))
    ad.var_names = ["GAPDH", "ACTB"]
    ad.var["gene_ids"] = ["ENSG00000111640", "ENSG00000075624"]
    ad.obs_names = ["AAACCTG-1"]
    out = write_10x_mtx(ad, tmp_path / "mtx")
    assert (out / "features.tsv.gz").exists()
    assert not (out / "genes.tsv").exists()
    text = gzip.open(out / "features.tsv.gz", "rt").read()
    assert "ENSG00000111640\tGAPDH\tGene Expression" in text
    assert "ENSG00000075624\tACTB\tGene Expression" in text


def test_write_10x_mtx_v2_genes_tsv_uses_gene_ids(tmp_path):
    ad = AnnData(sparse.csr_matrix([[1.0, 2.0]]))
    ad.var_names = ["GAPDH", "ACTB"]
    ad.var["gene_ids"] = ["ENSG00000111640", "ENSG00000075624"]
    ad.obs_names = ["AAACCTG-1"]
    out = write_10x_mtx(ad, tmp_path / "mtx", version=2)
    text = (out / "genes.tsv").read_text()
    assert "ENSG00000111640\tGAPDH" in text
    assert "ENSG00000075624\tACTB" in text
    assert "GAPDH\tGAPDH" not in text


def test_write_10x_mtx_v2_without_gene_ids_repeats_symbol(tmp_path):
    ad = AnnData(sparse.csr_matrix([[1.0]]))
    ad.var_names = ["GAPDH"]
    ad.obs_names = ["AAACCTG-1"]
    out = write_10x_mtx(ad, tmp_path / "mtx", version=2)
    assert (out / "genes.tsv").read_text() == "GAPDH\tGAPDH\n"


@pytest.mark.parametrize("value", [0.2, -1.0, np.nan, np.inf])
def test_write_10x_mtx_rejects_non_count_values(tmp_path, value):

    ad = AnnData(sparse.csr_matrix([[value]]))
    with pytest.raises(ValueError, match="counts must"):
        write_10x_mtx(ad, tmp_path / "bad")
