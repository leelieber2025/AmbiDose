import json

import numpy as np
import pytest
import scanpy as sc
from scipy import sparse

import ambidose as amdose
from ambidose.cli import main
from ambidose.datasets import make_toy
from ambidose.io import read_manifest, write_h5ad
from ambidose.pp import CHI_KEY, DROPLET_KEY, _write_rho_trust
from ambidose.reporting import _stable_top_n, inspect_input, summarize, write_report


def _denoised_toy():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    return adata, cells


def test_classify_accepts_whitelist_path(tmp_path):
    adata = make_toy(n_samples=1, n_empty=30, n_cells=10, seed=4)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    path = tmp_path / "cells.tsv"
    path.write_text("\n".join(cells) + "\n")
    amdose.classify_droplets(adata, cell_barcodes=path)
    assert (adata.obs.loc[cells, "ambidose_droplet"].astype(str) == "cell").all()


def test_denoise_passes_empty_umi_max_to_call_cells_lower(monkeypatch):
    # denoise(empty_umi_max=...) must reach call_cells(lower=...) too, not
    # only classify_droplets -- otherwise chi-refinement's ambient pool
    # keeps using the default 100 regardless of what the caller asked for.
    adata = make_toy(n_samples=1, n_empty=40, n_cells=12, seed=31)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()

    seen_lower = []
    real_call_cells = amdose.pp.call_cells

    def spy(*args, **kwargs):
        seen_lower.append(kwargs.get("lower"))
        return real_call_cells(*args, **kwargs)

    monkeypatch.setattr("ambidose.pp.call_cells", spy)
    amdose.denoise(
        adata,
        cell_barcodes=cells,
        empty_umi_max=50,
        type_key="cell_type",
        sample_key=None,
    )
    assert seen_lower == [50]


def test_analysis_ready_uses_stored_droplet_and_layer_keys():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    amdose.denoise(
        adata,
        type_key="cell_type",
        sample_key=None,
        droplet_key="droplet",
        layer_out="cleaned",
    )
    n_cell = int((adata.obs["droplet"].astype(str) == "cell").sum())
    result = amdose.analysis_ready(adata)
    assert result.n_obs == n_cell
    assert "cleaned" in result.layers
    assert (result.layers["cleaned"] != result.X).nnz == 0


def test_summarize_uses_input_layer_not_x():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X.copy()
    adata.X.data = np.zeros_like(adata.X.data)
    amdose.denoise(
        adata,
        cell_barcodes=cells,
        type_key="cell_type",
        sample_key=None,
        layer="counts",
    )
    result = summarize(adata)
    assert result["total_removed_fraction"] > 0
    assert result["input_layer"] == "counts"
    assert float(adata.X.sum()) == 0.0
    assert float(adata.layers["counts"].sum()) > 0.0
    assert float(adata.layers["ambidose_denoised"].sum()) > 0.0


def test_summarize_after_analysis_ready_keeps_removed_fraction():
    adata, cells = _denoised_toy()
    before = summarize(adata)["total_removed_umi"]
    assert before > 0
    ready = amdose.analysis_ready(adata)
    after = summarize(ready)["total_removed_umi"]
    assert after == before


def test_analysis_ready_preserves_raw_and_uses_denoised_x():
    adata, cells = _denoised_toy()
    result = amdose.analysis_ready(adata)
    assert result.n_obs == len(cells)
    assert "raw_counts" in result.layers
    assert (result.layers["ambidose_denoised"] != result.X).nnz == 0
    assert (result.layers["raw_counts"] < result.X).nnz == 0


def test_analysis_ready_uses_input_layer_not_x():
    # X is a zeroed decoy; the real raw counts denoise() actually used live
    # in a named layer. analysis_ready()'s raw_counts output must come from
    # that layer (via raw_count_matrix()), not from copying the decoy X.
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X.copy()
    adata.X.data = np.zeros_like(adata.X.data)
    amdose.denoise(
        adata,
        cell_barcodes=cells,
        type_key="cell_type",
        sample_key=None,
        layer="counts",
    )
    result = amdose.analysis_ready(adata)
    assert float(result.layers["raw_counts"].sum()) > 0


def test_write_report_custom_droplet_key(tmp_path):
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=26)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(
        adata,
        type_key="cell_type",
        sample_key=None,
        droplet_key="droplet",
    )
    assert adata.uns["ambidose"]["droplet_key"] == "droplet"
    path = write_report(adata, tmp_path / "report.html")
    assert path.exists()
    result = summarize(adata)
    assert result["n_cells"] == len(cells)
    assert result["n_empty"] == int((adata.obs["droplet"].astype(str) == "empty").sum())
    assert (
        result["n_trust_ok"]
        + result["n_trust_low_evidence"]
        + result["n_trust_ceiling_risk"]
        + result["n_trust_type_structure_risk"]
        + result["n_trust_under_execution"]
        + result["n_trust_over_removal"]
        == result["n_cells"]
    )
    assert result["droplet_key"] == "droplet"


def test_summary_and_html_report(tmp_path):
    adata, cells = _denoised_toy()
    result = summarize(adata)
    assert result["n_cells"] == len(cells)
    assert result["n_inflated"] == 0
    assert 0 <= result["total_removed_fraction"] <= 1
    assert "n_trust_ok" in result
    assert set(result["cell_umi_retention_percentiles"]) == {"p1", "p5", "median", "p95", "p99"}
    assert result["n_cells_below_50pct_umi_retention"] >= 0
    assert (
        result["n_trust_ok"]
        + result["n_trust_low_evidence"]
        + result["n_trust_ceiling_risk"]
        + result["n_trust_type_structure_risk"]
        + result["n_trust_under_execution"]
        + result["n_trust_over_removal"]
        == result["n_cells"]
    )
    assert "native_genes_by_type" not in result.get("diagnostics", {})
    path = write_report(adata, tmp_path / "report.html")
    text = path.read_text()
    assert "AmbiDose QC report" in text
    assert '"native_genes_by_type"' not in text
    assert "n_native_genes_by_type" in text
    assert "How to read ambient-fraction QC" in text
    assert "Count-retention check" in text
    assert "zero-ambient simulations" in text
    assert "not cell-filtering recommendations" in text
    assert "data:image/png;base64" in text


def test_stable_top_n_breaks_ties_by_name_not_array_order():
    # Regression: summarize()'s and pl.ambient_profile()'s top-20 lists used
    # plain np.argsort(...)[::-1][:n], so tied values resolved by incidental
    # gene column order rather than something reproducible. _stable_top_n
    # must break ties by name, independent of input order.
    values = np.array([5.0, 3.0, 5.0, 1.0, 5.0])
    names = np.array(["zeta", "beta", "alpha", "delta", "gamma"])
    top = _stable_top_n(values, names, 3)
    # All three ties (5.0: zeta/alpha/gamma) must appear, ordered
    # alphabetically among themselves, ahead of the non-tied beta/delta.
    assert list(names[top]) == ["alpha", "gamma", "zeta"]

    # Same values, different physical order -> identical chosen set/order.
    values2 = np.array([5.0, 5.0, 5.0, 3.0, 1.0])
    names2 = np.array(["gamma", "alpha", "zeta", "beta", "delta"])
    top2 = _stable_top_n(values2, names2, 3)
    assert list(names2[top2]) == ["alpha", "gamma", "zeta"]


def test_summarize_top_removed_genes_tie_break_is_deterministic(monkeypatch):
    adata, cells = _denoised_toy()
    # Force an exact tie across every gene's removed-UMI count so the
    # resulting order is pure tie-break behavior, not real signal.
    adata.var["ambidose_removed_umi"] = 7.0
    result = summarize(adata)
    top_names = [row["gene"] for row in result["top_removed_genes"]]
    assert top_names == sorted(top_names)


def test_summarize_without_denoise_raises():
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=1)
    with pytest.raises(KeyError, match="droplet_key"):
        summarize(adata)


def test_inspect_matches_external_whitelist(tmp_path):
    adata = make_toy(n_samples=1, n_empty=30, n_cells=10, seed=4)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    raw = tmp_path / "raw.h5ad"
    whitelist = tmp_path / "cells.tsv"
    write_h5ad(adata, raw)
    whitelist.write_text("\n".join(cells) + "\n")
    result = inspect_input(raw, cell_barcodes=whitelist)
    assert result["status"] == "ok"
    assert result["whitelist_mode"] == "external"
    assert result["n_whitelist_matched"] == len(cells)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    expected_empty = (totals > 0) & (totals <= 100)
    expected_empty[adata.obs_names.isin(cells)] = False
    assert result["n_empty_candidates"] == int(expected_empty.sum())


def test_manifest_resolves_relative_paths(tmp_path):
    raw = tmp_path / "raw.h5ad"
    cells = tmp_path / "cells.tsv"
    raw.touch()
    cells.touch()
    manifest = tmp_path / "libraries.tsv"
    manifest.write_text("library\traw_counts\tcell_barcodes\nL1\traw.h5ad\tcells.tsv\n")
    rows = read_manifest(manifest)
    assert rows[0].library == "L1"
    assert rows[0].raw_counts == raw
    assert rows[0].cell_barcodes == cells


def test_cli_analysis_ready_report_and_json(tmp_path):
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=5)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    raw = tmp_path / "raw.h5ad"
    whitelist = tmp_path / "cells.tsv"
    output = tmp_path / "cleaned.h5ad"
    report = tmp_path / "report.html"
    summary_json = tmp_path / "summary.json"
    write_h5ad(adata, raw)
    whitelist.write_text("\n".join(cells) + "\n")
    assert (
        main(
            [
                "denoise",
                "--input",
                str(raw),
                "--cell-barcodes",
                str(whitelist),
                "--type-key",
                "cell_type",
                "--cells-only",
                "--report",
                str(report),
                "--summary-json",
                str(summary_json),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = sc.read_h5ad(output)
    assert result.n_obs == len(cells)
    assert "raw_counts" in result.layers
    assert report.exists()
    assert json.loads(summary_json.read_text())["n_inflated"] == 0


def test_cli_10x_output_is_corrected_cells(tmp_path):
    adata, cells = _denoised_toy()
    result = amdose.analysis_ready(adata)
    out = tmp_path / "mtx"
    amdose.io.write_10x_mtx(result, out)
    matrix = sc.read_10x_mtx(out, var_names="gene_symbols")
    assert matrix.n_obs == len(cells)
    assert sparse.issparse(matrix.X)
    assert np.issubdtype(matrix.X.dtype, np.number)


def test_denoise_cells_only_without_chi_raises():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=8)
    cells = adata[adata.obs["droplet"].astype(str) == "cell"].copy()
    # Whitelist refinement runs by default; a cells-only object where every
    # barcode is whitelisted has no non-whitelist droplets to build an
    # ambient pool from, so the chi step itself raises "at least 10" first
    # (same underlying problem estimate_chi would otherwise raise on: no
    # empty droplets to estimate ambient from).
    with pytest.raises(ValueError, match="at least 10"):
        amdose.denoise(
            cells, cell_barcodes=list(cells.obs_names), type_key="cell_type", sample_key=None
        )


def test_estimate_chi_few_empties_raises():
    adata = make_toy(n_samples=1, n_empty=5, n_cells=20, seed=9)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"]
    with pytest.raises(ValueError, match="need at least 10 empty"):
        amdose.estimate_chi(adata, sample_key=None)


def test_denoise_filtered_matrix_without_empties_raises():
    adata = make_toy(n_samples=1, n_empty=0, n_cells=30, seed=10)
    with pytest.raises(ValueError, match="cell_barcodes"):
        amdose.denoise(adata, type_key="cell_type", sample_key=None)


def test_denoise_second_call_with_cell_barcodes_raises():
    adata, cells = _denoised_toy()
    with pytest.raises(ValueError, match="already completed"):
        amdose.denoise(
            adata,
            cell_barcodes=cells[:10],
            type_key="cell_type",
            sample_key=None,
        )


def test_denoise_twice_raises_instead_of_double_subtracting():
    adata, _ = _denoised_toy()
    raw_before = adata.layers["raw_counts"].copy()
    den_before = adata.layers["ambidose_denoised"].copy()
    with pytest.raises(ValueError, match="already completed"):
        amdose.denoise(adata, type_key="cell_type", sample_key=None)
    assert (adata.layers["raw_counts"] != raw_before).nnz == 0
    assert (adata.layers["ambidose_denoised"] != den_before).nnz == 0


def test_denoise_rejects_conflicting_preexisting_raw_counts():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=44)
    adata.layers["counts"] = adata.X.copy()
    adata.layers["raw_counts"] = sparse.csr_matrix(adata.X.shape, dtype=adata.X.dtype)
    with pytest.raises(ValueError, match="differs from the requested input"):
        amdose.denoise(
            adata,
            layer="counts",
            type_key="cell_type",
            sample_key=None,
            empty_umi_max=100,
        )


def test_denoise_rejects_output_overwriting_input_or_raw():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=45)
    adata.layers["counts"] = adata.X.copy()
    with pytest.raises(ValueError, match="must not overwrite"):
        amdose.denoise(adata, layer="counts", layer_out="counts", sample_key=None)
    with pytest.raises(ValueError, match="reserved"):
        amdose.denoise(adata, layer_out="raw_counts", sample_key=None)


def test_denoise_rejects_expect_cells_with_diem_atomically():
    adata = make_toy(n_samples=1, n_empty=20, n_cells=20, seed=50)
    obs_before = adata.obs.copy(deep=True)

    with pytest.raises(ValueError, match="expect_cells is not used"):
        amdose.denoise(
            adata, cell_calling="diem", expect_cells=10, type_key="cell_type", sample_key=None
        )

    assert adata.obs.equals(obs_before)
    assert "ambidose" not in adata.uns


def test_denoise_rejects_overwriting_existing_output_layer():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=46)
    original = adata.X.copy()
    adata.layers["spliced"] = original.copy()

    with pytest.raises(ValueError, match="already exists"):
        amdose.denoise(
            adata, layer_out="spliced", type_key="cell_type", sample_key=None, empty_umi_max=100
        )

    assert (adata.layers["spliced"] != original).nnz == 0


def test_denoised_layer_does_not_alias_x_after_normalization():
    adata, _ = _denoised_toy()
    den_before = adata.layers["ambidose_denoised"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    assert (adata.layers["ambidose_denoised"] != den_before).nnz == 0


@pytest.mark.parametrize("bad_label", [np.nan, None, "", "   "])
def test_denoise_rejects_missing_or_empty_sample_labels(bad_label):
    adata = make_toy(n_samples=2, n_empty=40, n_cells=20, seed=46)
    adata.obs["sample"] = adata.obs["sample"].astype(object)
    adata.obs.iloc[0, adata.obs.columns.get_loc("sample")] = bad_label
    with pytest.raises(ValueError, match="missing or empty sample labels"):
        amdose.denoise(adata, type_key="cell_type", empty_umi_max=100, sample_key="sample")


def test_denoise_rejects_log_counts():
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=14)
    adata.X = adata.X.astype(np.float64)
    adata.X.data = np.log1p(adata.X.data)
    with pytest.raises(ValueError, match="raw integer UMI"):
        amdose.denoise(adata, empty_umi_max=80, sample_key=None)


def test_denoise_writes_rho_trust(capsys):
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(adata, cell_barcodes=cells, type_key="cell_type", sample_key=None)
    assert "ambidose_rho_trust" in adata.obs
    assert set(adata.obs.loc[cells, "ambidose_rho_trust"].astype(str)) <= {
        "ok",
        "low_evidence",
        "ceiling_risk",
        "type_structure_risk",
        "under_execution",
        "over_removal",
    }
    err = capsys.readouterr().err
    assert "denoise completed" in err
    assert "QC for estimated ambient fractions" in err
    assert "corrected counts" in err
    assert "counts" in adata.uns["ambidose"]["trust"]


def test_over_removal_is_directionally_flagged():
    adata, cells = _denoised_toy()
    idx = adata.obs_names.get_loc(cells[0])
    den = adata.layers["ambidose_denoised"].tolil()
    den[idx, :] = 0
    adata.layers["ambidose_denoised"] = den.tocsr()
    adata.obs.iloc[idx, adata.obs.columns.get_loc("ambidose_d")] = 1.0
    _write_rho_trust(
        adata,
        droplet_key="ambidose_droplet",
        cell_label="cell",
        layer_out="ambidose_denoised",
    )
    assert str(adata.obs.iloc[idx]["ambidose_rho_trust"]) == "over_removal"
    assert adata.obs.iloc[idx]["ambidose_dose_execution_ratio"] > 1.05
    assert adata.obs.iloc[idx]["ambidose_removed_fraction"] > 0.5


def test_denoise_cells_only_with_stored_chi_does_not_invent_empties():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=11)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"]
    amdose.estimate_chi(adata, sample_key=None)
    cells = adata[adata.obs["droplet"].astype(str) == "cell"].copy()
    # Drop droplet labels: this is the FAQ cells-only + stored χ path.
    del cells.obs[DROPLET_KEY]
    # One real cell with UMI below the default empty cutoff; it must stay a cell.
    x = cells.X.tolil()
    x[0, :] = 0
    x[0, 0] = 40
    cells.X = x.tocsr()
    amdose.denoise(cells, type_key="cell_type", sample_key=None)
    assert (cells.obs[DROPLET_KEY].astype(str) == "cell").all()
    assert "ambidose_denoised" in cells.layers
    assert np.isfinite(cells.obs["ambidose_d"].to_numpy()).all()


def test_denoise_one_type_library_returns_minimal_schema():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=12)
    adata.obs["cell_type"] = np.where(adata.obs["droplet"].astype(str) == "cell", "t0", "none")
    amdose.denoise(
        adata,
        cell_barcodes=adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist(),
        type_key="cell_type",
        sample_key=None,
    )
    cells = adata.obs[DROPLET_KEY].astype(str) == "cell"
    public = {
        "ambidose_droplet",
        "ambidose_d",
        "ambidose_rho",
        "ambidose_removed_umi",
        "ambidose_removed_fraction",
        "ambidose_dose_execution_ratio",
        "ambidose_rho_trust",
    }
    assert {column for column in adata.obs if column.startswith("ambidose_")} == public
    assert CHI_KEY in adata.var.columns
    rho = adata.obs.loc[cells, "ambidose_rho"].to_numpy(dtype=float)
    # 1-type χ-mixture EM reports ρ≈1; the quantile-floor fallback must not.
    assert float(np.median(rho)) < 0.5


def test_denoise_rejects_view():
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=13)
    with pytest.raises(ValueError, match="got a view"):
        amdose.denoise(adata[:5], type_key="cell_type", sample_key=None)


def test_cli_manifest_uses_per_library_whitelists(tmp_path):
    lines = ["library\traw_counts\tcell_barcodes"]
    expected = 0
    for library, seed in [("L1", 21), ("L2", 22)]:
        adata = make_toy(n_samples=1, n_empty=30, n_cells=12, seed=seed)
        cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
        raw = tmp_path / f"{library}.h5ad"
        whitelist = tmp_path / f"{library}.tsv"
        write_h5ad(adata, raw)
        whitelist.write_text("\n".join(cells) + "\n")
        lines.append(f"{library}\t{raw.name}\t{whitelist.name}")
        expected += len(cells)
    manifest = tmp_path / "libraries.tsv"
    manifest.write_text("\n".join(lines) + "\n")
    output = tmp_path / "combined.h5ad"
    assert (
        main(
            [
                "denoise",
                "--manifest",
                str(manifest),
                "--type-key",
                "cell_type",
                "--cells-only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = sc.read_h5ad(output)
    assert result.n_obs == expected
    assert set(result.obs["sample"].astype(str)) == {"L1", "L2"}


def test_cli_manifest_default_refines_whitelist_off_trusts_it(tmp_path):
    adata = make_toy(n_samples=1, n_empty=40, n_cells=12, seed=31)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    empties = adata.obs_names[adata.obs["droplet"].astype(str) == "empty"].tolist()
    inflated = cells + empties[:15]
    raw = tmp_path / "L1.h5ad"
    whitelist = tmp_path / "L1.tsv"
    write_h5ad(adata, raw)
    whitelist.write_text("\n".join(inflated) + "\n")
    manifest = tmp_path / "libraries.tsv"
    manifest.write_text(f"library\traw_counts\tcell_barcodes\nL1\t{raw.name}\t{whitelist.name}\n")

    refined = tmp_path / "refined.h5ad"
    assert (
        main(
            [
                "denoise",
                "--manifest",
                str(manifest),
                "--type-key",
                "cell_type",
                "--cells-only",
                "--output",
                str(refined),
            ]
        )
        == 0
    )
    # Whitelist refinement against empty-droplet chi is on by default (no
    # --cell-calling needed) -- the soup-like barcodes get dropped.
    n_refined = sc.read_h5ad(refined).n_obs
    assert n_refined < len(inflated)
    assert n_refined >= len(cells) - 2

    trusted = tmp_path / "trusted.h5ad"
    assert (
        main(
            [
                "denoise",
                "--manifest",
                str(manifest),
                "--type-key",
                "cell_type",
                "--cell-calling",
                "off",
                "--cells-only",
                "--output",
                str(trusted),
            ]
        )
        == 0
    )
    # --cell-calling off is the only way to trust the manifest whitelist
    # as-is -- the soup-like barcodes stay in.
    assert sc.read_h5ad(trusted).n_obs == len(inflated)


def test_denoise_cli_input_auto_detected_whitelist_refined_by_default(tmp_path):
    from ambidose.io import write_10x_mtx

    adata = make_toy(n_samples=1, n_empty=40, n_cells=12, seed=33)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    empties = adata.obs_names[adata.obs["droplet"].astype(str) == "empty"].tolist()
    inflated = cells + empties[:15]

    raw_dir = tmp_path / "raw_feature_bc_matrix"
    write_10x_mtx(adata, raw_dir)
    filt_dir = tmp_path / "filtered_feature_bc_matrix"
    filt_dir.mkdir()
    (filt_dir / "barcodes.tsv").write_text("\n".join(inflated) + "\n")

    refined = tmp_path / "refined.h5ad"
    assert (
        main(
            [
                "denoise",
                "--input",
                str(raw_dir),
                "--cells-only",
                "--output",
                str(refined),
            ]
        )
        == 0
    )
    # Auto-detected Cell Ranger filtered barcodes are refined against
    # empty-droplet chi by default -- previously the default ('diem')
    # discarded them entirely and ran the from-scratch mixture model
    # instead (auto-detected barcodes only got used when --cell-calling
    # chi was passed explicitly).
    n_refined = sc.read_h5ad(refined).n_obs
    assert n_refined < len(inflated)
    assert n_refined >= len(cells) - 2

    trusted = tmp_path / "trusted.h5ad"
    assert (
        main(
            [
                "denoise",
                "--input",
                str(raw_dir),
                "--cell-calling",
                "off",
                "--cells-only",
                "--output",
                str(trusted),
            ]
        )
        == 0
    )
    # --cell-calling off is the only way to trust the auto-detected
    # Cell Ranger whitelist as-is.
    assert sc.read_h5ad(trusted).n_obs == len(inflated)


def test_coarse_leiden_resolution_is_locked():
    from ambidose.pp import (
        DROPLET_KEY,
        LEIDEN_RESOLUTION_COARSE,
        LEIDEN_RESOLUTION_FINE,
        LEIDEN_RESOLUTION_MEDIUM,
        _resolve_coarse_resolution,
    )

    assert LEIDEN_RESOLUTION_COARSE == 0.08
    assert LEIDEN_RESOLUTION_MEDIUM == 0.2
    assert LEIDEN_RESOLUTION_FINE == 0.35
    adata = make_toy(n_samples=1, n_empty=20, n_cells=12, seed=3)
    adata.obs[DROPLET_KEY] = ["cell"] * adata.n_obs
    assert _resolve_coarse_resolution(adata, droplet_key=DROPLET_KEY, cell_label="cell") == 0.35


def test_typing_fast_is_noop_below_cell_threshold(monkeypatch):
    from ambidose.pp import CLUSTER_KEY, DROPLET_KEY, _embed_coarse_hvg, resolve_type_key

    seen: list[bool] = []
    orig = _embed_coarse_hvg

    def wrapped(sub, *, typing_fast):
        seen.append(bool(typing_fast))
        return orig(sub, typing_fast=typing_fast)

    monkeypatch.setattr("ambidose._typing._embed_coarse_hvg", wrapped)
    adata = make_toy(n_samples=1, n_empty=40, n_cells=80, n_genes=50, seed=4)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    resolve_type_key(adata)
    assert CLUSTER_KEY in adata.obs
    assert seen == [False]
    clustering = adata.uns.get("ambidose", {}).get("clustering", {})
    assert clustering.get("typing_fast") in (False, None)


def test_auto_grouping_is_label_free_leiden():
    from ambidose.pp import CLUSTER_KEY, TYPE_KEY, resolve_type_key

    adata = make_toy(n_samples=1, n_empty=40, n_cells=30, n_genes=50, seed=9)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"].astype(str)
    assert resolve_type_key(adata) == CLUSTER_KEY
    assert CLUSTER_KEY in adata.obs
    assert TYPE_KEY not in adata.obs
    from ambidose._shared import EMPTY_TYPE, _validated_type_values

    noncells = adata.obs["ambidose_droplet"].astype(str).to_numpy() != "cell"
    validated = _validated_type_values(adata, CLUSTER_KEY).to_numpy()
    assert all(value is EMPTY_TYPE for value in validated[noncells])


def test_auto_typing_runs_independently_per_library(monkeypatch):
    from ambidose.pp import CLUSTER_KEY, DROPLET_KEY, resolve_type_key

    adata = make_toy(n_samples=2, n_empty=20, n_cells=20, n_genes=30, seed=41)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    calls = []

    def fake_single(sub, **kwargs):
        sample = sub.obs["sample"].astype(str).unique().tolist()
        assert len(sample) == 1
        calls.append(sample[0])
        is_cell = sub.obs[DROPLET_KEY].astype(str).to_numpy() == "cell"
        labels = np.array(["-1"] * sub.n_obs, dtype=object)
        labels[is_cell] = "0"
        sub.obs[CLUSTER_KEY] = labels
        return CLUSTER_KEY

    monkeypatch.setattr("ambidose._typing._annotate_coarse_types_single", fake_single)
    assert resolve_type_key(adata, sample_key="sample") == CLUSTER_KEY
    assert len(calls) == 2
    cells = adata.obs[DROPLET_KEY].astype(str) == "cell"
    by_sample = (
        adata.obs.loc[cells]
        .groupby("sample", observed=True)[CLUSTER_KEY]
        .agg(lambda x: set(x.astype(str)))
    )
    assert by_sample.iloc[0].isdisjoint(by_sample.iloc[1])
    diag = adata.uns["ambidose"]["type_sample_dependence"]
    assert diag["scope"] == "per_sample"
    assert diag["cramers_v"] == pytest.approx(1.0)


def test_type_sample_dependence_excludes_missing_types():
    from ambidose.pp import _type_sample_dependence

    adata = make_toy(n_samples=2, n_empty=10, n_cells=10, seed=47)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    is_cell = adata.obs[DROPLET_KEY].astype(str).to_numpy() == "cell"
    adata.obs["qc_type"] = np.where(is_cell, adata.obs["cell_type"], None)
    first_cell = np.flatnonzero(is_cell)[0]
    adata.obs.iloc[first_cell, adata.obs.columns.get_loc("qc_type")] = None

    _type_sample_dependence(
        adata,
        type_key="qc_type",
        sample_key="sample",
        droplet_key=DROPLET_KEY,
        cell_label="cell",
        scope="provided",
    )

    table = adata.uns["ambidose"]["type_sample_dependence"]["contingency"]
    assert "nan" not in table.index
    assert table.to_numpy().sum() == int(is_cell.sum()) - 1


def test_provided_types_report_library_dependence():
    from ambidose.pp import DROPLET_KEY, resolve_type_key

    adata = make_toy(n_samples=2, n_empty=20, n_cells=20, n_genes=30, seed=42)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    adata.obs["provided_type"] = adata.obs["sample"].astype(str)
    assert resolve_type_key(adata, type_key="provided_type", sample_key="sample") == "provided_type"
    diag = adata.uns["ambidose"]["type_sample_dependence"]
    assert diag["scope"] == "provided"
    assert diag["cramers_v"] == pytest.approx(1.0)
    assert diag["median_type_sample_fraction"] == pytest.approx(1.0)


def test_analysis_ready_twice_does_not_clobber_raw():
    adata, _cells = _denoised_toy()
    once = amdose.analysis_ready(adata)
    raw_once = once.layers["raw_counts"].copy()
    den_sum = float(once.X.sum())
    raw_sum = float(raw_once.sum())
    assert raw_sum > den_sum
    twice = amdose.analysis_ready(once)
    assert (twice.layers["raw_counts"] != raw_once).nnz == 0
    assert float(twice.layers["raw_counts"].sum()) == raw_sum
    assert float(twice.X.sum()) == den_sum


def test_barcode_rank_uses_raw_counts_after_analysis_ready():
    import matplotlib.pyplot as plt

    adata, _cells = _denoised_toy()
    ready = amdose.analysis_ready(adata)
    raw_tot = np.asarray(ready.layers["raw_counts"].sum(axis=1)).ravel()
    den_tot = np.asarray(ready.X.sum(axis=1)).ravel()
    assert not np.allclose(raw_tot, den_tot)
    ax = amdose.pl.barcode_rank(ready)
    plotted = np.asarray(ax.lines[0].get_ydata())
    expected = np.sort(np.maximum(raw_tot, 1.0))[::-1]
    np.testing.assert_allclose(plotted, expected)
    den_expected = np.sort(np.maximum(den_tot, 1.0))[::-1]
    assert not np.allclose(plotted, den_expected)
    plt.close(ax.figure)


def test_summarize_includes_doublet_fields_when_marked():
    adata, _cells = _denoised_toy()
    assert "doublet_marking_run" not in summarize(adata)
    amdose.mark_doublets(adata, type_key="cell_type", sample_key=None)
    data = summarize(adata)
    assert data["doublet_marking_run"] is True
    assert "n_doublet_scrublet" in data
    assert "n_type_residual_scored" in data
    assert (
        data["n_type_residual_doublet_leaning"] + data["n_type_residual_soup_leaning"]
        <= data["n_type_residual_scored"]
    )


def test_report_renders_doublet_diagnostic_panel(tmp_path):
    adata, _cells = _denoised_toy()
    amdose.mark_doublets(adata, type_key="cell_type", sample_key=None)
    out = tmp_path / "report.html"
    write_report(adata, out)
    assert out.is_file()
    # The extra panel is rasterized into the embedded PNG (not literal HTML
    # text); check the machine-readable summary payload embedded alongside
    # it instead, which does carry the doublet QC fields as visible JSON.
    assert "doublet_marking_run" in out.read_text()


def test_denoise_raw_kwarg_rejects_layer():
    raw = make_toy(n_samples=1, n_empty=30, n_cells=20, seed=120)
    cells = raw.obs["droplet"].astype(str) == "cell"
    filtered = raw[cells].copy()
    filtered.layers["counts"] = filtered.X.copy()

    with pytest.raises(ValueError, match="raw pool X"):
        amdose.denoise(filtered, raw=raw, layer="counts", sample_key=None)


def test_denoise_raw_kwarg_matches_traditional_path(tmp_path):
    from ambidose.io import read_10x_mtx, write_10x_mtx
    from ambidose.pp import LAYER_OUT

    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    cell_barcodes = raw.obs_names[cell_mask].tolist()

    raw_dir = tmp_path / "raw_feature_bc_matrix"
    write_10x_mtx(raw, raw_dir)
    filtered = read_10x_mtx(raw_dir)[cell_barcodes].copy()
    filtered.obs["my_own_annotation"] = "kept"

    out = amdose.denoise(filtered, raw=raw_dir, sample_key=None)
    assert LAYER_OUT in out.layers
    assert (out.obs["my_own_annotation"] == "kept").all()

    raw2 = raw.copy()
    amdose.denoise(raw2, cell_barcodes=cell_barcodes, cell_calling="off", sample_key=None)
    ref = amdose.analysis_ready(raw2)[out.obs_names, out.var_names]
    assert (out.layers[LAYER_OUT] != ref.layers[LAYER_OUT]).nnz == 0

    with pytest.raises(ValueError, match="pass only one of"):
        amdose.denoise(filtered, raw=raw_dir, cell_barcodes=cell_barcodes, sample_key=None)


def test_denoise_raw_kwarg_normalizes_barcode_suffix_mismatch():
    """Copy-back uses normalize_barcode() so '-1' suffix conventions can differ."""
    from ambidose.pp import LAYER_OUT

    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    cell_barcodes = raw.obs_names[cell_mask].tolist()

    raw_suffixed = raw.copy()
    raw_suffixed.obs_names = [f"{b}-1" for b in raw_suffixed.obs_names]
    filtered_bare = raw[cell_barcodes].copy()
    out = amdose.denoise(filtered_bare, raw=raw_suffixed, sample_key=None)
    assert out.n_obs == len(cell_barcodes)
    assert LAYER_OUT in out.layers

    filtered_suffixed = raw[cell_barcodes].copy()
    filtered_suffixed.obs_names = [f"{b}-1" for b in filtered_suffixed.obs_names]
    out2 = amdose.denoise(filtered_suffixed, raw=raw.copy(), sample_key=None)
    assert out2.n_obs == len(cell_barcodes)


def test_denoise_raw_kwarg_keeps_multi_sample_chi():
    from ambidose.pp import CHI_KEY

    raw = make_toy(n_samples=2, n_empty=50, n_cells=40, seed=9)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    cell_barcodes = raw.obs_names[cell_mask].tolist()
    filtered = raw[cell_barcodes].copy()

    out = amdose.denoise(filtered, raw=raw.copy(), sample_key="sample")
    assert CHI_KEY in out.uns
    assert out.uns[CHI_KEY].shape[0] == 2


def test_denoise_report_kwarg_writes_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=17)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(adata, cell_barcodes=cells, sample_key=None, report=True)
    default_path = tmp_path / "ambidose_report.html"
    assert default_path.is_file()

    custom_path = tmp_path / "custom.html"
    adata2 = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=18)
    cells2 = adata2.obs_names[adata2.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(adata2, cell_barcodes=cells2, sample_key=None, report=custom_path)
    assert custom_path.is_file()


def test_denoise_raw_kwarg_report_includes_empty_droplets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    cell_barcodes = raw.obs_names[cell_mask].tolist()
    filtered = raw[cell_barcodes].copy()

    report_path = tmp_path / "report.html"
    out = amdose.denoise(filtered, raw=raw.copy(), sample_key=None, report=report_path)
    assert report_path.is_file()
    assert out.n_obs == len(cell_barcodes)
    assert "<th>n_empty</th><td>50</td>" in report_path.read_text()


def test_denoise_raw_kwarg_rejects_duplicate_symbols_without_gene_ids():
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    names = list(filtered.var_names)
    names[0] = names[5]
    filtered.var_names = names

    with pytest.raises(ValueError, match="duplicated var_names"):
        amdose.denoise(filtered, raw=raw.copy(), sample_key=None)


def test_denoise_raw_kwarg_rejects_duplicate_raw_symbols_without_gene_ids():
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    names = list(raw.var_names)
    names[1] = names[10]
    raw.var_names = names

    with pytest.raises(ValueError, match="duplicated var_names"):
        amdose.denoise(filtered, raw=raw, sample_key=None)


def test_denoise_raw_kwarg_uses_raw_pool_counts_after_filtered_log1p():
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=70)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    expected_raw = filtered.X.copy()
    sc.pp.normalize_total(filtered, target_sum=1e4)
    sc.pp.log1p(filtered)

    out = amdose.denoise(filtered, raw=raw.copy(), sample_key=None)
    assert (out.layers["raw_counts"] != expected_raw).nnz == 0
    raw_values = out.layers["raw_counts"].data
    np.testing.assert_allclose(raw_values, np.rint(raw_values))


def test_denoise_raw_kwarg_requires_every_barcode_and_gene():
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=71)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    filtered_bad_barcode = filtered.copy()
    names = list(filtered_bad_barcode.obs_names)
    names[0] = "NOT_IN_RAW"
    filtered_bad_barcode.obs_names = names
    with pytest.raises(ValueError, match="every filtered barcode"):
        amdose.denoise(filtered_bad_barcode, raw=raw.copy(), sample_key=None)

    with pytest.raises(ValueError, match="every filtered feature"):
        amdose.denoise(filtered, raw=raw[:, :-1].copy(), sample_key=None)


def test_denoise_raw_kwarg_rejects_normalized_barcode_collision():
    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=72)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    names = list(filtered.obs_names)
    names[0] = "COLLIDE"
    names[1] = "COLLIDE-1"
    filtered.obs_names = names
    with pytest.raises(ValueError, match="non-injective"):
        amdose.denoise(filtered, raw=raw.copy(), sample_key=None)


def test_denoise_restores_scanpy_global_n_jobs():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=73)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    before = sc.settings.n_jobs
    amdose.denoise(
        adata,
        cell_barcodes=cells,
        cell_calling="off",
        type_key="cell_type",
        sample_key=None,
        n_jobs=7,
    )
    assert sc.settings.n_jobs == before


def test_denoise_raw_kwarg_puts_denoised_counts_in_x():
    """Unlike the traditional cell_barcodes= path (X left untouched), raw=
    should match analysis_ready()'s own convention: X is denoised, the
    original input moves to layers['raw_counts']. Requested by Lee
    directly (2026-08-30)."""
    from ambidose.pp import LAYER_OUT

    raw = make_toy(n_samples=1, n_empty=50, n_cells=40, seed=7)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    cell_barcodes = raw.obs_names[cell_mask].tolist()
    filtered = raw[cell_barcodes].copy()
    filtered_x_before = filtered.X.copy()

    out = amdose.denoise(filtered, raw=raw.copy(), sample_key=None)
    assert (out.layers[LAYER_OUT] != out.X).nnz == 0
    assert "raw_counts" in out.layers
    assert (out.layers["raw_counts"] != filtered_x_before).nnz == 0
    assert (out.layers["raw_counts"] < out.X).nnz == 0


def test_chi_storage_is_mutually_exclusive_between_modes():
    adata = make_toy(n_samples=2, n_empty=20, n_cells=10, seed=91)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"]
    amdose.estimate_chi(adata, sample_key=None)
    assert CHI_KEY in adata.var and CHI_KEY not in adata.uns
    amdose.estimate_chi(adata, sample_key="sample")
    assert CHI_KEY in adata.uns and CHI_KEY not in adata.var
    amdose.estimate_chi(adata, sample_key=None)
    assert CHI_KEY in adata.var and CHI_KEY not in adata.uns


def test_multisample_denoise_rejects_global_chi():
    adata = make_toy(n_samples=2, n_empty=20, n_cells=10, seed=92)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"]
    amdose.estimate_chi(adata, sample_key=None)
    with pytest.raises(ValueError, match="sample-specific profiles"):
        amdose.denoise(adata, sample_key="sample", type_key="cell_type")


def test_failed_denoise_does_not_mutate_existing_run_metadata():
    import copy

    adata, cells = _denoised_toy()
    before = copy.deepcopy(adata.uns["ambidose"])
    with pytest.raises(ValueError, match="already completed"):
        amdose.denoise(adata, cell_barcodes=cells, layer_out="foo", sample_key=None)
    assert adata.uns["ambidose"] == before


def test_summary_actual_removed_matches_count_difference():
    from ambidose.reporting import summarize

    adata, _ = _denoised_toy()
    result = summarize(adata)
    labels = adata.obs["ambidose_droplet"].astype(str).to_numpy()
    raw_total = np.asarray(adata.layers["raw_counts"].sum(axis=1)).ravel()
    den_total = np.asarray(adata.layers["ambidose_denoised"].sum(axis=1)).ravel()
    expected = float(np.median((raw_total - den_total)[labels == "cell"]))
    assert result["median_actual_removed_umi"] == expected
    assert "median_predicted_ambient_umi" in result
    assert "median_removed_umi" not in result


@pytest.mark.parametrize("mode", ["whitelist_off", "whitelist_refine", "diem", "empty_umi"])
def test_denoise_respects_custom_droplet_key(mode):
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=103)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    kwargs = {"droplet_key": "custom_status", "sample_key": None, "type_key": "cell_type"}
    if mode == "whitelist_off":
        kwargs.update(cell_barcodes=cells, cell_calling="off")
    elif mode == "whitelist_refine":
        kwargs.update(cell_barcodes=cells)
    elif mode == "diem":
        kwargs.update(cell_calling="diem")
    else:
        kwargs.update(empty_umi_max=100)
    amdose.denoise(adata, **kwargs)
    assert "custom_status" in adata.obs
    assert "ambidose_droplet" not in adata.obs


def test_denoise_passes_n_jobs_to_cell_calling(monkeypatch):
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=104)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    seen = {}
    original = amdose.pp.call_cells

    def wrapped(*args, **kwargs):
        seen["n_jobs"] = kwargs.get("n_jobs")
        return original(*args, **kwargs)

    monkeypatch.setattr(amdose.pp, "call_cells", wrapped)
    amdose.denoise(
        adata,
        cell_barcodes=cells,
        sample_key=None,
        type_key="cell_type",
        n_jobs=1,
    )
    assert seen["n_jobs"] == 1


def test_default_single_sample_api_needs_no_sample_column():
    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=105)
    del adata.obs["sample"]
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(adata, cell_barcodes=cells, type_key="cell_type")
    assert CHI_KEY in adata.var
    assert CHI_KEY not in adata.uns


def test_analysis_ready_rejects_raw_denoised_layer_collision():
    adata, _ = _denoised_toy()
    with pytest.raises(ValueError, match="raw_layer must differ"):
        amdose.analysis_ready(
            adata,
            raw_layer="ambidose_denoised",
            denoised_layer="ambidose_denoised",
        )


def test_analysis_ready_rejects_overwriting_existing_raw_layer():
    adata, _ = _denoised_toy()
    adata.layers["counts"] = adata.X.copy()
    before = adata.layers["counts"].copy()

    with pytest.raises(ValueError, match="already exists"):
        amdose.analysis_ready(adata, raw_layer="counts")

    assert (adata.layers["counts"] != before).nnz == 0


@pytest.mark.parametrize("missing_layer", ["raw_counts", "ambidose_denoised"])
def test_completed_run_rejects_rerun_when_artifact_is_missing(missing_layer):
    adata, _ = _denoised_toy()
    del adata.layers[missing_layer]
    with pytest.raises(RuntimeError, match="completed AmbiDose run"):
        amdose.denoise(adata, sample_key=None)


def test_raw_count_matrix_honors_empty_string_input_layer():
    from anndata import AnnData

    from ambidose.pp import raw_count_matrix

    adata = AnnData(sparse.csr_matrix([[0, 0]], dtype=int))
    adata.layers[""] = sparse.csr_matrix([[2, 3]], dtype=int)
    adata.uns["ambidose"] = {"input_layer": ""}

    np.testing.assert_array_equal(raw_count_matrix(adata).toarray(), [[2, 3]])


def test_completed_raw_snapshot_is_authoritative_over_input_layer():
    from ambidose.pp import raw_count_matrix

    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=106)
    adata.layers["counts"] = adata.X.copy()
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    amdose.denoise(
        adata,
        layer="counts",
        cell_barcodes=cells,
        cell_calling="off",
        sample_key=None,
        type_key="cell_type",
    )
    snapshot = adata.layers["raw_counts"].copy()
    adata.layers["counts"] = sparse.csr_matrix(adata.shape, dtype=adata.X.dtype)
    assert (raw_count_matrix(adata) != snapshot).nnz == 0


def test_invalid_sample_key_failure_is_object_state_atomic():
    import copy
    import pickle

    import pandas as pd

    adata = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=107)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    before_obs = adata.obs.copy(deep=True)
    before_uns = pickle.dumps(copy.deepcopy(adata.uns))
    before_layers = set(adata.layers)
    with pytest.raises(KeyError, match="does_not_exist"):
        amdose.denoise(
            adata,
            cell_barcodes=cells,
            sample_key="does_not_exist",
            type_key="cell_type",
        )
    pd.testing.assert_frame_equal(adata.obs, before_obs)
    assert pickle.dumps(adata.uns) == before_uns
    assert set(adata.layers) == before_layers


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_n": 0},
        {"quantile": -0.1},
        {"quantile": 1.1},
        {"soup_quantile": -0.1},
        {"min_genes": 0},
        {"min_valid": 0},
        {"min_chi": -1.0},
        {"max_type_mean": -1.0},
    ],
)
def test_estimate_dose_rejects_invalid_statistical_parameters(kwargs):
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=108)
    with pytest.raises(ValueError):
        amdose.estimate_dose(adata, **kwargs)


def test_raw_mode_replaces_stale_single_sample_namespace():
    import pandas as pd

    raw = make_toy(n_samples=1, n_empty=50, n_cells=30, seed=109)
    cells = raw.obs_names[raw.obs["droplet"].astype(str) == "cell"].tolist()
    filtered = raw[cells].copy()
    filtered.uns[CHI_KEY] = pd.DataFrame(
        [np.full(raw.n_vars, 1.0 / raw.n_vars)],
        index=["stale"],
        columns=raw.var_names,
    )
    filtered.obs["ambidose_fake_old_result"] = 123
    out = amdose.denoise(filtered, raw=raw.copy(), sample_key=None, type_key="cell_type")
    assert CHI_KEY in out.var
    assert CHI_KEY not in out.uns
    assert "ambidose_fake_old_result" not in out.obs


def test_raw_mode_replaces_stale_multisample_namespace():
    raw = make_toy(n_samples=2, n_empty=30, n_cells=20, seed=110)
    cells = raw.obs_names[raw.obs["droplet"].astype(str) == "cell"].tolist()
    filtered = raw[cells].copy()
    filtered.var[CHI_KEY] = np.full(raw.n_vars, 1.0 / raw.n_vars)
    out = amdose.denoise(filtered, raw=raw.copy(), sample_key="sample", type_key="cell_type")
    assert CHI_KEY not in out.var
    assert CHI_KEY in out.uns


def test_default_type_key_does_not_guess_from_stale_cluster_column():
    from ambidose.pp import _default_type_key

    adata = make_toy(n_samples=1, n_empty=10, n_cells=10, seed=111)
    adata.obs["ambidose_cluster"] = "old"
    assert _default_type_key(adata) is None


def test_estimate_dose_refreshes_n_umi_for_selected_layer():
    adata = make_toy(n_samples=1, n_empty=30, n_cells=20, seed=112)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    adata.layers["counts"] = adata.X.copy()
    amdose.estimate_chi(adata, sample_key=None, layer="counts")
    adata.obs["n_umi"] = -1.0
    amdose.estimate_dose(adata, sample_key=None, layer="counts")
    expected = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    np.testing.assert_array_equal(adata.obs["n_umi"].to_numpy(), expected)


def test_mixture_does_not_converge_while_native_profile_is_still_moving():
    from scipy import sparse as sp

    from ambidose.pp import _two_component_mixture_em

    counts = sp.coo_matrix(np.array([[0.0, 1.0, 1.0]]))
    ambient = np.array([3.0, 2.0, 4.0]) / 9.0
    native = np.array([4.0, 3.0, 4.0]) / 11.0
    rho, fitted_native, n_iter, converged, inferred = _two_component_mixture_em(
        coo=counts,
        n_cells=np.array([2.0]),
        n_vars=3,
        native=native,
        ambient=ambient,
        max_iter=500,
        convergence=1e-3,
        initial_rho=0.5,
        pseudocount=1e-8,
    )
    assert n_iter > 1
    assert converged
    assert rho[0] < 0.1
    assert np.isclose(fitted_native.sum(), 1.0)
    assert np.isclose(inferred.sum(), 1.0)


@pytest.mark.parametrize("kwargs", [{"convergence": 0}, {"pseudocount": 0}, {"pseudocount": -1}])
def test_mixture_rejects_invalid_numerical_parameters(kwargs):
    from ambidose.pp import estimate_dose_mixture

    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=113)
    with pytest.raises(ValueError):
        estimate_dose_mixture(adata, type_key="cell_type", **kwargs)


def test_denoise_rejects_dual_chi_state_before_mutation():
    import copy
    import pickle

    import pandas as pd

    adata = make_toy(n_samples=1, n_empty=30, n_cells=20, seed=114)
    adata.var[CHI_KEY] = np.full(adata.n_vars, 1.0 / adata.n_vars)
    adata.uns[CHI_KEY] = pd.DataFrame(
        [np.full(adata.n_vars, 1.0 / adata.n_vars)],
        index=["s0"],
        columns=adata.var_names,
    )
    before_obs = adata.obs.copy(deep=True)
    before_uns = pickle.dumps(copy.deepcopy(adata.uns))
    with pytest.raises(ValueError, match="ambiguous χ state"):
        amdose.denoise(adata, sample_key=None, type_key="cell_type")
    pd.testing.assert_frame_equal(adata.obs, before_obs)
    assert pickle.dumps(adata.uns) == before_uns


def test_raw_mode_rejects_conflicting_sample_assignments():
    raw = make_toy(n_samples=2, n_empty=30, n_cells=20, seed=113)
    cells = raw.obs_names[raw.obs["droplet"].astype(str) == "cell"].tolist()
    filtered = raw[cells].copy()
    filtered.obs.loc[filtered.obs_names[0], "sample"] = "wrong_sample"

    with pytest.raises(ValueError, match="sample assignments disagree"):
        amdose.denoise(filtered, raw=raw.copy(), sample_key="sample")


def test_raw_mode_copies_authoritative_sample_key_to_result():
    raw = make_toy(n_samples=2, n_empty=30, n_cells=20, seed=114)
    cells = raw.obs_names[raw.obs["droplet"].astype(str) == "cell"].tolist()
    expected = raw.obs.loc[cells, "sample"].astype(str).to_numpy()
    filtered = raw[cells].copy()
    del filtered.obs["sample"]

    out = amdose.denoise(filtered, raw=raw.copy(), sample_key="sample")

    np.testing.assert_array_equal(out.obs["sample"].astype(str), expected)
    assert out.uns["ambidose"]["sample_key"] == "sample"


def test_raw_mode_uses_internal_sentinel_for_noncell_types(monkeypatch):
    import ambidose.pp as pp
    from ambidose._shared import EMPTY_TYPE

    raw = make_toy(n_samples=1, n_empty=20, n_cells=15, seed=71)
    cell_mask = raw.obs["droplet"].astype(str).to_numpy() == "cell"
    filtered = raw[cell_mask].copy()
    original = pp.estimate_dose_adaptive
    seen = []

    def capture(target, **kwargs):
        noncell = target.obs[pp.DROPLET_KEY].astype(str).to_numpy() != "cell"
        seen.extend(target.obs.loc[noncell, "cell_type"].tolist())
        return original(target, **kwargs)

    monkeypatch.setattr(pp, "estimate_dose_adaptive", capture)
    amdose.denoise(filtered, raw=raw, type_key="cell_type", sample_key=None)
    assert seen
    assert all(value is EMPTY_TYPE for value in seen)


def test_report_failure_preserves_completed_subtraction(monkeypatch, tmp_path):
    adata = make_toy(n_samples=1, n_empty=20, n_cells=15, seed=72)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()

    def fail_report(*args, **kwargs):
        raise RuntimeError("report failed")

    monkeypatch.setattr("ambidose.reporting.write_report", fail_report)
    with pytest.raises(RuntimeError, match="report failed"):
        amdose.denoise(
            adata,
            cell_barcodes=cells,
            type_key="cell_type",
            sample_key=None,
            report=tmp_path / "report.html",
        )
    run = adata.uns.get("ambidose", {})
    assert run["subtraction_completed"] is True
    assert run["core_completed"] is True
    assert run["completed"] is True
    assert run["postprocess_completed"] is True
    assert run["report_completed"] is False
    ready = amdose.analysis_ready(adata)
    assert ready.n_obs > 0
    with pytest.raises(ValueError, match="already completed"):
        amdose.denoise(adata, sample_key=None)


@pytest.mark.parametrize(
    "missing", ["ambidose_rho", "ambidose_d", "raw_counts", "ambidose_denoised"]
)
def test_summarize_rejects_corrupted_completed_result(missing):
    adata, _ = _denoised_toy()
    if missing in adata.obs:
        del adata.obs[missing]
    else:
        del adata.layers[missing]

    with pytest.raises(RuntimeError, match="corrupted"):
        summarize(adata)


def test_summarize_excludes_missing_type_from_type_breakdown():
    adata = make_toy(n_samples=1, n_empty=5, n_cells=5, seed=73)
    adata.obs[DROPLET_KEY] = adata.obs["droplet"].astype(str)
    cells = adata.obs[DROPLET_KEY].astype(str).to_numpy() == "cell"
    adata.obs["reported_type"] = np.array([np.nan] * adata.n_obs, dtype=object)
    adata.obs.loc[adata.obs_names[cells][0], "reported_type"] = "known"
    adata.obs["ambidose_rho"] = 0.1
    adata.layers["ambidose_denoised"] = adata.X.copy()
    adata.uns["ambidose"] = {
        "dose_type_key": "reported_type",
        "droplet_key": DROPLET_KEY,
        "layer_out": "ambidose_denoised",
    }
    result = summarize(adata)
    assert result["median_rho_by_type"] == {"known": pytest.approx(0.1)}
