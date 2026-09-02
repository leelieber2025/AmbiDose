import pandas as pd
import scanpy as sc

from ambidose.cli import main
from ambidose.datasets import make_toy
from ambidose.io import write_h5ad


def _run(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        return 1


def test_version_exit_zero(capsys):
    assert _run(["--version"]) == 0
    assert "ambidose" in capsys.readouterr().out


def test_inspect_missing_input_exits_cleanly(capsys):
    assert _run(["inspect", "--input", "/no/such/ambidose-input.h5"]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_denoise_help_lists_cell_barcodes(capsys):
    assert _run(["denoise", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--cell-barcodes" in out
    assert "--root" in out
    assert "--full-typing" in out
    assert "--expected-cells" in out
    assert "--max-cells" in out
    assert "--n-jobs" in out
    assert "diem" in out
    assert "chi" in out
    assert "off" in out
    assert "emptydrops" in out


def test_estimate_chi_help_lists_cell_barcodes(capsys):
    assert _run(["estimate-chi", "--help"]) == 0
    assert "--cell-barcodes" in capsys.readouterr().out


def test_denoise_cli_mtx_requires_cells_only():
    assert (
        _run(
            [
                "denoise",
                "--input",
                "/tmp/raw.h5ad",
                "--output",
                "/tmp/out",
                "--output-format",
                "10x-mtx",
            ]
        )
        == 1
    )


def test_denoise_cli_off_without_barcodes_exits(tmp_path):
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=3)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--output",
                str(tmp_path / "out.h5ad"),
                "--cell-calling",
                "off",
            ]
        )
        == 1
    )


def test_denoise_cli_without_whitelist_uses_diem(tmp_path):
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=3)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert out.is_file()
    result = sc.read_h5ad(out)
    assert result.uns["ambidose"]["cell_calling"]["method"] == "diem"


def test_denoise_cli_expect_cells_without_cell_calling_uses_ordmag(tmp_path):
    # --expected-cells with no explicit --cell-calling must resolve to ordmag
    # (the CLI's own default used to be the literal string "diem", which
    # always overrode the intended ordmag default -- see cell_calling
    # resolution in _denoise()).
    adata = make_toy(n_samples=1, n_empty=20, n_cells=10, seed=3)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--expected-cells",
                "8",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    result = sc.read_h5ad(out)
    assert result.uns["ambidose"]["cell_calling"]["method"] == "ordmag"


def test_denoise_root_rejects_cell_barcodes():
    assert (
        _run(
            [
                "denoise",
                "--root",
                "/tmp",
                "--cell-barcodes",
                "/tmp/bc.tsv",
                "--output",
                "/tmp/out.h5ad",
            ]
        )
        == 1
    )


def test_denoise_cli_uses_cell_barcodes(tmp_path):
    adata = make_toy(n_samples=1, n_empty=40, n_cells=20, seed=5)
    true_cells = adata.obs_names[adata.obs["droplet"].to_numpy() == "cell"]
    keep = list(true_cells[:10])
    inp = tmp_path / "raw.h5ad"
    bc = tmp_path / "filtered_barcodes.tsv"
    out = tmp_path / "cleaned.h5ad"
    write_h5ad(adata, inp)
    bc.write_text("\n".join(keep) + "\n")
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--cell-barcodes",
                str(bc),
                "--type-key",
                "cell_type",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    cleaned = sc.read_h5ad(out)
    lab = cleaned.obs["ambidose_droplet"].astype(str)
    assert (lab.loc[keep] == "cell").all()
    leftover = true_cells[10:]
    assert (lab.loc[leftover] != "cell").all()
    assert "ambidose_denoised" in cleaned.layers
    assert "ambidose_d" in cleaned.obs


def test_estimate_chi_multi_sample_csv_keeps_sample_names(tmp_path, capsys):
    adata = make_toy(n_samples=2, n_empty=80, n_cells=30, seed=8)
    adata.obs["ambidose_droplet"] = adata.obs["droplet"]
    inp = tmp_path / "raw.h5ad"
    out = tmp_path / "chi.csv"
    write_h5ad(adata, inp)
    rc = _run(
        [
            "estimate-chi",
            "--input",
            str(inp),
            "--sample-key",
            "sample",
            "--empty-umi-max",
            "80",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "-- sample 's0' --" in printed
    assert "-- sample 's1' --" in printed
    df = pd.read_csv(out, index_col=0)
    assert set(df.index.astype(str)) == {"s0", "s1"}


def test_explicit_missing_sample_key_fails_instead_of_using_global_chi(tmp_path):
    adata = make_toy(n_samples=2, n_empty=40, n_cells=20, seed=44)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)

    assert (
        _run(
            [
                "estimate-chi",
                "--input",
                str(inp),
                "--sample-key",
                "sampel",
                "--empty-umi-max",
                "80",
            ]
        )
        == 1
    )


def _spy_call_cells_lower(monkeypatch, seen):
    from ambidose.pp import call_cells as real

    def spy(*args, **kwargs):
        seen.append(kwargs.get("lower"))
        return real(*args, **kwargs)

    monkeypatch.setattr("ambidose.pp.call_cells", spy)


def test_denoise_cli_manifest_passes_empty_umi_max_as_lower(tmp_path, monkeypatch):
    seen = []
    _spy_call_cells_lower(monkeypatch, seen)
    adata = make_toy(n_samples=1, n_empty=40, n_cells=12, seed=31)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    raw = tmp_path / "L1.h5ad"
    whitelist = tmp_path / "L1.tsv"
    write_h5ad(adata, raw)
    whitelist.write_text("\n".join(cells) + "\n")
    manifest = tmp_path / "libraries.tsv"
    manifest.write_text(f"library\traw_counts\tcell_barcodes\nL1\t{raw.name}\t{whitelist.name}\n")
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--manifest",
                str(manifest),
                "--type-key",
                "cell_type",
                "--empty-umi-max",
                "50",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert seen == [50]


def test_denoise_cli_root_passes_empty_umi_max_as_lower(tmp_path, monkeypatch):
    from ambidose.io import write_10x_mtx

    seen = []
    _spy_call_cells_lower(monkeypatch, seen)
    adata = make_toy(n_samples=1, n_empty=40, n_cells=12, seed=34)
    cells = adata.obs_names[adata.obs["droplet"].astype(str) == "cell"].tolist()
    sample_dir = tmp_path / "S1"
    write_10x_mtx(adata, sample_dir / "raw_feature_bc_matrix")
    filt = sample_dir / "filtered_feature_bc_matrix"
    filt.mkdir()
    (filt / "barcodes.tsv").write_text("\n".join(cells) + "\n")
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--root",
                str(tmp_path),
                "--empty-umi-max",
                "50",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert seen == [50]


def test_denoise_help_lists_mark_doublets(capsys):
    assert _run(["denoise", "--help"]) == 0
    assert "--mark-doublets" in capsys.readouterr().out


def test_denoise_cli_mark_doublets_writes_type_residual(tmp_path):
    adata = make_toy(n_samples=1, n_empty=60, n_cells=80, seed=13)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--type-key",
                "cell_type",
                "--mark-doublets",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    result = sc.read_h5ad(out)
    assert "ambidose_doublet_score" in result.obs
    assert "ambidose_type_residual" in result.obs


def test_denoise_cli_without_mark_doublets_skips_it(tmp_path):
    adata = make_toy(n_samples=1, n_empty=60, n_cells=80, seed=13)
    inp = tmp_path / "raw.h5ad"
    write_h5ad(adata, inp)
    out = tmp_path / "out.h5ad"
    assert (
        _run(
            [
                "denoise",
                "--input",
                str(inp),
                "--type-key",
                "cell_type",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    result = sc.read_h5ad(out)
    assert "ambidose_doublet_score" not in result.obs
    assert "ambidose_type_residual" not in result.obs
