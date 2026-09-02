"""Command-line entry: ``ambidose``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ambidose",
        description="Per-cell ambient dose removal using empty droplets for χ.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_inspect = sub.add_parser("inspect", help="Validate input and cell-whitelist matching")
    p_inspect.add_argument("--input")
    p_inspect.add_argument("--manifest")
    p_inspect.add_argument("--cell-barcodes", default=None)
    p_inspect.add_argument("--empty-umi-max", type=int, default=100)
    p_inspect.add_argument("--output-json", default=None)

    p_est = sub.add_parser("estimate-chi", help="SoupX-style χ from empty droplets")
    p_est.add_argument(
        "--input",
        required=True,
        help="raw .h5, .h5ad, MTX directory, or Cell Ranger outs directory",
    )
    p_est.add_argument("--output", help="CSV path for the ambient profile")
    p_est.add_argument(
        "--empty-umi-max",
        type=int,
        default=None,
        help="SoupX-style empty upper bound (default 100 if neither this nor --expected-cells)",
    )
    p_est.add_argument("--empty-umi-min", type=int, default=0)
    p_est.add_argument("--expected-cells", type=int, default=None)
    p_est.add_argument(
        "--cell-barcodes",
        default=None,
        help=(
            "cell whitelist TSV or filtered HDF5 (default: matching Cell Ranger filtered "
            "barcodes; trusted external caller accepted as fallback)"
        ),
    )
    p_est.add_argument("--sample-key", default=None)
    p_est.add_argument("--top", type=int, default=20, help="print top ambient genes")

    p_den = sub.add_parser("denoise", help="Estimate χ and per-cell dose, subtract d·χ")
    p_den.add_argument("--input", help="raw .h5 / .h5ad / mtx directory")
    p_den.add_argument("--root", help="directory of 10x sample folders")
    p_den.add_argument("--manifest", help="TSV: library, raw_counts, cell_barcodes")
    p_den.add_argument("--output", required=True)
    p_den.add_argument(
        "--output-format",
        choices=["h5ad", "10x-mtx"],
        default="h5ad",
        help="h5ad keeps raw X plus a denoised layer, all droplets, unless "
        "--cells-only. 10x-mtx cannot store layers: requires --cells-only",
    )
    p_den.add_argument(
        "--report",
        default=None,
        help="self-contained HTML QC report path (default: <stem>_report.html "
        "next to --output for h5ad, or ambidose_report.html inside the "
        "--output directory for 10x-mtx; pass --report off to skip it)",
    )
    p_den.add_argument("--summary-json", default=None)
    p_den.add_argument(
        "--cells-only",
        action="store_true",
        help="write only called cells with the denoised layer as X (via "
        "analysis_ready()), instead of the full object with raw X plus a "
        "denoised layer",
    )
    p_den.add_argument("--empty-umi-max", type=int, default=100)
    p_den.add_argument(
        "--expected-cells",
        type=int,
        default=None,
        help="call cells with Cell Ranger OrdMag (or --cell-calling force) "
        "when the filtered whitelist is missing or over-called",
    )
    p_den.add_argument(
        "--cell-calling",
        choices=["diem", "chi", "emptydrops", "ordmag", "force", "off"],
        default=None,
        help="unset (default): with a whitelist available (explicit "
        "--cell-barcodes, auto-detected Cell Ranger filtered barcodes, or a "
        "manifest/root library), refines it against empty-droplet χ -- same "
        "as passing chi explicitly. With --expected-cells and no whitelist, "
        "calls cells with Cell Ranger OrdMag. With neither, builds our own "
        "empty/debris/cell mixture whitelist (same as passing diem "
        "explicitly). "
        "off: trust the whitelist as-is, do not refine or call cells. "
        "emptydrops / ordmag / force: Lun 2019 or Cell Ranger step 1, only "
        "meaningful without a whitelist",
    )
    p_den.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="cap called cells (chip capacity, e.g. 20000 on a 20k kit)",
    )
    p_den.add_argument(
        "--cell-barcodes",
        default=None,
        help=(
            "cell whitelist TSV or filtered HDF5 for single-sample --input (default: matching "
            "Cell Ranger filtered barcodes; trusted external caller accepted "
            "as fallback)"
        ),
    )
    p_den.add_argument("--sample-key", default=None)
    p_den.add_argument("--type-key", dest="type_key", default=None)
    p_den.add_argument(
        "--full-typing",
        action="store_true",
        help="use the full-cell coarse graph (default: faster graph at >=20k cells)",
    )
    p_den.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="worker threads for typing and structure regression (default: "
        "auto-detect from usable host CPUs and available RAM)",
    )
    p_den.add_argument(
        "--mark-doublets",
        action="store_true",
        help="flag predicted doublets among called cells (scanpy Scrublet) and, "
        "when typing is available, add a heterotypic doublet-vs-soup QC score "
        "(ambidose_type_residual, leave-one-type residual vs chi shape); "
        "diagnostic only, does not change rho/dose or the output cell count "
        "(default: off)",
    )

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    try:
        if args.cmd == "inspect":
            return _inspect(args)
        if args.cmd == "estimate-chi":
            return _estimate_chi(args)
        if args.cmd == "denoise":
            return _denoise(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    parser.error(f"unknown command {args.cmd}")
    return 2


def _inspect(args: argparse.Namespace) -> int:
    import json

    from .io import read_manifest
    from .reporting import _aggregate_status, inspect_input, write_inspection_json

    if sum(value is not None for value in (args.input, args.manifest)) != 1:
        raise SystemExit("pass exactly one of --input or --manifest")
    if args.manifest:
        libraries = {
            row.library: inspect_input(
                row.raw_counts,
                cell_barcodes=row.cell_barcodes,
                empty_umi_max=args.empty_umi_max,
            )
            for row in read_manifest(args.manifest)
        }
        report = {
            "kind": "manifest",
            "input": str(args.manifest),
            "n_libraries": len(libraries),
            "libraries": libraries,
            "status": _aggregate_status(lib["status"] for lib in libraries.values()),
        }
    else:
        report = inspect_input(
            args.input, cell_barcodes=args.cell_barcodes, empty_umi_max=args.empty_umi_max
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        write_inspection_json(report, args.output_json)
    return 1 if report["status"] == "error" else 0


def _load(path: Path):
    import scanpy as sc

    from .io import read_10x_h5, read_10x_mtx, sniff_input

    resolved = sniff_input(path)
    if resolved.kind == "root":
        raise SystemExit(f"{path} contains multiple samples; pass it with --root")
    if resolved.kind == "h5ad":
        adata = sc.read_h5ad(resolved.raw)
    elif resolved.kind == "mtx":
        adata = read_10x_mtx(resolved.raw)
    else:
        adata = read_10x_h5(resolved.raw)
    return adata, resolved.filtered_barcodes


def _resolve_cli_sample_key(adata, requested: str | None) -> str | None:
    """Auto-detect the conventional column, but reject an explicit typo."""
    if requested is not None:
        if requested not in adata.obs.columns:
            raise SystemExit(f"--sample-key {requested!r} is not present in obs")
        return requested
    return "sample" if "sample" in adata.obs.columns else None


def _estimate_chi(args: argparse.Namespace) -> int:
    import pandas as pd

    from .pp import classify_droplets, estimate_chi

    adata, detected_barcodes = _load(Path(args.input))
    empty_max = args.empty_umi_max
    cell_bc = _read_barcode_file(args.cell_barcodes or detected_barcodes)
    if empty_max is None and args.expected_cells is None and cell_bc is None:
        empty_max = 100
    classify_droplets(
        adata,
        empty_umi_max=empty_max,
        empty_umi_min=args.empty_umi_min,
        expected_cells=args.expected_cells,
        cell_barcodes=cell_bc,
    )
    from .pp import CHI_KEY

    sample_key = _resolve_cli_sample_key(adata, args.sample_key)
    chi = estimate_chi(adata, sample_key=sample_key)
    # Multi-sample: estimate_chi already built and stored the properly
    # sample-indexed DataFrame in uns[CHI_KEY] -- the bare ndarray it also
    # returns has no sample names at all, so CSV output built from that
    # alone had 0/1/2/... row labels and no way to tell which row was which
    # sample; use the indexed version whenever it exists.
    chi_df = adata.uns.get(CHI_KEY) if chi.ndim == 2 else None
    _print_chi(adata, chi, top=args.top, chi_df=chi_df)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if chi_df is not None:
            chi_df.to_csv(out)
        elif chi.ndim == 1:
            pd.Series(chi, index=adata.var_names.astype(str), name="chi").to_csv(out)
        else:
            pd.DataFrame(chi, columns=adata.var_names.astype(str)).to_csv(out)
        print(out)
    return 0


def _print_chi(adata, chi, *, top: int, chi_df=None) -> None:
    import pandas as pd

    from .pp import DROPLET_KEY

    if DROPLET_KEY not in adata.obs:
        raise SystemExit(f"{DROPLET_KEY!r} missing after classify_droplets")
    n_empty = int((adata.obs[DROPLET_KEY] == "empty").sum())
    n_cell = int((adata.obs[DROPLET_KEY] == "cell").sum())
    print(f"barcodes: {adata.n_obs}  genes: {adata.n_vars}")
    print(f"empty: {n_empty}  cell-like: {n_cell}")
    if chi.ndim == 1:
        s = pd.Series(chi, index=adata.var_names.astype(str), name="chi")
        print(s.sort_values(ascending=False).head(top).to_string())
        if "genome" in adata.var.columns:
            g = adata.var["genome"].astype(str)
            for name in sorted(g.unique()):
                frac = float(s[g.to_numpy() == name].sum())
                print(f"chi mass {name}: {frac:.3f}")
    elif chi_df is not None:
        # Multi-sample: this branch used to print nothing at all (the
        # top-gene/chi-mass summary above only ever ran for chi.ndim==1).
        print(f"samples: {len(chi_df.index)}")
        for name in chi_df.index.astype(str):
            s = chi_df.loc[name].astype(float)
            top_genes = s.sort_values(ascending=False).head(top)
            print(f"-- sample {name!r} --")
            print(top_genes.to_string())
            if "genome" in adata.var.columns:
                g = adata.var["genome"].reindex(s.index).astype(str)
                for gname in sorted(g.unique()):
                    frac = float(s[g.to_numpy() == gname].sum())
                    print(f"  chi mass {gname}: {frac:.3f}")


def _read_barcode_file(path: str | None) -> list[str] | None:
    if path is None:
        return None
    from .io import read_10x_barcodes

    return read_10x_barcodes(path)


def _feature_ids(adata):
    values = (
        adata.var["gene_ids"].astype(str)
        if "gene_ids" in adata.var.columns
        else adata.var_names.astype(str)
    )
    if not values.is_unique:
        raise ValueError("library feature IDs must be unique")
    return values


def _require_identical_features(blocks, names):
    reference = _feature_ids(blocks[0])
    reference_set = set(reference)
    for name, block in zip(names[1:], blocks[1:], strict=True):
        current = _feature_ids(block)
        if not current.equals(reference):
            current_set = set(current)
            raise ValueError(
                f"library {name!r} has a different feature schema: "
                f"missing={len(reference_set - current_set)}, "
                f"additional={len(current_set - reference_set)}, "
                f"order_mismatch={reference_set == current_set}"
            )


def _denoise_root(args: argparse.Namespace):
    import pandas as pd
    from anndata import concat

    from .io import find_10x_mtx_samples, find_filtered_barcodes, read_10x_barcodes, read_10x_mtx
    from .pp import CHI_KEY, call_cells, classify_droplets, estimate_chi

    sample_key = args.sample_key or "sample"
    found = find_10x_mtx_samples(args.root)
    if not found:
        raise SystemExit(f"no 10x mtx samples under {args.root}")
    blocks = []
    chi_map = {}
    for name, path in found.items():
        print(f"read {name}")
        ad = read_10x_mtx(path)
        bc = find_filtered_barcodes(path)
        if bc is not None:
            print(f"  auto-detected filtered barcodes at {bc}")
            whitelist = read_10x_barcodes(bc)
            # Refined against ambient chi by default; --cell-calling off is
            # the only way to trust the Cell Ranger whitelist as-is.
            if str(args.cell_calling).strip().lower() != "off":
                print("  refining whitelist against empty-droplet χ", flush=True)
                whitelist = call_cells(
                    ad,
                    method="chi",
                    cell_barcodes=whitelist,
                    max_cells=args.max_cells,
                    lower=args.empty_umi_max if args.empty_umi_max is not None else 100,
                )
            classify_droplets(
                ad,
                empty_umi_max=args.empty_umi_max,
                cell_barcodes=whitelist,
            )
        else:
            raise SystemExit(
                f"{path}: no filtered barcodes; denoise --root needs Cell Ranger "
                "filtered lists (or pass each library through --input --cell-barcodes). "
                "UMI-threshold cell calling is not a product path"
            )
        estimate_chi(ad, sample_key=None)
        keep = ad.obs["ambidose_droplet"].astype(str).isin(["empty", "cell"])
        ad = ad[keep].copy()
        ad.obs[sample_key] = name
        chi_map[name] = pd.Series(ad.var[CHI_KEY].to_numpy(), index=ad.var_names.astype(str))
        blocks.append(ad)
        print(f"  empty+cells={ad.n_obs}")
    _require_identical_features(blocks, list(found))
    names = list(found)
    adata = concat(
        blocks,
        axis=0,
        join="inner",
        merge="same",
        label=sample_key,
        keys=names,
        index_unique="-",
    )
    adata.uns[CHI_KEY] = pd.DataFrame(
        {k: v.reindex(adata.var_names.astype(str)) for k, v in chi_map.items()}
    ).T
    if CHI_KEY in adata.var.columns:
        del adata.var[CHI_KEY]
    return adata


def _denoise_manifest(args: argparse.Namespace):
    import pandas as pd
    from anndata import concat

    from .io import read_manifest
    from .pp import CHI_KEY, call_cells, classify_droplets, estimate_chi

    sample_key = args.sample_key or "sample"
    rows = read_manifest(args.manifest)
    blocks = []
    chi_map = {}
    for number, row in enumerate(rows, start=1):
        print(f"[{number}/{len(rows)}] library {row.library}: reading raw counts")
        adata, detected = _load(row.raw_counts)
        source = row.cell_barcodes or detected
        if source is None:
            raise SystemExit(
                f"{row.library}: no cell whitelist on the manifest or next to the "
                "raw matrix; UMI-threshold cell calling is not a product path"
            )
        whitelist = _read_barcode_file(source)
        # Refined against ambient chi by default; --cell-calling off is the
        # only way to trust the manifest/auto-detected whitelist as-is.
        if str(args.cell_calling).strip().lower() != "off":
            print(f"[{number}/{len(rows)}] refining whitelist against empty-droplet χ", flush=True)
            whitelist = call_cells(
                adata,
                method="chi",
                cell_barcodes=whitelist,
                max_cells=args.max_cells,
                lower=args.empty_umi_max if args.empty_umi_max is not None else 100,
            )
        classify_droplets(
            adata,
            empty_umi_max=args.empty_umi_max,
            cell_barcodes=whitelist,
        )
        estimate_chi(adata, sample_key=None)
        keep = adata.obs["ambidose_droplet"].astype(str).isin(["empty", "cell"])
        adata = adata[keep].copy()
        adata.obs[sample_key] = row.library
        chi_map[row.library] = pd.Series(
            adata.var[CHI_KEY].to_numpy(), index=adata.var_names.astype(str)
        )
        blocks.append(adata)
    _require_identical_features(blocks, [row.library for row in rows])
    combined = concat(
        blocks,
        axis=0,
        join="inner",
        merge="same",
        label=sample_key,
        keys=[row.library for row in rows],
        index_unique="-",
    )
    combined.uns[CHI_KEY] = pd.DataFrame(
        {key: value.reindex(combined.var_names.astype(str)) for key, value in chi_map.items()}
    ).T
    if CHI_KEY in combined.var.columns:
        del combined.var[CHI_KEY]
    return combined


def _print_run_summary(data: dict) -> None:
    """Print post-processing results not already shown by denoise()."""
    removed = data.get("total_removed_fraction")
    median_removed = data.get("median_actual_removed_umi")
    if removed is not None:
        print(
            f"ambidose: removed {100.0 * removed:.1f}% of cell UMIs "
            f"(median {median_removed:.1f} UMI per cell)",
            file=sys.stderr,
        )
    if data.get("doublet_marking_run"):
        print(
            f"ambidose: doublet QC: Scrublet={data.get('n_doublet_scrublet')}, "
            f"type-residual={data.get('n_type_residual_doublet_leaning')}, "
            f"soup-like={data.get('n_type_residual_soup_leaning')}",
            file=sys.stderr,
        )


def _denoise(args: argparse.Namespace) -> int:
    from .io import write_10x_mtx, write_h5ad
    from .pp import analysis_ready, denoise, mark_doublets, require_run_keys
    from .reporting import summarize, write_report, write_summary_json

    modes = sum(value is not None for value in (args.input, args.root, args.manifest))
    if modes != 1:
        raise SystemExit("pass exactly one of --input, --root, or --manifest")
    if (args.root or args.manifest) and args.cell_barcodes:
        raise SystemExit("--cell-barcodes is for single-sample --input")
    if args.cell_barcodes and args.expected_cells:
        raise SystemExit("pass only one of --cell-barcodes or --expected-cells")
    if args.output_format == "10x-mtx" and not args.cells_only:
        raise SystemExit(
            "10x-mtx writes the denoised cell matrix as X (no layers). "
            "Pass --cells-only, or use --output-format h5ad"
        )
    # None means "not specified"; denoise() already resolves that correctly
    # per context (ordmag with --expected-cells, chi-refine with a whitelist)
    # -- only the no-whitelist/no-expected-cells case below must be resolved
    # to an explicit "diem" here, or denoise() falls through to its
    # smoke-only empty_umi_max path instead of the diem mixture model.
    cell_calling = args.cell_calling
    if args.manifest:
        adata = _denoise_manifest(args)
        sample_key = args.sample_key or "sample"
        detected_barcodes = None
    elif args.root:
        adata = _denoise_root(args)
        sample_key = args.sample_key or "sample"
        detected_barcodes = None
    else:
        print("[1/6] reading input")
        adata, detected_barcodes = _load(Path(args.input))
        src = args.cell_barcodes or detected_barcodes
        if args.expected_cells is not None:
            print(
                f"[2/6] calling cells with {cell_calling or 'ordmag'} "
                f"expected_cells={args.expected_cells}"
                + (f" max_cells={args.max_cells}" if args.max_cells else "")
            )
            detected_barcodes = None
        elif cell_calling == "off":
            if src is None:
                raise SystemExit(
                    "--cell-calling off needs --cell-barcodes or a Cell Ranger "
                    "outs/ directory with filtered barcodes"
                )
            print(f"[2/6] using external whitelist {src} (--cell-calling off)")
        elif src is not None:
            # A whitelist (explicit or auto-detected) is refined against
            # ambient chi by default; only --cell-calling off (above) skips
            # that and trusts it as-is.
            label = (
                "external whitelist"
                if args.cell_barcodes is not None
                else "auto-detected filtered barcodes"
            )
            print(f"[2/6] using {label} {src}")
            print("[2/6] refining whitelist against empty-droplet χ", flush=True)
        elif cell_calling in ("diem", "emptydrops") or cell_calling is None:
            cell_calling = cell_calling or "diem"
            print(f"[2/6] calling cells with {cell_calling}", flush=True)
            detected_barcodes = None
        elif args.cell_barcodes is None:
            raise SystemExit(
                "pass --cell-barcodes, --cell-calling off (with Cell Ranger "
                "filtered barcodes), --expected-cells, or --cell-calling diem"
            )
        sample_key = _resolve_cli_sample_key(adata, args.sample_key)
    print("[3-5/6] classifying droplets, estimating dose, and subtracting ambient counts")
    denoise(
        adata,
        sample_key=sample_key,
        empty_umi_max=args.empty_umi_max,
        cell_barcodes=(
            _read_barcode_file(args.cell_barcodes or detected_barcodes) if args.input else None
        ),
        expect_cells=args.expected_cells,
        cell_calling=cell_calling,
        max_cells=args.max_cells,
        type_key=args.type_key,
        typing_fast=not args.full_typing,
        n_jobs=args.n_jobs,
    )
    if args.mark_doublets:
        from .pp import CLUSTER_KEY as _CLUSTER_KEY
        from .pp import _default_type_key

        type_col = _default_type_key(adata)
        if type_col == _CLUSTER_KEY:
            type_col = None
        print(
            "[6/6] marking doublets"
            + (f" (type residual via {type_col!r})" if type_col else " (no typing; Scrublet only)"),
            flush=True,
        )
        mark_doublets(adata, type_key=type_col, sample_key=sample_key, layer="raw_counts")
    report_path = args.report
    if report_path is None:
        out_path = Path(args.output)
        report_path = str(
            out_path / "ambidose_report.html"
            if args.output_format == "10x-mtx"
            else out_path.with_name(out_path.stem + "_report.html")
        )
    if str(report_path).strip().lower() != "off":
        write_report(adata, report_path)
        print(f"QC report: {report_path}", file=sys.stderr)
        print(report_path)
    _print_run_summary(summarize(adata))
    if args.summary_json:
        write_summary_json(adata, args.summary_json)
        print(args.summary_json)
    print("[6/6] writing output")
    require_run_keys(adata)
    if args.output_format == "10x-mtx":
        write_10x_mtx(analysis_ready(adata), args.output, sample_key=sample_key)
    else:
        output = analysis_ready(adata) if args.cells_only else adata
        write_h5ad(output, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
