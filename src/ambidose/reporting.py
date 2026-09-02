"""Inspection, summaries, and self-contained HTML reports."""

from __future__ import annotations

import json
from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._shared import EMPTY_TYPES, _validated_type_values
from .io import normalize_barcode, read_10x_barcodes, read_10x_h5, read_10x_mtx, sniff_input


def _stable_top_n(values: np.ndarray, names, n: int) -> np.ndarray:
    """Indices of the top-``n`` values, descending; ties broken by name so
    the result doesn't depend on incidental gene order in the object."""
    names_arr = np.asarray(names, dtype=str)
    order = np.lexsort((names_arr, -np.asarray(values, dtype=float)))
    return order[:n]


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return {
            "index": [str(x) for x in value.index],
            "columns": [str(x) for x in value.columns],
            "data": value.to_numpy().tolist(),
        }
    if isinstance(value, pd.Series):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize(adata: AnnData, *, layer: str | None = None) -> dict:
    """Return a stable, JSON-serializable result summary."""
    from .pp import require_run_keys

    key, layer_out = require_run_keys(adata)
    if layer is None:
        layer = layer_out
    run = adata.uns.get("ambidose", {})
    is_completed = isinstance(run, dict) and (
        run.get("subtraction_completed") is True
        or run.get("core_completed") is True
        or run.get("completed") is True
    )
    if is_completed:
        missing_obs = [
            name for name in ("ambidose_rho", "ambidose_d") if name not in adata.obs.columns
        ]
        missing_layers = [name for name in ("raw_counts", layer_out) if name not in adata.layers]
        if missing_obs or missing_layers:
            raise RuntimeError(
                "completed AmbiDose object is corrupted: "
                f"missing obs columns={missing_obs}, missing layers={missing_layers}"
            )
    if layer not in adata.layers:
        raise KeyError(f"summary layer={layer!r} missing from adata.layers")
    labels = adata.obs[key]
    counts = labels.astype(str).value_counts().to_dict()
    is_cell = labels.astype(str).to_numpy() == "cell"
    rho = np.asarray(adata.obs.get("ambidose_rho", np.zeros(adata.n_obs)), dtype=float)
    dose = np.asarray(adata.obs.get("ambidose_d", np.zeros(adata.n_obs)), dtype=float)
    out = {
        "n_droplets": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_cells": int(counts.get("cell", int(is_cell.sum()))),
        "n_empty": int(counts.get("empty", 0)),
        "n_other": int(counts.get("other", 0)),
        "n_doublet": int(counts.get("doublet", 0)),
        "median_rho": float(np.median(rho[is_cell])) if is_cell.any() else None,
        "median_predicted_ambient_umi": float(np.median(dose[is_cell])) if is_cell.any() else None,
    }
    trust = adata.obs.get("ambidose_rho_trust")
    if trust is not None:
        trust_s = trust.astype(str).to_numpy()
        cell_trust = trust_s[is_cell]
        out["n_trust_ok"] = int((cell_trust == "ok").sum())
        out["n_trust_low_evidence"] = int((cell_trust == "low_evidence").sum())
        out["n_trust_ceiling_risk"] = int((cell_trust == "ceiling_risk").sum())
        out["n_trust_type_structure_risk"] = int((cell_trust == "type_structure_risk").sum())
        out["n_trust_under_execution"] = int((cell_trust == "under_execution").sum())
        out["n_trust_over_removal"] = int((cell_trust == "over_removal").sum())
        out["rho_trust_prompt"] = (
            "QC flags apply to quantitative interpretation of ambidose_rho; "
            "they are not cell-filtering recommendations."
        )
    run = adata.uns.get("ambidose", {})
    removal = run.get("removal", {}) if isinstance(run, dict) else {}
    ratio_percentiles = removal.get("execution_ratio_percentiles", {})
    if ratio_percentiles:
        out["dose_execution_ratio_percentiles"] = {
            key: float(value) for key, value in ratio_percentiles.items()
        }
    dose_meta = run.get("dose", {}) if isinstance(run, dict) else {}
    selection = dose_meta.get("selection", {}) if isinstance(dose_meta, dict) else {}
    if "n_disagreement" in selection:
        out["n_dose_disagreement"] = int(selection["n_disagreement"])
    dose_type_key = adata.uns.get("ambidose", {}).get("dose_type_key")
    if dose_type_key in adata.obs.columns and is_cell.any():
        validated = _validated_type_values(adata, dose_type_key).to_numpy()
        valid_cell = is_cell & np.array([t not in EMPTY_TYPES for t in validated])
        types = validated[valid_cell]
        rho_cell = rho[valid_cell]
        out["median_rho_by_type"] = {
            str(t): float(np.median(rho_cell[types == t])) for t in pd.unique(types)
        }
    if layer in adata.layers:
        from .pp import raw_count_matrix

        raw = raw_count_matrix(adata)
        den = (
            adata.layers[layer].tocsr()
            if sparse.issparse(adata.layers[layer])
            else sparse.csr_matrix(adata.layers[layer])
        )
        raw_total = float(raw[is_cell].sum())
        removed = float(raw[is_cell].sum() - den[is_cell].sum())
        raw_per_cell = np.asarray(raw.sum(axis=1)).ravel().astype(float)
        den_per_cell = np.asarray(den.sum(axis=1)).ravel().astype(float)
        removed_per_cell = raw_per_cell - den_per_cell
        retention = np.ones(adata.n_obs, dtype=float)
        positive_raw = raw_per_cell > 0
        retention[positive_raw] = den_per_cell[positive_raw] / raw_per_cell[positive_raw]
        cell_retention = retention[is_cell]
        out["cell_umi_retention_percentiles"] = (
            dict(
                zip(
                    ("p1", "p5", "median", "p95", "p99"),
                    np.percentile(cell_retention, [1, 5, 50, 95, 99]).tolist(),
                    strict=True,
                )
            )
            if cell_retention.size
            else {}
        )
        out["n_cells_below_50pct_umi_retention"] = int((cell_retention < 0.5).sum())
        out["median_actual_removed_umi"] = (
            float(np.median(removed_per_cell[is_cell])) if is_cell.any() else None
        )
        out["total_removed_umi"] = removed
        out["total_removed_fraction"] = removed / raw_total if raw_total else 0.0
        out["n_inflated"] = int((den > raw).nnz)
        gene_removed = (
            adata.var["ambidose_removed_umi"].to_numpy(dtype=float)
            if "ambidose_removed_umi" in adata.var
            else np.asarray(raw[is_cell].sum(axis=0) - den[is_cell].sum(axis=0)).ravel()
        )
        top = _stable_top_n(gene_removed, adata.var_names, 20)
        out["top_removed_genes"] = [
            {"gene": str(adata.var_names[i]), "removed_umi": int(gene_removed[i])}
            for i in top
            if gene_removed[i] > 0
        ]
    chi = adata.var.get("ambidose_chi")
    if chi is not None:
        values = np.asarray(chi, dtype=float)
        top = _stable_top_n(values, adata.var_names, 20)
        out["top_ambient_genes"] = [
            {"gene": str(adata.var_names[i]), "chi": float(values[i])} for i in top
        ]
    elif "ambidose_chi" in adata.uns:
        frame = adata.uns["ambidose_chi"]

        def _top_for_library(name):
            row = frame.loc[name]
            top = _stable_top_n(row.to_numpy(dtype=float), row.index, 20)
            return [{"gene": str(row.index[i]), "chi": float(row.iloc[i])} for i in top]

        out["top_ambient_genes_by_library"] = {
            str(name): _top_for_library(name) for name in frame.index
        }
    out["droplet_key"] = key
    out["layer_out"] = layer
    out["input_layer"] = adata.uns.get("ambidose", {}).get("input_layer")
    cell_calling = adata.uns.get("ambidose", {}).get("cell_calling")
    if isinstance(cell_calling, dict):
        out["cell_calling_method"] = cell_calling.get("method")
        if "n_whitelist" in cell_calling:
            out["cell_calling_n_whitelist"] = cell_calling["n_whitelist"]
        if "n_called" in cell_calling:
            out["cell_calling_n_called"] = cell_calling["n_called"]
        if "n_dropped" in cell_calling:
            out["cell_calling_n_dropped"] = cell_calling["n_dropped"]
    if "ambidose_doublet" in adata.obs:
        out["doublet_marking_run"] = True
        out["n_doublet_scrublet"] = int(
            adata.obs["ambidose_doublet"].fillna(False).to_numpy(dtype=bool).sum()
        )
        if "ambidose_type_residual" in adata.obs:
            residual = adata.obs["ambidose_type_residual"].to_numpy(dtype=float)
            finite = np.isfinite(residual)
            out["n_type_residual_scored"] = int(finite.sum())
            out["n_type_residual_doublet_leaning"] = int((residual[finite] > 0).sum())
            out["n_type_residual_soup_leaning"] = int((residual[finite] < 0).sum())
    out["diagnostics"] = _report_diagnostics(adata.uns.get("ambidose", {}))
    return out


def _report_diagnostics(uns) -> dict:
    """Keep QC-useful uns fields; drop per-gene native index lists."""
    if not isinstance(uns, dict):
        return {}
    skip = {"native_genes_by_type"}
    out = {str(k): _jsonable(v) for k, v in uns.items() if k not in skip}
    native = uns.get("native_genes_by_type")
    if isinstance(native, dict):
        out["n_native_genes_by_type"] = {str(k): int(len(v)) for k, v in native.items()}
    return out


def write_summary_json(adata: AnnData, path: str | Path) -> Path:
    """Write :func:`summarize` as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summarize(adata), indent=2, sort_keys=True) + "\n")
    return out


def guess_gene_symbol_style(var_names) -> str:
    """``"human-like"``/``"mouse-like"``/``"undetermined"`` from mito gene case.

    Human symbols are ``MT-*``, mouse are ``mt-*``. Used by ``inspect``.
    """
    names = [str(g) for g in var_names]
    if sum(g.startswith("MT-") for g in names) > 2:
        return "human-like"
    if sum(g.startswith("mt-") for g in names) > 2:
        return "mouse-like"
    return "undetermined"


def _aggregate_status(statuses) -> str:
    statuses = set(statuses)
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def inspect_input(path: str | Path, *, cell_barcodes=None, empty_umi_max: int = 100) -> dict:
    """Inspect one input without mutating it or running denoising."""
    resolved = sniff_input(path)
    if resolved.kind == "root":
        libraries = {
            name: inspect_input(lib_path, empty_umi_max=empty_umi_max)
            for name, lib_path in (resolved.samples or {}).items()
        }
        return {
            "kind": "root",
            "input": str(Path(path)),
            "n_libraries": len(libraries),
            "libraries": libraries,
            "status": _aggregate_status(lib["status"] for lib in libraries.values()),
        }
    if resolved.kind == "h5ad":
        import scanpy as sc

        adata = sc.read_h5ad(resolved.raw)
    elif resolved.kind == "mtx":
        adata = read_10x_mtx(resolved.raw)
    else:
        adata = read_10x_h5(resolved.raw)
    source = cell_barcodes if cell_barcodes is not None else resolved.filtered_barcodes
    barcodes = read_10x_barcodes(source) if isinstance(source, (str, Path)) else source
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    empty_candidates = (totals > 0) & (totals <= empty_umi_max)
    if barcodes is not None:
        wanted_for_empty = {normalize_barcode(x) for x in barcodes}
        in_whitelist = np.array(
            [normalize_barcode(x) in wanted_for_empty for x in adata.obs_names.astype(str)]
        )
        empty_candidates &= ~in_whitelist
    matrix_bytes = (
        adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes
        if sparse.issparse(adata.X)
        else adata.X.nbytes
    )
    report = {
        "kind": resolved.kind,
        "input": str(Path(path)),
        "raw_matrix": str(resolved.raw),
        "n_droplets": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "estimated_matrix_memory_gib": float(matrix_bytes / 1024**3),
        "empty_umi_max": int(empty_umi_max),
        "n_empty_candidates": int(empty_candidates.sum()),
        "whitelist_source": None if source is None else str(source),
        "whitelist_mode": "missing"
        if source is None
        else "external"
        if cell_barcodes is not None
        else "cellranger_auto",
        "feature_types": (
            adata.var["feature_types"].astype(str).value_counts().to_dict()
            if "feature_types" in adata.var
            else {"Gene Expression": int(adata.n_vars)}
        ),
        "gene_symbol_style": guess_gene_symbol_style(adata.var_names),
        "obs_columns": [str(x) for x in adata.obs.columns],
    }
    if barcodes is not None:
        wanted = {normalize_barcode(x) for x in barcodes}
        observed = {normalize_barcode(x) for x in adata.obs_names}
        matched = len(wanted & observed)
        report.update(
            n_whitelist=int(len(wanted)),
            n_whitelist_matched=int(matched),
            whitelist_match_fraction=float(matched / len(wanted)) if wanted else 0.0,
        )
    problems = []
    if source is None:
        problems.append("no cell whitelist found; UMI-threshold mode is smoke-only")
    elif report.get("n_whitelist_matched", 0) == 0:
        problems.append("no whitelist barcodes matched the raw matrix")
    if report["n_empty_candidates"] < 10:
        problems.append("fewer than 10 candidate empty droplets")
    report["problems"] = problems
    if any("no whitelist barcodes" in p for p in problems):
        report["status"] = "error"
    elif problems:
        report["status"] = "warn"
    else:
        report["status"] = "ok"
    return report


def write_inspection_json(report: dict, path: str | Path) -> Path:
    """Write an inspect_input result as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    return out


def write_report(adata: AnnData, path: str | Path) -> Path:
    """Write a self-contained HTML QC report for a completed run."""
    import matplotlib.pyplot as plt

    from . import pl

    fig = pl.summary(adata)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    image = b64encode(buffer.getvalue()).decode("ascii")
    data = summarize(adata)
    rows = "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in data.items()
        if not isinstance(v, (dict, list))
    )
    payload = escape(json.dumps(data, indent=2, sort_keys=True))
    n_ok = data.get("n_trust_ok")
    n_cells = data.get("n_cells")
    banner = (
        "<p><strong>How to read ambient-fraction QC.</strong> "
        "Corrected counts are stored in <code>ambidose_denoised</code>. "
        "<code>ambidose_rho_trust</code> identifies cells where "
        "<code>ambidose_rho</code> should be interpreted cautiously. "
        "These flags are not cell-filtering recommendations.</p>"
    )
    if n_ok is not None and n_cells:
        banner += (
            f"<p>Ambient fraction suitable for interpretation: "
            f"<strong>{n_ok}/{n_cells} cells</strong>. "
            f"Interpret with caution: {n_cells - n_ok} cells "
            f"(too few informative genes: {data.get('n_trust_low_evidence', 0)}; "
            f"near upper limit: {data.get('n_trust_ceiling_risk', 0)}; "
            f"unresolved cell group: {data.get('n_trust_type_structure_risk', 0)}; "
            f"under-executed dose: {data.get('n_trust_under_execution', 0)}; "
            f"over-removal: {data.get('n_trust_over_removal', 0)}).</p>"
        )
    dependence = data.get("diagnostics", {}).get("type_sample_dependence", {})
    if dependence.get("n_samples", 1) > 1:
        scope = dependence.get("scope", "unknown")
        banner += (
            "<p><strong>Type-library dependence.</strong> "
            f"Typing scope: {escape(str(scope))}; "
            f"Cramers V={float(dependence.get('cramers_v', 0.0)):.3f}; "
            f"median maximum library fraction per type="
            f"{100.0 * float(dependence.get('median_type_sample_fraction', 0.0)):.1f}%. "
            + (
                "Automatic types were constructed independently per library, so library-exclusive cluster labels are expected and are not themselves a warning."
                if scope == "per_sample"
                else "Strong dependence may reflect batch structure or real composition differences; inspect the contingency table."
            )
            + "</p>"
        )
    retention = data.get("cell_umi_retention_percentiles", {})
    if retention:
        banner += (
            "<p><strong>Count-retention check.</strong> "
            f"Total cell UMI retained: {100.0 * (1.0 - data.get('total_removed_fraction', 0.0)):.1f}%. "
            f"Per-cell retention: p1={100.0 * retention['p1']:.1f}%, "
            f"p5={100.0 * retention['p5']:.1f}%, "
            f"median={100.0 * retention['median']:.1f}%. "
            f"Cells below 50% retention: {data.get('n_cells_below_50pct_umi_retention', 0)}. "
            "Validated zero-ambient simulations still lose about 5-7% of total UMIs; "
            "compare corrected counts with raw_counts for expression-sensitive analyses.</p>"
        )
    by_type = data.get("median_rho_by_type")
    type_rows = (
        "<h2>Median &rho; by type</h2><table>"
        + "".join(
            f"<tr><th>{escape(str(t))}</th><td>{escape(f'{v:.3f}')}</td></tr>"
            for t, v in sorted(by_type.items())
        )
        + "</table>"
        if by_type
        else ""
    )
    top_genes = data.get("top_removed_genes")
    gene_rows = (
        "<h2>Top ambient-removed genes</h2>"
        "<table><tr><th>gene</th><th>removed UMI (cells)</th></tr>"
        + "".join(
            f"<tr><td>{escape(str(g['gene']))}</td><td>{g['removed_umi']}</td></tr>"
            for g in top_genes[:20]
        )
        + "</table>"
        if top_genes
        else ""
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>AmbiDose QC report</title><style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}th,td{{padding:.35rem .7rem;border:1px solid #ccc;text-align:left}}img{{max-width:100%}}pre{{background:#f5f5f5;padding:1rem;overflow:auto}}.banner{{background:#fff3cd;border:1px solid #ffc107;padding:1rem;margin:1rem 0}}</style></head><body><h1>AmbiDose QC report</h1><div class="banner">{banner}</div><table>{rows}</table><h2>Diagnostics</h2><img alt="AmbiDose summary plots" src="data:image/png;base64,{image}">{type_rows}{gene_rows}<h2>Machine-readable summary</h2><pre>{payload}</pre></body></html>"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
