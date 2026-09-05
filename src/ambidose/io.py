"""Read Cell Ranger raw HDF5 and write h5ad."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, concat

# Cell Ranger raw/filtered directory or .h5 basename pairs, newest layout
# first. v3+ ("feature") replaced v2's genome-subdirectory ("gene_bc")
# layout; both are still seen in the wild.
_RAW_FILTERED_PAIRS = [
    ("raw_feature_bc_matrix", "filtered_feature_bc_matrix"),
    ("raw_gene_bc_matrices", "filtered_gene_bc_matrices"),
]


def list_10x_genomes(path: str | Path) -> list[str]:
    """Return Cell Ranger v2 genome groups, or ``[]`` for a v3 single matrix."""
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            return []
        return [k for k, v in f.items() if isinstance(v, h5py.Group) and "barcodes" in v]


def read_10x_h5(
    path: str | Path,
    *,
    genome: str | None = None,
    gex_only: bool = True,
) -> AnnData:
    """Read a Cell Ranger feature-barcode matrix ``.h5`` (raw or filtered).

    Barnyard / multi-genome v2 files (e.g. hgmm12k) are concatenated on genes
    and tagged with ``var['genome']``.
    """
    path = Path(path)
    genomes = list_10x_genomes(path)
    if genome is not None or len(genomes) <= 1:
        adata = sc.read_10x_h5(str(path), genome=genome, gex_only=gex_only)
        if len(genomes) == 1:
            adata.var["genome"] = genomes[0]
        adata.var_names_make_unique()
        return adata

    ads = []
    for g in genomes:
        ad = sc.read_10x_h5(str(path), genome=g, gex_only=gex_only)
        ad.var_names_make_unique()
        ad.var["genome"] = g
        ads.append(ad)
    adata = concat(ads, axis=1, merge="same")
    adata.var_names_make_unique()
    return adata


def _resolve_maybe_gz(path: Path) -> Path:
    """Resolve a TSV path, preferring the gzipped Cell Ranger v3+ name.

    If both ``barcodes.tsv`` and ``barcodes.tsv.gz`` exist, the ``.gz`` file
    is used. A missing preferred name falls back to the other suffix.
    """
    path = Path(path)
    if path.suffix == ".gz":
        if path.exists():
            return path
        bare = path.with_suffix("")
        return bare if bare.exists() else path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    return path


def read_10x_barcodes(path: str | Path) -> list[str]:
    """Read barcodes from Cell Ranger TSV[.gz] or feature-matrix HDF5."""
    path = _resolve_maybe_gz(Path(path))
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            if "matrix" in f and "barcodes" in f["matrix"]:
                values = f["matrix/barcodes"][:]
            else:
                blocks = [
                    group["barcodes"][:]
                    for group in f.values()
                    if isinstance(group, h5py.Group) and "barcodes" in group
                ]
                if not blocks:
                    raise ValueError(f"no Cell Ranger barcode dataset found in {path}")
                values = np.concatenate(blocks)
        return list(dict.fromkeys(v.decode() if isinstance(v, bytes) else str(v) for v in values))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return list(dict.fromkeys(ln.strip() for ln in f if ln.strip()))


def normalize_barcode(b: str) -> str:
    """Strip whitespace and a trailing GEM-well ``-1`` suffix for matching.

    The single source of truth for barcode matching across ``pp`` and
    ``reporting`` -- keep them using this so a whitelist that matches in
    ``inspect`` also matches in ``call_cells``/``classify_droplets``.
    """
    b = str(b).strip()
    return b[:-2] if b.endswith("-1") else b


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        hit = _resolve_maybe_gz(c)
        if hit.exists():
            return hit
    return None


def find_filtered_barcodes(raw_path: str | Path) -> Path | None:
    """Best-effort match of a raw input's filtered (cell-called) barcodes.

    ``raw_path`` may be a raw ``.h5``, a raw mtx directory, or a Cell Ranger
    ``outs/`` directory containing a ``raw_*`` child. Tries, in order: a
    same-directory ``raw_*`` -> ``filtered_*`` name swap (dir or ``.h5``,
    v3-style and v2-style), then the same swap one level up (v2's
    ``<root>/<sample>/raw_gene_bc_matrices/<genome>/`` layout, where
    ``filtered_gene_bc_matrices/<genome>/`` is a sibling of
    ``raw_gene_bc_matrices/``, not of the genome directory itself). Returns
    the filtered barcodes.tsv[.gz] path if found, or the filtered ``.h5``
    path (barcodes are read from it directly by the caller), else ``None``.
    Callers must treat ``None`` as missing: the product path requires a
    whitelist (``cell_barcodes`` or a Cell Ranger filtered pair). UMI-
    threshold cell calling is smoke-only and not a CLI fallback.
    """
    raw_path = Path(raw_path)
    if raw_path.is_file():
        # xxx/raw_feature_bc_matrix.h5 -> xxx/filtered_feature_bc_matrix.h5
        for raw_name, filt_name in _RAW_FILTERED_PAIRS:
            if raw_name in raw_path.stem:
                cand = raw_path.with_name(raw_path.name.replace(raw_name, filt_name))
                if cand.exists():
                    return cand
        return None

    if not raw_path.is_dir():
        return None

    # same-level swap: <parent>/raw_feature_bc_matrix/ -> <parent>/filtered_feature_bc_matrix/
    for raw_name, filt_name in _RAW_FILTERED_PAIRS:
        if raw_path.name == raw_name:
            filt_dir = raw_path.parent / filt_name
            hit = _first_existing(filt_dir / "barcodes.tsv")
            if hit:
                return hit
            h5 = _first_existing(raw_path.parent / f"{filt_name}.h5")
            if h5:
                return h5

    # v2 layout: <sample>/raw_gene_bc_matrices/<genome>/ -- filtered is a
    # sibling of raw_gene_bc_matrices/, keyed by the same <genome> name.
    if raw_path.parent.name == "raw_gene_bc_matrices":
        filt_dir = raw_path.parent.parent / "filtered_gene_bc_matrices" / raw_path.name
        hit = _first_existing(filt_dir / "barcodes.tsv")
        if hit:
            return hit

    return None


def _has_mtx(d: Path) -> bool:
    return (d / "matrix.mtx").exists() or (d / "matrix.mtx.gz").exists()


@dataclass
class ResolvedInput:
    """What :func:`sniff_input` found. ``kind`` is ``"h5"``, ``"h5ad"``,
    ``"mtx"``, or ``"root"`` (multi-sample; ``samples`` is set, ``raw`` is
    the root directory itself and otherwise unused).
    """

    kind: str
    raw: Path
    filtered_barcodes: Path | None = None
    samples: dict[str, Path] | None = None


def sniff_input(path: str | Path) -> ResolvedInput:
    """Auto-detect what kind of 10x input ``path`` is, and pair it with
    filtered barcodes if a sibling filtered dataset can be found.

    Recognizes, in order: a ``.h5``/``.h5ad`` file; a directory that is
    itself a raw mtx dir (``matrix.mtx[.gz]`` directly inside); a Cell Ranger
    sample directory containing a ``raw_feature_bc_matrix``/
    ``raw_gene_bc_matrices`` child (v3 ``outs/`` layout, or the same without
    the ``outs/`` wrapper) -- the common case of "point me at your
    Cell Ranger output," which also auto-pairs the filtered barcodes so the
    caller doesn't need a second ``--cell-barcodes`` flag; and finally a
    multi-sample root (subdirectories that each resolve to one of the
    above). Raises ``ValueError`` with what *was* found, rather than
    guessing, when nothing recognizable is under ``path``.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix == ".h5ad":
            return ResolvedInput(kind="h5ad", raw=path)
        if path.suffix == ".h5":
            return ResolvedInput(
                kind="h5", raw=path, filtered_barcodes=find_filtered_barcodes(path)
            )
        raise ValueError(f"unrecognized file type (expected .h5 or .h5ad): {path}")

    if not path.is_dir():
        raise FileNotFoundError(path)

    if _has_mtx(path):
        return ResolvedInput(kind="mtx", raw=path, filtered_barcodes=find_filtered_barcodes(path))

    for raw_name, _filt_name in _RAW_FILTERED_PAIRS:
        raw_h5 = path / f"{raw_name}.h5"
        if raw_h5.exists():
            return ResolvedInput(
                kind="h5", raw=raw_h5, filtered_barcodes=find_filtered_barcodes(raw_h5)
            )
        raw_dir = path / raw_name
        if raw_dir.is_dir() and _has_mtx(raw_dir):
            return ResolvedInput(
                kind="mtx", raw=raw_dir, filtered_barcodes=find_filtered_barcodes(raw_dir)
            )

    # v2 barnyard-style: raw_gene_bc_matrices/<genome>/ for one or more genomes.
    v2_container = path / "raw_gene_bc_matrices"
    if v2_container.is_dir():
        genome_dirs = [d for d in sorted(v2_container.iterdir()) if d.is_dir() and _has_mtx(d)]
        if len(genome_dirs) == 1:
            return ResolvedInput(
                kind="mtx",
                raw=genome_dirs[0],
                filtered_barcodes=find_filtered_barcodes(genome_dirs[0]),
            )
        if len(genome_dirs) > 1:
            names = ", ".join(d.name for d in genome_dirs)
            raise ValueError(
                f"{v2_container} has multiple genomes ({names}); point --input "
                f"at one directly, e.g. {genome_dirs[0]}"
            )

    samples = find_10x_mtx_samples(path)
    if samples:
        return ResolvedInput(kind="root", raw=path, samples=samples)

    raise ValueError(
        f"could not recognize a 10x dataset under {path} (looked for matrix.mtx[.gz], "
        "raw_feature_bc_matrix[.h5], raw_gene_bc_matrices/<genome>, and multi-sample "
        "subdirectories)"
    )


def find_10x_mtx_samples(root: str | Path) -> dict[str, Path]:
    """Map sample name -> raw Cell Ranger mtx directory under ``root``.

    Prefers Cell Ranger v3+ (``<sample>/outs/raw_feature_bc_matrix/matrix.mtx.gz``,
    or the same without ``outs/``). Still finds v2
    (``<sample>/raw_gene_bc_matrices/<genome>/matrix.mtx``). Sample name is
    the directory two levels above the raw mtx dir for v3-style
    layouts (``outs/raw_feature_bc_matrix`` -- 2 levels) and the Cell Ranger
    sample dir for v2 (``raw_gene_bc_matrices/<genome>`` -- 2 levels too),
    so both share the same ``parents[2]`` rule; the no-``outs/``-wrapper v3
    variant is only 1 level down, handled separately.
    """
    root = Path(root)
    found: dict[str, Path] = {}
    patterns = [
        ("*/outs/raw_feature_bc_matrix/matrix.mtx.gz", 2),
        ("*/outs/raw_feature_bc_matrix/matrix.mtx", 2),
        ("*/raw_feature_bc_matrix/matrix.mtx.gz", 1),
        ("*/raw_feature_bc_matrix/matrix.mtx", 1),
        ("*/raw_gene_bc_matrices/*/matrix.mtx.gz", 2),
        ("*/raw_gene_bc_matrices/*/matrix.mtx", 2),
    ]
    for pattern, depth in patterns:
        for mtx in sorted(root.glob(pattern)):
            name = mtx.parents[depth].name
            found.setdefault(name, mtx.parent)
    return found


def read_10x_mtx(path: str | Path, *, var_names: str = "gene_symbols") -> AnnData:
    """Read a Cell Ranger ``matrix.mtx`` directory (``genes.tsv`` or ``features.tsv``)."""
    path = Path(path)
    adata = sc.read_10x_mtx(str(path), var_names=var_names, make_unique=True)
    adata.var_names_make_unique()
    genome = path.name
    if genome in {"GRCh38", "hg19", "hg38", "mm10", "GRCm38", "mm39"}:
        adata.var["genome"] = genome
    return adata


def write_h5ad(adata: AnnData, path: str | Path, *, compression: str | None = "gzip") -> None:
    """Write AnnData to ``.h5ad``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(path), compression=compression)


def write_10x_mtx(
    adata: AnnData,
    path: str | Path,
    *,
    version: int = 3,
    sample_key: str | None = None,
) -> Path:
    """Write a Cell Ranger MTX directory.

    Default ``version=3``: gzipped ``matrix.mtx.gz``, ``barcodes.tsv.gz``,
    and three-column ``features.tsv.gz`` (id, name, feature type). Pass
    ``version=2`` for uncompressed ``matrix.mtx``, ``barcodes.tsv``, and
    two-column ``genes.tsv``.

    ``sample_key``: also write ``sample.tsv`` (one line per barcode, same
    order as ``barcodes.tsv``) from ``adata.obs[sample_key]`` when present.
    The 10x mtx format has no ``obs`` slot, so a multi-library run's sample
    assignment would otherwise be unrecoverable from the output alone.
    """
    from scipy.io import mmwrite

    if version not in (2, 3):
        raise ValueError("version must be 2 or 3")
    if sample_key is not None and sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    x = _as_int_csr(adata.X).T.tocsc()
    gene_id = (
        adata.var["gene_ids"].astype(str)
        if "gene_ids" in adata.var.columns
        else adata.var_names.astype(str)
    )
    symbol = adata.var_names.astype(str)
    if "feature_types" in adata.var.columns:
        ftype = adata.var["feature_types"].astype(str)
    else:
        ftype = np.full(adata.n_vars, "Gene Expression", dtype=object)

    if version == 3:
        with gzip.open(out / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, x, field="integer")
        with gzip.open(out / "barcodes.tsv.gz", "wt") as f:
            f.write("\n".join(adata.obs_names.astype(str)) + "\n")
        with gzip.open(out / "features.tsv.gz", "wt") as f:
            for gid, sym, kind in zip(gene_id, symbol, ftype, strict=True):
                f.write(f"{gid}\t{sym}\t{kind}\n")
    else:
        mmwrite(out / "matrix.mtx", x, field="integer")
        (out / "barcodes.tsv").write_text("\n".join(adata.obs_names.astype(str)) + "\n")
        with (out / "genes.tsv").open("w") as f:
            for gid, sym in zip(gene_id, symbol, strict=True):
                f.write(f"{gid}\t{sym}\n")
    if sample_key is not None:
        (out / "sample.tsv").write_text("\n".join(adata.obs[sample_key].astype(str)) + "\n")
    return out


def cast_int32_counts(data: np.ndarray) -> np.ndarray:
    """Validate count values and cast them to int32."""
    values = np.asarray(data)
    if values.size:
        if not np.isfinite(values).all():
            raise ValueError("counts must be finite; found NaN or infinity")
        if np.any(values < 0):
            raise ValueError("counts must be nonnegative")
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
            raise ValueError(
                "counts must be integer-valued; normalized or log-transformed "
                "matrices cannot be exported as 10x counts"
            )
        info = np.iinfo(np.int32)
        mx = float(np.max(rounded))
        mn = float(np.min(rounded))
        if mx > info.max or mn < info.min:
            raise ValueError(
                f"count {mx if mx > info.max else mn} does not fit in int32 "
                f"[{info.min}, {info.max}]; refusing to wrap to a negative or "
                "truncated value"
            )
    return np.rint(values).astype(np.int32, copy=False)


def _as_int_csr(x):
    from scipy import sparse

    x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)
    out = x.copy()
    out.data = cast_int32_counts(out.data)
    return out


@dataclass(frozen=True)
class ManifestRow:
    """One independent GEM/library and its optional explicit whitelist."""

    library: str
    raw_counts: Path
    cell_barcodes: Path | None


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """Read a TSV with library, raw_counts, and cell_barcodes columns."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"library", "raw_counts", "cell_barcodes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("manifest missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("manifest has no libraries")
    if frame["library"].duplicated().any() or (frame["library"].str.strip() == "").any():
        raise ValueError("manifest library names must be non-empty and unique")
    base = path.parent
    rows = []
    for item in frame.itertuples(index=False):
        raw = Path(item.raw_counts)
        raw = raw if raw.is_absolute() else base / raw
        bc = Path(item.cell_barcodes) if item.cell_barcodes else None
        if bc is not None and not bc.is_absolute():
            bc = base / bc
        if not raw.exists():
            raise FileNotFoundError(raw)
        if bc is not None and not bc.exists():
            raise FileNotFoundError(bc)
        rows.append(ManifestRow(str(item.library), raw, bc))
    return rows
