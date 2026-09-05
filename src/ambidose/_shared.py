"""Shared constants and low-level helpers for preprocessing."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

DROPLET_KEY = "ambidose_droplet"
CHI_KEY = "ambidose_chi"
DOSE_KEY = "ambidose_d"
RHO_KEY = "ambidose_rho"
LAYER_OUT = "ambidose_denoised"
SAMPLE_KEY_DEFAULT = None
TYPE_KEY = "ambidose_type"
CLUSTER_KEY = "ambidose_cluster"
RHO_TRUST_KEY = "ambidose_rho_trust"
TRUST_OK = "ok"
TRUST_LOW_EVIDENCE = "low_evidence"
TRUST_CEILING = "ceiling_risk"
TRUST_TYPE = "type_structure_risk"
TRUST_UNDER_EXECUTION = "under_execution"
TRUST_OVER_REMOVAL = "over_removal"
TRUST_NOT_CELL = "not_cell"
UNDER_EXECUTION_RATIO = 0.5
OVER_EXECUTION_RATIO = 1.05
OVER_REMOVAL_FRACTION = 0.5
# remain_P / (n χ_P) at or below this looks like soup sitting on
# protected native genes (χ is estimated from empty droplets).
NATIVE_SOUP_RATIO = 1.25
MIN_PROTECTED_CHI = 0.02
SHRINK_K = 8.0
MIN_GENES = 8
MIN_VALID = 5
MIN_TYPE_CELLS = 10
# Exclusive ownership needs a gap over the runner-up meta/fragment.
# Housekeeping and unowned injection genes sit at ~1x across groups;
# a 1.8x lineage marker (CD3D-like) still clears 1.2x.
OWNER_MIN_FOLD = 1.2
# A fragment can only inherit its meta-group's ownership grant if its own
# raw mean is within this fold of the meta-group's strongest individual
# fragment. Meta-group merging (_split_noise_meta_ids) is intentionally
# permissive on small/noisy clusters -- their split-half profile noise is
# large, so complete-linkage can transitively pull a biologically distinct
# cluster into the same meta-group as a real marker-expressing one. Without
# this floor, ownership (and the unconditional native_confidence=1.0
# protection it grants) spreads to every member regardless of whether that
# member itself shows any of the signal, protecting off-target ambient
# leakage from subtraction. Measured on real data: false-positive grants
# (owning cluster's eval identity != the marker's documented owner, with
# the true-owner type present and outranked in the same sample) were 1734/
# 1925 on kidney and the dominant share of 4709 on fetal liver; the losing
# fragment's own mean sat 16x-2600x below the true owner's mean in the
# same meta-group. 10x keeps headroom for genuine same-identity variation
# (the docstring case this was built for -- erythroid maturation stages an
# order of magnitude apart) while excluding every measured false positive,
# all of which start above 16x. See CHANGELOG.
OWNER_FRAGMENT_MIN_SHARE_FOLD = 10.0
# EXPERIMENTAL (not wired into the frozen product path by default): a gene
# ranks as "owned" by a meta-group if it's in that group's own top-K genes
# by mean, regardless of how other groups compare -- lets biologically
# shared markers (e.g. erythroid maturation fragments) be owned by
# multiple groups at once instead of one cross-group magnitude winner.
OWNER_TOP_K = 50
# NOT WIRED IN (tried and rejected, kept as a documented negative result --
# see CHANGELOG). Was meant to cap how much extra a single gene can absorb
# from _realloc_unspent_rank1's leftover pool, as a multiple of that gene's
# own base rank-1 share (dose*chi_g). Rejected: on real kidney data the
# "legitimate" (must-win-needed) and "harmful" (kidney/fetal-liver
# over-correcting) realloc multiples occupy the same numeric range (median
# 2.96x, p90 5.76x across ~400k real events), so no fold threshold
# separates them -- _realloc_unspent_rank1's actual fix uses a structural
# `leftover_cap` (frozen single-winner ownership's own leftover) instead.
REALLOC_CAP_FOLD = 5.0
# Apply first-inflection only when the 3-component mixture is inflated
# relative to the rank-curve cliff (debris-heavy libraries). Below this
# the mixture's second mode is kept (heterogeneous high-RNA heads).
MIX_INFLATION_RATIO = 2.5
# Three-tier label-free Leiden resolution, keyed off scFair's estimated
# population count. Coarse / medium / fine are 0.08 / 0.2 / 0.35.
LEIDEN_RESOLUTION_COARSE = 0.08
LEIDEN_RESOLUTION_MEDIUM = 0.2
LEIDEN_RESOLUTION_FINE = 0.35
MEDIUM_N_DENSITY_POPS = 12
FINE_N_DENSITY_POPS = 20
# Coarse-typing speed trial. Off below this many cells even if requested;
# the shortcuts change the graph, so small libraries keep the default path.
TYPING_FAST_N_CELLS = 20_000
EMPTY_TYPE = object()
EMPTY_TYPES = frozenset({EMPTY_TYPE})


class _StageProgress:
    """Elapsed-time reporting and a quiet heartbeat for long product stages."""

    def __init__(self, label: str, *, heartbeat_seconds: float = 30.0):
        self.label = label
        self.heartbeat_seconds = heartbeat_seconds
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.started = time.monotonic()
        print(f"ambidose: {self.label}...", file=sys.stderr, flush=True)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            elapsed = time.monotonic() - self.started
            print(
                f"ambidose: still {self.label} (elapsed {elapsed:.0f}s)...",
                file=sys.stderr,
                flush=True,
            )

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        elapsed = time.monotonic() - self.started
        status = "completed" if exc_type is None else "failed"
        print(
            f"ambidose: {self.label} {status} in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def _reject_view(adata: AnnData, fname: str) -> None:
    """Refuse to mutate an AnnData view in place (e.g. ``adata[mask]`` passed
    without ``.copy()``): anndata silently converts the view to an actual
    object on first write (``ImplicitModificationWarning``), which "works"
    but means the caller's own object now aliases -- or, depending on the
    slice, no longer aliases -- whatever they sliced from, a easy-to-miss
    footgun for a library whose whole job is writing new obs/var/layers
    columns onto its input.
    """
    if getattr(adata, "is_view", False):
        raise ValueError(
            f"{fname}(adata) got a view (e.g. from `adata[mask]` without "
            "`.copy()`); pass adata.copy() instead -- writing obs/var/layers "
            "columns onto a view silently realizes it into a new object, "
            "which is easy to miss since the call otherwise looks like it "
            "mutated your original object in place"
        )


def _as_csr(x):
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(x)


def droplet_key_used(adata: AnnData) -> str | None:
    """Column ``denoise()`` used for droplet labels, or ``None`` if unknown."""
    stored = adata.uns.get("ambidose", {})
    if isinstance(stored, dict):
        key = stored.get("droplet_key")
        if isinstance(key, str) and key in adata.obs.columns:
            return key
    return None


def require_run_keys(adata: AnnData) -> tuple[str, str]:
    """Return ``(droplet_key, layer_out)`` written by ``denoise()``.

    Raises if those keys are missing instead of treating every barcode as a cell.
    """
    stored = adata.uns.get("ambidose", {})
    if not isinstance(stored, dict) or "droplet_key" not in stored:
        raise KeyError(
            "uns['ambidose']['droplet_key'] missing; run denoise() before summarize/write_report"
        )
    key = stored["droplet_key"]
    if key not in adata.obs.columns:
        raise KeyError(f"obs[{key!r}] missing; uns says denoise used that droplet_key")
    layer_out = stored.get("layer_out", LAYER_OUT)
    return str(key), str(layer_out)


def raw_count_matrix(adata: AnnData):
    """CSR input UMIs for the recorded run.

    An explicit ``input_layer`` is authoritative. ``raw_counts`` is the
    preserved copy written by a completed run, not a stronger claim than the
    layer the caller explicitly selected for that run.
    """
    stored = adata.uns.get("ambidose", {})
    input_layer = stored.get("input_layer") if isinstance(stored, dict) else None
    if isinstance(stored, dict) and (
        stored.get("core_completed") is True or stored.get("completed") is True
    ):
        if "raw_counts" not in adata.layers:
            raise RuntimeError(
                "completed AmbiDose object is corrupted: layers['raw_counts'] is missing"
            )
        return _as_csr(adata.layers["raw_counts"])
    if input_layer is not None:
        if input_layer not in adata.layers:
            raise KeyError(
                f"uns['ambidose']['input_layer']={input_layer!r} missing from "
                "layers; summarize cannot find the matrix denoise() used"
            )
        return _as_csr(adata.layers[input_layer])
    if "raw_counts" in adata.layers:
        return _as_csr(adata.layers["raw_counts"])
    return _as_csr(adata.X)


def _same_matrix(left, right) -> bool:
    """Exact equality for dense/sparse count matrices."""
    a = _as_csr(left)
    b = _as_csr(right)
    return a.shape == b.shape and (a != b).nnz == 0


def _validated_group_values(
    adata: AnnData, key: str, *, kind: str, allow_missing: bool
) -> pd.Series:
    """Return stable string labels without merging distinct source identities."""
    values = adata.obs[key]
    missing = values.isna()
    stripped_empty = values.astype("string").str.strip().eq("").fillna(False)
    invalid = missing | stripped_empty
    if not allow_missing and bool(invalid.any()):
        raise ValueError(
            f"obs[{key!r}] contains {int(invalid.sum())} missing or empty {kind} labels"
        )
    labels = values.astype(str).astype(object)
    labels.loc[invalid] = EMPTY_TYPE
    original = values[~invalid]
    converted = labels[~invalid]
    if int(original.nunique(dropna=False)) != int(converted.nunique(dropna=False)):
        raise ValueError(
            f"obs[{key!r}] contains distinct {kind} labels that collide after string conversion"
        )
    return labels


def _validated_sample_values(adata: AnnData, sample_key: str) -> pd.Series:
    return _validated_group_values(adata, sample_key, kind="sample", allow_missing=False)


def _validated_type_values(adata: AnnData, type_key: str) -> pd.Series:
    return _validated_group_values(adata, type_key, kind="type", allow_missing=True)


def _record_cluster_diag(adata: AnnData, **fields) -> None:
    uns = dict(adata.uns.get("ambidose", {}))
    clustering = dict(uns.get("clustering", {}))
    clustering.update(fields)
    uns["clustering"] = clustering
    adata.uns["ambidose"] = uns


def _available_ram_bytes() -> int | None:
    """Free RAM the process may still use, or None if it cannot be read.

    Includes ``MemAvailable``, cgroup remaining, and ``RLIMIT_AS`` (``ulimit
    -v``) minus current virtual size, so a 24 GiB address-space cap is
    visible even when the machine has more RAM.
    """
    remaining: list[int] = []
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    remaining.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    for max_path, used_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            cap = max_path.read_text().strip()
            if cap == "max":
                continue
            limit = int(cap)
            if limit >= (1 << 62):
                continue
            used = int(used_path.read_text())
            remaining.append(limit - used)
        except (OSError, ValueError):
            continue
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            vm = None
            with open("/proc/self/status", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("VmSize:"):
                        vm = int(line.split()[1]) * 1024
                        break
            if vm is not None:
                remaining.append(max(0, int(soft) - vm))
    except (OSError, ValueError):
        pass
    if not remaining:
        return None
    return max(0, min(remaining))


def _require_ram(n_bytes: int, *, where: str) -> None:
    """Raise before a dense allocation that will not fit in free RAM."""
    need = int(n_bytes)
    if need <= 0:
        return
    avail = _available_ram_bytes()
    if avail is None or need <= avail:
        return
    raise MemoryError(
        f"ambidose: {where} needs {need / (1024**3):.1f} GiB free RAM; "
        f"{avail / (1024**3):.1f} GiB available. Use Cell Ranger filtered "
        "barcodes so empty droplets are not treated as cells, or run with more memory"
    )


# Bound dense intermediates as a fraction of memory available when the stage
# starts. This scales from laptops to large-memory hosts while leaving room for
# AnnData, sparse inputs, and Python itself. Only block width changes; every
# gene still uses the same formula.
_DENSE_WORKSPACE_RAM_FRACTION = 1.0 / 3.0


def _dense_chunk_columns(
    n_rows: int,
    n_cols: int,
    *,
    arrays_per_value: int,
    max_cols: int,
) -> int:
    """Columns fitting a conservative float64 dense-workspace budget."""
    if n_rows <= 0 or n_cols <= 0:
        return 1
    cap = max(1, min(int(max_cols), int(n_cols)))
    available = _available_ram_bytes()
    if available is None:
        return cap
    budget = int(available * _DENSE_WORKSPACE_RAM_FRACTION)
    bytes_per_col = max(1, int(n_rows) * 8 * max(1, int(arrays_per_value)))
    return max(1, min(cap, budget // bytes_per_col))


def _usable_cpu_count() -> int:
    """Cores this process may actually schedule onto, not just the host total.

    ``os.sched_getaffinity`` reflects a container/cgroup CPU-set restriction
    that ``os.cpu_count()`` does not; fall back to it on platforms without
    ``sched_getaffinity`` (e.g. macOS).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _configure_scanpy_n_jobs(n_jobs: int | None) -> int:
    """Resolve Scanpy workers; ``None`` and ``-1`` use every available CPU."""
    if n_jobs is None or n_jobs == -1:
        return _usable_cpu_count()
    resolved = int(n_jobs)
    if resolved < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    return resolved


def _mc_worker_count(n_jobs: int) -> int:
    """Parallel Monte Carlo workers, capped by CPU count and free RAM.

    Each worker chunks multinomial simulations to a 128 MiB count matrix.
    Budget twice that size for likelihood temporaries and allocator overhead,
    and use at most half of currently available RAM across workers.
    """
    if n_jobs <= 1:
        return 1
    per_worker = 256 * 1024**2
    cpu_n = _usable_cpu_count()
    avail = _available_ram_bytes()
    mem_n = max(1, int(avail // 2 // per_worker)) if avail is not None else cpu_n
    return max(1, min(cpu_n, mem_n, n_jobs))


def _thread_worker_count(
    n_jobs: int | None,
    *,
    n_tasks: int,
    per_worker_bytes: int,
) -> int:
    """Threads allowed by affinity, requested jobs, task count, and RAM."""
    if n_tasks <= 1:
        return 1
    cpu_n = _usable_cpu_count()
    requested = cpu_n if n_jobs is None else max(1, int(n_jobs))
    available = _available_ram_bytes()
    if available is None:
        mem_n = cpu_n
    else:
        budget = int(available * _DENSE_WORKSPACE_RAM_FRACTION)
        mem_n = max(1, budget // max(1, int(per_worker_bytes)))
    return max(1, min(cpu_n, requested, int(n_tasks), mem_n))


def _require_raw_integer_counts(adata: AnnData, *, layer: str | None, fname: str) -> None:
    """Reject non-finite, negative, or non-integer count matrices."""
    x = _as_csr(adata.layers[layer] if layer is not None else adata.X)
    if x.data.size == 0:
        return
    if not np.isfinite(x.data).all():
        raise ValueError(f"{fname}() requires raw integer UMI counts; found non-finite values")
    if (x.data < 0).any():
        raise ValueError(f"{fname}() requires raw integer UMI counts; found negative values")
    # rtol=0: default rtol=1e-5 scales with magnitude, so CPM/CP10K values
    # around 1e4-1e5 can look "integer" (e.g. 100000.4 ~ 100000.0).
    if not np.allclose(x.data, np.rint(x.data), rtol=0, atol=1e-6):
        hint = ""
        if layer is None:
            raw_looking = [name for name in ("counts", "raw_counts", "raw") if name in adata.layers]
            if raw_looking:
                hint = f" adata.layers has {raw_looking!r}; pass layer={raw_looking[0]!r}?"
        raise ValueError(
            f"{fname}() requires raw integer UMI counts in adata.X "
            "(or the given layer). Normalized, log1p, or scaled matrices are "
            f"rejected because rounding them can invent counts.{hint}"
        )


def _stable_count_order(adata: AnnData, totals: np.ndarray, idx=None) -> np.ndarray:
    """Indices ordered by count descending and stable barcode identity."""
    names = adata.obs_names.astype(str)
    if not names.is_unique:
        raise ValueError("obs_names must be unique for deterministic count ordering")
    selected = np.arange(adata.n_obs) if idx is None else np.asarray(idx, dtype=np.int64)
    order = np.lexsort((names.to_numpy()[selected], -np.asarray(totals)[selected]))
    return selected[order]


def _stable_subsample_indices(obs_names, size: int, *, seed: int = 0) -> np.ndarray:
    names = pd.Index(obs_names).astype(str)
    if not names.is_unique:
        raise ValueError("obs_names must be unique for deterministic subsampling")
    identity_order = np.argsort(names.to_numpy(), kind="stable")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(identity_order.size, size=size, replace=False)
    return np.sort(identity_order[chosen])


def _sample_storage_id(sample: str | None) -> str:
    """HDF5-safe internal key; the biological label is stored as a value."""
    if sample is None:
        return "global"
    digest = hashlib.blake2b(str(sample).encode(), digest_size=10).hexdigest()
    return f"sample_{digest}"


def _validate_output_layer(*, layer: str | None, layer_out: str) -> None:
    """Protect the raw snapshot and the selected input layer from writes."""
    if layer_out == "raw_counts":
        raise ValueError("layer_out='raw_counts' is reserved for immutable input counts")
    if layer is not None and layer_out == layer:
        raise ValueError("layer_out must not overwrite the input layer")


def _chi_vector(adata: AnnData) -> np.ndarray:
    if CHI_KEY in adata.uns:
        raise ValueError("ambiguous χ state: both var and uns representations are present")
    if CHI_KEY not in adata.var.columns:
        raise KeyError(f"{CHI_KEY!r} missing; run estimate_chi first")
    return _validate_chi_vector(
        adata.var[CHI_KEY].to_numpy(), adata.n_vars, where=f"var[{CHI_KEY!r}]"
    )


def _chi_for_obs(adata: AnnData, *, sample_key: str | None, sample_name: str | None) -> np.ndarray:
    """Validated χ for one explicit sample or the single-library profile."""
    if sample_key is None:
        return _chi_vector(adata)
    if sample_name is None:
        raise ValueError("sample_name is required when sample_key is set")
    frame = _validate_chi_frame(adata, sample_key)
    if sample_name not in frame.index:
        raise KeyError(f"sample {sample_name!r} missing from uns[{CHI_KEY!r}]")
    return frame.loc[sample_name].to_numpy(dtype=np.float64)


def _resolve_cell_mask(adata: AnnData, droplet_key: str | None, cell_label: str) -> np.ndarray:
    """Resolve cells explicitly; only None means a cells-only object."""
    if droplet_key is None:
        return np.ones(adata.n_obs, dtype=bool)
    if droplet_key not in adata.obs.columns:
        raise KeyError(f"droplet_key={droplet_key!r} not in adata.obs")
    return adata.obs[droplet_key].astype(str).to_numpy() == cell_label


def _sample_names(adata: AnnData, sample_key: str | None) -> np.ndarray | None:
    if sample_key is None:
        return None
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")
    if CHI_KEY not in adata.uns:
        raise ValueError(f"explicit sample_key requires sample-specific χ in uns[{CHI_KEY!r}]")
    return _validated_sample_values(adata, sample_key).to_numpy()


def _profile(x_empty: sparse.spmatrix, *, sample: str | None = None) -> np.ndarray:
    totals = np.asarray(x_empty.sum(axis=0)).ravel().astype(np.float64)
    s = totals.sum()
    if s <= 0:
        where = f"sample {sample!r}: " if sample is not None else ""
        raise ValueError(f"{where}empty droplets have zero total UMI")
    return totals / s


def _validate_chi_vector(values, n_vars: int, *, where: str) -> np.ndarray:
    chi = np.asarray(values, dtype=np.float64)
    if chi.shape != (n_vars,):
        raise ValueError(f"{where} must have shape ({n_vars},), got {chi.shape}")
    if not np.isfinite(chi).all():
        raise ValueError(f"{where} must contain only finite values")
    if np.any(chi < 0):
        raise ValueError(f"{where} must be nonnegative")
    total = float(chi.sum())
    if total <= 0 or not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{where} must sum to 1, got {total}")
    return chi


def _validate_chi_frame(adata: AnnData, sample_key: str) -> pd.DataFrame:
    if CHI_KEY in adata.var.columns:
        raise ValueError("ambiguous χ state: both var and uns representations are present")
    frame = adata.uns.get(CHI_KEY)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"uns[{CHI_KEY!r}] must be a sample-by-gene DataFrame")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"uns[{CHI_KEY!r}] must have unique sample and gene labels")
    samples = _validated_sample_values(adata, sample_key)
    expected_samples = set(samples.unique())
    actual_samples = set(frame.index.astype(str))
    if actual_samples != expected_samples:
        raise ValueError(
            f"uns[{CHI_KEY!r}] sample labels do not match obs[{sample_key!r}]: "
            f"missing={sorted(expected_samples - actual_samples)}, "
            f"extra={sorted(actual_samples - expected_samples)}"
        )
    genes = adata.var_names.astype(str)
    if set(frame.columns.astype(str)) != set(genes):
        raise ValueError(f"uns[{CHI_KEY!r}] gene labels must exactly match adata.var_names")
    aligned = frame.copy()
    aligned.columns = aligned.columns.astype(str)
    aligned.index = aligned.index.astype(str)
    if not aligned.index.is_unique:
        raise ValueError(f"uns[{CHI_KEY!r}] sample labels collide after string conversion")
    if not aligned.columns.is_unique:
        raise ValueError(f"uns[{CHI_KEY!r}] gene labels collide after string conversion")
    aligned = aligned.loc[sorted(expected_samples), genes]
    for sample, row in aligned.iterrows():
        _validate_chi_vector(row.to_numpy(), adata.n_vars, where=f"χ for sample {sample!r}")
    return aligned
