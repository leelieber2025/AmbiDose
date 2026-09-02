"""Continuous and integer ambient-removal budget allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _alloc_budget(tgt: float, room: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """Spend ``tgt`` on buckets with capacity ``room``, weights ``ws``.

    SoupX ``alloc``: if some genes saturate, leftover mass is reassigned to
    genes that still have room. Callers must already zero-out weights on
    protected genes so leftover cannot land on native markers.
    """
    n = int(room.size)
    out = np.zeros(n, dtype=np.float64)
    if tgt <= 0 or n == 0:
        return out
    room = np.clip(np.asarray(room, dtype=np.float64), 0.0, None)
    ws = np.clip(np.asarray(ws, dtype=np.float64), 0.0, None)
    ws = np.where(room > 0, ws, 0.0)
    s = float(ws.sum())
    if s <= 0:
        return out
    ws = ws / s
    want = tgt * ws
    if np.all(want <= room + 1e-12):
        return np.minimum(want, room)
    o = np.argsort(np.divide(room, ws, out=np.full(n, np.inf), where=ws > 0))
    w = ws[o]
    y = room[o]
    cy = np.concatenate([[0.0], np.cumsum(y[:-1])])
    # Tail weight sum(w[i:]) via a reverse cumsum, not "1 - cumsum(w[:i])":
    # the latter cancels two O(1) sums to recover a possibly tiny remainder
    # (an near-zero-but-nonzero last weight rounds its own tail to exactly
    # 0.0 once the running total saturates to 1.0 in float64), which
    # collapses that bucket's saturation breakpoint onto its predecessor's
    # and lets `sat` admit one bucket too many when `tgt` lands on that
    # spurious tie -- overspending the budget by that bucket's whole room.
    tail_w = np.cumsum(w[::-1])[::-1]
    k = np.full(n, np.inf, dtype=np.float64)
    nz = w > 0
    k[nz] = y[nz] / w[nz] * tail_w[nz] + cy[nz]
    sat = k <= tgt
    resid = tgt - float(y[sat].sum())
    w_left = float(w[~sat].sum())
    extra = np.zeros(n, dtype=np.float64)
    extra[sat] = y[sat]
    if resid > 0 and w_left > 0:
        extra[~sat] = resid * (w[~sat] / w_left)
    out[o] = extra
    return np.minimum(out, room)


def _confidence_weighted_take(
    observed: np.ndarray,
    chi: np.ndarray,
    dose: float,
    native_confidence: np.ndarray,
    is_u: np.ndarray | None = None,
) -> np.ndarray:
    """Rank-1 take scaled by native confidence; optional soupOnly on U genes.

    ``dose`` is a χ-direction budget (typically type-median ρ times the
    group's library sum), not a per-gene allocation of per-cell ``d_c``.
    Confidence 0 takes ``dose·χ``. Confidence 1 takes ``0.1·dose·χ``.
    Unexpressed unowned genes (``is_u``) are extra-cleared to the observed
    count; that extra-clear is outside the rank-1 budget.
    """
    confidence = np.clip(native_confidence, 0.0, 1.0)
    protected_take = dose * chi * (1.0 - 0.9 * confidence)
    take = np.minimum(observed, protected_take)
    if is_u is not None:
        take = np.where(is_u, observed, take)
    return take


def _realloc_unspent_rank1(
    take: np.ndarray,
    observed: np.ndarray,
    chi: np.ndarray,
    dose: float,
    is_p: np.ndarray,
    is_u: np.ndarray,
) -> np.ndarray:
    """Spend leftover χ-budget on non-protected, non-soupOnly genes.

    Protected genes keep their rank-1 slice. U genes stay on the soupOnly
    path. Leftover is the unused part of ``dose`` after the clipped take.
    """
    leftover = max(0.0, float(dose) - float(np.sum(take)))
    if leftover <= 0:
        return take
    blocked = is_p | is_u
    room = np.where(blocked, 0.0, np.maximum(observed - take, 0.0))
    weights = np.where(blocked, 0.0, chi)
    return take + _alloc_budget(leftover, room, weights)


def _subtract_row(y: np.ndarray, chi_g: np.ndarray, d: float) -> np.ndarray:
    """Rank-1 take. Used by the type-naive path and by research scripts."""
    return np.minimum(y, d * chi_g)


def _alloc_integer_budget(
    tgt: float, room: np.ndarray, ws: np.ndarray, *, tie_keys=None
) -> np.ndarray:
    """Spend the feasible half-up-rounded budget on positive-weight buckets."""
    room_i = np.rint(np.clip(room, 0.0, None)).astype(np.int64)
    weights = np.clip(np.asarray(ws, dtype=np.float64), 0.0, None)
    keys = (
        np.arange(room_i.size).astype(str) if tie_keys is None else np.asarray(tie_keys).astype(str)
    )
    if keys.shape != room_i.shape:
        raise ValueError("tie_keys must match room")
    if not pd.Index(keys).is_unique:
        raise ValueError("tie_keys must be unique after string conversion")
    return _alloc_integer_budget_validated(tgt, room_i, weights, keys)


def _alloc_integer_budget_validated(
    tgt: float,
    room_i: np.ndarray,
    weights: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    """Integer allocation after shape/type/key validation by the caller."""
    active = (room_i > 0) & (weights > 0)
    out = np.zeros(room_i.size, dtype=np.int64)
    if not active.any():
        return out
    target = min(
        int(np.floor(max(tgt, 0.0) + 0.5)),
        int(room_i[active].sum()),
    )
    continuous = _alloc_budget(float(target), room_i[active], weights[active])
    allocated = np.floor(continuous).astype(np.int64)
    remaining = target - int(allocated.sum())
    if remaining > 0:
        eligible = np.flatnonzero(allocated < room_i[active])
        fractions = continuous[eligible] - allocated[eligible]
        active_keys = keys[active]
        order = np.lexsort((active_keys[eligible], -fractions))
        allocated[eligible[order[:remaining]]] += 1
    out[np.flatnonzero(active)] = allocated
    if int(out.sum()) != target:
        raise RuntimeError("integer budget allocation failed to spend feasible target")
    return out


def _integerize_corrected(raw, corrected, *, gene_keys):
    """Convert continuous removal to deterministic integer UMI removal per cell."""
    raw = raw.tocsr(copy=True)
    corrected = corrected.tocsr(copy=True)
    if not (
        np.array_equal(raw.indptr, corrected.indptr)
        and np.array_equal(raw.indices, corrected.indices)
    ):
        raise RuntimeError("corrected matrix lost raw CSR sparsity alignment")
    continuous = np.clip(raw.data - corrected.data, 0.0, raw.data)
    removed = np.floor(continuous).astype(np.int64)
    fractions = continuous - removed
    gene_keys = np.asarray(gene_keys).astype(str)
    if gene_keys.shape != (raw.shape[1],):
        raise ValueError("gene_keys must have length n_vars")
    for row in range(raw.shape[0]):
        start, stop = int(raw.indptr[row]), int(raw.indptr[row + 1])
        target = int(np.floor(continuous[start:stop].sum() + 0.5))
        remaining = target - int(removed[start:stop].sum())
        if remaining <= 0:
            continue
        local = np.arange(start, stop)
        eligible = local[removed[start:stop] < raw.data[start:stop]]
        order = np.lexsort((gene_keys[raw.indices[eligible]], -fractions[eligible]))
        chosen = eligible[order[:remaining]]
        removed[chosen] += 1
    corrected.data = raw.data.astype(np.int64) - removed
    corrected.eliminate_zeros()
    return corrected


def _selected_data_positions(x, rows: np.ndarray):
    """CSR map from selected matrix entries to their positions in ``x.data``."""
    rows = np.asarray(rows, dtype=np.int64)
    counts = np.diff(x.indptr)[rows]
    total = int(counts.sum())
    row_counts = np.zeros(x.shape[0], dtype=np.int64)
    row_counts[rows] = counts
    indptr = np.concatenate([[0], np.cumsum(row_counts)])
    if total == 0:
        return sparse.csr_matrix(x.shape, dtype=np.int64)
    packed_offsets = np.repeat(np.cumsum(counts) - counts, counts)
    data_idx = np.repeat(x.indptr[rows], counts) + np.arange(total, dtype=np.int64) - packed_offsets
    return sparse.csr_matrix(
        (data_idx, x.indices[data_idx], indptr),
        shape=x.shape,
        copy=False,
    )


def _expand_take_to_cells(
    x,
    idx: np.ndarray,
    take_g: np.ndarray,
    weights: np.ndarray,
    *,
    data_positions=None,
    cell_keys=None,
) -> None:
    """Write type-level take through sparse gene columns onto CSR data."""
    gidx = np.flatnonzero(take_g > 1e-12)
    if gidx.size == 0 or idx.size == 0:
        return
    idx = np.asarray(idx, dtype=np.int64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    keys = (
        np.arange(idx.size).astype(str) if cell_keys is None else np.asarray(cell_keys).astype(str)
    )
    if keys.shape != (idx.size,):
        raise ValueError("cell_keys must match idx")
    if not pd.Index(keys).is_unique:
        raise ValueError("cell_keys must be unique after string conversion")
    x.sort_indices()
    if data_positions is None:
        data_positions = sparse.csr_matrix(
            (
                np.arange(x.nnz, dtype=np.int64),
                x.indices,
                x.indptr,
            ),
            shape=x.shape,
            copy=False,
        )
    chunk_size = 3000
    for start in range(0, gidx.size, chunk_size):
        chunk = gidx[start : start + chunk_size]
        positions = data_positions[idx][:, chunk].tocsc()
        for j, gene in enumerate(chunk):
            a, b = int(positions.indptr[j]), int(positions.indptr[j + 1])
            if a == b:
                continue
            data_idx = positions.data[a:b]
            values = x.data[data_idx]
            has_room = values > 0
            if not has_room.any():
                continue
            local_rows = positions.indices[a:b][has_room]
            room_i = np.rint(values[has_room]).astype(np.int64)
            local_weights = np.clip(w[local_rows], 0.0, None)
            take = _alloc_integer_budget_validated(
                float(take_g[gene]),
                room_i,
                local_weights,
                keys[local_rows],
            )
            x.data[data_idx[has_room]] = room_i - take
