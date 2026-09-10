"""Continuous and integer ambient-removal budget allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ._shared import MIN_TYPE_CELLS

# Blend weight for dose-enrichment scaling of unowned rank-1 and soupOnly
# takes. 0 keeps the pooled take; 1 replaces it with take × max(Pearson(y/n, ρ), 0).
ENRICH_STRENGTH = 0.1
# Pre-enrich take/capacity at or above this is type-pooled; below it, rank-1
# take is spent inside each cell so fractional χ-budget is not taken from
# other cells of the same type.
SAT_FRAC = 1.0
# Unbounded soupOnly extra-clear covers this χ-mass prefix of U genes.
# Ownership single-winner stays at CHI_MASS_SINGLE_WINNER (0.15). Wider
# extra-clear recovers soup-dominant markers (e.g. globin) that sit
# outside the 15% prefix and were remaining-d_c capped.
SOUP_ONLY_CHI_MASS = 0.8
# Promoted in 0.3.13: gate _type_masks's abundant_collision/ceiling exception
# (protects a gene whose type mean approaches the ρ=1 ambient ceiling) on
# cross-type specificity via _ownership._ceiling_cross_type_gate (median-of-
# others background comparison), so a type with genuinely elevated per-cell
# ρ can no longer get unconditional protection on pure soup just because no
# other type happens to be compared. See _ownership._type_masks's
# ceiling_gate docstring and DEVLOG.md's 2026-09-09 entries for the full
# design history (a rejected single-winner version, the T-cell/NK false-
# competition root cause, and the median redesign's full-suite evaluation
# before promotion: fetal erythroid Hb and realistic_gt clustering-quality
# regressions eliminated; GSE218853 macro ARS/ERS cost real but halved
# from the rejected version, +2.0pp/-1.6pp, judged acceptable).
CEILING_CROSS_TYPE_GATE = True


def _high_chi_u_mask(
    is_u: np.ndarray, chi: np.ndarray, mass: float = SOUP_ONLY_CHI_MASS
) -> np.ndarray:
    """U genes inside the smallest χ-prefix covering ``mass`` of total χ."""
    from ._ownership import _chi_mass_prefix_mask

    return np.asarray(is_u, dtype=bool) & _chi_mass_prefix_mask(chi, mass)


def _revoke_u_with_expressing_subset(
    x,
    idx: np.ndarray,
    library: np.ndarray,
    chi: np.ndarray,
    is_u: np.ndarray,
    *,
    min_cells: int = MIN_TYPE_CELLS,
) -> np.ndarray:
    """Drop high-χ U when a subset of the type exceeds the type χ ceiling.

    Type-mean U gates miss a native minority merged into a larger type:
    the mean is diluted below n̄χ, soupOnly then wipes the minority.
    Compare to n̄χ, not n_c χ: native cells' own identity UMIs inflate
    n_c so y < n_c χ even when they clearly express the gene.
    """
    is_u = np.asarray(is_u, dtype=bool).copy()
    idx = np.asarray(idx, dtype=np.int64)
    genes = np.flatnonzero(_high_chi_u_mask(is_u, chi))
    if genes.size == 0 or idx.size < min_cells:
        return is_u
    n = np.clip(np.asarray(library, dtype=np.float64)[idx], 1e-12, None)
    block = x[idx][:, genes]
    y = np.asarray(block.toarray() if sparse.issparse(block) else block, dtype=np.float64)
    thresh = float(n.mean()) * np.asarray(chi, dtype=np.float64)[genes]
    n_hot = np.count_nonzero(y > thresh[None, :], axis=0)
    is_u[genes[n_hot >= int(min_cells)]] = False
    return is_u


def _cap_take_to_remaining(take: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    """Scale a type-pooled take so it cannot exceed remaining per-cell dose."""
    take = np.asarray(take, dtype=np.float64).copy()
    cap = float(np.clip(np.asarray(remaining, dtype=np.float64), 0.0, None).sum())
    tgt = float(take.sum())
    if cap <= 0.0 or tgt <= 0.0:
        take[:] = 0.0
        return take
    if tgt > cap:
        take *= cap / tgt
    return take


def _dose_enrichment_scales(
    x, idx, take: np.ndarray, dose: np.ndarray, library: np.ndarray
) -> np.ndarray:
    """Per-gene scale in [0, 1] = max(Pearson(y/n, ρ), 0) within a type."""
    scales = np.ones(take.shape[0], dtype=np.float64)
    genes = np.flatnonzero(take > 1e-12)
    if genes.size == 0 or idx.size < 3:
        return scales
    n = np.clip(np.asarray(library, dtype=np.float64)[idx], 1e-12, None)
    rho = np.asarray(dose, dtype=np.float64)[idx] / n
    rho_c = rho - rho.mean()
    var_rho = float(np.dot(rho_c, rho_c))
    if var_rho <= 0:
        return scales
    block = x[idx][:, genes].tocsr()
    inv_n = 1.0 / n
    rate = block.multiply(inv_n[:, np.newaxis])
    mean_rate = np.asarray(rate.mean(axis=0)).ravel()
    cov = np.asarray(rate.T.dot(rho_c)).ravel()
    sum_sq = np.asarray(rate.multiply(rate).sum(axis=0)).ravel()
    var_rate = np.maximum(sum_sq - idx.size * mean_rate**2, 0.0)
    corr = np.zeros(genes.size, dtype=np.float64)
    ok = var_rate > 0
    corr[ok] = cov[ok] / np.sqrt(var_rate[ok] * var_rho)
    scales[genes] = np.clip(corr, 0.0, None)
    return scales


def _preserve_take_sum(
    take: np.ndarray, mass: float, observed: np.ndarray | None = None
) -> np.ndarray:
    """Scale ``take`` so its sum is ``mass``, optionally clipped to observed."""
    take = np.asarray(take, dtype=np.float64).copy()
    mass = float(mass)
    if mass <= 0:
        take[:] = 0.0
        return take
    tgt = float(take.sum())
    if tgt <= 0:
        return take
    take *= mass / tgt
    if observed is not None:
        take = np.minimum(take, np.asarray(observed, dtype=np.float64))
        tgt = float(take.sum())
        if tgt > mass:
            take *= mass / tgt
    return take


def _apply_dose_enrichment(
    x,
    idx,
    take: np.ndarray,
    dose: np.ndarray,
    library: np.ndarray,
    *,
    strength: float = ENRICH_STRENGTH,
    renormalize: bool = False,
    observed: np.ndarray | None = None,
) -> np.ndarray:
    """Reweight unowned pooled take when counts do not track per-cell dose.

    Default shrinks the type take. ``renormalize`` keeps the original sum
    (responsibility is allocation, not a smaller budget).
    """
    take = np.asarray(take, dtype=np.float64)
    if strength <= 0:
        return take.copy()
    mass = float(take.sum())
    scales = _dose_enrichment_scales(x, idx, take, dose, library)
    out = (1.0 - strength) * take + strength * take * scales
    if renormalize:
        return _preserve_take_sum(out, mass, observed)
    return out


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
    r_t: np.ndarray | None = None,
    *,
    leftover_cap: float | None = None,
) -> np.ndarray:
    """Spend leftover χ-budget on non-protected, non-soupOnly genes.

    Protected genes keep their rank-1 slice. U genes stay on the soupOnly
    path. Leftover is the unused part of ``dose`` after the clipped take.

    ``r_t`` (``mean/expected``, from :func:`_type_masks`) down-weights genes
    whose observed level sits well above the pure-ambient ceiling even
    though they didn't clear the ``native_confidence`` significance test --
    an under-powered real marker (r_t >> 1) should not be treated the same
    as a gene that genuinely looks like ambient (r_t ~= 1) just because
    both failed to reach significance. Without this, leftover mass
    concentrates on whichever unprotected genes have the highest χ,
    regardless of how implausible "this is pure ambient" already looks for
    that specific gene -- the mechanism behind on-target markers (e.g. a
    cell-type's own canonical genes in an under-powered cluster) being
    fully zeroed out by reallocation despite never being flagged is_u.

    A flat per-gene fold cap (a multiple of the gene's own base rank-1
    share) was tried and rejected: on real kidney data, the "legitimate"
    and "harmful" realloc multiples occupy the *same* range (median 2.96x,
    p90 5.76x across ~400k real gene-cluster events) -- there is no
    magnitude threshold that separates them, so any cap tight enough to
    matter for kidney/fetal-liver also costs must-win, and any cap loose
    enough to spare must-win (5x, at the measured p90) does not move
    kidney/fetal-liver's aggregate leak_ratio at all (see CHANGELOG).

    ``leftover_cap``, when given, bounds the *total* leftover actually
    redistributed (not any one gene's share of it) -- the caller computes
    what leftover the frozen single-winner ownership rule would have
    produced and passes it here, so the gap-cascade's wider ownership
    (more genes protected per type) can never push realloc's total
    footprint past what the already-validated baseline had. This is a
    structural cap tied to *why* extra leftover exists (newly-protected
    genes freeing up budget), not to any single gene's magnitude.
    """
    leftover = max(0.0, float(dose) - float(np.sum(take)))
    if leftover_cap is not None:
        leftover = min(leftover, max(0.0, float(leftover_cap)))
    if leftover <= 0:
        return take
    blocked = is_p | is_u
    room = np.where(blocked, 0.0, np.maximum(observed - take, 0.0))
    weights = np.where(blocked, 0.0, chi)
    if r_t is not None:
        weights = weights / np.maximum(1.0, r_t)
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
    raw = raw.tocsr(copy=False)
    corrected = corrected.tocsr(copy=False)
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


def _pre_enrich_sat_mask(
    take: np.ndarray, observed: np.ndarray, sat_frac: float = SAT_FRAC
) -> np.ndarray:
    """Genes whose type-level take already meets ``sat_frac`` of observed capacity."""
    take = np.asarray(take, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    return (take > 1e-12) & (take / np.maximum(observed, 1e-12) >= sat_frac)


def _expand_unsat_cell_carry(
    x,
    idx: np.ndarray,
    take_g: np.ndarray,
    weights: np.ndarray,
    *,
    data_positions,
    gene_names: np.ndarray,
) -> None:
    """Spend unsaturated type take as per-cell budgets across remaining genes."""
    gidx = np.flatnonzero(take_g > 1e-12)
    if gidx.size == 0 or idx.size == 0:
        return
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return
    budget = w * (float(take_g[gidx].sum()) / w_sum)
    gene_w = take_g[gidx]
    names = np.asarray(gene_names).astype(str)[gidx]
    loc = data_positions[idx][:, gidx].tocsr()
    for i in range(idx.size):
        if budget[i] <= 1e-12:
            continue
        a, b = int(loc.indptr[i]), int(loc.indptr[i + 1])
        if a == b:
            continue
        data_idx = loc.data[a:b]
        local_genes = loc.indices[a:b]
        room = np.rint(x.data[data_idx]).astype(np.int64)
        take_i = _alloc_integer_budget(
            float(budget[i]),
            room.astype(float),
            gene_w[local_genes],
            tie_keys=names[local_genes],
        )
        x.data[data_idx] = room - take_i


def _expand_take_to_cells(
    x,
    idx: np.ndarray,
    take_g: np.ndarray,
    weights: np.ndarray,
    *,
    data_positions=None,
    cell_keys=None,
    sat_mask=None,
    gene_names=None,
) -> None:
    """Write type-level take through sparse gene columns onto CSR data.

    ``sat_mask`` True genes stay on the type-level integer allocator.
    The rest of a positive take is spent inside each cell (unsaturated
    rank-1). ``sat_mask is None`` type-pools every gene.
    """
    take_g = np.asarray(take_g, dtype=np.float64)
    if sat_mask is not None:
        sat_mask = np.asarray(sat_mask, dtype=bool)
        if sat_mask.shape != take_g.shape:
            raise ValueError("sat_mask must match take_g")
        if gene_names is None:
            raise ValueError("gene_names required when sat_mask is set")
        take_sat = np.where(sat_mask, take_g, 0.0)
        take_unsat = np.where(~sat_mask, take_g, 0.0)
        _expand_take_to_cells(
            x, idx, take_sat, weights, data_positions=data_positions, cell_keys=cell_keys
        )
        idx = np.asarray(idx, dtype=np.int64)
        x.sort_indices()
        if data_positions is None:
            data_positions = sparse.csr_matrix(
                (np.arange(x.nnz, dtype=np.int64), x.indices, x.indptr),
                shape=x.shape,
                copy=False,
            )
        _expand_unsat_cell_carry(
            x, idx, take_unsat, weights, data_positions=data_positions, gene_names=gene_names
        )
        return
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
