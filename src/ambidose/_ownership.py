"""Dominant-gene ownership and native-gene protection masks."""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ._dose import SOUP_ONLY_MAX_RT, U_MAX_LIBRARY_FRAC, _top_chi_indices, _unexpressed_mask
from ._shared import (
    _DENSE_WORKSPACE_RAM_FRACTION,
    EMPTY_TYPES,
    MIN_PROTECTED_CHI,
    MIN_TYPE_CELLS,
    NATIVE_SOUP_RATIO,
    OWNER_FRAGMENT_MIN_SHARE_FOLD,
    OWNER_MIN_FOLD,
    OWNER_TOP_K,
    _available_ram_bytes,
    _dense_chunk_columns,
    _require_ram,
    _thread_worker_count,
)

_MT_GENE_RE = re.compile(r"(^|_)mt[-_]", re.IGNORECASE)


def _mt_gene_mask(var_names) -> np.ndarray:
    """Identify mitochondrial symbols while excluding MTOR/MT1A-like names.

    Mitochondrial genes are protected independently of type ownership because
    their expression is often shared across cell types.
    """
    return np.array([bool(_MT_GENE_RE.search(str(n))) for n in var_names])


CROSS_TYPE_ANCHOR_GAP = 3.0
CROSS_TYPE_ANCHOR_K = 2.5
CROSS_TYPE_ANCHOR_MIN_RHO = 0.1
CROSS_TYPE_ANCHOR_MIN_ADJUSTED_R2 = 0.01
STRUCTURE_CHUNK = 3000


def _structure_chunk_genes(n_cells: int, n_cand: int) -> int:
    """Gene block size for target/fitted/LAPACK and residual temporaries."""
    return _dense_chunk_columns(
        n_cells,
        n_cand,
        arrays_per_value=5,
        max_cols=STRUCTURE_CHUNK,
    )


def _cross_cell_structure_mask(
    x,
    n: np.ndarray,
    idx: np.ndarray,
    candidates: np.ndarray,
    reference_genes: np.ndarray,
    *,
    min_adjusted_r2: float = CROSS_TYPE_ANCHOR_MIN_ADJUSTED_R2,
    n_components: int = 20,
    chunk_size: int | None = None,
    solver: str = "qr",
    n_jobs: int | None = None,
) -> np.ndarray:
    """Select genes explained by cell programs learned without the candidates.

    Candidate columns are independent in the least-squares fit, so they are
    processed in gene blocks instead of one dense ``n_cells × n_candidates``
    array.
    """
    keep = reference_genes.copy()
    keep[candidates] = False
    n_comp = min(n_components, idx.size - 2, int(keep.sum()) - 1)
    passed = np.zeros(x.shape[1], dtype=bool)
    if candidates.size == 0 or n_comp < 2:
        return passed

    from sklearn.decomposition import TruncatedSVD

    scale = np.divide(1e4, n[idx], out=np.zeros(idx.size, dtype=np.float64), where=n[idx] > 0)
    z = x[idx].multiply(scale[:, None]).tocsr()
    z.data = np.log1p(z.data)
    scores = TruncatedSVD(n_components=n_comp, random_state=0).fit_transform(z[:, keep])
    design = np.column_stack([np.ones(idx.size), scores])
    if solver not in {"qr", "lstsq"}:
        raise ValueError("solver must be 'qr' or 'lstsq'")
    q = None
    if solver == "qr":
        q, r = np.linalg.qr(design, mode="reduced")
        if np.linalg.matrix_rank(r) < r.shape[0]:
            solver = "lstsq"
    n_cand = int(candidates.size)
    min_task_genes = 16
    max_useful_tasks = max(1, (n_cand + min_task_genes - 1) // min_task_genes)
    if chunk_size is not None:
        block = max(1, int(chunk_size))
        workers = _thread_worker_count(
            n_jobs,
            n_tasks=(n_cand + block - 1) // block,
            per_worker_bytes=idx.size * block * 8 * 5,
        )
    else:
        workers = _thread_worker_count(
            n_jobs,
            n_tasks=max_useful_tasks,
            per_worker_bytes=idx.size * min_task_genes * 8 * 5,
        )
        available = _available_ram_bytes()
        memory_block = STRUCTURE_CHUNK
        if available is not None:
            memory_block = max(
                1,
                int(available * _DENSE_WORKSPACE_RAM_FRACTION)
                // workers
                // max(1, idx.size * 8 * 5),
            )
        balanced_block = (n_cand + workers - 1) // workers
        block = min(STRUCTURE_CHUNK, memory_block, balanced_block)
    block = max(1, block)
    r2 = np.zeros(n_cand, dtype=np.float64)

    def run_block(start: int) -> tuple[int, np.ndarray]:
        sl = candidates[start : start + block]
        _require_ram(
            idx.size * int(sl.size) * 8 * 5 * workers,
            where="parallel cross-type structure protection",
        )
        target = z[:, sl].toarray()
        if solver == "qr":
            fitted = q @ (q.T @ target)
        else:
            fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        denom = np.square(target - target.mean(axis=0)).sum(axis=0)
        valid = denom > 0
        out = np.zeros(sl.size, dtype=np.float64)
        out[valid] = 1.0 - np.square(target[:, valid] - fitted[:, valid]).sum(axis=0) / denom[valid]
        if solver == "qr" and valid.any():
            factor = (idx.size - 1) / (idx.size - n_comp - 1)
            adjusted = 1.0 - (1.0 - out) * factor
            verify = valid & (np.abs(adjusted - min_adjusted_r2) <= 1e-10)
            if verify.any():
                exact_target = target[:, verify]
                exact_fitted = design @ np.linalg.lstsq(design, exact_target, rcond=None)[0]
                out[verify] = (
                    1.0 - np.square(exact_target - exact_fitted).sum(axis=0) / denom[verify]
                )
        return start, out

    try:
        starts = list(range(0, n_cand, block))
        if workers <= 1 or len(starts) <= 1:
            for start in starts:
                pos, values = run_block(start)
                r2[pos : pos + values.size] = values
        else:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as executor:
                for pos, values in executor.map(run_block, starts):
                    r2[pos : pos + values.size] = values
    except MemoryError as exc:
        if str(exc).startswith("ambidose:"):
            raise
        raise MemoryError(
            "ambidose: cross-type structure protection ran out of memory "
            f"on {idx.size} cells x {n_cand} genes. Use Cell "
            "Ranger filtered barcodes so empty droplets are not treated as "
            "cells, or run with more memory"
        ) from exc
    adjusted_r2 = 1.0 - (1.0 - r2) * (idx.size - 1) / (idx.size - n_comp - 1)
    passed[candidates[adjusted_r2 >= min_adjusted_r2]] = True
    return passed


def _confident_owner_mask(
    type_means: dict[str, np.ndarray],
    n_genes: int,
    *,
    max_type_mean: float,
    gap: float = CROSS_TYPE_ANCHOR_GAP,
) -> dict[str, np.ndarray]:
    """Per type, True for genes where that type is the top expresser by at
    least a ``gap``-fold multiplicative margin over the runner-up.

    Deliberately stricter than ``_dominant_owner_masks``'s plain argmax test
    (which allows near-ties): this pool anchors ``_cross_type_anchor_mask``'s
    independent ``rho_t`` estimate below, so a housekeeping-shaped gene
    sneaking in via a noise-driven near-tie would bias that anchor upward
    toward the gene's own (non-ambient) level, defeating the mechanism.
    Confirmed via toy simulation before this gap requirement was added:
    without it, rho_t_anchor came out ~0.65 regardless of the true
    simulated rho, because housekeeping genes (uniformly ~equal across
    types) leaked into the "confirmed owner elsewhere" pool via ties.
    """
    if len(type_means) < 2:
        return {t: np.zeros(n_genes, dtype=bool) for t in type_means}
    stacked = np.vstack(list(type_means.values()))
    sorted_desc = np.sort(stacked, axis=0)[::-1]
    top1 = sorted_desc[0]
    top2 = sorted_desc[1]
    return {
        t: (mean_t >= top1) & (top1 > max_type_mean) & (top1 >= gap * np.maximum(top2, 1e-9))
        for t, mean_t in type_means.items()
    }


def _cross_type_anchor_mask(
    x,
    n: np.ndarray,
    chi: np.ndarray,
    types_s: np.ndarray,
    type_means_s: dict[str, np.ndarray],
    u_masks_s: dict[str, np.ndarray],
    *,
    type_indices: dict[str, np.ndarray] | None = None,
    max_type_mean: float,
    min_chi: float,
    k: float = CROSS_TYPE_ANCHOR_K,
    min_rho_t_anchor: float = CROSS_TYPE_ANCHOR_MIN_RHO,
    n_jobs: int | None = None,
) -> dict[str, np.ndarray]:
    """Extra native-gene protection per type, via an independently-anchored
    cross-type ``rho_t`` estimate (sixth #77 candidate, CHANGELOG).

    ``_dominant_owner_masks`` protects a gene in the empirically calibrated owner groups where it is
    the global argmax; a gene that is ambiguously mid-level in several types
    at once (the actual housekeeping-gene bias mechanism, #77) is not the
    argmax anywhere and gets no protection at all. This mechanism adds a
    second, independent test per type: build an "anchor pool" of genes with
    a *confident* owner in some OTHER type (``_confident_owner_mask``, so the
    anchor itself cannot be contaminated by the ambiguous genes this is
    trying to protect), estimate that type's own ambient dose ``rho_t``
    purely from those anchor genes, and flag any of this type's own
    unexpressed-by-the-ambient-ceiling-test candidate genes as native
    (protected) if its apparent per-gene ratio exceeds ``k`` times that
    independent anchor -- evidence the gene carries real signal beyond what
    this type's own ambient dose would explain.

    ``min_rho_t_anchor`` prevents unstable ratio tests when the independently
    estimated ambient fraction is near zero. The 0.1 default separates the
    low-signal barnyard cases from the cell-type structure seen in PBMC data.

    ``types_s``: the full-length per-cell type label array, pre-masked to
    the current sample group (non-members set to a value not equal to any
    real type name) -- so ``np.flatnonzero(types_s == t)`` yields indices
    directly into the full ``x``/``n``/``chi`` arrays, matching every other
    per-sample-group loop in this module.
    """
    n_genes = chi.size

    def _idx(t: str) -> np.ndarray:
        if type_indices is not None:
            return type_indices[t]
        return np.flatnonzero(types_s == t)

    # Ownership and extra-clear already ignore groups smaller than
    # MIN_TYPE_CELLS. The anchor pool must use the same catalog: a 2–9
    # cell Leiden fragment can spike one gene by Poisson noise and pass
    # the 3-fold confident-owner gap, which then pollutes every other
    # type's rho_t_anchor.
    usable_means = {
        t: mean_t for t, mean_t in type_means_s.items() if _idx(t).size >= MIN_TYPE_CELLS
    }
    conf_owner = _confident_owner_mask(usable_means, n_genes, max_type_mean=max_type_mean)
    has_owner_anywhere = np.zeros(n_genes, dtype=bool)
    for mask in conf_owner.values():
        has_owner_anywhere |= mask

    extra: dict[str, np.ndarray] = {}
    for t, mean_t in type_means_s.items():
        idx = _idx(t)
        if idx.size < MIN_TYPE_CELLS:
            extra[t] = np.zeros(n_genes, dtype=bool)
            continue
        u_t = u_masks_s[t]
        owned_elsewhere = has_owner_anywhere & ~conf_owner[t]
        anchor_pool = np.flatnonzero(u_t & owned_elsewhere & (chi >= min_chi))
        protect = np.zeros(n_genes, dtype=bool)
        if anchor_pool.size >= 8:
            y_anchor = float(np.asarray(x[idx][:, anchor_pool].sum()))
            mass_anchor = float((n[idx, None] * chi[anchor_pool][None, :]).sum())
            rho_t_anchor = y_anchor / mass_anchor if mass_anchor > 0 else float("nan")
            if np.isfinite(rho_t_anchor) and rho_t_anchor >= min_rho_t_anchor:
                n_bar_t = float(n[idx].mean())
                cand = np.flatnonzero(u_t & (chi >= min_chi))
                has_structure = _cross_cell_structure_mask(
                    x,
                    n,
                    idx,
                    cand,
                    has_owner_anywhere,
                    n_jobs=n_jobs,
                )
                r_t = mean_t[cand] / (n_bar_t * chi[cand])
                protect[cand[(r_t > k * rho_t_anchor) & has_structure[cand]]] = True
        extra[t] = protect
    return extra


def _type_means(x, types: np.ndarray) -> dict[str, np.ndarray]:
    """Per-type mean expression, GLOBAL across the whole object (all samples).

    One evaluation of the type catalog per ``subtract()`` call, matching the
    project's own "aggregate by label across the current object" convention
    -- computed once and reused for every (sample, type) pair in the main
    loop, not recomputed per sample.
    """
    means: dict[str, np.ndarray] = {}
    for t in pd.unique(types):
        if t in EMPTY_TYPES:
            continue
        idx_t = np.flatnonzero(types == t)
        if idx_t.size < MIN_TYPE_CELLS:
            continue
        means[t] = np.asarray(x[idx_t].mean(axis=0)).ravel()
    return means


def _type_sd(x, idx: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Bessel-corrected per-gene sample sd on one type's cells."""
    nt = idx.size
    sq = x[idx].copy()
    sq.data **= 2
    m2 = np.asarray(sq.mean(axis=0)).ravel()
    var = np.maximum(m2 - mean**2, 0.0) * (nt / max(nt - 1, 1))
    return np.sqrt(var)


def _exclusive_owner_masks(
    group_means: np.ndarray,
    type_means: dict[str, np.ndarray],
    member_of: np.ndarray,
    *,
    max_type_mean: float,
) -> dict[str, np.ndarray]:
    """Owner group(s) per gene: a bottom-up gap cascade, not just the
    single global argmax.

    The same broad cell type can land in several meta-groups at very
    different expression magnitudes for the same gene (e.g. erythroid
    maturation stages differ an order of magnitude in hemoglobin) --
    requiring a single global winner left every meta-group but the largest
    fully unprotected, even when each one is, on its own, far above any
    genuinely different, non-owning type's level.

    Comparing every candidate against the single lowest group's mean was
    tried first and rejected: on real data it also let genuinely
    off-target, contamination-level groups qualify whenever the true
    minimum happened to sit near zero, which is common -- that widened
    the FOLD bar into triviality and measurably *worsened* kidney marker
    leak_ratio (0.0064 -> 0.0082, see CHANGELOG). Instead, sort each
    gene's group means descending and scan gaps from the bottom up: the
    owner/background boundary is the lowest-ranked adjacent pair whose
    ratio already clears ``OWNER_MIN_FOLD``; every group at or above that
    rank shares ownership, everything below it does not. This reduces to
    the original single-winner rule when there is one clear leader over a
    tight background cluster, and extends it correctly when several
    groups (fragmented same-identity clusters) all sit well above the
    lowest, undifferentiated tier.

    A winning meta-group's eligibility is shared by every fragment merged
    into it, but a fragment only *inherits* that share if its own raw mean
    is within ``OWNER_FRAGMENT_MIN_SHARE_FOLD`` of the meta-group's
    strongest individual fragment (see that constant's docstring). Without
    this, a biologically distinct, low-expressing fragment that
    complete-linkage merged in only because its own profile was too noisy
    to separate (small n -> large split-half noise -> a wide compatibility
    window) rides along on a real marker-expressing fragment's grant and
    gets unconditional protection (``native_confidence=1.0``) for a gene it
    doesn't actually express -- directly protecting ambient contamination
    of an off-target gene from subtraction.
    """
    names = list(type_means)
    stacked = np.asarray(group_means)
    n_genes = next(iter(type_means.values())).size
    if stacked.ndim != 2 or stacked.shape[0] < 2:
        eligible = np.zeros((1, n_genes), dtype=bool)
    else:
        n_meta = stacked.shape[0]
        order = np.argsort(-stacked, axis=0)
        sorted_vals = np.take_along_axis(stacked, order, axis=0)
        cutoff_rank = np.full(n_genes, -1, dtype=np.int64)
        for r in range(n_meta - 1, 0, -1):
            ratio_ok = sorted_vals[r - 1] >= OWNER_MIN_FOLD * np.maximum(sorted_vals[r], 1e-9)
            newly_set = ratio_ok & (cutoff_rank == -1)
            cutoff_rank[newly_set] = r - 1
        top_val = sorted_vals[np.clip(cutoff_rank, 0, n_meta - 1), np.arange(n_genes)]
        valid = (cutoff_rank >= 0) & (top_val > max_type_mean)
        ranks = np.arange(n_meta)[:, None]
        eligible_sorted = (ranks <= cutoff_rank[None, :]) & valid[None, :]
        eligible = np.zeros((n_meta, n_genes), dtype=bool)
        np.put_along_axis(eligible, order, eligible_sorted, axis=0)
    n_meta_actual = stacked.shape[0] if stacked.ndim == 2 else 1
    group_max = np.zeros((n_meta_actual, n_genes), dtype=np.float64)
    for i, name in enumerate(names):
        m = int(member_of[i])
        group_max[m] = np.maximum(group_max[m], type_means[name])
    return {
        name: eligible[int(member_of[i])]
        & (type_means[name] > max_type_mean)
        & (type_means[name] * OWNER_FRAGMENT_MIN_SHARE_FOLD >= group_max[int(member_of[i])])
        for i, name in enumerate(names)
    }


def _single_winner_owner_masks(
    group_means: np.ndarray,
    type_means: dict[str, np.ndarray],
    member_of: np.ndarray,
    *,
    max_type_mean: float,
) -> dict[str, np.ndarray]:
    """Original single global-argmax winner rule (the frozen baseline
    before the gap cascade). Used alongside it, not instead of it: see
    ``_dominant_owner_masks(..., also_single_winner=True)`` and
    ``_realloc_unspent_rank1``'s ``leftover_cap`` -- the gap cascade's
    wider ownership is computed and used for the actual take, but this
    narrower rule bounds how much *extra* leftover dose-budget that wider
    ownership is allowed to create for realloc to redirect elsewhere, so
    protecting more fragmented markers can't push realloc's total
    footprint past what the already-validated frozen baseline had (see
    CHANGELOG).
    """
    names = list(type_means)
    stacked = np.asarray(group_means)
    if stacked.ndim != 2 or stacked.shape[0] < 2:
        winner = np.zeros(next(iter(type_means.values())).size, dtype=np.int64)
        exclusive = np.zeros(winner.size, dtype=bool)
    else:
        winner = stacked.argmax(axis=0)
        ranked = np.sort(stacked, axis=0)[::-1]
        exclusive = (ranked[0] > max_type_mean) & (
            ranked[0] >= OWNER_MIN_FOLD * np.maximum(ranked[1], 1e-9)
        )
    return {
        name: (winner == int(member_of[i])) & exclusive & (type_means[name] > max_type_mean)
        for i, name in enumerate(names)
    }


def _topk_owner_masks(
    group_means: np.ndarray,
    type_means: dict[str, np.ndarray],
    member_of: np.ndarray,
    *,
    max_type_mean: float,
    top_k: int = OWNER_TOP_K,
) -> dict[str, np.ndarray]:
    """NOT called from ``_dominant_owner_masks`` -- kept as a documented,
    fully-tried negative result, not live code. See CHANGELOG for the full
    writeup; summary below.

    Owner group(s) per gene: gap-cascade eligible (genuine cross-group
    specificity), capped to that group's own top ``top_k`` among those.

    A pure self-referential magnitude floor (a multiple of the group's own
    median candidate mean) was tried and rejected: a real, specific marker
    isn't necessarily the group's own highest-expressed gene -- coexisting
    non-specific, uniformly high genes (housekeeping-like) can sit above it
    within the same group, so comparing a candidate against its own
    group's median wrongly disqualifies genes that are clearly specific
    when compared against *other* groups. Specificity is inherently a
    cross-group question, not a within-group one. Rank alone (no floor at
    all) was tried first and rejected too: with a small candidate pool,
    "top-K" degenerates to "every candidate."

    This version reuses the already-validated gap-cascade eligibility test
    (see ``_exclusive_owner_masks``, safe on must-win, multi-owner-capable,
    the one actually wired in) as the specificity floor, then additionally
    caps each group to its own top ``top_k`` genes among what it's
    eligible for. Rejected for a different reason than the first two
    attempts: any fixed absolute ``top_k`` is the wrong scale whenever
    ownership breadth varies by orders of magnitude across contexts.
    Verified directly on Mixture (barnyard): unrestricted gap-cascade
    correctly gives hg19 10,551 owned genes and mm10 9,438 (real
    species-exclusive gene counts, not noise), so any small top_k (30, 50)
    truncates ~99.5% of genuinely-owned genes and collapsed Mixture's
    barnyard precision 0.90 -> 0.55 -- not a small-candidate-pool
    degenerate case (that was ruled out by direct inspection: both
    hg19/mm10 hit exactly top_k=50, i.e. the cap bound, not the candidate
    count). A real cell-type marker panel (kidney/fetal liver COARSE_MARKERS
    scale) and two whole non-overlapping species transcriptomes are
    ownership problems that differ by orders of magnitude in scale; no
    single fixed K serves both.
    """
    names = list(type_means)
    stacked = np.asarray(group_means)
    n_genes = next(iter(type_means.values())).size
    if stacked.ndim != 2 or stacked.shape[0] < 2:
        eligible = np.zeros((1, n_genes), dtype=bool)
    else:
        n_meta = stacked.shape[0]
        order = np.argsort(-stacked, axis=0)
        sorted_vals = np.take_along_axis(stacked, order, axis=0)
        cutoff_rank = np.full(n_genes, -1, dtype=np.int64)
        for r in range(n_meta - 1, 0, -1):
            ratio_ok = sorted_vals[r - 1] >= OWNER_MIN_FOLD * np.maximum(sorted_vals[r], 1e-9)
            newly_set = ratio_ok & (cutoff_rank == -1)
            cutoff_rank[newly_set] = r - 1
        top_val = sorted_vals[np.clip(cutoff_rank, 0, n_meta - 1), np.arange(n_genes)]
        valid = (cutoff_rank >= 0) & (top_val > max_type_mean)
        ranks = np.arange(n_meta)[:, None]
        gap_eligible_sorted = (ranks <= cutoff_rank[None, :]) & valid[None, :]
        gap_eligible = np.zeros((n_meta, n_genes), dtype=bool)
        np.put_along_axis(gap_eligible, order, gap_eligible_sorted, axis=0)

        eligible = np.zeros((n_meta, n_genes), dtype=bool)
        for m in range(n_meta):
            candidates = np.flatnonzero(gap_eligible[m])
            if candidates.size == 0:
                continue
            row = stacked[m]
            k = min(top_k, candidates.size)
            top_idx = candidates[np.argpartition(-row[candidates], k - 1)[:k]]
            eligible[m, top_idx] = True
    return {
        name: eligible[int(member_of[i])] & (type_means[name] > max_type_mean)
        for i, name in enumerate(names)
    }


def _split_noise_meta_ids(
    x,
    n: np.ndarray,
    names: list[str],
    type_indices: dict[str, np.ndarray],
    *,
    cell_keys=None,
    n_splits: int = 8,
) -> np.ndarray:
    """Complete-linkage meta ids: merge fragments whose profiles sit in split noise.

    Half-splits estimate each group's whole-profile sampling variation.
    Two groups merge only if every pair in the resulting cluster is within
    ``min`` of their split noises. Used both to share gene ownership inside
    a meta-group and to relabel Leiden leaves for subtract identity.
    """
    n_names = len(names)
    if n_names <= 1:
        return np.zeros(n_names, dtype=np.int64)
    stable_cells = (
        np.arange(x.shape[0]).astype(str)
        if cell_keys is None
        else np.asarray(cell_keys).astype(str)
    )
    norm_means = []
    profile_noise = []
    for name in names:
        idx = np.asarray(type_indices[name], dtype=np.int64)
        idx = idx[np.argsort(stable_cells[idx], kind="stable")]
        seed = int.from_bytes(hashlib.blake2b(str(name).encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        scale = np.divide(1e4, n[idx], out=np.zeros(idx.size), where=n[idx] > 0)
        scaled = x[idx].multiply(scale[:, None]).tocsr()
        mean = np.asarray(scaled.mean(axis=0)).ravel()
        norm_means.append(mean)
        distance = 0.0
        if idx.size >= 4:
            for _ in range(n_splits):
                order = rng.permutation(idx.size)
                cut = idx.size // 2
                left = np.asarray(scaled[order[:cut]].mean(axis=0)).ravel()
                right = np.asarray(scaled[order[cut:]].mean(axis=0)).ravel()
                lp = np.log1p(left)
                rp = np.log1p(right)
                denom = float(np.linalg.norm(lp) * np.linalg.norm(rp))
                split_distance = 1.0 - float(lp @ rp) / denom if denom > 0 else 1.0
                distance = max(distance, split_distance)
        profile_noise.append(distance)
    means = np.vstack(norm_means)
    profiles = np.log1p(means)
    norms = np.linalg.norm(profiles, axis=1)
    denom = norms[:, None] * norms[None, :]
    similarity = np.divide(
        profiles @ profiles.T,
        denom,
        out=np.zeros((n_names, n_names)),
        where=denom > 0,
    )
    noise = np.maximum(np.asarray(profile_noise, dtype=np.float64), 1e-3)
    compatible = (1.0 - similarity) <= np.minimum.outer(noise, noise)
    np.fill_diagonal(compatible, True)
    return _complete_linkage_labels(compatible, keys=names)


def _complete_linkage_labels(compatible: np.ndarray, *, keys=None) -> np.ndarray:
    """Complete-linkage merge with a stable identity order."""
    n = compatible.shape[0]
    stable_keys = np.arange(n).astype(str) if keys is None else np.asarray(keys).astype(str)
    if stable_keys.shape != (n,):
        raise ValueError("keys must match compatibility matrix")
    if not pd.Index(stable_keys).is_unique:
        raise ValueError("keys must be unique after string conversion")
    order = np.argsort(stable_keys, kind="stable")
    compatible = compatible[np.ix_(order, order)]
    labels = np.arange(n, dtype=np.int64)
    merged = True
    while merged:
        merged = False
        uniq = np.unique(labels)
        for a_i, a in enumerate(uniq):
            members_a = np.flatnonzero(labels == a)
            for b in uniq[a_i + 1 :]:
                members_b = np.flatnonzero(labels == b)
                if not compatible[np.ix_(members_a, members_b)].all():
                    continue
                labels[labels == b] = a
                merged = True
                break
            if merged:
                break
    _, labels = np.unique(labels, return_inverse=True)
    restored = np.empty(n, dtype=np.int64)
    restored[order] = labels
    return restored


def _dominant_owner_masks(
    x,
    n: np.ndarray,
    types: np.ndarray,
    type_means: dict[str, np.ndarray],
    *,
    type_indices: dict[str, np.ndarray] | None = None,
    cell_keys=None,
    max_type_mean: float,
    n_splits: int = 8,
    also_single_winner: bool = False,
) -> tuple[dict[str, np.ndarray], int] | tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Owners after complete-linkage merge of groups inside split noise.

    Half-splits estimate each group's whole-profile sampling variation.
    Two groups merge only if *every* pair in the resulting cluster is
    within ``min`` of their split noises (complete linkage, not a
    max-noise single-linkage chain). Ownership is exclusive between the
    resulting meta-groups and shared by fragments inside a winner.
    A gene with no unique owner (near-tie across metas, including unowned
    injection genes) is not protected. If merge collapses to one
    meta-group, fall back to per-fragment argmax with the same uniqueness
    fold so identity genes still have an owner. A sample with only one
    group of at least ``MIN_TYPE_CELLS`` cells owns every gene above
    ``max_type_mean`` in that group.

    ``also_single_winner=True`` additionally computes ``_single_winner_owner_masks``
    on the same (expensive to compute) meta-groups and returns it as a
    second dict -- ``(masks, masks_single_winner, n_meta)`` instead of
    ``(masks, n_meta)`` -- so a caller can bound how much *extra* leftover
    the wider gap-cascade ownership creates for ``_realloc_unspent_rank1``
    relative to the frozen single-winner baseline, without paying for
    ``_split_noise_meta_ids`` twice.
    """
    type_means = {name: type_means[name] for name in sorted(type_means)}
    names = list(type_means)
    n_genes = x.shape[1]
    empty = {name: np.zeros(n_genes, dtype=bool) for name in names}
    if len(names) == 0:
        return (empty, empty, 0) if also_single_winner else (empty, 0)
    idx_map = (
        {name: type_indices[name] for name in names}
        if type_indices is not None
        else {name: np.flatnonzero(types == name) for name in names}
    )
    if len(names) == 1:
        name = names[0]
        idx = idx_map[name]
        n_bar = float(n[idx].mean()) if idx.size else 0.0
        floor = max(float(max_type_mean), U_MAX_LIBRARY_FRAC * n_bar)
        masks = {name: type_means[name] > floor}
        return (masks, masks, 1) if also_single_winner else (masks, 1)

    raw_means = np.vstack([type_means[name] for name in names])
    sizes = [int(idx_map[name].size) for name in names]
    labels = _split_noise_meta_ids(x, n, names, idx_map, cell_keys=cell_keys, n_splits=n_splits)
    n_meta = int(np.unique(labels).size)

    if n_meta < 2:
        group_means, member_of = raw_means, np.arange(len(names))
    else:
        meta_means = []
        for lab in range(n_meta):
            members = np.flatnonzero(labels == lab)
            meta_means.append(
                np.average(raw_means[members], axis=0, weights=np.asarray(sizes)[members])
            )
        group_means, member_of = np.vstack(meta_means), labels

    masks = _exclusive_owner_masks(group_means, type_means, member_of, max_type_mean=max_type_mean)
    if not also_single_winner:
        return masks, n_meta
    masks_sw = _single_winner_owner_masks(
        group_means, type_means, member_of, max_type_mean=max_type_mean
    )
    return masks, masks_sw, n_meta


def _p_set_is_soup_like(
    x, n: np.ndarray, chi: np.ndarray, idx: np.ndarray, is_p: np.ndarray
) -> bool:
    """True if leftover mass on P genes is near the empty-droplet χ ceiling."""
    p = np.flatnonzero(is_p)
    if p.size == 0 or idx.size == 0:
        return False
    chi_p = float(chi[p].sum())
    if chi_p < MIN_PROTECTED_CHI:
        return False
    remain = float(np.asarray(x[idx][:, p].sum()))
    soup = float(n[idx].sum()) * chi_p
    return soup > 0 and remain <= NATIVE_SOUP_RATIO * soup


def _type_masks(
    x,
    n: np.ndarray,
    chi: np.ndarray,
    idx: np.ndarray,
    *,
    max_type_mean: float,
    min_chi: float,
    top_n: int,
    exclude: np.ndarray | None = None,
    empirical_margin: bool = False,
    collision_exception: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unexpressed genes, diagnostic native mask, native confidence, and r_t.

    ``r_t`` (``mean/expected``) is also returned so callers can down-weight
    leftover-budget reallocation onto genes that failed the significance
    test but still sit well above the ambient ceiling (see
    ``_realloc_unspent_rank1``'s ``r_t`` parameter) -- a low-confidence gene
    with r_t far above 1 looks more like an under-powered real marker than
    like ambient, even though it didn't clear ``native_confidence``.

    U is ranked by χ among unexpressed genes (same rule as dose). Genes
    sitting on the ρ=1 ceiling (``mean/(n̄χ) > 0.85``), or abundant genes
    closer to that ceiling than to true soup (``mean > U_MAX_LIBRARY_FRAC
    · n̄`` and ``mean/(n̄χ) > 0.5``), are not U. Extra-clear is further
    restricted to ``mean/(n̄χ) < SOUP_ONLY_MAX_RT`` so mid-ceiling
    leftovers are not wiped. True soup has ``mean/(n̄χ) ≈ ρ``. Do not intersect with the global top-χ set: those are
    the majority type's markers, so the intersection is empty for the type
    that dominates soup.
    ``exclude`` (e.g. this type's dominantly-expressed genes) is never
    eligible for U even if its mean sits at/below the ambient ceiling --
    also folded into ``is_p`` so it gets the native-gene protection in
    ``_expand_take_to_cells`` too, not just soupOnly exemption.

    ``empirical_margin`` uses the observed per-gene SD. It requires
    ``exclude`` so highly variable native genes cannot enter the ambient pool.

    A per-cell-dose-based ambient baseline (``mean(d_c)*chi`` instead of
    ``n̄*chi``) was tried here and reverted -- see CHANGELOG. It correctly
    rescued real markers that are also a sample's dominant ambient
    contaminant (e.g. hemoglobin in blood-rich tissue) in their own true
    cluster, but had no way to distinguish that case from a gene that is
    genuinely uniform ambient with a chi-dominant, dose-underestimated
    profile: both look "significant" under the dose baseline in most or
    all types simultaneously, not just the true owner, so a cross-type
    uniformity gate (reject the rescue when 2+ types independently pass)
    does not separate them either -- verified directly (HBA1/HBA2/ALAS2/
    GYPA passed the dose-based test in 4-6 of 6 real cell types on a
    fetal-liver sample, not just Erythroid) and broke a synthetic test's
    genuinely-uniform ambient gene (`den[:,2]` dropped from an expected 0
    to 38-39 of 40, essentially unremoved).
    """
    mean = np.asarray(x[idx].mean(axis=0)).ravel()
    expected = float(n[idx].mean()) * chi
    dominant_exclusion_applied = exclude is not None
    if empirical_margin and not dominant_exclusion_applied:
        raise ValueError(
            "_type_masks: empirical_margin=True requires exclude to be given "
            "(a dominant-type exclusion mask) -- without it, the widened "
            "margin has no safety net against real highly-expressed, "
            "highly-variable native genes being reclassified as ambient."
        )
    margin_sd = None
    if empirical_margin:
        # Bessel-corrected sample SD without densifying the cell-by-gene block.
        margin_sd = _type_sd(x, idx, mean)
    unexp = _unexpressed_mask(
        mean,
        expected,
        idx.size,
        max_type_mean=max_type_mean,
        margin_sd=margin_sd,
        dominant_exclusion_applied=dominant_exclusion_applied,
    )
    if margin_sd is not None:
        se = margin_sd / np.sqrt(idx.size)
    else:
        se = np.sqrt(np.maximum(expected, 0.0) / idx.size)
    z = np.divide(mean - expected, se, out=np.zeros_like(mean), where=se > 0)
    z[(se == 0) & (mean > expected)] = np.inf
    # One-sided evidence that expression exceeds ambient. Equality (z=0)
    # is ambient-consistent, so confidence is 0, not 0.5.
    native_confidence = np.clip(2.0 * ndtr(z) - 1.0, 0.0, 1.0)
    native_confidence[mean <= max_type_mean] = 0.0

    n_bar = float(n[idx].mean()) if idx.size else 0.0
    r_t = np.divide(mean, expected, out=np.zeros_like(mean), where=expected > 0)
    if n_bar > 0 and collision_exception:
        abundant_collision = (mean > U_MAX_LIBRARY_FRAC * n_bar) & (r_t > 0.5)
        ceiling = r_t > 0.85
        unexp = unexp & ~abundant_collision & ~ceiling
        native_confidence[abundant_collision] = 1.0
    is_p = ~unexp
    cand_mask = unexp & (chi >= min_chi)
    if n_bar > 0:
        cand_mask = cand_mask & (r_t < SOUP_ONLY_MAX_RT)
    if exclude is not None:
        cand_mask = cand_mask & ~exclude
        is_p = is_p | exclude
        native_confidence[exclude] = 1.0
    cand = np.flatnonzero(cand_mask)
    is_u = np.zeros(chi.size, dtype=bool)
    cand = _top_chi_indices(chi, cand, top_n)
    is_u[cand] = True
    return is_u, is_p, native_confidence, r_t
