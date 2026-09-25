"""
Graph-structure features: competition and corroboration.

Both exploit facts about the *generating process* that a purely pairwise model cannot see.
The EDA (``docs/05_EDA_FINDINGS.md``) verified them on the real labels, which is what makes
it safe to lean on them:

**Competition (Stage R).** Ground truth contains 7,638,365 links and **zero** Source 2/3
records used by more than one Source 1 entity -- the true match sets are exactly pairwise
disjoint. So every candidate record has at most one legitimate parent, and the
*competition* among Source 1 entities for a candidate is evidence. Normalisation runs down
the **columns** (over the entities competing for one candidate) with a null "dustbin"
option so a candidate may match nothing. The asymmetry is deliberate and correct: columns
are constrained to at most one parent, rows are not, since a Source 1 entity legitimately
has many matches. A symmetric Sinkhorn would encode the wrong constraint.

The ``margin`` feature this produces is the precision workhorse. Two genuinely distinct
Source 1 entities that look alike (two branches of one chain) yield a near-zero margin --
exactly the configuration that causes the catastrophic false merges F_0.5 punishes hardest.

**Corroboration (Stage T).** Source 2 and Source 3 describe the same businesses, so a
weakly-linked Source 2 record that looks identical to a strongly-linked Source 3 record is
probably a match too. Rather than an O(n^2) all-pairs similarity among an entity's
candidates, corroboration is computed by *grouping* candidates on cheap exact keys (postal
code, numeric signature, name skeleton) and asking "what is the best probability among my
group-mates from the other source?". That is O(n log n), captures the signal that matters,
and adds no measurable cost.

All of these enter as **features of a stacked stage-2 model** rather than as direct score
edits, so the model learns when they are informative instead of being forced to trust them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


GRAPH_FEATURE_NAMES: Tuple[str, ...] = (
    "p1",
    "p1_logit",
    # ---- competition over Source-1 entities for this candidate ------------------
    "compete_count",
    "compete_rank",
    "compete_is_best",
    "compete_margin",
    "compete_ratio",
    "compete_softmax",
    # ---- the entity's own candidate profile -------------------------------------
    "entity_cand_count",
    "entity_p_rank",
    "entity_p_max",
    "entity_p_sum",
    "entity_p_margin_to_max",
    "entity_p_mean",
    "entity_rank_in_source",
    "entity_source_p_max",
    # ---- corroboration from the other source ------------------------------------
    "corr_postal_count",
    "corr_postal_best_other_p",
    "corr_digit_count",
    "corr_digit_best_other_p",
    "corr_skel_count",
    "corr_skel_best_other_p",
    "other_source_best_p",
    "other_source_count",
)

N_GRAPH_FEATURES = len(GRAPH_FEATURE_NAMES)
GRAPH_INDEX: Dict[str, int] = {n: i for i, n in enumerate(GRAPH_FEATURE_NAMES)}


@dataclass
class GraphConfig:
    softmax_temperature: float = 0.6
    null_logit: float = 0.0          # dustbin score: how attractive "no parent" is
    enable_competition: bool = True
    enable_corroboration: bool = True


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _group_stats(group_key: np.ndarray, value: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-element statistics of ``value`` within groups defined by ``group_key``.

    Returns ``(count, rank, best, second_best)`` aligned with the input order.
    ``rank`` is 0 for the largest value in the group. ``second_best`` is ``-inf`` for
    singleton groups.
    """
    n = len(group_key)
    if n == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z, z
    order = np.lexsort((-value, group_key))
    gk = group_key[order]
    vv = value[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(gk[1:], gk[:-1], out=first[1:])
    gid = np.cumsum(first) - 1
    n_groups = int(gid[-1]) + 1

    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, n))
    rank_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)

    best = vv[starts]
    second = np.full(n_groups, -np.inf, dtype=np.float64)
    has_second = counts > 1
    second[has_second] = vv[starts[has_second] + 1]

    out_count = np.empty(n, dtype=np.float32)
    out_rank = np.empty(n, dtype=np.float32)
    out_best = np.empty(n, dtype=np.float32)
    out_second = np.empty(n, dtype=np.float32)

    out_count[order] = np.repeat(counts, counts).astype(np.float32)
    out_rank[order] = rank_sorted.astype(np.float32)
    out_best[order] = np.repeat(best, counts).astype(np.float32)
    out_second[order] = np.repeat(second, counts).astype(np.float32)
    return out_count, out_rank, out_best, out_second


def _group_max_excluding_self(group_key: np.ndarray, value: np.ndarray,
                              eligible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each element, the max ``value`` over *other* eligible members of its group.

    Used for corroboration: "the best probability among my group-mates **from the other
    source**". Returns ``(best_other, n_other)``.
    """
    n = len(group_key)
    best_other = np.zeros(n, dtype=np.float32)
    n_other = np.zeros(n, dtype=np.float32)
    if n == 0:
        return best_other, n_other

    order = np.argsort(group_key, kind="stable")
    gk = group_key[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(gk[1:], gk[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, n))

    vv = value[order]
    ee = eligible[order]
    ob = np.zeros(n, dtype=np.float32)
    on = np.zeros(n, dtype=np.float32)

    for s, c in zip(starts, counts):
        if c < 2:
            continue
        sl = slice(s, s + c)
        v = vv[sl]
        e = ee[sl]
        if not e.any():
            continue
        ve = np.where(e, v, -1.0)
        # top two eligible values let us exclude self in O(c)
        i1 = int(np.argmax(ve))
        m1 = ve[i1]
        ve2 = ve.copy()
        ve2[i1] = -1.0
        m2 = float(ve2.max())
        tot = float(e.sum())
        res = np.where(np.arange(c) == i1, m2, m1)
        res = np.where(res < 0.0, 0.0, res)
        ob[sl] = res
        on[sl] = np.where(e, tot - 1.0, tot)

    best_other[order] = ob
    n_other[order] = on
    return best_other, n_other


def build_graph_features(q_row: np.ndarray, c_row: np.ndarray, p1: np.ndarray,
                         cand_src: np.ndarray,
                         cand_postal_code: np.ndarray,
                         cand_digit_code: np.ndarray,
                         cand_skel_code: np.ndarray,
                         cfg: GraphConfig) -> np.ndarray:
    """Compute the ``(n_pairs, N_GRAPH_FEATURES)`` graph design matrix.

    Parameters
    ----------
    q_row, c_row
        Candidate pair indices (query row, index row).
    p1
        Stage-1 calibrated-ish probability for each pair.
    cand_src
        Source tag (2 or 3) of each candidate record.
    cand_postal_code, cand_digit_code, cand_skel_code
        Integer hashes of the candidate's postal code / numeric signature / name
        skeleton. Zero means "absent", and absent keys never form a corroboration group.
    """
    n = len(q_row)
    G = np.zeros((n, N_GRAPH_FEATURES), dtype=np.float32)
    if n == 0:
        return G
    F = GRAPH_INDEX

    p1 = p1.astype(np.float32)
    lg = _logit(p1)
    G[:, F["p1"]] = p1
    G[:, F["p1_logit"]] = lg

    # ---------------------------------------------------------------- competition ----
    if cfg.enable_competition:
        cnt, rank, best, second = _group_stats(c_row.astype(np.int64), lg.astype(np.float64))
        G[:, F["compete_count"]] = cnt
        G[:, F["compete_rank"]] = rank
        G[:, F["compete_is_best"]] = (rank == 0).astype(np.float32)
        margin = np.where(np.isfinite(second), lg - np.maximum(second, -30.0), 30.0)
        # for non-best entries the margin to the best is negative and equally informative
        margin = np.where(rank == 0, margin, lg - best)
        G[:, F["compete_margin"]] = margin.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(best != 0, lg / best, 1.0)
        G[:, F["compete_ratio"]] = np.clip(np.nan_to_num(ratio), -10.0, 10.0)

        # column softmax with a null dustbin -- the probabilistic projection of the
        # one-parent-per-candidate constraint
        tau = max(cfg.softmax_temperature, 1e-3)
        z = np.exp(np.clip((lg - best) / tau, -60.0, 0.0))          # shifted for stability
        denom = np.zeros(int(c_row.max()) + 2, dtype=np.float64)
        np.add.at(denom, c_row, z)
        null_term = np.exp(np.clip((cfg.null_logit - best) / tau, -60.0, 30.0))
        G[:, F["compete_softmax"]] = (z / (denom[c_row] + null_term)).astype(np.float32)

    # ---------------------------------------------------------- entity-side profile ----
    e_cnt, e_rank, e_best, _ = _group_stats(q_row.astype(np.int64), p1.astype(np.float64))
    G[:, F["entity_cand_count"]] = e_cnt
    G[:, F["entity_p_rank"]] = e_rank
    G[:, F["entity_p_max"]] = e_best
    G[:, F["entity_p_margin_to_max"]] = p1 - e_best
    p_sum = np.zeros(int(q_row.max()) + 2, dtype=np.float64)
    np.add.at(p_sum, q_row, p1)
    G[:, F["entity_p_sum"]] = p_sum[q_row].astype(np.float32)
    G[:, F["entity_p_mean"]] = (p_sum[q_row] / np.maximum(e_cnt, 1)).astype(np.float32)

    # rank within the entity *and* within the candidate's own source: |T| is distributed
    # across both sources (0-5 from S2, 0-6 from S3), so per-source rank is informative
    src_key = q_row.astype(np.int64) * 4 + cand_src.astype(np.int64)
    s_cnt, s_rank, s_best, _ = _group_stats(src_key, p1.astype(np.float64))
    G[:, F["entity_rank_in_source"]] = s_rank
    G[:, F["entity_source_p_max"]] = s_best

    # ---------------------------------------------------------------- corroboration ----
    if cfg.enable_corroboration:
        is_s2 = cand_src == 2
        other_is_s2 = ~is_s2
        # "best probability among this entity's candidates from the *other* source"
        other_key = q_row.astype(np.int64)
        b2, n2 = _group_max_excluding_self(other_key, p1, other_is_s2)
        b3, n3 = _group_max_excluding_self(other_key, p1, is_s2)
        G[:, F["other_source_best_p"]] = np.where(is_s2, b2, b3)
        G[:, F["other_source_count"]] = np.where(is_s2, n2, n3)

        for key_arr, cnt_name, best_name in (
                (cand_postal_code, "corr_postal_count", "corr_postal_best_other_p"),
                (cand_digit_code, "corr_digit_count", "corr_digit_best_other_p"),
                (cand_skel_code, "corr_skel_count", "corr_skel_best_other_p")):
            present = key_arr != 0
            gk = q_row.astype(np.int64) * np.int64(1 << 21) + (
                key_arr.astype(np.int64) % np.int64((1 << 21) - 1))
            # candidates without the key get a unique group so they never corroborate
            gk = np.where(present, gk, -(np.arange(n, dtype=np.int64) + 1))
            elig = present & np.ones(n, dtype=bool)
            bo, no = _group_max_excluding_self(gk, p1, elig)
            G[:, F[cnt_name]] = no
            G[:, F[best_name]] = bo

    return np.nan_to_num(G, nan=0.0, posinf=30.0, neginf=-30.0)


# --------------------------------------------------------------------------------------
# hard disjointness repair
# --------------------------------------------------------------------------------------

def repair_disjointness(q_row: np.ndarray, c_row: np.ndarray, selected: np.ndarray,
                        p: np.ndarray) -> np.ndarray:
    """Enforce "each Source 2/3 record has at most one parent" on the emitted sets.

    The ground truth satisfies this with zero exceptions, so any candidate selected by two
    Source 1 entities is necessarily at least one error. Keeping only the highest-probability
    claim cannot reduce the number of true positives and strictly reduces the number of
    predictions, so under a precision-weighted metric it is a safe, monotone improvement.
    """
    out = selected.copy()
    sel_idx = np.flatnonzero(out)
    if len(sel_idx) == 0:
        return out
    cc = c_row[sel_idx]
    pp = p[sel_idx]
    order = np.lexsort((-pp, cc))
    cc_s = cc[order]
    keep_first = np.empty(len(cc_s), dtype=bool)
    keep_first[0] = True
    np.not_equal(cc_s[1:], cc_s[:-1], out=keep_first[1:])
    drop = sel_idx[order[~keep_first]]
    out[drop] = False
    return out


__all__ = ["GraphConfig", "GRAPH_FEATURE_NAMES", "GRAPH_INDEX", "N_GRAPH_FEATURES",
           "build_graph_features", "repair_disjointness"]
