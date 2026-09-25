"""
Expected-F_beta set selection: the decision layer.

Why this exists
---------------
The leaderboard metric is macro-averaged F_0.5 computed **per Source 1 entity**. The
usual "threshold the pairwise probability" rule optimises the wrong thing, because the
value of adding one more candidate depends on how many you already emitted for *that*
entity and on how large that entity's true set is likely to be. Here the optimal
decision is computed exactly instead.

The algebra that makes it possible
----------------------------------
For a predicted set ``S`` and true set ``T``, substituting ``P = |S n T| / |S|`` and
``R = |S n T| / |T|`` into ``F_beta = (1+b^2) P R / (b^2 P + R)`` with ``b = 0.5``::

    F_0.5(S, T) = 1.25 * |S n T| / (0.25 * |T| + |S|)

and ``F_0.5(empty, empty) = 1``. The numerator is *linear* in the true-positive count and
the denominator depends only on the two set **sizes** -- not on which elements were
chosen. Two consequences:

**(1) For a fixed size k, the optimal S is the top-k by probability.**
    ``|T|`` is a property of nature and does not depend on our choice, so with ``k``
    fixed the denominator's distribution is unaffected by *which* k candidates we pick.
    Since ``F(t+1, f-1, k) - F(t, f, k) = 1.25 / (0.25(t+f)+k) > 0``, swapping a selected
    candidate for an unselected one of higher probability weakly improves the objective.

**(2) Therefore only k = 0..n need be evaluated** -- ``n+1`` candidates instead of ``2^n``.

With ``Y_i ~ Bernoulli(p_i)`` independent, ``TP ~ PoissonBinomial(p_1..p_k)`` and
``FN ~ PoissonBinomial(p_{k+1}..p_n) + M``, where ``M`` models true matches that blocking
never retrieved. Both pmfs come from the standard ``O(n^2)`` Poisson-binomial recurrence,
and ``E[F|k]`` is the quadratic form ``pmf_TP^T . F_k . pmf_FN``.

The singleton decision is not special-cased: ``k = 0`` gives
``E[F|0] = P(|T| = 0) = prod(1 - p_i) * P(M = 0)``, so abstaining wins exactly when the
entity most likely has no match at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Upper bound on the candidate count carried into the DP. Candidates below
# ``prune_epsilon`` are folded into the missing-mass term instead, which keeps the
# effective n small (typically < 10) and the whole layer cheap.
_TABLE_MAX = 64

_F_TABLE: np.ndarray | None = None


def _f_table(beta: float) -> np.ndarray:
    """Precompute ``F[k, t, f] = (1+b^2) t / (b^2 (t+f) + k)``.

    Cached for the default beta; tiny (64^3 float64 ~ 2 MB).
    """
    global _F_TABLE
    if beta == 0.5 and _F_TABLE is not None:
        return _F_TABLE
    b2 = beta * beta
    coef = 1.0 + b2
    n = _TABLE_MAX + 1
    k = np.arange(n).reshape(n, 1, 1)
    t = np.arange(n).reshape(1, n, 1)
    f = np.arange(n).reshape(1, 1, n)
    denom = b2 * (t + f) + k
    with np.errstate(divide="ignore", invalid="ignore"):
        tab = coef * t / denom
    tab[~np.isfinite(tab)] = 0.0
    # k = 0, f = 0, t = 0 -> both sets empty -> perfect score by definition
    tab[0, 0, 0] = 1.0
    if beta == 0.5:
        _F_TABLE = tab
    return tab


def poisson_binomial(probs: Sequence[float]) -> np.ndarray:
    """pmf of the number of successes among independent Bernoulli trials."""
    pmf = np.zeros(len(probs) + 1, dtype=np.float64)
    pmf[0] = 1.0
    for i, p in enumerate(probs):
        # in-place backward update avoids a temporary per trial
        pmf[1:i + 2] = pmf[1:i + 2] * (1.0 - p) + pmf[0:i + 1] * p
        pmf[0] *= (1.0 - p)
    return pmf


def _add_bernoulli(pmf: np.ndarray, p: float) -> np.ndarray:
    if p <= 0.0:
        return pmf
    out = np.zeros(len(pmf) + 1, dtype=np.float64)
    out[:-1] = pmf * (1.0 - p)
    out[1:] += pmf * p
    return out


@dataclass
class DecisionConfig:
    """Parameters of the decision layer."""

    beta: float = 0.5
    rule: str = "expected_f"        # expected_f | global_threshold | top1 | topk
    prune_epsilon: float = 0.01     # probabilities below this go into the miss term
    max_emit: int = 25              # hard cap on |S|
    global_threshold: float = 0.5   # used by rule='global_threshold'
    fixed_k: int = 3                # used by rule='topk'
    probability_power: float = 1.0  # p -> p**gamma escape hatch if p is over-confident
    missing_mass: float = 0.0       # expected count of true matches blocking missed


def expected_f_curve(probs: np.ndarray, cfg: DecisionConfig
                     ) -> Tuple[np.ndarray, int]:
    """Return ``(E[F | k] for k = 0..n, argmax k)`` for one entity.

    ``probs`` must already be sorted descending.
    """
    beta = cfg.beta
    b2 = beta * beta
    tab = _f_table(beta)

    n = len(probs)
    if n == 0:
        return np.array([1.0]), 0

    # suffix Poisson-binomials: pmf of FN for every k, built from the tail inwards
    suffix: List[np.ndarray] = [None] * (n + 1)          # type: ignore[list-item]
    acc = np.array([1.0])
    suffix[n] = acc
    for i in range(n - 1, -1, -1):
        acc = _add_bernoulli(acc, float(probs[i]))
        suffix[i] = acc

    ev = np.empty(n + 1, dtype=np.float64)
    tp = np.array([1.0])
    max_k = min(n, cfg.max_emit)
    for k in range(n + 1):
        if k > 0:
            tp = _add_bernoulli(tp, float(probs[k - 1]))
        if k > max_k:
            ev[k] = -1.0
            continue
        fn = _add_bernoulli(suffix[k], cfg.missing_mass) if cfg.missing_mass else suffix[k]
        kt = min(k, _TABLE_MAX)
        t_len = min(len(tp), _TABLE_MAX + 1)
        f_len = min(len(fn), _TABLE_MAX + 1)
        block = tab[kt, :t_len, :f_len]
        ev[k] = float(tp[:t_len] @ block @ fn[:f_len])

    return ev, int(np.argmax(ev))


def decide_entity(probs: np.ndarray, cfg: DecisionConfig
                  ) -> Tuple[np.ndarray, float, np.ndarray]:
    """Choose the emitted subset for one entity.

    Parameters
    ----------
    probs
        Calibrated match probabilities for this entity's candidates, in the candidate
        array's own order.

    Returns
    -------
    (selected_positions, expected_f, ev_curve)
        ``selected_positions`` indexes into ``probs``.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 1.0, np.array([1.0])

    p = np.asarray(probs, dtype=np.float64)
    if cfg.probability_power != 1.0:
        p = np.clip(p, 0.0, 1.0) ** cfg.probability_power

    order = np.argsort(-p, kind="stable")
    p_sorted = p[order]

    if cfg.rule == "top1":
        sel = order[:1] if p_sorted[0] >= cfg.global_threshold else order[:0]
        return sel.astype(np.int32), 0.0, np.array([0.0])
    if cfg.rule == "topk":
        k = min(cfg.fixed_k, n)
        return order[:k].astype(np.int32), 0.0, np.array([0.0])
    if cfg.rule == "global_threshold":
        keep = p_sorted >= cfg.global_threshold
        sel = order[keep][:cfg.max_emit]
        return sel.astype(np.int32), 0.0, np.array([0.0])

    # expected_f: fold negligible probabilities into the missing mass
    eps = cfg.prune_epsilon
    n_keep = int(np.searchsorted(-p_sorted, -eps, side="right")) if eps > 0 else n
    n_keep = min(max(n_keep, 0), _TABLE_MAX)
    dropped = p_sorted[n_keep:]
    local = DecisionConfig(**{**cfg.__dict__,
                             "missing_mass": cfg.missing_mass + float(dropped.sum())})
    ev, k_star = expected_f_curve(p_sorted[:n_keep], local)
    return order[:k_star].astype(np.int32), float(ev[k_star]), ev


def decide_groups(probs: np.ndarray, group_starts: np.ndarray,
                  group_ends: np.ndarray, cfg: DecisionConfig,
                  missing_mass: np.ndarray | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised driver over many entities.

    ``probs`` is the flat candidate probability array; ``group_starts``/``group_ends``
    delimit each entity's slice.

    Returns ``(selected_mask, k_star, expected_f)`` where ``selected_mask`` aligns with
    ``probs``.
    """
    n_groups = len(group_starts)
    mask = np.zeros(len(probs), dtype=bool)
    k_star = np.zeros(n_groups, dtype=np.int16)
    exp_f = np.zeros(n_groups, dtype=np.float32)

    base_missing = cfg.missing_mass
    for g in range(n_groups):
        st, en = int(group_starts[g]), int(group_ends[g])
        if en <= st:
            exp_f[g] = 1.0
            continue
        if missing_mass is not None:
            cfg_g = DecisionConfig(**{**cfg.__dict__,
                                      "missing_mass": float(missing_mass[g])})
        else:
            cfg_g = cfg
        sel, ef, _ = decide_entity(probs[st:en], cfg_g)
        if len(sel):
            mask[st + sel] = True
        k_star[g] = len(sel)
        exp_f[g] = ef
    cfg.missing_mass = base_missing
    return mask, k_star, exp_f


# --------------------------------------------------------------------------------------
# reference implementations used by the property tests
# --------------------------------------------------------------------------------------

def f_beta_sets(pred: Sequence[str], truth: Sequence[str], beta: float = 0.5) -> float:
    """Metric exactly as written in the problem statement (precision/recall form)."""
    ps, ts = set(pred), set(truth)
    if not ps and not ts:
        return 1.0
    if not ps or not ts:
        return 0.0
    tp = len(ps & ts)
    if tp == 0:
        return 0.0
    precision = tp / len(ps)
    recall = tp / len(ts)
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def f_beta_closed(n_pred: int, n_truth: int, tp: int, beta: float = 0.5) -> float:
    """The closed form used throughout: ``(1+b^2) tp / (b^2 |T| + |S|)``."""
    if n_pred == 0 and n_truth == 0:
        return 1.0
    b2 = beta * beta
    denom = b2 * n_truth + n_pred
    return 0.0 if denom == 0 else (1.0 + b2) * tp / denom


def brute_force_best_subset(probs: Sequence[float], beta: float = 0.5,
                            missing_mass: float = 0.0) -> Tuple[frozenset, float]:
    """Exhaustive ``O(2^n * 2^n)`` reference: the true argmax over all subsets.

    Used only in tests, for small ``n``.
    """
    import itertools

    n = len(probs)
    best_set, best_ev = frozenset(), -1.0
    for mask in range(1 << n):
        sel = frozenset(i for i in range(n) if mask >> i & 1)
        ev = 0.0
        # enumerate every realisation of the latent labels
        for truth_mask in range(1 << n):
            prob = 1.0
            truth = set()
            for i in range(n):
                if truth_mask >> i & 1:
                    prob *= probs[i]
                    truth.add(i)
                else:
                    prob *= (1.0 - probs[i])
            if prob == 0.0:
                continue
            # missing_mass as a single extra Bernoulli that is always a false negative
            for extra, w in ((0, 1.0 - missing_mass), (1, missing_mass)):
                if w <= 0.0:
                    continue
                tp = len(sel & truth)
                ev += prob * w * f_beta_closed(len(sel), len(truth) + extra, tp, beta)
        if ev > best_ev:
            best_ev, best_set = ev, sel
    return best_set, best_ev


__all__ = [
    "DecisionConfig", "decide_entity", "decide_groups", "expected_f_curve",
    "poisson_binomial", "f_beta_sets", "f_beta_closed", "brute_force_best_subset",
]
