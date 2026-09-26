"""
Scoring: macro-averaged F_beta exactly as the leaderboard computes it, plus the
per-entity breakdowns the error analysis needs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


def f_beta(n_pred: int, n_truth: int, tp: int, beta: float = 0.5) -> float:
    """``(1+b^2)*tp / (b^2*|T| + |S|)``; 1.0 when both sets are empty."""
    if n_pred == 0 and n_truth == 0:
        return 1.0
    b2 = beta * beta
    denom = b2 * n_truth + n_pred
    if denom == 0:
        return 0.0
    return (1.0 + b2) * tp / denom


VERDICTS = ("exact", "partial", "false_merge_on_singleton", "false_merge_on_matched",
            "missed_all", "singleton_ok")


def classify(pred: Set[str], truth: Set[str]) -> str:
    """Bucket one entity's outcome for error analysis."""
    if not truth:
        return "singleton_ok" if not pred else "false_merge_on_singleton"
    if not pred:
        return "missed_all"
    if pred == truth:
        return "exact"
    if not (pred & truth):
        return "false_merge_on_matched"
    return "partial"


@dataclass
class Score:
    """Aggregate result of scoring a prediction set against ground truth."""

    macro_f: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    n_entities: int = 0
    per_country: Dict[str, Tuple[float, int]] = field(default_factory=dict)
    verdicts: Dict[str, int] = field(default_factory=dict)
    singleton_n: int = 0
    singleton_correct: int = 0
    non_singleton_macro_f: float = 0.0
    k_star_hist: Dict[int, int] = field(default_factory=dict)
    truth_size_hist: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "macro_f05": round(self.macro_f, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "n_entities": self.n_entities,
            "per_country": {k: {"macro_f05": round(v[0], 6), "n": v[1]}
                            for k, v in sorted(self.per_country.items())},
            "verdicts": dict(sorted(self.verdicts.items())),
            "singletons": {
                "n": self.singleton_n,
                "correct": self.singleton_correct,
                "credit": round(self.singleton_correct / self.singleton_n, 6)
                if self.singleton_n else None,
            },
            "non_singleton_macro_f05": round(self.non_singleton_macro_f, 6),
            "k_star_hist": dict(sorted(self.k_star_hist.items())),
            "truth_size_hist": dict(sorted(self.truth_size_hist.items())),
        }


def score_predictions(pred: Mapping[str, Iterable[str]],
                      truth: Mapping[str, Iterable[str]],
                      countries: Mapping[str, str] | None = None,
                      beta: float = 0.5) -> Score:
    """Macro-average F_beta over **every entity present in ``truth``**.

    Entities missing from ``pred`` are scored as an empty prediction, mirroring the
    leaderboard's treatment (and the validator's requirement that every Source 1 entity
    appear exactly once).
    """
    total = 0.0
    total_p = 0.0
    total_r = 0.0
    n = 0
    verdicts: Counter[str] = Counter()
    per_country_sum: Dict[str, float] = {}
    per_country_n: Counter[str] = Counter()
    singleton_n = singleton_correct = 0
    non_single_sum = 0.0
    non_single_n = 0
    k_hist: Counter[int] = Counter()
    t_hist: Counter[int] = Counter()

    for eid, t_ids in truth.items():
        ts = set(t_ids)
        ps = set(pred.get(eid, ()))
        tp = len(ps & ts)
        f = f_beta(len(ps), len(ts), tp, beta)
        total += f
        total_p += (tp / len(ps)) if ps else (1.0 if not ts else 0.0)
        total_r += (tp / len(ts)) if ts else (1.0 if not ps else 0.0)
        n += 1
        verdicts[classify(ps, ts)] += 1
        k_hist[len(ps)] += 1
        t_hist[len(ts)] += 1
        if not ts:
            singleton_n += 1
            if not ps:
                singleton_correct += 1
        else:
            non_single_sum += f
            non_single_n += 1
        if countries is not None:
            c = countries.get(eid, "?")
            per_country_sum[c] = per_country_sum.get(c, 0.0) + f
            per_country_n[c] += 1

    if n == 0:
        return Score()

    return Score(
        macro_f=total / n,
        macro_precision=total_p / n,
        macro_recall=total_r / n,
        n_entities=n,
        per_country={c: (per_country_sum[c] / per_country_n[c], per_country_n[c])
                     for c in per_country_sum},
        verdicts=dict(verdicts),
        singleton_n=singleton_n,
        singleton_correct=singleton_correct,
        non_singleton_macro_f=(non_single_sum / non_single_n) if non_single_n else 0.0,
        k_star_hist=dict(k_hist),
        truth_size_hist=dict(t_hist),
    )


def score_flat(sel_mask: np.ndarray, labels: np.ndarray,
               group_starts: np.ndarray, group_ends: np.ndarray,
               truth_sizes: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Fast per-entity F_beta from flat candidate arrays.

    ``truth_sizes`` must be the **full** true-set size, including true matches that
    blocking never retrieved -- otherwise recall is silently inflated.
    """
    n_groups = len(group_starts)
    out = np.empty(n_groups, dtype=np.float64)
    b2 = beta * beta
    coef = 1.0 + b2
    for g in range(n_groups):
        st, en = int(group_starts[g]), int(group_ends[g])
        m = sel_mask[st:en]
        n_pred = int(m.sum())
        n_truth = int(truth_sizes[g])
        if n_pred == 0 and n_truth == 0:
            out[g] = 1.0
            continue
        denom = b2 * n_truth + n_pred
        if denom == 0:
            out[g] = 0.0
            continue
        tp = int(labels[st:en][m].sum())
        out[g] = coef * tp / denom
    return out


def blocking_recall(labels: np.ndarray, group_starts: np.ndarray,
                    group_ends: np.ndarray, truth_sizes: np.ndarray) -> dict:
    """Recall ceiling imposed by candidate generation.

    ``micro`` is the fraction of all true links retrieved; ``macro`` averages per-entity
    recall, which is the quantity that actually bounds the macro-averaged score.
    """
    n_groups = len(group_starts)
    found = np.empty(n_groups, dtype=np.int32)
    for g in range(n_groups):
        found[g] = int(labels[group_starts[g]:group_ends[g]].sum())
    total_true = int(truth_sizes.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        per_entity = np.where(truth_sizes > 0, found / np.maximum(truth_sizes, 1), 1.0)
    return {
        "micro_recall": found.sum() / total_true if total_true else 1.0,
        "macro_recall": float(per_entity.mean()),
        "entities_full_recall": int((per_entity >= 1.0).sum()),
        "entities_zero_recall": int(((truth_sizes > 0) & (found == 0)).sum()),
        "n_true_links": total_true,
        "n_found_links": int(found.sum()),
    }


def parse_ground_truth(path: str) -> Dict[str, List[str]]:
    """Read ``*_ground_truth.tsv`` into ``{s1_id: [matched ids]}``."""
    from .paths import open_text
    out: Dict[str, List[str]] = {}
    with open_text(path) as fh:
        next(fh)
        for line in fh:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            out[s1] = rest.split(",") if rest else []
    return out


__all__ = ["f_beta", "classify", "VERDICTS", "Score", "score_predictions", "score_flat",
           "blocking_recall", "parse_ground_truth"]
