"""
Decision-tuner endpoints.

The decision layer consumes only the stored calibrated probabilities, so it can be
re-solved without touching the model or recomputing features. That is what makes the tuner
interactive, and it is the reason the training script persists ``val_pairs.npz``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import ensure_src_on_path, settings
from ..store import load_metrics, load_val_pairs

ensure_src_on_path()
from trice.decide import DecisionConfig, decide_groups, expected_f_curve  # noqa: E402
from trice.evaluate import classify, score_flat                          # noqa: E402
from trice.graph import repair_disjointness                              # noqa: E402

router = APIRouter(prefix="/api", tags=["decision"])


class SimulateRequest(BaseModel):
    rule: str = Field("expected_f",
                      pattern="^(expected_f|global_threshold|top1|topk)$")
    beta: float = Field(0.5, gt=0.0, le=4.0)
    prune_epsilon: float = Field(0.01, ge=0.0, le=0.5)
    max_emit: int = Field(25, ge=1, le=64)
    global_threshold: float = Field(0.5, ge=0.0, le=1.0)
    fixed_k: int = Field(3, ge=1, le=32)
    probability_power: float = Field(1.0, gt=0.0, le=5.0)
    use_missing_mass: bool = True
    disjointness_repair: bool = True
    use_stage1_probabilities: bool = False
    sample_entities: Optional[int] = Field(None, ge=200, le=1_000_000)


def _solve(vp, req: SimulateRequest, sample: Optional[np.ndarray]) -> Dict[str, Any]:
    cfg = DecisionConfig(
        beta=req.beta, rule=req.rule, prune_epsilon=req.prune_epsilon,
        max_emit=req.max_emit, global_threshold=req.global_threshold,
        fixed_k=req.fixed_k, probability_power=req.probability_power)

    probs = vp.p_stage1 if req.use_stage1_probabilities else vp.p_cal
    starts, ends = vp.starts, vp.ends
    truth = vp.val_truth_sizes
    miss = vp.val_missing if req.use_missing_mass else None
    countries = vp.val_countries

    if sample is not None:
        # restrict to a contiguous gather of the sampled entities
        keep_pairs = np.zeros(len(probs), dtype=bool)
        for g in sample:
            keep_pairs[starts[g]:ends[g]] = True
        idx = np.flatnonzero(keep_pairs)
        sub_probs = probs[idx]
        sub_q = vp.q_dense[idx]
        sub_label = vp.label[idx]
        sub_c = vp.c[idx]
        # renumber the sampled entities densely
        remap = np.full(vp.n_entities, -1, dtype=np.int64)
        remap[sample] = np.arange(len(sample))
        sub_q = remap[sub_q]
        order = np.argsort(sub_q, kind="stable")
        sub_q, sub_probs = sub_q[order], sub_probs[order]
        sub_label, sub_c = sub_label[order], sub_c[order]
        n_ent = len(sample)
        a = np.arange(n_ent)
        s2 = np.searchsorted(sub_q, a, side="left").astype(np.int64)
        e2 = np.searchsorted(sub_q, a, side="right").astype(np.int64)
        truth = truth[sample]
        countries = countries[sample]
        miss = miss[sample] if miss is not None else None
        probs, starts, ends = sub_probs, s2, e2
        labels, c_row, q_row = sub_label, sub_c, sub_q
    else:
        n_ent = vp.n_entities
        labels, c_row, q_row = vp.label, vp.c, vp.q_dense

    mask, k_star, exp_f = decide_groups(probs, starts, ends, cfg, missing_mass=miss)
    if req.disjointness_repair:
        mask = repair_disjointness(q_row.astype(np.int64), c_row.astype(np.int64),
                                   mask, probs)

    per = score_flat(mask, labels, starts, ends, truth)

    sel_n = np.zeros(n_ent, dtype=np.int64)
    np.add.at(sel_n, q_row[mask], 1)
    tp_n = np.zeros(n_ent, dtype=np.int64)
    np.add.at(tp_n, q_row[mask & (labels == 1)], 1)

    prec = np.where(sel_n > 0, tp_n / np.maximum(sel_n, 1),
                    np.where(truth == 0, 1.0, 0.0))
    recl = np.where(truth > 0, tp_n / np.maximum(truth, 1),
                    np.where(sel_n == 0, 1.0, 0.0))

    # verdict buckets
    verdicts: Dict[str, int] = {}
    empty_pred = sel_n == 0
    singleton = truth == 0
    exact = (tp_n == truth) & (sel_n == truth)
    verdicts["exact"] = int((exact & ~singleton).sum())
    verdicts["singleton_ok"] = int((singleton & empty_pred).sum())
    verdicts["false_merge_on_singleton"] = int((singleton & ~empty_pred).sum())
    verdicts["missed_all"] = int((~singleton & empty_pred).sum())
    verdicts["false_merge_on_matched"] = int((~singleton & ~empty_pred &
                                              (tp_n == 0)).sum())
    verdicts["partial"] = int(n_ent - sum(verdicts.values()))

    by_country = []
    for c in sorted(set(countries.tolist())):
        m = countries == c
        by_country.append({
            "country": c, "n": int(m.sum()),
            "macro_f05": round(float(per[m].mean()), 6),
            "macro_precision": round(float(prec[m].mean()), 6),
            "macro_recall": round(float(recl[m].mean()), 6),
            "mean_k": round(float(sel_n[m].mean()), 4),
        })

    k_hist: Dict[int, int] = {}
    vals, cnts = np.unique(sel_n, return_counts=True)
    for v, n in zip(vals.tolist(), cnts.tolist()):
        k_hist[int(v)] = int(n)

    t_hist: Dict[int, int] = {}
    vals, cnts = np.unique(truth, return_counts=True)
    for v, n in zip(vals.tolist(), cnts.tolist()):
        t_hist[int(v)] = int(n)

    # is the rule's own expected value honest? bin predicted E[F] vs realised F
    ev_cal = []
    if req.rule == "expected_f":
        edges = np.linspace(0.0, 1.0, 11)
        which = np.clip(np.digitize(exp_f, edges) - 1, 0, 9)
        for b in range(10):
            m = which == b
            if not m.any():
                continue
            ev_cal.append({
                "bin": round(float(edges[b] + 0.05), 3),
                "predicted_ef": round(float(exp_f[m].mean()), 5),
                "realised_f": round(float(per[m].mean()), 5),
                "n": int(m.sum()),
            })

    return {
        "n_entities": int(n_ent),
        "sampled": sample is not None,
        "score": {
            "macro_f05": round(float(per.mean()), 6),
            "macro_precision": round(float(prec.mean()), 6),
            "macro_recall": round(float(recl.mean()), 6),
            "mean_k": round(float(sel_n.mean()), 4),
            "predicted_pairs": int(mask.sum()),
        },
        "by_country": by_country,
        "verdicts": verdicts,
        "k_histogram": k_hist,
        "truth_size_histogram": t_hist,
        "ev_calibration": ev_cal,
    }


@router.post("/runs/{run_id}/decision/simulate")
def simulate(run_id: str, req: SimulateRequest) -> Dict[str, Any]:
    vp = load_val_pairs(run_id)
    if vp is None:
        raise HTTPException(404, f"no validation arrays for run {run_id}")
    n_sample = req.sample_entities or settings().simulate_sample_entities
    sample = None
    if n_sample < vp.n_entities:
        rng = np.random.default_rng(12345)          # fixed so slider moves are comparable
        sample = np.sort(rng.choice(vp.n_entities, size=n_sample, replace=False))
    return _solve(vp, req, sample)


@router.get("/runs/{run_id}/decision")
def decision_summary(run_id: str) -> Dict[str, Any]:
    """Stored decision-stage results from training, plus the rule comparison table."""
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")
    rules = m.get("decision_rules", {})
    comparison = []
    for name, r in rules.items():
        comparison.append({
            "rule": name,
            "macro_f05": r.get("val_macro_f05"),
            "macro_precision": r.get("val_macro_precision"),
            "macro_recall": r.get("val_macro_recall"),
            "mean_k": r.get("mean_k"),
        })
    comparison.sort(key=lambda d: -(d["macro_f05"] or 0))
    return {
        "chosen": rules.get("expected_f"),
        "best_threshold_rule": m.get("best_threshold_rule"),
        "comparison": comparison,
        "truth_size_hist": m.get("truth_size_hist"),
    }


@router.get("/runs/{run_id}/decision/curve/{entity_index}")
def entity_curve(run_id: str, entity_index: int) -> Dict[str, Any]:
    """``E[F | k]`` for one validation entity -- the plot behind the decision."""
    vp = load_val_pairs(run_id)
    if vp is None:
        raise HTTPException(404, f"no validation arrays for run {run_id}")
    if not (0 <= entity_index < vp.n_entities):
        raise HTTPException(404, "entity index out of range")
    lo, hi = int(vp.starts[entity_index]), int(vp.ends[entity_index])
    p = np.sort(vp.p_cal[lo:hi])[::-1]
    cfg = DecisionConfig(prune_epsilon=0.0)
    curve, k_star = expected_f_curve(p, cfg)
    return {
        "entity_id": str(vp.val_entity_ids[entity_index]),
        "k_star": int(k_star),
        "curve": [{"k": i, "expected_f": round(float(v), 6)}
                  for i, v in enumerate(curve)],
        "probabilities": [round(float(x), 6) for x in p.tolist()],
    }
