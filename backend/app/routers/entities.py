"""
Per-entity drill-down.

Macro-averaged F_0.5 concentrates its penalties in a small number of individual entities,
so being able to find and read those entities is the most useful debugging affordance in
the whole tool. Every row here is filterable by error type.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..config import ensure_src_on_path
from ..store import load_val_pairs

ensure_src_on_path()
from trice.decide import DecisionConfig, decide_groups, expected_f_curve  # noqa: E402
from trice.evaluate import f_beta                                        # noqa: E402
from trice.graph import repair_disjointness                              # noqa: E402

router = APIRouter(prefix="/api", tags=["entities"])

FILTERS = ("all", "exact", "partial", "false_merge_on_singleton",
           "false_merge_on_matched", "missed_all", "singleton_ok")


def _decide_all(vp, max_emit: int = 25, prune: float = 0.01,
                repair: bool = True):
    cfg = DecisionConfig(rule="expected_f", prune_epsilon=prune, max_emit=max_emit)
    mask, k_star, exp_f = decide_groups(vp.p_cal, vp.starts, vp.ends, cfg,
                                        missing_mass=vp.val_missing)
    if repair:
        mask = repair_disjointness(vp.q_dense.astype(np.int64),
                                   vp.c.astype(np.int64), mask, vp.p_cal)
    return mask, k_star, exp_f


_CACHE: Dict[str, Any] = {}


def _cached_decision(run_id: str):
    if run_id not in _CACHE:
        vp = load_val_pairs(run_id)
        if vp is None:
            raise HTTPException(404, f"no validation arrays for run {run_id}")
        mask, k_star, exp_f = _decide_all(vp)
        n = vp.n_entities
        sel_n = np.zeros(n, dtype=np.int64)
        np.add.at(sel_n, vp.q_dense[mask], 1)
        tp_n = np.zeros(n, dtype=np.int64)
        np.add.at(tp_n, vp.q_dense[mask & (vp.label == 1)], 1)
        truth = vp.val_truth_sizes
        f05 = np.array([f_beta(int(s), int(t), int(tp))
                        for s, t, tp in zip(sel_n, truth, tp_n)])
        verdict = np.empty(n, dtype=object)
        for i in range(n):
            s, t, tp = int(sel_n[i]), int(truth[i]), int(tp_n[i])
            if t == 0:
                verdict[i] = "singleton_ok" if s == 0 else "false_merge_on_singleton"
            elif s == 0:
                verdict[i] = "missed_all"
            elif tp == 0:
                verdict[i] = "false_merge_on_matched"
            elif tp == t and s == t:
                verdict[i] = "exact"
            else:
                verdict[i] = "partial"
        _CACHE[run_id] = {"vp": vp, "mask": mask, "k_star": k_star, "exp_f": exp_f,
                          "sel_n": sel_n, "tp_n": tp_n, "f05": f05,
                          "verdict": verdict}
    return _CACHE[run_id]


@router.get("/runs/{run_id}/entities")
def list_entities(run_id: str,
                  page: int = Query(1, ge=1),
                  page_size: int = Query(50, ge=1, le=200),
                  filter: str = Query("all"),
                  country: Optional[str] = None,
                  q: Optional[str] = None,
                  sort: str = Query("f05", pattern="^(f05|n_cand|k_star|entity)$"),
                  ascending: bool = True) -> Dict[str, Any]:
    if filter not in FILTERS:
        raise HTTPException(422, f"filter must be one of {FILTERS}")
    st = _cached_decision(run_id)
    vp = st["vp"]
    n = vp.n_entities

    keep = np.ones(n, dtype=bool)
    if filter != "all":
        keep &= st["verdict"] == filter
    if country:
        keep &= vp.val_countries == country
    if q:
        needle = q.lower()
        keep &= np.array([needle in str(e).lower() for e in vp.val_entity_ids])

    idx = np.flatnonzero(keep)
    counts = (vp.ends - vp.starts)
    if sort == "f05":
        key = st["f05"][idx]
    elif sort == "n_cand":
        key = counts[idx]
    elif sort == "k_star":
        key = st["sel_n"][idx]
    else:
        key = idx.astype(float)
    order = np.argsort(key, kind="stable")
    if not ascending:
        order = order[::-1]
    idx = idx[order]

    total = len(idx)
    lo = (page - 1) * page_size
    page_idx = idx[lo:lo + page_size]

    rows = []
    for i in page_idx:
        i = int(i)
        a, b = int(vp.starts[i]), int(vp.ends[i])
        sel = st["mask"][a:b]
        ids = vp.candidate_ids(a, b)
        rows.append({
            "index": i,
            "entity_id": str(vp.val_entity_ids[i]),
            "country": str(vp.val_countries[i]),
            "n_candidates": int(b - a),
            "k_star": int(st["sel_n"][i]),
            "truth_size": int(vp.val_truth_sizes[i]),
            "tp": int(st["tp_n"][i]),
            "expected_f": round(float(st["exp_f"][i]), 5),
            "realised_f": round(float(st["f05"][i]), 5),
            "verdict": str(st["verdict"][i]),
            "predicted": [ids[j] for j in np.flatnonzero(sel).tolist()],
        })

    return {"total": total, "page": page, "page_size": page_size, "rows": rows,
            "filters": list(FILTERS),
            "countries": sorted(set(vp.val_countries.tolist()))}


@router.get("/runs/{run_id}/entities/{entity_index}")
def entity_detail(run_id: str, entity_index: int) -> Dict[str, Any]:
    st = _cached_decision(run_id)
    vp = st["vp"]
    if not (0 <= entity_index < vp.n_entities):
        raise HTTPException(404, "entity index out of range")
    a, b = int(vp.starts[entity_index]), int(vp.ends[entity_index])
    ids = vp.candidate_ids(a, b)
    sel = st["mask"][a:b]
    p = vp.p_cal[a:b]
    order = np.argsort(-p)

    cands = []
    for j in order.tolist():
        cands.append({
            "entity_id": ids[j],
            "source": int(vp.cand_src[a + j]),
            "p_calibrated": round(float(vp.p_cal[a + j]), 6),
            "p_stage1": round(float(vp.p_stage1[a + j]), 6),
            "block_score": round(float(vp.block_score[a + j]), 6),
            "selected": bool(sel[j]),
            "is_true": bool(vp.label[a + j] == 1),
        })

    curve, k_star = expected_f_curve(np.sort(p)[::-1], DecisionConfig(prune_epsilon=0.0))
    return {
        "index": entity_index,
        "entity_id": str(vp.val_entity_ids[entity_index]),
        "country": str(vp.val_countries[entity_index]),
        "truth_size": int(vp.val_truth_sizes[entity_index]),
        "n_candidates": int(b - a),
        "k_star": int(st["sel_n"][entity_index]),
        "realised_f": round(float(st["f05"][entity_index]), 5),
        "expected_f": round(float(st["exp_f"][entity_index]), 5),
        "verdict": str(st["verdict"][entity_index]),
        "missing_mass": round(float(vp.val_missing[entity_index]), 5),
        "candidates": cands,
        "ev_curve": [{"k": i, "expected_f": round(float(v), 6)}
                     for i, v in enumerate(curve)],
        "ev_curve_k_star": int(k_star),
        "note": ("Retrieved candidates only. truth_size is the full true-set size, so "
                 "truth_size greater than the number of true candidates here means "
                 "blocking missed some links."),
    }
