"""Run listing, detail, blocking, features and calibration endpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..store import blocking_reports, list_runs, load_inference, load_metrics

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs")
def get_runs() -> Dict[str, Any]:
    return {"runs": list_runs()}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")
    inf = load_inference(run_id)
    return {
        "run_id": run_id,
        "metrics": m,
        "inference": inf,
        "artifacts": _artifacts(run_id),
    }


def _artifacts(run_id: str) -> List[Dict[str, Any]]:
    from ..store import run_dir
    d = run_dir(run_id)
    out = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append({"name": name, "bytes": os.path.getsize(p)})
    return out


@router.get("/runs/{run_id}/blocking")
def get_blocking(run_id: str) -> Dict[str, Any]:
    """Recall ceiling, reduction ratio and per-channel cost for each country partition."""
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")

    per_country = []
    for country, rec in (m.get("blocking") or {}).items():
        stats = (m.get("blocking_stats") or {}).get(country, {})
        channels = []
        for cname, cs in stats.items():
            channels.append({
                "channel": cname,
                "vocab": cs.get("vocab"),
                "nnz": cs.get("nnz"),
                "index_gb": cs.get("index_gb"),
                "pairs": cs.get("pairs"),
                "seconds": cs.get("seconds"),
            })
        per_country.append({
            "country": country,
            "macro_recall": rec.get("macro_recall"),
            "micro_recall": rec.get("micro_recall"),
            "n_true_links": rec.get("n_true_links"),
            "n_found_links": rec.get("n_found_links"),
            "entities_full_recall": rec.get("entities_full_recall"),
            "entities_zero_recall": rec.get("entities_zero_recall"),
            "channels": channels,
        })
    per_country.sort(key=lambda d: d["country"])

    # the ceiling a given recall imposes on macro F0.5 at the achieved precision
    rules = m.get("decision_rules", {}).get("expected_f", {})
    precision = rules.get("val_macro_precision") or 1.0
    curve = []
    for r in [i / 100 for i in range(50, 101, 2)]:
        denom = 0.25 * precision + r
        curve.append({"recall": round(r, 3),
                      "f05_ceiling": round(1.25 * precision * r / denom, 5)
                      if denom else 0.0})

    return {
        "overall_macro_recall": m.get("blocking_recall_overall"),
        "per_country": per_country,
        "achieved_precision": precision,
        "f05_ceiling_curve": curve,
        "sweeps": blocking_reports(),
    }


@router.get("/runs/{run_id}/features")
def get_features(run_id: str, limit: int = Query(40, ge=1, le=200)) -> Dict[str, Any]:
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")
    return {
        "model": {
            "kind": m.get("model_kind"),
            "stage1": m.get("stage1"),
            "stage2": m.get("stage2"),
            "use_stage2": m.get("use_stage2"),
            "stage1_parameters": m.get("stage1_parameters"),
            "stage2_parameters": m.get("stage2_parameters"),
        },
        "importances_stage1": (m.get("importances_stage1") or [])[:limit],
        "importances_stage2": (m.get("importances_stage2") or [])[:limit],
        "separation": (m.get("feature_separation") or [])[:limit],
    }


@router.get("/runs/{run_id}/calibration")
def get_calibration(run_id: str) -> Dict[str, Any]:
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")
    cal = m.get("calibration") or {}
    groups = []
    for key, v in cal.items():
        groups.append({
            "group": "overall" if key == "__overall__" else key,
            "is_overall": key == "__overall__",
            "n": v.get("n"),
            "ece": v.get("ece"),
            "brier": v.get("brier"),
            "has_own_curve": v.get("calibrated_in_group"),
            "points": v.get("points", []),
        })
    groups.sort(key=lambda g: (not g["is_overall"], g["group"]))
    return {"groups": groups}


@router.get("/runs/{run_id}/ablation")
def get_ablation(run_id: str) -> Dict[str, Any]:
    """Stage-by-stage comparison, all on the same validation entities."""
    m = load_metrics(run_id)
    if m is None:
        raise HTTPException(404, f"run {run_id} not found")
    rules: Dict[str, Any] = m.get("decision_rules", {})
    exp = (rules.get("expected_f") or {}).get("val_macro_f05")
    best_thr_key = m.get("best_threshold_rule")
    rows = []

    def add(label: str, value, note: str = "") -> None:
        rows.append({
            "config": label,
            "macro_f05": value,
            "delta_vs_expected_f": (round(value - exp, 5)
                                    if (value is not None and exp is not None) else None),
            "note": note,
        })

    add("TRICE (expected-F0.5 selection)", exp, "chosen configuration")
    if best_thr_key:
        add(f"best tuned global threshold ({best_thr_key.split('@')[-1]})",
            (rules.get(best_thr_key) or {}).get("val_macro_f05"),
            "requires a validation sweep to locate")
    add("stage-1 only (no graph features)",
        (rules.get("stage1_only_expected_f") or {}).get("val_macro_f05"),
        "isolates competition + corroboration")
    add("expected-F without missing-mass term",
        (rules.get("expected_f_no_miss") or {}).get("val_macro_f05"),
        "isolates the blocking-recall coupling")
    add("expected-F without disjointness repair",
        (rules.get("expected_f_no_repair") or {}).get("val_macro_f05"),
        "isolates the one-parent constraint")
    for k in ("top1", "top2", "top3", "top4"):
        if k in rules:
            add(f"fixed cardinality {k}", rules[k].get("val_macro_f05"),
                "no per-entity adaptation")
    return {"rows": rows, "all_rules": rules}
