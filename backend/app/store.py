"""
Run registry and artifact access.

Runs are plain directories under ``artifacts/runs/<run_id>`` written by
``scripts/05_train.py``; there is no database. The service discovers them from the
filesystem and caches the heavier artifacts (the validation pair arrays) in memory.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

from .config import settings


def run_dir(run_id: str) -> str:
    safe = os.path.basename(run_id)
    return os.path.join(settings().runs_dir, safe)


def list_runs() -> List[Dict[str, Any]]:
    """Every run with a readable ``metrics.json``, newest first."""
    base = settings().runs_dir
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        mpath = os.path.join(base, name, "metrics.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:
            continue
        rules = m.get("decision_rules", {})
        chosen = rules.get("expected_f", {})
        best_thr_key = m.get("best_threshold_rule")
        out.append({
            "run_id": name,
            "created": m.get("created"),
            "countries": m.get("countries", []),
            "n_entities": m.get("n_entities"),
            "n_val_entities": m.get("n_val_entities"),
            "n_pairs": m.get("n_pairs"),
            "model_kind": m.get("model_kind"),
            "use_stage2": m.get("use_stage2"),
            "headline": {
                "val_macro_f05": chosen.get("val_macro_f05"),
                "val_macro_precision": chosen.get("val_macro_precision"),
                "val_macro_recall": chosen.get("val_macro_recall"),
                "mean_k": chosen.get("mean_k"),
                "blocking_recall": m.get("blocking_recall_overall"),
                "stage1_auc": (m.get("stage1") or {}).get("auc"),
                "stage2_auc": (m.get("stage2") or {}).get("auc"),
                "best_threshold_rule": best_thr_key,
                "best_threshold_f05": (rules.get(best_thr_key) or {}).get("val_macro_f05")
                if best_thr_key else None,
            },
            "has_inference": os.path.isfile(os.path.join(base, name, "inference.json")),
            "has_val_pairs": os.path.isfile(os.path.join(base, name, "val_pairs.npz")),
        })
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return out


def load_metrics(run_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir(run_id), "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_inference(run_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir(run_id), "inference.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class ValPairs:
    """Validation candidate arrays, plus derived per-entity indexing.

    Holds only the validation slice written by the training script, which is what the
    decision tuner and entity explorer need. Nothing here requires the model.
    """

    def __init__(self, npz: Dict[str, np.ndarray]) -> None:
        self.q = npz["q"].astype(np.int64)
        self.c = npz["c"].astype(np.int64)
        self.cand_src = npz["cand_src"]
        self.cand_num = npz["cand_num"]
        self.p_cal = npz["p_cal"]
        self.p_stage1 = npz["p_stage1"]
        self.label = npz["label"]
        self.block_score = npz["block_score"]
        self.entity_ids = npz["entity_ids"]
        self.entity_country = npz["entity_country"].astype(str)
        self.truth_sizes = npz["truth_sizes"]
        self.val_entity = npz["val_entity"]
        self.missing_mass = npz["missing_mass"]

        # entities that actually appear in the validation slice, in ascending id order
        self.val_ids = np.flatnonzero(self.val_entity).astype(np.int64)
        # dense remap so group bounds are contiguous
        remap = np.full(len(self.entity_ids), -1, dtype=np.int64)
        remap[self.val_ids] = np.arange(len(self.val_ids))
        self.q_dense = remap[self.q]
        order = np.argsort(self.q_dense, kind="stable")
        for attr in ("q", "c", "cand_src", "cand_num", "p_cal", "p_stage1", "label",
                     "block_score", "q_dense"):
            setattr(self, attr, getattr(self, attr)[order])
        n = len(self.val_ids)
        idx = np.arange(n)
        self.starts = np.searchsorted(self.q_dense, idx, side="left").astype(np.int64)
        self.ends = np.searchsorted(self.q_dense, idx, side="right").astype(np.int64)
        self.val_truth_sizes = self.truth_sizes[self.val_ids]
        self.val_countries = self.entity_country[self.val_ids]
        self.val_entity_ids = self.entity_ids[self.val_ids]
        self.val_missing = self.missing_mass[self.val_ids]

    @property
    def n_entities(self) -> int:
        return len(self.val_ids)

    def candidate_ids(self, lo: int, hi: int) -> List[str]:
        return [f"S{s}-{n}" for s, n in zip(self.cand_src[lo:hi].tolist(),
                                            self.cand_num[lo:hi].tolist())]


@lru_cache(maxsize=4)
def load_val_pairs(run_id: str) -> Optional[ValPairs]:
    path = os.path.join(run_dir(run_id), "val_pairs.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    return ValPairs(data)


def blocking_reports() -> List[Dict[str, Any]]:
    """Standalone blocking sweep artifacts written by ``scripts/04_blocking_eval.py``."""
    base = settings().blocking_dir
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(base, name), encoding="utf-8") as fh:
                out.append({"file": name, "data": json.load(fh)})
        except Exception:
            continue
    return out
