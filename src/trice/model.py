"""
Matching model: gradient-boosted trees plus probability calibration.

Two stages, both GBDT:

* **Stage 1** scores the raw pairwise features from :mod:`trice.features`.
* **Stage 2** re-scores using the stage-1 output *plus* the graph features from
  :mod:`trice.graph` (competition rank/margin, Source 2 <-> Source 3 corroboration).
  Stacking rather than hand-blending lets the model learn *when* corroboration and
  competition are informative instead of assuming a fixed weight.

Calibration matters more than usual here. The decision layer in :mod:`trice.decide` is
only optimal if the probabilities it consumes are honest, and the test set contains a
country (France) with no labelled examples, where a tree ensemble's scores drift. So
calibration is fitted per country group on held-out folds, with a pooled fallback for
groups that are small or unseen.

Licensing: ``HistGradientBoostingClassifier`` (scikit-learn, BSD-3) is the default and has
no native dependency; LightGBM (MIT) is used automatically when importable. Both are
comfortably inside the challenge's "MIT/Apache-2.0, <= 8 B parameters" rule -- a GBDT has
on the order of 10^5 parameters.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import lightgbm as lgb
    _HAVE_LGB = True
except Exception:                                            # pragma: no cover
    _HAVE_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ModelConfig:
    kind: str = "auto"              # auto | lightgbm | hgb
    max_iter: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    max_bins: int = 255
    early_stopping_rounds: int = 30
    random_state: int = 20260925
    n_jobs: int = -1

    def resolve(self) -> str:
        if self.kind != "auto":
            return self.kind
        return "lightgbm" if _HAVE_LGB else "hgb"


class GBDT:
    """Thin wrapper so LightGBM and scikit-learn share one interface."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.kind = cfg.resolve()
        self.model = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            feature_names: Sequence[str] | None = None) -> "GBDT":
        self.feature_names = list(feature_names or [])
        if self.kind == "lightgbm":
            params = dict(
                objective="binary", metric=["auc", "average_precision"],
                learning_rate=self.cfg.learning_rate,
                num_leaves=self.cfg.max_leaf_nodes,
                min_data_in_leaf=self.cfg.min_samples_leaf,
                lambda_l2=self.cfg.l2_regularization,
                max_bin=self.cfg.max_bins,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                num_threads=0 if self.cfg.n_jobs == -1 else self.cfg.n_jobs,
                seed=self.cfg.random_state, verbosity=-1,
            )
            dtrain = lgb.Dataset(X, label=y, feature_name=self.feature_names or "auto")
            valid = []
            if X_val is not None and y_val is not None:
                valid = [lgb.Dataset(X_val, label=y_val, reference=dtrain)]
            callbacks = [lgb.log_evaluation(period=0)]
            if valid:
                callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds,
                                                   verbose=False))
            self.model = lgb.train(params, dtrain, num_boost_round=self.cfg.max_iter,
                                   valid_sets=valid, callbacks=callbacks)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.cfg.max_iter, learning_rate=self.cfg.learning_rate,
                max_leaf_nodes=self.cfg.max_leaf_nodes,
                min_samples_leaf=self.cfg.min_samples_leaf,
                l2_regularization=self.cfg.l2_regularization,
                max_bins=self.cfg.max_bins,
                early_stopping=X_val is not None,
                n_iter_no_change=self.cfg.early_stopping_rounds,
                validation_fraction=0.1 if X_val is None else None,
                random_state=self.cfg.random_state,
            )
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "lightgbm":
            return self.model.predict(X, num_iteration=getattr(
                self.model, "best_iteration", None)).astype(np.float32)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    # -------------------------------------------------------------- introspection ----
    def importances(self) -> List[Tuple[str, float]]:
        if self.kind == "lightgbm":
            gains = self.model.feature_importance(importance_type="gain")
            names = self.model.feature_name()
        else:
            # permutation-free fallback: HGB exposes no native gains, so use the
            # per-feature bin split counts collected during fitting
            names = self.feature_names or [f"f{i}" for i in range(
                self.model.n_features_in_)]
            gains = np.zeros(len(names), dtype=float)
            for stage in self.model._predictors:
                for pred in stage:
                    nodes = pred.nodes
                    for node in nodes:
                        if not node["is_leaf"]:
                            gains[node["feature_idx"]] += node["gain"]
        total = float(gains.sum()) or 1.0
        out = [(n, float(g) / total) for n, g in zip(names, gains)]
        out.sort(key=lambda t: -t[1])
        return out

    def n_parameters(self) -> int:
        """Rough parameter count, for the model-size rule in the challenge constraints."""
        if self.kind == "lightgbm":
            return int(self.model.num_trees() * self.cfg.max_leaf_nodes * 2)
        total = 0
        for stage in self.model._predictors:
            for pred in stage:
                total += len(pred.nodes) * 2
        return int(total)


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------

class GroupCalibrator:
    """Isotonic calibration per group (country), with a pooled fallback.

    Unseen groups -- the France case -- fall back to the pooled curve, and small groups
    are shrunk toward it. Isotonic is preferred over Platt scaling because the raw GBDT
    score distribution here is strongly bimodal, which a single sigmoid fits poorly.
    """

    def __init__(self, min_group: int = 5_000) -> None:
        self.min_group = min_group
        self.pooled: IsotonicRegression | None = None
        self.per_group: Dict[str, IsotonicRegression] = {}
        self.group_n: Dict[str, int] = {}

    def fit(self, scores: np.ndarray, labels: np.ndarray,
            groups: np.ndarray) -> "GroupCalibrator":
        self.pooled = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.pooled.fit(scores, labels)
        for g in np.unique(groups):
            m = groups == g
            n = int(m.sum())
            self.group_n[str(g)] = n
            if n < self.min_group or labels[m].min() == labels[m].max():
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(scores[m], labels[m])
            self.per_group[str(g)] = iso
        return self

    def transform(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        out = self.pooled.predict(scores).astype(np.float32)
        for g, iso in self.per_group.items():
            m = groups == g
            if m.any():
                out[m] = iso.predict(scores[m]).astype(np.float32)
        return np.clip(out, 1e-6, 1.0 - 1e-6)

    def reliability(self, scores: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray, n_bins: int = 15) -> Dict[str, dict]:
        """Reliability curves + expected calibration error, overall and per group."""
        out: Dict[str, dict] = {}
        cal = self.transform(scores, groups)
        for key, mask in [("__overall__", np.ones(len(scores), dtype=bool))] + \
                         [(str(g), groups == g) for g in np.unique(groups)]:
            if not mask.any():
                continue
            p = cal[mask]
            y = labels[mask]
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            which = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
            pts = []
            ece = 0.0
            for b in range(n_bins):
                bm = which == b
                n = int(bm.sum())
                if n == 0:
                    continue
                pp = float(p[bm].mean())
                pt = float(y[bm].mean())
                pts.append({"p_pred": round(pp, 5), "p_true": round(pt, 5), "n": n})
                ece += (n / len(p)) * abs(pp - pt)
            out[key] = {
                "n": int(mask.sum()), "ece": round(ece, 5),
                "brier": round(float(((p - y) ** 2).mean()), 5),
                "calibrated_in_group": key in self.per_group,
                "points": pts,
            }
        return out


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def rank_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if labels.min() == labels.max():
        return {"auc": float("nan"), "average_precision": float("nan")}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def feature_separation(X: np.ndarray, y: np.ndarray,
                       names: Sequence[str]) -> List[dict]:
    """Per-feature positive/negative means and single-feature AUC."""
    pos = y == 1
    out = []
    for j, n in enumerate(names):
        col = X[:, j]
        try:
            auc = float(roc_auc_score(y, col)) if y.min() != y.max() else float("nan")
        except Exception:
            auc = float("nan")
        out.append({
            "feature": n,
            "pos_mean": float(col[pos].mean()) if pos.any() else 0.0,
            "neg_mean": float(col[~pos].mean()) if (~pos).any() else 0.0,
            "auc": round(auc, 5) if auc == auc else None,
        })
    out.sort(key=lambda d: -(d["auc"] or 0.0))
    return out


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------

def save_bundle(path: str, **objects) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(objects, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = ["ModelConfig", "GBDT", "GroupCalibrator", "rank_metrics",
           "feature_separation", "save_bundle", "load_bundle", "_HAVE_LGB"]
