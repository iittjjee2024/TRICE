"""Sanity-check both GBDT backends and the LightGBM->HGB fallback."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import trice.model as M  # noqa: E402
from trice.model import GBDT, ModelConfig, rank_metrics  # noqa: E402

rng = np.random.default_rng(0)
X = rng.random((4000, 10)).astype("float32")
y = (X[:, 0] + 0.2 * rng.random(4000) > 0.6).astype("int8")
Xv = rng.random((1000, 10)).astype("float32")
yv = (Xv[:, 0] + 0.2 * rng.random(1000) > 0.6).astype("int8")
names = [f"f{i}" for i in range(10)]

g = GBDT(ModelConfig(max_iter=60)).fit(X, y, Xv, yv, names)
print("default kind:", g.kind,
      "| AUC", round(rank_metrics(g.predict(Xv), yv)["auc"], 4),
      "| params", g.n_parameters(), "| top feature", g.importances()[0])

# force the HGB path
g2 = GBDT(ModelConfig(max_iter=60, kind="hgb")).fit(X, y, Xv, yv, names)
print("hgb kind:", g2.kind,
      "| AUC", round(rank_metrics(g2.predict(Xv), yv)["auc"], 4),
      "| params", g2.n_parameters(), "| top feature", g2.importances()[0])

# simulate LightGBM being unavailable: the wrapper should silently use HGB
M._HAVE_LGB = False
g3 = GBDT(ModelConfig(max_iter=60))          # resolve() -> hgb when _HAVE_LGB False
g3 = g3.fit(X, y, Xv, yv, names)
print("no-lgbm kind:", g3.kind,
      "| AUC", round(rank_metrics(g3.predict(Xv), yv)["auc"], 4))

assert g.kind in ("lightgbm", "hgb")
assert g2.kind == "hgb"
assert g3.kind == "hgb"
print("\nALL MODEL CHECKS PASSED")
