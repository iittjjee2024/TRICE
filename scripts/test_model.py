"""Sanity-check both GBDT backends and the LightGBM->HGB fallback."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import trice.model as M  # noqa: E402
from trice.model import (GBDT, EnsembleMatcher, ModelConfig,  # noqa: E402
                         make_matcher, rank_metrics)

rng = np.random.default_rng(0)
X = rng.random((4000, 10)).astype("float32")
y = (X[:, 0] + 0.2 * rng.random(4000) > 0.6).astype("int8")
Xv = rng.random((1000, 10)).astype("float32")
yv = (Xv[:, 0] + 0.2 * rng.random(1000) > 0.6).astype("int8")
names = [f"f{i}" for i in range(10)]


def auc(m):
    return round(rank_metrics(m.predict(Xv), yv)["auc"], 4)


# each single backend
single = {}
for kind in ("lightgbm", "xgboost", "catboost", "hgb"):
    g = GBDT(ModelConfig(max_iter=60, kind=kind)).fit(X, y, Xv, yv, names)
    single[kind] = auc(g)
    print(f"{kind:10s} kind={g.kind:9s} AUC={single[kind]} params={g.n_parameters()}")

# the ensemble (all available backends, rank-averaged)
ens = make_matcher(ModelConfig(max_iter=60, kind="ensemble"))
ens.fit(X, y, Xv, yv, names)
ens_auc = auc(ens)
print(f"ensemble   backends={ens.backends} AUC={ens_auc} "
      f"params={ens.n_parameters()} top={ens.importances()[0]}")
assert isinstance(ens, EnsembleMatcher)
assert len(ens.members) >= 1

# fallback: no boosting libs installed -> ensemble degrades to a single hgb model
M._HAVE_LGB = M._HAVE_XGB = M._HAVE_CAT = False
ens2 = make_matcher(ModelConfig(max_iter=60, kind="ensemble")).fit(X, y, Xv, yv, names)
print("no-libs ensemble backends:", ens2.backends, "AUC", auc(ens2))
assert ens2.backends == ["hgb"]

print("\nALL MODEL CHECKS PASSED")
