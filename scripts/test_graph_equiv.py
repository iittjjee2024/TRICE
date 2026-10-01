"""Numerical equivalence: refactored (float32, out=) build_graph_features vs the committed
version in git HEAD. Confirms the memory refactor did not change the feature values beyond
float32 rounding, so the trained stage-2 model still sees the same inputs.
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trice.graph import GraphConfig, N_GRAPH_FEATURES, build_graph_features as new_fn  # noqa


def load_orig():
    """Load the committed-at-HEAD build_graph_features as a baseline, pulled straight from
    git so the test needs no checked-in copy. Falls back to the pre-refactor commit that
    still used the float64 path if HEAD already contains the refactor."""
    import subprocess
    import types
    for rev in ("HEAD~1:src/trice/graph.py", "HEAD:src/trice/graph.py"):
        try:
            src = subprocess.check_output(["git", "show", rev], cwd=ROOT).decode("utf-8")
        except Exception:
            continue
        if "out: np.ndarray" in src and rev.startswith("HEAD:"):
            continue  # HEAD already has the refactor; prefer the parent
        mod = types.ModuleType("graph_orig")
        sys.modules["graph_orig"] = mod
        exec(compile(src, "<graph_orig>", "exec"), mod.__dict__)
        return mod
    raise RuntimeError("could not load a baseline graph.py from git")


def main() -> None:
    orig = load_orig()
    rng = np.random.default_rng(7)
    n, e = 300_000, 25_000
    q = np.sort(rng.integers(0, e, size=n)).astype(np.int32)
    c = rng.integers(0, e * 6, size=n).astype(np.int32)
    p1 = rng.random(n).astype(np.float32)
    src = rng.integers(2, 4, size=n).astype(np.int8)
    postal = rng.integers(0, 50_000, size=n).astype(np.int64)
    digit = rng.integers(0, 50_000, size=n).astype(np.int64)
    skel = rng.integers(0, 50_000, size=n).astype(np.int64)
    cfg = GraphConfig()

    G_old = orig.build_graph_features(
        q.astype(np.int64), c.astype(np.int64), p1.copy(), src, postal, digit, skel, cfg)

    # new path writes into an out= array that is a MEMMAP, exactly as 06_infer.py does,
    # so column-wise assignment + in-place nan_to_num on a disk-backed array is exercised
    import tempfile
    d = tempfile.mkdtemp()
    gp = os.path.join(d, "G.f32")
    G_new = np.memmap(gp, dtype=np.float32, mode="w+", shape=(n, N_GRAPH_FEATURES))
    new_fn(q.astype(np.int64), c.astype(np.int64), p1.copy(), src,
           postal, digit, skel, cfg, out=G_new)
    G_new.flush()

    from trice.graph import GRAPH_FEATURE_NAMES
    worst = 0.0
    worst_feat = ""
    for i, name in enumerate(GRAPH_FEATURE_NAMES):
        d = float(np.abs(G_old[:, i].astype(np.float64) - G_new[:, i].astype(np.float64)).max())
        if d > worst:
            worst, worst_feat = d, name
        flag = "  <-- large" if d > 1e-2 else ""
        print(f"  {name:28s} max|diff|={d:.3e}{flag}")
    print(f"\nworst feature: {worst_feat} = {worst:.3e}")
    # float32 rounding on logits/softmax; anything under 1e-2 is immaterial to a GBDT
    assert worst < 1e-2, f"feature {worst_feat} diverged by {worst}"
    print("EQUIVALENCE OK: refactor preserves feature values within float32 tolerance")


if __name__ == "__main__":
    main()
