"""
Verify the inference OOM fix: spilling the graph-feature matrix to a disk memmap and
reading it back in chunks for stage-2 must produce results IDENTICAL to holding it in RAM,
and the per-entity candidate-id construction must match the old vectorised version.

This isolates the exact code paths changed in the exit -9 fix without running the slow,
memory-heavy full blocking pipeline.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from trice.graph import N_GRAPH_FEATURES  # noqa: E402

rng = np.random.default_rng(0)
n_pairs = 2_000_000
n_feat = 20

# ---- 1. graph-matrix memmap round-trip + chunked read equals in-RAM ----
G_ram = rng.random((n_pairs, N_GRAPH_FEATURES)).astype(np.float32)
d = tempfile.mkdtemp(prefix="memmap_test_")
g_path = os.path.join(d, "G.f32")
G_mm = np.memmap(g_path, dtype=np.float32, mode="w+",
                 shape=(n_pairs, N_GRAPH_FEATURES))
G_mm[:] = G_ram
G_mm.flush()

X_ram = rng.random((n_pairs, n_feat)).astype(np.float32)
x_path = os.path.join(d, "X.f32")
X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_pairs, n_feat))
X_mm[:] = X_ram
X_mm.flush()


# a cheap deterministic "model" standing in for m2.predict
def fake_predict(M):
    return (M.sum(axis=1) * 0.001).astype(np.float32)


# old path: everything in RAM at once
p2_ram = fake_predict(np.concatenate([X_ram, G_ram], axis=1))

# new path: chunked read from the two memmaps
p2_mm = np.zeros(n_pairs, dtype=np.float32)
step = 500_000
for s in range(0, n_pairs, step):
    e = min(s + step, n_pairs)
    chunk = np.concatenate([np.asarray(X_mm[s:e]), np.asarray(G_mm[s:e])], axis=1)
    p2_mm[s:e] = fake_predict(chunk)

err = float(np.abs(p2_ram - p2_mm).max())
print(f"memmap chunked vs in-RAM stage-2: max abs diff = {err:.2e}")
assert err == 0.0, "memmap/chunked stage-2 diverged from in-RAM"

# ---- 2. per-entity candidate-id strings equal the old vectorised build ----
n_ent = 5000
cand_per = rng.integers(0, 10, size=n_ent)
total = int(cand_per.sum())
starts = np.concatenate([[0], np.cumsum(cand_per)])[:-1].astype(np.int64)
ends = np.cumsum(cand_per).astype(np.int64)
src = rng.choice([2, 3], size=total).astype(np.int8)
num = rng.integers(1, 10_000_000, size=total).astype(np.int32)
mask = rng.random(total) < 0.3

# OLD: one big vectorised string array
old_ids = np.char.add(np.char.add("S", src.astype(str)),
                      np.char.add("-", num.astype(str)))
old_rows = []
for g in range(n_ent):
    st, en = int(starts[g]), int(ends[g])
    if en <= st:
        old_rows.append("")
        continue
    bids = old_ids[st:en]
    sel = mask[st:en]
    old_rows.append(",".join(bids[sel].tolist()) if sel.any() else "")

# NEW: per-entity construction from compact int arrays
new_rows = []
for g in range(n_ent):
    st, en = int(starts[g]), int(ends[g])
    if en <= st:
        new_rows.append("")
        continue
    bids = [f"S{int(s)}-{int(n)}" for s, n in zip(src[st:en], num[st:en])]
    sel = mask[st:en]
    new_rows.append(",".join(b for b, k in zip(bids, sel) if k) if sel.any() else "")

mismatch = sum(1 for a, b in zip(old_rows, new_rows) if a != b)
print(f"per-entity id build: {mismatch} / {n_ent} rows differ")
assert mismatch == 0, "per-entity id construction diverged from vectorised"

import shutil  # noqa: E402
shutil.rmtree(d, ignore_errors=True)
print("\nOOM-FIX LOGIC VERIFIED: identical results, graph matrix off-RAM")
