"""
Verify the chunked candidate union against a slow, obviously-correct reference.

The chunked implementation exists to bound peak memory on the largest partition; this
checks the optimisation did not change the semantics (per-pair max score, OR'd probe
masks, per-query top-N by score).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from trice.blocking import BlockingConfig, union_channels  # noqa: E402

FAIL = 0


def reference(results, n_queries, cap, channel_order):
    """Dict-based reference: unambiguous, O(n log n) in Python."""
    bit_of = {name: (1 << k) for k, name in enumerate(channel_order)}
    acc: dict[tuple[int, int], list] = {}
    for name, q, i, s in results:
        bit = bit_of[name]
        for qq, ii, ss in zip(q.tolist(), i.tolist(), s.tolist()):
            key = (qq, ii)
            if key in acc:
                acc[key][0] = max(acc[key][0], ss)
                acc[key][1] |= bit
            else:
                acc[key] = [ss, bit]
    by_q: dict[int, list] = {}
    for (qq, ii), (ss, mm) in acc.items():
        by_q.setdefault(qq, []).append((ss, ii, mm))
    out_q, out_i, out_s, out_m = [], [], [], []
    for qq in range(n_queries):
        rows = by_q.get(qq, [])
        rows.sort(key=lambda t: (-t[0], t[1]))
        for ss, ii, mm in rows[:cap]:
            out_q.append(qq)
            out_i.append(ii)
            out_s.append(ss)
            out_m.append(mm)
    return (np.array(out_q, dtype=np.int32), np.array(out_i, dtype=np.int32),
            np.array(out_s, dtype=np.float32), np.array(out_m, dtype=np.uint8))


def check(cond: bool, msg: str) -> None:
    global FAIL
    if not cond:
        FAIL += 1
        print(f"  FAIL  {msg}")


rng = np.random.default_rng(20260925)
print("comparing chunked union against the reference on random cases")

for trial in range(60):
    n_queries = int(rng.integers(5, 400))
    n_index = int(rng.integers(20, 3000))
    cap = int(rng.integers(2, 12))
    block = int(rng.integers(1, max(2, n_queries // 2 + 1)))

    results = []
    for name in ("name", "addr"):
        k = int(rng.integers(1, 9))
        rows = []
        for q in range(n_queries):
            if rng.random() < 0.12:
                continue
            kk = int(rng.integers(1, k + 1))
            idxs = rng.choice(n_index, size=kk, replace=False)
            for ii in idxs:
                rows.append((q, int(ii), float(np.round(rng.random(), 3))))
        if not rows:
            results.append((name, np.zeros(0, np.int32), np.zeros(0, np.int32),
                            np.zeros(0, np.float32)))
            continue
        rows.sort(key=lambda t: t[0])
        q = np.array([r[0] for r in rows], dtype=np.int32)
        i = np.array([r[1] for r in rows], dtype=np.int32)
        s = np.array([r[2] for r in rows], dtype=np.float32)
        results.append((name, q, i, s))

    cfg = BlockingConfig(max_candidates=cap)
    gq, gi, gs, gm = union_channels(results, n_queries, cfg, ["name", "addr"],
                                    query_block=block)
    rq, ri, rs, rm = reference(results, n_queries, cap, ["name", "addr"])

    check(len(gq) == len(rq),
          f"trial {trial}: length {len(gq)} vs reference {len(rq)}")
    if len(gq) != len(rq):
        continue
    check(np.array_equal(gq, rq), f"trial {trial}: query rows differ")
    ok = True
    for q in range(n_queries):
        a = {(int(i), round(float(s), 6), int(m))
             for i, s, m in zip(gi[gq == q], gs[gq == q], gm[gq == q])}
        b = {(int(i), round(float(s), 6), int(m))
             for i, s, m in zip(ri[rq == q], rs[rq == q], rm[rq == q])}
        if a != b:
            ok = False
            print(f"  trial {trial} query {q}: {sorted(a)} vs {sorted(b)}")
            break
    check(ok, f"trial {trial}: candidate sets differ")
    for q in np.unique(gq):
        sc = gs[gq == q]
        check(bool(np.all(np.diff(sc) <= 1e-6)),
              f"trial {trial}: scores not descending for query {q}")

print("  60 random cases compared")

print("\nchunk-size invariance")
n_queries, n_index = 300, 900
results = []
for name in ("name", "addr"):
    q = np.repeat(np.arange(n_queries, dtype=np.int32), 6)
    i = rng.choice(n_index, size=len(q)).astype(np.int32)
    s = np.round(rng.random(len(q)), 3).astype(np.float32)
    results.append((name, q, i, s))
cfg = BlockingConfig(max_candidates=7)
base = union_channels(results, n_queries, cfg, ["name", "addr"], query_block=n_queries)
for blk in (1, 7, 33, 150, 299, 1000):
    got = union_channels(results, n_queries, cfg, ["name", "addr"], query_block=blk)
    same = all(np.array_equal(a, b) for a, b in zip(base, got))
    check(same, f"chunk size {blk} changed the result")
print("  invariant for chunk sizes 1, 7, 33, 150, 299, 1000")

print()
if FAIL:
    print(f"RESULT: {FAIL} check(s) FAILED")
    sys.exit(1)
print("RESULT: all checks passed")
