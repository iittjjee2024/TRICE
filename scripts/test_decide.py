"""
Property tests for the decision layer.

The novel contribution rests on two mathematical claims, so they are tested rather than
trusted:

1.  ``F_0.5(S,T) = 1.25 |S n T| / (0.25|T| + |S|)`` equals the precision/recall form
    printed in the problem statement, on random sets.
2.  ``argmax_k E[F | top-k]`` equals the true ``argmax`` over all ``2^n`` subsets.

Also checks the worked example from the problem statement and the Poisson-binomial pmf.
"""
from __future__ import annotations

import itertools
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from trice.decide import (DecisionConfig, brute_force_best_subset, decide_entity,  # noqa
                          expected_f_curve, f_beta_closed, f_beta_sets,
                          poisson_binomial)

FAIL = 0


def check(cond: bool, msg: str) -> None:
    global FAIL
    if not cond:
        FAIL += 1
        print(f"  FAIL  {msg}")


# ---------------------------------------------------------------- 0. worked example ----
print("0. worked example from the problem statement")
got = f_beta_sets(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
print(f"   predicted 3, truth 2, tp 2 -> F_0.5 = {got:.6f}  (statement says 0.714)")
check(abs(got - 0.7142857142857143) < 1e-9, f"expected 0.714286, got {got}")
check(abs(f_beta_closed(3, 2, 2) - got) < 1e-12, "closed form disagrees on the example")

# ------------------------------------------------- 1. closed form == pr form ----
print("\n1. closed form vs precision/recall form, 20000 random set pairs")
rng = random.Random(7)
universe = [f"X{i}" for i in range(12)]
worst = 0.0
for _ in range(20000):
    a = rng.sample(universe, rng.randint(0, 8))
    b = rng.sample(universe, rng.randint(0, 8))
    lhs = f_beta_sets(a, b)
    rhs = f_beta_closed(len(set(a)), len(set(b)), len(set(a) & set(b)))
    worst = max(worst, abs(lhs - rhs))
print(f"   max abs difference = {worst:.3e}")
check(worst < 1e-12, f"closed form deviates by {worst}")

# ------------------------------------------------- 2. poisson binomial ----
print("\n2. Poisson-binomial pmf")
for probs in ([0.3], [0.5, 0.5], [0.1, 0.2, 0.7], [0.9, 0.8, 0.4, 0.25, 0.05]):
    pmf = poisson_binomial(probs)
    # brute force
    n = len(probs)
    ref = np.zeros(n + 1)
    for mask in range(1 << n):
        p = 1.0
        c = 0
        for i in range(n):
            if mask >> i & 1:
                p *= probs[i]
                c += 1
            else:
                p *= 1 - probs[i]
        ref[c] += p
    err = float(np.abs(pmf - ref).max())
    check(err < 1e-12, f"pmf mismatch for {probs}: {err}")
    check(abs(pmf.sum() - 1.0) < 1e-12, f"pmf does not sum to 1 for {probs}")
print("   pmf matches brute force and sums to 1 for all cases")

# ------------------------------------------------- 3. top-k optimality ----
print("\n3. top-k prefix optimality vs exhaustive 2^n search")
cfg = DecisionConfig(beta=0.5, prune_epsilon=0.0, max_emit=64)
rng = random.Random(1234)
n_cases = 0
n_ev_mismatch = 0
worst_ev_gap = 0.0
for trial in range(400):
    n = rng.randint(1, 7)
    probs = [round(rng.random(), 3) for _ in range(n)]
    miss = rng.choice([0.0, 0.0, 0.15, 0.4])

    local = DecisionConfig(beta=0.5, prune_epsilon=0.0, max_emit=64, missing_mass=miss)
    sel, ev, curve = decide_entity(np.array(probs), local)

    best_set, best_ev = brute_force_best_subset(probs, 0.5, miss)
    n_cases += 1
    gap = best_ev - ev
    worst_ev_gap = max(worst_ev_gap, gap)
    if gap > 1e-9:
        n_ev_mismatch += 1
        if n_ev_mismatch <= 3:
            print(f"   MISMATCH probs={probs} miss={miss} "
                  f"ours={ev:.6f} brute={best_ev:.6f}")
print(f"   {n_cases} random cases, expected-value gap max = {worst_ev_gap:.3e}, "
      f"mismatches = {n_ev_mismatch}")
check(n_ev_mismatch == 0, "top-k prefix is not optimal in some case")

# ------------------------------------------------- 4. monotone / sanity ----
print("\n4. behavioural sanity checks")
c = DecisionConfig(prune_epsilon=0.0)
# all-tiny probabilities -> abstain
sel, ev, _ = decide_entity(np.array([0.02, 0.01, 0.005]), c)
check(len(sel) == 0, f"expected abstention on tiny probs, emitted {len(sel)}")
print(f"   tiny probs      -> k*={len(sel)}  E[F]={ev:.4f}   (abstains)")
# one strong -> emit it
sel, ev, _ = decide_entity(np.array([0.97, 0.02]), c)
check(len(sel) == 1, f"expected 1, got {len(sel)}")
print(f"   one strong      -> k*={len(sel)}  E[F]={ev:.4f}")
# five moderate: the 0.25|T| term makes extra predictions cheaper than for a lone
# moderate candidate, so more should be emitted
sel_lone, _, _ = decide_entity(np.array([0.55]), c)
sel_many, ev_many, _ = decide_entity(np.array([0.55] * 5), c)
print(f"   one @0.55       -> k*={len(sel_lone)}")
print(f"   five @0.55      -> k*={len(sel_many)}  E[F]={ev_many:.4f}")
check(len(sel_many) >= len(sel_lone), "group discount not observed")
# strong + moderate: should usually keep only the strong one under beta=0.5
sel, ev, _ = decide_entity(np.array([0.95, 0.45]), c)
print(f"   0.95 + 0.45     -> k*={len(sel)}  E[F]={ev:.4f}")
# curve shape
ev_curve, k = expected_f_curve(np.array([0.9, 0.8, 0.3, 0.1]), c)
print("   E[F|k] for [0.9,0.8,0.3,0.1] = "
      + ", ".join(f"k={i}:{v:.4f}" for i, v in enumerate(ev_curve)) + f"  -> k*={k}")
check(k == int(np.argmax(ev_curve)), "argmax disagrees with curve")

# ------------------------------------------------- 5. singleton falls out ----
print("\n5. k=0 expected value equals P(no true match)")
for probs in ([0.1, 0.05], [0.4, 0.3, 0.2], [0.01]):
    curve, _ = expected_f_curve(np.array(sorted(probs, reverse=True)),
                                DecisionConfig(prune_epsilon=0.0))
    prod = float(np.prod([1 - p for p in probs]))
    check(abs(curve[0] - prod) < 1e-12,
          f"E[F|0]={curve[0]} != prod(1-p)={prod} for {probs}")
print("   confirmed for all cases: E[F|0] = prod(1-p_i)")

# ------------------------------------------------- 6. speed ----
print("\n6. throughput")
import time
rng = np.random.default_rng(0)
batch = [np.sort(rng.random(rng.integers(5, 40)))[::-1] * rng.uniform(0.2, 1.0)
         for _ in range(20000)]
cfgs = DecisionConfig(prune_epsilon=0.01, max_emit=25)
t0 = time.time()
for p in batch:
    decide_entity(p, cfgs)
dt = time.time() - t0
print(f"   {len(batch):,} entities in {dt:.2f}s = {len(batch)/dt:,.0f} entities/s")
print(f"   -> 1,732,544 test entities would take ~{1_732_544/(len(batch)/dt):.0f}s")

print()
if FAIL:
    print(f"RESULT: {FAIL} check(s) FAILED")
    sys.exit(1)
print("RESULT: all checks passed")
