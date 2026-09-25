"""Ground-truth structure: match-set sizes, singleton rate, disjointness, S2/S3 mix."""
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "student_resource", "dataset", "train", "train_ground_truth.tsv")

size_hist = Counter()
per_source_hist = Counter()
combo_hist = Counter()
n_rows = 0
n_matched_ids = 0

# compact storage of every matched id: source tag (2/3) * 1e10 + numeric part
matched_codes = []

with open(GT, encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.partition("\t")
        n_rows += 1
        rest = rest.rstrip("\n")
        if not rest:
            size_hist[0] += 1
            combo_hist[(0, 0)] += 1
            continue
        ids = rest.split(",")
        n2 = n3 = 0
        codes = np.empty(len(ids), dtype=np.int64)
        for i, mid in enumerate(ids):
            tag = mid[1]           # '2' or '3'
            num = int(mid[3:])
            codes[i] = (2 if tag == "2" else 3) * 10_000_000_000 + num
            if tag == "2":
                n2 += 1
            else:
                n3 += 1
        matched_codes.append(codes)
        size_hist[len(ids)] += 1
        per_source_hist[("S2", n2)] += 1
        per_source_hist[("S3", n3)] += 1
        combo_hist[(n2, n3)] += 1
        n_matched_ids += len(ids)

all_codes = np.concatenate(matched_codes)
del matched_codes

print(f"ground-truth rows (= train S1 entities): {n_rows:,}")
print(f"total matched ids                      : {n_matched_ids:,}")
print(f"mean matches per S1 entity             : {n_matched_ids / n_rows:.3f}")

singletons = size_hist[0]
print(f"\nSINGLETONS: {singletons:,} ({singletons / n_rows:.2%}) "
      f"-> max achievable macro-F0.5 from singletons alone = {singletons / n_rows:.4f}")

print("\nmatch-set size distribution:")
cum = 0
for k in sorted(size_hist):
    cum += size_hist[k]
    print(f"  |T|={k:>3}: {size_hist[k]:>10,}  ({size_hist[k] / n_rows:6.2%})  cum {cum / n_rows:7.2%}")
    if k > 14:
        break

print("\nper-source match counts (how many S2 / how many S3 per entity):")
for src in ("S2", "S3"):
    tot = sum(v for (s, _), v in per_source_hist.items() if s == src)
    line = []
    for k in range(0, 8):
        v = per_source_hist.get((src, k), 0)
        line.append(f"{k}:{v / tot:.1%}")
    print(f"  {src}  " + "  ".join(line))

print("\nmost common (n_S2, n_S3) combinations:")
for combo, v in combo_hist.most_common(12):
    print(f"  {combo}: {v:,} ({v / n_rows:.2%})")

# ---- disjointness: does any S2/S3 record belong to more than one S1 entity? ----
all_codes.sort()
uniq, counts = np.unique(all_codes, return_counts=True)
n_multi = int((counts > 1).sum())
print(f"\nDISJOINTNESS CHECK")
print(f"  distinct matched ids       : {len(uniq):,}")
print(f"  ids used by >1 S1 entity   : {n_multi:,} ({n_multi / len(uniq):.4%})")
print(f"  max reuse of a single id   : {int(counts.max())}")
print("  -> true match sets are "
      + ("PAIRWISE DISJOINT (assumption holds)" if n_multi == 0
         else "NOT strictly disjoint"))

n_s2 = int((uniq // 10_000_000_000 == 2).sum())
n_s3 = len(uniq) - n_s2
print(f"\n  distinct S2 ids matched: {n_s2:,}   distinct S3 ids matched: {n_s3:,}")
print(f"  (train S2 has 5,034,616 rows -> {n_s2 / 5_034_616:.1%} of S2 participates in a match)")
print(f"  (train S3 has 5,285,603 rows -> {n_s3 / 5_285_603:.1%} of S3 participates in a match)")
