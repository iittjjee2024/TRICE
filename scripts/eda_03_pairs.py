"""Pull real match groups so we can see the actual corruption process, plus machine specs."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "student_resource", "dataset", "train")

N_GROUPS = 40
SEED = 11

# ---------------- machine ----------------
import multiprocessing
try:
    import psutil
    ram = psutil.virtual_memory().total / 1e9
except Exception:
    ram = None
print(f"cpu logical cores: {multiprocessing.cpu_count()}")
if ram:
    print(f"total RAM: {ram:.1f} GB")
print(f"python: {sys.version.split()[0]}")

from unidecode import unidecode
for s in ["राम मार्केटिंग प्राइवेट लिमिटेड", "मॉडर्न फाइनेंस", "आदित्य प्रॉपर्टीज एलएलपी",
          "Société Générale Ecole", "Léarning Àmicale"]:
    print(f"  unidecode({s!r}) = {unidecode(s)!r}")

# ---------------- sample ground-truth groups ----------------
rng = random.Random(SEED)
gt_path = os.path.join(D, "train_ground_truth.tsv")

# reservoir sample of non-singleton rows
sample = []
n_seen = 0
with open(gt_path, encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.partition("\t")
        rest = rest.rstrip("\n")
        if not rest:
            continue
        n_seen += 1
        if len(sample) < N_GROUPS:
            sample.append((s1, rest.split(",")))
        else:
            j = rng.randrange(n_seen)
            if j < N_GROUPS:
                sample[j] = (s1, rest.split(","))

want_s1 = {s for s, _ in sample}
want_other = {m for _, ms in sample for m in ms}
print(f"\nsampled {len(sample)} groups, need {len(want_other)} S2/S3 records")

records = {}


def scan(path, wanted):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            eid, _, rest = line.partition("\t")
            if eid in wanted:
                parts = rest.rstrip("\n").split("\t")
                records[eid] = parts  # name, address, country
                if len(records) == len(want_s1) + len(want_other):
                    return


scan(os.path.join(D, "train_source1.tsv"), want_s1)
print("  scanned source1")
scan(os.path.join(D, "train_source2.tsv"), want_other)
print("  scanned source2")
scan(os.path.join(D, "train_source3.tsv"), want_other)
print("  scanned source3")

print("\n" + "=" * 100)
for s1, mids in sample:
    r = records.get(s1)
    if not r:
        continue
    print(f"\n### {s1}   [{r[2]}]")
    print(f"  S1  NAME: {r[0]}")
    print(f"      ADDR: {r[1]}")
    for m in mids:
        rm = records.get(m)
        if not rm:
            print(f"  {m}  <missing>")
            continue
        print(f"  {m[:2]}  NAME: {rm[0]}")
        print(f"      ADDR: {rm[1]}")
