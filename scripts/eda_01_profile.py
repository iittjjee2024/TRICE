"""Fast first-pass profile of the challenge dataset: row counts, columns, country mix."""
import os
import sys
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "student_resource", "dataset")

FILES = [
    ("train", "train_source1.tsv"),
    ("train", "train_source2.tsv"),
    ("train", "train_source3.tsv"),
    ("train", "train_ground_truth.tsv"),
    ("test", "test_source1.tsv"),
    ("test", "test_source2.tsv"),
    ("test", "test_source3.tsv"),
]


def profile(path):
    n = 0
    countries = Counter()
    header = None
    ncol_bad = 0
    empty_name = 0
    empty_addr = 0
    samples = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        ncols = len(header)
        ci = header.index("country") if "country" in header else None
        ni = header.index("business_name") if "business_name" in header else None
        ai = header.index("business_address") if "business_address" in header else None
        for line in f:
            n += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != ncols:
                ncol_bad += 1
                continue
            if ci is not None:
                countries[parts[ci]] += 1
            if ni is not None and not parts[ni].strip():
                empty_name += 1
            if ai is not None and not parts[ai].strip():
                empty_addr += 1
            if len(samples) < 4:
                samples.append(parts)
    return dict(n=n, header=header, countries=countries, ncol_bad=ncol_bad,
                empty_name=empty_name, empty_addr=empty_addr, samples=samples)


def main():
    for split, name in FILES:
        path = os.path.join(DATA, split, name)
        r = profile(path)
        print("=" * 90)
        print(f"{split}/{name}")
        print(f"  rows            : {r['n']:,}")
        print(f"  header          : {r['header']}")
        print(f"  malformed rows  : {r['ncol_bad']:,}")
        if r["countries"]:
            tot = sum(r["countries"].values())
            mix = ", ".join(f"{k}={v:,} ({v / tot:.1%})"
                            for k, v in r["countries"].most_common())
            print(f"  countries       : {mix}")
        print(f"  empty name/addr : {r['empty_name']:,} / {r['empty_addr']:,}")
        for s in r["samples"][:3]:
            print("   |", " || ".join(x[:60] for x in s))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
