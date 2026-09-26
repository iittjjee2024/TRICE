"""
Mine token aliases (state/city abbreviations, transliterations, name variants)
from the training ground truth.

Writes ``artifacts/variants.json``.

Usage:
    python scripts/02_mine_variants.py [--sample 300000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.mine import VariantMiner                       # noqa: E402
from trice.normalize import normalize_address, normalize_name  # noqa: E402
from trice.paths import artifacts_dir, dataset_dir        # noqa: E402

DATA = os.path.join(dataset_dir(ROOT), "train")
ART = artifacts_dir(ROOT)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300_000,
                    help="number of S1 training entities to sample")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--min-score", type=float, default=0.12)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    rng = random.Random(args.seed)

    # ---------------------------------------------------------------- sample GT ----
    log(f"sampling {args.sample:,} ground-truth groups")
    groups: list[tuple[str, list[str]]] = []
    n_seen = 0
    with open(os.path.join(DATA, "train_ground_truth.tsv"),
              encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if not rest:
                continue
            n_seen += 1
            if len(groups) < args.sample:
                groups.append((s1, rest.split(",")))
            else:
                j = rng.randrange(n_seen)
                if j < args.sample:
                    groups[j] = (s1, rest.split(","))

    need_s1 = {g[0] for g in groups}
    need_other = {m for _, ms in groups for m in ms}
    log(f"  need {len(need_s1):,} S1 and {len(need_other):,} S2/S3 records")

    # ---------------------------------------------------------------- load records ----
    # Store only the normalised token bags we need, never the raw strings, to stay
    # inside the memory budget.
    addr_tokens: dict[str, tuple[str, ...]] = {}
    name_tokens: dict[str, tuple[str, ...]] = {}

    def scan(fname: str, wanted: set[str]) -> None:
        t0 = time.time()
        kept = 0
        with open(os.path.join(DATA, fname), encoding="utf-8", errors="replace") as fh:
            next(fh)
            for line in fh:
                eid, tab, rest = line.partition("\t")
                if not tab or eid not in wanted:
                    continue
                parts = rest.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                nm = normalize_name(parts[0])
                ad = normalize_address(parts[1])
                name_tokens[eid] = tuple(nm.core_tokens)
                addr_tokens[eid] = tuple(ad.alpha_tokens)
                kept += 1
        log(f"  {fname}: kept {kept:,} in {time.time() - t0:.0f}s")

    scan("train_source1.tsv", need_s1)
    scan("train_source2.tsv", need_other)
    scan("train_source3.tsv", need_other)

    # ---------------------------------------------------------------- corpus freq ----
    addr_freq: Counter[str] = Counter()
    name_freq: Counter[str] = Counter()
    for toks in addr_tokens.values():
        addr_freq.update(toks)
    for toks in name_tokens.values():
        name_freq.update(toks)

    # ---------------------------------------------------------------- mine ----
    log("mining address aliases")
    addr_miner = VariantMiner()
    log("mining name aliases")
    name_miner = VariantMiner()

    n_used = 0
    for s1, mids in groups:
        a_addr = addr_tokens.get(s1)
        a_name = name_tokens.get(s1)
        if a_addr is None:
            continue
        for m in mids:
            b_addr = addr_tokens.get(m)
            if b_addr is not None:
                addr_miner.add(a_addr, b_addr)
            b_name = name_tokens.get(m)
            if a_name is not None and b_name is not None:
                name_miner.add(a_name, b_name)
        n_used += 1
    log(f"  fed {n_used:,} groups; addr pairs used={addr_miner.n_pairs:,} "
        f"name pairs used={name_miner.n_pairs:,}")

    addr_pairs = addr_miner.pairs(args.min_count, args.min_score)
    name_pairs = name_miner.pairs(args.min_count, args.min_score)
    log(f"  surviving alias pairs: address={len(addr_pairs):,} name={len(name_pairs):,}")

    addr_map = addr_miner.token_map(args.min_count, args.min_score, addr_freq)
    name_map = name_miner.token_map(args.min_count, args.min_score, name_freq)
    log(f"  token map sizes: address={len(addr_map):,} name={len(name_map):,}")

    print("\nTOP 60 MINED ADDRESS ALIASES (count, score, x <-> y)")
    for x, y, c, s in addr_pairs[:60]:
        print(f"  {c:>7,}  {s:.3f}  {x!r} <-> {y!r}")

    print("\nTOP 40 MINED NAME ALIASES")
    for x, y, c, s in name_pairs[:40]:
        print(f"  {c:>7,}  {s:.3f}  {x!r} <-> {y!r}")

    out = {
        "meta": {
            "sampled_groups": len(groups),
            "groups_used": n_used,
            "min_count": args.min_count,
            "min_score": args.min_score,
            "seed": args.seed,
        },
        "token_map": addr_map,
        "name_map": name_map,
        "address_pairs": [
            {"a": x, "b": y, "count": c, "score": round(s, 4)} for x, y, c, s in addr_pairs
        ],
        "name_pairs": [
            {"a": x, "b": y, "count": c, "score": round(s, 4)} for x, y, c, s in name_pairs
        ],
    }
    path = os.path.join(ART, "variants.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
