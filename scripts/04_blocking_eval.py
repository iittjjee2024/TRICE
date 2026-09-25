"""
Blocking recall / cost sweep on a training sample.

Blocking caps the recall of everything downstream, so this is the gate. The expensive
document-frequency count is done once per channel and reused across every configuration
in the sweep.

Usage:
    python scripts/04_blocking_eval.py --country US --queries 6000
    python scripts/04_blocking_eval.py --country India --queries 6000 --sweep wide
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.blocking import (BlockingConfig, ChannelConfig, build_channel_index,  # noqa
                            candidate_rank, count_df, encode_ids, group_bounds,
                            label_candidates, query_channel, union_channels,
                            vocabulary_from_df)

from trice.paths import artifacts_dir, dataset_dir  # noqa: E402

STORE = os.path.join(artifacts_dir(ROOT), "store")
GT = os.path.join(dataset_dir(ROOT), "train", "train_ground_truth.tsv")
COLS = ["num", "src", "country", "name_core", "name_skel", "name_nospace",
        "addr_alpha", "addr_digits", "postal", "house"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_truth_for(s1_ids: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(GT, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            s1, _, rest = line.partition("\t")
            if s1 in s1_ids:
                rest = rest.rstrip("\n")
                out[s1] = rest.split(",") if rest else []
                if len(out) == len(s1_ids):
                    break
    return out


# Each entry: label -> (name channel kwargs, addr channel kwargs, max_candidates,
#                       use_skeleton_shingles)
SWEEPS: dict[str, list[tuple]] = {
    "fast": [
        ("noshingle-3k",
         dict(df_cap=3_000, ns_caps={"w": 200, "S": 200}, top_k=22, min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, False),
        ("shingle1.2k",
         dict(df_cap=4_000, ns_caps={"k": 1_200, "w": 200, "S": 200}, top_k=22,
              min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, True),
    ],
    "wide": [
        ("noshingle-3k",
         dict(df_cap=3_000, ns_caps={"w": 200, "S": 200}, top_k=22, min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, False),
        ("noshingle-12k",
         dict(df_cap=12_000, ns_caps={"w": 200, "S": 200}, top_k=22, min_score=0.06),
         dict(df_cap=20_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, False),
        ("shingle600",
         dict(df_cap=4_000, ns_caps={"k": 600, "w": 200, "S": 200}, top_k=22,
              min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, True),
        ("shingle1.2k",
         dict(df_cap=4_000, ns_caps={"k": 1_200, "w": 200, "S": 200}, top_k=22,
              min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, True),
        ("shingle1.2k-k40",
         dict(df_cap=4_000, ns_caps={"k": 1_200, "w": 200, "S": 200}, top_k=28,
              min_score=0.05),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=24, min_score=0.08),
         44, True),
    ],
    "final": [
        ("chosen",
         dict(df_cap=4_000, ns_caps={"k": 1_200, "w": 200, "S": 200}, top_k=22,
              min_score=0.06),
         dict(df_cap=6_000, ns_caps={"H": 200, "D": 200}, top_k=18, min_score=0.10),
         32, True),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="US")
    ap.add_argument("--queries", type=int, default=6_000)
    ap.add_argument("--sweep", default="fast", choices=sorted(SWEEPS))
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    base = BlockingConfig()

    log(f"loading train store, country={args.country}")
    t0 = time.time()
    s1 = pd.read_parquet(os.path.join(STORE, "train_source1.parquet"), columns=COLS)
    s1 = s1[s1["country"] == args.country].reset_index(drop=True)
    idx_frames = []
    for src in (2, 3):
        f = pd.read_parquet(os.path.join(STORE, f"train_source{src}.parquet"),
                            columns=COLS)
        f = f[f["country"] == args.country].reset_index(drop=True)
        idx_frames.append(f)
    n_index = sum(len(f) for f in idx_frames)
    log(f"  S1={len(s1):,}  index={n_index:,}  ({time.time() - t0:.0f}s)")

    # index row -> (src, num) for labelling
    idx_src = np.concatenate([f["src"].to_numpy(np.int8) for f in idx_frames])
    idx_num = np.concatenate([f["num"].to_numpy(np.int32) for f in idx_frames])

    rng = np.random.default_rng(args.seed)
    n_q = min(args.queries, len(s1))
    q_pos = np.sort(rng.choice(len(s1), size=n_q, replace=False))
    queries = s1.iloc[q_pos].reset_index(drop=True)
    q_ids = [f"S1-{n}" for n in queries["num"].to_numpy()]

    log("loading ground truth for the sample")
    truth = load_truth_for(set(q_ids))
    truth_sizes = np.array([len(truth.get(i, [])) for i in q_ids], dtype=np.int32)
    truth_codes = []
    for i in q_ids:
        ids = truth.get(i, [])
        if not ids:
            truth_codes.append(np.zeros(0, dtype=np.int64))
            continue
        src = np.array([2 if x[1] == "2" else 3 for x in ids], dtype=np.int8)
        num = np.array([int(x[3:]) for x in ids], dtype=np.int64)
        truth_codes.append(np.sort(encode_ids(src, num)))
    log(f"  mean |T|={truth_sizes.mean():.3f}  singletons={(truth_sizes == 0).mean():.2%}")

    # ------------------------------------------------ df counts, once per channel ----
    counts = {}
    for spec in base.channels:
        log(f"counting df for channel '{spec.name}' (namespaces={spec.namespaces})")
        t0 = time.time()
        df, n_docs = count_df(idx_frames, base, spec.namespaces)
        counts[spec.name] = (df, n_docs)
        log(f"  {len(df):,} distinct tokens in {time.time() - t0:.0f}s")

    rows = []
    for (label, name_kw, addr_kw, max_cand, shingles) in SWEEPS[args.sweep]:
        cfg = BlockingConfig(
            use_skeleton_shingles=shingles,
            max_candidates=max_cand,
            channels=[
                ChannelConfig("name", ("n", "k", "w", "S"), **name_kw),
                ChannelConfig("addr", ("a", "d", "H", "D"), **addr_kw),
            ])
        tag = (f"{label}: name(cap={name_kw['df_cap']},k={name_kw['top_k']}) "
               f"addr(cap={addr_kw['df_cap']},k={addr_kw['top_k']}) "
               f"cap={max_cand} shingles={shingles}")
        print()
        log(f"=== {tag}")

        results = []
        build_s = query_s = 0.0
        per_channel = {}
        for spec in cfg.channels:
            df, n_docs = counts[spec.name]
            # NOTE: when shingles are off the 'k' tokens simply never get generated, so
            # the cached count is still a superset and the vocabulary filter is harmless.
            vocab = vocabulary_from_df(df, n_docs, cfg, spec)
            t0 = time.time()
            index = build_channel_index(idx_frames, vocab, cfg, spec)
            build_s += time.time() - t0
            t0 = time.time()
            qr, ir, sc = query_channel(index, queries, cfg, spec)
            query_s += time.time() - t0
            results.append((spec.name, qr, ir, sc))

            # per-channel recall
            cc = encode_ids(idx_src[ir], idx_num[ir])
            st, en = group_bounds(qr, n_q)
            lab = np.zeros(len(qr), dtype=np.int8)
            for g in range(n_q):
                if en[g] > st[g]:
                    lab[st[g]:en[g]] = label_candidates(cc[st[g]:en[g]], truth_codes[g])
            found = np.zeros(n_q, dtype=np.int32)
            np.add.at(found, qr[lab == 1], 1)
            per = np.where(truth_sizes > 0, found / np.maximum(truth_sizes, 1), 1.0)
            per_channel[spec.name] = {
                "vocab": len(vocab), "nnz": int(index.matrix.nnz),
                "gb": round(index.nbytes() / 1e9, 2),
                "pairs": int(len(qr)), "macro_recall": float(per.mean()),
            }
            log(f"    {spec.name}: vocab={len(vocab):,} nnz={index.matrix.nnz:,} "
                f"({index.nbytes() / 1e9:.2f}GB) pairs={len(qr):,} "
                f"macro_recall={per.mean():.4f}")
            del index, vocab

        q_row, i_row, score, mask = union_channels(
            results, n_q, cfg, [c.name for c in cfg.channels])
        del results

        cand_codes = encode_ids(idx_src[i_row], idx_num[i_row])
        starts, ends = group_bounds(q_row, n_q)
        labels = np.zeros(len(q_row), dtype=np.int8)
        for g in range(n_q):
            if ends[g] > starts[g]:
                labels[starts[g]:ends[g]] = label_candidates(
                    cand_codes[starts[g]:ends[g]], truth_codes[g])

        found = np.zeros(n_q, dtype=np.int32)
        np.add.at(found, q_row[labels == 1], 1)
        per_entity = np.where(truth_sizes > 0, found / np.maximum(truth_sizes, 1), 1.0)
        n_cand = ends - starts
        qps = n_q / max(query_s, 1e-9)

        # channel attribution: true links found ONLY by one channel
        excl = {}
        for i, c in enumerate(cfg.channels):
            bit = 1 << i
            only = (mask == bit) & (labels == 1)
            excl[c.name] = int(only.sum())

        print(f"    UNION  pairs={len(q_row):,}  mean_cand={n_cand.mean():.2f}  "
              f"macro_recall={per_entity.mean():.4f}")
        print(f"           zero-recall entities={int(((truth_sizes > 0) & (found == 0)).sum()):,}"
              f" ({((truth_sizes > 0) & (found == 0)).mean():.2%})"
              f"   full-recall={int((per_entity >= 1).sum()):,} "
              f"({(per_entity >= 1).mean():.1%})")
        print(f"           positives={int((labels == 1).sum()):,} "
              f"({(labels == 1).mean():.3%} of pairs)")
        print(f"           exclusive true links by channel: {excl}")
        print(f"           build={build_s:.0f}s query={query_s:.0f}s "
              f"({qps:,.0f} q/s) -> full test ~{1_732_544 / qps / 60:.0f} min")

        rows.append({
            "tag": tag, "label": label, "name": name_kw, "addr": addr_kw,
            "max_cand": max_cand, "shingles": shingles,
            "pairs": int(len(q_row)), "mean_cand": float(n_cand.mean()),
            "macro_recall": float(per_entity.mean()),
            "micro_recall": float(found.sum() / max(1, truth_sizes.sum())),
            "zero_recall_frac": float(((truth_sizes > 0) & (found == 0)).mean()),
            "positive_frac": float((labels == 1).mean()),
            "queries_per_s": float(qps), "build_s": build_s, "query_s": query_s,
            "per_channel": per_channel, "exclusive_true": excl,
        })

    print()
    print("=" * 110)
    print(f"SUMMARY  country={args.country}  queries={n_q:,}  index={n_index:,}")
    print("=" * 110)
    print(f"{'config':<58} {'recall':>7} {'cand':>6} {'pos%':>6} {'q/s':>7} {'test min':>9}")
    for r in rows:
        print(f"{r['tag']:<58} {r['macro_recall']:>7.4f} {r['mean_cand']:>6.1f} "
              f"{r['positive_frac'] * 100:>6.2f} {r['queries_per_s']:>7.0f} "
              f"{1_732_544 / r['queries_per_s'] / 60:>9.0f}")

    outdir = os.path.join(artifacts_dir(ROOT), "blocking")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"sweep_{args.country}_{args.sweep}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"country": args.country, "n_queries": n_q, "n_index": n_index,
                   "rows": rows}, fh, indent=1, default=float)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
