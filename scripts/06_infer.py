"""
Full test-set inference: 1.73 M Source-1 entities against ~10 M Source-2/3 records.

Memory strategy
---------------
Peak memory, not CPU, is the binding constraint (16.9 GB). Per country:

* **Pass 1** streams query batches. For each batch only the index records that batch
  actually references are prepared, features are built, stage-1 scores them, and the heavy
  raw feature matrix is discarded immediately -- only the stage-1 probability and the
  curated ``STAGE2_RAW_FEATURES`` slice are retained.
* **Graph features** are then computed once over the *whole* country, so competition
  normalisation sees every Source-1 entity contending for a candidate. Batching this step
  would silently weaken it, which is why the pass is split this way rather than doing
  everything in one sweep.
* **Pass 2** applies stage 2, calibration and the expected-F_0.5 decision.

Output is written in ``test_source1.tsv`` order so every required entity appears exactly
once, including entities for which blocking found nothing.

Usage:
    python scripts/06_infer.py --run-id <train run id>
    python scripts/06_infer.py --run-id r-... --countries France      # single partition
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

from trice.blocking import BlockingConfig, ChannelConfig                      # noqa: E402
from trice.decide import DecisionConfig                                       # noqa: E402
from trice.export import (read_entity_ids, validate_submission,               # noqa: E402
                          write_candidate_pairs, write_matching_results)
from trice.features import FEATURE_NAMES, prepare_side                        # noqa: E402
from trice.graph import GraphConfig, build_graph_features, repair_disjointness  # noqa
from trice.model import load_bundle                                           # noqa: E402
from trice.pipeline import (Candidates, build_feature_matrix, compute_idf,     # noqa: E402
                           estimate_missing_mass, generate_candidates,
                           hash_codes, load_partition, split_countries)

STORE = os.path.join(ROOT, "artifacts", "store")
RUNS = os.path.join(ROOT, "artifacts", "runs")
TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")
OUTPUT = os.path.join(ROOT, "output")

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--query-batch", type=int, default=120_000)
    ap.add_argument("--output-dir", default=OUTPUT)
    ap.add_argument("--max-emit", type=int, default=25)
    ap.add_argument("--prune-epsilon", type=float, default=0.01)
    ap.add_argument("--check-ids", action="store_true",
                    help="verify every emitted id exists in the test set (memory heavy)")
    ap.add_argument("--scratch", default=os.path.join(ROOT, "artifacts", "scratch"),
                    help="directory for the on-disk feature spill")
    args = ap.parse_args()
    scratch = args.scratch
    os.makedirs(scratch, exist_ok=True)

    rundir = os.path.join(RUNS, args.run_id)
    bundle = load_bundle(os.path.join(rundir, "model.pkl"))
    m1 = bundle["stage1"]
    m2 = bundle["stage2"]
    cal = bundle["calibrator"]
    blk = bundle["blocking"]
    use_stage2 = bool(bundle.get("use_stage2", True))
    recall_ceiling = float(bundle.get("overall_blocking_recall", 0.88))
    mm_model = bundle.get("missing_mass_model")
    mm_scale = float(bundle.get("missing_mass_scale", 1.0))
    log(f"loaded bundle from {rundir} (model={m1.kind}, use_stage2={use_stage2}, "
        f"blocking_recall={recall_ceiling:.4f}, "
        f"missing_mass={'fitted' if mm_model is not None else 'heuristic'} "
        f"scale={mm_scale})")

    # rebuild the blocking config exactly as trained
    block_cfg = BlockingConfig(
        use_skeleton_shingles=blk["use_skeleton_shingles"],
        shingle_size=blk["shingle_size"], max_shingles=blk["max_shingles"],
        w_name=blk["w_name"], w_shingle=blk["w_shingle"], w_nospace=blk["w_nospace"],
        w_skel=blk["w_skel"], w_addr=blk["w_addr"], w_digit=blk["w_digit"],
        w_house_anchor=blk["w_house_anchor"], w_digit_sig=blk["w_digit_sig"],
        min_df=blk["min_df"], max_candidates=blk["max_candidates"],
        channels=[ChannelConfig(**c) for c in blk["channels"]])
    graph_cfg = GraphConfig(**bundle["graph_cfg"])

    # Prefer the configuration chosen by scripts/07_tune_decision.py over CLI defaults, so
    # inference cannot silently diverge from what validation actually selected.
    tuned = bundle.get("decision", {}) or {}
    dec_cfg = DecisionConfig(
        beta=0.5, rule="expected_f",
        prune_epsilon=float(tuned.get("prune_epsilon", args.prune_epsilon)),
        max_emit=int(tuned.get("max_emit", args.max_emit)),
        probability_power=float(tuned.get("probability_power", 1.0)))
    do_repair = bool(bundle.get("disjointness_repair", True))
    log(f"decision: prune_eps={dec_cfg.prune_epsilon} max_emit={dec_cfg.max_emit} "
        f"gamma={dec_cfg.probability_power} repair={do_repair} "
        f"missing_mass={bundle.get('missing_mass_variant', 'fitted')}")

    countries = args.countries or split_countries(STORE, "test")
    log(f"test countries: {countries}")

    # Results are written to per-partition shard files on disk rather than accumulated in
    # RAM. Holding all three partitions' id-string lists (~785 k matched lists, ~12 M
    # candidate ids) alongside the blocking matmul buffers overran the 16 GB budget; each
    # partition is now fully freed before the next begins, and the final TSVs are assembled
    # from the shards in test_source1.tsv order at the very end.
    shard_dir = os.path.join(scratch, f"shards_{args.run_id}")
    os.makedirs(shard_dir, exist_ok=True)
    for old in os.listdir(shard_dir):
        os.remove(os.path.join(shard_dir, old))
    per_country_stats = {}

    for country in countries:
        log(f"================ partition {country}")
        t_c = time.time()
        s1, idx_frames = load_partition(STORE, "test", country)
        n_index = sum(len(f) for f in idx_frames)
        log(f"  S1={len(s1):,} index={n_index:,}")

        log("  blocking")
        cand, bstats = generate_candidates(s1, idx_frames, block_cfg, log)
        log(f"  candidates: {len(cand):,} (mean {cand.counts.mean():.2f}/entity, "
            f"{int((cand.counts == 0).sum()):,} entities with none)")

        log("  computing IDF")
        idf = compute_idf([s1] + idx_frames)

        idx_all = pd.concat(idx_frames, ignore_index=True)
        del idx_frames
        idx_src = idx_all["src"].to_numpy(np.int8)
        idx_num = idx_all["num"].to_numpy(np.int32)
        postal_code = hash_codes(idx_all["postal"].fillna("").to_numpy(dtype=object))
        digit_code = hash_codes(idx_all["addr_digits"].fillna("").to_numpy(dtype=object))
        skel_code = hash_codes(idx_all["name_skel"].fillna("").to_numpy(dtype=object))

        n_pairs = len(cand)
        n_ent = len(s1)
        p1 = np.zeros(n_pairs, dtype=np.float32)

        # The full pairwise matrix is spilled to disk rather than held in RAM: stage 2
        # needs it again after the graph features (which require globally-computed stage-1
        # probabilities), and at ~40 M pairs x 55 float32 it is ~9 GB -- fine on disk,
        # impossible in a 16.9 GB budget alongside everything else.
        mm_path = os.path.join(scratch, f"X_{country}.f32")
        X_mm = np.memmap(mm_path, dtype=np.float32, mode="w+",
                         shape=(n_pairs, len(FEATURE_NAMES)))
        log(f"  pass 1: features + stage 1 in batches of {args.query_batch:,} entities"
            f"  (spilling {n_pairs * len(FEATURE_NAMES) * 4 / 1e9:.1f} GB to {mm_path})")

        q_side_full = prepare_side(s1)
        counts_per_pair = np.repeat(cand.counts, cand.counts).astype(np.float32)
        from trice.features import FeatureBuilder
        fb = FeatureBuilder(idf)
        t0 = time.time()
        done_pairs = 0
        for b_start in range(0, n_ent, args.query_batch):
            b_end = min(b_start + args.query_batch, n_ent)
            lo = int(cand.starts[b_start])
            hi = int(cand.ends[b_end - 1])
            if hi <= lo:
                continue
            sl = slice(lo, hi)
            ref, inv = np.unique(cand.c_row[sl], return_inverse=True)
            c_side = prepare_side(idx_all.iloc[ref].reset_index(drop=True))
            inv = inv.astype(np.int32)

            q_b = cand.q_row[sl]
            sc_b = cand.score[sl]
            pr_b = cand.probe[sl]
            rk_b = cand.rank[sl].astype(np.float32)
            cn_b = counts_per_pair[sl]
            step = 400_000
            for s in range(0, hi - lo, step):
                e = min(s + step, hi - lo)
                Xb = fb.build(q_side_full, c_side, q_b[s:e], inv[s:e],
                              sc_b[s:e], pr_b[s:e], rk_b[s:e], cn_b[s:e])
                p1[lo + s:lo + e] = m1.predict(Xb)
                X_mm[lo + s:lo + e] = Xb
                del Xb
            del c_side, inv
            done_pairs += hi - lo
            el = time.time() - t0
            log(f"    {b_end:,}/{n_ent:,} entities, {done_pairs:,} pairs "
                f"({done_pairs / max(el, 1e-9):,.0f} pairs/s)")
        X_mm.flush()
        del q_side_full, idf, idx_all, fb, counts_per_pair

        # ------------------------------------------------------------- graph ----
        log("  graph features (global competition within partition)")
        t0 = time.time()
        G = build_graph_features(
            cand.q_row.astype(np.int64), cand.c_row.astype(np.int64), p1,
            idx_src[cand.c_row], postal_code[cand.c_row], digit_code[cand.c_row],
            skel_code[cand.c_row], graph_cfg)
        log(f"    {G.shape} in {time.time() - t0:.0f}s ({G.nbytes / 1e9:.2f} GB)")
        del postal_code, digit_code, skel_code

        # ------------------------------------------------------------- pass 2 ----
        if use_stage2:
            log("  pass 2: stage 2 (full features + graph) + calibration")
            p2 = np.zeros(n_pairs, dtype=np.float32)
            step = 1_000_000
            for s in range(0, n_pairs, step):
                e = min(s + step, n_pairs)
                p2[s:e] = m2.predict(
                    np.concatenate([np.asarray(X_mm[s:e]), G[s:e]], axis=1))
        else:
            log("  pass 2: stage 2 disabled by training-time validation; using stage 1")
            p2 = p1
        del G
        del X_mm
        try:
            os.remove(mm_path)
        except OSError:
            pass
        groups = np.full(n_pairs, country, dtype=object)
        p_cal = cal.transform(p2, groups)
        del p2, groups
        log(f"    calibrated; own curve for {country}: "
            f"{country in cal.per_group}")

        # ------------------------------------------------------------- decide ----
        log("  decision layer")
        counts = cand.counts
        p_sum = np.zeros(n_ent)
        np.add.at(p_sum, cand.q_row, p_cal)
        variant = bundle.get("missing_mass_variant")
        if variant == "off" or mm_scale == 0.0:
            miss = None
        elif variant == "heuristic" or mm_scale < 0.0:
            miss = estimate_missing_mass(counts, p_sum, recall_ceiling)
        elif mm_model is not None:
            miss = mm_model.missing_mass(counts, p_sum, mm_scale)
        else:
            miss = estimate_missing_mass(counts, p_sum, recall_ceiling)
        log(f"    missing-mass mean="
            f"{(miss.mean() if miss is not None else 0.0):.3f} "
            f"(Sum p={p_sum.mean():.3f})")
        from trice.decide import decide_groups
        t0 = time.time()
        mask, k_star, exp_f = decide_groups(p_cal, cand.starts, cand.ends, dec_cfg,
                                            missing_mass=miss)
        n_before = int(mask.sum())
        if do_repair:
            mask = repair_disjointness(cand.q_row, cand.c_row, mask, p_cal)
        log(f"    selected {int(mask.sum()):,} pairs "
            f"({n_before - int(mask.sum()):,} dropped by disjointness repair) "
            f"in {time.time() - t0:.0f}s")
        log(f"    mean k*={k_star.mean():.3f}  empty entities="
            f"{int((k_star == 0).sum()):,} ({(k_star == 0).mean():.2%})")

        # ------------------------------------------------- write partition shard ----
        # One TSV shard per partition: "S1-<num> \t match_ids \t candidate_ids". Assembled
        # into the final, ordered output files after every partition is done.
        ent_num = s1["num"].to_numpy()
        cand_id_str = np.char.add(
            np.char.add("S", idx_src[cand.c_row].astype(str)),
            np.char.add("-", idx_num[cand.c_row].astype(str)))
        shard_path = os.path.join(shard_dir, f"{country}.tsv")
        with open(shard_path, "w", encoding="utf-8", newline="") as sh:
            for g in range(n_ent):
                st, en = int(cand.starts[g]), int(cand.ends[g])
                eid = f"S1-{ent_num[g]}"
                if en <= st:
                    sh.write(f"{eid}\t\t\n")
                    continue
                block_ids = cand_id_str[st:en]
                cand_join = ",".join(block_ids.tolist())
                sel = mask[st:en]
                match_join = ",".join(block_ids[sel].tolist()) if sel.any() else ""
                sh.write(f"{eid}\t{match_join}\t{cand_join}\n")
        log(f"  wrote shard {shard_path}")
        del cand_id_str, ent_num

        per_country_stats[country] = {
            "n_entities": n_ent, "n_index": n_index, "n_pairs": int(n_pairs),
            "mean_candidates": float(counts.mean()),
            "entities_without_candidates": int((counts == 0).sum()),
            "selected_pairs": int(mask.sum()),
            "mean_k": float(k_star.mean()),
            "empty_predictions": int((k_star == 0).sum()),
            "blocking": bstats,
            "seconds": round(time.time() - t_c, 1),
        }
        log(f"  partition done in {time.time() - t_c:.0f}s")
        del cand, p_cal, mask, k_star, exp_f, s1, idx_src, idx_num

    # ------------------------------------------------- assemble from shards ----
    # Load the compact shards (one line per entity, match + candidate lists) into two
    # dicts and emit in test_source1.tsv order so every required entity appears exactly
    # once. The shards hold only id strings, a small fraction of the peak inference memory.
    log("assembling submission files from partition shards")
    matched: dict[str, list[str]] = {}
    candidates_out: dict[str, list[str]] = {}
    for country in countries:
        shard_path = os.path.join(shard_dir, f"{country}.tsv")
        if not os.path.isfile(shard_path):
            continue
        with open(shard_path, encoding="utf-8") as sh:
            for line in sh:
                eid, match_join, cand_join = line.rstrip("\n").split("\t")
                matched[eid] = match_join.split(",") if match_join else []
                candidates_out[eid] = cand_join.split(",") if cand_join else []

    log("writing submission files in test_source1.tsv order")
    required = read_entity_ids(os.path.join(TEST_DIR, "test_source1.tsv"))
    log(f"  {len(required):,} required Source-1 entities")

    os.makedirs(args.output_dir, exist_ok=True)
    mpath = os.path.join(args.output_dir, "matching_results.tsv")
    cpath = os.path.join(args.output_dir, "candidate_pairs.tsv")
    n_rows, n_non = write_matching_results(
        mpath, required, (matched.get(e, []) for e in required))
    log(f"  {mpath}: {n_rows:,} rows, {n_non:,} non-empty")
    cn_rows, cn_non = write_candidate_pairs(
        cpath, required, (candidates_out.get(e, []) for e in required))
    log(f"  {cpath}: {cn_rows:,} rows, {cn_non:,} non-empty")

    # ---------------------------------------------------------------- validate ----
    log("validating output format")
    targets = None
    if args.check_ids:
        targets = set(read_entity_ids(os.path.join(TEST_DIR, "test_source2.tsv")))
        targets |= set(read_entity_ids(os.path.join(TEST_DIR, "test_source3.tsv")))
        log(f"  loaded {len(targets):,} valid S2/S3 ids")
    report = validate_submission(mpath, cpath, set(required), targets)
    print(json.dumps(report, indent=1))

    summary = {
        "run_id": args.run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "countries": countries,
        "per_country": per_country_stats,
        "total_entities": len(required),
        "total_predicted_nonempty": n_non,
        "validation": report,
        "seconds": round(time.time() - T0, 1),
    }
    with open(os.path.join(rundir, "inference.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    log(f"TOTAL {time.time() - T0:.0f}s   ok={report['ok']}")


if __name__ == "__main__":
    main()
