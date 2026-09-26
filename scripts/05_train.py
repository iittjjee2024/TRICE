"""
Train the TRICE matcher and score a held-out validation split.

Phases
------
A. per country: block -> label -> pairwise features            (record frames discarded)
B. train stage-1 GBDT on the pooled pairs
C. per country: stage-1 scores -> graph features
D. train stage-2 GBDT (stacked)
E. per-country isotonic calibration
F. decision layer + rule comparison + ablation
G. persist bundle, metrics and the validation arrays the UI needs

Usage
-----
    python scripts/05_train.py --entities 80000
    python scripts/05_train.py --entities 60000 --holdout-country India   # zero-shot sim
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.blocking import BlockingConfig                                    # noqa: E402
from trice.decide import DecisionConfig                                      # noqa: E402
from trice.evaluate import blocking_recall, f_beta, score_flat               # noqa: E402
from trice.features import (FEATURE_NAMES, STAGE2_RAW_FEATURES,              # noqa: E402
                            STAGE2_RAW_INDEX, prepare_side)
from trice.graph import (GRAPH_FEATURE_NAMES, GraphConfig,                   # noqa: E402
                         build_graph_features, repair_disjointness)
from trice.model import (GBDT, GroupCalibrator, ModelConfig,                 # noqa: E402
                         feature_separation, rank_metrics, save_bundle)
from trice.pipeline import (Candidates, MissingMassEstimator, attach_labels,  # noqa: E402
                           build_feature_matrix, compact_candidates, compute_idf,
                           estimate_missing_mass, generate_candidates, hash_codes,
                           load_partition, load_ground_truth_subset,
                           split_countries, truth_arrays)

from trice.paths import artifacts_dir, dataset_dir  # noqa: E402

STORE = os.path.join(artifacts_dir(ROOT), "store")
GT = os.path.join(dataset_dir(ROOT), "train", "train_ground_truth.tsv")
RUNS = os.path.join(artifacts_dir(ROOT), "runs")

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=80_000,
                    help="training S1 entities sampled per country")
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--holdout-country", default=None,
                    help="train without this country and validate only on it "
                         "(zero-shot simulation for France)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-emit", type=int, default=25)
    ap.add_argument("--train-rows", type=int, default=2_200_000,
                    help="cap on pairs used to FIT each GBDT (all positives are kept; "
                         "negatives are subsampled). Scoring and calibration always use "
                         "the full set, so the base rate the calibrator sees is correct.")
    ap.add_argument("--val-rows", type=int, default=600_000,
                    help="cap on pairs used for early stopping")
    args = ap.parse_args()

    run_id = args.run_id or time.strftime("r-%Y%m%d-%H%M%S")
    outdir = os.path.join(RUNS, run_id)
    os.makedirs(outdir, exist_ok=True)

    block_cfg = BlockingConfig()
    model_cfg = ModelConfig()
    graph_cfg = GraphConfig()
    rng = np.random.default_rng(args.seed)

    countries = split_countries(STORE, "train")
    log(f"run {run_id}; train countries = {countries}")
    if not countries:
        raise SystemExit(
            f"no country partitions found in the record store at {STORE}.\n"
            f"  The store is empty or missing - re-run scripts/03_prepare.py --force "
            f"(a previous broken run may have left empty Parquet files).")

    # ==================================================================== phase A ====
    parts = []
    c_offset = 0
    block_stats_all = {}
    for country in countries:
        log(f"--- partition {country}")
        t0 = time.time()
        s1, idx_frames = load_partition(STORE, "train", country)
        n_index = sum(len(f) for f in idx_frames)
        log(f"    S1={len(s1):,} index={n_index:,} loaded in {time.time() - t0:.0f}s")

        n_q = min(args.entities, len(s1))
        pos = np.sort(rng.choice(len(s1), size=n_q, replace=False))
        queries = s1.iloc[pos].reset_index(drop=True)
        del s1
        q_ids = [f"S1-{n}" for n in queries["num"].to_numpy()]

        truth = load_ground_truth_subset(GT, set(q_ids))
        codes, sizes = truth_arrays(q_ids, truth)
        del truth
        log(f"    sampled {n_q:,} entities, mean |T|={sizes.mean():.3f}, "
            f"singletons={(sizes == 0).mean():.2%}")

        cand, bstats = generate_candidates(queries, idx_frames, block_cfg, log)
        block_stats_all[country] = bstats
        log(f"    candidates: {len(cand):,} "
            f"(mean {cand.counts.mean():.2f}/entity)")

        idx_src_all = np.concatenate([f["src"].to_numpy(np.int8) for f in idx_frames])
        idx_num_all = np.concatenate([f["num"].to_numpy(np.int32) for f in idx_frames])
        attach_labels(cand, idx_src_all, idx_num_all, codes)
        rec = blocking_recall(cand.labels, cand.starts, cand.ends, sizes)
        log(f"    blocking macro recall={rec['macro_recall']:.4f} "
            f"micro={rec['micro_recall']:.4f} positives={cand.labels.mean():.3%}")

        # IDF must be measured over the whole partition, before subsetting
        log("    computing IDF")
        idf = compute_idf([queries] + idx_frames)

        # keep only the index records candidates actually reference
        idx_all = pd.concat(idx_frames, ignore_index=True)
        del idx_frames
        referenced = compact_candidates(cand)
        idx_sub = idx_all.iloc[referenced].reset_index(drop=True)
        del idx_all
        log(f"    compacted index side: {len(referenced):,} referenced records "
            f"(of {n_index:,})")

        cand_src = idx_src_all[referenced][cand.c_row]
        cand_num = idx_num_all[referenced][cand.c_row]
        del idx_src_all, idx_num_all
        postal_code = hash_codes(idx_sub["postal"].fillna("").to_numpy(dtype=object))
        digit_code = hash_codes(idx_sub["addr_digits"].fillna("").to_numpy(dtype=object))
        skel_code = hash_codes(idx_sub["name_skel"].fillna("").to_numpy(dtype=object))

        log("    pairwise features")
        q_side = prepare_side(queries)
        c_side = prepare_side(idx_sub)
        del idx_sub
        t0 = time.time()
        X = build_feature_matrix(cand, q_side, c_side, idf, chunk=500_000)
        log(f"    features {X.shape} in {time.time() - t0:.0f}s "
            f"({len(cand) / max(time.time() - t0, 1e-9):,.0f} pairs/s)")

        # keep a small sample of readable records for the entity explorer
        sample_pairs = np.sort(rng.choice(len(cand), size=min(40_000, len(cand)),
                                         replace=False))
        explorer = {
            "pair_idx": sample_pairs.astype(np.int64),
            "q_name": np.array([q_side["name_core"][i] for i in cand.q_row[sample_pairs]],
                               dtype=object),
            "q_addr": np.array([q_side["addr_alpha"][i] for i in cand.q_row[sample_pairs]],
                               dtype=object),
            "c_name": np.array([c_side["name_core"][i] for i in cand.c_row[sample_pairs]],
                               dtype=object),
            "c_addr": np.array([c_side["addr_alpha"][i] for i in cand.c_row[sample_pairs]],
                               dtype=object),
        }
        del q_side, c_side, idf

        parts.append({
            "country": country,
            "q_ids": np.array(q_ids, dtype=object),
            "truth_sizes": sizes,
            "X": X,
            "y": cand.labels,
            "q_row": cand.q_row.astype(np.int64),
            "c_row": cand.c_row.astype(np.int64) + c_offset,
            "cand_src": cand_src,
            "postal_code": postal_code[cand.c_row],
            "digit_code": digit_code[cand.c_row],
            "skel_code": skel_code[cand.c_row],
            "cand_num": cand_num,
            "starts": cand.starts, "ends": cand.ends,
            "blocking_recall": rec,
            "explorer": explorer,
        })
        c_offset += len(referenced)
        del cand, postal_code, digit_code, skel_code, cand_src, cand_num, queries

    # ------------------------------------------------------------- pool partitions ----
    log("--- pooling partitions")
    ent_off = 0
    pair_parts = []
    for p in parts:
        p["q_global"] = p["q_row"] + ent_off
        p["entity_country"] = np.full(len(p["q_ids"]), p["country"], dtype=object)
        ent_off += len(p["q_ids"])
        pair_parts.append(p)

    X = np.concatenate([p["X"] for p in pair_parts])
    y = np.concatenate([p["y"] for p in pair_parts]).astype(np.int8)
    q_global = np.concatenate([p["q_global"] for p in pair_parts])
    c_global = np.concatenate([p["c_row"] for p in pair_parts])
    cand_src = np.concatenate([p["cand_src"] for p in pair_parts])
    cand_num = np.concatenate([p["cand_num"] for p in pair_parts])
    postal_code = np.concatenate([p["postal_code"] for p in pair_parts])
    digit_code = np.concatenate([p["digit_code"] for p in pair_parts])
    skel_code = np.concatenate([p["skel_code"] for p in pair_parts])
    entity_ids = np.concatenate([p["q_ids"] for p in pair_parts])
    entity_country = np.concatenate([p["entity_country"] for p in pair_parts])
    truth_sizes = np.concatenate([p["truth_sizes"] for p in pair_parts])
    for p in pair_parts:
        del p["X"], p["y"]
    n_entities = len(entity_ids)
    log(f"    pooled: {len(X):,} pairs over {n_entities:,} entities, "
        f"{X.nbytes / 1e9:.2f} GB")

    # ------------------------------------------------------------- train/val split ----
    if args.holdout_country:
        val_ent = entity_country == args.holdout_country
        if not val_ent.any():
            raise SystemExit(f"holdout country {args.holdout_country} not in train data")
        log(f"    ZERO-SHOT mode: validating only on {args.holdout_country}")
    else:
        val_ent = rng.random(n_entities) < args.val_fraction
    pair_is_val = val_ent[q_global]
    log(f"    entities: train={int((~val_ent).sum()):,} val={int(val_ent.sum()):,}")
    log(f"    pairs:    train={int((~pair_is_val).sum()):,} val={int(pair_is_val.sum()):,}")

    # ==================================================================== phase B ====
    tr = ~pair_is_val
    va = pair_is_val

    # Fitting rows are capped: 4.9 M x 79 features upcast to float64 by some backends
    # exceeds the memory budget, and a GBDT converges long before that many rows. All
    # positives are kept and negatives are subsampled; scoring and calibration still run
    # over the full set so the calibrator sees the true base rate.
    def fit_subset(mask: np.ndarray, cap: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) <= cap:
            return idx
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        room = max(cap - len(pos), cap // 10)
        if len(neg) > room:
            neg = rng.choice(neg, size=room, replace=False)
        out = np.concatenate([pos, neg])
        out.sort()
        return out

    fit_idx = fit_subset(tr, args.train_rows)
    es_idx = fit_subset(va, args.val_rows)
    log(f"    fitting rows: {len(fit_idx):,} (positives "
        f"{int(y[fit_idx].sum()):,} = {y[fit_idx].mean():.2%}); "
        f"early-stopping rows: {len(es_idx):,}")

    log("--- stage 1: pairwise GBDT")
    m1 = GBDT(model_cfg)
    t0 = time.time()
    m1.fit(X[fit_idx], y[fit_idx], X[es_idx], y[es_idx],
           feature_names=list(FEATURE_NAMES))
    log(f"    trained ({m1.kind}) in {time.time() - t0:.0f}s, "
        f"~{m1.n_parameters():,} parameters")
    p1 = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), 1_000_000):
        e = min(s + 1_000_000, len(X))
        p1[s:e] = m1.predict(X[s:e])
    s1_metrics = rank_metrics(p1[va], y[va])
    log(f"    stage1 val AUC={s1_metrics['auc']:.5f} "
        f"AP={s1_metrics['average_precision']:.5f}")

    # ==================================================================== phase C ====
    log("--- graph features (competition + corroboration)")
    t0 = time.time()
    G = build_graph_features(q_global.astype(np.int64), c_global.astype(np.int64),
                             p1, cand_src, postal_code, digit_code, skel_code, graph_cfg)
    log(f"    graph matrix {G.shape} in {time.time() - t0:.0f}s")

    # ==================================================================== phase D ====
    # Stage 2 sees the FULL pairwise matrix plus the graph features. Inference affords this
    # by spilling the pairwise matrix to an on-disk memmap (see scripts/06_infer.py); an
    # earlier version fed stage 2 only a 12-column slice and the truncation cost more than
    # the graph features added.
    all_names = list(FEATURE_NAMES) + list(GRAPH_FEATURE_NAMES)
    log(f"--- stage 2: stacked GBDT on {len(all_names)} features")
    m2 = GBDT(model_cfg)
    t0 = time.time()
    # Only the fitting slices are materialised as a combined matrix, never all 6.6 M rows.
    m2.fit(np.concatenate([X[fit_idx], G[fit_idx]], axis=1), y[fit_idx],
           np.concatenate([X[es_idx], G[es_idx]], axis=1), y[es_idx],
           feature_names=all_names)
    log(f"    trained in {time.time() - t0:.0f}s, ~{m2.n_parameters():,} parameters")
    p2 = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), 1_000_000):
        e = min(s + 1_000_000, len(X))
        p2[s:e] = m2.predict(np.concatenate([X[s:e], G[s:e]], axis=1))
    del G
    s2_metrics = rank_metrics(p2[va], y[va])
    log(f"    stage2 val AUC={s2_metrics['auc']:.5f} "
        f"AP={s2_metrics['average_precision']:.5f}  "
        f"(stage1 AP was {s1_metrics['average_precision']:.5f})")

    # Adopt stage 2 only if it genuinely beats stage 1 on held-out average precision.
    # Recorded in the bundle so inference cannot silently run the weaker model.
    use_stage2 = bool(s2_metrics["average_precision"] >= s1_metrics["average_precision"])
    log(f"    use_stage2 = {use_stage2}")
    if not use_stage2:
        log("    stage 2 did not improve AP; falling back to stage-1 probabilities")
        p2 = p1
        s2_metrics = dict(s1_metrics)

    # ==================================================================== phase E ====
    log("--- calibration (per country, isotonic)")
    pair_country = entity_country[q_global]
    cal = GroupCalibrator(min_group=20_000)
    # fit on the TRAIN pairs only, so the validation score stays honest
    cal.fit(p2[tr], y[tr].astype(float), pair_country[tr])
    p_cal = cal.transform(p2, pair_country)
    rel = cal.reliability(p2[va], y[va].astype(float), pair_country[va])
    for k, v in rel.items():
        log(f"    {k:>10}: n={v['n']:>9,} ECE={v['ece']:.4f} brier={v['brier']:.4f} "
            f"own_curve={v['calibrated_in_group']}")

    # ==================================================================== phase F ====
    log("--- decision layer")
    # rebuild per-entity group bounds over the pooled, q_global-sorted arrays
    order = np.argsort(q_global, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    qs = q_global[order]
    starts = np.searchsorted(qs, np.arange(n_entities), side="left").astype(np.int64)
    ends = np.searchsorted(qs, np.arange(n_entities), side="right").astype(np.int64)

    cand_sorted = Candidates(
        q_row=qs.astype(np.int32), c_row=c_global[order].astype(np.int32),
        score=X[order][:, 0], probe=np.zeros(len(order), dtype=np.uint8),
        rank=np.zeros(len(order), dtype=np.int32), starts=starts, ends=ends,
        labels=y[order])
    p_sorted = p_cal[order]

    counts = ends - starts
    p_sum = np.zeros(n_entities)
    np.add.at(p_sum, qs, p_sorted)
    overall_recall = float(np.mean([p["blocking_recall"]["macro_recall"]
                                    for p in pair_parts]))

    # Missing-mass estimate. Fitted on TRAIN entities only; see
    # trice.pipeline.MissingMassEstimator for why this is the highest-leverage correction.
    mm = MissingMassEstimator().fit(counts[~val_ent], p_sum[~val_ent],
                                    truth_sizes[~val_ent])
    miss_fitted = mm.missing_mass(counts, p_sum)
    miss_heuristic = estimate_missing_mass(counts, p_sum, overall_recall)
    exp_T = mm.expected_truth(counts, p_sum)
    log(f"    missing mass: fitted mean={miss_fitted[val_ent].mean():.3f} "
        f"heuristic mean={miss_heuristic[val_ent].mean():.3f}")
    log(f"    E[|T|] estimate vs actual on val: "
        f"{exp_T[val_ent].mean():.3f} vs {truth_sizes[val_ent].mean():.3f} "
        f"(Sum p = {p_sum[val_ent].mean():.3f})")
    miss = miss_fitted

    val_mask_ent = val_ent
    results = {}

    def run_rule(name: str, cfg: DecisionConfig, use_miss: bool = True,
                 repair: bool = True,
                 miss_override: np.ndarray | None = None) -> dict:
        mm_arg = miss_override if miss_override is not None else (miss if use_miss else None)
        mask, k_star, exp_f = __import__("trice.decide", fromlist=["decide_groups"]) \
            .decide_groups(p_sorted, starts, ends, cfg, missing_mass=mm_arg)
        if repair:
            mask = repair_disjointness(cand_sorted.q_row, cand_sorted.c_row, mask,
                                       p_sorted)
        per = score_flat(mask, cand_sorted.labels, starts, ends, truth_sizes)
        sel_n = np.zeros(n_entities, dtype=np.int32)
        np.add.at(sel_n, qs[mask], 1)
        tp_n = np.zeros(n_entities, dtype=np.int32)
        np.add.at(tp_n, qs[mask & (cand_sorted.labels == 1)], 1)
        prec = np.where(sel_n > 0, tp_n / np.maximum(sel_n, 1),
                        np.where(truth_sizes == 0, 1.0, 0.0))
        recl = np.where(truth_sizes > 0, tp_n / np.maximum(truth_sizes, 1),
                        np.where(sel_n == 0, 1.0, 0.0))
        v = val_mask_ent
        out = {
            "rule": name,
            "val_macro_f05": float(per[v].mean()),
            "val_macro_precision": float(prec[v].mean()),
            "val_macro_recall": float(recl[v].mean()),
            "train_macro_f05": float(per[~v].mean()) if (~v).any() else None,
            "mean_k": float(sel_n[v].mean()),
            "exact": int((tp_n[v] == truth_sizes[v]) & (sel_n[v] == truth_sizes[v])).sum()
            if False else int(((tp_n == truth_sizes) & (sel_n == truth_sizes))[v].sum()),
        }
        results[name] = out
        log(f"    {name:<28} val macro F0.5={out['val_macro_f05']:.5f} "
            f"P={out['val_macro_precision']:.4f} R={out['val_macro_recall']:.4f} "
            f"mean_k={out['mean_k']:.2f}")
        return out

    base_dec = DecisionConfig(beta=0.5, rule="expected_f", prune_epsilon=0.01,
                              max_emit=args.max_emit)
    run_rule("expected_f", base_dec)
    run_rule("expected_f_no_miss", base_dec, use_miss=False)
    run_rule("expected_f_heuristic_miss", base_dec, miss_override=miss_heuristic)
    run_rule("expected_f_no_repair", base_dec, repair=False)

    # A single scalar on the fitted missing mass, tuned on validation. One degree of
    # freedom against ~35 k entities, so the overfitting risk is negligible.
    #
    # NOTE: the candidate set includes scale 0, i.e. *disabling* the term. On this dataset
    # that is what wins -- E[|T|] is already well estimated by the retrieved probabilities
    # alone (3.462 predicted vs 3.462 actual), so the extra mass only pushes the rule to
    # over-emit, and under beta = 0.5 the precision loss exceeds the recall gain.
    for scale in (0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
        run_rule(f"expected_f_miss_x{scale}", base_dec,
                 miss_override=mm.missing_mass(counts, p_sum, scale))

    variants = {
        "off": None,
        "heuristic": miss_heuristic,
        **{f"x{s}": mm.missing_mass(counts, p_sum, s)
           for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)},
    }
    variant_scores = {
        "off": results["expected_f_no_miss"]["val_macro_f05"],
        "heuristic": results["expected_f_heuristic_miss"]["val_macro_f05"],
        "x1.0": results["expected_f"]["val_macro_f05"],
        **{f"x{s}": results[f"expected_f_miss_x{s}"]["val_macro_f05"]
           for s in (0.5, 0.75, 1.25, 1.5, 2.0, 3.0)},
    }
    best_variant = max(variant_scores, key=lambda k: variant_scores[k])
    best_scale = 0.0 if best_variant == "off" else (
        -1.0 if best_variant == "heuristic" else float(best_variant[1:]))
    log(f"    missing-mass variants: "
        + ", ".join(f"{k}={v:.5f}" for k, v in sorted(variant_scores.items(),
                                                      key=lambda t: -t[1])))
    log(f"    best = {best_variant} ({variant_scores[best_variant]:.5f})")
    miss = variants[best_variant]
    chosen_key = {"off": "expected_f_no_miss",
                  "heuristic": "expected_f_heuristic_miss",
                  "x1.0": "expected_f"}.get(best_variant,
                                            f"expected_f_miss_{best_variant}")
    results["expected_f"] = dict(results[chosen_key], rule="expected_f")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        run_rule(f"threshold@{thr}", DecisionConfig(rule="global_threshold",
                                                   global_threshold=thr,
                                                   max_emit=args.max_emit))
    run_rule("top1", DecisionConfig(rule="top1", global_threshold=0.5))
    for k in (2, 3, 4):
        run_rule(f"top{k}", DecisionConfig(rule="topk", fixed_k=k))

    # ablation: stage-1 probabilities only (no graph stage), same decision rule
    log("--- ablation: stage-1 only (no graph features)")
    cal1 = GroupCalibrator(min_group=20_000)
    cal1.fit(p1[tr], y[tr].astype(float), pair_country[tr])
    p1_cal_sorted = cal1.transform(p1, pair_country)[order]
    from trice.decide import decide_groups as _dg
    m_s1, _, _ = _dg(p1_cal_sorted, starts, ends, base_dec, missing_mass=miss)
    m_s1 = repair_disjointness(cand_sorted.q_row, cand_sorted.c_row, m_s1, p1_cal_sorted)
    per_s1 = score_flat(m_s1, cand_sorted.labels, starts, ends, truth_sizes)
    results["stage1_only_expected_f"] = {
        "rule": "stage1_only_expected_f",
        "val_macro_f05": float(per_s1[val_mask_ent].mean()),
    }
    log(f"    stage1_only expected_f  val macro F0.5="
        f"{per_s1[val_mask_ent].mean():.5f}")

    best_thr = max((k for k in results if k.startswith("threshold@")),
                   key=lambda k: results[k]["val_macro_f05"])
    log("")
    log("=" * 74)
    log(f"HEADLINE  expected_f = {results['expected_f']['val_macro_f05']:.5f}")
    log(f"          best fixed threshold ({best_thr}) = "
        f"{results[best_thr]['val_macro_f05']:.5f}")
    log(f"          gain from decision layer = "
        f"{results['expected_f']['val_macro_f05'] - results[best_thr]['val_macro_f05']:+.5f}")
    log(f"          gain from graph stage    = "
        f"{results['expected_f']['val_macro_f05'] - results['stage1_only_expected_f']['val_macro_f05']:+.5f}")
    log("=" * 74)

    # ==================================================================== phase G ====
    log("--- persisting")
    save_bundle(os.path.join(outdir, "model.pkl"),
                stage1=m1, stage2=m2, calibrator=cal, calibrator_stage1=cal1,
                feature_names=list(FEATURE_NAMES),
                graph_feature_names=list(GRAPH_FEATURE_NAMES),
                stage2_feature_names=all_names,
                use_stage2=use_stage2,
                blocking=asdict(block_cfg), model_cfg=asdict(model_cfg),
                graph_cfg=asdict(graph_cfg), decision=asdict(base_dec),
                overall_blocking_recall=overall_recall,
                missing_mass_model=mm, missing_mass_scale=best_scale)

    imp1 = m1.importances()
    imp2 = m2.importances()
    # index-slice rather than boolean-mask: X[va] would copy 1.6 M rows
    sep_idx = np.flatnonzero(va)[:400_000]
    sep = feature_separation(X[sep_idx], y[sep_idx], list(FEATURE_NAMES))

    metrics = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "countries": countries,
        "n_entities": int(n_entities),
        "n_pairs": int(len(X)),
        "n_val_entities": int(val_ent.sum()),
        "blocking": {c: p["blocking_recall"] for c, p in
                     zip([p["country"] for p in pair_parts], pair_parts)},
        "blocking_stats": block_stats_all,
        "blocking_recall_overall": overall_recall,
        "stage1": s1_metrics, "stage2": s2_metrics, "use_stage2": use_stage2,
        "stage1_parameters": m1.n_parameters(), "stage2_parameters": m2.n_parameters(),
        "model_kind": m1.kind,
        "calibration": rel,
        "decision_rules": results,
        "best_threshold_rule": best_thr,
        "missing_mass": {
            "scale": best_scale,
            "fitted_mean_val": float(miss_fitted[val_ent].mean()),
            "heuristic_mean_val": float(miss_heuristic[val_ent].mean()),
            "expected_truth_mean_val": float(exp_T[val_ent].mean()),
            "actual_truth_mean_val": float(truth_sizes[val_ent].mean()),
            "p_sum_mean_val": float(p_sum[val_ent].mean()),
        },
        "importances_stage1": imp1[:40],
        "importances_stage2": imp2[:40],
        "feature_separation": sep[:40],
        "truth_size_hist": {int(k): int(v) for k, v in
                            zip(*np.unique(truth_sizes, return_counts=True))},
    }
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, default=float)

    # validation arrays so the UI's decision tuner and entity explorer work offline
    val_pair = pair_is_val[order]
    np.savez_compressed(
        os.path.join(outdir, "val_pairs.npz"),
        q=qs[val_pair].astype(np.int32),
        c=cand_sorted.c_row[val_pair],
        cand_src=cand_src[order][val_pair],
        cand_num=cand_num[order][val_pair],
        p_cal=p_sorted[val_pair].astype(np.float32),
        p_stage1=p1[order][val_pair].astype(np.float32),
        label=cand_sorted.labels[val_pair],
        block_score=cand_sorted.score[val_pair],
        entity_ids=entity_ids, entity_country=entity_country.astype(str),
        truth_sizes=truth_sizes, val_entity=val_ent,
        missing_mass=miss.astype(np.float32),
        missing_mass_heuristic=miss_heuristic.astype(np.float32),
        expected_truth=exp_T.astype(np.float32),
    )
    log(f"wrote {outdir}")
    log(f"TOTAL {time.time() - T0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Persist the full traceback so it can be inspected even if the notebook truncated
        # the streamed output (`cat artifacts/last_error.txt`).
        import traceback
        tb = traceback.format_exc()
        try:
            os.makedirs(ART_FALLBACK := os.environ.get(
                "TRICE_ARTIFACTS_DIR", os.path.join(ROOT, "artifacts")), exist_ok=True)
            with open(os.path.join(ART_FALLBACK, "last_error.txt"), "w",
                      encoding="utf-8") as _fh:
                _fh.write(tb)
        except Exception:
            pass
        print(tb, flush=True)
        raise
