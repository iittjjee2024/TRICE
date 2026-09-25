"""
Tune the decision layer against a finished run, without re-training.

The decision stage consumes only stored calibrated probabilities, so its configuration can
be searched exhaustively in seconds rather than re-running the ~30 minute training. The
winning configuration is written back into the run's model bundle and metrics so that
``scripts/06_infer.py`` picks it up automatically.

Searched: missing-mass variant (including *off*), probability power gamma, prune epsilon,
max emit, and the disjointness repair toggle. All on the held-out validation entities.

Usage:
    python scripts/07_tune_decision.py --run-id main
    python scripts/07_tune_decision.py --run-id main --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.decide import DecisionConfig, decide_groups          # noqa: E402
from trice.evaluate import score_flat                           # noqa: E402
from trice.graph import repair_disjointness                     # noqa: E402
from trice.model import load_bundle, save_bundle                # noqa: E402
from trice.paths import artifacts_dir                           # noqa: E402

RUNS = os.path.join(artifacts_dir(ROOT), "runs")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--sample", type=int, default=0,
                    help="validation entities to use (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rundir = os.path.join(RUNS, args.run_id)
    npz_path = os.path.join(rundir, "val_pairs.npz")
    if not os.path.isfile(npz_path):
        raise SystemExit(f"{npz_path} not found")

    with np.load(npz_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}

    val_entity = d["val_entity"]
    val_ids = np.flatnonzero(val_entity).astype(np.int64)
    remap = np.full(len(d["entity_ids"]), -1, dtype=np.int64)
    remap[val_ids] = np.arange(len(val_ids))

    q = remap[d["q"].astype(np.int64)]
    order = np.argsort(q, kind="stable")
    q = q[order]
    c = d["c"].astype(np.int64)[order]
    p = d["p_cal"][order].astype(np.float64)
    label = d["label"][order]

    n_ent = len(val_ids)
    idx = np.arange(n_ent)
    starts = np.searchsorted(q, idx, side="left").astype(np.int64)
    ends = np.searchsorted(q, idx, side="right").astype(np.int64)
    truth = d["truth_sizes"][val_ids]
    countries = d["entity_country"].astype(str)[val_ids]
    counts = ends - starts

    p_sum = np.zeros(n_ent)
    np.add.at(p_sum, q, p)

    miss_fitted = d["missing_mass"][val_ids].astype(np.float64)
    miss_heur = (d["missing_mass_heuristic"][val_ids].astype(np.float64)
                 if "missing_mass_heuristic" in d else None)

    if args.sample and args.sample < n_ent:
        rng = np.random.default_rng(7)
        keep = np.sort(rng.choice(n_ent, size=args.sample, replace=False))
        log(f"using a {len(keep):,}-entity sample of {n_ent:,}")
    else:
        keep = None

    log(f"validation entities = {n_ent:,}, pairs = {len(p):,}, "
        f"mean |T| = {truth.mean():.3f}, Sum p = {p_sum.mean():.3f}")

    bundle = load_bundle(os.path.join(rundir, "model.pkl"))
    mm = bundle.get("missing_mass_model")

    miss_variants: dict[str, np.ndarray | None] = {"off": None}
    if miss_heur is not None:
        miss_variants["heuristic"] = miss_heur
    if mm is not None:
        for s in (0.25, 0.5, 1.0, 1.5):
            miss_variants[f"fitted_x{s}"] = mm.missing_mass(counts, p_sum, s)
    else:
        miss_variants["fitted"] = miss_fitted

    def evaluate(cfg: DecisionConfig, miss: np.ndarray | None,
                 repair: bool) -> tuple[float, float, float, float]:
        mask, k_star, _ = decide_groups(p, starts, ends, cfg, missing_mass=miss)
        if repair:
            mask = repair_disjointness(q, c, mask, p)
        per = score_flat(mask, label, starts, ends, truth)
        sel = np.zeros(n_ent, dtype=np.int64)
        np.add.at(sel, q[mask], 1)
        tp = np.zeros(n_ent, dtype=np.int64)
        np.add.at(tp, q[mask & (label == 1)], 1)
        prec = np.where(sel > 0, tp / np.maximum(sel, 1),
                        np.where(truth == 0, 1.0, 0.0))
        rec = np.where(truth > 0, tp / np.maximum(truth, 1),
                       np.where(sel == 0, 1.0, 0.0))
        s = keep if keep is not None else slice(None)
        return (float(per[s].mean()), float(prec[s].mean()),
                float(rec[s].mean()), float(sel[s].mean()))

    grid = list(itertools.product(
        list(miss_variants),                       # missing-mass variant
        (0.85, 1.0, 1.15, 1.3),                    # probability power gamma
        (0.005, 0.01, 0.03),                       # prune epsilon
        (True, False),                             # disjointness repair
    ))
    log(f"evaluating {len(grid)} configurations")

    rows = []
    t0 = time.time()
    for i, (mv, gamma, eps, repair) in enumerate(grid, 1):
        cfg = DecisionConfig(beta=0.5, rule="expected_f", prune_epsilon=eps,
                             max_emit=25, probability_power=gamma)
        f, pr, rc, mk = evaluate(cfg, miss_variants[mv], repair)
        rows.append({"missing_mass": mv, "gamma": gamma, "prune_epsilon": eps,
                     "repair": repair, "macro_f05": f, "precision": pr,
                     "recall": rc, "mean_k": mk})
        if i % 8 == 0 or i == len(grid):
            log(f"  {i}/{len(grid)}  ({time.time() - t0:.0f}s elapsed)")

    rows.sort(key=lambda r: -r["macro_f05"])
    print()
    print(f"{'miss':<14}{'gamma':>7}{'eps':>7}{'repair':>8}"
          f"{'F0.5':>10}{'P':>9}{'R':>9}{'mean|S|':>9}")
    for r in rows[:15]:
        print(f"{r['missing_mass']:<14}{r['gamma']:>7.2f}{r['prune_epsilon']:>7.3f}"
              f"{str(r['repair']):>8}{r['macro_f05']:>10.5f}{r['precision']:>9.4f}"
              f"{r['recall']:>9.4f}{r['mean_k']:>9.3f}")

    best = rows[0]
    print()
    log(f"BEST  F0.5={best['macro_f05']:.5f}  "
        f"miss={best['missing_mass']} gamma={best['gamma']} "
        f"eps={best['prune_epsilon']} repair={best['repair']}")

    # per-country breakdown for the winner
    cfg = DecisionConfig(beta=0.5, rule="expected_f",
                         prune_epsilon=best["prune_epsilon"], max_emit=25,
                         probability_power=best["gamma"])
    mask, _, _ = decide_groups(p, starts, ends, cfg,
                               missing_mass=miss_variants[best["missing_mass"]])
    if best["repair"]:
        mask = repair_disjointness(q, c, mask, p)
    per = score_flat(mask, label, starts, ends, truth)
    print()
    print(f"{'country':<10}{'n':>9}{'F0.5':>10}")
    by_country = {}
    for ctry in sorted(set(countries.tolist())):
        m = countries == ctry
        by_country[ctry] = {"n": int(m.sum()), "macro_f05": float(per[m].mean())}
        print(f"{ctry:<10}{int(m.sum()):>9,}{per[m].mean():>10.5f}")

    if args.dry_run:
        log("dry run: bundle not modified")
        return

    bundle["decision"] = {
        **bundle.get("decision", {}),
        "beta": 0.5, "rule": "expected_f",
        "prune_epsilon": best["prune_epsilon"],
        "max_emit": 25,
        "probability_power": best["gamma"],
    }
    bundle["missing_mass_variant"] = best["missing_mass"]
    bundle["missing_mass_scale"] = (
        0.0 if best["missing_mass"] == "off"
        else (-1.0 if best["missing_mass"] == "heuristic"
              else float(best["missing_mass"].split("_x")[-1])))
    bundle["disjointness_repair"] = bool(best["repair"])
    bundle["tuned_val_macro_f05"] = best["macro_f05"]
    save_bundle(os.path.join(rundir, "model.pkl"), **bundle)
    log("wrote tuned decision configuration into model.pkl")

    mpath = os.path.join(rundir, "metrics.json")
    with open(mpath, encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["decision_tuning"] = {
        "best": best, "by_country": by_country,
        "grid_size": len(grid), "top": rows[:15],
    }
    metrics.setdefault("decision_rules", {})["expected_f"] = {
        "rule": "expected_f",
        "val_macro_f05": best["macro_f05"],
        "val_macro_precision": best["precision"],
        "val_macro_recall": best["recall"],
        "mean_k": best["mean_k"],
    }
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, default=float)
    log(f"updated {mpath}")


if __name__ == "__main__":
    main()
