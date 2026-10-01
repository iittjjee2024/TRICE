"""
Assemble output/ from the previously-generated TSVs in docs/ instead of rerunning the full
Kaggle inference.

* ``output/matching_results.tsv`` is ``docs/matching_results_combined.tsv`` verbatim --
  this is the only file the leaderboard scores, so it is copied unchanged.
* ``output/candidate_pairs.tsv`` is ``docs/candidate_pairs.tsv`` with a repair pass: any
  matched id that is missing from an entity's candidate list is appended, so the
  ``matches ⊆ candidates`` invariant the validator checks holds for every row. The two
  docs files came from different runs, which left 2091 entities violating it; the scored
  matches are untouched, only the (unscored) candidate lists grow.

Writes in test_source1.tsv order so every required entity appears exactly once, then runs
the same format validator scripts/08_package.py uses.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.export import (read_entity_ids, validate_submission,            # noqa: E402
                          write_candidate_pairs, write_matching_results)
from trice.paths import dataset_dir, output_dir                           # noqa: E402

DOCS = os.path.join(ROOT, "docs")
OUTPUT = output_dir(ROOT)
TEST_DIR = os.path.join(dataset_dir(ROOT), "test")

SRC_MATCHING = os.path.join(DOCS, "matching_results_combined.tsv")
SRC_CANDIDATES = os.path.join(DOCS, "candidate_pairs.tsv")


def read_map(path: str) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh)  # header
        for line in fh:
            s1, tab, rest = line.partition("\t")
            if not tab:
                continue
            rest = rest.rstrip("\n")
            m[s1] = rest.split(",") if rest else []
    return m


def main() -> None:
    for p in (SRC_MATCHING, SRC_CANDIDATES):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    print("reading docs/matching_results_combined.tsv …")
    matched = read_map(SRC_MATCHING)
    print(f"  {len(matched):,} rows")
    print("reading docs/candidate_pairs.tsv …")
    candidates = read_map(SRC_CANDIDATES)
    print(f"  {len(candidates):,} rows")

    required = read_entity_ids(os.path.join(TEST_DIR, "test_source1.tsv"))
    print(f"  {len(required):,} required Source-1 entities")

    # repair: candidate list must be a superset of the matched list for each entity
    repaired = 0
    added_ids = 0
    for s1, mids in matched.items():
        if not mids:
            continue
        clist = candidates.get(s1, [])
        cset = set(clist)
        extra = [i for i in mids if i not in cset]
        if extra:
            candidates[s1] = clist + extra
            repaired += 1
            added_ids += len(extra)
    print(f"  repaired {repaired:,} entities, appended {added_ids:,} matched ids "
          f"to candidate lists")

    os.makedirs(OUTPUT, exist_ok=True)
    mpath = os.path.join(OUTPUT, "matching_results.tsv")
    cpath = os.path.join(OUTPUT, "candidate_pairs.tsv")

    nr, nn = write_matching_results(
        mpath, required, (matched.get(e, []) for e in required))
    print(f"wrote {mpath}: {nr:,} rows, {nn:,} non-empty")
    cnr, cnn = write_candidate_pairs(
        cpath, required, (candidates.get(e, []) for e in required))
    print(f"wrote {cpath}: {cnr:,} rows, {cnn:,} non-empty")

    print("\nvalidating …")
    report = validate_submission(mpath, cpath, set(required), None)
    import json
    print(json.dumps(report, indent=1))
    if not report["ok"]:
        raise SystemExit("validation FAILED")
    if report["checks"].get("matches_subset_of_candidates") is False:
        raise SystemExit("repair did not fully close matches⊆candidates")
    print("\nOK — output/ ready; run scripts/08_package.py to build the zip")


if __name__ == "__main__":
    main()
