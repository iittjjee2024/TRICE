"""
Assemble the final submission archive.

Produces ``Vortex_submission.zip`` in the layout the challenge requires::

    Vortex_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       │   ├── trice/          the pipeline library
    │       │   └── scripts/        ordered stages + tests
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Refuses to build if the submission files are missing or fail format validation, so a broken
archive cannot be produced by accident.

Usage:
    python scripts/08_package.py
    python scripts/08_package.py --skip-validation      # not recommended
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trice.export import read_entity_ids, validate_submission  # noqa: E402
from trice.paths import dataset_dir, output_dir               # noqa: E402

TEAM = "Vortex"
OUTPUT = output_dir(ROOT)
TEST_DIR = os.path.join(dataset_dir(ROOT), "test")

# scripts that belong in the reproducible package (the workbench is excluded)
PIPELINE_SCRIPTS = [
    "02_mine_variants.py",
    "03_prepare.py",
    "04_blocking_eval.py",
    "05_train.py",
    "06_infer.py",
    "07_tune_decision.py",
    "test_normalize.py",
    "test_decide.py",
]

CODE_README = """# Business Entity Resolution — team Vortex

Self-contained pipeline. `src/trice/` is a plain Python library with no web dependencies.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\\Scripts\\pip
```

Python 3.11. Peak memory ~10 GB; ~25 GB of free disk is used for the record store and the
inference feature spill.

## Expected data layout

Relative to this folder's parent (or pass explicit paths):

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## Reproduce both output files

```bash
# 1. mine token aliases from the training ground truth        (~2 min)
python src/scripts/02_mine_variants.py --sample 250000

# 2. normalise all 24 M records into a Parquet record store   (~4 min, 13 workers)
python src/scripts/03_prepare.py --workers 13

# 3. train the matcher and score a held-out split             (~29 min)
python src/scripts/05_train.py --entities 70000 --run-id main

# 4. search the decision-layer configuration on validation    (~10 min)
python src/scripts/07_tune_decision.py --run-id main

# 5. full test-set inference -> output/                       (~110 min)
python src/scripts/06_infer.py --run-id main
```

Step 5 writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the
format validator itself.

## Correctness tests

```bash
python src/scripts/test_normalize.py    # normalisation against real match groups
python src/scripts/test_decide.py       # metric algebra + top-k optimality vs brute force
```

`test_decide.py` is the important one: it proves the closed-form metric matches the problem
statement's precision/recall formula and that the top-k argmax equals the true argmax over all
2^n subsets.

## Stage map

| stage | module | what it does |
|---|---|---|
| normalise | `trice/normalize.py` | unicode folding, consonant skeleton (cross-script), DBA split, domain de-concatenation, legal-suffix separation, address component bag |
| mine | `trice/mine.py` | token aliases learned from ground-truth matched pairs |
| store | `trice/records.py` | streaming TSV → Parquet, memory-bounded |
| block | `trice/blocking.py` | two IDF-weighted sparse channels with per-namespace df caps |
| features | `trice/features.py` | 55 pairwise features |
| model | `trice/model.py` | LightGBM/HistGB + per-country isotonic calibration |
| graph | `trice/graph.py` | column-softmax competition with dustbin, S2↔S3 corroboration |
| decide | `trice/decide.py` | Poisson-binomial DP, exact expected-F_0.5 set selection |
| evaluate | `trice/evaluate.py` | macro F_beta, blocking recall |
| export | `trice/export.py` | TSV writers + format validator |

## Fair play

No external database, API, geocoder or internet corpus is consulted. All corpus statistics
(IDF weights, token aliases, calibration curves) are derived from the provided train/test TSVs.
Models are gradient-boosted trees (LightGBM, MIT; scikit-learn HistGradientBoosting, BSD-3)
totalling ~80,000 parameters — far inside the 8-billion / MIT-Apache constraint. No pretrained
weights are downloaded.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=TEAM)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    matching = os.path.join(OUTPUT, "matching_results.tsv")
    candidates = os.path.join(OUTPUT, "candidate_pairs.tsv")

    for p in (matching, candidates):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} — run scripts/06_infer.py first")

    if not args.skip_validation:
        print("validating submission files…")
        required = set(read_entity_ids(os.path.join(TEST_DIR, "test_source1.tsv")))
        report = validate_submission(matching, candidates, required, None)
        for w in report["warnings"]:
            print(f"  WARNING: {w}")
        if not report["ok"]:
            for i, issue in enumerate(report["issues"], 1):
                print(f"  {i}. {issue}")
            raise SystemExit("validation FAILED — refusing to package")
        m = report["matching"]
        print(f"  PASS — {m['rows']:,} rows, {m['nonempty']:,} non-empty, "
              f"{m['total_ids']:,} ids, mean {m['mean_ids_per_nonempty']:.3f} per non-empty")

    zip_path = args.out or os.path.join(ROOT, f"{args.team}_submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    base = "code/business_engine" if False else "code/business_entity_resolution"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # ---- output ----
        z.write(matching, "output/matching_results.tsv")
        z.write(candidates, "output/candidate_pairs.tsv")
        n += 2

        # ---- library ----
        trice_dir = os.path.join(ROOT, "src", "trice")
        for fname in sorted(os.listdir(trice_dir)):
            if not fname.endswith(".py"):
                continue
            z.write(os.path.join(trice_dir, fname), f"{base}/src/trice/{fname}")
            n += 1
        # make it importable as a package even without an __init__
        if not os.path.isfile(os.path.join(trice_dir, "__init__.py")):
            z.writestr(f"{base}/src/trice/__init__.py",
                       '"""TRICE — business entity resolution pipeline."""\n')
            n += 1

        # ---- scripts ----
        for fname in PIPELINE_SCRIPTS:
            src = os.path.join(ROOT, "scripts", fname)
            if os.path.isfile(src):
                z.write(src, f"{base}/src/scripts/{fname}")
                n += 1

        # ---- metadata ----
        z.writestr(f"{base}/README.md", CODE_README)
        z.write(os.path.join(ROOT, "backend", "requirements.txt"),
                f"{base}/requirements.txt")
        n += 2

        doc = os.path.join(ROOT, "Documentation_template.md")
        if not os.path.isfile(doc):
            raise SystemExit("Documentation_template.md missing")
        z.write(doc, "Documentation_template.md")
        n += 1

        # design docs are useful context for the reviewers
        docs_dir = os.path.join(ROOT, "docs")
        if os.path.isdir(docs_dir):
            for fname in sorted(os.listdir(docs_dir)):
                if fname.endswith(".md"):
                    z.write(os.path.join(docs_dir, fname), f"{base}/docs/{fname}")
                    n += 1

    size = os.path.getsize(zip_path) / 1e6
    print(f"\nwrote {zip_path}")
    print(f"  {n} entries, {size:,.1f} MB")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            print(f"    {info.filename:<62}{info.file_size / 1e3:>12,.1f} KB")


if __name__ == "__main__":
    main()
