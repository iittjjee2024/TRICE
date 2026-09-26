# TRICE — Business Entity Resolution (Amazon ML Challenge 2026)

**Team Vortex** · Aryan Bansal (lead) · Sachin Kumar · Daksh Tandon · Parth Aggarwal

Match every Source-1 business record to the set of Source-2 / Source-3 records describing
the same real-world business — across **1.73 M** test entities and **~10 M** candidate
records — scored by macro-averaged **F<sub>0.5</sub>**.

> **Leaderboard: macro F<sub>0.5</sub> = 0.887** on the full 1,732,544-entity test set.
> **Held-out validation: macro F<sub>0.5</sub> = 0.917** (precision 0.955, recall 0.845).
> Ships a runnable pipeline **and** a FastAPI + React analysis workbench.

## Results

| Metric | Value |
|---|---|
| **Leaderboard macro F<sub>0.5</sub>** (full test set) | **0.887** |
| Validation macro F<sub>0.5</sub> (held-out) | 0.917 |
| Validation precision / recall | 0.955 / 0.845 |
| Test entities scored | 1,732,544 (1,603,183 with matches, 129,361 singletons) |
| Blocking macro-recall ceiling | ≈ 0.88 |
| Model | LightGBM GBDT, two stacked stages, ≈ 8 × 10<sup>4</sup> parameters |

For reference, a `top-1` baseline scores ≈ 0.67 and a fixed `top-3` ≈ 0.75 on the same
validation split; the exact expected-F<sub>0.5</sub> decision layer plus the graph
competition/corroboration features are what carry it to 0.917. The small validation →
leaderboard gap (0.917 → 0.887) is expected generalization on unseen test data; the
leaderboard run additionally used the earlier-epoch model for the US partition (France and
India used the full-data model), so a fully-consistent full-data inference pass is the most
likely lever for a further gain. Recall — bounded by the ≈ 0.88 blocking ceiling — is now
the binding constraint, not precision.

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Repository layout](#2-repository-layout)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — get the code](#4-step-1--get-the-code)
5. [Step 2 — upload the dataset](#5-step-2--upload-the-dataset)
6. [Step 3 — Python environment](#6-step-3--python-environment)
7. [Step 4 — train and test (the pipeline)](#7-step-4--train-and-test-the-pipeline)
8. [Step 5 — the submittable TSV](#8-step-5--the-submittable-tsv)
9. [Step 6 — the analysis workbench (optional)](#9-step-6--the-analysis-workbench-optional)
10. [Approach in brief](#10-approach-in-brief)
11. [Runtime & hardware](#11-runtime--hardware)
12. [Troubleshooting](#12-troubleshooting)
13. [Fair play & licensing](#13-fair-play--licensing)

---

## 1. What you get

- **`output/matching_results.tsv`** — the file you upload to the leaderboard. One row per
  Source-1 entity: `source1_entity_id <TAB> matched_entity_ids` (comma-joined S2/S3 ids,
  empty for a no-match).
- **`output/candidate_pairs.tsv`** — the blocking candidate set the model scored (required in
  the submission zip).
- A **reproducible pipeline** (`src/trice/` + `scripts/`) that regenerates both from the raw
  data.
- A **workbench** (`backend/` FastAPI + `frontend/` React) for inspecting blocking recall,
  feature importance, per-country calibration, the decision layer, and individual entities.
- A one-command **submission packager** (`scripts/08_package.py`).

---

## 2. Repository layout

```
Amazon ML Challenge/
├── README.md                     ← this file
├── Documentation_template.md     filled-in methodology (the challenge write-up)
├── docs/                         design + analysis write-ups (01–05)
├── src/trice/                    the pipeline library (no web dependencies)
│   ├── normalize.py  mine.py  records.py  blocking.py  features.py
│   ├── model.py  graph.py  decide.py  evaluate.py  export.py  pipeline.py
├── scripts/                      ordered pipeline stages + correctness tests
│   ├── 02_mine_variants.py  03_prepare.py  04_blocking_eval.py
│   ├── 05_train.py  06_infer.py  07_tune_decision.py  08_package.py
│   └── test_normalize.py  test_decide.py  test_union.py
├── backend/                      FastAPI service + pinned requirements.txt
├── frontend/                     React + Vite workbench UI
├── student_resource/             ← YOU place the dataset here (see step 2)
│   ├── dataset/                    (git-ignored; not in the repo)
│   └── utils/validate_submission.py   official validator (provided)
├── artifacts/                    generated: record store, runs, models (git-ignored)
└── output/                       matching_results.tsv, candidate_pairs.tsv
```

Note: `src/trice/` imports nothing from FastAPI, so it drops straight into the challenge
submission zip under `code/business_entity_resolution/src/`.

---

## Run on Kaggle (no local setup)

Prefer to train/test in the cloud? Use [`kaggle/trice_kaggle_runner.ipynb`](kaggle/trice_kaggle_runner.ipynb).
It clones this repo, auto-detects a dataset you attach as a Kaggle *Input*, runs the full
pipeline, and writes `matching_results.tsv` to the notebook output. Setup steps are in
[`kaggle/README.md`](kaggle/README.md).

---

## 3. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.10–3.12 work; the venv keeps it isolated |
| Node.js | ≥ 18 (tested on 24) | only for the optional workbench UI |
| RAM | 16 GB | pipeline is memory-bound; see [§11](#11-runtime--hardware) |
| Free disk | ~25 GB | record store + on-disk feature spill during inference |
| Git | any | to clone / push |

Windows PowerShell commands are shown below; macOS/Linux equivalents are in comments.

---

## 4. Step 1 — get the code

```powershell
git clone https://github.com/<your-org>/trice-entity-resolution.git
cd trice-entity-resolution
```

Pushing this repo to GitHub for the first time:

```powershell
git init
git add .                       # the dataset is git-ignored, so this is code only
git commit -m "TRICE business entity resolution pipeline + workbench"
git branch -M main
git remote add origin https://github.com/<your-org>/trice-entity-resolution.git
git push -u origin main
```

The dataset (~2.3 GB) and all generated artifacts are excluded by `.gitignore`, so the repo
stays small. Collaborators download the data themselves (next step).

---

## 5. Step 2 — upload the dataset

The challenge dataset is **not** committed to the repo. Place the provided files exactly
here (the folder names and file names must match):

```
student_resource/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv          Source-1 training records (deduplicated reference)
    │   ├── train_source2.tsv          Source-2 training records
    │   ├── train_source3.tsv          Source-3 training records
    │   └── train_ground_truth.tsv     match labels for the training set
    └── test/
        ├── test_source1.tsv           Source-1 test records (predict for every one)
        ├── test_source2.tsv           Source-2 test records
        └── test_source3.tsv           Source-3 test records
```

If you have the challenge zip, unzip it so the layout matches. To verify:

```powershell
Get-ChildItem -Recurse student_resource\dataset | Select-Object FullName, Length
```

Expected row counts (sanity check): train S1 = 2,206,821 · test S1 = 1,732,544. Every file
is **tab-separated**; do not re-save them as CSV.

---

## 6. Step 3 — Python environment

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

```bash
# macOS / Linux
python3 -m venv backend/.venv
backend/.venv/bin/pip install --upgrade pip
backend/.venv/bin/pip install -r backend/requirements.txt
```

Verify the imports resolve:

```powershell
backend\.venv\Scripts\python.exe -c "import numpy, pandas, sklearn, scipy, rapidfuzz, lightgbm, pyarrow; print('ok')"
```

`PY=backend\.venv\Scripts\python.exe` is used as shorthand below.

---

## 7. Step 4 — train and test (the pipeline)

Run these in order. Each stage writes to `artifacts/` and prints progress; times are for the
reference 16-core / 16 GB machine.

```powershell
# 4a. mine token aliases from the training ground truth  ->  artifacts/variants.json   (~2 min)
$PY scripts\02_mine_variants.py --sample 250000

# 4b. normalise all 24 M records into a Parquet record store  ->  artifacts/store/     (~4 min)
$PY scripts\03_prepare.py --workers 13

# 4c. (optional) inspect blocking recall vs cost on a training sample
$PY scripts\04_blocking_eval.py --country US --queries 6000

# 4d. TRAIN the matcher and TEST it on a held-out validation split  ->  artifacts/runs/main   (~29 min)
$PY scripts\05_train.py --entities 70000 --run-id main

# 4e. search the decision-layer configuration on the validation split (fast, no re-training) (~10 min)
$PY scripts\07_tune_decision.py --run-id main
```

Step **4d** is the train/test step. It samples training entities per country, blocks, builds
features, trains a two-stage gradient-boosted matcher, calibrates per country, runs the
expected-F<sub>0.5</sub> decision layer, and **scores a held-out split it never trained on**.
It prints, and saves to `artifacts/runs/main/metrics.json`:

```
stage1 val AUC=0.99968 AP=0.99553
stage2 val AUC=0.99981 AP=0.99732
HEADLINE  expected_f = 0.91698     (precision 0.955, recall 0.845)
```

To simulate the zero-shot France setting locally, hold out a country you *do* have labels
for:

```powershell
$PY scripts\05_train.py --entities 60000 --run-id india_zeroshot --holdout-country India
```

Correctness tests (run any time — they need no dataset):

```powershell
$PY scripts\test_normalize.py   # normalisation vs real corrupted match groups
$PY scripts\test_decide.py      # metric algebra + top-k optimality vs brute-force 2^n search
$PY scripts\test_union.py       # chunked candidate union vs reference
```

---

## 8. Step 5 — the submittable TSV

Run full inference over the test set, then validate:

```powershell
# 5a. full test inference  ->  output/matching_results.tsv + output/candidate_pairs.tsv   (~90–110 min)
$PY scripts\06_infer.py --run-id main

# 5b. official validator (must print PASS)
cd student_resource
python utils\validate_submission.py `
    --matching ..\output\matching_results.tsv `
    --candidate ..\output\candidate_pairs.tsv `
    --test-dir dataset\test
cd ..
```

`06_infer.py` runs the validator internally and writes `artifacts/runs/main/inference.json`.
The produced `output/matching_results.tsv` looks exactly like the challenge portal example:

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
```

**Upload `output/matching_results.tsv`** to the leaderboard portal.

Build the full submission archive (`Vortex_submission.zip`, with `output/`, the runnable
`code/`, and the methodology doc) — it refuses to build unless the files pass validation:

```powershell
$PY scripts\08_package.py
```

---

## 9. Step 6 — the analysis workbench (optional)

A local, single-user tool for reading the results. **Binds to loopback and has no
authentication** — do not expose it on a public interface.

```powershell
# terminal 1 — API
$PY backend\run.py                       # http://127.0.0.1:8000/docs

# terminal 2 — UI
cd frontend
npm install
npm run dev                              # http://127.0.0.1:5173
```

| Screen | Answers |
|---|---|
| Dashboard | Which run scores best; the ablation table |
| Data | File/store status, country mix, live **normalisation playground**, mined aliases |
| Blocking | Recall ceiling per country, channel cost, F<sub>0.5</sub>-vs-recall curve |
| Model | Feature importance by family, single-feature separation, parameter count |
| Calibration | Per-country reliability diagram (watch the France panel) |
| Decision tuner | Live macro F<sub>0.5</sub> as decision parameters change |
| Entities | Per-entity drill-down filtered by error type, with the `E[F\|k]` curve |
| Submission | Write status, rule-by-rule validation, download |
| Docs | The design documents rendered in-app |

Production build of the UI: `npm run build` (outputs to `frontend/dist/`).

---

## 10. Approach in brief

Full derivation in [`docs/02_SOLUTION_IDEA.md`](docs/02_SOLUTION_IDEA.md); measured findings in
[`docs/05_EDA_FINDINGS.md`](docs/05_EDA_FINDINGS.md); methodology write-up in
[`Documentation_template.md`](Documentation_template.md).

1. **Normalisation** — unicode folding plus a *consonant-skeleton* transform that maps
   Devanagari and Latin spellings of a name onto one string
   (`राम मार्केटिंग प्राइवेट लिमिटेड` → `rmrktng` ← `ram marketing private limited`); DBA
   splitting, domain de-concatenation (`edwardshintzedougherty.com`), legal-suffix separation,
   order-free address component bags.
2. **Mined aliases** — `texas↔tx`, `mysore↔mysuru`, Devanagari state names, etc., learned from
   ground-truth matched pairs, not from any external gazetteer.
3. **Blocking** — two independent IDF-weighted sparse channels (name, address) with
   per-namespace document-frequency caps, unioned; ~0.88 macro recall ceiling.
4. **Matcher** — LightGBM gradient-boosted trees over ~55 pairwise features dominated by
   IDF-weighted rarity overlap.
5. **Graph stage** — the ground truth has zero Source-2/3 records with more than one parent
   across 7.6 M links, so disjointness is enforced via a column-softmax with a null dustbin,
   plus Source-2 ↔ Source-3 corroboration. This is the largest single win (+0.011).
6. **Decision layer** — because `F_0.5(S,T) = 1.25·|S∩T| / (0.25·|T| + |S|)`, the optimal
   emitted set is computed exactly: evaluate `E[F | top-k]` for every `k` with a
   Poisson-binomial DP and take the argmax. No threshold to tune; the singleton decision falls
   out at `k = 0`.

---

## 11. Runtime & hardware

| Stage | Time (16 cores) | Peak RAM | Disk |
|---|---|---|---|
| `02_mine_variants` | ~2 min | ~3 GB | — |
| `03_prepare` | ~4 min | ~5 GB | ~1.6 GB (store) |
| `05_train` | ~29 min | ~10 GB | small |
| `07_tune_decision` | ~10 min | ~3 GB | — |
| `06_infer` | ~90–110 min | ~11 GB | ~9 GB (temporary feature spill) |

Memory, not CPU, is the binding constraint. Everything is partitioned by country and streamed
in blocks to stay within 16 GB. Lower `--entities` or `--train-rows` in `05_train.py`, or
`--query-batch` in `06_infer.py`, if you have less.

---

## 12. Troubleshooting

- **`MemoryError` during inference** — lower `--query-batch` (e.g. `--query-batch 60000`) on
  `scripts/06_infer.py`. The candidate union and feature build are already chunked; a smaller
  batch reduces peak further.
- **`numpy.dtype size changed` on import** — a numpy/pandas ABI clash in a *global* Python.
  Always use the project venv (`backend\.venv`), never the system interpreter.
- **`variants.json missing`** — run `scripts/02_mine_variants.py` before `03_prepare.py`.
- **Validator says "not TAB-separated"** — the dataset was re-saved as CSV; re-extract the
  originals. Every file is `.tsv`.
- **UI can't reach the API** — start `backend\run.py` first; the Vite dev server proxies
  `/api` to `http://127.0.0.1:8000`.
- **Tailwind build error on `npm run build`** — pinned to `tailwindcss@4.1.18`; run
  `npm install` again if you changed versions.

---

## 13. Fair play & licensing

No external database, API, geocoder, or internet corpus is consulted. Every statistic — IDF
weights, token aliases, calibration curves, the |T| prior — is derived from the provided
`train_*.tsv` and `test_*.tsv`. Using the *test records themselves* for unsupervised corpus
statistics is transductive learning on given data, not external augmentation.

The models are gradient-boosted trees: **LightGBM** (MIT) when available, otherwise
scikit-learn's **HistGradientBoostingClassifier** (BSD-3-Clause). Total parameter count is
~8 × 10⁴ — far inside the challenge's 8-billion / MIT-Apache limit. No pretrained weights are
downloaded.
