# Plan & system design — TRICE Workbench

The deliverable is a **FastAPI + React workbench** that runs the TRICE pipeline
described in [`02_SOLUTION_IDEA.md`](02_SOLUTION_IDEA.md) end to end, makes every
intermediate stage inspectable, and emits the two submission TSVs.

Why a workbench and not a script: the pipeline has five separable stages whose value is
only visible by *comparing* configurations on a validation split. The metric is
precision-heavy and macro-averaged, so the failure modes are concentrated in a small
number of individual entities. A tool that lets you find and read those entities is
worth more over a 3-day challenge than a faster training loop.

---

## 1. Repository layout

```
Amazon ML Challenge/
├── docs/
│   ├── 01_PROBLEM_STATEMENT.md
│   ├── 02_SOLUTION_IDEA.md
│   ├── 03_PLAN_AND_DESIGN.md         ← this file
│   └── 04_API_CONTRACT.md
├── backend/
│   ├── .venv/                        # isolated env (global env has a numpy/pandas ABI clash)
│   ├── requirements.txt              # pinned
│   ├── run.py                        # uvicorn entrypoint
│   └── app/
│       ├── main.py                   # FastAPI app, CORS, router mounting
│       ├── config.py                 # settings, paths
│       ├── schemas.py                # pydantic request/response models
│       ├── store.py                  # run registry + artifact persistence (SQLite + npz/parquet)
│       ├── jobs.py                   # background job runner, progress + log streaming
│       ├── routers/
│       │   ├── datasets.py
│       │   ├── runs.py
│       │   ├── analysis.py           # blocking / model / calibration / decision analytics
│       │   ├── pairs.py              # pair explorer + review queue
│       │   └── exports.py            # TSV writers + submission validator
│       └── er/
│           ├── textnorm.py           # unicode fold, abbreviation canon, tokenisation
│           ├── address.py            # address componentisation
│           ├── synth.py              # noise operators + synthetic dataset generator
│           ├── datasets.py           # TSV IO, splits, ground-truth parsing
│           ├── idf.py                # transductive IDF over train ∪ test
│           ├── blocking.py           # multi-probe candidate generation + metrics
│           ├── features.py           # pairwise feature extraction
│           ├── model.py              # stage-1 / stage-2 GBDT + calibration
│           ├── graph.py              # tripartite messages + competition normalisation
│           ├── decide.py             # Poisson-binomial DP, expected-F0.5 set selection
│           ├── evaluate.py           # macro F_0.5, per-entity breakdown
│           └── pipeline.py           # orchestration of the whole run
├── frontend/
│   ├── package.json
│   ├── vite.config.ts                # dev proxy /api → :8000
│   ├── tailwind.config.js
│   └── src/
│       ├── main.tsx, App.tsx
│       ├── api/client.ts             # typed fetch layer
│       ├── components/               # shared UI primitives
│       └── pages/                    # the nine screens in §5
├── data/                             # datasets live here (generated or dropped in)
├── artifacts/                        # per-run outputs, gitignored
└── output/                           # matching_results.tsv, candidate_pairs.tsv
```

`data/` mirrors the challenge layout exactly (`train/train_source1.tsv`, …,
`test/test_source1.tsv`, …) so a real challenge dataset can be dropped in without any
code change.

### 1.1 Mapping to the required submission package

| Required | Produced by |
|---|---|
| `output/matching_results.tsv` | `exports.py` → `write_matching_results` |
| `output/candidate_pairs.tsv` | `exports.py` → `write_candidate_pairs` (the **final** candidate set, post-prune) |
| `code/business_entity_resolution/src/` | `backend/app/er/` (self-contained, no web deps) |
| `code/business_entity_resolution/requirements.txt` | `backend/requirements.txt` |
| `Documentation_template.md` | generated from `docs/01`–`docs/03` + the live run report |

`app/er/` deliberately imports nothing from FastAPI or the routers, so it can be lifted
into the submission zip as a pure library with a thin CLI.

---

## 2. Data model

### 2.1 Core tables (in-memory, persisted per run)

**`records`** — all S1/S2/S3 records from a split, unified.

| field | type | note |
|---|---|---|
| `entity_id` | str | `S1-…`/`S2-…`/`S3-…` |
| `source` | int8 | derived from prefix — never a column in the input |
| `business_name`, `business_address`, `country` | str | raw |
| `name_norm`, `addr_norm` | str | normalised |
| `name_tokens`, `addr_tokens` | list[str] | |
| `postal`, `house_no`, `street`, `locality` | str | extracted components |

**`candidates`** — output of blocking, one row per `(s1, cand)` pair.

| field | type |
|---|---|
| `s1_idx`, `cand_idx` | int32 |
| `probe_mask` | uint8 (which of the 4 key families fired) |
| `block_score` | float32 |

**`pairs`** — candidates enriched with features and scores.

| field | type |
|---|---|
| `features` | float32[n_pairs, n_features] |
| `p1` | stage-1 probability |
| `s2_score` | stage-2 score |
| `p_cal` | calibrated probability |
| `q_compete` | column-softmax competition probability |
| `label` | int8, `-1` when unknown (test) |

**`decisions`** — per S1 entity.

| field | type |
|---|---|
| `s1_entity_id` | str |
| `selected` | list[str] |
| `k_star`, `expected_f` | chosen size and its `E[F]` |
| `ev_curve` | float32[n_cand+1] — `E[F|k]` for every `k`, for the UI |
| `realised_f` | float, validation only |

Storage: Parquet for tabular artifacts, `.npz` for float matrices, SQLite for the run
registry and review annotations. No database server to install.

### 2.2 Run lifecycle

A **run** is one full pipeline execution with a frozen config.

```
queued → normalising → blocking → featurising → training
       → scoring → graph → calibrating → deciding → done
                                                  ↘ failed
```

Runs are immutable once `done`. Config is stored as JSON alongside metrics so any
leaderboard number is reproducible and comparable in the UI. The decision stage is
**re-runnable in isolation** against a finished run — that is the whole point of the
Decision Tuner screen, and it is fast because it only touches `p_cal`.

---

## 3. Backend design decisions

| Decision | Rationale |
|---|---|
| **`er/` is framework-free** | Must lift cleanly into the submission zip; also makes it unit-testable without a server. |
| **Jobs are in-process background tasks with a thread pool** | No Celery/Redis for a 3-day challenge tool. Progress + logs polled over HTTP, with an SSE endpoint for live log tailing. |
| **NumPy arrays, not per-pair Python objects** | Candidate counts reach millions; features are one dense `float32` matrix built columnwise. |
| **`HistGradientBoostingClassifier` default, LightGBM optional** | sklearn ships wheels everywhere and needs no compiler. LightGBM is auto-detected and used when importable. |
| **Cross-encoder behind an interface, off by default** | Keeps the default path CPU-only with zero downloads, which matters for reproducibility and for the no-external-data rule. |
| **Synthetic dataset generator ships with the app** | There is no dataset in this workspace. The generator implements the documented noise patterns, including an unseen third country, so the pipeline is verifiable end to end and the France scenario is testable. |
| **Decision stage decoupled from scoring** | Re-deciding a finished run is ~10 ms; re-scoring is minutes. Enables interactive tuning. |

### 3.1 Numerical care in `decide.py`

The Poisson-binomial DP underflows for large `n` with small `p`. Mitigations:

- prune candidates with `p < ε` into the `M` term rather than carrying them in the DP,
- cap `n` at the per-entity candidate cap (default 40),
- run the DP in `float64` and renormalise the pmf each step,
- vectorise `E[F|k]` as an outer product `pmf_tp ⊗ pmf_fn` against a precomputed
  `F(t,f,k)` table.

A property test asserts the closed form `1.25·TP/(0.25|T|+|S|)` agrees with the naive
`1.25PR/(0.25P+R)` on random sets, and that brute-force enumeration over all `2ⁿ` subsets
for small `n` agrees with the top-`k` argmax. The whole idea rests on those two claims,
so they get tested rather than trusted.

---

## 4. REST surface

Full detail in [`04_API_CONTRACT.md`](04_API_CONTRACT.md). Summary:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness + capability flags (lightgbm present, etc.) |
| `GET` | `/api/datasets` | list datasets in `data/` with row counts per source/country |
| `POST` | `/api/datasets/generate` | synthesise a challenge-shaped dataset |
| `POST` | `/api/datasets/validate` | schema/format check on a dataset folder |
| `GET` | `/api/datasets/{id}/preview` | sample records per source |
| `POST` | `/api/runs` | start a run from a config |
| `GET` | `/api/runs` | list runs with headline metrics |
| `GET` | `/api/runs/{id}` | run detail: status, stage timings, config, metrics |
| `GET` | `/api/runs/{id}/logs` | SSE log stream |
| `DELETE` | `/api/runs/{id}` | delete a run and its artifacts |
| `GET` | `/api/runs/{id}/blocking` | recall ceiling, reduction ratio, probe contribution, per-country |
| `GET` | `/api/runs/{id}/features` | feature importances + per-feature separation |
| `GET` | `/api/runs/{id}/calibration` | reliability curves, per country |
| `GET` | `/api/runs/{id}/decision` | score, k\* histogram, E[F] vs realised F, threshold sweep comparison |
| `POST` | `/api/runs/{id}/decision/simulate` | re-decide with different decision params, returns new metrics |
| `GET` | `/api/runs/{id}/ablation` | stage-toggle comparison table |
| `GET` | `/api/runs/{id}/entities` | paged per-entity results, filterable by error type |
| `GET` | `/api/runs/{id}/entities/{s1_id}` | one entity: candidates, features, scores, E[F] curve, truth |
| `POST` | `/api/runs/{id}/review` | record a human verdict on a pair |
| `POST` | `/api/runs/{id}/export` | write the two TSVs + run the validator |
| `GET` | `/api/runs/{id}/export/{file}` | download a produced TSV |
| `GET` | `/api/docs-md/{name}` | serve the markdown docs to the UI |

---

## 5. Frontend design

React 19 + TypeScript + Vite, TailwindCSS, TanStack Query for server state, Recharts for
plots, React Router for navigation. Dark, dense, data-first layout — this is an analysis
tool, not a marketing page.

| Screen | What it answers |
|---|---|
| **Dashboard** | What runs exist, which scores best, what changed between them. |
| **Datasets** | What data do I have; generate or validate a dataset; preview records per source and country. |
| **New run** | Configure every stage toggle and hyper-parameter, launch, watch the live log and stage timings. |
| **Blocking** | Is my recall ceiling high enough, and how much did each probe family contribute? Per-country reduction ratio. This gates everything downstream. |
| **Model & features** | Which features matter; per-feature positive/negative separation; stage-1 vs stage-2 lift. |
| **Calibration** | Are my probabilities honest — per country? Reliability diagram + ECE. The France panel is the one to watch. |
| **Decision tuner** | The centrepiece. Live macro-F<sub>0.5</sub> as decision params change; expected-F<sub>0.5</sub> vs best global threshold side by side; `k*` distribution; predicted-vs-realised `E[F]` calibration of the decision rule itself. |
| **Entity explorer** | Per-entity drill-down: candidate table with every feature and score, the `E[F|k]` curve with the chosen `k*` marked, ground truth when available, and a filter for the specific error types (false merge / missed / wrong-singleton). |
| **Export** | Write the two TSVs, show validator output, download. |
| **Docs** | The three markdown docs rendered in-app so the methodology stays next to the numbers. |

### 5.1 Interaction notes

- Every long operation is a job with a progress bar and a cancel affordance; nothing
  blocks the UI thread.
- The Decision Tuner calls `POST /decision/simulate`, which is fast enough (~10 ms) to
  feel synchronous — debounced at 200 ms.
- Entity explorer rows link by `s1_entity_id`, so a finding is shareable as a URL.
- Accessibility: semantic landmarks, labelled form controls, visible focus rings, charts
  paired with a data table, and status changes announced via `aria-live`. Colour is never
  the only signal for match/no-match — icons and text accompany it.

---

## 6. Build order

1. Scaffold: venv, pinned `requirements.txt`, Vite app, dev proxy. ✱ verify both servers boot
2. `textnorm` + `address` + `synth` + `datasets` → **generate a dataset**, eyeball it
3. `blocking` + metrics → recall ceiling on the synthetic set must exceed 0.95
4. `features` + `model` → stage-1 AUC sanity check
5. `graph` + `decide` + `evaluate` → **property tests for the metric algebra and the argmax**
6. `pipeline` + `store` + `jobs` + routers → full run over HTTP
7. Frontend screens against the live API
8. End-to-end verification: run, score, export, validate, build

Step 5's tests are non-negotiable — they are the correctness foundation of the novel
claim. Step 3's threshold is the gate on everything downstream, because no amount of
modelling recovers recall that blocking discarded.

---

## 7. What is deliberately out of scope

- Multi-user auth and any notion of tenancy. This is a **local single-user analysis
  tool** bound to `127.0.0.1`; it has no authentication, so it must not be exposed on a
  network interface. Noted explicitly because an unauthenticated API that reads and
  writes local files is a real risk if bound to `0.0.0.0`.
- Distributed training / GPU orchestration.
- Live leaderboard integration (submissions are manual uploads by design).
- A cross-encoder as a load-bearing stage. The interface exists; the weights do not
  ship.
