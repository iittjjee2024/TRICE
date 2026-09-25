# API contract

Base URL `http://127.0.0.1:8000`. All payloads JSON unless noted. Interactive schema at
`/docs` (FastAPI auto-generated).

> The server binds to loopback only and has **no authentication**. It reads and writes
> files under the project directory. Do not bind it to a public interface.

---

## Conventions

- Ids: `dataset_id` and `run_id` are slugs (`^[a-z0-9][a-z0-9_-]{0,63}$`).
- Errors: `{"detail": "..."}` with standard HTTP codes. `404` unknown id, `409` illegal
  state transition, `422` validation error (FastAPI default).
- Timestamps: ISO-8601 UTC.
- All floats are JSON numbers; `NaN`/`Infinity` are never emitted (nulls instead).

---

## Health

### `GET /api/health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "capabilities": {
    "lightgbm": false,
    "rapidfuzz": true,
    "cross_encoder": false
  },
  "paths": { "data": "data", "artifacts": "artifacts", "output": "output" }
}
```

---

## Datasets

### `GET /api/datasets`

```json
{
  "datasets": [
    {
      "dataset_id": "synthetic_demo",
      "path": "data/synthetic_demo",
      "has_train": true,
      "has_test": true,
      "has_ground_truth": true,
      "counts": {
        "train": { "source1": 4000, "source2": 5200, "source3": 4800 },
        "test":  { "source1": 1500, "source2": 1950, "source3": 1800 }
      },
      "countries": { "train": ["India", "US"], "test": ["France", "India", "US"] },
      "generated": true,
      "created_at": "2026-09-25T09:12:04Z"
    }
  ]
}
```

### `POST /api/datasets/generate`

Synthesises a challenge-shaped dataset using the mined noise-operator vocabulary.

```json
{
  "dataset_id": "synthetic_demo",
  "n_train_entities": 4000,
  "n_test_entities": 1500,
  "train_countries": ["US", "India"],
  "test_only_countries": ["France"],
  "singleton_rate": 0.28,
  "noise_level": 1.0,
  "seed": 20260925,
  "overwrite": false
}
```

`201` → the dataset summary object above.

Field notes:

| field | default | meaning |
|---|---|---|
| `singleton_rate` | `0.28` | fraction of S1 entities with **no** matches — drives how much of the score is singleton credit |
| `noise_level` | `1.0` | global multiplier on operator firing rates |
| `test_only_countries` | `["France"]` | countries present in test but **not** train, reproducing the zero-shot condition |

### `POST /api/datasets/validate`

```json
{ "dataset_id": "synthetic_demo" }
```

→

```json
{
  "ok": true,
  "issues": [],
  "warnings": ["train_source3.tsv: 12 records have an empty business_address"]
}
```

### `GET /api/datasets/{dataset_id}/preview?split=train&source=1&limit=25&country=India`

```json
{
  "columns": ["entity_id", "business_name", "business_address", "country"],
  "rows": [["S1-000001", "Ramesh Traders Pvt Ltd", "12 MG Rd, Bengaluru 560001", "India"]],
  "total": 4000
}
```

### `DELETE /api/datasets/{dataset_id}`

`204`. Refuses with `409` if a run references it.

---

## Runs

### `POST /api/runs`

```json
{
  "dataset_id": "synthetic_demo",
  "name": "trice-full",
  "notes": "all stages on",
  "config": {
    "validation": {
      "mode": "random",
      "holdout_fraction": 0.25,
      "holdout_country": null,
      "seed": 7
    },
    "blocking": {
      "max_candidates_per_entity": 40,
      "tfidf_char_ngram": [2, 4],
      "tfidf_top_k": 30,
      "token_probe": true,
      "postal_probe": true,
      "phonetic_probe": true,
      "min_block_score": 0.02
    },
    "features": { "use_address": true, "use_rarity": true, "use_cross_field": true },
    "model": {
      "kind": "hgb",
      "max_iter": 300,
      "learning_rate": 0.08,
      "max_leaf_nodes": 31,
      "stage2": true
    },
    "graph": {
      "tripartite_messages": true,
      "competition_features": true,
      "softmax_temperature": 0.15
    },
    "augmentation": { "noise_replay": true, "n_synthetic_pairs": 20000 },
    "calibration": { "method": "isotonic", "per_country": true },
    "decision": {
      "rule": "expected_f",
      "beta": 0.5,
      "prune_epsilon": 0.01,
      "missing_mass": "auto",
      "global_threshold": 0.5,
      "max_emit": 25,
      "disjointness_repair": true
    }
  }
}
```

`mode` ∈ `random` | `by_country` | `none`. `by_country` with `holdout_country: "India"`
reproduces the zero-shot setting locally — the honest France proxy.

`decision.rule` ∈ `expected_f` | `global_threshold` | `top1` — the last two exist to
quantify the gain from the decision layer.

`201` →

```json
{ "run_id": "r-20260925-091204-a3f1", "status": "queued" }
```

### `GET /api/runs`

```json
{
  "runs": [
    {
      "run_id": "r-20260925-091204-a3f1",
      "name": "trice-full",
      "dataset_id": "synthetic_demo",
      "status": "done",
      "created_at": "2026-09-25T09:12:04Z",
      "duration_s": 74.2,
      "headline": {
        "val_macro_f05": 0.8471,
        "val_precision": 0.9012,
        "val_recall": 0.7788,
        "blocking_recall": 0.9863,
        "reduction_ratio": 0.9991,
        "n_test_entities": 1500
      }
    }
  ]
}
```

### `GET /api/runs/{run_id}`

```json
{
  "run_id": "r-...",
  "name": "trice-full",
  "dataset_id": "synthetic_demo",
  "status": "done",
  "stage": "done",
  "progress": 1.0,
  "config": { "...": "as submitted, with defaults filled in" },
  "stages": [
    { "name": "normalising",  "status": "done", "seconds": 1.9 },
    { "name": "blocking",     "status": "done", "seconds": 12.4 },
    { "name": "featurising",  "status": "done", "seconds": 21.7 },
    { "name": "training",     "status": "done", "seconds": 18.2 },
    { "name": "graph",        "status": "done", "seconds": 6.1 },
    { "name": "calibrating",  "status": "done", "seconds": 1.1 },
    { "name": "deciding",     "status": "done", "seconds": 0.9 }
  ],
  "metrics": { "...": "see /decision and /blocking for detail" },
  "error": null
}
```

### `GET /api/runs/{run_id}/logs` — `text/event-stream`

```
event: log
data: {"ts":"2026-09-25T09:12:07Z","level":"info","stage":"blocking","msg":"probe tfidf: 58,201 pairs"}

event: status
data: {"status":"running","stage":"featurising","progress":0.34}

event: done
data: {"status":"done"}
```

Also available non-streaming: `GET /api/runs/{run_id}/logs?tail=500`.

### `POST /api/runs/{run_id}/cancel` → `202`
### `DELETE /api/runs/{run_id}` → `204`

---

## Analysis

### `GET /api/runs/{run_id}/blocking`

```json
{
  "overall": {
    "recall_ceiling": 0.9863,
    "reduction_ratio": 0.9991,
    "n_candidate_pairs": 412903,
    "n_possible_pairs": 462150000,
    "mean_candidates_per_entity": 27.5,
    "entities_with_zero_candidates": 41
  },
  "by_country": [
    { "country": "US", "recall_ceiling": 0.991, "mean_candidates": 26.2, "n_entities": 700 },
    { "country": "India", "recall_ceiling": 0.978, "mean_candidates": 29.8, "n_entities": 800 }
  ],
  "by_probe": [
    { "probe": "tfidf_char", "pairs": 380120, "unique_true_found": 5120, "exclusive_true": 318 },
    { "probe": "rare_token", "pairs": 91043,  "unique_true_found": 4880, "exclusive_true": 201 },
    { "probe": "postal",     "pairs": 44210,  "unique_true_found": 3102, "exclusive_true": 96 },
    { "probe": "phonetic",   "pairs": 61885,  "unique_true_found": 4401, "exclusive_true": 57 }
  ],
  "candidate_count_histogram": [[0, 41], [1, 120], [2, 260]],
  "missed_examples": [
    { "s1_entity_id": "S1-000317", "missed_id": "S2-001902",
      "s1_name": "...", "missed_name": "...", "reason": "no shared rare token" }
  ]
}
```

`exclusive_true` — true matches found by **only** this probe. It is the number that
justifies keeping a probe family.

### `GET /api/runs/{run_id}/features`

```json
{
  "model": { "kind": "hgb", "stage1_auc": 0.9932, "stage2_auc": 0.9971, "n_features": 58 },
  "importances": [
    { "feature": "name_idf_soft_cosine", "importance": 0.184, "stage": 1 },
    { "feature": "compete_margin",       "importance": 0.121, "stage": 2 }
  ],
  "separation": [
    { "feature": "name_idf_soft_cosine", "pos_mean": 0.88, "neg_mean": 0.21, "auc": 0.951 }
  ],
  "score_histogram": {
    "bins": [0.0, 0.05, "..."],
    "positive": [12, 30, "..."],
    "negative": [90120, 4200, "..."]
  }
}
```

### `GET /api/runs/{run_id}/calibration`

```json
{
  "overall": { "ece": 0.0134, "brier": 0.0271, "method": "isotonic" },
  "curves": [
    {
      "group": "US",
      "n": 120431,
      "ece": 0.011,
      "points": [ { "p_pred": 0.05, "p_true": 0.041, "n": 41022 } ]
    },
    { "group": "France", "n": 38210, "ece": 0.029, "points": [], "note": "calibrated from synthetic replay" }
  ]
}
```

### `GET /api/runs/{run_id}/decision`

```json
{
  "score": {
    "macro_f05": 0.8471,
    "macro_precision": 0.9012,
    "macro_recall": 0.7788,
    "n_entities": 1000,
    "singletons": { "n": 280, "correct": 251, "credit": 0.8964 },
    "non_singletons": { "n": 720, "macro_f05": 0.8279 }
  },
  "by_country": [ { "country": "India", "macro_f05": 0.8312, "n": 400 } ],
  "k_star_histogram": [[0, 302], [1, 410], [2, 180]],
  "ev_calibration": [
    { "bin": 0.85, "predicted_ef": 0.85, "realised_f": 0.83, "n": 220 }
  ],
  "rule_comparison": [
    { "rule": "expected_f",              "macro_f05": 0.8471 },
    { "rule": "global_threshold@best",    "macro_f05": 0.8092, "threshold": 0.62 },
    { "rule": "top1",                     "macro_f05": 0.7415 }
  ],
  "threshold_sweep": [ { "threshold": 0.5, "macro_f05": 0.79, "precision": 0.84, "recall": 0.80 } ],
  "error_breakdown": {
    "false_merge_on_singleton": 29,
    "false_merge_on_matched": 96,
    "missed_all": 74,
    "partial": 210,
    "exact": 591
  }
}
```

### `POST /api/runs/{run_id}/decision/simulate`

Re-runs **only** the decision stage against stored calibrated probabilities. Fast
(~10 ms), so the UI can drive it from sliders.

```json
{
  "rule": "expected_f",
  "beta": 0.5,
  "prune_epsilon": 0.01,
  "missing_mass": 0.02,
  "max_emit": 25,
  "global_threshold": 0.5,
  "probability_power": 1.0,
  "disjointness_repair": true
}
```

→ the same shape as `GET /decision`, plus `"applied": false` (simulation does not mutate
the run). `POST /api/runs/{run_id}/decision/commit` with the same body persists it.

`probability_power` raises `p → p^γ` before the DP — a single-parameter escape hatch if
the independence assumption proves optimistic on validation.

### `GET /api/runs/{run_id}/ablation`

```json
{
  "rows": [
    { "config": "baseline (block + stage1 + τ)", "macro_f05": 0.7712, "delta": null },
    { "config": "+ expected-F0.5 selection",     "macro_f05": 0.8134, "delta": 0.0422 },
    { "config": "+ competition features",        "macro_f05": 0.8301, "delta": 0.0167 },
    { "config": "+ tripartite messages",         "macro_f05": 0.8389, "delta": 0.0088 },
    { "config": "+ per-country calibration",     "macro_f05": 0.8471, "delta": 0.0082 }
  ]
}
```

Computed from stored artifacts where possible (decision-layer rows are free); model-level
rows require a re-run and are returned as `null` with `"needs_run": true` until executed.

---

## Entities & review

### `GET /api/runs/{run_id}/entities`

Query: `page`, `page_size` (≤200), `filter` ∈ `all|exact|partial|false_merge|missed|singleton_ok|singleton_bad`,
`country`, `q` (substring on name/id), `sort` ∈ `f05|n_cand|k_star`.

```json
{
  "total": 1000,
  "page": 1,
  "page_size": 50,
  "rows": [
    {
      "s1_entity_id": "S1-000317",
      "business_name": "Ramesh Traders Pvt Ltd",
      "country": "India",
      "n_candidates": 18,
      "k_star": 2,
      "expected_f": 0.91,
      "realised_f": 0.83,
      "predicted": ["S2-001902", "S3-004410"],
      "truth": ["S2-001902", "S3-004410", "S3-004411"],
      "verdict": "partial"
    }
  ]
}
```

### `GET /api/runs/{run_id}/entities/{s1_entity_id}`

```json
{
  "s1": { "entity_id": "S1-000317", "business_name": "...", "business_address": "...",
          "country": "India", "name_norm": "...", "addr_norm": "...",
          "components": { "postal": "560001", "house_no": "12", "street": "mg road" } },
  "ev_curve": [ { "k": 0, "expected_f": 0.12 }, { "k": 1, "expected_f": 0.68 } ],
  "k_star": 2,
  "candidates": [
    {
      "entity_id": "S2-001902",
      "business_name": "Ramesh Traders Private Limited",
      "business_address": "12 M.G. Road, Bangalore",
      "country": "India",
      "probes": ["tfidf_char", "rare_token"],
      "p1": 0.94, "s2_score": 4.1, "p_cal": 0.97, "q_compete": 0.95,
      "selected": true,
      "is_true": true,
      "top_features": [ { "feature": "name_idf_soft_cosine", "value": 0.93 } ]
    }
  ],
  "truth": ["S2-001902", "S3-004410", "S3-004411"],
  "realised_f": 0.83
}
```

`top_features` is the per-pair contribution view — enough to explain *why* a pair scored
the way it did without shipping a full SHAP dependency.

### `POST /api/runs/{run_id}/review`

```json
{ "s1_entity_id": "S1-000317", "candidate_entity_id": "S2-001902", "verdict": "match", "note": "same branch" }
```

`verdict` ∈ `match` | `no_match` | `unsure`. Stored in SQLite. Reviews are **advisory
annotations for error analysis** — they are never fed back into training, to keep runs
reproducible from config alone.

### `GET /api/runs/{run_id}/review` → all recorded verdicts.

---

## Export

### `POST /api/runs/{run_id}/export`

```json
{ "split": "test", "output_dir": "output", "run_validator": true }
```

→

```json
{
  "files": [
    { "name": "matching_results.tsv", "path": "output/matching_results.tsv", "rows": 1500, "bytes": 61204 },
    { "name": "candidate_pairs.tsv",  "path": "output/candidate_pairs.tsv",  "rows": 1500, "bytes": 184311 }
  ],
  "validation": {
    "ok": true,
    "issues": [],
    "warnings": [],
    "checks": {
      "one_row_per_s1": true,
      "all_test_s1_present": true,
      "no_duplicate_rows": true,
      "no_duplicate_ids_within_list": true,
      "only_s2_s3_ids": true,
      "all_ids_exist_in_test": true,
      "matches_subset_of_candidates": true
    }
  }
}
```

The `checks` map mirrors the rules in `01_PROBLEM_STATEMENT.md` §3.1 one-for-one, so a
green panel here means the official `utils/validate_submission.py` will pass too.

### `GET /api/runs/{run_id}/export/{filename}` — `text/tab-separated-values` download.

---

## Docs

### `GET /api/docs-md` → `{"docs":[{"name":"01_PROBLEM_STATEMENT","title":"..."}]}`
### `GET /api/docs-md/{name}` → `{"name":"...","markdown":"..."}`
