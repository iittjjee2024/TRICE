# Amazon ML Challenge 2026 — Business Entity Resolution

Detailed breakdown of the problem, extracted from:

- `6ab5628d5a817_amazon_ml_challenge_problem_statement_copy.pdf`
- `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026_copy.pdf`

---

## 1. The task in one paragraph

Business identity data arrives from three independent sources. Each source holds
partial, noisy fragments describing the same real-world businesses, and **the sources
share no common identifier**. Source 1 is already deduplicated and acts as the
reference. For every Source 1 record we must output the set of Source 2 and Source 3
records that refer to the same real-world business. The answer set for a Source 1
entity may be empty, a single record, or many records.

This is classic **Entity Resolution (ER)**, but with three structural twists that
drive the whole solution design:

1. It is **one-to-many set prediction**, not pairwise classification. The unit of
   evaluation is the *set* emitted for each Source 1 entity.
2. Scoring is **macro-averaged F<sub>0.5</sub>** per Source 1 entity, so every entity
   carries equal weight regardless of how many matches it has, and **singletons
   score a full 1.0 when you correctly predict nothing**.
3. The test set contains a **country that never appears in training** (France), so the
   model must generalise zero-shot across address/name conventions.

---

## 2. Data

### 2.1 Files

All files are **tab-separated** (`.tsv`). Tabs are used because both the address field
and the ID-list columns contain commas.

```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Reading without `sep="\t"` silently collapses each line into a single column.

| Path | Contents |
|---|---|
| `dataset/train/train_source1.tsv` | Source 1 training records (deduplicated reference source) |
| `dataset/train/train_source2.tsv` | Source 2 training records |
| `dataset/train/train_source3.tsv` | Source 3 training records |
| `dataset/train/train_ground_truth.tsv` | Ground-truth match labels for training |
| `dataset/test/test_source1.tsv` | Source 1 test records — **must produce a row for every one of these** |
| `dataset/test/test_source2.tsv` | Source 2 test records |
| `dataset/test/test_source3.tsv` | Source 3 test records |

No ground truth is given for the test set. Self-evaluation requires holding out a
validation split from the training data and scoring it with the F<sub>0.5</sub>
formula below.

### 2.2 Source record schema

Every `*_source{1,2,3}.tsv` has four columns:

| Column | Description |
|---|---|
| `entity_id` | Unique record id. The **prefix encodes the source**: `S1-`, `S2-`, `S3-`. |
| `business_name` | Business name. May contain abbreviations, legal suffixes, typos, transliterations. |
| `business_address` | Address. May be partial, reformatted, missing components, or landmark-based. |
| `country` | Country label. Training covers **US** and **India**; test *additionally* has **France**. |

There is **no `source` column**. A record's source comes from its `entity_id` prefix
and from which file it lives in.

Explicit instruction on `country`: treat it as an **open set of string labels**. Do not
hard-code, filter, or one-hot the pipeline to `{US, India}`, and every test entity —
France included — must appear in the submission.

### 2.3 Ground-truth schema

`train_ground_truth.tsv` has two columns:

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching `entity_id`s from Source 2 and/or Source 3; **empty** when the entity has no matches |

### 2.4 Noise patterns explicitly called out

**Name variations**

- Abbreviations: `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`
- Legal-suffix inconsistencies (present in one source, absent in another)
- DBA / trade names
- Punctuation differences (`&` vs `and`)
- Word-order transpositions
- Typos

**Address variations**

- Abbreviations: `Rd` vs `Road`, `St` vs `Street`
- Transliteration variants
- Missing components (no PIN code, no state)
- Landmark-based references (`Near SBI ATM`)
- Municipal numbering formats
- Component reordering

---

## 3. Required output

Two tab-separated files, both in `output/` of the final submission package.

### 3.1 `matching_results.tsv` — the scored file

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching `entity_id`s from Source 2 and/or Source 3 |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
```

Rules:

- Exactly **one row per Source 1 test entity** — missing entities cause rejection.
- `matched_entity_ids` **empty** for singletons.
- **No duplicate ids** within a single list, and no duplicate `source1_entity_id` rows.
- Lists may contain **only S2-/S3- ids that exist in the test set**. Self-matches to
  Source 1 are rejected.
- Single tab between columns; commas between ids; **no quoting**.

This is the only file uploaded to the leaderboard during the challenge.

### 3.2 `candidate_pairs.tsv` — the blocking audit file

Same shape, with `candidate_entity_ids` instead of `matched_entity_ids`.

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `candidate_entity_ids` | Comma-separated candidate `entity_id`s from Source 2 and/or Source 3 |

Critical definition: this must be the **final** candidate set — the exact set of
records the matching model runs inference over, *not* the raw output of an early
blocking pass that is later filtered. If the pipeline has several blocking/filtering
stages, `candidate_pairs.tsv` is the last one. Therefore **every id in
`matching_results.tsv` must also appear here**; a matched id that was never a
candidate signals a pipeline bug and the validator warns about it.

Not scored on the leaderboard. Used by the organisers to analyse blocking quality
(recall ceiling, reduction ratio) and to verify the pipeline.

### 3.3 Local validation

A stdlib-only helper is provided in the student resources:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Prints `PASS` (exit 0) or a numbered list of issues (exit 1). It checks format only —
it does not compute a score.

---

## 4. Evaluation metric

Submissions are scored with **F<sub>β</sub>, β = 0.5** — precision-weighted, because
falsely merging two distinct businesses is more damaging than missing a link.

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed **per Source 1 entity**, then **macro-averaged** over all Source 1 entities
in the evaluation set. Singletons are included in the average.

Edge cases stated in the problem:

| True set | Predicted set | Score |
|---|---|---|
| empty | empty | **1.0** |
| empty | non-empty | **0.0** |
| non-empty | empty | **0.0** |

Worked example from the problem statement:

- Predicted `[S2-00047, S2-00193, S3-00812]`
- Truth `[S2-00047, S3-00812]`
- Precision = 2/3, Recall = 2/2 = 1.0
- F<sub>0.5</sub> = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

### 4.1 A useful algebraic simplification

Let `S` be the predicted set, `T` the true set, and `TP = |S ∩ T|`. Substituting
`P = TP/|S|` and `R = TP/|T|`:

```
          1.25 · (TP/|S|) · (TP/|T|)         1.25 · TP² / (|S||T|)              1.25 · TP
F_0.5 = ------------------------------- = ------------------------------- = -------------------
         0.25 · (TP/|S|) + (TP/|T|)        TP · (0.25|T| + |S|) / (|S||T|)    0.25 · |T| + |S|
```

So for a Source 1 entity:

```
F_0.5(S, T) = 1.25 · |S ∩ T| / (0.25 · |T| + |S|)          when |S| > 0 or |T| > 0
F_0.5(S, T) = 1                                            when |S| = |T| = 0
```

This closed form is linear in `TP` with a denominator that only depends on the two set
*sizes*. That is what makes the decision layer described in
[`02_SOLUTION_IDEA.md`](02_SOLUTION_IDEA.md) exactly optimisable rather than
heuristic — it is the single most exploitable property of this metric.

### 4.2 What the metric implies

- **A global probability threshold is the wrong tool.** The marginal value of adding a
  candidate depends on how many candidates you already emitted for *that* entity and
  on the likely size of that entity's true set. Adding a p = 0.5 candidate to an entity
  that already has 4 confident matches is a very different trade than adding it as the
  only prediction.
- **Singletons are free points.** With correct empty prediction they pay a full 1.0.
  Any false positive on a singleton costs the entire entity, so precision discipline
  on low-evidence entities matters more than squeezing recall on rich entities.
- **Big entities are worth no more than small ones.** Macro-averaging removes the
  incentive to optimise for high-degree entities. Per-entity calibration beats
  global tuning.

---

## 5. Constraints and rules

### 5.1 Hard constraints

1. Output format must match exactly; failing validation means no evaluation. A correct
   submission shows `SCORED` status with an F<sub>0.5</sub> score.
2. `matched_entity_ids` may reference only Source 2 / Source 3 entities. Self-matches
   to Source 1, and ids absent from the test set, are rejected.
3. Every Source 1 entity must appear in the submission.
4. No duplicate entity ids in any list, no duplicate `source1_entity_id` rows.
5. The final model must be **MIT / Apache-2.0 licensed** and **≤ 8 billion parameters**.

### 5.2 Academic integrity — strictly prohibited

External data lookup of any kind is banned and results in **immediate
disqualification**:

- Commercial entity-resolution APIs or services
- Looking up business registrations in government databases
- **Geocoding APIs to normalise addresses**
- Any external data augmentation from internet sources

Everything must be derived from the provided training and test data. Note the
important distinction this leaves open: *transductive* use of the provided **test
records themselves** (unsupervised structure, corpus statistics, self-supervised
augmentation) is not external data — it is the given data. The solution design leans
on this.

### 5.3 Logistics

- **Challenge window:** 25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST (3 days).
- **Max 5 submissions per day.** Keep version history of every submission.
- **Two leaderboards:** public (subset of test, live) and private (remaining portion,
  revealed at the end). Final ranking is the **private** leaderboard. Predictions are
  submitted for the full test set either way; the split is applied at scoring time.
- Desktop/laptop only; **no simultaneous logins**.
- Top 100 teams must submit methodology, blocking strategy, model architecture and
  feature engineering write-ups.

### 5.4 Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file uploaded to leaderboard)
│   └── candidate_pairs.tsv         # blocking candidate set fed to the model
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # end-to-end reproduction steps
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled-in methodology write-up
```

Anyone must be able to regenerate both output files from the training/test data using
only what is in `code/business_entity_resolution/`. No page limit on the methodology
document — depth is preferred over brevity.

---

## 6. Tips given by the organisers

- Invest in blocking / candidate generation — it sets the **upper bound on recall**.
- Explore string-similarity features (Jaccard, Levenshtein, TF-IDF cosine) for names
  and addresses.
- Pay attention to **country-specific address patterns**.
- Weigh the precision/recall trade-off deliberately; F<sub>0.5</sub> favours precision.
- Do not neglect singletons — a correct "no match" is worth a full 1.0.
- Validate output format locally before spending a submission.

---

## 7. Restating the problem formally

Let

- `A = {a₁ … a_n}` be Source 1 records (deduplicated, so distinct real businesses),
- `B = {b₁ … b_m}` be Source 2 records, `C = {c₁ … c_k}` be Source 3 records,
- `x(e) = (name, address, country)` the observed noisy attributes of record `e`.

We must learn `f : A → 2^(B ∪ C)` maximising

```
          1                    1.25 · |f(a) ∩ T(a)|
J(f) = ------- ·  Σ    ──────────────────────────────────
         |A|      a∈A    0.25 · |T(a)| + |f(a)|
```

with the convention that the summand is 1 when `f(a) = T(a) = ∅`.

Two latent structural facts, never stated as constraints but true of the generating
process, are exploitable:

- **Source 1 is deduplicated** ⇒ distinct `a` are distinct businesses ⇒ the true sets
  `{T(a)}` are **pairwise disjoint**. A Source 2/3 record belongs to at most one
  Source 1 entity. This turns the problem into a constrained *assignment*, which is
  strictly more information than independent pairwise classification.
- **B and C describe the same businesses** ⇒ a strong `b ↔ c` similarity is evidence
  that `b` and `c` share a Source 1 parent, even when neither links confidently to
  Source 1 on its own. The tripartite graph carries information that no pairwise
  `(a, ·)` model can see.

These two facts, plus the closed form in §4.1, are the foundation of the approach in
[`02_SOLUTION_IDEA.md`](02_SOLUTION_IDEA.md).
