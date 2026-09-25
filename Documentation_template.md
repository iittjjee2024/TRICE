# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Vortex
**Team Members:** Aryan Bansal (lead), Sachin Kumar, Daksh Tandon, Parth Aggarwal
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 1 businesses against Source 2/3 with a four-stage pipeline: script-aware
normalisation (including a consonant-skeleton transform that collapses Devanagari and Latin
renderings of the same name onto one string), dual-channel IDF-weighted sparse blocking, a
gradient-boosted matcher, and a **decision layer that computes the Bayes-optimal emitted set
per entity in closed form** — exploiting the fact that
`F_0.5(S,T) = 1.25·|S∩T| / (0.25·|T| + |S|)`, whose denominator depends only on set sizes.
Our other main contribution encodes a property we verified in the labels — Source 2/3 records
have **exactly one** parent across all 7,638,365 ground-truth links — as a column-softmax
competition normalisation with a null dustbin. On a held-out validation split these reach
**macro F<sub>0.5</sub> = 0.91698** (precision 0.9548, recall 0.8447), versus 0.7524 for the
best fixed-cardinality rule and 0.9156 for the best *tuned* global threshold.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled all 24 M records and read hundreds of real ground-truth match groups before
writing any modelling code. Findings that changed the design:

**Scale.** Test is 1,732,544 Source 1 entities against 9,969,589 Source 2/3 records —
1.7 × 10<sup>13</sup> naive pairs. Available hardware: 16 cores, **16.9 GB RAM**. Memory, not
CPU, was the binding constraint throughout.

**Ground truth is exactly one-to-many, never many-to-many.**

```
distinct matched ids     : 7,638,365
ids used by >1 S1 entity : 0  (0.0000%)
max reuse of a single id : 1
```

Additionally 73.4 % of Source 2 and 74.6 % of Source 3 records participate in a match, so the
true structure is close to a near-perfect assignment. This is far stronger information than
generic ER assumes, and we exploit it directly (§4, Stage R).

**Singletons are rare; cardinality is the real problem.** Only 5.58 % of entities have no
match. Mean |T| = 3.461, max 11:

| \|T\| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| share | 5.58 % | 5.40 % | 17.00 % | 24.05 % | 21.94 % | 14.59 % | 7.47 % | 2.90 % | 1.06 % |

So the decision layer's job is choosing the correct **set size**, not deciding whether to
abstain. A `top-1` rule is capped near 0.67 and a fixed `top-3` near 0.75.

**Noise operators, catalogued from real match groups.** Name: junk framing (`...`, `--`, `##`,
`<<`, `M/s`, leading `THE`); legal-suffix swap/drop (`Limited`↔`Ltd`, `Private Limited`↔`Pvt
Ltd`); *added* generic descriptors (`Services`, `Center`, `(Partners)`); **DBA with an
unrelated prefix** (`Halonex dba Marnie Baynes Keystone Bnb Inc`); **domainification**
(`edwardshintzedougherty.com`); token truncation (`Ace Foods Limited` → `Ace`); word-order
transposition; 1–2 character typos (`Marnie`→`Mamie`, `World`→`Woerd`, `EDWARDS`→`EDWARD5`,
`Golden`→`G0lden`); and **full Devanagari transliteration** of the name.

Address: street-type abbreviation (`Drive`↔`Dr`); **ordinal ↔ word** (`14th`↔`FOURTEENTH`) and
even wrong expansion (`St`→`SAINT`); state abbreviation ↔ full name ↔ Devanagari
(`TX`↔`Texas`, `TN`↔`Tamil Nadu`↔`तमिलनाडु`); city renames (`Mysore`↔`Mysuru`); **component
reordering** (`44 Goodrich Avenue, Auburn, ME` → `ME, Auburn, 44 Goodrich Avenue`); house-number
junk (`HN 823`, `##84-11`); digit corruption (`1329`→`132`); literal `<NULL>`; and ~3.4 % of
Source 2/3 addresses entirely empty.

**Transliteration needs more than `unidecode`.**
`unidecode('राम मार्केटिंग प्राइवेट लिमिटेड') = 'raama maarkettiNg praaivett limitteda'`, which
does **not** equal `ram marketing private limited`. Our fix is described in §2.2.

**France is 15 % of the test set with zero labelled examples** (259,452 of 1,732,544 Source 1
entities), so `country` is treated strictly as an opaque partition key — never hard-coded,
filtered or one-hot encoded.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — multi-channel blocking + GBDT classifier + graph-structure
stacking + decision-theoretic set selection.

**Core Innovation:** Two things.

**(1) Exact expected-F<sub>0.5</sub> set selection.** Substituting `P = |S∩T|/|S|` and
`R = |S∩T|/|T|` into the metric gives

```
F_0.5(S,T) = 1.25·|S∩T| / (0.25·|T| + |S|)        F_0.5(∅,∅) = 1
```

The numerator is linear in the true-positive count and the denominator depends only on the two
set *sizes*. Two consequences make the optimum computable:

* `|T|` is a property of nature and does not depend on our choice, so for a **fixed** size `k`
  the denominator's distribution is unaffected by *which* `k` candidates we pick. Since
  `F(t+1,f−1,k) − F(t,f,k) = 1.25/(0.25(t+f)+k) > 0`, swapping a selected candidate for an
  unselected one of higher probability weakly improves the objective — so the **top-`k` prefix
  is optimal**.
* Therefore only `k = 0 … n` need evaluating, not `2^n`.

With `Yᵢ ~ Bernoulli(pᵢ)`, `TP ~ PoissonBinomial(p₁…p_k)` and
`FN ~ PoissonBinomial(p_{k+1}…pₙ) + M`, and `E[F|k]` is the quadratic form
`pmf_TP ᵀ · F_k · pmf_FN`. Both claims are verified against exhaustive 2<sup>n</sup>
enumeration in `scripts/test_decide.py` (400 random cases, max expected-value gap
6.7 × 10⁻¹⁶, zero mismatches). The singleton decision is *not* special-cased:
`E[F|0] = Π(1−pᵢ)`.

**(2) Disjointness as a column softmax with a null dustbin.** Because every Source 2/3 record
has at most one parent, we normalise **down the columns** (over the Source 1 entities competing
for one candidate) with a null option so a candidate may match nothing:

```
q(a,b) = exp(s(a,b)/τ) / ( exp(s_∅(b)/τ) + Σ_{a'∈cand(b)} exp(s(a',b)/τ) )
```

The asymmetry is deliberate: columns are constrained to one parent, rows are not, since an
entity legitimately has many matches. A symmetric Sinkhorn would encode the wrong constraint.
This turned out to be the single most valuable feature we added (§4).

**Consonant skeleton for cross-script matching.** Dropping the spurious `unidecode` capital-`N`
artefact, removing vowels and collapsing doubled consonants maps both renderings to one string:

```
consonant_skeleton('ram marketing private limited')                  -> 'rmrktng'
consonant_skeleton(unidecode('राम मार्केटिंग प्राइवेट लिमिटेड'))          -> 'rmrktng'
```

**Data-derived alias mining instead of a gazetteer.** Rather than importing geographic
reference data — which would sit uncomfortably close to the ban on external augmentation — we
*learn* aliases from ground-truth matched pairs. For a true match the two addresses describe the
same place, so tokens appearing on exactly one side are candidate aliases; aggregated over
250,000 groups with a cosine-style association score `c/√(n_x·n_y)`, the real aliases separate
cleanly. Mined automatically: `texas↔tx`, `ohio↔oh`, `illinois↔il`, `maharashtra↔mh`,
`karnataka↔ka`, `tamil↔tmilllnaattu`, `mhaaraassttr↔maharashtra`, `dillii↔delhi`,
`odisha↔orissa`, `mysore↔mysuru`, `fort↔ft`, plus name aliases `maarketting↔marketing`,
`fuudds↔foods`, `faainens↔finance`, `eksportts↔exports` — 361 address and 602 name aliases in
total, zero external lookups.

Full pipeline:

```
normalise → mine aliases → block (2 channels) → stage-1 GBDT
   → graph features (competition + corroboration) → stage-2 GBDT
   → per-country isotonic calibration → expected-F_0.5 set selection → TSV
```

---

## 3. Candidate Generation (Blocking)

Retrieval runs as **two independent channels**, each an L2-normalised IDF-weighted sparse
vector space, whose top-*k* results are unioned. A single blended similarity cannot serve this
data: some matches have an intact address but an unusable name (Devanagari, domainified,
truncated to one token), others an intact name but an empty or reordered address.

- **Blocking keys used:**

  | channel | namespace | key |
  |---|---|---|
  | name | `n` | core name token (legal suffixes stripped, aliases folded) |
  | name | `k` | 6-char n-gram of the name consonant skeleton |
  | name | `w` | whole space-free core name (recovers `edwardshintzedougherty.com`) |
  | name | `S` | whole name skeleton (recovers Devanagari renderings) |
  | addr | `a` | canonical address token (street types, ordinals, aliases normalised) |
  | addr | `d` | numeric address token |
  | addr | `H` | house number + address token |
  | addr | `D` | full sorted numeric signature |

  Candidate generation and a usable similarity score both fall out of one sparse product
  `Q @ Xᵀ`. Its cost is `Σ_t df_query(t)·df_index(t)`, so **per-namespace document-frequency
  caps** are the primary performance control — and they double as an automatic stop list,
  removing `services`/`colony`/`road` without a hand-written one. Production caps: name tokens
  4,000, skeleton n-grams 1,200, anchors 200, address tokens 6,000. Tightening the skeleton cap
  alone cut estimated full-test query time from **165 min to 12 min** for 0.02 recall.

  Both operands are CSR so SciPy takes its efficient SMMP path; the index is built **already
  transposed** to avoid a multi-gigabyte transpose copy. Everything is partitioned by the
  `country` label, which cuts work ~3×.

- **Candidate pairs generated:** 47.0 per entity after the union and a cap of 48.
  On the validation build, **6,584,804 pairs** over 140,000 entities. On the full test set,
  **≈ 80 M pairs** over 1,732,544 entities — a reduction ratio of ≈ 0.9999954.

- **How we ensured true matches were not lost:**
  The two channels are complementary, not redundant. Measured alone they reach only 0.65
  (name) and 0.69 (address) macro recall, but their union reaches 0.88; exclusive true-link
  attribution was 3,551 name-only and 4,205 address-only, so neither can be dropped. Achieved
  recall ceiling:

  | partition | macro recall | micro recall | true links | found |
  |---|---|---|---|---|
  | India | 0.8640 | 0.8555 | 242,608 | 207,540 |
  | US | 0.8943 | 0.8877 | 242,052 | 214,875 |
  | **overall** | **0.8792** | — | — | — |

  We deliberately stopped chasing recall at ~0.88. With achieved precision `P`, retrieving a
  fraction `r` of true matches caps the score at `1.25·r/(0.25+r)` for perfect selection;
  at our operating point another 2 points of recall is worth ≈ 0.008 F<sub>0.5</sub>, while a
  single false merge costs an entire entity. Raising `top_k` from 22/18 to 30/26 and the cap
  from 32 to 48 did lift recall (US 0.879 → 0.894, India 0.841 → 0.864) and was worth taking,
  but the marginal return falls off quickly.

---

## 4. Matching Model

**Features used** (55 pairwise + 24 graph = 79 at stage 2):

- **Name features:** IDF-weighted soft cosine and containment over core tokens; max/sum IDF of
  shared tokens; max IDF of *unshared* tokens on each side; Jaccard and containment; token
  count and delta; `rapidfuzz` battery (ratio, token-set, token-sort, partial, Jaro-Winkler);
  greedy fuzzy token alignment returning both the best and the **weakest** aligned pair (the
  weakest link exposes two generic names aligning loosely — the false-merge signature);
  consonant-skeleton ratio / equality / prefix equality; space-free equality, containment and
  ratio; acronym match; legal-suffix agree / conflict / both-absent.
- **Address features:** treated as an **unordered bag** because component reordering is
  endemic — IDF cosine, Jaccard, containment, shared-token count and max shared IDF; full-string
  ratio and token-set ratio; numeric-signature Jaccard / containment / shared count; house-number
  exact and prefix equality; postal exact, 3-prefix and both-present.
- **Other:** blocking score, which channels fired, within-entity candidate rank and count;
  address-empty flags for each side; cross-field similarity (Source 3 sometimes packs the name
  into the address); country agreement.
- **Graph features (stage 2 only):** competition count / rank / is-best / **margin** / ratio /
  **column-softmax-with-dustbin**; the entity's own candidate profile (count, rank, max, sum,
  mean, margin-to-max, rank within the candidate's own source); and corroboration — for each of
  postal code, numeric signature and name skeleton, the group size and the best probability among
  group-mates **from the other source**, plus the other source's best probability overall. This
  is `O(n log n)` grouping rather than an `O(n²)` all-pairs similarity.

**Model type:** LightGBM (MIT) gradient-boosted trees, two stacked stages; scikit-learn
`HistGradientBoostingClassifier` (BSD-3-Clause) is the automatic fallback. **50,148 + 30,744 =
80,892 parameters** — about 10⁻⁵ of the 8-billion cap. No pretrained weights are downloaded and
no external corpus is consulted.

Stage 2 is **adopted only if it beats stage 1 on held-out average precision**; the decision is
recorded in the model bundle so inference cannot silently run the weaker model. (An earlier
version fed stage 2 only a 12-column slice of the raw features to save memory; the truncation
cost more than the graph features added — AP 0.98886 vs 0.99441 — so we spill the full pairwise
matrix to an on-disk `float32` memmap during inference instead.)

| | AUC | Average precision |
|---|---|---|
| stage 1 (pairwise) | 0.99968 | 0.99553 |
| stage 2 (+ graph) | **0.99981** | **0.99732** |

Stage-2 gain importance is dominated by `p1_logit` (0.387), **`compete_softmax` (0.381)** and
`p1` (0.218) — the competition normalisation is nearly as informative as the stage-1 score
itself, which is why the graph stage is worth +0.0109 F<sub>0.5</sub>.

**Calibration.** Isotonic regression fitted **per country on training pairs only**, with a
pooled fallback for small or unseen groups — this is what covers France. Calibration is
load-bearing rather than cosmetic: the decision layer consumes probabilities, and
miscalibration moves the `argmax` over `k` while leaving AUC untouched.

| group | pairs | ECE | Brier |
|---|---|---|---|
| overall | 1,640,527 | 0.00078 | 0.00240 |
| India | 817,664 | 0.00084 | 0.00244 |
| US | 822,863 | 0.00074 | 0.00236 |

**Threshold selection method:** **none — the threshold is eliminated.** Instead of tuning a
cut-off we compute `argmax_k E[F_0.5 | top-k]` per entity from the Poisson-binomial. The only
decision-stage parameters are searched once on validation by `scripts/07_tune_decision.py`
(144 configurations over missing-mass variant, probability power γ, prune ε and the
disjointness-repair toggle); the winner was γ = 1.0, ε = 0.005, repair on — γ = 1.0 confirming
that the calibrated probabilities need no shrinkage and the independence assumption holds.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.91698** on 34,866 held-out validation entities
  (precision 0.95484, recall 0.84472, mean |S| 2.956).

  | country | entities | macro F<sub>0.5</sub> |
  |---|---|---|
  | US | 17,533 | 0.92700 |
  | India | 17,333 | 0.90685 |

  Ablation, all on the same validation entities:

  | configuration | macro F<sub>0.5</sub> | Δ |
  |---|---|---|
  | **TRICE (full)** | **0.91698** | — |
  | best *tuned* global threshold (0.7) | 0.91563 | −0.00135 |
  | without disjointness repair | 0.91586 | −0.00112 |
  | stage-1 only (no graph features) | 0.90610 | **−0.01088** |
  | fixed `top-3` | 0.75239 | −0.16459 |
  | fixed `top-2` | 0.75025 | −0.16673 |
  | fixed `top-4` | 0.70324 | −0.21374 |
  | `top-1` | 0.66739 | −0.24959 |

  Read honestly: the **graph stage is the largest single win (+0.0109)**. The decision layer
  beats a *tuned* threshold by only +0.00135, because the matcher is so accurate that its
  probabilities are strongly bimodal — every threshold from 0.5 to 0.8 lands within 0.003 of
  the others. Its practical value is that it beats fixed-cardinality rules by 0.16–0.25 and
  reaches its operating point **without any validation sweep**, so it cannot be mis-tuned and
  it transfers to France where no labels exist to tune on.

  We also tested a learned missing-mass term (isotonic `Σp → E[|T|]`, fitted on train
  entities). It estimates the aggregate correctly (3.462 predicted vs 3.462 actual) but is
  **score-neutral** (0.91698 vs 0.91695 with the term off): the retrieved probabilities already
  capture `E[|T|]`, so the extra mass only encourages over-emission, which β = 0.5 punishes. We
  report it as a negative result rather than claiming it as a gain.

- **Common false positives (wrong merges):** Only 0.7 % of entities are false merges on a
  singleton and 0.1 % are complete false merges. The residual cases are (a) genuine
  near-duplicates — two branches of one chain sharing a name and a street, where the
  competition margin is near zero by construction; (b) generic names whose distinguishing
  tokens are all low-IDF (`Global Services Pvt Ltd`) combined with a partial address; (c) DBA
  records where the *prefix* company is itself a real entity in Source 1.

- **Common false negatives (missed matches):** The dominant error is **partial** — right
  entity, incomplete set (≈ 42 % of entities). Two distinct causes: roughly 12 % of true links
  are never retrieved by blocking (recall ceiling 0.879), and among retrieved links the
  precision-weighted objective correctly declines to emit genuinely uncertain candidates.
  Complete misses (`missed_all`) run ≈ 3 %, concentrated in records whose address is empty
  **and** whose name was truncated to a single low-IDF token — cases with essentially no signal
  left. India trails the US by 3.0 points of blocking recall (0.864 vs 0.894) and 2.0 points of
  final score, attributable to Devanagari names plus landmark-based addresses
  (`Near SBI ATM`, `choubepada`) that carry no matchable structure.

---

## 6. Conclusion

Reading the metric's algebra rather than accepting it as a black box turned threshold tuning
into an exactly solvable per-entity decision, and verifying the disjointness of the label
structure turned a modelling assumption into an enforceable constraint that proved to be the
single most valuable feature we added. The biggest practical lesson was that our initial
intuitions were repeatedly wrong in measurable ways — singletons were 5.6 % not 28 %, recall
was the binding constraint rather than precision, a truncated stage-2 input cost more than the
features it made room for, and our learned missing-mass term was score-neutral — so every
design choice in the final pipeline is one we measured on held-out data rather than one we
reasoned our way to.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships under `code/business_entity_resolution/`. `src/trice/` imports
nothing from any web framework, so it is a self-contained library.

```
code/business_entity_resolution/
├── README.md                 exact reproduction steps
├── requirements.txt          pinned dependencies
└── src/
    ├── trice/
    │   ├── normalize.py      unicode folding, consonant skeleton, DBA/domain handling,
    │   │                     legal-suffix separation, address componentisation
    │   ├── mine.py           alias mining from ground-truth matched pairs
    │   ├── records.py        streaming TSV → Parquet record store (memory-bounded)
    │   ├── blocking.py       multi-channel inverted index, df caps, union
    │   ├── features.py       55 pairwise features
    │   ├── graph.py          competition normalisation + corroboration + repair
    │   ├── model.py          GBDT wrapper, per-group isotonic calibration
    │   ├── decide.py         Poisson-binomial DP, expected-F_β set selection
    │   ├── evaluate.py       macro F_β, blocking recall, verdict classification
    │   ├── export.py         TSV writers + self-contained format validator
    │   └── pipeline.py       stage orchestration
    └── scripts/
        ├── 02_mine_variants.py    → artifacts/variants.json
        ├── 03_prepare.py          → artifacts/store/*.parquet        (~4 min, 24 M records)
        ├── 04_blocking_eval.py    blocking recall / cost sweep
        ├── 05_train.py            train + score validation           (~29 min)
        ├── 06_infer.py            full test inference → output/*.tsv
        ├── 07_tune_decision.py    decision-layer search on stored probabilities
        ├── test_normalize.py      normalisation vs real match groups
        └── test_decide.py         metric algebra + top-k optimality proofs
```

Entry points to regenerate both output files:

```bash
python scripts/02_mine_variants.py --sample 250000
python scripts/03_prepare.py --workers 13
python scripts/05_train.py --entities 70000 --run-id main
python scripts/07_tune_decision.py --run-id main
python scripts/06_infer.py --run-id main            # writes output/*.tsv
```

We also built a **FastAPI + React analysis workbench** (`backend/`, `frontend/`) over the same
library: blocking recall analysis, feature-importance breakdown, per-country reliability
diagrams, a live decision tuner that re-solves the expected-F objective from stored
probabilities, and a per-entity explorer filterable by error type that plots each entity's
`E[F|k]` curve. It is a development tool, not part of the scored pipeline.

### B. Additional Results

**Correctness tests for the novel claims** (`scripts/test_decide.py`):

```
0. worked example from the problem statement -> F_0.5 = 0.714286   (statement says 0.714)
1. closed form vs precision/recall form, 20000 random set pairs: max diff 2.220e-16
2. Poisson-binomial pmf matches brute force, sums to 1
3. top-k prefix optimality vs exhaustive 2^n search:
     400 random cases, expected-value gap max 6.661e-16, mismatches = 0
4. behaviour:  tiny probs -> k*=0 (abstains);  one @0.55 -> k*=1;  five @0.55 -> k*=5
               0.95+0.45 -> k*=1
5. E[F|0] = Π(1−pᵢ) confirmed for all cases
```

The `five @0.55 → k*=5` versus `one @0.55 → k*=1` contrast is the behaviour no fixed threshold
can reproduce: the `0.25·|T|` term makes an additional prediction cheaper for an entity that
already has several plausible matches.

**Blocking configuration sweep** (US partition, 5,000 sampled entities, full 6.19 M index):

| configuration | macro recall | cand/entity | queries/s | est. full test |
|---|---|---|---|---|
| single blended channel, df cap 40 k | 0.8991 | 30.0 | 175 | 165 min |
| two channels, no skeleton n-grams | 0.8592 | 29.2 | 2,783 | 10 min |
| two channels, n-grams capped at df 1.2 k | 0.8795 | 31.2 | 2,344 | 12 min |
| **production (top_k 30/26, cap 48)** | **0.8943** | 46.9 | — | — |

**Dataset profile.**

| file | rows |
|---|---|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

Country mix — train: US 60.0 %, India 40.0 %. Test Source 1: India 46.8 %, US 38.3 %,
**France 15.0 %**.

**Fair play.** No external database, API, geocoder or internet corpus is used. Every statistic
— IDF weights, token aliases, calibration curves, the |T| prior — is derived from the provided
`train_*.tsv` and `test_*.tsv`. Computing corpus statistics over the provided *test records* is
transductive learning on given data, not external augmentation. Models are gradient-boosted
trees under MIT / BSD-3 licences totalling ~8 × 10⁴ parameters.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
