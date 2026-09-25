# TRICE — the solution idea

**TRICE** = **T**ripartite message passing · **R**eciprocal competition normalisation ·
**I**n-domain noise replay · **C**ountry-conditional calibration ·
**E**xpected-F<sub>0.5</sub> set selection.

---

## 0. The one-sentence version

Most entity-resolution pipelines end with *"score every pair, threshold, emit"*. TRICE
replaces the final threshold with an **exact Bayes-optimal set decision** derived from
the closed form of the scoring metric, and feeds that decision with probabilities that
have been corrected for two things a pairwise model structurally cannot see: **that
Source 2/3 records compete for a single Source 1 parent**, and **that Source 2 and
Source 3 records corroborate each other**.

---

## 1. Why the obvious pipeline leaves points on the table

The default approach is:

```
block  →  pairwise features  →  GBDT / cross-encoder  →  p(match) > τ  →  emit
```

Three specific losses:

| Loss | Cause |
|---|---|
| **Wrong decision rule.** | A single global `τ` is being used to optimise a metric that is *per-entity, non-linear and size-dependent*. The optimal `τ` genuinely differs entity by entity. |
| **Independence assumption violated.** | Source 1 is deduplicated, so true match sets are pairwise **disjoint** — each S2/S3 record has at most one S1 parent. A pairwise model scores `(a,b)` without knowing that `b` looks even better next to some other `a'`. |
| **Half the graph is ignored.** | `b ∈ S2` and `c ∈ S3` that are obviously the same business corroborate each other. A model that only ever looks at `(S1, S2)` and `(S1, S3)` edges never uses the `S2–S3` edges. |

Plus the zero-shot problem: the matcher and — more damagingly — its **probability
calibration** are fit on US/India and then applied to France. A miscalibrated `p` breaks
any decision rule, optimal or not.

TRICE addresses each of these as an explicit, separable stage.

---

## 2. Stage E first: the decision layer (the core idea)

### 2.1 The metric has a closed form that is linear in TP

From [`01_PROBLEM_STATEMENT.md` §4.1](01_PROBLEM_STATEMENT.md), for predicted set `S`
and true set `T`:

$$
F_{0.5}(S,T)=\frac{1.25\,|S\cap T|}{0.25\,|T| + |S|},\qquad
F_{0.5}(\varnothing,\varnothing)=1
$$

The denominator depends **only on the two set sizes**. This is what makes exact
optimisation possible.

### 2.2 The decision problem

For one Source 1 entity with candidate records `1…n` and calibrated marginal match
probabilities `p₁ … pₙ`, choose `S ⊆ {1…n}` maximising

$$
\mathbb{E}\big[F_{0.5}(S,T)\big]
=\mathbb{E}\left[\frac{1.25\cdot \mathrm{TP}_S}{0.25\,|T| + |S|}\right]
$$

where `TP_S = Σ_{i∈S} Yᵢ`, `|T| = Σ_{i=1}^{n} Yᵢ + M`, `Yᵢ ~ Bernoulli(pᵢ)`, and `M` is
the number of true matches **that blocking never retrieved**.

### 2.3 Two facts that collapse a 2ⁿ search to n+1 evaluations

**Fact 1 — for a fixed size `k = |S|`, the optimal `S` is the top-`k` by probability.**

Note `|T|` does not depend on `S` — it is a property of nature, not of our choice. So
with `k` fixed, the denominator is a random variable independent of *which* `k`
candidates we pick. Maximising `E[F]` therefore reduces to stochastically maximising
`TP_S`, and

$$
F(t{+}1, f{-}1, k) - F(t, f, k) = \frac{1.25}{0.25(t+f)+k} > 0
$$

so swapping any selected candidate for an unselected one with higher probability weakly
increases the objective. The top-`k` prefix of the descending-`p` ordering is optimal.

**Fact 2 — `k` only ranges over `0…n`.** So we evaluate `n+1` candidate sets and take
the argmax. Exactly, not greedily.

### 2.4 Computing `E[F | k]` exactly

Sort so `p₁ ≥ p₂ ≥ … ≥ pₙ`. With `S = {1…k}`:

- `TP ~ PoissonBinomial(p₁…p_k)`
- `FN ~ PoissonBinomial(p_{k+1}…pₙ) + M`, independent of `TP`

Both pmfs come from the standard O(n²) Poisson-binomial DP
(`dp[j] ← dp[j](1−p) + dp[j−1]p`). Then

$$
\mathbb{E}[F\mid k]=\sum_{t=0}^{k}\sum_{f=0}^{n-k}
\Pr[\mathrm{TP}{=}t]\Pr[\mathrm{FN}{=}f]\cdot
\begin{cases}
1 & k=0,\ f=0\\[2pt]
\dfrac{1.25\,t}{0.25(t+f)+k} & \text{otherwise}
\end{cases}
$$

`n` is the per-entity candidate cap (tens, not thousands), so this is microseconds per
entity and fully vectorisable as an outer product.

### 2.5 Why this is the interesting part

**The singleton decision falls out of the same formula.** Setting `k = 0` gives

$$
\mathbb{E}[F\mid k{=}0]=\Pr[|T|=0]=\prod_{i=1}^{n}(1-p_i)\cdot\Pr[M=0]
$$

So "predict nothing" wins exactly when the probability that this entity has *no* match
at all exceeds the best achievable expected score from predicting something. No separate
singleton classifier, no hand-tuned singleton threshold — and singletons are worth a
full 1.0 each under macro-averaging, which is a large slice of the score.

**It produces per-entity adaptive behaviour for free.** The implied threshold moves with
the evidence profile:

| Entity's probability profile | What the rule does |
|---|---|
| one candidate at `p = 0.55`, rest near 0 | often emits it — `E[F|1] ≈ 1.25(0.55)/(0.25·1+1) ≈ 0.55` vs `E[F|0] ≈ 0.45` |
| five candidates at `p = 0.55` | emits **more** of them: the `0.25|T|` term grows, so marginal cost of an extra prediction falls |
| all candidates `p < 0.2` | emits nothing, banking the singleton credit |
| one at `0.95`, one at `0.45` | often emits only the first — adding the second risks halving precision on an entity that is otherwise nearly perfect |

Hand-tuning a threshold can approximate the *average* of these behaviours. It cannot
reproduce them.

**It gives blocking recall a principled home.** `M` — expected unretrieved true matches
— is measured on the validation split as a function of blocking signals (candidate
count, best similarity, country). Including it makes the rule appropriately less eager
to declare a singleton when blocking was likely thin for that entity.

### 2.6 Pseudocode

```python
def choose_set(probs, p_missing):
    p = sorted(probs, reverse=True)
    n = len(p)
    suffix_pb = poisson_binomial_suffixes(p)      # pmf of FN for each k
    best_k, best_ev = 0, -1.0
    sel_pb = [1.0]                                # pmf of TP, grows with k
    for k in range(n + 1):
        if k > 0:
            sel_pb = pb_add(sel_pb, p[k - 1])
        fn_pb = pb_add_bernoulli(suffix_pb[k], p_missing)
        ev = expected_f_beta(sel_pb, fn_pb, k)    # the double sum of §2.4
        if ev > best_ev:
            best_k, best_ev = k, ev
    return top_k_ids(best_k), best_ev
```

Everything upstream of this exists to make `probs` *calibrated*, because the rule is
only optimal if the probabilities are honest. That is stages T, R, I and C.

---

## 3. Stage T — tripartite message passing

Source 2 and Source 3 describe the same businesses, so `S2–S3` edges carry evidence
about `S1` membership. Concretely: `S2-4471` links weakly to `S1-88` (address is
landmark-only), but `S3-9912` links *strongly* to `S1-88` and is near-identical to
`S2-4471`. Transitivity resolves `S2-4471`.

One round of message passing over the tripartite graph, computed on the blocked
candidate graph only:

```
m(a,b) = max over c ∈ cand₃(a)  [ sim₂₃(b,c) · p⁰(a,c) ]        # S3 → S2 support
m(a,c) = max over b ∈ cand₂(a)  [ sim₂₃(b,c) · p⁰(a,b) ]        # S2 → S3 support
```

These messages — along with `sum`, `top-2 mean`, and the count of corroborating
partners above a similarity floor — become **features of a second-stage model**, not a
hand-weighted blend. Stacking keeps the first-stage scores honest and lets the model
learn when corroboration is informative (chains, generic names) versus misleading.

Cost: cheap. `sim₂₃` is only evaluated for `(b, c)` pairs that already share a Source 1
candidate, which the blocking index gives us for free.

---

## 4. Stage R — reciprocal competition normalisation

Source 1 is deduplicated, so **each S2/S3 record has at most one S1 parent**. The true
sets are disjoint. A pairwise probability ignores this entirely.

Build the sparse candidate score matrix and normalise **down the columns** (over the
S1 entities competing for a given S2/S3 record `b`), with a **null "dustbin" column**
so that `b` is allowed to match nothing — the optimal-transport-with-dustbin trick from
graph matching, which is the correct probabilistic projection of the disjointness
constraint:

$$
q(a,b)=\frac{\exp(s(a,b)/\tau)}{\exp(s_\varnothing(b)/\tau)+\sum_{a'\in \mathrm{cand}(b)}\exp(s(a',b)/\tau)}
$$

Note the asymmetry is deliberate and correct: **columns** are constrained (one parent
per S2/S3 record) while **rows** are not (an S1 entity may legitimately have many
matches). This is why a plain symmetric Sinkhorn would be wrong here.

Derived features fed to the second-stage model: `q(a,b)`, the rank of `a` among `b`'s
competitors, the margin `s(a,b) − s(a₂,b)` to the runner-up, and `b`'s competitor count.

The margin feature is the precision workhorse. Two near-duplicate Source 1 entities
(e.g. two genuinely distinct branches of one chain) produce a near-zero margin — exactly
the configuration that causes catastrophic false merges under F<sub>0.5</sub>, and the
model learns to back off there.

Optionally, a final **greedy disjointness repair**: if a record ends up assigned to two
S1 entities, keep the higher `q`. Applied after set selection, guarded by validation.

---

## 5. Stage I — in-domain noise replay (the France problem)

The test set contains France; training does not. External data is banned. But the test
records themselves are *provided data*, so **transductive** use is legitimate — this is
the lever.

**Step 1 — mine a noise-operator inventory from training.** For each ground-truth
matched pair, align tokens (Hungarian assignment on token-level edit distance) and
record what actually happened:

| Operator | Mined example |
|---|---|
| token substitution | `corporation → corp`, `private → pvt`, `road → rd` |
| suffix deletion | drop `limited`, `llc`, `inc` |
| punctuation rewrite | `&` ↔ `and` |
| token transposition | word-order swap, with observed displacement distribution |
| character typo | insert/delete/substitute/transpose, with per-position rates |
| component drop | remove PIN/ZIP, remove state |
| landmark insertion | prepend `near <entity>`, `opp <entity>` |
| transliteration variant | vowel/digraph folding classes |

Each with an empirical firing rate — and, importantly, rates conditioned on country,
because Indian addresses degrade differently from US ones.

**Step 2 — replay them on the unseen country.** Take the French test records, apply
sampled operators, and generate labelled positive pairs. Generate hard negatives by
pulling near-miss records from the same blocking neighbourhood.

**Step 3 — train on the union** of real training pairs plus in-domain synthetic pairs,
with instance weights, and use the synthetic French pairs to **fit the France
calibration curve**.

Why this matters more than it sounds: the second-stage model may rank France pairs
acceptably even without this, but its *probabilities* will be systematically off, and
Stage E consumes probabilities. Calibration, not ranking, is the thing that breaks
zero-shot — and this is the only way to fix it without external data.

**Operator rates are mined from the generic, country-agnostic operator vocabulary and
re-estimated per country from data.** No `if country == "France"` branch anywhere; the
country label is just a grouping key, which also satisfies the explicit
"treat country as an open set" instruction.

---

## 6. Stage C — country-conditional calibration

Isotonic regression per country group, fit on held-out validation folds, with
pooled shrinkage for groups with little data and the synthetic curve for unseen
countries. Reliability diagrams per country are a first-class artefact in the UI,
because a calibration bug here is invisible in AUC and devastating in final score.

---

## 7. Full pipeline

```
                       ┌────────────────────────────────────────────┐
  S1 / S2 / S3  ──────▶│ A. Normalisation                           │
  train + test         │   unicode fold, legal-suffix canon, token  │
                       │   expansion, address componentisation,     │
                       │   transductive IDF over train ∪ test       │
                       └────────────────────┬───────────────────────┘
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │ B. Multi-probe blocking            (recall)│
                       │   char 3-gram TF-IDF ANN                   │
                       │   rare-token inverted index                │
                       │   postal / house-number key                │
                       │   phonetic key                             │
                       │   → union, per-entity cap                  │
                       └────────────────────┬───────────────────────┘
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │ C. Stage-1 pairwise matcher                │
                       │   ~60 similarity features → GBDT           │
                       │   → p⁰(a,b)      ── cheap, high recall     │
                       └────────────────────┬───────────────────────┘
                                            ▼
                    ┌───────────────────────┴────────────────────────┐
                    ▼                                               ▼
       ┌─────────────────────────┐                   ┌──────────────────────────┐
       │ T. tripartite messages  │                   │ R. column-softmax +      │
       │    S2↔S3 corroboration  │                   │    dustbin, rank, margin │
       └────────────┬────────────┘                   └────────────┬─────────────┘
                    └───────────────────┬───────────────────────  ┘
                                        ▼
                       ┌────────────────────────────────────────────┐
                       │ D. Stage-2 stacked matcher                 │
                       │   stage-1 score + graph + competition      │
                       │   features → GBDT → s(a,b)                 │
                       └────────────────────┬───────────────────────┘
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │ C. country-conditional isotonic calibration│
                       │   → p(a,b)   honest probabilities          │
                       └────────────────────┬───────────────────────┘
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │ E. expected-F_0.5 set selection  (exact)   │
                       │   per entity: argmax_k E[F | top-k]        │
                       └────────────────────┬───────────────────────┘
                                            ▼
                            matching_results.tsv + candidate_pairs.tsv
```

`I.` (noise replay) injects extra labelled pairs into the training of `C`/`D` and into
the calibration fit — drawn as a side input rather than a pipeline stage.

### 7.1 Feature groups for the stage-1 matcher

| Group | Features |
|---|---|
| **Name — lexical** | Jaro-Winkler, normalised Levenshtein, token-set ratio, token-sort ratio, partial ratio, LCS ratio |
| **Name — token** | Jaccard, containment, **IDF-weighted soft cosine** (fuzzy token alignment so typos still align), max/mean IDF of shared tokens, count of shared tokens with IDF above a percentile |
| **Name — structural** | acronym match (`IBM` ↔ `International Business Machines`), initials match, legal-suffix agree/conflict, token-count delta, word-order displacement |
| **Address — components** | postal-code exact/prefix match, house-number exact match, street-name similarity, city/state similarity, component-presence overlap mask |
| **Address — lexical** | same lexical battery on the normalised address, plus a landmark-stripped variant |
| **Rarity** | IDF-weighted overlap on address tokens, rarest shared token's IDF, shared-digit-sequence indicator |
| **Cross-field** | name↔address cross-contamination similarity (source 3 sometimes packs the name into the address) |
| **Context** | country agree, both-country-known, candidate rank in blocking, number of blocking probes that fired, blocking score |

The **rarity** group is doing most of the work. A shared `"Vaidyanathan"` is
overwhelming evidence; a shared `"Services"` is nearly none. Transductive IDF over
`train ∪ test` measures this without any external corpus.

### 7.2 Optional stage-1.5 cross-encoder

A small multilingual sentence encoder or cross-encoder (MIT/Apache-2.0, well under the
8B cap) re-scoring only the survivors of the GBDT prune, with its score folded in as a
stage-2 feature. Architecturally isolated behind an interface and **off by default**:
it is a recall/precision refinement on hard pairs, not load-bearing, and the GBDT path
must stay reproducible on CPU with no downloads.

---

## 8. What is genuinely novel here

| | Claim |
|---|---|
| 1 | **Exact expected-F<sub>0.5</sub> set selection.** The metric's closed form makes the Bayes-optimal per-entity set computable in closed form via a Poisson-binomial DP over `n+1` candidate sizes. Replaces threshold tuning with a decision-theoretic optimum, and derives the singleton decision as a special case. |
| 2 | **Disjointness as a column-softmax with a dustbin.** Encodes "Source 1 is deduplicated" as a one-sided normalisation with a null option — the structurally correct projection, and the source of the margin feature that suppresses the false merges F<sub>0.5</sub> punishes hardest. |
| 3 | **S2↔S3 corroboration.** Uses the half of the tripartite graph that pairwise pipelines discard, via message passing on the already-blocked graph at near-zero cost. |
| 4 | **Mined noise operators replayed on the unseen country.** Turns the banned-external-data constraint into a design: learn the corruption process from training pairs, replay it transductively on French test records to obtain in-domain labelled pairs and, crucially, an in-domain **calibration** curve. |
| 5 | **Blocking recall enters the decision rule.** `M`, the expected number of unretrieved true matches, is a term in the objective rather than an offline diagnostic, so blocking quality and the emit/abstain decision are coupled. |

Items 1 and 2 are the ones that should move the leaderboard most, because they attack
precision on singletons and near-duplicates — where macro-averaged F<sub>0.5</sub>
concentrates its penalties.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Independence assumption in the Poisson-binomial is wrong (candidates are correlated) | The rule only needs the *expectation* to be ranked correctly across `k`. Validate `E[F]` against realised F on held-out data; the UI plots predicted-vs-actual per `k`. If badly off, add a shrinkage exponent on `p` fit on validation. |
| Calibration drift on France | Stage I synthetic curve + reliability diagrams per country in the UI; fall back to pooled isotonic if the synthetic fit looks degenerate. |
| Message passing amplifies errors | One round only, messages enter as *features* of a stacked model rather than as score updates; ablation toggle. |
| Blocking misses cap recall | Multi-probe union with four independent key families; recall ceiling and reduction ratio reported per country before any modelling. |
| Overfitting to the public leaderboard | All tuning on internal CV; the decision layer has essentially no free parameters to overfit, which is a side benefit of replacing `τ`. |

---

## 10. Ablation plan

Each row is a toggle in the workbench, scored on the same validation split:

| Configuration | Purpose |
|---|---|
| blocking + stage-1 + global `τ` | baseline |
| + expected-F<sub>0.5</sub> selection | isolate the decision layer (expected: largest single gain) |
| + competition features | isolate Stage R |
| + tripartite messages | isolate Stage T |
| + country-conditional calibration | isolate Stage C |
| + noise-replay augmentation | isolate Stage I (measure on a held-out *country*, simulating France by holding out India) |

The last point deserves emphasis: **simulate the zero-shot setting by training on US
only and validating on India.** That is the only honest way to estimate France
performance before submitting, and it is a first-class evaluation mode in the tool.
