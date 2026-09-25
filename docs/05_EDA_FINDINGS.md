# EDA findings on the real challenge data

Produced by `scripts/eda_01_profile.py`, `scripts/eda_02_groundtruth.py`,
`scripts/eda_03_pairs.py`.

---

## 1. Scale

| File | Rows | Size |
|---|---|---|
| `train_source1.tsv` | 2,206,821 | 200 MB |
| `train_source2.tsv` | 5,034,616 | 467 MB |
| `train_source3.tsv` | 5,285,603 | 480 MB |
| `train_ground_truth.tsv` | 2,206,821 | 121 MB |
| `test_source1.tsv` | **1,732,544** | 167 MB |
| `test_source2.tsv` | 4,887,273 | 486 MB |
| `test_source3.tsv` | 5,082,316 | 483 MB |

No malformed rows, no missing names. `business_address` is empty for ~3.4 % of
Source 2/3 records (168,967 / 175,916 in train) and never empty in Source 1.

Naive comparison space for test: 1.73 M × 9.97 M ≈ **1.7 × 10¹³ pairs**. Blocking must
achieve a reduction ratio around 1 − 10⁻⁶ while keeping recall high. This is the
dominant engineering constraint.

Hardware available: 16 logical cores, **16.9 GB RAM** — memory, not CPU, is binding.
Record stores must use integer ids and Arrow-backed strings, never Python dicts of
10 M strings.

## 2. Country mix

| Split | US | India | France |
|---|---|---|---|
| train | 60.0 % | 40.0 % | — |
| test S1 | 38.3 % | 46.8 % | **15.0 %** |

France is 259,452 of the 1,732,544 test Source 1 entities — **15 % of the score** rides
on a country with zero labelled examples.

## 3. Ground-truth structure — the three decisive facts

### 3.1 Match sets are *exactly* pairwise disjoint

```
distinct matched ids     : 7,638,365
ids used by >1 S1 entity : 0  (0.0000%)
max reuse of a single id : 1
```

Every Source 2/3 record belongs to **at most one** Source 1 entity, with zero
exceptions across 7.64 M labelled links. The disjointness assumption behind Stage R
(competition normalisation) is not an approximation — it is a hard property of the data,
and it is safe to enforce as a constraint rather than merely encourage as a feature.

Furthermore **73.4 % of Source 2** and **74.6 % of Source 3** records participate in a
match. So the true structure is close to a *near-perfect assignment*: most records on
both sides have exactly one partner. That is far more exploitable than generic ER.

### 3.2 Singletons are rare — 5.58 %, not a third

| \|T\| | count | share | cumulative |
|---|---|---|---|
| 0 | 123,247 | 5.58 % | 5.58 % |
| 1 | 119,157 | 5.40 % | 10.98 % |
| 2 | 375,212 | 17.00 % | 27.99 % |
| 3 | 530,841 | 24.05 % | 52.04 % |
| 4 | 484,115 | 21.94 % | 73.98 % |
| 5 | 321,957 | 14.59 % | 88.57 % |
| 6 | 164,868 | 7.47 % | 96.04 % |
| 7 | 63,968 | 2.90 % | 98.94 % |
| 8+ | 23,456 | 1.06 % | 100 % |

Mean **3.461** matches per entity; maximum 11. Per source, 0–5 S2 records and 0–6 S3
records, with `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` the four most common shapes (43 % of all
entities combined).

**Design consequence:** the earlier plan over-weighted singleton abstention. Only 5.6 %
of the score is singleton credit. The decision layer's real job is choosing the correct
**cardinality** of a multi-match set — which is exactly what the expected-F<sub>0.5</sub>
argmax over `k` does, and it is a *harder* and more valuable job than abstaining. A
`top-1` baseline is capped near 0.40; predicting a fixed 3 is capped well below optimal
because `|T|` varies.

### 3.3 Prior over \|T\| is strong and usable

The distribution above is a genuine prior. Because the metric's denominator contains
`0.25·|T|`, and because `E[|T|]` is well estimated by the candidate-score profile, the
decision rule benefits from this prior directly — it is folded in through the
Poisson-binomial over calibrated probabilities plus the `M` (missed-mass) term.

---

## 4. The observed corruption process

Real examples, Source 1 → its true matches.

```
S1  Alpaugh Golden Vance LLC          7031 Invitational Drive, Fl 0, Saint Louis, MO
S2  Alpaugh  Golden                   7031 INVITATIONAL DRIVE, SAINT LOUIS, MO
S2  Alpaugh G0lden  Vance             7031 INVITATIONAL DRIVE, SAINT LOUIS, MO
```

```
S1  Marnie Baynes Keystone Bnb Inc    1901 Vista Villas Drive, Leander, TX
S2  ... THE MARNIE BAYNES KEYSTONE BNB INC   1901 VISTA VILLAS DR, LEANDER, TX
S2  Marnie Baynes-Keystone Bnb Inc    LEANDER, TX, VISTA VILLAS DRIVE
S3  Marnie Baynes Keystone            1901 Vista Villas Dr, Leander, Texas
S3  Mamie Baynes  Keystone Bnb Inc    1901 Vista Villas Drive, Leander, Texas
S3  Halonex dba Marnie Baynes Keystone Bnb Inc   TX, Leander, 1901 Vista Villas Drive
```

```
S1  Ace Foods Limited                 B-6, Ananthi Apartments, 13/12, Zakariah Colony, Iii Street, Choolaimedu, Chennai - 600 094., Tamil Nadu
S2  एस फूड्स लिमिटेड                    B-6, ANANTHI APARTMENTS, ..., CHENNAI - 600 094., तमिलनाडु
S3  Center Ace Limited                B-6, Ananthi Apartments, ..., Choolaimedu, Chennai - 600 094., TN
S3  Ace                               Tamil Nadu, Choolaimedu, Chennai - 600 094., Ananthi Apartments, 13/12, Zakariah Colony, Iii Street, B-6
```

```
S1  Edwards, Hintze & Dougherty       541 14th Street, Newport, OR
S2  Edwards, Hintze + Dougherty Co    <empty>
S2  edwardshintzedougherty.com        FOURTEENTH STREET, NEWPORT, OR
S2  #edwardshintze                    541 FOURTEENTH SAINT, NEWPORT, OR
S2  Hintze Edwards, Dougherty Services  541 FOURTEENTH ST, NEWPORT, OR
S2  EDWARD5, HINTZE & DOUGHERTY       ...
```

### 4.1 Catalogued operators

**Name**

| Operator | Real examples |
|---|---|
| junk prefix/suffix | `...`, `--`, `##`, `<<`, `#`, `M/s`, leading `THE` |
| legal suffix swap / drop | `Limited`↔`Ltd`, `Private Limited`↔`Pvt Ltd`, `Inc`, `LLC`↔`Llc`, `PC`, `SARL`, `S.A.S`, `LLP`↔`एलएलपी` |
| generic descriptor **added** | `Services`, `Service`, `Center`, `Co`, `Group`, `(Partners)`, `[Services]` appended or prepended |
| **DBA prefix with a foreign name** | `Halonex dba Marnie Baynes…`, `Quocalo Co dba Meta's Foods` — the prefix is an *unrelated* company name |
| **domainification** | `tejrajchits.com`, `edwardshintzedougherty.com`, `wilfordhancock.com` — tokens concatenated, TLD appended |
| token subset / truncation | `Alpaugh Golden Vance LLC` → `Alpaugh Golden`; `Ace Foods Limited` → `Ace` |
| word-order transposition | `Edwards, Hintze & Dougherty` → `Hintze Edwards, Dougherty` |
| punctuation | `&` ↔ `+` ↔ `and`; hyphen inserted; `.` added |
| typo (1–2 chars) | `Marnie`→`Mamie`, `Golden`→`G0lden`, `World`→`Woerd`, `Tbk`→`To,k`, `EDWARDS`→`EDWARD5`, `Culbert`→`CÙLBERT` |
| case | full upper in S2, title in S3 |
| **Devanagari transliteration of the whole name** | `राम मार्केटिंग प्राइवेट लिमिटेड` = Ram Marketing Private Limited |

**Address**

| Operator | Real examples |
|---|---|
| street-type abbrev | `Drive`↔`Dr`, `Avenue`↔`Ave`, `Street`↔`St`, `Road`↔`Rd` |
| **ordinal ↔ word** | `14th` ↔ `FOURTEENTH` |
| **wrong expansion** | `St` → `SAINT` (in `FOURTEENTH SAINT` from `14th St`) |
| state abbrev ↔ full | `TX`↔`Texas`, `ME`↔`Maine`, `IL`↔`Illinois`, `OH`↔`Ohio`, `KA`↔`Karnataka`, `TN`↔`Tamil Nadu`, `UP`↔`Uttar Pradesh` |
| state transliterated | `तमिलनाडु`, `उत्तर प्रदेश` |
| city variants | `Mysore`↔`Mysuru`, `Bangalore`↔`Bengaluru` |
| **component reordering** | `44 Goodrich Avenue, Auburn, ME` → `ME, Auburn, 44 Goodrich Avenue` |
| component drop | city, state, PIN, or house number removed |
| house-number prefix junk | `HN 823`, `H.NO`, `H.no B3/801`, `##84-11`, `Fl 0` |
| digit corruption | `1329`→`132`, `84-11`→`84-11-88`, `613/11`→`13/11` |
| typo in locality | `LUCKNOW`→`LCUKNOW`, `QUINCY`→`QUNCY`, `PARK`→`PAKR` |
| literal null | `<NULL>` |
| empty | ~3.4 % of S2/S3 |

### 4.2 What this implies for features

1. **Rare "core" name tokens survive.** After stripping junk, legal suffixes and generic
   descriptors, the remaining proper-noun tokens are preserved (possibly with a 1–2
   character typo) in essentially every true match. IDF-weighted fuzzy core-token
   overlap is the single strongest signal.
2. **Addresses are near-copies, modulo reordering.** Treating the address as an
   *unordered bag* of canonicalised components — house number, digit signature, street
   core, locality — removes most of the noise. The numeric signature
   (`{7031}`, `{13/12, 600094, B-6}`) is highly discriminative.
3. **Domainified names need de-concatenation.** `edwardshintzedougherty.com` only matches
   if we compare a space-free form of the core name. A `name_nospace` field plus a
   containment test handles it and recovers a whole failure class cheaply.
4. **DBA needs splitting.** `X dba Y` should produce *two* candidate name forms, and the
   right-hand side is the one that matches.
5. **Transliteration is mandatory, not optional.** `unidecode` maps
   `राम मार्केटिंग प्राइवेट लिमिटेड` → `raama maarkettiNg praaivett limitteda`. That is
   *not* equal to `ram marketing private limited`, so naive folding fails. It needs an
   additional **vowel/consonant-class squeeze** (collapse doubled vowels, strip
   `N`/`a` artefacts) to produce a comparable skeleton — see §5.
6. **A single global similarity threshold cannot work** because per-entity difficulty
   varies enormously — compare the near-identical `Alpaugh` group with the
   `Edwards, Hintze & Dougherty` group where one match has an empty address and another
   is a domain string.

---

## 5. Transliteration: `unidecode` is necessary but not sufficient

```
unidecode('राम मार्केटिंग प्राइवेट लिमिटेड') = 'raama maarkettiNg praaivett limitteda'
target                                      = 'ram marketing private limited'

unidecode('मॉडर्न फाइनेंस')                   = 'moNddrn phaaineNs'
target                                      = 'modern finance'

unidecode('आदित्य प्रॉपर्टीज एलएलपी')           = 'aadity proNprttiij elelpii'
target                                      = 'aditya properties llp'
```

So a **skeleton form** is required on top of folding:

- collapse repeated vowels (`aa`→`a`, `ii`→`i`)
- drop the `N` nasal artefact and trailing schwa `a`
- collapse doubled consonants (`tt`→`t`, `ll`→`l`)
- map `ph`→`f`, `v`↔`w`, `z`↔`j`

Applied to both sides this turns `raamaamaarkettiNg` and `ram marketing` into comparable
skeletons. French folding is simpler: `Société Générale` → `Societe Generale`,
`Léarning Àmicale` → `Learning Amicale`, which plain `unidecode` handles correctly.

The skeleton function is country-agnostic and applied to every record, satisfying the
"treat country as an open set" instruction while fixing the Devanagari class.

---

## 6. Revisions to the plan forced by this EDA

| Original assumption | Reality | Change |
|---|---|---|
| Singletons ≈ 28 % of entities, big share of score | **5.58 %** | Decision layer retargeted at set *cardinality*, not abstention. Singleton logic still falls out of `k = 0`. |
| Disjointness is a soft prior worth a feature | **Hard, 0 violations in 7.6 M links** | Promote to an enforced constraint: global one-parent assignment repair, not just a feature. |
| Datasets fit comfortably in memory | 10 M records/side, 16.9 GB RAM | Integer id encoding, Arrow strings, chunked scoring, memory-mapped candidate arrays. |
| Char-ngram TF-IDF ANN over all pairs | 1.7 × 10¹³ pairs | Inverted-index blocking on *rare* keys with posting-list caps; no dense ANN at this scale. |
| Latin-only text normalisation | Devanagari names and state names | Skeleton transliteration stage is mandatory. |
| Hand-coded abbreviation tables | Long tail (`Mysore`↔`Mysuru`, `14th`↔`FOURTEENTH`, Devanagari states) | **Mine variant pairs from ground-truth matched groups** — data-driven, fully within fair-play rules, and covers the tail. |

---

## 7. Blocking calibration results (real data, US partition)

Measured by `scripts/04_blocking_eval.py` on 5,000 sampled training entities against the
full US index (6,186,873 Source 2/3 records).

| config | macro recall | cand/entity | positives | queries/s | est. full test |
|---|---|---|---|---|---|
| single blended channel, df cap 40 k, k=30 | 0.8991 | 30.0 | 10.35 % | 175 | **165 min** |
| two channels, no shingles, df cap 3 k | 0.8592 | 29.2 | 10.02 % | 2,783 | 10 min |
| two channels, shingles capped at df 1.2 k | **0.8795** | 31.2 | 9.60 % | 2,344 | **12 min** |

Three conclusions:

**Per-namespace df caps are worth 13x.** Retrieval cost is `Σ_t df_query(t)·df_index(t)`,
and skeleton n-grams are ~2 orders of magnitude more frequent than core name tokens. One
shared cap either throttles the useful tokens or lets the frequent ones dominate. Capping
`k` (shingles) at df 1,200 while leaving name tokens at 4,000 cut the estimated full-test
query time from 165 min to 12 min for 0.02 recall.

**The two channels are genuinely complementary, not redundant.** Alone they reach only
0.65 (name) and 0.69 (address), but together 0.88. Exclusive true-link attribution:

```
name channel only : 3,551 true links
addr channel only : 4,205 true links
```

Neither channel can be dropped. This validates the multi-channel design over a single
blended vector — the blended index scored *higher alone* (0.899) but cost 13x more, and
the union gets most of the way there for a fraction of the compute.

**Recall is not the binding constraint — precision is.** This is the important one. If
candidate generation retrieves a fraction `r` of an entity's true matches and the matcher
then selects exactly those, the achieved score is

```
F_0.5 = 1.25 · (r·t) / (0.25·t + r·t) = 1.25 r / (0.25 + r)
```

| macro recall `r` | ceiling on macro F<sub>0.5</sub> |
|---|---|
| 0.80 | 0.952 |
| 0.86 | 0.968 |
| **0.88** | **0.973** |
| 0.90 | 0.978 |
| 0.95 | 0.990 |

Because β = 0.5 discounts recall, an extra 2 points of blocking recall is worth ~0.005
F<sub>0.5</sub>, whereas a false merge costs an entire entity. Chasing recall past ~0.88
is therefore a poor use of both compute and candidate slots — every extra candidate is
another chance for the matcher to make a precision error. The production configuration is
fixed at the two-channel shingle-capped setting and effort moves to the matcher and the
decision layer.

Production blocking configuration:

```python
ChannelConfig("name", ("n", "k", "w", "S"), df_cap=4_000,
              ns_caps={"k": 1_200, "w": 200, "S": 200}, top_k=22, min_score=0.06)
ChannelConfig("addr", ("a", "d", "H", "D"), df_cap=6_000,
              ns_caps={"H": 200, "D": 200},   top_k=18, min_score=0.10)
max_candidates = 32
```

---

## 8. First end-to-end run (smoke, 4,000 entities/country)

```
blocking macro recall   India 0.8413   US 0.8792
stage-1 val AUC         0.99935      AP 0.99441
stage-2 val AUC         0.99748      AP 0.98886     <-- WORSE than stage 1
calibration ECE         0.0063 overall, 0.0064 India, 0.0063 US

val macro F0.5
  expected_f                 0.89520   P=0.9347  R=0.8242  mean_k=2.95
  best fixed threshold(0.5)  0.89469   P=0.9333  R=0.8262
  threshold@0.3 ... @0.8     0.89411 ... 0.89451   (flat)
  top1                       0.64999   P=0.9445  R=0.3455
  top2                       0.72200
  top3                       0.71720
  top4                       0.65976
  stage-1 only + expected_f  0.89455
```

### 8.1 The matcher is far more accurate than anticipated

AUC 0.99935 / AP 0.99441 on real held-out data. The normalisation plus IDF-weighted rarity
features separate matches from non-matches almost perfectly. Consequences:

* The probability distribution is strongly **bimodal**, which is why every fixed threshold
  from 0.3 to 0.8 lands within 0.0006 of the others — there is almost nothing in the
  middle to threshold.
* The expected-F<sub>0.5</sub> layer therefore wins by only **+0.0005** over the best
  *tuned* threshold. Reported honestly: against a tuned global threshold the decision layer
  is near-neutral on this dataset, because the probabilities are nearly deterministic.
* Against naive fixed-cardinality rules the gap is enormous (**+0.17 vs top-3, +0.25 vs
  top-1**), and unlike a tuned threshold it needs no validation sweep to find its operating
  point. That is where its practical value sits.

### 8.2 Recall is now the binding constraint, not precision

At P = 0.935, R = 0.824:

| | F<sub>0.5</sub> at P=0.935 |
|---|---|
| R = 0.824 (current) | 0.8949 |
| R = 0.86 | 0.9117 |
| R = 0.90 | 0.9276 |
| R = 0.95 | 0.9465 |

`dF/dR ≈ 0.40`, `dF/dP ≈ 0.75`. Precision is still worth ~2x per unit — but precision is
already 0.935 with little headroom, whereas recall has 0.09 of headroom up to the blocking
ceiling and more beyond it. **This reverses the §7 conclusion.** §7 assumed perfect
precision within the candidate set, which made recall look cheap; with a real matcher at
P = 0.935 the arithmetic favours buying recall.

Action: raise `top_k` (name 22→30, addr 18→26) and `max_candidates` (32→48). The matcher's
accuracy makes extra candidates low-risk — it rejects them reliably, so the cost is compute
rather than precision.

### 8.3 Stage 2 regressed because its input was truncated

Stage 2 was fed only 12 curated raw features plus the graph features, to keep test-set
memory in budget. That truncation cost more than the graph features added
(AP 0.98886 vs 0.99441).

Fix: give stage 2 the **full** feature matrix plus graph features, and make inference
affordable by spilling the pairwise matrix to an on-disk `float32` memmap during pass 1
instead of keeping it in RAM. At 48 candidates the largest partition (India) needs ~8.6 GB
of disk for this — trivially available (117 GB free) and it removes the compromise entirely.

Additionally, stage 2 is now **adopted only if it beats stage 1 on validation average
precision**; the choice is recorded in the model bundle so inference cannot silently use a
worse model.

### 8.4 India lags US by 3.8 points of blocking recall

0.8413 vs 0.8792, and India is 47 % of the test set. Devanagari names and landmark-based
addresses are the likely cause. The address channel already carries these cases; the higher
`top_k` should help most here.
