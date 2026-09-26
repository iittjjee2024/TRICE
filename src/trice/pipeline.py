"""
Run orchestration: the stages that turn a record store into predictions.

Kept free of any web-framework import so this package can be lifted verbatim into the
submission zip under ``code/business_entity_resolution/src/``. The FastAPI service and the
command-line scripts are both thin drivers over the functions here.

Stage order mirrors ``docs/02_SOLUTION_IDEA.md``::

    load -> block -> label -> features -> stage-1 -> graph -> stage-2
         -> calibrate -> decide -> (score | export)

Everything is partitioned by country so peak memory stays bounded; the largest partition
(India, test) is ~4.7 M index records.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .blocking import (BlockingConfig, ChannelConfig, build_channel_index,
                       candidate_rank, count_df, encode_ids, group_bounds,
                       label_candidates, query_channel, union_channels,
                       vocabulary_from_df)
from .decide import DecisionConfig, decide_groups
from .features import FEATURE_NAMES, FeatureBuilder, prepare_side
from .graph import (GRAPH_FEATURE_NAMES, GraphConfig, build_graph_features,
                    repair_disjointness)
from .model import GBDT, GroupCalibrator, ModelConfig
from .paths import open_text

STORE_COLUMNS = ["num", "src", "country", "name_core", "name_skel", "name_nospace",
                 "legal", "addr_alpha", "addr_digits", "postal", "house"]

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Everything needed to reproduce a run."""

    store_dir: str = "artifacts/store"
    dataset_dir: str = "student_resource/dataset"
    countries: List[str] | None = None            # None = all found in the split

    # training sample sizes (per country); the full split is used for inference
    train_entities_per_country: int = 120_000
    val_fraction: float = 0.2
    holdout_country: str | None = None            # zero-shot simulation
    seed: int = 20260925

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    stage2: bool = True
    calibrate_per_country: bool = True
    disjointness_repair: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------------------
# candidate container
# --------------------------------------------------------------------------------------

@dataclass
class Candidates:
    """Flat candidate-pair arrays for one country partition."""

    q_row: np.ndarray            # int32, index into the query frame
    c_row: np.ndarray            # int32, index into the concatenated index frames
    score: np.ndarray            # float32, blocking similarity
    probe: np.ndarray            # uint8, channel bitmask
    rank: np.ndarray             # int32, within-entity rank by blocking score
    starts: np.ndarray           # int64, per-query slice start
    ends: np.ndarray             # int64, per-query slice end
    labels: np.ndarray | None = None      # int8, 1 = true match (train only)

    def __len__(self) -> int:
        return len(self.q_row)

    @property
    def counts(self) -> np.ndarray:
        return self.ends - self.starts


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_partition(store_dir: str, split: str, country: str | None,
                   columns: Sequence[str] = tuple(STORE_COLUMNS)
                   ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Load one country partition: the Source-1 frame and the Source-2/3 index frames."""
    cols = list(columns)
    frames: List[pd.DataFrame] = []
    s1 = pd.read_parquet(os.path.join(store_dir, f"{split}_source1.parquet"),
                         columns=cols)
    if country is not None:
        s1 = s1[s1["country"] == country]
    s1 = s1.reset_index(drop=True)
    for src in (2, 3):
        f = pd.read_parquet(os.path.join(store_dir, f"{split}_source{src}.parquet"),
                            columns=cols)
        if country is not None:
            f = f[f["country"] == country]
        frames.append(f.reset_index(drop=True))
    return s1, frames


def split_countries(store_dir: str, split: str) -> List[str]:
    s1 = pd.read_parquet(os.path.join(store_dir, f"{split}_source1.parquet"),
                         columns=["country"])
    return sorted(str(c) for c in s1["country"].dropna().unique())


def load_ground_truth_subset(path: str, wanted: set[str]) -> Dict[str, List[str]]:
    """Read only the ground-truth rows we need, streaming (the file is 121 MB)."""
    out: Dict[str, List[str]] = {}
    with open_text(path) as fh:
        next(fh)
        for line in fh:
            s1, _, rest = line.partition("\t")
            if s1 in wanted:
                rest = rest.rstrip("\n")
                out[s1] = rest.split(",") if rest else []
                if len(out) == len(wanted):
                    break
    return out


def truth_arrays(q_ids: Sequence[str], truth: Dict[str, List[str]]
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Per-query sorted truth codes plus the full true-set size.

    ``truth_sizes`` is the *full* size including links blocking never retrieved, which is
    what keeps recall honest.
    """
    codes: List[np.ndarray] = []
    sizes = np.zeros(len(q_ids), dtype=np.int32)
    for i, qid in enumerate(q_ids):
        ids = truth.get(qid, [])
        sizes[i] = len(ids)
        if not ids:
            codes.append(np.zeros(0, dtype=np.int64))
            continue
        src = np.array([2 if x[1] == "2" else 3 for x in ids], dtype=np.int8)
        num = np.array([int(x[3:]) for x in ids], dtype=np.int64)
        codes.append(np.sort(encode_ids(src, num)))
    return codes, sizes


# --------------------------------------------------------------------------------------
# blocking stage
# --------------------------------------------------------------------------------------

def generate_candidates(queries: pd.DataFrame, idx_frames: Sequence[pd.DataFrame],
                        cfg: BlockingConfig, log: Logger = _noop
                        ) -> Tuple[Candidates, Dict[str, dict]]:
    """Run every blocking channel over one partition and union the results."""
    stats: Dict[str, dict] = {}
    results = []
    for spec in cfg.channels:
        t0 = time.time()
        df, n_docs = count_df(idx_frames, cfg, spec.namespaces)
        vocab = vocabulary_from_df(df, n_docs, cfg, spec)
        del df
        t_vocab = time.time() - t0

        t0 = time.time()
        index = build_channel_index(idx_frames, vocab, cfg, spec)
        t_build = time.time() - t0

        t0 = time.time()
        qr, ir, sc = query_channel(index, queries, cfg, spec)
        t_query = time.time() - t0

        stats[spec.name] = {
            "vocab": len(vocab), "nnz": int(index.matrix.nnz),
            "index_gb": round(index.nbytes() / 1e9, 3),
            "pairs": int(len(qr)),
            "seconds": {"vocab": round(t_vocab, 1), "build": round(t_build, 1),
                        "query": round(t_query, 1)},
        }
        log(f"    channel {spec.name}: vocab={len(vocab):,} nnz={index.matrix.nnz:,} "
            f"pairs={len(qr):,} "
            f"(vocab {t_vocab:.0f}s build {t_build:.0f}s query {t_query:.0f}s)")
        results.append((spec.name, qr, ir, sc))
        del index, vocab

    n_q = len(queries)
    q_row, c_row, score, probe = union_channels(
        results, n_q, cfg, [c.name for c in cfg.channels])
    del results

    starts, ends = group_bounds(q_row, n_q)
    rank = candidate_rank(q_row, starts, ends - starts)
    return Candidates(q_row=q_row, c_row=c_row, score=score, probe=probe,
                      rank=rank, starts=starts, ends=ends), stats


def attach_labels(cand: Candidates, idx_src: np.ndarray, idx_num: np.ndarray,
                  truth_codes: Sequence[np.ndarray]) -> None:
    """Fill ``cand.labels`` from per-query sorted truth codes."""
    codes = encode_ids(idx_src[cand.c_row], idx_num[cand.c_row])
    labels = np.zeros(len(cand), dtype=np.int8)
    for g in range(len(cand.starts)):
        st, en = int(cand.starts[g]), int(cand.ends[g])
        if en > st:
            labels[st:en] = label_candidates(codes[st:en], truth_codes[g])
    cand.labels = labels


# --------------------------------------------------------------------------------------
# feature stage
# --------------------------------------------------------------------------------------

def compute_idf(frames: Sequence[pd.DataFrame]) -> Dict[str, float]:
    """Namespaced IDF over the partition, used by the pairwise features.

    Transductive corpus statistics over the provided files -- not external data.
    """
    from collections import Counter
    import math

    df: Counter = Counter()
    n_docs = 0
    for f in frames:
        n_docs += len(f)
        for s in f["name_core"].fillna("").to_numpy(dtype=object):
            if s:
                df.update({"n" + t for t in s.split()})
        for s in f["addr_alpha"].fillna("").to_numpy(dtype=object):
            if s:
                df.update({"a" + t for t in s.split()})
        for s in f["addr_digits"].fillna("").to_numpy(dtype=object):
            if s:
                df.update({"d" + t for t in s.split(",") if t})
    return {t: math.log((n_docs + 1.0) / (d + 1.0)) + 1.0 for t, d in df.items()}


def build_feature_matrix(cand: Candidates, q_side: Dict[str, np.ndarray],
                         c_side: Dict[str, np.ndarray], idf: Dict[str, float],
                         chunk: int = 1_000_000, log: Logger = _noop) -> np.ndarray:
    """Pairwise design matrix, built in chunks so memory stays flat."""
    fb = FeatureBuilder(idf)
    n = len(cand)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    counts = np.repeat(cand.counts, cand.counts).astype(np.float32)
    t0 = time.time()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        X[start:stop] = fb.build(
            q_side, c_side,
            cand.q_row[start:stop], cand.c_row[start:stop],
            cand.score[start:stop], cand.probe[start:stop],
            cand.rank[start:stop].astype(np.float32), counts[start:stop])
        el = time.time() - t0
        log(f"      features {stop:,}/{n:,} ({stop / max(el, 1e-9):,.0f} pairs/s)")
    return X


def compact_candidates(cand: Candidates) -> np.ndarray:
    """Remap ``cand.c_row`` onto only the index records actually referenced.

    Returns the array of original index rows, so the caller can subset the index frame
    with ``frame.iloc[referenced]``. This is a large memory saving, not a micro-
    optimisation: a partition holds up to 6.2 M index records but a training sample
    references only a small fraction of them, and :func:`trice.features.prepare_side`
    allocates per-record token structures for everything it is handed.
    """
    referenced, inverse = np.unique(cand.c_row, return_inverse=True)
    cand.c_row = inverse.astype(np.int32)
    return referenced


def hash_codes(values: np.ndarray) -> np.ndarray:
    """Map a string column to non-zero int64 codes ('' -> 0)."""
    codes, _ = pd.factorize(pd.Series(values), use_na_sentinel=False)
    codes = codes.astype(np.int64) + 1
    empty = np.array([v == "" or v is None for v in values], dtype=bool)
    codes[empty] = 0
    return codes


def graph_matrix(cand: Candidates, p1: np.ndarray, idx_src: np.ndarray,
                 postal_code: np.ndarray, digit_code: np.ndarray,
                 skel_code: np.ndarray, cfg: GraphConfig) -> np.ndarray:
    return build_graph_features(
        cand.q_row, cand.c_row, p1, idx_src[cand.c_row],
        postal_code[cand.c_row], digit_code[cand.c_row], skel_code[cand.c_row], cfg)


# --------------------------------------------------------------------------------------
# decision stage
# --------------------------------------------------------------------------------------

def apply_decision(cand: Candidates, probs: np.ndarray, cfg: DecisionConfig,
                   missing_mass: np.ndarray | None = None,
                   disjointness_repair: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the expected-F set selection, then optionally enforce disjointness."""
    mask, k_star, exp_f = decide_groups(probs, cand.starts, cand.ends, cfg,
                                        missing_mass=missing_mass)
    if disjointness_repair:
        mask = repair_disjointness(cand.q_row, cand.c_row, mask, probs)
    return mask, k_star, exp_f


def estimate_missing_mass(counts: np.ndarray, p_sum: np.ndarray,
                          recall_ceiling: float) -> np.ndarray:
    """Heuristic fallback: expected true matches that blocking never retrieved.

    If blocking recovers a fraction ``r`` of true links, then for an entity whose retrieved
    candidates carry ``sum(p)`` expected true matches, roughly ``sum(p)·(1-r)/r`` more are
    missing. Cheap, but it applies the *average* recall to every entity and so badly
    underestimates the missing mass for entities where blocking did poorly.
    :class:`MissingMassEstimator` is the fitted replacement; this remains as a baseline for
    the ablation.
    """
    r = float(np.clip(recall_ceiling, 1e-3, 1.0))
    factor = (1.0 - r) / r
    out = (p_sum * factor).astype(np.float64)
    out[counts == 0] = np.maximum(out[counts == 0], factor)
    return np.clip(out, 0.0, 8.0)


class MissingMassEstimator:
    """Learned estimate of ``E[|T|]`` per entity, hence of the unretrieved true mass.

    Why this matters. ``E[F | k]`` is driven by the ``0.25·|T|`` term in the denominator,
    and the Poisson-binomial over *retrieved* candidates only ever sees the true matches
    blocking actually found. For an entity with four true matches of which one was
    retrieved, the rule believes ``|T| ≈ 1``, concludes that emitting that single candidate
    scores ~1.0, and emits one -- while the realised score is
    ``1.25·1/(0.25·4 + 1) = 0.625``. Error analysis showed this is the dominant failure
    mode: ~42 % of entities land in the "partial" bucket.

    Supplying an honest ``|T|`` estimate makes the rule aware that more matches probably
    exist, which lowers the relative cost of an extra prediction and pushes it to emit more.

    The estimator is a one-dimensional isotonic regression from ``Σp`` (the expected number
    of retrieved true matches) to the observed ``|T|``, fitted on **training** entities and
    binned by candidate count so that thin-candidate entities get their own curve. Monotone
    by construction, so it cannot invert the ordering, and it has effectively no capacity to
    overfit.
    """

    def __init__(self, count_bins: Sequence[int] = (1, 8, 20, 36, 10_000)) -> None:
        self.count_bins = list(count_bins)
        self.models: Dict[int, object] = {}
        self.global_model: object | None = None
        self.empty_mean: float = 0.0

    def _bin(self, counts: np.ndarray) -> np.ndarray:
        return np.digitize(counts, self.count_bins)

    def fit(self, counts: np.ndarray, p_sum: np.ndarray,
            truth_sizes: np.ndarray) -> "MissingMassEstimator":
        from sklearn.isotonic import IsotonicRegression

        has_cand = counts > 0
        self.empty_mean = float(truth_sizes[~has_cand].mean()) if (~has_cand).any() else 0.0

        self.global_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=12.0)
        if has_cand.any():
            self.global_model.fit(p_sum[has_cand], truth_sizes[has_cand])

        bins = self._bin(counts)
        for b in np.unique(bins[has_cand]):
            m = has_cand & (bins == b)
            if int(m.sum()) < 2_000:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=12.0)
            iso.fit(p_sum[m], truth_sizes[m])
            self.models[int(b)] = iso
        return self

    def expected_truth(self, counts: np.ndarray, p_sum: np.ndarray) -> np.ndarray:
        out = np.zeros(len(counts), dtype=np.float64)
        has_cand = counts > 0
        if has_cand.any() and self.global_model is not None:
            out[has_cand] = self.global_model.predict(p_sum[has_cand])
        bins = self._bin(counts)
        for b, iso in self.models.items():
            m = has_cand & (bins == b)
            if m.any():
                out[m] = iso.predict(p_sum[m])
        out[~has_cand] = self.empty_mean
        return out

    def missing_mass(self, counts: np.ndarray, p_sum: np.ndarray,
                     scale: float = 1.0) -> np.ndarray:
        """``max(0, E[|T|] − Σp) · scale``, clipped to a sane range."""
        expected = self.expected_truth(counts, p_sum)
        miss = np.maximum(expected - p_sum, 0.0) * float(scale)
        return np.clip(miss, 0.0, 10.0)


__all__ = [
    "RunConfig", "Candidates", "STORE_COLUMNS",
    "load_partition", "split_countries", "load_ground_truth_subset", "truth_arrays",
    "generate_candidates", "attach_labels", "compute_idf", "build_feature_matrix",
    "hash_codes", "graph_matrix", "apply_decision", "estimate_missing_mass",
    "compact_candidates", "MissingMassEstimator",
]
