"""
Candidate generation (blocking) at 10^13-pair scale.

The naive comparison space for the test split is 1.73 M x 9.97 M ~= 1.7e13 pairs, so
blocking must discard ~99.9999 % of it while keeping nearly every true link.

Multi-channel retrieval
-----------------------
A single blended similarity cannot serve this dataset. Some true matches have an intact
address but a destroyed name (Devanagari rendering, ``edwardshintzedougherty.com``, a
truncation down to one token); others have an intact name but an empty or reordered
address (~3.4 % of Source 2/3 rows have no address at all). Blending both into one vector
with fixed weights compromises on both failure modes.

So retrieval runs as **independent channels**, each with its own vocabulary, IDF
weighting, document-frequency cap and top-k, whose results are then unioned:

===========  ==========================================  ==================================
channel      namespaces                                  recovers
===========  ==========================================  ==================================
``name``     ``n`` core token, ``k`` skeleton n-gram,     name overlap, typos,
             ``w`` space-free name, ``S`` whole skeleton  cross-script, domainified names
``addr``     ``a`` address token, ``d`` numeric token,    locality/street overlap, and the
             ``H`` house+token, ``D`` digit signature     "name is unusable" cases
===========  ==========================================  ==================================

Within a channel, records are L2-normalised IDF-weighted sparse vectors, so candidate
generation *and* a meaningful similarity score both fall out of one sparse product
``Q @ Xᵀ``. Its cost is ``sum_t df_query(t) * df_index(t)``, which is why the
document-frequency cap is the primary performance control -- and it doubles as an
automatic stop list, removing ``services``/``colony``/``road`` without a hand-written one.

Both operands are CSR so SciPy takes its efficient SMMP path; the index is therefore built
**already transposed** (an inverted index of shape ``vocab x records``) to avoid a
multi-gigabyte transpose copy.

Partitioning by the ``country`` label cuts the work about threefold. The label is treated
as an opaque partition key discovered from the data, so an unseen value such as ``France``
flows through unchanged -- nothing is special-cased or one-hot encoded.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """One retrieval channel.

    ``df_cap`` is the default document-frequency cap; ``ns_caps`` overrides it per
    namespace. Per-namespace control is necessary rather than cosmetic: skeleton n-grams
    (``k``) are two orders of magnitude more frequent than core name tokens (``n``), and
    since retrieval cost is ``sum_t df_query(t) * df_index(t)``, a single shared cap either
    throttles the useful tokens or lets the frequent ones dominate the run time.
    """

    name: str
    namespaces: Tuple[str, ...]
    df_cap: int = 20_000
    ns_caps: Dict[str, int] = field(default_factory=dict)
    top_k: int = 20
    min_score: float = 0.05
    query_chunk: int = 2_500

    def cap_for(self, token: str) -> int:
        return self.ns_caps.get(token[0], self.df_cap)


@dataclass
class BlockingConfig:
    """Tunable blocking parameters."""

    # ---- token generation -------------------------------------------------------
    use_skeleton_shingles: bool = True
    shingle_size: int = 6
    max_shingles: int = 10

    w_name: float = 1.0
    w_shingle: float = 0.45
    w_nospace: float = 1.6
    w_skel: float = 1.2
    w_addr: float = 0.85
    w_digit: float = 1.0
    w_house_anchor: float = 1.3
    w_digit_sig: float = 1.5

    min_df: int = 1

    # ---- channels ---------------------------------------------------------------
    # top_k / max_candidates are set for recall rather than compute: with the matcher at
    # AUC ~0.999 an extra candidate is very unlikely to be mistaken for a match, so the
    # cost of a wider candidate set is CPU, not precision. See docs/05_EDA_FINDINGS.md §8.2.
    channels: List[ChannelConfig] = field(default_factory=lambda: [
        ChannelConfig("name", ("n", "k", "w", "S"), df_cap=4_000,
                      ns_caps={"k": 1_200, "w": 200, "S": 200},
                      top_k=30, min_score=0.045),
        ChannelConfig("addr", ("a", "d", "H", "D"), df_cap=6_000,
                      ns_caps={"H": 200, "D": 200},
                      top_k=26, min_score=0.08),
    ])

    # ---- union ------------------------------------------------------------------
    max_candidates: int = 48

    def channel(self, name: str) -> ChannelConfig:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)


# --------------------------------------------------------------------------------------
# token generation
# --------------------------------------------------------------------------------------

def _shingles(s: str, n: int, limit: int) -> List[str]:
    if not s:
        return []
    if len(s) <= n:
        return [s]
    out = [s[i:i + n] for i in range(len(s) - n + 1)]
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def record_tokens(name_core: str, name_skel: str, name_nospace: str,
                  addr_alpha: str, addr_digits: str, house: str,
                  cfg: BlockingConfig,
                  namespaces: Iterable[str] | None = None
                  ) -> List[Tuple[str, float]]:
    """Namespaced ``(token, base_weight)`` pairs for one record.

    ``namespaces`` restricts generation to a channel's namespaces, so no work is done for
    tokens the channel would discard anyway.
    """
    ns = None if namespaces is None else set(namespaces)
    out: List[Tuple[str, float]] = []

    def want(p: str) -> bool:
        return ns is None or p in ns

    if want("n"):
        for t in name_core.split():
            out.append(("n" + t, cfg.w_name))

    if want("k") and cfg.use_skeleton_shingles and name_skel:
        for g in _shingles(name_skel, cfg.shingle_size, cfg.max_shingles):
            out.append(("k" + g, cfg.w_shingle))

    if want("w") and len(name_nospace) >= 7:
        out.append(("w" + name_nospace, cfg.w_nospace))
    if want("S") and len(name_skel) >= 6:
        out.append(("S" + name_skel, cfg.w_skel))

    a_toks = addr_alpha.split() if (want("a") or want("H")) else []
    if want("a"):
        for t in a_toks:
            out.append(("a" + t, cfg.w_addr))

    d_toks = ([x for x in addr_digits.split(",") if x]
              if (addr_digits and (want("d") or want("D"))) else [])
    if want("d"):
        for t in d_toks:
            out.append(("d" + t, cfg.w_digit))

    if want("H") and house and a_toks:
        for t in a_toks[:3]:
            out.append((f"H{house}|{t}", cfg.w_house_anchor))
    if want("D") and len(d_toks) >= 2:
        out.append(("D" + ",".join(sorted(set(d_toks))), cfg.w_digit_sig))

    return out


def frame_columns(frame: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """Extract the raw object arrays token generation needs, once per frame."""
    return (
        frame["name_core"].fillna("").to_numpy(dtype=object),
        frame["name_skel"].fillna("").to_numpy(dtype=object),
        frame["name_nospace"].fillna("").to_numpy(dtype=object),
        frame["addr_alpha"].fillna("").to_numpy(dtype=object),
        frame["addr_digits"].fillna("").to_numpy(dtype=object),
        frame["house"].fillna("").to_numpy(dtype=object),
    )


# --------------------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------------------

@dataclass
class Vocabulary:
    """Token -> column id with IDF weights, built from the index side only."""

    ids: Dict[str, int]
    idf: np.ndarray
    n_docs: int

    def __len__(self) -> int:
        return len(self.ids)


def count_df(frames: Sequence[pd.DataFrame], cfg: BlockingConfig,
             namespaces: Iterable[str] | None = None,
             progress=None) -> Tuple[Counter, int]:
    """Document-frequency counter over the index side.

    Kept separate from :func:`vocabulary_from_df` so a parameter sweep can reuse one
    expensive count across several caps.
    """
    df: Counter = Counter()
    n_docs = 0
    for frame in frames:
        cols = frame_columns(frame)
        n_docs += len(frame)
        for i, row in enumerate(zip(*cols)):
            df.update({t for t, _ in record_tokens(*row, cfg, namespaces)})
            if progress is not None and (i & 0x3FFFF) == 0x3FFFF:
                progress(i)
    return df, n_docs


def vocabulary_from_df(df: Counter, n_docs: int, cfg: BlockingConfig,
                       channel: ChannelConfig) -> Vocabulary:
    """Apply the per-namespace document-frequency caps and compute IDF."""
    ids: Dict[str, int] = {}
    idf_vals: List[float] = []
    min_df = cfg.min_df
    for tok, d in df.items():
        if d < min_df or d > channel.cap_for(tok):
            continue
        ids[tok] = len(ids)
        idf_vals.append(math.log((n_docs + 1.0) / (d + 1.0)) + 1.0)
    return Vocabulary(ids=ids, idf=np.asarray(idf_vals, dtype=np.float32), n_docs=n_docs)


# --------------------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------------------

def _triples(frame: pd.DataFrame, vocab: Vocabulary, cfg: BlockingConfig,
             namespaces: Iterable[str] | None, row_offset: int = 0
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(col_ids, row_ids, weights)`` for a record frame, L2-normalised per row."""
    cols = frame_columns(frame)
    vid = vocab.ids
    idf = vocab.idf

    c_acc: List[np.ndarray] = []
    r_acc: List[np.ndarray] = []
    w_acc: List[np.ndarray] = []

    for i, row in enumerate(zip(*cols)):
        best: Dict[int, float] = {}
        for tok, base in record_tokens(*row, cfg, namespaces):
            j = vid.get(tok)
            if j is None:
                continue
            v = base * idf[j]
            if v > best.get(j, 0.0):
                best[j] = v
        if not best:
            continue
        n = len(best)
        cj = np.fromiter(best.keys(), dtype=np.int32, count=n)
        wv = np.fromiter(best.values(), dtype=np.float32, count=n)
        norm = math.sqrt(float((wv * wv).sum()))
        if norm > 0.0:
            wv /= norm
        c_acc.append(cj)
        r_acc.append(np.full(n, i + row_offset, dtype=np.int32))
        w_acc.append(wv)

    if not c_acc:
        z32 = np.zeros(0, dtype=np.int32)
        return z32, z32, np.zeros(0, dtype=np.float32)
    return np.concatenate(c_acc), np.concatenate(r_acc), np.concatenate(w_acc)


@dataclass
class ChannelIndex:
    """Inverted index for one channel: CSR of shape ``(vocab, n_records)``."""

    channel: str
    matrix: sp.csr_matrix
    vocab: Vocabulary
    n_records: int

    def nbytes(self) -> int:
        m = self.matrix
        return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def build_channel_index(frames: Sequence[pd.DataFrame], vocab: Vocabulary,
                        cfg: BlockingConfig, channel: ChannelConfig,
                        block: int = 400_000, progress=None) -> ChannelIndex:
    """Build one channel's inverted index over the concatenated index-side frames."""
    c_parts: List[np.ndarray] = []
    r_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []

    offset = 0
    for frame in frames:
        for start in range(0, len(frame), block):
            chunk = frame.iloc[start:start + block]
            c, r, w = _triples(chunk, vocab, cfg, channel.namespaces,
                               row_offset=offset + start)
            c_parts.append(c)
            r_parts.append(r)
            w_parts.append(w)
            if progress is not None:
                progress(offset + start + len(chunk))
        offset += len(frame)

    cols = np.concatenate(c_parts); c_parts.clear()
    rows = np.concatenate(r_parts); r_parts.clear()
    wts = np.concatenate(w_parts); w_parts.clear()

    n_vocab = max(1, len(vocab))
    coo = sp.coo_matrix((wts, (cols, rows)), shape=(n_vocab, offset), dtype=np.float32)
    del cols, rows, wts
    matrix = coo.tocsr()
    del coo
    return ChannelIndex(channel=channel.name, matrix=matrix, vocab=vocab,
                        n_records=offset)


def query_channel(index: ChannelIndex, frame: pd.DataFrame, cfg: BlockingConfig,
                  channel: ChannelConfig, progress=None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``k`` index records per query row for one channel.

    Returns ``(query_row, index_row, score)`` with ``query_row`` ascending.
    """
    c, r, w = _triples(frame, index.vocab, cfg, channel.namespaces)
    Q = sp.coo_matrix((w, (r, c)), shape=(len(frame), max(1, len(index.vocab))),
                      dtype=np.float32).tocsr()
    XT = index.matrix
    k = channel.top_k
    step = channel.query_chunk

    q_out: List[np.ndarray] = []
    i_out: List[np.ndarray] = []
    s_out: List[np.ndarray] = []

    for start in range(0, Q.shape[0], step):
        stop = min(start + step, Q.shape[0])
        blk = Q[start:stop]
        if blk.nnz == 0:
            if progress is not None:
                progress(stop)
            continue
        R = blk @ XT
        if channel.min_score > 0.0:
            R.data[R.data < channel.min_score] = 0.0
            R.eliminate_zeros()
        indptr, indices, data = R.indptr, R.indices, R.data
        for row in range(R.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                continue
            d = data[lo:hi]
            ix = indices[lo:hi]
            if len(d) > k:
                sel = np.argpartition(-d, k - 1)[:k]
                d, ix = d[sel], ix[sel]
            q_out.append(np.full(len(ix), start + row, dtype=np.int32))
            i_out.append(ix.astype(np.int32))
            s_out.append(d.astype(np.float32))
        del R
        if progress is not None:
            progress(stop)

    if not q_out:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32)
    return np.concatenate(q_out), np.concatenate(i_out), np.concatenate(s_out)


# --------------------------------------------------------------------------------------
# union across channels
# --------------------------------------------------------------------------------------

def _union_block(q: np.ndarray, i: np.ndarray, s: np.ndarray, m: np.ndarray,
                 q_lo: int, q_hi: int, cap: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dedupe and cap one contiguous range of query rows."""
    if len(q) == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8)

    # (query, index) -> one int64 key. A single argsort replaces a two-key lexsort, which
    # roughly halves the peak allocation.
    key = (q.astype(np.int64) - q_lo) * np.int64(1 << 32) + i.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    q, i, s, m = q[order], i[order], s[order], m[order]

    first = np.empty(len(key), dtype=bool)
    first[0] = True
    np.not_equal(key[1:], key[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    del key

    # per (query, index) group: max score, OR'd probe masks
    s = np.maximum.reduceat(s, starts)
    m = np.bitwise_or.reduceat(m, starts)
    q, i = q[starts], i[starts]
    del first, starts, order

    # order by query ascending then score descending, again with one argsort: quantise the
    # score into the low bits so a single integer sort expresses both orderings
    QBITS = np.int64(1 << 22)
    sc = ((1.0 - np.clip(s, 0.0, 1.0)) * float(int(QBITS) - 1)).astype(np.int64)
    rank_key = (q.astype(np.int64) - q_lo) * QBITS + sc
    order = np.argsort(rank_key, kind="stable")
    q, i, s, m = q[order], i[order], s[order], m[order]
    del rank_key, sc, order

    n_local = q_hi - q_lo
    idx = np.arange(n_local)
    g_start = np.searchsorted(q, idx + q_lo, side="left").astype(np.int64)
    g_end = np.searchsorted(q, idx + q_lo, side="right").astype(np.int64)
    counts = g_end - g_start
    rank = np.arange(len(q), dtype=np.int64) - np.repeat(g_start, counts)
    keep = rank < np.repeat(np.minimum(counts, cap), counts)
    return q[keep], i[keep], s[keep], m[keep]


def union_channels(results: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                   n_queries: int, cfg: BlockingConfig, channel_order: Sequence[str],
                   query_block: int = 150_000
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge per-channel results.

    Returns ``(query_row, index_row, score, probe_mask)`` sorted by query then descending
    score. ``score`` is the per-pair maximum across channels; ``probe_mask`` has bit *i*
    set when channel ``channel_order[i]`` retrieved the pair -- a feature the matcher uses,
    and the statistic that shows whether a channel is earning its cost.

    Dedupe and capping are both per-query operations, so the work is done in blocks of
    query rows: the results are exact, but peak memory is a fraction of what sorting all
    ~45 M pairs of the largest partition at once requires.
    """
    if not results:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8)

    bit_of = {name: (1 << k) for k, name in enumerate(channel_order)}
    # each channel's output is already ascending in query row, so slicing by query range
    # is a pair of searchsorted calls
    per_channel = [(r[1], r[2], r[3], bit_of[r[0]]) for r in results]

    out_q: List[np.ndarray] = []
    out_i: List[np.ndarray] = []
    out_s: List[np.ndarray] = []
    out_m: List[np.ndarray] = []

    for q_lo in range(0, n_queries, query_block):
        q_hi = min(q_lo + query_block, n_queries)
        qs, is_, ss, ms = [], [], [], []
        for cq, ci, cs, bit in per_channel:
            a = int(np.searchsorted(cq, q_lo, side="left"))
            b = int(np.searchsorted(cq, q_hi, side="left"))
            if b <= a:
                continue
            qs.append(cq[a:b])
            is_.append(ci[a:b])
            ss.append(cs[a:b])
            ms.append(np.full(b - a, bit, dtype=np.uint8))
        if not qs:
            continue
        bq, bi, bs, bm = _union_block(
            np.concatenate(qs), np.concatenate(is_), np.concatenate(ss),
            np.concatenate(ms), q_lo, q_hi, cfg.max_candidates)
        if len(bq):
            out_q.append(bq)
            out_i.append(bi)
            out_s.append(bs)
            out_m.append(bm)

    if not out_q:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8)
    return (np.concatenate(out_q), np.concatenate(out_i),
            np.concatenate(out_s), np.concatenate(out_m))


# --------------------------------------------------------------------------------------
# labelling helpers
# --------------------------------------------------------------------------------------

def encode_ids(src: np.ndarray, num: np.ndarray) -> np.ndarray:
    """Pack ``(source, numeric id)`` into one sortable int64 code."""
    return src.astype(np.int64) * np.int64(1 << 32) + num.astype(np.int64)


def label_candidates(cand_codes: np.ndarray, truth_codes: np.ndarray) -> np.ndarray:
    """1 where a candidate code appears in the sorted ``truth_codes``."""
    if len(truth_codes) == 0 or len(cand_codes) == 0:
        return np.zeros(len(cand_codes), dtype=np.int8)
    pos = np.searchsorted(truth_codes, cand_codes)
    pos = np.clip(pos, 0, len(truth_codes) - 1)
    return (truth_codes[pos] == cand_codes).astype(np.int8)


def group_bounds(q_row: np.ndarray, n_queries: int) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end offsets of each query's block in a ``q_row``-sorted array."""
    idx = np.arange(n_queries)
    starts = np.searchsorted(q_row, idx, side="left").astype(np.int64)
    ends = np.searchsorted(q_row, idx, side="right").astype(np.int64)
    return starts, ends


def candidate_rank(q_row: np.ndarray, starts: np.ndarray,
                   counts: np.ndarray) -> np.ndarray:
    """Within-entity rank (0-based) for each candidate."""
    return (np.arange(len(q_row), dtype=np.int64) - np.repeat(starts, counts)
            ).astype(np.int32)


__all__ = [
    "BlockingConfig", "ChannelConfig", "Vocabulary", "ChannelIndex",
    "count_df", "vocabulary_from_df", "build_channel_index", "query_channel",
    "union_channels", "record_tokens", "frame_columns",
    "encode_ids", "label_candidates", "group_bounds", "candidate_rank",
]
