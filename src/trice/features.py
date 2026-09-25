"""
Pairwise feature extraction.

The feature set is driven directly by the corruption operators catalogued in
``docs/05_EDA_FINDINGS.md``. Design priorities, in order:

1.  **Rarity beats similarity.** A shared ``vaidyanathan`` is near-proof; a shared
    ``services`` is worth nothing. IDF-weighted overlap statistics computed
    transductively over the provided corpus carry most of the signal, so they get the
    richest treatment.
2.  **Order-free address comparison.** Component reordering is one of the most common
    corruptions, so every address feature treats the address as a bag.
3.  **Cheap before expensive.** Features are computed on ``numpy`` arrays over a whole
    candidate chunk. The ``rapidfuzz`` batch helpers release the GIL and are the only
    per-pair string work; everything else is set algebra on pre-split token lists.

``FEATURE_NAMES`` is the authoritative column order for the design matrix.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import JaroWinkler as _JW
    _HAVE_RF = True
except Exception:                                           # pragma: no cover
    _HAVE_RF = False


FEATURE_NAMES: Tuple[str, ...] = (
    # ---- blocking context -------------------------------------------------------
    "block_score",
    "probe_vector",
    "probe_anchor",
    "cand_rank",
    "cand_count",
    # ---- name: token / rarity ---------------------------------------------------
    "name_jaccard",
    "name_containment",
    "name_idf_cosine",
    "name_idf_containment",
    "name_shared_max_idf",
    "name_shared_sum_idf",
    "name_unshared_max_idf_q",
    "name_unshared_max_idf_c",
    "name_n_shared",
    "name_tok_count_q",
    "name_tok_count_c",
    "name_tok_count_diff",
    # ---- name: fuzzy string -----------------------------------------------------
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "name_jaro",
    # ---- name: skeleton / cross-script -----------------------------------------
    "skel_ratio",
    "skel_prefix_eq",
    "skel_equal",
    # ---- name: structural -------------------------------------------------------
    "nospace_equal",
    "nospace_containment",
    "nospace_ratio",
    "acronym_match",
    "legal_agree",
    "legal_conflict",
    "legal_both_empty",
    "best_token_pair_ratio",
    "worst_shared_token_ratio",
    # ---- address: bag -----------------------------------------------------------
    "addr_jaccard",
    "addr_containment",
    "addr_idf_cosine",
    "addr_shared_max_idf",
    "addr_n_shared",
    "addr_ratio",
    "addr_token_set",
    # ---- address: numeric / anchors ---------------------------------------------
    "digit_jaccard",
    "digit_containment",
    "digit_n_shared",
    "house_equal",
    "house_prefix_eq",
    "postal_equal",
    "postal_prefix3",
    "both_have_postal",
    # ---- missingness ------------------------------------------------------------
    "addr_empty_q",
    "addr_empty_c",
    "addr_empty_any",
    # ---- cross-field ------------------------------------------------------------
    "cross_name_in_addr",
    "cross_addr_in_name",
    # ---- country ----------------------------------------------------------------
    "country_equal",
)

N_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}

# ---------------------------------------------------------------------------------------
# Stage-2 input contract
# ---------------------------------------------------------------------------------------
# The stacked stage-2 model sees the graph features plus only this curated slice of the
# raw pairwise features -- not the full matrix.
#
# This is a hard memory constraint, not a modelling preference. Test inference produces
# ~55 M candidate pairs; carrying all 55 raw features at float32 costs ~12 GB, which does
# not fit alongside everything else in 16.9 GB. Graph features need globally-computed
# stage-1 probabilities, so they cannot be produced in the same streaming pass that builds
# the raw features -- meaning the raw features would have to be kept or recomputed.
# Carrying 12 columns instead of 55 costs ~2.6 GB and makes the two-pass structure fit.
#
# The columns are the strongest and least redundant signals per feature family.
STAGE2_RAW_FEATURES: Tuple[str, ...] = (
    "block_score",
    "name_idf_cosine",
    "name_idf_containment",
    "name_shared_max_idf",
    "name_token_set",
    "nospace_equal",
    "skel_equal",
    "addr_idf_cosine",
    "digit_jaccard",
    "house_equal",
    "addr_empty_any",
    "legal_conflict",
)

STAGE2_RAW_INDEX: Tuple[int, ...] = tuple(FEATURE_INDEX[n] for n in STAGE2_RAW_FEATURES)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _acronym(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens if t)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _fuzz.ratio(a, b) * 0.01 if _HAVE_RF else float(a == b)


def _token_set(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _fuzz.token_set_ratio(a, b) * 0.01 if _HAVE_RF else float(a == b)


def _token_sort(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _fuzz.token_sort_ratio(a, b) * 0.01 if _HAVE_RF else float(a == b)


def _partial(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _fuzz.partial_ratio(a, b) * 0.01 if _HAVE_RF else float(a == b)


def _jaro(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _JW.similarity(a, b) if _HAVE_RF else float(a == b)


def _best_token_pair_ratio(qt: Sequence[str], ct: Sequence[str]) -> Tuple[float, float]:
    """Greedy fuzzy token alignment.

    Returns ``(best, worst_of_best)``: the best single-token similarity found, and the
    weakest link among each query token's best partner. The second value is what exposes
    a pair whose tokens *all* align loosely -- the signature of a false merge between two
    businesses with generic names.
    """
    if not qt or not ct or not _HAVE_RF:
        return 0.0, 0.0
    best_overall = 0.0
    worst = 1.0
    for a in qt:
        local = 0.0
        for b in ct:
            r = _fuzz.ratio(a, b) * 0.01
            if r > local:
                local = r
                if local >= 1.0:
                    break
        best_overall = max(best_overall, local)
        worst = min(worst, local)
    return best_overall, worst


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------

class FeatureBuilder:
    """Builds the pairwise design matrix for candidate pairs.

    ``idf`` maps a namespaced token (``'n'`` + name token, ``'a'`` + address token,
    ``'d'`` + digit token) to its inverse document frequency, computed over the union of
    the split's records -- a transductive corpus statistic, not external data.
    """

    def __init__(self, idf: Dict[str, float], default_idf: float = 1.0) -> None:
        self.idf = idf
        self.default_idf = default_idf

    def _tok_idf(self, prefix: str, tok: str) -> float:
        return self.idf.get(prefix + tok, self.default_idf)

    def build(self,
              q_rec: Dict[str, np.ndarray],
              c_rec: Dict[str, np.ndarray],
              q_idx: np.ndarray,
              c_idx: np.ndarray,
              block_score: np.ndarray,
              probe_mask: np.ndarray,
              cand_rank: np.ndarray,
              cand_count: np.ndarray) -> np.ndarray:
        """Compute the ``(n_pairs, N_FEATURES)`` float32 design matrix.

        ``q_rec`` / ``c_rec`` are column dicts of *pre-split* record fields (see
        :func:`prepare_side`), indexed by ``q_idx`` / ``c_idx``.
        """
        n = len(q_idx)
        X = np.zeros((n, N_FEATURES), dtype=np.float32)

        qn_t = q_rec["name_tokens"]; cn_t = c_rec["name_tokens"]
        qa_t = q_rec["addr_tokens"]; ca_t = c_rec["addr_tokens"]
        qd_t = q_rec["digit_tokens"]; cd_t = c_rec["digit_tokens"]
        qn_s = q_rec["name_core"]; cn_s = c_rec["name_core"]
        qa_s = q_rec["addr_alpha"]; ca_s = c_rec["addr_alpha"]
        qk = q_rec["name_skel"]; ck = c_rec["name_skel"]
        qw = q_rec["name_nospace"]; cw = c_rec["name_nospace"]
        ql = q_rec["legal_set"]; cl = c_rec["legal_set"]
        qh = q_rec["house"]; ch = c_rec["house"]
        qp = q_rec["postal"]; cp = c_rec["postal"]
        qc = q_rec["country"]; cc = c_rec["country"]

        F = FEATURE_INDEX
        idf_get = self.idf.get
        d_idf = self.default_idf

        X[:, F["block_score"]] = block_score
        X[:, F["probe_vector"]] = (probe_mask & 1) > 0
        X[:, F["probe_anchor"]] = (probe_mask & 2) > 0
        X[:, F["cand_rank"]] = cand_rank
        X[:, F["cand_count"]] = cand_count

        for r in range(n):
            qi = q_idx[r]
            ci = c_idx[r]

            # ---------------- name tokens / rarity ----------------
            a_tok: List[str] = qn_t[qi]
            b_tok: List[str] = cn_t[ci]
            sa, sb = set(a_tok), set(b_tok)
            shared = sa & sb
            union = sa | sb

            wa = {t: idf_get("n" + t, d_idf) for t in sa}
            wb = {t: idf_get("n" + t, d_idf) for t in sb}
            sum_sh = sum(wa[t] for t in shared)
            norm_a = sum(v * v for v in wa.values()) ** 0.5
            norm_b = sum(v * v for v in wb.values()) ** 0.5
            dot = sum(wa[t] * wb[t] for t in shared)

            X[r, F["name_jaccard"]] = _safe_div(len(shared), len(union))
            X[r, F["name_containment"]] = _safe_div(len(shared), min(len(sa), len(sb)) or 1)
            X[r, F["name_idf_cosine"]] = _safe_div(dot, norm_a * norm_b)
            X[r, F["name_idf_containment"]] = _safe_div(
                sum_sh, min(sum(wa.values()), sum(wb.values())) or 1.0)
            X[r, F["name_shared_max_idf"]] = max((wa[t] for t in shared), default=0.0)
            X[r, F["name_shared_sum_idf"]] = sum_sh
            X[r, F["name_unshared_max_idf_q"]] = max((wa[t] for t in sa - shared),
                                                     default=0.0)
            X[r, F["name_unshared_max_idf_c"]] = max((wb[t] for t in sb - shared),
                                                     default=0.0)
            X[r, F["name_n_shared"]] = len(shared)
            X[r, F["name_tok_count_q"]] = len(sa)
            X[r, F["name_tok_count_c"]] = len(sb)
            X[r, F["name_tok_count_diff"]] = abs(len(sa) - len(sb))

            # ---------------- name fuzzy ----------------
            a_s, b_s = qn_s[qi], cn_s[ci]
            X[r, F["name_ratio"]] = _ratio(a_s, b_s)
            X[r, F["name_token_set"]] = _token_set(a_s, b_s)
            X[r, F["name_token_sort"]] = _token_sort(a_s, b_s)
            X[r, F["name_partial"]] = _partial(a_s, b_s)
            X[r, F["name_jaro"]] = _jaro(a_s, b_s)

            # ---------------- skeleton ----------------
            a_k, b_k = qk[qi], ck[ci]
            X[r, F["skel_ratio"]] = _ratio(a_k, b_k)
            X[r, F["skel_equal"]] = float(bool(a_k) and a_k == b_k)
            if a_k and b_k:
                p = min(6, len(a_k), len(b_k))
                X[r, F["skel_prefix_eq"]] = float(a_k[:p] == b_k[:p])

            # ---------------- structural ----------------
            a_w, b_w = qw[qi], cw[ci]
            X[r, F["nospace_equal"]] = float(bool(a_w) and a_w == b_w)
            if a_w and b_w:
                short, long_ = (a_w, b_w) if len(a_w) <= len(b_w) else (b_w, a_w)
                X[r, F["nospace_containment"]] = float(short in long_)
                X[r, F["nospace_ratio"]] = _ratio(a_w, b_w)
            if a_tok and b_tok:
                X[r, F["acronym_match"]] = float(
                    _acronym(a_tok) == b_w or _acronym(b_tok) == a_w
                    or _acronym(a_tok) == _acronym(b_tok))

            la, lb = int(ql[qi]), int(cl[ci])
            X[r, F["legal_agree"]] = float((la & lb) != 0)
            X[r, F["legal_conflict"]] = float(la != 0 and lb != 0 and (la & lb) == 0)
            X[r, F["legal_both_empty"]] = float(la == 0 and lb == 0)

            best, worst = _best_token_pair_ratio(a_tok, b_tok)
            X[r, F["best_token_pair_ratio"]] = best
            X[r, F["worst_shared_token_ratio"]] = worst

            # ---------------- address bag ----------------
            aa: List[str] = qa_t[qi]
            ba: List[str] = ca_t[ci]
            s_aa, s_ba = set(aa), set(ba)
            sh_a = s_aa & s_ba
            un_a = s_aa | s_ba
            X[r, F["addr_jaccard"]] = _safe_div(len(sh_a), len(un_a))
            X[r, F["addr_containment"]] = _safe_div(
                len(sh_a), min(len(s_aa), len(s_ba)) or 1)
            X[r, F["addr_n_shared"]] = len(sh_a)
            if sh_a:
                wsa = {t: idf_get("a" + t, d_idf) for t in s_aa}
                wsb = {t: idf_get("a" + t, d_idf) for t in s_ba}
                na = sum(v * v for v in wsa.values()) ** 0.5
                nb = sum(v * v for v in wsb.values()) ** 0.5
                dt = sum(wsa[t] * wsb[t] for t in sh_a)
                X[r, F["addr_idf_cosine"]] = _safe_div(dt, na * nb)
                X[r, F["addr_shared_max_idf"]] = max(wsa[t] for t in sh_a)

            a_as, b_as = qa_s[qi], ca_s[ci]
            X[r, F["addr_ratio"]] = _ratio(a_as, b_as)
            X[r, F["addr_token_set"]] = _token_set(a_as, b_as)

            # ---------------- digits / anchors ----------------
            ad: List[str] = qd_t[qi]
            bd: List[str] = cd_t[ci]
            s_ad, s_bd = set(ad), set(bd)
            sh_d = s_ad & s_bd
            X[r, F["digit_jaccard"]] = _safe_div(len(sh_d), len(s_ad | s_bd))
            X[r, F["digit_containment"]] = _safe_div(
                len(sh_d), min(len(s_ad), len(s_bd)) or 1)
            X[r, F["digit_n_shared"]] = len(sh_d)

            h1, h2 = qh[qi], ch[ci]
            X[r, F["house_equal"]] = float(bool(h1) and h1 == h2)
            if h1 and h2:
                X[r, F["house_prefix_eq"]] = float(h1[:2] == h2[:2])
            p1, p2 = qp[qi], cp[ci]
            X[r, F["postal_equal"]] = float(bool(p1) and p1 == p2)
            X[r, F["both_have_postal"]] = float(bool(p1) and bool(p2))
            if p1 and p2:
                X[r, F["postal_prefix3"]] = float(p1[:3] == p2[:3])

            # ---------------- missingness ----------------
            e1 = not a_as
            e2 = not b_as
            X[r, F["addr_empty_q"]] = float(e1)
            X[r, F["addr_empty_c"]] = float(e2)
            X[r, F["addr_empty_any"]] = float(e1 or e2)

            # ---------------- cross-field ----------------
            # Source 3 occasionally packs the business name into the address field.
            if sa and s_ba:
                X[r, F["cross_name_in_addr"]] = _safe_div(len(sa & s_ba), len(sa))
            if s_aa and sb:
                X[r, F["cross_addr_in_name"]] = _safe_div(len(s_aa & sb), len(sb))

            X[r, F["country_equal"]] = float(qc[qi] == cc[ci])

        return X


# --------------------------------------------------------------------------------------
# record-side preparation
# --------------------------------------------------------------------------------------

_LEGAL_BIT: Dict[str, int] = {}


def _legal_mask(s: str) -> int:
    """Encode a comma-joined legal-suffix code list as an int bitmask.

    A bitmask rather than a ``frozenset`` because this array is built for millions of
    records: the set version costs ~200 bytes per record and dominated peak memory.
    """
    if not s:
        return 0
    m = 0
    for code in s.split(","):
        if not code:
            continue
        bit = _LEGAL_BIT.get(code)
        if bit is None:
            bit = 1 << len(_LEGAL_BIT)
            _LEGAL_BIT[code] = bit
        m |= bit
    return m


def prepare_side(df, columns: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    """Pre-split a record frame into the arrays :class:`FeatureBuilder` expects.

    Splitting once per record instead of once per pair matters: a record participates in
    many candidate pairs, so tokenisation would otherwise be repeated dozens of times.

    Callers should pass only the records actually referenced by candidate pairs -- see
    :func:`trice.pipeline.compact_candidates`. Preparing a whole 6 M-row source frame when
    ~2 M rows are referenced exhausts the memory budget.
    """
    name_core = df["name_core"].fillna("").to_numpy(dtype=object)
    addr_alpha = df["addr_alpha"].fillna("").to_numpy(dtype=object)
    digits = df["addr_digits"].fillna("").to_numpy(dtype=object)
    legal = df["legal"].fillna("").to_numpy(dtype=object)

    out = {
        "name_core": name_core,
        "addr_alpha": addr_alpha,
        "name_skel": df["name_skel"].fillna("").to_numpy(dtype=object),
        "name_nospace": df["name_nospace"].fillna("").to_numpy(dtype=object),
        "house": df["house"].fillna("").to_numpy(dtype=object),
        "postal": df["postal"].fillna("").to_numpy(dtype=object),
        "country": df["country"].astype(str).to_numpy(dtype=object),
        "name_tokens": np.array([s.split() for s in name_core], dtype=object),
        "addr_tokens": np.array([s.split() for s in addr_alpha], dtype=object),
        "digit_tokens": np.array([[x for x in s.split(",") if x] for s in digits],
                                 dtype=object),
        "legal_set": np.fromiter((_legal_mask(s) for s in legal), dtype=np.int64,
                                 count=len(legal)),
    }
    return out


__all__ = ["FEATURE_NAMES", "FEATURE_INDEX", "N_FEATURES", "FeatureBuilder",
           "prepare_side", "STAGE2_RAW_FEATURES", "STAGE2_RAW_INDEX"]
