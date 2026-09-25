"""
Mine token-variant pairs from the training ground truth.

Motivation
----------
Normalisation needs to know that ``texas`` and ``tx`` are the same thing, and likewise
``tamil nadu``/``tn``, ``mysore``/``mysuru``, ``new york``/``ny``, plus the Devanagari
renderings of Indian state names. Hard-coding those tables would mean importing
geographic reference data, which sits uncomfortably close to the challenge's ban on
external data augmentation.

Instead we *derive* the table from the labelled data we were given. For a true match
``(a, b)`` the two addresses describe the same place, so tokens appearing on exactly one
side are candidate aliases of each other. Aggregated over hundreds of thousands of
matched pairs, the genuine aliases separate cleanly from noise.

This is fully inside the fair-play rules: the only input is ``train_ground_truth.tsv``
plus the provided source files.

Association measure
-------------------
For a candidate alias pair ``(x, y)`` with co-occurrence count ``c`` and marginal
"appeared alone" counts ``n_x``, ``n_y``::

    score(x, y) = c / sqrt(n_x * n_y)

a cosine-style statistic in ``[0, 1]``. Alias pairs sit near the top; incidental
co-occurrences (two unrelated dropped tokens) score near zero because their marginals are
large relative to the joint count.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Candidate pairs are only considered when they are *orthographically plausible*
# aliases. Without this filter, frequent unrelated pairs (two common dropped tokens)
# survive the statistical test through sheer volume.
_MIN_LEN = 2


def _plausible(x: str, y: str) -> bool:
    """Cheap orthographic gate on an alias pair."""
    if len(x) < _MIN_LEN or len(y) < _MIN_LEN:
        return False
    if x.isdigit() and y.isdigit():
        return False          # digit variants are handled numerically, not by aliasing
    if x.isdigit() != y.isdigit():
        return False
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    # abbreviation: every character of the short form appears in order in the long form
    # ('tx' in 'texas', 'ny' in 'new york' once concatenated, 'ka' in 'karnataka')
    it = iter(long_)
    if all(ch in it for ch in short):
        return True
    # shared prefix of >=3 ('mysore' / 'mysuru', 'bengaluru' / 'bengalooru')
    if len(short) >= 3 and short[:3] == long_[:3]:
        return True
    # shared first character is a weak but necessary condition for the rest
    return short[0] == long_[0] and len(short) >= 3


class VariantMiner:
    """Accumulates alias statistics over matched record pairs."""

    def __init__(self, max_only: int = 3) -> None:
        # max_only: skip pairs where either side has too many unmatched tokens -- those
        # are uninformative (whole-address rewrites) and would add quadratic noise.
        self.max_only = max_only
        self.joint: Dict[Tuple[str, str], int] = defaultdict(int)
        self.marg: Dict[str, int] = defaultdict(int)
        self.n_pairs = 0

    def add(self, tokens_a: Sequence[str], tokens_b: Sequence[str]) -> None:
        """Feed one *true* match pair's token bags."""
        sa, sb = set(tokens_a), set(tokens_b)
        if not sa or not sb:
            return
        only_a = sa - sb
        only_b = sb - sa
        if not only_a or not only_b:
            return
        if len(only_a) > self.max_only or len(only_b) > self.max_only:
            return
        self.n_pairs += 1
        for x in only_a:
            self.marg[x] += 1
        for y in only_b:
            self.marg[y] += 1
        for x in only_a:
            for y in only_b:
                if _plausible(x, y):
                    key = (x, y) if x < y else (y, x)
                    self.joint[key] += 1

    def pairs(self, min_count: int = 15, min_score: float = 0.12
              ) -> List[Tuple[str, str, int, float]]:
        """Return surviving alias pairs as ``(x, y, count, score)``."""
        out: List[Tuple[str, str, int, float]] = []
        for (x, y), c in self.joint.items():
            if c < min_count:
                continue
            nx, ny = self.marg[x], self.marg[y]
            if nx == 0 or ny == 0:
                continue
            score = c / math.sqrt(nx * ny)
            if score >= min_score:
                out.append((x, y, c, score))
        out.sort(key=lambda t: -t[2])
        return out

    # ---------------------------------------------------------------- clustering ----
    def token_map(self, min_count: int = 15, min_score: float = 0.12,
                  token_freq: Dict[str, int] | None = None) -> Dict[str, str]:
        """Collapse alias pairs into a ``token -> canonical token`` map.

        Union-find over the surviving pairs; the canonical representative of each
        cluster is its most frequent member (by corpus frequency when available, else by
        alias marginal), which keeps the map stable and interpretable.
        """
        parent: Dict[str, str] = {}

        def find(t: str) -> str:
            parent.setdefault(t, t)
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for x, y, _c, _s in self.pairs(min_count, min_score):
            union(x, y)

        clusters: Dict[str, List[str]] = defaultdict(list)
        for t in parent:
            clusters[find(t)].append(t)

        freq = token_freq or self.marg
        mapping: Dict[str, str] = {}
        for members in clusters.values():
            if len(members) < 2:
                continue
            canonical = max(members, key=lambda t: (freq.get(t, 0), -len(t), t))
            for m in members:
                if m != canonical:
                    mapping[m] = canonical
        return mapping


__all__ = ["VariantMiner"]
