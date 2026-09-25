"""
Text normalisation for business entity resolution.

Design notes
------------
Everything here is derived from the corruption operators catalogued in
``docs/05_EDA_FINDINGS.md`` by reading real ground-truth match groups. Three
choices matter more than the rest:

1.  **No hand-written stop-word / generic-descriptor list.** Noise tokens such as
    ``services``, ``center``, ``group`` are suppressed automatically by IDF computed
    over the provided corpus (see :mod:`trice.idf`). Hard-coding them would be both
    less complete and less defensible.

2.  **Consonant skeleton with double-collapse** is the mechanism that makes
    cross-script matching work. ``unidecode`` alone is not enough::

        unidecode('राम मार्केटिंग प्राइवेट लिमिटेड') -> 'raama maarkettiNg praaivett limitteda'

    which does not equal ``ram marketing private limited``. But dropping the ``N``
    anusvara artefact, removing vowels and collapsing doubled consonants maps *both*
    strings to ``rmmrktngprvtlmtd``.

3.  **Legal suffixes are separated, not deleted.** Suffix agreement/conflict is a real
    feature; suffix presence is pure noise. Splitting keeps both usable.

Only the abbreviation classes explicitly named in the problem statement are hard-coded.
The long tail (``Mysore``/``Mysuru``, ``14th``/``FOURTEENTH``, Devanagari state names) is
*mined from the training ground truth* by :mod:`trice.mine`, keeping the pipeline inside
the fair-play rules -- no external data source is consulted.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from unidecode import unidecode

# --------------------------------------------------------------------------------------
# character-level helpers
# --------------------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")

# Junk framing observed in the data: '...', '--', '##', '<<', '#', '[', ']', '(', ')'
_JUNK_EDGE = re.compile(r"^[\s\-\.\#\<\>\*\|\+\~\^\_/\\,;:!\?\"'\[\]\(\)\{\}]+|"
                        r"[\s\-\.\#\<\>\*\|\+\~\^\_/\\,;:!\?\"'\[\]\(\)\{\}]+$")
_MULTISPACE = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGIT_RUN = re.compile(r"\d+")
_HAS_DIGIT = re.compile(r"\d")

# 'M/s' (Indian) and a leading definite article carry no identity information.
_LEADING_NOISE = re.compile(r"^(?:m/s\.?|messrs\.?|the)\s+")

# ``X dba Y`` / ``X d/b/a Y`` -- Y is the trade name that actually matches.
_DBA = re.compile(r"\b(?:dba|d/b/a|d\.b\.a\.?|doing\s+business\s+as|trading\s+as|t/a)\b")

# Domainified names: 'edwardshintzedougherty.com', 'tejrajchits.com'
_DOMAIN = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\."
    r"(?:com|net|org|biz|info|io|co|in|co\.in|fr|us|uk|shop|online|store)$"
)

_LITERAL_NULL = frozenset({"<null>", "null", "n/a", "na", "none", "-", "--", "nil", ""})

# Devanagari transliteration artefacts, applied while case is still meaningful so the
# capital-N marker can be told apart from a real 'n'.
#
# ``unidecode`` emits a capital ``N`` for two different Devanagari features: the anusvara
# nasal (where it *is* a real ``n`` -- ``मार्केटिंग`` -> ``maarkettiNg`` = "marketing") and
# certain vowel signs (where it is spurious -- ``मॉडर्न`` -> ``moNddrn`` = "modern").
# Mapping it to ``n`` is correct in the first, far more common case and leaves a
# single-character discrepancy in the second, which fuzzy similarity absorbs.
_TRANSLIT_FIX = (
    ("N", "n"),
    ("ph", "f"),    # 'phuudds' -> 'fuudds' = "foods"
    ("PH", "F"),
)

# Only the script-independent part is reapplied inside the skeleton functions; the
# capital-N rule has already been handled by :func:`fold`.
_SKELETON_FIX = (("ph", "f"),)


def fold(text: str) -> str:
    """Lower-case, strip accents and transliterate to ASCII.

    Handles French diacritics (``Société`` -> ``societe``) and Devanagari
    (``राम`` -> ``raama``) in one pass.

    The anusvara artefact matters here. ``unidecode`` renders the Devanagari nasal as a
    capital ``N`` (``maarkettiNg``), which is spurious -- the Latin spelling has no such
    letter. It must be removed *before* case folding, because afterwards it is
    indistinguishable from a legitimate ``n``. The fix is applied only when the input was
    actually non-ASCII, so a genuine capital ``N`` in ``NEW DELHI`` is never touched.
    """
    if not text:
        return ""
    # NFKC first so composed/decomposed accents behave identically.
    text = unicodedata.normalize("NFKC", text)
    if not text.isascii():
        text = unidecode(text)
        for src, dst in _TRANSLIT_FIX:
            if src in text:
                text = text.replace(src, dst)
    return _MULTISPACE.sub(" ", text.lower()).strip()


def strip_junk(text: str) -> str:
    """Remove junk framing characters and leading noise words."""
    if not text:
        return ""
    prev = None
    while prev != text:
        prev = text
        text = _JUNK_EDGE.sub("", text)
    text = _LEADING_NOISE.sub("", text)
    return text.strip()


def consonant_skeleton(text: str) -> str:
    """Vowel-free, double-collapsed consonant signature.

    This is the cross-script equaliser. Both the Latin and the Devanagari rendering of
    a name collapse to the same string::

        >>> consonant_skeleton('ram marketing private limited')
        'rmmrktngprvtlmtd'
        >>> consonant_skeleton(unidecode('राम मार्केटिंग प्राइवेट लिमिटेड'))
        'rmmrktngprvtlmtd'

    ``y`` counts as a consonant; digits are kept (they are highly discriminative).
    """
    if not text:
        return ""
    for src, dst in _SKELETON_FIX:
        if src in text:
            text = text.replace(src, dst)
    text = text.lower()
    out: List[str] = []
    last = ""
    for ch in text:
        if ch in _VOWELS:
            continue
        if not (ch.isalnum()):
            continue
        if ch == last:          # collapse doubled consonants / digits
            continue
        out.append(ch)
        last = ch
    return "".join(out)


def vowel_squeeze(text: str) -> str:
    """Collapse repeated letters but keep vowels -- a softer skeleton than
    :func:`consonant_skeleton`, useful as a second blocking key.

    ``'raama maarkettiNg'`` -> ``'ramamarketing'``
    """
    if not text:
        return ""
    for src, dst in _SKELETON_FIX:
        if src in text:
            text = text.replace(src, dst)
    text = text.lower()
    out: List[str] = []
    last = ""
    for ch in text:
        if not ch.isalnum():
            continue
        if ch == last:
            continue
        out.append(ch)
        last = ch
    return "".join(out)


# --------------------------------------------------------------------------------------
# legal suffixes
# --------------------------------------------------------------------------------------

# Canonical code -> surface variants. Covers the US / India / France forms seen in the
# data plus the abbreviation pairs the problem statement names explicitly.
LEGAL_FORMS: Dict[str, Tuple[str, ...]] = {
    "inc": ("inc", "inc.", "incorporated", "incorp"),
    "llc": ("llc", "l.l.c", "l.l.c.", "lc"),
    "corp": ("corp", "corp.", "corporation", "corpn"),
    "ltd": ("ltd", "ltd.", "limited", "limitee", "limiteda", "ltda"),
    "pvt": ("pvt", "pvt.", "private", "pravate", "pvtltd"),
    "llp": ("llp", "l.l.p", "elelpi", "elelpii"),
    "lp": ("lp", "l.p"),
    "plc": ("plc", "p.l.c"),
    "pc": ("pc", "p.c"),
    "pllc": ("pllc",),
    "pa": ("pa", "p.a"),
    "co": ("co", "co.", "company", "kompany"),
    "opc": ("opc",),
    "huf": ("huf",),
    "gmbh": ("gmbh",),
    "sarl": ("sarl", "s.a.r.l", "sarlu"),
    "sas": ("sas", "s.a.s", "sasu"),
    "sa": ("sa", "s.a"),
    "sci": ("sci", "s.c.i"),
    "eurl": ("eurl",),
    "snc": ("snc",),
    "scop": ("scop",),
    "scm": ("scm",),
    "selarl": ("selarl",),
    "trust": ("trust",),
    "assn": ("assn", "association"),
}

LEGAL_LOOKUP: Dict[str, str] = {}
for _code, _variants in LEGAL_FORMS.items():
    for _v in _variants:
        LEGAL_LOOKUP[_v] = _code
        LEGAL_LOOKUP[_v.replace(".", "")] = _code

# Suffix codes that are country-specific enough that a *conflict* is informative
# (an ``inc`` never matches a ``sarl``), versus interchangeable pairs.
LEGAL_EQUIV: Dict[str, str] = {
    "ltd": "ltd", "pvt": "ltd", "limited": "ltd",
    "inc": "inc", "corp": "inc",
    "llc": "llc", "pllc": "llc",
    "llp": "llp", "lp": "llp",
}

# Skeleton-keyed lookup so transliterated suffixes are still recognised:
# 'praaivett' -> skeleton 'prvt' -> same as 'private'; 'limitteda' -> 'lmtd' -> 'limited'.
# Only consulted for long tokens with a long skeleton, otherwise short core words
# collide (e.g. 'ace' skeletonises to 'c', which would masquerade as 'co').
_LEGAL_SKELETON_MIN_TOKEN = 5
_LEGAL_SKELETON_MIN_SKEL = 3

LEGAL_SKELETON: Dict[str, str] = {}
for _code, _variants in LEGAL_FORMS.items():
    for _v in _variants:
        if len(_v) < _LEGAL_SKELETON_MIN_TOKEN:
            continue
        _sk = consonant_skeleton(_v)
        if len(_sk) >= _LEGAL_SKELETON_MIN_SKEL:
            LEGAL_SKELETON.setdefault(_sk, _code)


def lookup_legal(tok: str) -> str | None:
    """Return the canonical legal-suffix code for ``tok``, or ``None``.

    Tries the literal surface form first, then the consonant skeleton for tokens long
    enough that a skeleton collision is implausible.
    """
    code = LEGAL_LOOKUP.get(tok)
    if code is not None:
        return code
    if len(tok) >= _LEGAL_SKELETON_MIN_TOKEN:
        sk = consonant_skeleton(tok)
        if len(sk) >= _LEGAL_SKELETON_MIN_SKEL:
            return LEGAL_SKELETON.get(sk)
    return None


# --------------------------------------------------------------------------------------
# address vocabulary explicitly named in the problem statement
# --------------------------------------------------------------------------------------

STREET_TYPES: Dict[str, str] = {
    "street": "st", "st": "st", "str": "st", "saint": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "boulvard": "blvd", "bd": "blvd",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq",
    "suite": "ste", "ste": "ste",
    "floor": "fl", "flr": "fl", "fl": "fl",
    "apartment": "apt", "apartments": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg",
    "block": "blk", "blk": "blk",
    "phase": "ph", "sector": "sec", "sec": "sec",
    "nagar": "nagar", "colony": "colony", "marg": "rd", "galli": "ln",
    "rue": "rue", "r": "rue", "chemin": "chemin", "allee": "allee",
    "impasse": "impasse", "quai": "quai", "cours": "cours",
    "house": "hno", "hno": "hno", "hn": "hno", "h": "hno",
    "number": "no", "no": "no", "num": "no", "nos": "no",
    "near": "", "opp": "", "opposite": "", "behind": "", "beside": "",
    "next": "", "above": "", "below": "", "landmark": "",
}

# Ordinal / cardinal words -> digits. Covers the observed ``14th`` <-> ``FOURTEENTH``.
_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen")
_ORDINALS = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
             "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth")
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_TENS_ORD = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
             "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}

NUMBER_WORDS: Dict[str, str] = {}
for _i, _w in enumerate(_UNITS):
    NUMBER_WORDS[_w] = str(_i)
for _i, _w in enumerate(_ORDINALS):
    NUMBER_WORDS[_w] = str(_i)
for _w, _v in _TENS.items():
    NUMBER_WORDS[_w] = str(_v)
for _w, _v in _TENS_ORD.items():
    NUMBER_WORDS[_w] = str(_v)

_ORDINAL_SUFFIX = re.compile(r"^(\d+)(?:st|nd|rd|th)$")

# Roman numerals appear as street/phase markers ('Iii Street', 'Phase V').
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6",
          "vii": "7", "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12"}


def canon_addr_token(tok: str) -> str:
    """Canonicalise one address token. Returns ``''`` for tokens to drop."""
    if not tok:
        return ""
    m = _ORDINAL_SUFFIX.match(tok)
    if m:
        return m.group(1)
    if tok in NUMBER_WORDS:
        return NUMBER_WORDS[tok]
    st = STREET_TYPES.get(tok)
    if st is not None:
        return st
    if tok in _ROMAN:
        return _ROMAN[tok]
    return tok


# --------------------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------------------

class NameParts:
    """Normalised views of a business name."""

    __slots__ = ("raw", "core_tokens", "legal", "skeleton", "squeeze",
                 "nospace", "is_domain", "had_dba")

    def __init__(self, raw: str, core_tokens: List[str], legal: Set[str],
                 skeleton: str, squeeze: str, nospace: str,
                 is_domain: bool, had_dba: bool) -> None:
        self.raw = raw
        self.core_tokens = core_tokens
        self.legal = legal
        self.skeleton = skeleton
        self.squeeze = squeeze
        self.nospace = nospace
        self.is_domain = is_domain
        self.had_dba = had_dba

    def core(self) -> str:
        return " ".join(self.core_tokens)

    def as_dict(self) -> dict:
        return {
            "core": self.core(),
            "legal": ",".join(sorted(self.legal)),
            "skeleton": self.skeleton,
            "squeeze": self.squeeze,
            "nospace": self.nospace,
            "is_domain": self.is_domain,
            "had_dba": self.had_dba,
        }


def normalize_name(raw: str) -> NameParts:
    """Parse a business name into core tokens, legal suffixes and skeletons.

    Applied operators, in order: unicode fold -> junk strip -> DBA split ->
    domain de-concatenation -> ``&``/``+`` unification -> tokenise ->
    legal-suffix separation -> skeletons.
    """
    folded = fold(raw)
    had_dba = False

    # 'Halonex dba Marnie Baynes Keystone Bnb Inc' -> keep the trade name (right side).
    parts = _DBA.split(folded)
    if len(parts) > 1:
        had_dba = True
        folded = max(parts[1:], key=len).strip() or parts[0]

    folded = strip_junk(folded)

    # Domainified name: strip the TLD, keep the concatenated stem.
    is_domain = False
    dm = _DOMAIN.match(folded.replace(" ", ""))
    if dm:
        is_domain = True
        folded = dm.group(1).replace("-", " ")

    # '&' / '+' both stand for 'and'; drop them (they carry no identity signal).
    folded = folded.replace("&", " ").replace("+", " ")

    raw_tokens = [t for t in _NONALNUM.split(folded) if t]

    core_tokens: List[str] = []
    legal: Set[str] = set()
    for tok in raw_tokens:
        code = lookup_legal(tok)
        if code is not None:
            legal.add(LEGAL_EQUIV.get(code, code))
            continue
        core_tokens.append(tok)

    # A name consisting only of legal suffixes keeps them as core, otherwise we would
    # have nothing left to match on.
    if not core_tokens and raw_tokens:
        core_tokens = raw_tokens

    core_str = " ".join(core_tokens)
    return NameParts(
        raw=raw,
        core_tokens=core_tokens,
        legal=legal,
        skeleton=consonant_skeleton(core_str),
        squeeze=vowel_squeeze(core_str),
        nospace=_NONALNUM.sub("", core_str),
        is_domain=is_domain,
        had_dba=had_dba,
    )


# --------------------------------------------------------------------------------------
# address normalisation
# --------------------------------------------------------------------------------------

class AddrParts:
    """Normalised views of a business address.

    The address is treated as an **unordered bag** of canonical components, because
    component reordering is one of the most common corruptions in this dataset
    (``44 Goodrich Avenue, Auburn, ME`` -> ``ME, Auburn, 44 Goodrich Avenue``).
    """

    __slots__ = ("raw", "tokens", "digits", "postal", "house", "alpha_tokens",
                 "skeleton", "is_empty")

    def __init__(self, raw: str, tokens: List[str], digits: List[str], postal: str,
                 house: str, alpha_tokens: List[str], skeleton: str,
                 is_empty: bool) -> None:
        self.raw = raw
        self.tokens = tokens
        self.digits = digits
        self.postal = postal
        self.house = house
        self.alpha_tokens = alpha_tokens
        self.skeleton = skeleton
        self.is_empty = is_empty

    def as_dict(self) -> dict:
        return {
            "tokens": " ".join(self.tokens),
            "digits": ",".join(self.digits),
            "postal": self.postal,
            "house": self.house,
            "alpha": " ".join(self.alpha_tokens),
            "skeleton": self.skeleton,
            "is_empty": self.is_empty,
        }


def _extract_postal(folded: str) -> str:
    """Pull a plausible postal code out of an address.

    India PIN is 6 digits and is often written with a space (``600 094``); US ZIP is 5;
    France is 5. Longest-digit-run-wins, preferring 6 then 5, and joining split runs.
    """
    # Join 'NNN NNN' / 'NNN-NNN' style splits before scanning.
    joined = re.sub(r"(\d{3})[\s\-](\d{3})\b", r"\1\2", folded)
    runs = _DIGIT_RUN.findall(joined)
    for want in (6, 5):
        for r in runs:
            if len(r) == want:
                return r
    return ""


def normalize_address(raw: str, postal_hint: str = "") -> AddrParts:
    """Parse an address into a canonical component bag."""
    folded = fold(raw)
    if folded in _LITERAL_NULL:
        folded = ""
    if not folded:
        return AddrParts(raw, [], [], "", "", [], "", True)

    folded = folded.replace("<null>", " ")
    postal = _extract_postal(folded) or postal_hint

    # House number: first digit-bearing token of the first component, with the
    # 'HN'/'H.NO'/'##' prefixes stripped.
    first_component = folded.split(",", 1)[0]
    house = ""
    for tok in _NONALNUM.split(first_component):
        if tok and _HAS_DIGIT.search(tok):
            house = tok
            break

    tokens: List[str] = []
    digits: List[str] = []
    alpha: List[str] = []
    for tok in _NONALNUM.split(folded):
        if not tok:
            continue
        c = canon_addr_token(tok)
        if not c:
            continue
        tokens.append(c)
        if _HAS_DIGIT.search(c):
            digits.append(c)
        elif len(c) > 1:
            alpha.append(c)

    # Deduplicate while keeping first-seen order (the bag view).
    tokens = list(dict.fromkeys(tokens))
    digits = list(dict.fromkeys(digits))
    alpha = list(dict.fromkeys(alpha))

    return AddrParts(
        raw=raw,
        tokens=tokens,
        digits=digits,
        postal=postal,
        house=house,
        alpha_tokens=alpha,
        skeleton=consonant_skeleton(" ".join(alpha)),
        is_empty=False,
    )


# --------------------------------------------------------------------------------------
# variant map (mined) application
# --------------------------------------------------------------------------------------

class VariantMap:
    """Token -> canonical token map mined from the training ground truth.

    Discovers the long tail that is not worth hand-coding and would be questionable to
    source externally: ``texas``->``tx``, ``mysuru``->``mysore``, ``tamilnadu``->``tn``,
    plus the Devanagari state renderings.
    """

    def __init__(self, mapping: Dict[str, str] | None = None) -> None:
        self.mapping: Dict[str, str] = mapping or {}

    def apply(self, tokens: Sequence[str]) -> List[str]:
        m = self.mapping
        if not m:
            return list(tokens)
        return [m.get(t, t) for t in tokens]

    def __len__(self) -> int:
        return len(self.mapping)

    @classmethod
    def load(cls, path: str) -> "VariantMap":
        import json
        import os
        if not os.path.isfile(path):
            return cls({})
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data.get("token_map", {}))


__all__ = [
    "fold", "strip_junk", "consonant_skeleton", "vowel_squeeze",
    "normalize_name", "normalize_address", "canon_addr_token", "lookup_legal",
    "NameParts", "AddrParts", "VariantMap",
    "LEGAL_LOOKUP", "LEGAL_SKELETON", "LEGAL_EQUIV", "STREET_TYPES", "NUMBER_WORDS",
]
