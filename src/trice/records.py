"""
Streaming normalisation of the raw TSVs into a compact columnar record store.

Scale constraints
-----------------
The full dataset is ~24 M records across 7 files on a 16.9 GB machine, so the store is
built to be *narrow*:

* ``entity_id`` is split into an ``int32`` numeric part plus an ``int8`` source tag; the
  original string is never retained (it is reconstructed as ``f"S{src}-{num}"``).
* strings are written as Arrow-backed dictionary/plain columns in Parquet, which keeps
  them near raw byte size instead of the ~3x overhead of Python ``str`` objects.
* the work is spread over a process pool in line blocks, because normalisation is
  CPU-bound pure Python.

Output columns per source file
------------------------------
=================  ======  =========================================================
column             dtype   meaning
=================  ======  =========================================================
``num``            int32   numeric part of ``entity_id``
``src``            int8    1, 2 or 3
``country``        int8    index into the returned country vocabulary
``name_core``      str     space-joined core name tokens (legal suffixes removed)
``name_skel``      str     consonant skeleton of the core name (cross-script key)
``name_nospace``   str     core name with separators removed (domainified names)
``legal``          str     comma-joined canonical legal-suffix codes
``addr_alpha``     str     space-joined canonical non-numeric address tokens
``addr_digits``    str     comma-joined numeric address tokens
``postal``         str     extracted postal code ('' when absent)
``house``          str     house number ('' when absent)
=================  ======  =========================================================
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_address, normalize_name
from .paths import open_text

SOURCE_FILES = {
    ("train", 1): "train_source1.tsv",
    ("train", 2): "train_source2.tsv",
    ("train", 3): "train_source3.tsv",
    ("test", 1): "test_source1.tsv",
    ("test", 2): "test_source2.tsv",
    ("test", 3): "test_source3.tsv",
}

# Kept deliberately small and closed-world-free: countries are discovered from the data,
# never hard-coded, per the explicit "treat country as an open set" instruction.
_COUNTRY_VOCAB_LIMIT = 127


def _norm_block(args: Tuple[List[str], int, Dict[str, str], Dict[str, str]]) -> dict:
    """Normalise one block of raw TSV lines. Runs in a worker process."""
    lines, src, addr_map, name_map = args
    n = len(lines)
    num = np.empty(n, dtype=np.int64)
    country = [""] * n
    name_core: List[str] = [""] * n
    name_skel: List[str] = [""] * n
    name_nospace: List[str] = [""] * n
    legal: List[str] = [""] * n
    addr_alpha: List[str] = [""] * n
    addr_digits: List[str] = [""] * n
    postal: List[str] = [""] * n
    house: List[str] = [""] * n

    w = 0
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        eid, raw_name, raw_addr, ctry = parts[0], parts[1], parts[2], parts[3]
        try:
            num[w] = int(eid[3:])
        except ValueError:
            continue

        nm = normalize_name(raw_name)
        ad = normalize_address(raw_addr)

        # Apply the mined alias maps: address tokens and name tokens are folded onto
        # their canonical representative so 'texas'/'tx' and 'marketing'/'maarketting'
        # become literally equal.
        core = nm.core_tokens
        if name_map:
            core = [name_map.get(t, t) for t in core]
        alpha = ad.alpha_tokens
        if addr_map:
            alpha = [addr_map.get(t, t) for t in alpha]
            alpha = list(dict.fromkeys(alpha))

        country[w] = ctry
        name_core[w] = " ".join(core)
        name_skel[w] = nm.skeleton
        name_nospace[w] = nm.nospace
        legal[w] = ",".join(sorted(nm.legal))
        addr_alpha[w] = " ".join(alpha)
        addr_digits[w] = ",".join(ad.digits)
        postal[w] = ad.postal
        house[w] = ad.house
        w += 1

    return {
        "num": num[:w],
        "src": np.full(w, src, dtype=np.int8),
        "country": country[:w],
        "name_core": name_core[:w],
        "name_skel": name_skel[:w],
        "name_nospace": name_nospace[:w],
        "legal": legal[:w],
        "addr_alpha": addr_alpha[:w],
        "addr_digits": addr_digits[:w],
        "postal": postal[:w],
        "house": house[:w],
    }


def iter_blocks(path: str, block_lines: int = 50_000) -> Iterator[List[str]]:
    """Yield blocks of raw data lines (header skipped)."""
    with open_text(path, newline="") as fh:
        fh.readline()
        block: List[str] = []
        for line in fh:
            block.append(line)
            if len(block) >= block_lines:
                yield block
                block = []
        if block:
            yield block


STR_COLUMNS = ("name_core", "name_skel", "name_nospace", "legal",
               "addr_alpha", "addr_digits", "postal", "house")

_ARROW_SCHEMA = None


def _schema():
    """Arrow schema for the record store (built lazily so pyarrow stays optional)."""
    global _ARROW_SCHEMA
    if _ARROW_SCHEMA is None:
        import pyarrow as pa
        fields = [pa.field("num", pa.int32()), pa.field("src", pa.int8()),
                  pa.field("country", pa.string())]
        fields += [pa.field(c, pa.string()) for c in STR_COLUMNS]
        _ARROW_SCHEMA = pa.schema(fields)
    return _ARROW_SCHEMA


def _to_batch(out: dict):
    """Convert one worker result dict into an Arrow RecordBatch."""
    import pyarrow as pa
    num = out["num"]
    if len(num) and int(num.max()) > np.iinfo(np.int32).max:
        raise ValueError("entity_id numeric part exceeds int32")
    arrays = [pa.array(num.astype(np.int32)), pa.array(out["src"]),
              pa.array(out["country"], type=pa.string())]
    arrays += [pa.array(out[c], type=pa.string()) for c in STR_COLUMNS]
    return pa.RecordBatch.from_arrays(arrays, schema=_schema())


def normalize_file_to_parquet(path: str, src: int, out_path: str,
                              addr_map: Dict[str, str], name_map: Dict[str, str],
                              pool=None, block_lines: int = 50_000,
                              row_group: int = 250_000,
                              progress=None) -> int:
    """Normalise one source TSV straight into a Parquet file.

    Batches are written incrementally so peak memory stays proportional to
    ``block_lines * workers`` rather than to the file size -- necessary because a single
    source file holds up to 5.3 M records and materialising it as Python lists exhausts
    the 16.9 GB budget.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks = ((block, src, addr_map, name_map) for block in iter_blocks(path, block_lines))
    mapper = pool.imap(_norm_block, tasks, chunksize=1) if pool else map(_norm_block, tasks)

    tmp = out_path + ".tmp"
    writer = pq.ParquetWriter(tmp, _schema(), compression="zstd")
    pending: List = []
    pending_rows = 0
    total = 0
    try:
        for out in mapper:
            batch = _to_batch(out)
            pending.append(batch)
            pending_rows += batch.num_rows
            total += batch.num_rows
            if pending_rows >= row_group:
                writer.write_table(pa.Table.from_batches(pending, schema=_schema()))
                pending.clear()
                pending_rows = 0
            if progress:
                progress(total)
        if pending:
            writer.write_table(pa.Table.from_batches(pending, schema=_schema()))
    finally:
        writer.close()
    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(tmp, out_path)
    return total


def store_path(store_dir: str, split: str, src: int) -> str:
    return os.path.join(store_dir, f"{split}_source{src}.parquet")


def load_store(store_dir: str, split: str, src: int,
               columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(store_path(store_dir, split, src), columns=columns)


def ids_to_strings(num: np.ndarray, src: np.ndarray | int) -> np.ndarray:
    """Reconstruct ``S{src}-{num}`` entity id strings from the compact encoding."""
    if isinstance(src, int):
        prefix = np.full(len(num), f"S{src}-", dtype=object)
    else:
        prefix = np.char.add("S", np.char.add(src.astype(str), "-")).astype(object)
    return np.char.add(prefix.astype(str), num.astype(str))


__all__ = [
    "SOURCE_FILES", "normalize_file_to_parquet", "iter_blocks", "store_path",
    "load_store", "ids_to_strings", "STR_COLUMNS",
]
