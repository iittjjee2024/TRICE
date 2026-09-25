"""Dataset inspection: raw file stats, record-store preview, normalisation demo."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import ensure_src_on_path, settings

ensure_src_on_path()
from trice.normalize import normalize_address, normalize_name  # noqa: E402

router = APIRouter(prefix="/api", tags=["datasets"])

RAW_FILES = {
    ("train", 1): "train_source1.tsv",
    ("train", 2): "train_source2.tsv",
    ("train", 3): "train_source3.tsv",
    ("test", 1): "test_source1.tsv",
    ("test", 2): "test_source2.tsv",
    ("test", 3): "test_source3.tsv",
}


@router.get("/datasets")
def datasets() -> Dict[str, Any]:
    """Raw TSV presence/size plus record-store presence and row counts."""
    s = settings()
    out: List[Dict[str, Any]] = []
    for (split, src), fname in sorted(RAW_FILES.items()):
        raw = os.path.join(s.dataset_dir, split, fname)
        store = os.path.join(s.store_dir, f"{split}_source{src}.parquet")
        entry: Dict[str, Any] = {
            "split": split, "source": src, "file": fname,
            "raw_exists": os.path.isfile(raw),
            "raw_mb": round(os.path.getsize(raw) / 1e6, 1) if os.path.isfile(raw) else None,
            "store_exists": os.path.isfile(store),
            "store_mb": round(os.path.getsize(store) / 1e6, 1)
            if os.path.isfile(store) else None,
        }
        if entry["store_exists"]:
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(store)
                entry["rows"] = pf.metadata.num_rows
            except Exception:
                entry["rows"] = None
        out.append(entry)

    gt = os.path.join(s.dataset_dir, "train", "train_ground_truth.tsv")
    return {
        "files": out,
        "ground_truth": {
            "exists": os.path.isfile(gt),
            "mb": round(os.path.getsize(gt) / 1e6, 1) if os.path.isfile(gt) else None,
        },
        "prepared": all(f["store_exists"] for f in out),
    }


@router.get("/datasets/countries")
def countries() -> Dict[str, Any]:
    """Country mix per split, read from the record store."""
    s = settings()
    out: Dict[str, Any] = {}
    for split in ("train", "test"):
        per_source = {}
        for src in (1, 2, 3):
            path = os.path.join(s.store_dir, f"{split}_source{src}.parquet")
            if not os.path.isfile(path):
                continue
            df = pd.read_parquet(path, columns=["country"])
            vc = df["country"].value_counts()
            per_source[f"source{src}"] = {str(k): int(v) for k, v in vc.items()}
        if per_source:
            out[split] = per_source
    return {"countries": out}


@router.get("/datasets/preview")
def preview(split: str = Query("test", pattern="^(train|test)$"),
            source: int = Query(1, ge=1, le=3),
            country: Optional[str] = None,
            limit: int = Query(25, ge=1, le=200),
            offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Normalised records from the store, next to the values they were derived from."""
    s = settings()
    path = os.path.join(s.store_dir, f"{split}_source{source}.parquet")
    if not os.path.isfile(path):
        raise HTTPException(404, "record store not built; run scripts/03_prepare.py")
    df = pd.read_parquet(path)
    if country:
        df = df[df["country"] == country]
    total = len(df)
    df = df.iloc[offset:offset + limit]
    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "entity_id": f"S{r.src}-{r.num}",
            "country": str(r.country),
            "name_core": r.name_core, "name_skel": r.name_skel,
            "name_nospace": r.name_nospace, "legal": r.legal,
            "addr_alpha": r.addr_alpha, "addr_digits": r.addr_digits,
            "postal": r.postal, "house": r.house,
        })
    return {"total": int(total), "offset": offset, "rows": rows}


class NormalizeRequest(BaseModel):
    name: str = ""
    address: str = ""


@router.post("/datasets/normalize")
def normalize_demo(req: NormalizeRequest) -> Dict[str, Any]:
    """Run the normaliser on arbitrary text.

    Exposed because the normalisation stage is where most of this solution's accuracy
    comes from, and being able to poke at it directly (Devanagari, French accents, DBA,
    domainified names) is worth more than reading about it.
    """
    n = normalize_name(req.name or "")
    a = normalize_address(req.address or "")
    return {
        "name": {
            "raw": req.name,
            "core_tokens": n.core_tokens,
            "core": n.core(),
            "legal": sorted(n.legal),
            "skeleton": n.skeleton,
            "squeeze": n.squeeze,
            "nospace": n.nospace,
            "is_domain": n.is_domain,
            "had_dba": n.had_dba,
        },
        "address": {
            "raw": req.address,
            "tokens": a.tokens,
            "alpha_tokens": a.alpha_tokens,
            "digits": a.digits,
            "postal": a.postal,
            "house": a.house,
            "skeleton": a.skeleton,
            "is_empty": a.is_empty,
        },
    }


@router.get("/datasets/variants")
def variants(limit: int = Query(60, ge=1, le=500)) -> Dict[str, Any]:
    """Token aliases mined from the training ground truth."""
    import json
    path = os.path.join(settings().artifacts_dir, "variants.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "variants.json not found; run scripts/02_mine_variants.py")
    with open(path, encoding="utf-8") as fh:
        v = json.load(fh)
    return {
        "meta": v.get("meta", {}),
        "n_address_map": len(v.get("token_map", {})),
        "n_name_map": len(v.get("name_map", {})),
        "address_pairs": v.get("address_pairs", [])[:limit],
        "name_pairs": v.get("name_pairs", [])[:limit],
    }
