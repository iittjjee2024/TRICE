"""Submission file inspection, validation and download."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import ensure_src_on_path, settings
from ..store import load_inference

ensure_src_on_path()
from trice.export import read_entity_ids, validate_submission  # noqa: E402

router = APIRouter(prefix="/api", tags=["exports"])

FILES = ("matching_results.tsv", "candidate_pairs.tsv")


def _out_path(name: str) -> str:
    if name not in FILES:
        raise HTTPException(404, f"unknown output file {name}")
    return os.path.join(settings().output_dir, name)


@router.get("/output")
def output_status() -> Dict[str, Any]:
    """Presence, size and a head sample of each submission file."""
    out: List[Dict[str, Any]] = []
    for name in FILES:
        p = _out_path(name)
        entry: Dict[str, Any] = {"name": name, "exists": os.path.isfile(p)}
        if entry["exists"]:
            entry["bytes"] = os.path.getsize(p)
            entry["mb"] = round(entry["bytes"] / 1e6, 2)
            head: List[str] = []
            with open(p, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 11:
                        break
                    head.append(line.rstrip("\n"))
            entry["head"] = head
        out.append(entry)
    return {"files": out, "output_dir": settings().output_dir}


@router.get("/output/validate")
def validate(check_ids: bool = Query(False)) -> Dict[str, Any]:
    """Re-check both files against every rule in the problem statement.

    ``check_ids`` additionally verifies every emitted id exists in the test set; it loads
    ~10 M ids into memory, so it is off by default -- the same trade-off the official
    ``utils/validate_submission.py`` makes.
    """
    s = settings()
    m = _out_path("matching_results.tsv")
    c = _out_path("candidate_pairs.tsv")
    if not os.path.isfile(m):
        raise HTTPException(404, "matching_results.tsv not found; run scripts/06_infer.py")
    test_s1 = os.path.join(s.dataset_dir, "test", "test_source1.tsv")
    if not os.path.isfile(test_s1):
        raise HTTPException(404, "test_source1.tsv not found")
    required = set(read_entity_ids(test_s1))
    targets = None
    if check_ids:
        targets = set(read_entity_ids(os.path.join(s.dataset_dir, "test",
                                                   "test_source2.tsv")))
        targets |= set(read_entity_ids(os.path.join(s.dataset_dir, "test",
                                                    "test_source3.tsv")))
    report = validate_submission(m, c if os.path.isfile(c) else None, required, targets)
    report["id_existence_checked"] = bool(check_ids)
    return report


@router.get("/output/{name}")
def download(name: str) -> FileResponse:
    p = _out_path(name)
    if not os.path.isfile(p):
        raise HTTPException(404, f"{name} not found")
    return FileResponse(p, media_type="text/tab-separated-values", filename=name)


@router.get("/runs/{run_id}/inference")
def inference(run_id: str) -> Dict[str, Any]:
    inf = load_inference(run_id)
    if inf is None:
        raise HTTPException(404, f"no inference record for run {run_id}")
    return inf
