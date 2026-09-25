"""Serve the project markdown documents to the UI."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..config import settings

router = APIRouter(prefix="/api", tags=["docs"])

_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _doc_path(name: str) -> str:
    if not _SAFE.match(name):
        raise HTTPException(400, "invalid document name")
    # constrain to the docs directory; reject anything that escapes it
    base = os.path.realpath(settings().docs_dir)
    path = os.path.realpath(os.path.join(base, f"{name}.md"))
    if not path.startswith(base + os.sep):
        raise HTTPException(400, "invalid document name")
    return path


@router.get("/docs-md")
def list_docs() -> Dict[str, Any]:
    base = settings().docs_dir
    out: List[Dict[str, Any]] = []
    if os.path.isdir(base):
        for fname in sorted(os.listdir(base)):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(base, fname)
            title = fname[:-3]
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            out.append({"name": fname[:-3], "title": title,
                        "bytes": os.path.getsize(path)})
    # the filled-in methodology template lives outside docs/
    tpl = os.path.join(settings().root, "Documentation_template.md")
    if os.path.isfile(tpl):
        out.append({"name": "__methodology__", "title": "Methodology (submission)",
                    "bytes": os.path.getsize(tpl)})
    return {"docs": out}


@router.get("/docs-md/{name}")
def get_doc(name: str) -> Dict[str, Any]:
    if name == "__methodology__":
        path = os.path.join(settings().root, "Documentation_template.md")
    else:
        path = _doc_path(name)
    if not os.path.isfile(path):
        raise HTTPException(404, f"document {name} not found")
    with open(path, encoding="utf-8") as fh:
        return {"name": name, "markdown": fh.read()}
