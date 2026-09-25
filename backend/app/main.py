"""
TRICE workbench API.

A local analysis tool for the Amazon ML Challenge 2026 Business Entity Resolution task.
It reads the artifacts produced by ``scripts/0*.py`` and exposes them for the React UI.

Security note: this service binds to loopback and has **no authentication** while reading
and writing files under the project directory. It is a single-user developer tool; do not
bind it to a public interface.
"""

from __future__ import annotations

import os
import platform
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ensure_src_on_path, settings
from .routers import datasets, decision, docs, entities, exports, runs

ensure_src_on_path()

app = FastAPI(
    title="TRICE — Business Entity Resolution workbench",
    version=settings().version,
    description=__doc__,
)

# The Vite dev server runs on a different port; both loopback spellings are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(decision.router)
app.include_router(entities.router)
app.include_router(datasets.router)
app.include_router(exports.router)
app.include_router(docs.router)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    s = settings()
    caps = {}
    try:
        import lightgbm  # noqa: F401
        caps["lightgbm"] = True
    except Exception:
        caps["lightgbm"] = False
    for mod in ("rapidfuzz", "sklearn", "scipy", "pyarrow"):
        try:
            __import__(mod)
            caps[mod] = True
        except Exception:
            caps[mod] = False

    from .store import list_runs
    runs_list = list_runs()
    return {
        "status": "ok",
        "version": s.version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "capabilities": caps,
        "paths": {
            "root": s.root,
            "store": s.store_dir,
            "runs": s.runs_dir,
            "output": s.output_dir,
        },
        "state": {
            "store_built": os.path.isdir(s.store_dir)
            and bool(os.listdir(s.store_dir)) if os.path.isdir(s.store_dir) else False,
            "n_runs": len(runs_list),
            "latest_run": runs_list[0]["run_id"] if runs_list else None,
            "output_written": os.path.isfile(
                os.path.join(s.output_dir, "matching_results.tsv")),
        },
    }
