"""Service configuration and filesystem layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# repo root = two levels above this file's package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class Settings:
    root: str = ROOT
    src_dir: str = os.path.join(ROOT, "src")
    docs_dir: str = os.path.join(ROOT, "docs")
    artifacts_dir: str = os.path.join(ROOT, "artifacts")
    store_dir: str = os.path.join(ROOT, "artifacts", "store")
    runs_dir: str = os.path.join(ROOT, "artifacts", "runs")
    blocking_dir: str = os.path.join(ROOT, "artifacts", "blocking")
    output_dir: str = os.path.join(ROOT, "output")
    dataset_dir: str = os.path.join(ROOT, "student_resource", "dataset")
    version: str = "1.0.0"

    # The decision tuner re-solves the expected-F objective per entity in Python, so an
    # interactive slider samples entities instead of scoring the whole validation split.
    simulate_sample_entities: int = 6_000


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


def ensure_src_on_path() -> None:
    """Make the framework-free ``trice`` package importable.

    ``src/trice`` deliberately has no web dependencies so it can be copied straight into
    the submission zip; the service reaches it by path rather than by packaging.
    """
    import sys
    s = settings().src_dir
    if s not in sys.path:
        sys.path.insert(0, s)
