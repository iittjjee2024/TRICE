"""
Start the TRICE workbench API.

    python backend/run.py                 # http://127.0.0.1:8000  (docs at /docs)

Binds to loopback only: the service has no authentication and touches local files.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep on loopback: there is no auth)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {args.host}. This API has no authentication and "
              f"reads/writes local files. Prefer 127.0.0.1.", file=sys.stderr)

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload,
                log_level="info")


if __name__ == "__main__":
    main()
