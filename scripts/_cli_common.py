"""scripts/_cli_common.py — shared bootstrap for the retrieval/answer CLIs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import QueryRequest  # noqa: E402
from config.settings import settings  # noqa: E402


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("--repo", required=True, help="Repository identity, e.g. owner/name")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--author", action="append", default=[], dest="authors")
    parser.add_argument("--doc-type", action="append", default=[], dest="doc_types")
    parser.add_argument("--module", action="append", default=[], dest="modules")
    parser.add_argument("--kind", choices=["history", "snapshot"])
    parser.add_argument("--time-start")
    parser.add_argument("--time-end")
    parser.add_argument("--generations-dir", default=settings.generations_dir)


def request_from_args(args: argparse.Namespace, mode: str = "direct") -> QueryRequest:
    """Same validation as the API (a bad scope fails before any work)."""
    return QueryRequest(
        repo=args.repo,
        query=args.query,
        mode=mode,
        top_k=args.top_k,
        limit=args.limit,
        authors=args.authors,
        doc_types=args.doc_types,
        modules=args.modules,
        kind=args.kind,
        time_start=args.time_start,
        time_end=args.time_end,
    )


def utf8_stdout() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
