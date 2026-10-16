"""Scoped retrieval demo: prints what was retrieved and what fits the context."""
from __future__ import annotations

import argparse
import sys

try:  # run as a script: scripts/ is on sys.path
    from _cli_common import add_scope_arguments, request_from_args, utf8_stdout
except ImportError:  # imported as scripts.<name> (e.g. by interface/cli.py)
    from scripts._cli_common import add_scope_arguments, request_from_args, utf8_stdout

from api.answering import build_scope
from api.resources import RepoNotReady, ResourceRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitMind scoped retrieval demo")
    add_scope_arguments(parser)
    return parser.parse_args()


def main() -> int:
    utf8_stdout()
    args = parse_args()
    req = request_from_args(args)
    scope = build_scope(req)

    registry = ResourceRegistry(args.generations_dir)
    try:
        with registry.lease(req.repo) as entry:
            outcome = entry.service.retrieve(req.query, scope, top_k=req.top_k, limit=req.limit)
            included = set(outcome.assembled.included_ids)
            print(f"Repository: {req.repo}   generation: {entry.generation_id}")
            print(f"\nRetrieved {len(outcome.candidates)} candidates "
                  f"({len(included)} fit the context budget):")
            for c in outcome.candidates:
                m = c.metadata
                flag = "*" if c.chunk_id in included else " "
                print(f"{flag} {m.get('doc_type', '')}:{m.get('doc_id', '')} "
                      f"score={c.score:.4f} source={c.source} chunk={c.chunk_id}")
            for w in getattr(outcome, "warnings", []) or []:
                print(f"warning: {w}")
            print("\n(* = shown to the model)\n\nContext preview:\n")
            print(outcome.assembled.text)
    except RepoNotReady as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
