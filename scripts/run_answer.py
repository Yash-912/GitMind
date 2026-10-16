"""End-to-end answer demo: scoped retrieval, then validated generation."""
from __future__ import annotations

import argparse
import sys
import time

try:  # run as a script: scripts/ is on sys.path
    from _cli_common import add_scope_arguments, request_from_args, utf8_stdout
except ImportError:  # imported as scripts.<name> (e.g. by interface/cli.py)
    from scripts._cli_common import add_scope_arguments, request_from_args, utf8_stdout

from api.answering import DeadlineLLM, build_scope, generate_structured_answer
from api.resources import RepoNotReady, ResourceRegistry
from config.settings import settings
from embedding.models import EmbeddingError
from generation.answers import AnswerStatus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitMind end-to-end answer demo")
    add_scope_arguments(parser)
    parser.add_argument("--mode", default="direct", choices=["direct", "memo", "blame", "risk"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()
    args = parse_args(argv)
    req = request_from_args(args, args.mode)
    scope = build_scope(req)

    from generation.llm_client import LLMClient

    registry = ResourceRegistry(args.generations_dir)
    llm = LLMClient()
    try:
        with registry.lease(req.repo) as entry:
            outcome = entry.service.retrieve(req.query, scope, top_k=req.top_k, limit=req.limit)
            result = generate_structured_answer(
                req.mode,
                req.query,
                scope,
                outcome,
                entry,
                DeadlineLLM(llm, time.monotonic() + settings.query_timeout_s),
            )
            print(f"Status: {result.status.value}   model: {result.model or '-'}")
            if result.error:
                print(f"Error: {result.error}")
            print("\nAnswer:\n")
            print(result.text or "(none)")
            by_id = {c.chunk_id: c for c in outcome.candidates}
            if result.cited_ids:
                print("\nCited evidence:")
                for cid in result.cited_ids:
                    m = by_id[cid].metadata if cid in by_id else {}
                    print(f"- {m.get('doc_type', '')}:{m.get('doc_id', '')} chunk={cid}")
            print(f"\n(retrieved {len(result.retrieved_ids)}, shown to model "
                  f"{len(result.included_ids)}, cited {len(result.cited_ids)})")
            return 0 if result.status in (
                AnswerStatus.ANSWERED, AnswerStatus.INSUFFICIENT_EVIDENCE
            ) else 1
    except RepoNotReady as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EmbeddingError as exc:
        print(f"error: embedding provider unavailable: {exc}", file=sys.stderr)
        return 3
    finally:
        llm.close()
        registry.close()


if __name__ == "__main__":
    sys.exit(main())
