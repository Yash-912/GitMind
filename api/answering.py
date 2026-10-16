"""api/answering.py — answer generation shared by the REST API and the CLI scripts.

Given a retrieval outcome it runs the selected mode's generator and normalises
the result to a StructuredResult (status + validated citations), so every
caller reports the same outcomes.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from api.schemas import Mode, QueryRequest, SourceRef
from config.settings import settings
from generation.answers import (
    INSUFFICIENT_MESSAGE,
    AnswerStatus,
    StructuredResult,
    split_citations,
)
from generation.llm_client import LLMUnavailableError
from retrieval.scope import Scope


class DeadlineLLM:
    """Shared LLM client bound to one request's absolute deadline."""

    def __init__(self, llm: Any, deadline_at: float) -> None:
        self._llm, self._deadline_at = llm, deadline_at

    @property
    def providers(self) -> list[str]:
        return list(getattr(self._llm, "providers", []))

    def generate(self, prompt: str, **kwargs: Any):
        remaining = self._deadline_at - time.monotonic()
        if remaining <= 0:
            raise LLMUnavailableError(["request deadline exceeded"])
        kwargs["deadline_s"] = remaining
        return self._llm.generate(prompt, **kwargs)

    def close(self) -> None:  # the client is shared; its owner closes it
        pass


def source_ref(c: Any) -> SourceRef:
    m = c.metadata or {}
    paths = m.get("file_paths") or []
    return SourceRef(
        chunk_id=c.chunk_id,
        repo=str(m.get("repo", "")),
        doc_type=str(m.get("doc_type", "")),
        doc_id=str(m.get("doc_id", "")),
        timestamp=str(m.get("timestamp", "")),
        author=str(m.get("author", "")),
        source_url=str(m.get("source_url") or m.get("url") or ""),
        commit_sha=str(m.get("commit_sha", "")),
        path=str(m.get("source_path") or (paths[0] if paths else "")),
        line_start=m.get("line_start"),
        line_end=m.get("line_end"),
        score=round(float(c.score), 4),
        source=str(c.source),
        snippet=(c.text or "")[:200],
    )


def source_refs(candidates: list[Any], ids: list[str]) -> list[SourceRef]:
    by_id = {c.chunk_id: c for c in candidates}
    return [source_ref(by_id[i]) for i in dict.fromkeys(ids) if i in by_id]


def build_scope(req: QueryRequest) -> Scope:
    return Scope.build(
        req.repo,
        time_start=req.time_start,
        time_end=req.time_end,
        modules=req.modules,
        authors=req.authors,
        doc_types=req.doc_types,
        kind=req.kind,
    )


def coverage_complete(generation: Any) -> bool:
    """Whether the ingestion behind this generation covered the source fully."""
    flag = (generation.extra or {}).get("coverage_complete")
    if isinstance(flag, bool):
        return flag
    try:
        from ingestion.document_store import DocumentStore

        store = DocumentStore(settings.db_path)
        try:
            manifests = store.get_documents("ingestion_manifest", generation.repo)
        finally:
            store.close()
        if not manifests:
            return False
        latest = max(manifests, key=lambda m: m.get("started_at", ""))
        return latest.get("status") == "complete"
    except Exception:
        return False


def generate_structured_answer(
    mode: Mode,
    question: str,
    scope: Scope,
    outcome: Any,
    entry: Any,
    llm: Any,
    coverage_fn: Callable[[Any], bool] = coverage_complete,
) -> StructuredResult:
    """Run the mode's generator and normalise to a StructuredResult.

    `entry` provides `.records()` (all chunk records of the generation) and
    `.generation`; blame/risk statistics are computed from those records.
    """
    assembled = outcome.assembled
    included = list(assembled.included_ids)
    retrieved = list(outcome.retrieved_ids)
    base = dict(retrieved_ids=retrieved, included_ids=included)
    if not included or not assembled.text.strip():
        return StructuredResult(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE, text=INSUFFICIENT_MESSAGE, **base
        )

    from generation import (
        BlameMapGenerator,
        DecisionMemoGenerator,
        DirectQAGenerator,
        RiskReportGenerator,
    )

    if mode is Mode.DIRECT:
        return DirectQAGenerator(llm=llm).generate_structured(
            question, assembled.text, included, retrieved
        )

    if mode is Mode.MEMO:
        try:
            result = DecisionMemoGenerator(llm=llm, strict_json=False).generate(
                question, assembled.text
            )
        except LLMUnavailableError as exc:
            return StructuredResult(
                status=AnswerStatus.PROVIDER_UNAVAILABLE, error=str(exc), **base
            )
        if result.memo is None:
            return StructuredResult(
                status=AnswerStatus.INVALID_OUTPUT,
                error="memo output was not valid JSON",
                model=result.model,
                **base,
            )
        # The memo may only cite chunks the model was actually shown.
        valid, invalid = split_citations(result.memo.evidence, included)
        if invalid or not valid:
            return StructuredResult(
                status=AnswerStatus.INVALID_OUTPUT,
                error=(f"memo cited chunk ids that were not in the context: {invalid}"
                       if invalid else "memo cited no evidence"),
                invalid_citations=invalid,
                model=result.model,
                **base,
            )
        data = result.memo.model_dump()
        data["evidence"] = valid
        return StructuredResult(
            status=AnswerStatus.ANSWERED,
            text=f"{result.memo.decision}\n\n{result.memo.rationale}".strip(),
            data=data,
            model=result.model,
            cited_ids=valid,
            **base,
        )

    # blame / risk: figures are computed from every chunk record in scope.
    generator_cls = BlameMapGenerator if mode is Mode.BLAME else RiskReportGenerator
    module = scope.modules[0] if scope.modules else ""
    try:
        result = generator_cls(llm=llm).generate(
            module,
            assembled.text,
            records=entry.records(),
            scope=scope,
            coverage_complete=coverage_fn(entry.generation),
        )
    except LLMUnavailableError as exc:
        return StructuredResult(
            status=AnswerStatus.PROVIDER_UNAVAILABLE, error=str(exc), **base
        )
    stats = result.stats.to_dict() if result.stats is not None else None
    return StructuredResult(
        status=AnswerStatus.ANSWERED,
        text=result.raw_text,
        data={"statistics": stats} if stats is not None else None,
        model=result.model,
        **base,
    )
