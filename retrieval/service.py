"""service.py — Scope-aware retrieval pipeline loaded once per index generation.

``RetrievalService.retrieve`` is the single entry point the API calls:

    plan -> typed entities -> (narrowed) scope -> multi-query hybrid retrieval
         -> intent-routed graph expansion -> rerank -> budgeted context

Every branch obeys the same ``Scope``; the final candidate list is re-checked
against it. Failures that degrade quality (no reranker, no graph, LLM planner
down, module filter matched nothing) are returned as ``warnings`` rather than
hidden. Embedding failures and identity mismatches are NOT swallowed.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .chunk_store import ChunkStore
from .context_assembler import AssembledContext, ContextAssembler
from .graph_walker import GraphPathWalker, PathHop
from .scope import Scope, ScopeError
from .types import CandidateChunk, QueryPlan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    """How an intent changes retrieval."""

    doc_type_weights: dict[str, float] = field(default_factory=dict)
    expand_graph: bool = True
    causal: bool = False          # question asks for cause/rationale -> needs causal edges
    chronological: bool = False   # present context oldest-first


INTENT_ROUTES: dict[str, Route] = {
    "ownership": Route({"commit": 1.3, "pr": 1.1}, expand_graph=False),
    "risk": Route({"issue": 1.3, "pr": 1.2}),
    "decision": Route({"pr": 1.3, "issue": 1.3, "release": 1.1}, causal=True),
    "why": Route({"pr": 1.3, "issue": 1.3, "commit": 1.1}, causal=True),
    "temporal": Route(chronological=True),
    "general": Route(),
}


@dataclass
class RetrievalOutcome:
    plan: QueryPlan
    candidates: list[CandidateChunk]          # reranked, trimmed to top_k
    assembled: AssembledContext
    graph_paths: list[list[PathHop]]          # evidenced paths (never flat neighbours)
    warnings: list[str]
    reranker_available: bool
    retrieved_ids: list[str]                  # all candidates before reranking (fused + graph)
    scope: Scope | None = None                # scope actually applied (may be narrowed)


class RetrievalService:
    """Retrieval pipeline. Build with ``from_generation`` or inject every part."""

    def __init__(
        self,
        *,
        chunk_store: ChunkStore,
        retriever: Any,
        reranker: Any,
        decomposer: Any,
        resolver: Any = None,
        graph_walker: GraphPathWalker | None = None,
        assembler_factory: Callable[[str], ContextAssembler] | None = None,
        startup_warnings: list[str] | None = None,
        closers: list[Callable[[], None]] | None = None,
    ) -> None:
        self.chunk_store = chunk_store
        self.retriever = retriever
        self.reranker = reranker
        self.decomposer = decomposer
        self.resolver = resolver
        self.graph_walker = graph_walker
        self._assembler_factory = assembler_factory or (
            lambda order: ContextAssembler(max_tokens=3000, order=order)
        )
        self.startup_warnings = list(startup_warnings or [])
        self._closers = list(closers or [])

    # ------------------------------------------------------------ construction

    @classmethod
    def from_generation(
        cls,
        gen: Any,
        settings: Any,
        *,
        embedder: Any = None,
        reranker: Any = None,
        decomposer: Any = None,
        resolver: Any = None,
        graph: Any = None,
    ) -> "RetrievalService":
        """Load everything for one active generation (do this once, not per query).

        Raises the typed embedding errors (unavailable / identity mismatch)
        instead of continuing with a query vector space that does not match the
        collection.
        """
        from indexing.bm25_index import BM25Index
        from indexing.fts_index import FTSIndex
        from indexing.qdrant_store import QdrantStore

        from .graph_walker import GraphPathWalker
        from .hybrid_retriever import HybridRetriever
        from .query_decomposer import QueryDecomposer
        from .query_embedder import QueryEmbedder
        from .reranker import CrossEncoderReranker

        identity = dict(gen.embedding) if gen.embedding else None
        embedder = embedder or QueryEmbedder(expected_identity=identity)
        embedder.check_collection_identity(identity)  # fail fast, before opening stores

        warnings: list[str] = []
        closers: list[Callable[[], None]] = []
        try:
            chunk_store = ChunkStore(str(gen.chunks_path))
            bm25 = BM25Index(index_dir=str(gen.bm25_dir))
            fts = FTSIndex(db_path=str(gen.fts_path))
            closers.append(fts.close)
            qdrant_kwargs: dict[str, Any] = {}
            if identity and identity.get("dim"):
                qdrant_kwargs["semantic_dim"] = int(identity["dim"])
            qdrant = QdrantStore(
                path=settings.qdrant_path, collection_name=gen.qdrant_collection, **qdrant_kwargs
            )
            closers.append(qdrant.close)
            retriever = HybridRetriever(
                qdrant=qdrant, bm25=bm25, fts=fts, chunk_store=chunk_store, embedder=embedder
            )
            closers.append(retriever.close)

            reranker = reranker or CrossEncoderReranker()
            decomposer = decomposer or QueryDecomposer()
            closers.append(decomposer.close)

            if resolver is None:
                try:
                    from .entity_resolver import EntityResolver

                    resolver = EntityResolver(settings.db_path)
                    closers.append(resolver.close)
                except Exception as exc:  # noqa: BLE001 - reported, retrieval still works
                    warnings.append(f"entity resolver unavailable: {type(exc).__name__}: {exc}")

            walker = None
            if graph is None:
                try:
                    from entities.temporal_graph import TemporalGraph  # contract class

                    graph = TemporalGraph(settings.db_path)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"graph unavailable: {type(exc).__name__}: {exc}")
            if graph is not None:
                if hasattr(graph, "neighbors"):
                    walker = GraphPathWalker(graph, chunk_store)
                else:
                    warnings.append("graph has no neighbors(); graph expansion disabled")
            return cls(
                chunk_store=chunk_store, retriever=retriever, reranker=reranker,
                decomposer=decomposer, resolver=resolver, graph_walker=walker,
                startup_warnings=warnings, closers=closers,
            )
        except Exception:
            for close in reversed(closers):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    log.exception("cleanup failed while aborting service construction")
            raise

    # ----------------------------------------------------------------- query

    @property
    def reranker_available(self) -> bool:
        return bool(self.reranker is not None and getattr(self.reranker, "available", True))

    def retrieve(
        self, question: str, scope: Scope, top_k: int = 12, limit: int = 40
    ) -> RetrievalOutcome:
        warnings = list(self.startup_warnings)
        plan = self.decomposer.decompose(question)
        warnings.extend(plan.warnings)
        route = INTENT_ROUTES.get(plan.intent, INTENT_ROUTES["general"])

        self._classify_entities(question, plan, warnings)

        queries = list(dict.fromkeys([question, *plan.sub_queries]))
        boosts = [] if plan.authorship else list(plan.people)

        def run(sc: Scope) -> list[CandidateChunk]:
            return self.retriever.retrieve_many(
                queries, limit=limit, scope=sc,
                doc_type_weights=route.doc_type_weights, boost_terms=boosts,
            )

        effective = scope
        narrowed, notes = self._narrow(scope, plan, warnings)
        candidates: list[CandidateChunk] = []
        if narrowed is not scope:
            candidates = run(narrowed)
            if candidates:
                effective = narrowed
            else:
                warnings.append(
                    f"{notes} matched no candidates; retried without those filters"
                )
        if not candidates:
            candidates = run(scope)

        graph_paths: list[list[PathHop]] = []
        if self.graph_walker is not None and route.expand_graph and candidates:
            try:
                expansion = self.graph_walker.expand(candidates, effective, causal=route.causal)
                candidates = expansion.candidates
                graph_paths = expansion.paths
                warnings.extend(expansion.warnings)
            except Exception as exc:  # noqa: BLE001 - visible degradation
                warnings.append(f"graph expansion failed: {type(exc).__name__}: {exc}")
        elif route.causal and self.graph_walker is None:
            warnings.append("no causal evidence found: graph unavailable")

        # Final exclusion check across every branch, graph included.
        candidates = [c for c in candidates if effective.matches(c.metadata)]
        retrieved_ids = [c.chunk_id for c in candidates]

        if self.reranker is None:
            reranked = candidates[:top_k]
            warnings.append("reranker unavailable (not configured); using retrieval order")
        else:
            reranked = self.reranker.rerank(question, candidates, top_k=top_k)
            if not self.reranker_available:
                reason = getattr(self.reranker, "load_error", None)
                warnings.append(
                    "reranker unavailable" + (f" ({reason})" if reason else "") + "; using retrieval order"
                )

        assembler = self._assembler_factory("chronological" if route.chronological else "rank")
        assembled = assembler.assemble_with_report(reranked)
        return RetrievalOutcome(
            plan=plan,
            candidates=reranked,
            assembled=assembled,
            graph_paths=graph_paths,
            warnings=list(dict.fromkeys(warnings)),
            reranker_available=self.reranker_available,
            retrieved_ids=retrieved_ids,
            scope=effective,
        )

    # --------------------------------------------------------------- helpers

    def _classify_entities(self, question: str, plan: QueryPlan, warnings: list[str]) -> None:
        """Fill plan.modules / people / technologies from typed entity resolution (F05)."""
        if self.resolver is None:
            return
        try:
            if plan.entities:
                resolved = self.resolver.resolve_typed(plan.entities)
            else:
                resolved = self.resolver.resolve_from_query_typed(question)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"entity resolution failed: {type(exc).__name__}: {exc}")
            return
        for e in resolved:
            if e.kind == "module":
                plan.modules.append(e.name)
            elif e.kind == "person":
                plan.people.append(e.name)
            elif e.kind == "technology":
                plan.technologies.append(e.name)  # recorded only; never a filter

    @staticmethod
    def _narrow(scope: Scope, plan: QueryPlan, warnings: list[str]) -> tuple[Scope, str]:
        """Add soft narrowing (modules, authors on authorship queries, plan time range).

        The caller's own explicit filters always win and are never overridden.
        Returns the (possibly new) scope and a description of what was added.
        """
        changes: dict[str, Any] = {}
        notes: list[str] = []
        if plan.modules and not scope.modules:
            changes["modules"] = tuple(dict.fromkeys(plan.modules))
            notes.append(f"module filter {list(changes['modules'])}")
        if plan.authorship and plan.people and not scope.authors:
            changes["authors"] = tuple(dict.fromkeys(plan.people))
            notes.append(f"author filter {list(changes['authors'])}")
        if (plan.time_start or plan.time_end) and not (scope.time_start or scope.time_end):
            changes["time_start"] = plan.time_start
            changes["time_end"] = plan.time_end
            notes.append("time filter")
        if not changes:
            return scope, ""
        try:
            return dataclasses.replace(scope, **changes), " and ".join(notes)
        except ScopeError as exc:
            warnings.append(f"ignored invalid plan filters: {exc}")
            return scope, ""

    def close(self) -> None:
        for close in reversed(self._closers):
            try:
                close()
            except Exception:  # noqa: BLE001
                log.exception("error while closing retrieval service")
        self._closers = []
