from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config.settings import settings

from .custom_metrics import (
    EntityConsistencyMetric,
    MultiHopCoverageMetric,
    TemporalAccuracyMetric,
)

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


class RagasUnavailableError(RuntimeError):
    """RAGAS (or its judge/embedding wiring) could not be loaded or run.

    The original exception is chained (``__cause__``) and named in the message.
    """


@dataclass
class RagasConfig:
    """Explicit judge + embedding configuration for RAGAS."""

    judge_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-2"
    api_key: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_settings(cls, cfg: Any = None) -> "RagasConfig":
        """Read from settings; RAGAS_JUDGE_MODEL / RAGAS_EMBEDDING_MODEL env override."""
        cfg = cfg or settings
        return cls(
            judge_model=os.environ.get("RAGAS_JUDGE_MODEL")
            or getattr(cfg, "ragas_judge_model", None)
            or cls.judge_model,
            embedding_model=os.environ.get("RAGAS_EMBEDDING_MODEL")
            or getattr(cfg, "ragas_embedding_model", None)
            or cls.embedding_model,
            api_key=getattr(cfg, "gemini_api_key", None),
        )

    def identities(self) -> dict[str, str]:
        return {
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "judge_temperature": str(self.temperature),
        }


class RagasBackend:
    """Real RAGAS execution. UNVERIFIED against a live ragas install in this repo.

    Imports are deferred to ``__init__`` and any failure is re-raised as
    ``RagasUnavailableError`` carrying the real exception text.
    """

    def __init__(self, config: RagasConfig) -> None:
        self.config = config
        if not config.api_key:
            raise RagasUnavailableError("RAGAS judge needs GEMINI_API_KEY (settings.gemini_api_key is empty)")
        try:
            import ragas  # type: ignore
            from datasets import Dataset  # type: ignore
            from langchain_google_genai import (  # type: ignore
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
            from ragas.llms import LangchainLLMWrapper  # type: ignore
            from ragas import metrics as m  # type: ignore

            self._ragas = ragas
            self._dataset_class = Dataset
            self._metrics = [getattr(m, name) for name in RAGAS_METRIC_NAMES]
            self._llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model=config.judge_model,
                    google_api_key=config.api_key,
                    temperature=config.temperature,
                )
            )
            self._embeddings = LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(model=config.embedding_model, google_api_key=config.api_key)
            )
        except Exception as exc:  # noqa: BLE001
            raise RagasUnavailableError(f"RAGAS setup failed: {type(exc).__name__}: {exc}") from exc

    def evaluate(self, data: dict[str, list]) -> dict[str, float]:
        try:
            result = self._ragas.evaluate(
                self._dataset_class.from_dict(data),
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )
            frame = result.to_pandas()
            return {name: float(frame[name].mean()) for name in RAGAS_METRIC_NAMES if name in frame}
        except Exception as exc:  # noqa: BLE001
            raise RagasUnavailableError(f"RAGAS evaluation failed: {type(exc).__name__}: {exc}") from exc


@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth: str
    context: str


@dataclass
class EvalReport:
    metrics: dict[str, float | None]
    per_question: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    identities: dict[str, str] = field(default_factory=dict)
    latency_s: dict[str, float] = field(default_factory=dict)
    mlflow_logged: bool | None = None


PipelineRunner = Callable[[list[dict]], tuple[list[EvalResult], dict]]


class RagasEvaluator:
    """Evaluate the pipeline with RAGAS plus the per-claim custom metrics.

    ``backend`` and ``pipeline_runner`` are injectable so the configuration and
    aggregation logic can be unit-tested offline. Nothing is skipped silently:
    RAGAS failures raise ``RagasUnavailableError`` with the real cause, and
    non-applicable / NaN metrics are reported as ``None`` with a warning.
    """

    def __init__(
        self,
        config: RagasConfig | None = None,
        backend: Any = None,
        pipeline_runner: PipelineRunner | None = None,
        logger: Any = None,
    ) -> None:
        self.config = config or RagasConfig.from_settings()
        self._backend = backend
        self._pipeline_runner = pipeline_runner
        self.logger = logger
        self.last_report: EvalReport | None = None

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = RagasBackend(self.config)
        return self._backend

    def evaluate(
        self,
        dataset_path: str,
        limit: int = 20,
        run_name: str | None = None,
        extra_params: dict[str, Any] | None = None,
        split: str | None = "eval",
    ) -> dict:
        pairs = self._load_pairs(dataset_path)
        if split:
            pairs = [p for p in pairs if p.get("split", "eval") == split]
        pairs = pairs[:limit]
        backend = self.backend  # fail fast, with the real error, before running the pipeline
        runner = self._pipeline_runner or self._run_pipeline
        results, meta = runner(pairs)

        data = {
            "question": [r.question for r in results],
            "answer": [r.answer for r in results],
            "contexts": [[r.context] for r in results],
            "ground_truth": [r.ground_truth for r in results],
        }
        report_warnings: list[str] = []
        raw = backend.evaluate(data)
        merged: dict[str, float | None] = {}
        for name in RAGAS_METRIC_NAMES:
            if name not in raw:
                report_warnings.append(f"RAGAS did not return metric '{name}'")
                merged[name] = None
                continue
            val = raw[name]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                report_warnings.append(f"RAGAS metric '{name}' is NaN (judge call failed?)")
                merged[name] = None
            else:
                merged[name] = float(val)
        merged.update(self._compute_custom_metrics(meta, pairs, results))
        for name, val in merged.items():
            if val is None and name not in RAGAS_METRIC_NAMES:
                report_warnings.append(f"metric '{name}' not applicable to any question in this dataset")
        for w in report_warnings:
            warnings.warn(w, RuntimeWarning, stacklevel=2)

        per_question = self._per_question(pairs, results, meta)
        latencies = meta.get("latencies", [])
        report = EvalReport(
            metrics=merged,
            per_question=per_question,
            warnings=report_warnings,
            identities=self.config.identities(),
            latency_s={
                "total": float(sum(latencies)),
                "mean": float(sum(latencies) / len(latencies)) if latencies else 0.0,
            },
        )
        if self.logger is not None:
            report.mlflow_logged = self.logger.log_run(
                {**report.identities, "n_questions": len(results), "dataset": str(dataset_path)},
                {**merged, "latency_total_s": report.latency_s["total"], "latency_mean_s": report.latency_s["mean"]},
                run_name=run_name,
                extra_params=extra_params,
                dataset_path=dataset_path,
                per_question=per_question,
            )
        self.last_report = report
        return dict(merged)

    def _run_pipeline(self, pairs: list[dict]) -> tuple[list[EvalResult], dict]:
        from generation.direct_qa import DirectQAGenerator
        from indexing.bm25_index import BM25Index
        from indexing.fts_index import FTSIndex
        from indexing.qdrant_store import QdrantStore
        from retrieval import (
            ChunkStore,
            ContextAssembler,
            CrossEncoderReranker,
            GraphExpander,
            HybridRetriever,
        )

        chunk_store = ChunkStore(str(Path(settings.data_dir) / "chunks.jsonl"))
        qdrant = QdrantStore(path=settings.qdrant_path)
        bm25 = BM25Index(index_dir=settings.bm25_index_dir)
        fts = FTSIndex(db_path=settings.db_path)

        retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, fts=fts, chunk_store=chunk_store)
        expander = GraphExpander(db_path=settings.db_path, chunk_store=chunk_store, hop_depth=1)
        reranker = CrossEncoderReranker()
        assembler = ContextAssembler(max_tokens=3000)
        generator = DirectQAGenerator()

        results: list[EvalResult] = []
        meta: dict[str, list] = {
            "timestamps": [], "doc_types": [], "entity_tags": [], "answers": [],
            "cited_ids": [], "latencies": [],
        }
        try:
            for pair in pairs:
                q = pair.get("question", "")
                gt = pair.get("answer", "")
                t0 = time.perf_counter()
                candidates = retriever.retrieve(q, limit=40)
                expanded = expander.expand(candidates)
                reranked = reranker.rerank(q, expanded, top_k=12)
                context = assembler.assemble(reranked)
                gen = generator.generate(q, context)
                meta["latencies"].append(time.perf_counter() - t0)
                meta["timestamps"].append([str(c.metadata.get("timestamp", "")) for c in reranked])
                meta["doc_types"].append([c.metadata.get("doc_type", "") for c in reranked])
                meta["entity_tags"].append([c.metadata.get("entity_tags", []) for c in reranked])
                meta["answers"].append(gen.answer)
                evidence = set(gen.evidence)
                meta["cited_ids"].append(
                    [
                        c.chunk_id
                        for c in reranked
                        if f"{c.metadata.get('doc_type', '')}:{c.metadata.get('doc_id', '')}" in evidence
                        or c.chunk_id in gen.answer
                    ]
                )
                results.append(EvalResult(question=q, answer=gen.answer, ground_truth=gt, context=context))
        finally:
            expander.close()
            retriever.close()
            qdrant.close()
            fts.close()
            generator.close()
        return results, meta

    def _load_pairs(self, dataset_path: str) -> list[dict]:
        records: list[dict] = []
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _compute_custom_metrics(
        self, meta: dict, pairs: list[dict] | None = None, results: list[EvalResult] | None = None
    ) -> dict[str, float | None]:
        answers = meta.get("answers") or [r.answer for r in (results or [])]
        contexts = [r.context for r in (results or [])] or None
        truths = [r.ground_truth for r in (results or [])] or None
        questions = [r.question for r in (results or [])] or None
        required = [list((p or {}).get("required_chunk_ids") or []) for p in (pairs or [])]
        required += [[]] * (len(answers) - len(required))
        cited = list(meta.get("cited_ids") or []) + [[]] * (len(answers) - len(meta.get("cited_ids") or []))
        return {
            "temporal_accuracy": TemporalAccuracyMetric().score(
                answers, meta.get("timestamps", []), contexts, truths
            ),
            "multihop_coverage": MultiHopCoverageMetric().score(cited, required),
            "entity_consistency": EntityConsistencyMetric().score(
                answers, meta.get("entity_tags", []), contexts, truths, questions
            ),
        }

    def _per_question(self, pairs: list[dict], results: list[EvalResult], meta: dict) -> list[dict[str, Any]]:
        rows = []
        lat = meta.get("latencies", [])
        for i, r in enumerate(results):
            p = pairs[i] if i < len(pairs) else {}
            rows.append(
                {
                    "question": r.question,
                    "answer": r.answer,
                    "ground_truth": r.ground_truth,
                    "qa_type": p.get("qa_type", ""),
                    "split": p.get("split", ""),
                    "latency_s": lat[i] if i < len(lat) else None,
                    "cited_ids": (meta.get("cited_ids") or [[]] * len(results))[i],
                }
            )
        return rows
