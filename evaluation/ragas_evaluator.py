from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval import (
    ChunkStore,
    HybridRetriever,
    GraphExpander,
    CrossEncoderReranker,
    ContextAssembler,
)
from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex
from generation.direct_qa import DirectQAGenerator
from config.settings import settings


@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth: str
    context: str


class RagasEvaluator:
    """Evaluate the pipeline with RAGAS if available."""

    def __init__(self) -> None:
        self._ragas = None
        try:
            import ragas  # type: ignore
            from datasets import Dataset  # type: ignore
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            self._ragas = {
                "ragas": ragas,
                "dataset_class": Dataset,
                "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
            }
        except Exception:
            self._ragas = None

    def evaluate(self, dataset_path: str, limit: int = 20) -> dict:
        if self._ragas is None:
            raise RuntimeError("ragas is not installed")

        pairs = self._load_pairs(dataset_path)[:limit]
        results, meta = self._run_pipeline(pairs)

        ragas = self._ragas["ragas"]
        dataset_class = self._ragas["dataset_class"]
        metrics = self._ragas["metrics"]
        data = {
            "question": [r.question for r in results],
            "answer": [r.answer for r in results],
            "contexts": [[r.context] for r in results],
            "ground_truth": [r.ground_truth for r in results],
        }
        dataset = dataset_class.from_dict(data)
        scores = ragas.evaluate(dataset, metrics=metrics)
        custom = self._compute_custom_metrics(meta)
        merged = dict(scores)
        merged.update(custom)
        return merged

    def _run_pipeline(self, pairs: list[dict]) -> tuple[list[EvalResult], dict]:
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
        meta = {
            "timestamps": [],
            "doc_types": [],
            "entity_tags": [],
            "answers": [],
        }
        for pair in pairs:
            q = pair.get("question", "")
            gt = pair.get("answer", "")
            candidates = retriever.retrieve(q, limit=40)
            expanded = expander.expand(candidates)
            reranked = reranker.rerank(q, expanded, top_k=12)
            context = assembler.assemble(reranked)
            meta["timestamps"].append([c.metadata.get("timestamp", "") for c in reranked])
            meta["doc_types"].append([c.metadata.get("doc_type", "") for c in reranked])
            meta["entity_tags"].append([c.metadata.get("entity_tags", []) for c in reranked])
            answer = generator.generate(q, context).answer
            meta["answers"].append(answer)
            results.append(EvalResult(question=q, answer=answer, ground_truth=gt, context=context))

        expander.close()
        retriever.close()
        qdrant.close()
        fts.close()
        generator.close()
        return results, meta

    def _load_pairs(self, dataset_path: str) -> list[dict]:
        path = Path(dataset_path)
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _compute_custom_metrics(self, meta: dict) -> dict:
        from evaluation.custom_metrics.temporal_accuracy import TemporalAccuracyMetric
        from evaluation.custom_metrics.multihop_coverage import MultiHopCoverageMetric
        from evaluation.custom_metrics.entity_consistency import EntityConsistencyMetric

        temporal = TemporalAccuracyMetric().score(
            meta.get("answers", []),
            meta.get("timestamps", []),
        )
        multihop = MultiHopCoverageMetric().score(meta.get("doc_types", []))
        entity = EntityConsistencyMetric().score(
            meta.get("answers", []),
            meta.get("entity_tags", []),
        )
        return {
            "temporal_accuracy": temporal,
            "multihop_coverage": multihop,
            "entity_consistency": entity,
        }
