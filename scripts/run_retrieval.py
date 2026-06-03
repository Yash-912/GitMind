from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import settings
from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex
from retrieval import (
    ChunkStore,
    HybridRetriever,
    GraphExpander,
    CrossEncoderReranker,
    ContextAssembler,
    QueryDecomposer,
    EntityResolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitMind Phase 4 retrieval demo")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--chunks-path", default=str(Path(settings.data_dir) / "chunks.jsonl"))
    parser.add_argument("--db-path", default=settings.db_path)
    return parser.parse_args()


def main() -> None:
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    args = parse_args()

    chunk_store = ChunkStore(args.chunks_path)
    qdrant = QdrantStore(path=settings.qdrant_path)
    bm25 = BM25Index(index_dir=settings.bm25_index_dir)
    fts = FTSIndex(db_path=settings.db_path)

    decomposer = QueryDecomposer()
    plan = decomposer.decompose(args.query)
    resolver = EntityResolver(args.db_path)
    module_tags = resolver.resolve(plan.entities)

    retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, fts=fts, chunk_store=chunk_store)
    candidates = retriever.retrieve(
        args.query,
        limit=args.limit,
        filters={
            "time_start": plan.time_start,
            "time_end": plan.time_end,
            "module_tags": module_tags,
        },
    )

    expander = GraphExpander(db_path=args.db_path, chunk_store=chunk_store, hop_depth=1)
    expanded = expander.expand(candidates)

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(args.query, expanded, top_k=args.top_k)

    assembler = ContextAssembler(max_tokens=3000)
    context = assembler.assemble(reranked)

    print("\nTop chunks:")
    for c in reranked:
        doc_type = c.metadata.get("doc_type", "")
        doc_id = c.metadata.get("doc_id", "")
        print(f"- {doc_type}:{doc_id} score={c.score:.4f} source={c.source}")

    print("\nContext preview:\n")
    print(context)

    expander.close()
    retriever.close()
    decomposer.close()
    resolver.close()
    qdrant.close()
    fts.close()


if __name__ == "__main__":
    main()
