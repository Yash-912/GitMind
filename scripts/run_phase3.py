"""scripts/run_phase3.py — Phase 3 orchestration: embed → index into Qdrant + BM25 + FTS5.

Usage:
    python scripts/run_phase3.py

Reads chunked documents from data/chunks.jsonl (produced by Phase 2),
embeds them via Ollama, and populates all three indexes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from embedding.models import OllamaEmbeddingClient, PROSE_MODEL, CODE_MODEL
from embedding.embedding_cache import EmbeddingCache
from embedding.embedder import DualEmbedder
from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex


def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSONL file produced by Phase 2."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def run_phase3() -> None:
    print("=" * 60)
    print("GitMind — Phase 3: Embedding & Indexing")
    print("=" * 60)

    data_dir = Path(settings.data_dir)
    chunks_path = data_dir / "chunks.jsonl"

    if not chunks_path.exists():
        print(f"❌ {chunks_path} not found. Run Phase 2 first (scripts/run_phase2.py)")
        sys.exit(1)

    # ---- Load chunks ----
    print("\n[1/5] Loading chunks...")
    chunks = load_chunks(str(chunks_path))
    print(f"  Loaded {len(chunks)} chunks")

    # ---- Check Ollama readiness ----
    print("\n[2/5] Checking Ollama...")
    client = OllamaEmbeddingClient()
    if not client.is_available():
        print("  ❌ Ollama is not running! Start it with: ollama serve")
        print("  Then pull models:")
        print(f"    ollama pull {PROSE_MODEL}")
        print(f"    ollama pull {CODE_MODEL}")
        sys.exit(1)

    prose_ok = client.has_model(PROSE_MODEL)
    code_ok = client.has_model(CODE_MODEL)
    print(f"  {PROSE_MODEL}: {'✅' if prose_ok else '❌ (run: ollama pull ' + PROSE_MODEL + ')'}")
    print(f"  {CODE_MODEL}: {'✅' if code_ok else '❌ (run: ollama pull ' + CODE_MODEL + ')'}")
    if not (prose_ok and code_ok):
        print("  Pull missing models and re-run.")
        sys.exit(1)

    # ---- Embed ----
    print("\n[3/5] Embedding chunks...")
    cache = EmbeddingCache(cache_dir=str(data_dir / ".diskcache" / "embeddings"))
    embedder = DualEmbedder(client=client, cache=cache)
    embedded = embedder.embed_chunks(chunks, show_progress=True)
    print(f"  Embedded {len(embedded)} chunks total")
    code_count = sum(1 for e in embedded if e.code_vector and any(v != 0.0 for v in e.code_vector))
    print(f"  Of which {code_count} also have code vectors")

    # ---- Qdrant upsert ----
    print("\n[4/5] Upserting into Qdrant...")
    qdrant = QdrantStore(path=str(data_dir / ".qdrant"))
    upserted = qdrant.upsert_embedded_chunks(embedded)
    print(f"  Qdrant points: {qdrant.count()}")

    # ---- BM25 index ----
    print("\n[5a/5] Building BM25 index...")
    bm25 = BM25Index(index_dir=str(data_dir / "bm25_index"))
    chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    bm25.build(chunk_ids, chunk_texts)
    print(f"  BM25 index size: {bm25.size}")

    # ---- FTS5 index ----
    print("\n[5b/5] Building FTS5 index...")
    fts = FTSIndex(db_path=settings.db_path)
    fts_count = fts.add_chunks(chunks)
    print(f"  FTS5 chunks indexed: {fts.count()}")

    # ---- Summary ----
    print(f"\n✅ Phase 3 complete!")
    print(f"   Qdrant vectors:  {qdrant.count()}")
    print(f"   BM25 documents:  {bm25.size}")
    print(f"   FTS5 chunks:     {fts.count()}")
    print(f"   Embedding cache: {cache.size} entries")

    embedder.close()
    qdrant.close()
    fts.close()


if __name__ == "__main__":
    t0 = time.time()
    run_phase3()
    print(f"\n   Elapsed: {time.time() - t0:.1f}s")
