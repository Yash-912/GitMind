"""scripts/run_phase3.py — Phase 3: embed -> index one generation (Qdrant + BM25 + FTS5).

Usage:
    python scripts/run_phase3.py --repo owner/name --generation <id>
        [--root data/generations] [--allow-new-embedding-space] [--keep 2]

Phase 2 writes ``<root>/<repo_slug>/<generation>/chunks.jsonl`` into a *building*
generation.  This script indexes into that same generation only:

    BM25   -> <gen>/bm25/
    FTS5   -> <gen>/fts.db
    Qdrant -> collection ``gen.qdrant_collection``

Nothing is shared with, or overwrites, the currently ACTIVE generation.  The
generation is marked READY and published (atomic ACTIVE pointer swap) only
after *every* chunk was embedded with validated vectors and all three indexes
report the expected counts.  Any failure marks the generation FAILED, leaves
the previous ACTIVE generation serving, and exits non-zero.

Embedding space: the semantic (and code) embedding identity is stored in the
generation manifest.  If the previous ACTIVE generation used a different
identity (provider/model/dim), the run is refused unless
``--allow-new-embedding-space`` is given, because query vectors must match the
index.

Exit codes: 0 = published, 1 = failed (generation marked failed), 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from embedding.models import (
    OllamaEmbeddingClient,
    EmbeddingIdentity,
    EmbeddingIdentityMismatchError,
    CODE_DIM,
)
from embedding.embedding_cache import EmbeddingCache
from embedding.embedder import DualEmbedder
from indexing.generation import BUILDING, Generation, GenerationManager
from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex


class Phase3Error(RuntimeError):
    """A validation or integrity failure that must fail the generation."""


def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSONL file produced by Phase 2."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _validate_chunks(chunks: list[dict], repo: str) -> None:
    if not chunks:
        raise Phase3Error("chunks.jsonl contains no chunks")
    seen: set[str] = set()
    for i, c in enumerate(chunks):
        meta = c.get("metadata") or {}
        cid = meta.get("chunk_id", "")
        if not cid:
            raise Phase3Error(f"chunk #{i} has no metadata.chunk_id")
        if cid in seen:
            raise Phase3Error(f"duplicate chunk_id {cid!r}")
        seen.add(cid)
        if not (c.get("text") or "").strip():
            raise Phase3Error(f"chunk {cid} has empty text")
        if meta.get("repo") and meta["repo"] != repo:
            raise Phase3Error(f"chunk {cid} belongs to repo {meta['repo']!r}, not {repo!r}")


def _embedding_manifest(embedder: DualEmbedder) -> dict:
    sem, code = embedder.semantic_identity, embedder.code_identity
    return {
        "semantic": sem.to_dict() if sem else None,
        "code": code.to_dict() if code else None,
        "dual_vectors": code is not None,
    }


def _same_space(a: dict, b: dict) -> bool:
    def ident(d, k):
        return EmbeddingIdentity.from_dict(d[k]) if d.get(k) else None

    for key in ("semantic", "code"):
        x, y = ident(a, key), ident(b, key)
        if (x is None) != (y is None) or (x is not None and not x.matches(y)):
            return False
    return True


def run_phase3(
    generation_id: str,
    repo: str,
    *,
    root: str | Path | None = None,
    allow_new_embedding_space: bool = False,
    client: OllamaEmbeddingClient | None = None,
    qdrant_path: str | None = None,
    cache_dir: str | None = None,
    keep: int = 2,
    show_progress: bool = True,
) -> int:
    """Index generation *generation_id* of *repo*.  Returns a process exit code."""
    say = print if show_progress else (lambda *a, **k: None)
    root = Path(root) if root is not None else Path(settings.data_dir) / "generations"
    mgr = GenerationManager(root, repo)
    gen_path = mgr.base / generation_id
    if not (gen_path / "manifest.json").exists():
        print(f"ERROR: generation {generation_id!r} not found under {mgr.base}")
        return 2
    gen = Generation.load(gen_path)
    if gen.status != BUILDING:
        print(f"ERROR: generation {generation_id} is {gen.status!r}, expected 'building'")
        return 2
    if not gen.chunks_path.exists():
        print(f"ERROR: {gen.chunks_path} not found. Run Phase 2 into this generation first.")
        return 2

    embedder = None
    store = None
    created_collection = False
    try:
        say("=" * 60)
        say(f"GitMind - Phase 3: Embedding & Indexing  [{repo} / {generation_id}]")
        say("=" * 60)

        say("\n[1/5] Loading chunks...")
        chunks = load_chunks(str(gen.chunks_path))
        _validate_chunks(chunks, repo)
        say(f"  Loaded {len(chunks)} chunks")

        say("\n[2/5] Resolving embedding identity...")
        prev = mgr.active()
        prev_emb = dict(prev.embedding) if prev and prev.embedding else {}
        expected = (
            EmbeddingIdentity.from_dict(prev_emb["semantic"]) if prev_emb.get("semantic") else None
        )
        client = client or OllamaEmbeddingClient()
        cache = EmbeddingCache(cache_dir=str(cache_dir or settings.embedding_cache_dir))
        embedder = DualEmbedder(client=client, cache=cache, expected_semantic_identity=expected)
        refusal = (
            "embedding space differs from the active generation "
            f"({expected.key if expected else '?'}); re-run with "
            "--allow-new-embedding-space to build a new embedding space"
        )
        try:
            embedder.resolve_identities()
        except EmbeddingIdentityMismatchError as exc:
            if not allow_new_embedding_space:
                raise Phase3Error(f"{refusal} [{exc}]") from exc
            embedder.expected_semantic_identity = None
            embedder.semantic_identity = embedder.code_identity = None
            embedder.resolve_identities()
        embedding = _embedding_manifest(embedder)
        if prev_emb.get("semantic") and not _same_space(prev_emb, embedding):
            if not allow_new_embedding_space:
                raise Phase3Error(refusal)
            say("  NOTE: new embedding space (allowed by --allow-new-embedding-space)")
        say(f"  semantic: {embedder.semantic_identity.key}")
        say(f"  code:     {embedder.code_identity.key if embedder.code_identity else 'none (same model as prose)'}")

        say("\n[3/5] Embedding chunks...")
        report = embedder.embed_chunks_report(chunks, show_progress=show_progress)
        if report.failures:
            first = report.failures[0]
            raise Phase3Error(
                f"{len(report.failures)} embedding failure(s) over {len(chunks)} chunks "
                f"(first: {first.chunk_id} [{first.vector_kind}] {first.error_type}: {first.message})"
            )
        if len(report.chunks) != len(chunks):
            raise Phase3Error(
                f"embedded {len(report.chunks)} of {len(chunks)} chunks"
            )
        say(f"  Embedded {len(report.chunks)} chunks ({report.cache_hits} cache hits)")

        say("\n[4/5] Upserting into Qdrant...")
        sem, code = embedder.semantic_identity, embedder.code_identity
        store = QdrantStore(
            path=qdrant_path or settings.qdrant_path,
            collection=gen.qdrant_collection,
            semantic_dim=sem.dim,
            code_dim=code.dim if code else CODE_DIM,
            dual_vectors=code is not None,
            embedding=embedding,
            generation_id=gen.generation_id,
            repo=repo,
        )
        created_collection = True
        store.upsert_embedded_chunks(report.chunks)
        if store.count() != len(chunks):
            raise Phase3Error(f"Qdrant holds {store.count()} points, expected {len(chunks)}")

        say("\n[5/5] Building BM25 and FTS5 indexes...")
        ids = [c["metadata"]["chunk_id"] for c in chunks]
        BM25Index(str(gen.bm25_dir)).build(ids, [c["text"] for c in chunks])
        bm25_size = BM25Index(str(gen.bm25_dir)).size       # re-read from disk
        if bm25_size != len(chunks):
            raise Phase3Error(f"BM25 holds {bm25_size} documents, expected {len(chunks)}")
        fts = FTSIndex(str(gen.fts_path))
        try:
            fts.add_chunks(chunks)
            fts_count = fts.count()
        finally:
            fts.close()
        if fts_count != len(chunks):
            raise Phase3Error(f"FTS holds {fts_count} chunks, expected {len(chunks)}")

        # ---- Publish ----
        gen.embedding = embedding
        gen.mark_ready(chunk_count=len(chunks), dual_vectors=code is not None)
        mgr.publish(gen)
        removed = []
        try:
            removed = mgr.prune(keep=keep)
            for old in removed:
                if old.qdrant_collection:
                    store.delete_collection(old.qdrant_collection)
        except Exception as exc:  # noqa: BLE001 - published already; cleanup is best-effort
            print(f"WARNING: generation {generation_id} is ACTIVE but cleanup failed: {exc}")
        say(f"\nPhase 3 complete: generation {generation_id} is ACTIVE "
            f"({len(chunks)} chunks; pruned {len(removed)} old generation(s))")
        return 0

    except Exception as exc:  # noqa: BLE001 - any failure must fail the generation
        reason = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: Phase 3 failed; generation {generation_id} marked failed. {reason}")
        try:
            gen.mark_failed(reason)
        except Exception:
            pass
        if store is not None and created_collection:
            try:
                store.delete_collection()
            except Exception:
                pass
        return 1
    finally:
        if store is not None:
            store.close()
        if embedder is not None:
            embedder.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GitMind Phase 3: embed and index one generation")
    p.add_argument("--repo", required=True, help="repository identity, e.g. owner/name")
    p.add_argument("--generation", required=True, help="building generation id (from phase 2)")
    p.add_argument("--root", default=None, help="generations root (default: <data_dir>/generations)")
    p.add_argument("--allow-new-embedding-space", action="store_true",
                   help="permit a provider/model change relative to the ACTIVE generation")
    p.add_argument("--keep", type=int, default=2, help="generations to keep after publish")
    p.add_argument("--qdrant-path", default=None)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args(argv)
    return run_phase3(
        args.generation,
        args.repo,
        root=args.root,
        allow_new_embedding_space=args.allow_new_embedding_space,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        keep=args.keep,
    )


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"\n   Elapsed: {time.time() - t0:.1f}s")
    sys.exit(code)
