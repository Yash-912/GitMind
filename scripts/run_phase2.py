"""scripts/run_phase2.py — Phase 2 orchestration: parse → entities → chunk → graph.

Usage:
    python scripts/run_phase2.py

Reads raw documents from SQLite (populated in Phase 1), runs the full
Phase 2 pipeline, and writes:
  - entity registry to the same SQLite DB
  - temporal graph edges to the same SQLite DB
  - chunked documents to data/chunks.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from ingestion.document_store import DocumentStore, Document
from sqlmodel import Session, select

from parsing.multi_schema_parser import MultiSchemaParser
from entities.entity_extractor import EntityExtractor
from entities.entity_registry import EntityRegistry
from entities.temporal_graph import TemporalGraphBuilder
from chunking import (
    CommitChunker,
    DiffChunker,
    PRChunker,
    IssueChunker,
    ChangelogChunker,
    Chunk,
)


def load_documents(store: DocumentStore, doc_type: str) -> list[dict]:
    with Session(store.engine) as session:
        rows = session.exec(
            select(Document).where(Document.doc_type == doc_type)
        ).all()
        return [json.loads(r.payload_json) for r in rows]


def run_phase2() -> None:
    print("=" * 60)
    print("GitMind — Phase 2: Parsing, Entities & Chunking")
    print("=" * 60)

    db_path = settings.db_path
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    repo = settings.github_repo or "unknown"

    store = DocumentStore(db_path)
    parser = MultiSchemaParser()
    extractor = EntityExtractor()
    registry = EntityRegistry(db_path)
    graph_builder = TemporalGraphBuilder(db_path)

    chunkers = {
        "commit": CommitChunker(),
        "diff": DiffChunker(),
        "pr": PRChunker(),
        "issue": IssueChunker(),
        "release": ChangelogChunker(),
    }

    all_chunks: list[Chunk] = []

    # ------------------------------------------------------------------ commits
    print("\n[1/5] Processing commits...")
    commit_payloads = load_documents(store, "commit")
    print(f"  Found {len(commit_payloads)} commit documents")
    parsed_commits = []
    for payload in commit_payloads:
        pc = parser.parse_commit(payload)
        parsed_commits.append(pc)

        # Entity extraction on commit message
        entities = extractor.extract(pc.message, doc_id=pc.sha, doc_type="commit")
        for e in entities:
            canonical = registry.resolve(e.text, e.label.lower())
            registry.add_doc_reference(canonical, pc.sha)

        # Commit chunks (message)
        chunks = chunkers["commit"].chunk(pc, repo=repo)
        all_chunks.extend(chunks)

        # Diff chunks
        diff_chunks = chunkers["diff"].chunk(pc, repo=repo)
        all_chunks.extend(diff_chunks)

    print(f"  Generated {sum(1 for c in all_chunks if c.metadata.doc_type in ('commit', 'diff'))} commit+diff chunks")

    # ------------------------------------------------------------------ PRs
    print("\n[2/5] Processing pull requests...")
    pr_payloads = load_documents(store, "pr")
    print(f"  Found {len(pr_payloads)} PR documents")
    parsed_prs = []
    for payload in pr_payloads:
        ppr = parser.parse_pr(payload)
        parsed_prs.append(ppr)

        entities = extractor.extract(
            ppr.title + "\n" + ppr.body_clean,
            doc_id=str(ppr.number),
            doc_type="pr",
        )
        for e in entities:
            canonical = registry.resolve(e.text, e.label.lower())
            registry.add_doc_reference(canonical, str(ppr.number))

        chunks = chunkers["pr"].chunk(ppr, repo=repo)
        all_chunks.extend(chunks)

    print(f"  PR chunks so far: {sum(1 for c in all_chunks if c.metadata.doc_type == 'pr')}")

    # ------------------------------------------------------------------ Issues
    print("\n[3/5] Processing issues...")
    issue_payloads = load_documents(store, "issue")
    print(f"  Found {len(issue_payloads)} issue documents")
    parsed_issues = []
    for payload in issue_payloads:
        pi = parser.parse_issue(payload)
        parsed_issues.append(pi)

        text = pi.title + "\n" + pi.body_clean + "\n" + " ".join(pi.comments)
        entities = extractor.extract(text, doc_id=str(pi.number), doc_type="issue")
        for e in entities:
            canonical = registry.resolve(e.text, e.label.lower())
            registry.add_doc_reference(canonical, str(pi.number))

        chunks = chunkers["issue"].chunk(pi, repo=repo)
        all_chunks.extend(chunks)

    print(f"  Issue chunks so far: {sum(1 for c in all_chunks if c.metadata.doc_type == 'issue')}")

    # ------------------------------------------------------------------ Releases
    print("\n[4/5] Processing releases...")
    release_payloads = load_documents(store, "release")
    print(f"  Found {len(release_payloads)} release documents")
    for payload in release_payloads:
        pr_rel = parser.parse_release(payload)
        chunks = chunkers["release"].chunk(pr_rel, repo=repo)
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ Temporal graph
    print("\n[5/5] Building temporal graph...")
    # Load links stored in Phase 1
    link_payloads = load_documents(store, "link")
    class _FakeLinkRecord:
        def __init__(self, p):
            self.source_type = p["source_type"]
            self.source_id = p["source_id"]
            self.target_type = p["target_type"]
            self.target_id = p["target_id"]
            self.relation = p["relation"]

    fake_links = [_FakeLinkRecord(p) for p in link_payloads]
    graph_builder.add_edges_from_links(fake_links)

    # Also build PR→issue edges from parsed PRs
    graph_builder.build_from_parsed_issues(parsed_prs, parsed_issues)

    print(f"  Graph edges: {graph_builder.edge_count()}")

    # ------------------------------------------------------------------ Write chunks
    chunks_path = data_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            record = {"text": chunk.text, "metadata": chunk.metadata.to_dict()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Phase 2 complete!")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Chunks written to: {chunks_path}")
    print(f"   Entity registry entries: {len(registry.all_entities())}")
    print(f"   Graph edges: {graph_builder.edge_count()}")

    registry.close()
    graph_builder.close()


if __name__ == "__main__":
    t0 = time.time()
    run_phase2()
    print(f"\n   Elapsed: {time.time() - t0:.1f}s")
