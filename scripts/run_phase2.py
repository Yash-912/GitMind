"""scripts/run_phase2.py — Phase 2 orchestration: parse → entities → chunk → graph.

Usage:
    python scripts/run_phase2.py --repo owner/name

Reads raw documents of ONE repository from SQLite (populated in Phase 1), runs
the full Phase 2 pipeline, and writes:
  - repo-scoped entity registry to the same SQLite DB
  - repo-scoped temporal graph edges to the same SQLite DB
  - chunked documents to a new *generation*:
        <data_dir>/generations/<repo_slug>/<generation_id>/chunks.jsonl
    (plus the generation manifest). The generation is left in state "building":
    phase 3 indexes into the same generation (``--generation <id>``), marks it
    ready and publishes it. Phase 2 never publishes.

The generation id is printed as ``GENERATION_ID=<id>`` and returned by
``run_phase2``.
"""
from __future__ import annotations

import json
import os
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from ingestion.document_store import DocumentStore, Document
from sqlmodel import Session, select

from parsing.multi_schema_parser import MultiSchemaParser
from entities.entity_extractor import EntityExtractor
from entities.entity_registry import EntityRegistry
from entities.temporal_graph import TemporalGraphBuilder
from entities.tagging import (
    module_tags_for_paths,
    paths_in_text,
    resolve_entities,
    split_tags,
    stamp_tags,
)
from contextlib import nullcontext

from indexing.generation import BUILDING, Generation, GenerationManager
from scripts.run_snapshot import merge_into_generation, read_jsonl, run_snapshot
from chunking import (
    CommitChunker,
    DiffChunker,
    PRChunker,
    IssueChunker,
    ChangelogChunker,
    LogChunker,
    Chunk,
)


def load_documents(store: DocumentStore, doc_type: str, repo: str) -> list[dict]:
    with Session(store.engine) as session:
        rows = session.exec(
            select(Document).where(Document.doc_type == doc_type, Document.repo == repo)
        ).all()
        return [json.loads(r.payload_json) for r in rows]


def select_repository(store: DocumentStore, requested: str | None) -> str:
    """Never combine documents from unrelated repositories during parsing."""
    with Session(store.engine) as session:
        available = set(session.exec(select(Document.repo).distinct()).all())
    if requested == "legacy":
        if "" not in available:
            raise ValueError("No legacy documents are present")
        return ""
    if requested:
        from ingestion.repository import normalize_github_repo
        requested = requested if requested.startswith("file:") else normalize_github_repo(requested)
        if requested not in available:
            raise ValueError("No documents exist for this repository. Ingest it first; use --repo legacy only to inspect old unscoped data.")
        return requested
    scoped = available - {""}
    if len(scoped) == 1:
        return next(iter(scoped))
    if scoped:
        raise ValueError("Multiple repositories are present. Select one with --repo.")
    if "" in available:
        raise ValueError("Only unscoped legacy data is present. Re-ingest the repository, or explicitly select --repo legacy.")
    raise ValueError("No documents are present. Run ingestion first.")


def pinned_sha(store: DocumentStore, repo: str) -> str:
    """Pinned commit SHA of the most recent ingestion manifest that recorded one."""
    best = ("", "")
    for manifest in load_documents(store, "ingestion_manifest", repo):
        sha = ((manifest.get("sources") or {}).get("git") or {}).get("pinned_sha")
        if sha and (manifest.get("finished_at") or "") >= best[0]:
            best = (manifest.get("finished_at") or "", sha)
    return best[1]


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _open_building(mgr: GenerationManager, generation_id: str) -> Generation:
    gen = Generation.load(mgr.base / generation_id)
    if gen.repo != mgr.repo:
        raise ValueError("generation belongs to a different repository")
    if gen.status != BUILDING:
        raise ValueError(f"generation {generation_id} is {gen.status!r}, not building")
    return gen


def _register_snapshot(records: list[dict], registry: EntityRegistry, graph: TemporalGraphBuilder, repo: str) -> dict[str, int]:
    """Module/symbol entities and ``file_<path>`` nodes for snapshot (code) chunks."""
    modules: dict[tuple[str, str], None] = {}
    symbols: dict[tuple[str, str], None] = {}
    files: dict[str, None] = {}
    for rec in records:
        m = rec.get("metadata", {})
        path = m.get("source_path") or m.get("doc_id", "")
        if not path:
            continue
        files[path] = None
        for tag in m.get("module_tags", []):
            modules[(tag, path)] = None
        for name in m.get("entity_tags", []):
            if name:
                symbols[(name, path)] = None
    for path in files:
        graph.add_node("file", path, repo=repo)
    for name, path in modules:
        registry.register(name, "module")
        registry.add_doc_reference(name, path, doc_type="code", kind="module")
    for name, path in symbols:
        registry.register(name, "symbol")
        registry.add_doc_reference(name, path, doc_type="code", kind="symbol")
    return {"files": len(files), "modules": len({n for n, _ in modules}), "symbols": len({n for n, _ in symbols})}


def _tag_document(
    chunks: list[Chunk],
    extractor,
    registry: EntityRegistry,
    text: str,
    doc_id: str,
    doc_type: str,
    author: str = "",
    file_paths: list[str] | None = None,
) -> None:
    """Resolve entities for one document and stamp module/entity tags on its chunks."""
    resolved = resolve_entities(extractor, registry, text, doc_id, doc_type, author=author)
    modules, entities = split_tags(resolved)
    path_modules = module_tags_for_paths([*(file_paths or []), *paths_in_text(text)])
    for tag in path_modules:  # path-derived modules are exact; never fuzzy-merged
        registry.register(tag, "module")
        registry.add_doc_reference(tag, doc_id, doc_type=doc_type, kind="module")
    stamp_tags(chunks, [*path_modules, *modules], entities)


def run_phase2(
    repo_id: str | None = None,
    *,
    db_path: str | None = None,
    data_dir: str | Path | None = None,
    generation_id: str | None = None,
    repo_path: str | Path | None = None,
    ref: str = "HEAD",
) -> str:
    """Run phase 2 for one repository and return the generation id it wrote.

    ``generation_id`` writes into an existing *building* generation instead of
    creating one. ``repo_path`` also rebuilds the current-code (snapshot) view at
    ``ref`` through run_snapshot's shared merge logic. Without ``repo_path`` a new
    generation carries the active generation's snapshot records forward unchanged.
    """
    print("=" * 60)
    print("GitMind — Phase 2: Parsing, Entities & Chunking")
    print("=" * 60)

    db_path = db_path or settings.db_path
    data_dir = Path(data_dir or settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = DocumentStore(db_path)
    repo = select_repository(store, repo_id)
    if not repo:
        print("WARNING: Legacy documents have no verified repository identity.")
    parser = MultiSchemaParser()
    extractor = EntityExtractor()
    print(f"Entity extractor: {extractor.describe()}")
    registry = EntityRegistry(db_path, repo=repo)
    graph_builder = TemporalGraphBuilder(db_path)
    # Everything this run emits is tracked; stale edges/entities are pruned at the end.
    registry.begin_run()
    graph_builder.begin_run(repo)

    chunkers = {
        "commit": CommitChunker(),
        "diff": DiffChunker(),
        "pr": PRChunker(),
        "issue": IssueChunker(),
        "release": ChangelogChunker(),
        "cicd": LogChunker("cicd"),
        "pr_graphql": LogChunker("pr_graphql", node_type="pr"),
    }

    all_chunks: list[Chunk] = []
    node_times: dict[str, str] = {}

    # ------------------------------------------------------------------ GraphQL PR records
    # Canonical PR documents may not yet carry GraphQL closing references; keep the
    # GraphQL records so they can be merged in (and so PRs absent from the REST
    # collection are still represented, as node ``pr_<n>``).
    gql_by_number = {}
    for payload in load_documents(store, "pr_graphql", repo):
        pg = parser.parse_pr_graphql(payload)
        gql_by_number[pg.number] = pg

    # ------------------------------------------------------------------ commits
    print("\n[1/5] Processing commits...")
    commit_payloads = load_documents(store, "commit", repo)
    print(f"  Found {len(commit_payloads)} commit documents")
    parsed_commits = []
    for payload in commit_payloads:
        pc = parser.parse_commit(payload)
        parsed_commits.append(pc)
        node_times[f"commit_{pc.sha}"] = pc.timestamp.isoformat()

        # History ↔ code: the commit touches each file it changed (file_<path> node).
        for path in pc.file_paths:
            graph_builder.add_edge(
                "commit", pc.sha, "file", path, "touches", repo=repo,
                event_time=pc.timestamp.isoformat(),
                source_doc=f"commit_{pc.sha}",
            )

        chunks = chunkers["commit"].chunk(pc, repo=repo)
        chunks += chunkers["diff"].chunk(pc, repo=repo)
        _tag_document(
            chunks, extractor, registry, pc.message, pc.sha, "commit",
            author=pc.author, file_paths=pc.file_paths,
        )
        all_chunks.extend(chunks)

    print(f"  Generated {sum(1 for c in all_chunks if c.metadata.doc_type in ('commit', 'diff'))} commit+diff chunks")

    # ------------------------------------------------------------------ PRs
    print("\n[2/5] Processing pull requests...")
    pr_payloads = load_documents(store, "pr", repo)
    print(f"  Found {len(pr_payloads)} PR documents")
    parsed_prs = []
    for payload in pr_payloads:
        ppr = parser.parse_pr({**payload, "repo": repo})
        gql = gql_by_number.get(ppr.number)
        if gql is not None and not ppr.closing_issue_numbers:
            ppr.closing_issue_numbers = list(gql.closing_issue_numbers)
            ppr.closing_issues = list(gql.closing_issues)
            ppr.closing_issues_complete = gql.closing_issues_complete
            ppr.closing_source = "graphql"
        parsed_prs.append(ppr)
        if ppr.created_at:
            node_times[f"pr_{ppr.number}"] = ppr.created_at.isoformat()

        chunks = chunkers["pr"].chunk(ppr, repo=repo)
        _tag_document(
            chunks, extractor, registry, ppr.title + "\n" + ppr.body_clean,
            str(ppr.number), "pr", author=ppr.author,
        )
        all_chunks.extend(chunks)

    print(f"  PR chunks so far: {sum(1 for c in all_chunks if c.metadata.doc_type == 'pr')}")

    # ------------------------------------------------------------------ Issues
    print("\n[3/5] Processing issues...")
    issue_payloads = load_documents(store, "issue", repo)
    print(f"  Found {len(issue_payloads)} issue documents")
    parsed_issues = []
    for payload in issue_payloads:
        pi = parser.parse_issue(payload)
        parsed_issues.append(pi)
        if pi.created_at:
            node_times[f"issue_{pi.number}"] = pi.created_at.isoformat()

        text = pi.title + "\n" + pi.body_clean + "\n" + " ".join(pi.comments)
        chunks = chunkers["issue"].chunk(pi, repo=repo)
        _tag_document(
            chunks, extractor, registry, text, str(pi.number), "issue", author=pi.author,
        )
        all_chunks.extend(chunks)

    print(f"  Issue chunks so far: {sum(1 for c in all_chunks if c.metadata.doc_type == 'issue')}")

    # ------------------------------------------------------------------ Releases
    print("\n[4/5] Processing releases...")
    release_payloads = load_documents(store, "release", repo)
    print(f"  Found {len(release_payloads)} release documents")
    for payload in release_payloads:
        pr_rel = parser.parse_release(payload)
        chunks = chunkers["release"].chunk(pr_rel, repo=repo)
        _tag_document(
            chunks, extractor, registry, pr_rel.body_clean, pr_rel.tag, "release",
        )
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ CI/CD Runs
    print("\n[4b/5] Processing CI/CD runs...")
    cicd_payloads = load_documents(store, "cicd", repo)
    print(f"  Found {len(cicd_payloads)} CI/CD documents")
    for payload in cicd_payloads:
        pc = parser.parse_cicd(payload)
        chunks = chunkers["cicd"].chunk(pc, repo=repo)
        _tag_document(
            chunks, extractor, registry, pc.body_clean, str(pc.run_id), "cicd",
        )
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ GraphQL-only PRs
    known_prs = {p.number for p in parsed_prs}
    orphan_gql = [g for n, g in gql_by_number.items() if n not in known_prs]
    print(f"\n[4c/5] GraphQL PR records without a REST PR: {len(orphan_gql)}")
    for pg in orphan_gql:
        if pg.created_at:
            node_times[f"pr_{pg.number}"] = pg.created_at.isoformat()
        text = pg.title + "\n" + pg.body_clean + "\n" + " ".join(pg.closing_issues)
        chunks = chunkers["pr_graphql"].chunk(pg, repo=repo)
        _tag_document(
            chunks, extractor, registry, text, str(pg.number), "pr_graphql", author=pg.author,
        )
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ Temporal graph
    print("\n[5/5] Building temporal graph...")
    # Phase-1 links. PR→issue 'mentions' are re-derived below with the shared
    # repo-aware reference parser (code blocks / foreign refs excluded).
    links = [
        SimpleNamespace(**p)
        for p in load_documents(store, "link", repo)
        if not (p.get("source_type") == "pr" and p.get("relation") == "mentions")
    ]
    graph_builder.add_edges_from_links(links, repo=repo, node_times=node_times)

    # PR→issue edges: local mentions, foreign references, GraphQL closes
    graph_builder.build_from_parsed_issues(
        parsed_prs, parsed_issues, repo=repo, node_times=node_times
    )
    graph_builder.build_from_parsed_issues(
        [
            SimpleNamespace(
                number=g.number,
                linked_issue_numbers=[],
                external_issue_refs=[],
                closing_issue_numbers=g.closing_issue_numbers,
                closing_source="graphql",
            )
            for g in orphan_gql
        ],
        parsed_issues,
        repo=repo,
        node_times=node_times,
    )

    print(f"  Graph edges: {graph_builder.edge_count(repo)}")

    # ------------------------------------------------------------------ Write generation
    mgr = GenerationManager(root=data_dir / "generations", repo=repo)
    created = generation_id is None
    gen_ctx = mgr.build(commit=pinned_sha(store, repo)) if created else nullcontext(
        _open_building(mgr, generation_id)
    )
    with gen_ctx as gen:
        # History is always rebuilt. Snapshot records are only replaced when this
        # run rebuilds them (repo_path); otherwise an existing generation keeps its
        # own, and a new one carries the active generation's code view forward.
        carried = None
        active = mgr.active()
        if created and repo_path is None and active is not None:
            carried = [r for r in read_jsonl(active.chunks_path) if _is_snapshot(r)]
        history_records = [
            {"text": c.text, "metadata": c.metadata.to_dict()} for c in all_chunks
        ]
        merge_into_generation(gen, history_records, carried)

        snapshot_stats = None
        if repo_path is not None:
            report = run_snapshot(
                repo_path, repo, ref=ref, data_dir=data_dir,
                generation_id=gen.generation_id,
            )
            gen.commit = report["commit"]
            gen.save_manifest()
            print(
                f"Snapshot {report['commit'][:10]}: {report['selected_files']} files, "
                f"{report['chunks']['snapshot_total']} code chunks"
                + ("" if report["complete"] else " (INCOMPLETE: " + "; ".join(report["stops"]) + ")")
            )

        snapshot_records = [r for r in read_jsonl(gen.chunks_path) if _is_snapshot(r)]
        snapshot_stats = _register_snapshot(snapshot_records, registry, graph_builder, repo)

        # Remove what this run no longer emits (per repository only).
        graph_pruned = graph_builder.prune_stale()
        registry_pruned = registry.prune_stale()

        total = len(all_chunks) + len(snapshot_records)
        gen.chunk_count = total
        gen.extra["phase2"] = {
            "history_chunks": len(all_chunks),
            "snapshot_chunks": len(snapshot_records),
            "snapshot_entities": snapshot_stats,
            "graph_edges": graph_builder.edge_count(repo),
            "entities": len(registry.all_entities()),
            "extractor": extractor.describe(),
            "pruned": {"graph": graph_pruned, "registry": registry_pruned},
        }
        gen.save_manifest()  # stays "building": phase 3 marks ready and publishes

    print(f"\n✅ Phase 2 complete!")
    print(f"   History chunks: {len(all_chunks)}, code (snapshot) chunks: {len(snapshot_records)}")
    print(f"   Stale removed: graph {graph_pruned}, registry {registry_pruned}")
    print(f"   Chunks written to: {gen.chunks_path}")
    print(f"   Entity registry entries: {len(registry.all_entities())}")
    print(f"   Graph edges: {graph_builder.edge_count(repo)}")
    print(f"GENERATION_ID={gen.generation_id}")
    print(f"GENERATION_DIR={gen.path}")

    registry.close()
    graph_builder.close()
    store.close()
    return gen.generation_id


def _is_snapshot(record: dict) -> bool:
    return record.get("metadata", {}).get("scope") == "snapshot"


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Parse and chunk one ingested repository")
    args_parser.add_argument("--repo", help="Repository identity; use 'legacy' only for old unscoped data")
    args_parser.add_argument("--generation", default=None,
                             help="write into this existing building generation instead of a new one")
    args_parser.add_argument("--repo-path", default=None,
                             help="local clone; also rebuilds the current-code (snapshot) view at --ref")
    args_parser.add_argument("--ref", default="HEAD", help="commit/branch for the snapshot")
    args = args_parser.parse_args()
    if args.ref != "HEAD" and not args.repo_path:
        args_parser.error("--ref requires --repo-path")
    t0 = time.time()
    run_phase2(
        args.repo,
        generation_id=args.generation,
        repo_path=args.repo_path,
        ref=args.ref,
    )
    print(f"\n   Elapsed: {time.time() - t0:.1f}s")
