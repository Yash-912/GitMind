from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from entities.temporal_graph import TemporalGraphWalker

from .chunk_store import ChunkStore
from .types import CandidateChunk


class GraphExpander:
    """Expand candidates using temporal graph neighbors."""

    def __init__(
        self, db_path: str, chunk_store: ChunkStore, hop_depth: int = 1, repo: str = ""
    ) -> None:
        self.walker = TemporalGraphWalker(db_path, hop_depth=hop_depth)
        self.chunk_store = chunk_store
        self.repo = repo  # empty => taken from each candidate's metadata

    def expand(self, candidates: list[CandidateChunk]) -> list[CandidateChunk]:
        # The graph is repo-namespaced: always expand inside the seed's own repo.
        by_repo: dict[str, list[tuple[str, str]]] = {}
        for c in candidates:
            node_id = c.metadata.get("graph_node_id")
            if node_id and "_" in node_id:
                node_type, node_id_val = node_id.split("_", 1)
                repo = self.repo or c.metadata.get("repo", "")
                by_repo.setdefault(repo, []).append((node_type, node_id_val))

        neighbors: list[dict] = []
        for repo, nodes in by_repo.items():
            neighbors.extend(self.walker.expand_many(nodes, repo=repo))
        expanded: list[CandidateChunk] = list(candidates)
        seen = {c.chunk_id for c in candidates}

        for n in neighbors:
            nid = f"{n['type']}_{n['id']}"
            for chunk_id in self.chunk_store.by_node(nid):
                if chunk_id in seen:
                    continue
                record = self.chunk_store.get(chunk_id)
                if not record:
                    continue
                if record.get("metadata", {}).get("repo", "") not in by_repo:
                    continue  # same node id in another repository
                expanded.append(
                    CandidateChunk(
                        chunk_id=chunk_id,
                        score=0.0,
                        source="graph",
                        text=record.get("text", ""),
                        metadata=record.get("metadata", {}),
                    )
                )
                seen.add(chunk_id)

        return expanded

    def close(self) -> None:
        self.walker.close()


# ---------------------------------------------------------------------- #
# Evidenced graph paths (F12)                                             #
# ---------------------------------------------------------------------- #

# Only an explicit closing link is treated as causal evidence. Mentions and
# references are weak links; `parent` is ancestry. None of those may be
# presented as "X caused Y".
#
# Relation names are the real D1 vocabulary: merge_commit (commit->pr),
# mentions, closes (pr->issue), parent (commit->commit), references,
# follow_up (issue->commit that fixed it). `follow_up` is explicit fix
# provenance, so it is treated as causal evidence; `merge_commit` and `parent`
# are ancestry.
CAUSAL_RELATIONS = frozenset({"closes", "follow_up"})
ANCESTRY_RELATIONS = frozenset({"parent", "merge_commit"})
NO_CAUSAL_WARNING = "no causal evidence found, only mentions/ancestry"


class PathHop(NamedTuple):
    """One stored edge. src -> dst is the relation's true direction."""

    src: str
    relation: str
    dst: str
    event_time: Any
    source_doc: str


Path = list  # list[PathHop]


@dataclass
class GraphExpansion:
    candidates: list[CandidateChunk]          # input candidates + new graph candidates
    paths: list[list[PathHop]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def is_causal(path: list[PathHop]) -> bool:
    return any(h.relation in CAUSAL_RELATIONS for h in path)


def explain_path(path: list[PathHop]) -> str:
    """Citation-ready text for a path; weak links are labelled as non-causal."""
    parts = []
    for h in path:
        if h.relation in CAUSAL_RELATIONS:
            text = f"{h.src} {h.relation} {h.dst}"
        elif h.relation in ANCESTRY_RELATIONS:
            text = f"{h.src} --{h.relation}--> {h.dst} (ancestry, not causal)"
        else:
            text = f"{h.src} --{h.relation}--> {h.dst} (link only, not causal)"
        when = f"{h.event_time}" if h.event_time else "undated"
        src_doc = f"source: {h.source_doc}" if h.source_doc else "source: unknown"
        parts.append(f"{text} [{when}; {src_doc}]")
    return " ; ".join(parts)


class GraphPathWalker:
    """Expand candidates through typed graph edges and keep the evidenced paths.

    ``graph`` must provide ``neighbors(node_id, repo, hops=1, direction='both')``
    returning edges with ``repo, src, dst, relation, event_time, source_doc``.
    Chunk graph node ids (``graph_node_id``) are assumed to equal graph node
    ids. Neighbours/paths outside the scope are dropped: every node on a path
    must have at least one in-scope chunk.
    """

    def __init__(
        self,
        graph: Any,
        chunk_store: ChunkStore,
        hops: int = 1,
        max_paths: int = 30,
        max_seeds: int = 12,
    ) -> None:
        self.graph = graph
        self.chunk_store = chunk_store
        self.hops = hops
        self.max_paths = max_paths
        self.max_seeds = max_seeds

    def expand(self, candidates: list[CandidateChunk], scope: Any, causal: bool = False) -> GraphExpansion:
        expanded = list(candidates)
        seen = {c.chunk_id for c in candidates}
        paths: list[list[PathHop]] = []
        seen_paths: set[tuple] = set()
        seeds: list[str] = []
        for c in candidates:
            node = c.metadata.get("graph_node_id")
            if node and node not in seeds:
                seeds.append(node)

        for seed in seeds[: self.max_seeds]:
            edges = [
                e for e in self.graph.neighbors(seed, scope.repo, hops=self.hops, direction="both")
                if getattr(e, "repo", scope.repo) == scope.repo
            ]
            foreign = {e.dst for e in edges if getattr(e, "dst_repo", "")}
            for path in self._chain(seed, edges):
                nodes = {seed}
                for h in path:
                    nodes.update((h.src, h.dst))
                nodes.discard(seed)
                node_chunks = {n: self._in_scope_chunks(n, scope) for n in nodes}
                # Foreign-repo targets (dst_repo set, id like 'issue_other/repo#7')
                # are references, not retrievable chunks: keep them as evidence.
                if any(not chunks for n, chunks in node_chunks.items() if n not in foreign):
                    continue  # an endpoint is out of scope / not retrievable
                key = tuple(path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                paths.append(path)
                for chunks in node_chunks.values():
                    for rec in chunks:
                        meta = rec.get("metadata", {})
                        cid = meta.get("chunk_id")
                        if cid and cid not in seen:
                            seen.add(cid)
                            expanded.append(
                                CandidateChunk(
                                    chunk_id=cid, score=0.0, source="graph",
                                    text=rec.get("text", ""), metadata=meta,
                                )
                            )

        paths.sort(key=lambda p: (str(p[0].event_time or ""), len(p)))
        paths = paths[: self.max_paths]
        warnings: list[str] = []
        if causal and not any(is_causal(p) for p in paths):
            warnings.append(NO_CAUSAL_WARNING + ("" if paths else " (no graph links for retrieved chunks)"))
        return GraphExpansion(candidates=expanded, paths=paths, warnings=warnings)

    def _in_scope_chunks(self, node_id: str, scope: Any) -> list[dict]:
        recs = [self.chunk_store.get(cid) for cid in self.chunk_store.by_node(node_id)]
        return [r for r in recs if r and scope.matches(r.get("metadata", {}))]

    def _chain(self, seed: str, edges: list[Any]) -> list[list[PathHop]]:
        """Chains of edges starting at `seed`, up to `hops` long, no revisits."""
        hops_all = [
            PathHop(e.src, e.relation, e.dst, getattr(e, "event_time", None), getattr(e, "source_doc", ""))
            for e in edges
        ]
        result: list[list[PathHop]] = []
        frontier: list[tuple[list[PathHop], str, frozenset]] = [([], seed, frozenset({seed}))]
        for _ in range(max(1, self.hops)):
            nxt: list[tuple[list[PathHop], str, frozenset]] = []
            for path, tip, visited in frontier:
                for h in hops_all:
                    if h in path:
                        continue
                    if h.src == tip:
                        far = h.dst
                    elif h.dst == tip:
                        far = h.src
                    else:
                        continue
                    if far in visited:
                        continue
                    new = path + [h]
                    result.append(new)
                    nxt.append((new, far, visited | {far}))
            frontier = nxt
        return result
