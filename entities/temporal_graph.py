"""temporal_graph.py — Build and walk the temporal causality graph stored in SQLite."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal


EdgeRelation = Literal[
    "merge_commit",   # commit → pr
    "mentions",       # pr → issue  (or commit → issue)
    "closes",         # pr → issue
    "parent",         # commit → commit (parent commit)
    "references",     # commit → issue (referenced in message)
    "follow_up",      # issue → commit (commit that fixed the issue)
]


@dataclass
class GraphEdge:
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str

    @property
    def node_id_source(self) -> str:
        return f"{self.source_type}_{self.source_id}"

    @property
    def node_id_target(self) -> str:
        return f"{self.target_type}_{self.target_id}"


class TemporalGraphBuilder:
    """Builds and persists the temporal causality graph in SQLite.

    The graph is an adjacency list: (source_node, target_node, relation).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_tables()

    def _setup_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id   TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                relation    TEXT NOT NULL,
                UNIQUE(source_type, source_id, target_type, target_id, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target_type, target_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO graph_edges
                (source_type, source_id, target_type, target_id, relation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_type, source_id, target_type, target_id, relation),
        )
        self._conn.commit()

    def add_edges_from_links(self, links: list) -> None:
        """Accept LinkRecord objects from CrossReferenceLinker."""
        for link in links:
            self.add_edge(
                source_type=link.source_type,
                source_id=link.source_id,
                target_type=link.target_type,
                target_id=link.target_id,
                relation=link.relation,
            )

    def add_commit_parent_edges(self, commits: list) -> None:
        """Add parent→child commit edges from a list of CommitRecord or ParsedCommit."""
        # We don't have parent SHAs directly from CommitRecord — the GitCollector
        # iterates in reverse chronological order.  This method is a hook for
        # callers that already have parent SHA available.
        for commit in commits:
            parent_sha = getattr(commit, "parent_sha", None)
            if parent_sha:
                self.add_edge("commit", parent_sha, "commit", commit.sha, "parent")

    def build_from_parsed_issues(self, parsed_prs: list, parsed_issues: list) -> None:
        """Build PR→issue edges from parsed PR linked_issue_numbers."""
        issue_numbers = {str(i.number) for i in parsed_issues}
        for pr in parsed_prs:
            for issue_num in getattr(pr, "linked_issue_numbers", []):
                if str(issue_num) in issue_numbers:
                    self.add_edge("pr", str(pr.number), "issue", str(issue_num), "mentions")

    def edge_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]

    # ------------------------------------------------------------------
    # Querying / walking
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_type: str,
        node_id: str,
        direction: str = "both",
        relation: str | None = None,
    ) -> list[dict]:
        """Return neighboring nodes for one-hop expansion.

        direction: "out" | "in" | "both"
        """
        results: list[dict] = []
        base_filter = "AND relation = ?" if relation else ""
        params_rel = (relation,) if relation else ()

        if direction in ("out", "both"):
            rows = self._conn.execute(
                f"SELECT target_type, target_id, relation FROM graph_edges "
                f"WHERE source_type=? AND source_id=? {base_filter}",
                (node_type, node_id) + params_rel,
            ).fetchall()
            results += [
                {"type": r[0], "id": r[1], "relation": r[2], "direction": "out"}
                for r in rows
            ]

        if direction in ("in", "both"):
            rows = self._conn.execute(
                f"SELECT source_type, source_id, relation FROM graph_edges "
                f"WHERE target_type=? AND target_id=? {base_filter}",
                (node_type, node_id) + params_rel,
            ).fetchall()
            results += [
                {"type": r[0], "id": r[1], "relation": r[2], "direction": "in"}
                for r in rows
            ]

        return results

    def close(self) -> None:
        self._conn.close()


class TemporalGraphWalker:
    """Walk the temporal graph to expand retrieved nodes.

    Default: 1-hop expansion (configurable to 2-hop).
    """

    def __init__(self, db_path: str, hop_depth: int = 1) -> None:
        self._builder = TemporalGraphBuilder(db_path)
        self.hop_depth = hop_depth

    def expand(
        self,
        node_type: str,
        node_id: str,
    ) -> list[dict]:
        """Return all nodes reachable within hop_depth hops from (node_type, node_id)."""
        visited: set[tuple[str, str]] = set()
        frontier = [(node_type, node_id)]
        all_neighbors: list[dict] = []

        for _ in range(self.hop_depth):
            next_frontier: list[tuple[str, str]] = []
            for n_type, n_id in frontier:
                if (n_type, n_id) in visited:
                    continue
                visited.add((n_type, n_id))
                neighbors = self._builder.get_neighbors(n_type, n_id)
                for neighbor in neighbors:
                    key = (neighbor["type"], neighbor["id"])
                    if key not in visited:
                        all_neighbors.append(neighbor)
                        next_frontier.append(key)
            frontier = next_frontier

        # Deduplicate
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for n in all_neighbors:
            k = (n["type"], n["id"])
            if k not in seen:
                seen.add(k)
                deduped.append(n)

        return deduped

    def expand_many(self, nodes: list[tuple[str, str]]) -> list[dict]:
        """Expand multiple nodes and return a deduplicated flat list."""
        all_neighbors: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for node_type, node_id in nodes:
            for n in self.expand(node_type, node_id):
                k = (n["type"], n["id"])
                if k not in seen:
                    seen.add(k)
                    all_neighbors.append(n)
        return all_neighbors

    def close(self) -> None:
        self._builder.close()
