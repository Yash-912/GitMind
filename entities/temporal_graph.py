"""temporal_graph.py — Build and walk the temporal causality graph stored in SQLite."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal


EdgeRelation = Literal[
    "merge_commit",   # commit → pr
    "mentions",       # pr → issue  (or commit → issue)
    "closes",         # pr → issue
    "parent",         # commit → commit (parent commit)
    "references",     # commit → issue (referenced in message), pr → foreign issue
    "follow_up",      # issue → commit (commit that fixed the issue)
]


@dataclass
class Edge:
    """One repo-scoped graph edge as seen from a query node.

    ``src``/``dst`` are node ids (``"<type>_<id>"``) in stored orientation;
    ``direction`` is relative to the node being expanded ("out": that node is
    ``src``; "in": it is ``dst``). Foreign (other-repository) targets carry the
    namespace in the id, e.g. ``issue_other/repo#7``, and ``dst_repo`` set.
    """

    repo: str
    src: str
    dst: str
    relation: str
    event_time: str | None = None   # ISO time of the source document, if known
    source_doc: str = ""            # node id of the document that asserts the edge
    direction: str = "out"
    source: str = ""                # provenance of the assertion, e.g. "graphql"
    dst_repo: str = ""              # non-empty for edges into another repository


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


def _get_connection(db_path: str):
    """Return a sqlite3 connection, respecting DATABASE_URL env var.

    When DATABASE_URL is a PostgreSQL URL the temporal graph still uses
    an in-process SQLite file so that graph_edges traversal stays fast.
    """
    from config.settings import settings
    import os

    if settings.database_url and not settings.database_url.startswith("sqlite"):
        path = "data/graph.db"
    else:
        path = db_path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class TemporalGraphBuilder:
    """Builds and persists the temporal causality graph in SQLite.

    The graph is an adjacency list: (source_node, target_node, relation).
    Every edge and node belongs to one repository (``repo``; "" is the legacy
    namespace) so identically numbered PRs/issues of different repositories
    never collide.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = _get_connection(db_path)
        self._setup_tables()

    _EDGE_DDL = """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            repo        TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL,
            source_id   TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id   TEXT NOT NULL,
            relation    TEXT NOT NULL,
            src_node    TEXT NOT NULL,
            dst_node    TEXT NOT NULL,
            target_repo TEXT NOT NULL DEFAULT '',
            event_time  TEXT,
            source_doc  TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT '',
            UNIQUE(repo, source_type, source_id, target_type, target_id, relation)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(repo, src_node);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(repo, dst_node);
        CREATE TABLE IF NOT EXISTS graph_nodes (
            repo         TEXT NOT NULL DEFAULT '',
            node_id      TEXT NOT NULL,
            node_type    TEXT NOT NULL,
            node_ref     TEXT NOT NULL,
            foreign_repo TEXT NOT NULL DEFAULT '',
            event_time   TEXT,
            PRIMARY KEY (repo, node_id)
        );
    """

    def _setup_tables(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(graph_edges)")}
        if cols and "repo" not in cols:
            self._migrate_legacy_edges()
        self._conn.executescript(self._EDGE_DDL)
        self._conn.commit()

    def _migrate_legacy_edges(self) -> None:
        """Rebuild a pre-isolation graph_edges table; legacy rows get repo=''."""
        self._conn.executescript(
            """
            DROP INDEX IF EXISTS idx_graph_source;
            DROP INDEX IF EXISTS idx_graph_target;
            ALTER TABLE graph_edges RENAME TO graph_edges_legacy;
            """
        )
        self._conn.executescript(self._EDGE_DDL)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO graph_edges
                (repo, source_type, source_id, target_type, target_id, relation,
                 src_node, dst_node)
            SELECT '', source_type, source_id, target_type, target_id, relation,
                   source_type || '_' || source_id, target_type || '_' || target_id
            FROM graph_edges_legacy
            """
        )
        self._conn.execute("DROP TABLE graph_edges_legacy")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_type: str,
        node_ref: str,
        repo: str = "",
        event_time: str | None = None,
        foreign_repo: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO graph_nodes (repo, node_id, node_type, node_ref, foreign_repo, event_time)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo, node_id) DO UPDATE SET
                event_time = COALESCE(excluded.event_time, graph_nodes.event_time)
            """,
            (repo, f"{node_type}_{node_ref}", node_type, node_ref, foreign_repo, event_time),
        )

    def add_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        repo: str = "",
        event_time: str | None = None,
        source_doc: str = "",
        source: str = "",
        target_repo: str = "",
        commit: bool = True,
    ) -> None:
        """Insert an edge into ``repo``'s graph (idempotent).

        ``target_repo`` is set for edges into another repository; then
        ``target_id`` must already carry the namespace (``owner/repo#7``).
        Re-adding an edge refreshes its provenance when new values are given.
        """
        self._conn.execute(
            """
            INSERT INTO graph_edges
                (repo, source_type, source_id, target_type, target_id, relation,
                 src_node, dst_node, target_repo, event_time, source_doc, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo, source_type, source_id, target_type, target_id, relation)
            DO UPDATE SET
                event_time = COALESCE(excluded.event_time, graph_edges.event_time),
                source_doc = CASE WHEN excluded.source_doc != '' THEN excluded.source_doc
                                  ELSE graph_edges.source_doc END,
                source     = CASE WHEN excluded.source != '' THEN excluded.source
                                  ELSE graph_edges.source END
            """,
            (
                repo, source_type, source_id, target_type, target_id, relation,
                f"{source_type}_{source_id}", f"{target_type}_{target_id}",
                target_repo, event_time, source_doc, source,
            ),
        )
        self.add_node(source_type, source_id, repo)
        self.add_node(target_type, target_id, repo, foreign_repo=target_repo)
        if commit:
            self._conn.commit()

    def add_edges_from_links(
        self,
        links: list,
        repo: str = "",
        node_times: dict[str, str] | None = None,
    ) -> None:
        """Accept LinkRecord objects (or look-alikes) from CrossReferenceLinker.

        The asserting document is the edge source — except for ``parent`` edges,
        which are asserted by the child commit (the target). ``event_time`` is
        that document's timestamp from ``node_times`` (node id -> ISO time).
        """
        node_times = node_times or {}
        for link in links:
            src = f"{link.source_type}_{link.source_id}"
            dst = f"{link.target_type}_{link.target_id}"
            doc = dst if link.relation == "parent" else src
            self.add_edge(
                source_type=link.source_type,
                source_id=link.source_id,
                target_type=link.target_type,
                target_id=link.target_id,
                relation=link.relation,
                repo=repo,
                event_time=node_times.get(doc),
                source_doc=doc,
                source=getattr(link, "source", "") or "",
                commit=False,
            )
        self._conn.commit()

    def add_commit_parent_edges(self, commits: list, repo: str = "") -> None:
        """Add parent→child commit edges from a list of CommitRecord or ParsedCommit."""
        for commit in commits:
            parents = list(getattr(commit, "parent_shas", None) or [])
            single = getattr(commit, "parent_sha", None)
            if single:
                parents.append(single)
            when = getattr(commit, "timestamp", None)
            for parent_sha in parents:
                self.add_edge(
                    "commit", parent_sha, "commit", commit.sha, "parent",
                    repo=repo,
                    event_time=when.isoformat() if when else None,
                    source_doc=f"commit_{commit.sha}",
                )

    def build_from_parsed_issues(
        self,
        parsed_prs: list,
        parsed_issues: list,
        repo: str = "",
        node_times: dict[str, str] | None = None,
    ) -> None:
        """Build PR→issue edges from parsed PRs.

        * ``mentions``: local ``#N`` references to an issue present in ``repo``;
        * ``closes`` (source "graphql"/"text"): ``closing_issue_numbers``, which
          are authoritative so the issue need not be among ``parsed_issues``;
        * ``references``: foreign ``owner/repo#N`` refs, kept in their own
          namespace (never a local edge).
        """
        node_times = node_times or {}
        issue_numbers = {str(i.number) for i in parsed_issues}
        for pr in parsed_prs:
            pr_node = f"pr_{pr.number}"
            when = node_times.get(pr_node)
            for issue_num in getattr(pr, "linked_issue_numbers", []):
                if str(issue_num) in issue_numbers:
                    self.add_edge(
                        "pr", str(pr.number), "issue", str(issue_num), "mentions",
                        repo=repo, event_time=when, source_doc=pr_node, source="text",
                        commit=False,
                    )
            closing_source = getattr(pr, "closing_source", None) or "text"
            for issue_num in getattr(pr, "closing_issue_numbers", None) or []:
                self.add_edge(
                    "pr", str(pr.number), "issue", str(issue_num), "closes",
                    repo=repo, event_time=when, source_doc=pr_node,
                    source=closing_source, commit=False,
                )
            for ref in getattr(pr, "external_issue_refs", None) or []:
                foreign_repo = ref.rpartition("#")[0]
                self.add_edge(
                    "pr", str(pr.number), "issue", ref, "references",
                    repo=repo, event_time=when, source_doc=pr_node,
                    source="text", target_repo=foreign_repo, commit=False,
                )
        self._conn.commit()

    def edge_count(self, repo: str | None = None) -> int:
        if repo is None:
            return self._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE repo = ?", (repo,)
        ).fetchone()[0]

    def node_count(self, repo: str = "") -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM graph_nodes WHERE repo = ?", (repo,)
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Querying / walking  (every query is filtered by repo)
    # ------------------------------------------------------------------

    def _edges_of(self, node_id: str, repo: str, direction: str) -> list[Edge]:
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be 'out', 'in' or 'both', got {direction!r}")
        edges: list[Edge] = []
        cols = "repo, src_node, dst_node, relation, event_time, source_doc, source, target_repo"
        if direction in ("out", "both"):
            for r in self._conn.execute(
                f"SELECT {cols} FROM graph_edges WHERE repo=? AND src_node=? ORDER BY id",
                (repo, node_id),
            ):
                edges.append(Edge(r[0], r[1], r[2], r[3], r[4], r[5], "out", r[6], r[7]))
        if direction in ("in", "both"):
            for r in self._conn.execute(
                f"SELECT {cols} FROM graph_edges WHERE repo=? AND dst_node=? ORDER BY id",
                (repo, node_id),
            ):
                edges.append(Edge(r[0], r[1], r[2], r[3], r[4], r[5], "in", r[6], r[7]))
        return edges

    def neighbors(
        self,
        node_id: str,
        repo: str,
        hops: int = 1,
        direction: str = "both",
    ) -> list[Edge]:
        """Edges within ``hops`` of ``node_id`` ("<type>_<id>") in ``repo``'s graph."""
        seen_nodes = {node_id}
        seen_edges: set[tuple[str, str, str]] = set()
        frontier = [node_id]
        out: list[Edge] = []
        for _ in range(max(int(hops), 1)):
            nxt: list[str] = []
            for n in frontier:
                for e in self._edges_of(n, repo, direction):
                    key = (e.src, e.dst, e.relation)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    out.append(e)
                    other = e.dst if e.direction == "out" else e.src
                    if other not in seen_nodes:
                        seen_nodes.add(other)
                        nxt.append(other)
            frontier = nxt
        return out

    def get_neighbors(
        self,
        node_type: str,
        node_id: str,
        direction: str = "both",
        relation: str | None = None,
        repo: str = "",
    ) -> list[dict]:
        """Return neighboring nodes for one-hop expansion within ``repo``.

        direction: "out" | "in" | "both"
        """
        results: list[dict] = []
        for e in self._edges_of(f"{node_type}_{node_id}", repo, direction):
            if relation and e.relation != relation:
                continue
            other = e.dst if e.direction == "out" else e.src
            otype, _, oid = other.partition("_")
            results.append(
                {"type": otype, "id": oid, "relation": e.relation, "direction": e.direction}
            )
        return results

    def close(self) -> None:
        self._conn.close()


class TemporalGraphWalker:
    """Walk the temporal graph to expand retrieved nodes within one repository.

    Default: 1-hop expansion (configurable to 2-hop).
    """

    def __init__(self, db_path: str, hop_depth: int = 1) -> None:
        self._builder = TemporalGraphBuilder(db_path)
        self.hop_depth = hop_depth

    def neighbors(
        self, node_id: str, repo: str, hops: int | None = None, direction: str = "both"
    ) -> list[Edge]:
        return self._builder.neighbors(node_id, repo, hops or self.hop_depth, direction)

    def expand(
        self,
        node_type: str,
        node_id: str,
        repo: str = "",
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
                neighbors = self._builder.get_neighbors(n_type, n_id, repo=repo)
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

    def expand_many(self, nodes: list[tuple[str, str]], repo: str = "") -> list[dict]:
        """Expand multiple nodes and return a deduplicated flat list."""
        all_neighbors: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for node_type, node_id in nodes:
            for n in self.expand(node_type, node_id, repo=repo):
                k = (n["type"], n["id"])
                if k not in seen:
                    seen.add(k)
                    all_neighbors.append(n)
        return all_neighbors

    def close(self) -> None:
        self._builder.close()
