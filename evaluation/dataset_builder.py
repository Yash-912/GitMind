from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

NO_EVIDENCE_ANSWER = "I could not find evidence in the provided context."
_TOKEN = re.compile(r"[a-z0-9_]{4,}")
_STOP = frozenset(
    "what when where which who whom whose why how does did was were been being have has had "
    "this that these those with from into about after before during there their they them "
    "will would could should evidence context provided mentioned described".split()
)


@dataclass
class QAPair:
    question: str
    answer: str
    context: str
    source_chunk_id: str
    qa_type: str = "standard"
    required_chunk_ids: list[str] = field(default_factory=list)
    split: str = "eval"
    seed: int = 0
    dry_run: bool = False


class DatasetBuilder:
    """Build a QA dataset from existing chunks.

    Generation is injectable: pass ``generate`` (``str -> str``) or an ``llm``
    object with ``generate(prompt).text``; with neither, a real ``LLMClient`` is
    created lazily. Failed generations (bad JSON, blank/missing fields) are
    rejected and counted, never replaced by a truncated-chunk fallback.
    """

    def __init__(self, llm: Any = None, generate: Callable[[str], str] | None = None) -> None:
        self._generate = generate
        self._llm = llm
        self.rejected: Counter = Counter()

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from generation.llm_client import LLMClient

            self._llm = LLMClient()
        return self._llm

    def _call(self, prompt: str) -> str:
        if self._generate is not None:
            return self._generate(prompt)
        return self.llm.generate(prompt).text

    # ------------------------------------------------------------------ build

    def build(
        self,
        chunks_path: str,
        out_path: str,
        limit: int = 100,
        dry_run: bool = False,
        include_adversarial: bool = True,
        include_temporal: bool = True,
        include_multihop: bool = True,
        seed: int = 0,
        eval_fraction: float = 0.3,
        links: list[tuple[str, str]] | None = None,
        answerable_overlap: float = 0.6,
        max_attempts: int = 3,
    ) -> list[QAPair]:
        chunks = self._load_chunks(chunks_path)
        if not chunks:
            return []
        self.rejected = Counter()
        rng = random.Random(seed)
        usable = sorted(
            (c for c in chunks if c.get("text") and self._cid(c)), key=self._cid
        )
        sampled = rng.sample(usable, min(limit, len(usable)))
        by_id = {self._cid(c): c for c in usable}
        neighbors = self._neighbors(usable, links)
        corpus_index = self._corpus_index(usable)

        pairs: list[QAPair] = []

        def add(q_a: tuple[str, str] | None, text: str, cid: str, qa_type: str, required: list[str]) -> None:
            if q_a is None:
                return
            pairs.append(
                QAPair(q_a[0], q_a[1], text, cid, qa_type, required_chunk_ids=required, seed=seed, dry_run=dry_run)
            )

        for rec in sampled:
            cid, text = self._cid(rec), rec["text"]
            if dry_run:
                add(("What does this change describe?", text[:200]), text, cid, "standard", [cid])
            else:
                add(
                    self._qa(
                        "Create one concise question and answer based on the context below.\n\n"
                        f"Context:\n{text}\n\nReturn as JSON with fields: question, answer.",
                        "standard",
                    ),
                    text, cid, "standard", [cid],
                )

            if include_temporal:
                ts = str(rec.get("metadata", {}).get("timestamp", "") or "")
                if dry_run:
                    add(("When did this happen?", ts or "Unknown"), text, cid, "temporal", [cid])
                elif not ts:
                    self.rejected["temporal_no_timestamp"] += 1
                else:
                    add(
                        self._qa(
                            "Create a temporal question and answer based on the context below. "
                            "The answer must be grounded in the source timestamp.\n\n"
                            f"Source timestamp: {ts}\nContext:\n{text}\n\n"
                            "Return JSON with fields: question, answer.",
                            "temporal",
                        ),
                        f"[timestamp: {ts}]\n{text}", cid, "temporal", [cid],
                    )

            if include_multihop:
                cluster = self._cluster(cid, neighbors, rng)
                if len(cluster) < 2:
                    self.rejected["multihop_no_linked_cluster"] += 1
                else:
                    ctx = "\n\n".join(
                        f"[chunk {i}: {c}]\n{by_id[c]['text']}" for i, c in enumerate(cluster, 1)
                    )
                    if dry_run:
                        add(("What led to this change?", "Unknown"), ctx, cid, "multihop", cluster)
                    else:
                        add(
                            self._qa(
                                "Create a multi-hop question that can only be answered by combining "
                                "facts from at least two of the linked chunks below.\n\n"
                                f"{ctx}\n\nReturn JSON with fields: question, answer.",
                                "multihop",
                            ),
                            ctx, cid, "multihop", cluster,
                        )

        if include_adversarial:
            for rec in sampled[: max(1, len(sampled) // 10)]:
                pair = self._adversarial(rec, corpus_index, dry_run, answerable_overlap, max_attempts)
                if pair is not None:
                    add(pair, rec["text"], self._cid(rec), "adversarial", [])
                else:
                    self.rejected["adversarial_not_verified_unanswerable"] += 1

        self._assign_split(pairs, rng, eval_fraction)
        self._write_pairs(out_path, pairs)
        self._write_meta(out_path, chunks_path, usable, pairs, seed, limit, eval_fraction, dry_run, links is not None)
        return pairs

    # ------------------------------------------------------------- generation

    def _qa(self, prompt: str, kind: str) -> tuple[str, str] | None:
        """Parse a generated QA pair; None (and a counted rejection) on any failure."""
        try:
            raw = self._call(prompt)
        except Exception:  # noqa: BLE001
            self.rejected[f"{kind}_generator_error"] += 1
            return None
        data = self._parse_json(raw)
        q = str((data or {}).get("question", "")).strip()
        a = str((data or {}).get("answer", "")).strip()
        if data is None or not q or not a:
            self.rejected[f"{kind}_blank_or_unparseable"] += 1
            return None
        return q, a

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        raw = (raw or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------ adversarial

    def _adversarial(
        self, rec: dict, index: dict, dry_run: bool, overlap: float, attempts: int
    ) -> tuple[str, str] | None:
        text = rec["text"]
        if dry_run:
            return "Is there evidence of a database migration in this context?", NO_EVIDENCE_ANSWER
        for _ in range(attempts):
            qa = self._qa(
                "Create a plausible question about this project that is NOT answered by the "
                f"context below.\n\nContext:\n{text}\n\nReturn JSON with fields: question, answer.",
                "adversarial",
            )
            if qa is None:
                continue
            if self.answerable_in_corpus(qa[0], index, overlap):
                self.rejected["adversarial_answerable_elsewhere"] += 1
                continue
            return qa[0], NO_EVIDENCE_ANSWER
        return None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP}

    def _corpus_index(self, chunks: list[dict]) -> dict[str, set[int]]:
        index: dict[str, set[int]] = {}
        for i, c in enumerate(chunks):
            for t in self._tokens(c["text"]):
                index.setdefault(t, set()).add(i)
        return index

    def answerable_in_corpus(self, question: str, index: dict[str, set[int]], overlap: float = 0.6) -> bool:
        """Keyword check over the WHOLE corpus: does any chunk contain most question terms?"""
        terms = self._tokens(question)
        if not terms:
            return True  # cannot verify -> treat as answerable, i.e. reject
        hits: Counter = Counter()
        for t in terms:
            for i in index.get(t, ()):
                hits[i] += 1
        return bool(hits) and max(hits.values()) / len(terms) >= overlap

    # --------------------------------------------------------------- clusters

    @staticmethod
    def _cid(rec: dict) -> str:
        return rec.get("metadata", {}).get("chunk_id", "") or ""

    def _neighbors(self, chunks: list[dict], links: list[tuple[str, str]] | None) -> dict[str, list[str]]:
        """Link graph over chunk ids: explicit links, else shared doc_id / entity tag."""
        adj: dict[str, set[str]] = {}

        def connect(a: str, b: str) -> None:
            if a and b and a != b:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

        if links:
            for a, b in links:
                connect(a, b)
        else:
            groups: dict[tuple[str, str], list[str]] = {}
            for c in chunks:
                meta, cid = c.get("metadata", {}), self._cid(c)
                if meta.get("doc_id"):
                    groups.setdefault(("doc", str(meta["doc_id"])), []).append(cid)
                for e in meta.get("entity_tags") or []:
                    groups.setdefault(("entity", str(e).lower()), []).append(cid)
            for members in groups.values():
                if 1 < len(members) <= 20:  # skip hub entities that link everything
                    for m in members[1:]:
                        connect(members[0], m)
                        if len(members) > 2:
                            connect(members[1], m)
        return {k: sorted(v) for k, v in adj.items()}

    @staticmethod
    def _cluster(cid: str, neighbors: dict[str, list[str]], rng: random.Random, size: int = 3) -> list[str]:
        near = neighbors.get(cid, [])
        if not near:
            return [cid]
        picked = rng.sample(near, min(size - 1, len(near)))
        return [cid, *sorted(picked)]

    @staticmethod
    def links_from_graph(db_path: str, chunks: list[dict]) -> list[tuple[str, str]]:
        """Chunk-id links derived from graph_edges (node id = '<type>_<id>', unverified on real data)."""
        by_node: dict[str, list[str]] = {}
        for c in chunks:
            meta = c.get("metadata", {})
            if meta.get("graph_node_id") and meta.get("chunk_id"):
                by_node.setdefault(meta["graph_node_id"], []).append(meta["chunk_id"])
        out: list[tuple[str, str]] = []
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT source_type, source_id, target_type, target_id FROM graph_edges").fetchall()
        finally:
            conn.close()
        for st, sid, tt, tid in rows:
            for a in by_node.get(f"{st}_{sid}", []):
                for b in by_node.get(f"{tt}_{tid}", []):
                    out.append((a, b))
        return out

    # ----------------------------------------------------------- split / io

    @staticmethod
    def _assign_split(pairs: list[QAPair], rng: random.Random, eval_fraction: float) -> None:
        """Split by source chunk so no chunk's questions land in both train and eval."""
        sources = sorted({p.source_chunk_id for p in pairs})
        rng.shuffle(sources)
        n_eval = min(len(sources), max(1, round(len(sources) * eval_fraction))) if sources else 0
        eval_set = set(sources[:n_eval])
        for p in pairs:
            p.split = "eval" if p.source_chunk_id in eval_set else "train"

    def _load_chunks(self, chunks_path: str) -> list[dict]:
        path = Path(chunks_path)
        if not path.exists():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _write_pairs(self, out_path: str, pairs: list[QAPair]) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    def _write_meta(
        self, out_path: str, chunks_path: str, usable: list[dict], pairs: list[QAPair], seed: int,
        limit: int, eval_fraction: float, dry_run: bool, explicit_links: bool,
    ) -> None:
        ids = "\n".join(self._cid(c) for c in usable)
        meta = {
            "seed": seed,
            "limit": limit,
            "eval_fraction": eval_fraction,
            "dry_run": dry_run,
            "source_scope": {
                "chunks_path": str(chunks_path),
                "n_chunks": len(usable),
                "chunk_ids_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
            },
            "link_source": "explicit" if explicit_links else "shared_doc_or_entity",
            "counts": dict(Counter(f"{p.split}/{p.qa_type}" for p in pairs)),
            "rejected": dict(self.rejected),
        }
        Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def meta_path(out_path: str) -> str:
        return str(out_path) + ".meta.json"

    def close(self) -> None:
        if self._llm is not None and hasattr(self._llm, "close"):
            self._llm.close()
