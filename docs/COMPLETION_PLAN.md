# GitMind completion plan

Companion to [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (phases and acceptance map) and [PROJECT_AUDIT.md](PROJECT_AUDIT.md) (findings F01–F34). This document turns them into ordered, checkable work. A task is done only when its listed test passes; each audit reproduction in `docs/audit_checks.py` becomes a regression test with corrected expectations.

Ordering rule: each phase leaves the repo with a green test suite and a working end-to-end path. Do not start a phase until the previous exit gate passes.

## Phase 0 — Verification environment (prerequisite)

| # | Task | Done when |
|---|---|---|
| 0.1 | Create venv; install deps; lock with `pip-compile`/`uv lock` (pin all versions, pin Qdrant image tag) | Fresh install succeeds from lock file (F34) |
| 0.2 | Fix `sqlmodel`/`qdrant_client` install; run full `pytest` | 0 collection errors; record real pass count (replaces the stale 106) |
| 0.3 | Mark live-provider tests with a marker, skipped by default | Suite passes offline |
| 0.4 | Convert `docs/audit_checks.py` into `tests/test_audit_regressions.py` (xfail until fixed, one test per F-number) | 11 xfail tests present |
| 0.5 | Standardize entry points (`python -m scripts.x`, consistent `sys.path` bootstrapping) | All scripts run from repo root |

**Exit gate:** `pytest` green offline on a fresh clone.

## Phase 2 — Data identity, current-code ingestion, safe indexing

### 2A. Identity and index integrity (F01, F02, F06, F07, F08, F09, F10)

1. Deterministic chunk IDs: hash of `(repo, doc_type, doc_id, piece offset, chunker version, text hash)` in `chunk_metadata.py`; per-piece IDs in `changelog_chunker.py` (F01, F02).
2. Index generations: write chunks, BM25, FTS and Qdrant collection under a generation ID; atomically switch an `active` pointer; delete obsolete IDs/generations (F01, F09).
3. Typed embedding errors; validate dimension, finiteness, non-zero, batch cardinality; cache only valid results; never mark a generation ready after failures (F06).
4. Record actual provider/model/version/dim per collection and cache key; no silent cross-provider fallback at query time; changing provider requires a new generation (F07). Honor `settings.ollama_base_url` in ingestion.
5. Qdrant: use remote mode whenever a URL is set, API key optional; one owner per local store (F08).
6. Managed clones: `git fetch` + pin resolved commit; fail visibly without misreporting coverage (F10).
7. Fix parsing/chunking loss: commit body line 2 (F21), truncation with exact source ranges (F22), repo-aware reference regex (F23).

**Tests:** repeat/edit/rename/delete ingestion; interrupted publish leaves old generation live; embedding outage leaves index unpublished; Compose API startup then API-triggered ingestion; second run ingests a new remote commit.

### 2B. Repository isolation (F03)

Add `repo` (and snapshot commit) to graph nodes/edges, entity registry doc refs, chunk metadata and Qdrant payload; make repo a required query field in `QueryRequest`; per-repo chunk files/generations instead of one shared `data/chunks.jsonl`.
**Test:** two repos with identical PR/issue numbers and overlapping entities never cross-contaminate.

### 2C. Selective current-code ingestion (PRD gap)

1. Inventory at a pinned commit (git tree via API or sparse checkout); include/exclude rules applied before content transfer; incompleteness detection.
2. Budgets: max bytes, file size, file count, duration, workers; budget stops reported in the manifest.
3. Wire `parsing/code_parser.py` into the pipeline; add tree-sitter grammars (JS/TS, Go, Java at minimum) or declare unsupported languages explicitly.
4. Code/symbol chunks with exact `commit:path:line-range` citations.
5. Incremental updates: content-hash diff between snapshots; embed only added/changed; handle rename/delete; processing-version invalidation.
6. Populate canonical module/symbol/person/technology tags on chunks (F05).
7. Decide on a distinct code embedding model; keep it only if a measured benchmark shows a retrieval gain, otherwise drop the dual-vector claim.

**Exit gate:** second ingestion of an unchanged snapshot makes zero embedding calls; add/change/rename/delete tests pass; citations resolve to exact lines.

## Phase 3 — Retrieval correctness

1. One validated `Scope` object (repo, time, module, author, doc type, current-vs-history) applied to dense, BM25, FTS, graph expansion, plus a final exclusion check (F04).
2. Separate entity kinds in `EntityResolver`; only modules become module filters (F05).
3. Execute decomposed subqueries and route by intent; fuse with RRF (F11).
4. Preserve comment IDs, authors, timestamps, URLs in `github_collector.py` and chunkers (F13). Merge GraphQL closing refs into canonical PR records; fix `pr_graphql_<n>` node IDs (F14).
5. Temporal graph: edge timestamps, direction, provenance, relation type (mention/ancestry/closes); return paths; assemble context chronologically (F12).
6. Context assembler: skip/split oversized chunks, count separators, return included IDs (F15).
7. Persistent reranker loaded once at startup, availability exposed, labeled rerank vs retrieval scores, graph evidence not dropped (F20).
8. Bounded on-demand history expansion with budgets and gap reporting; current/history separation.

**Tests:** per-branch exclusion; multi-hop question needing linked evidence; chronological path; unsupported-causality returns a hedge; baseline comparison with/without each feature.

## Phase 4 — Verifiable answers

1. Pydantic schemas per mode (direct, memo, blame, risk); `strict_json` with validate-and-retry, visible failure on malformed output (F17).
2. Response separates `retrieved`, `included`, `cited` sources; validate cited IDs against included set (F16).
3. Distinguish outcomes: `answered`, `insufficient_evidence` (deterministic, pre-generation), `provider_unavailable` (HTTP 503) (F19). Gemini → Ollama fallback with total request deadline and the configured HTTP timeout (F31).
4. Compute ownership/churn/co-change/risk statistics from complete scoped data with denominator and date range; LLM only narrates (F18).
5. API validation: enum modes, trimmed non-empty queries, non-negative budgets rejected before work (F32).

**Tests:** known-answer set, unsupported questions, primary-only/fallback/total outage, forged citation rejected.

## Phase 5 — Evaluation

1. Explicit RAGAS judge and embeddings config; add `answer_correctness`; surface real import/runtime errors (F27).
2. Rewrite custom metrics: per-claim support, date-to-event association, path/citation-based multi-hop; define applicability (F28).
3. Reviewed held-out dataset: linked-cluster multi-hop, corpus-wide unanswerable cases, seeded sampling, train/eval split, reject failed generations (F29).
4. Integrate `MLflowLogger`: params, provider identities, datasets, per-question results, metrics, cost/latency; no silent disable (F30).
5. Run five configurations (e.g., dense-only, +BM25, +rerank, +graph, full) and publish results against PRD targets (faithfulness >0.85, relevancy >0.80, precision >0.75, recall >0.70, correctness >0.75). Any missed target is reported or the PRD is amended.

## Phase 6 — Deployment and operations

1. Durable jobs (persisted table), serialized publication per repo, queue bounds, progress, cancel/resume; Streamlit polls status (F24).
2. Separate `/health` (liveness) from `/ready` (active generation, embedding identity, coverage, generator availability) (F25).
3. Choose storage for all state: Postgres for graph/FTS/jobs, Qdrant for vectors, object or volume storage for chunks/BM25/cache; verify rebuild after restart; honor configured paths (F26).
4. Request lifecycle: shared read resources loaded once, `try/finally` cleanup, bounded batches, measured peak memory (F31).
5. Authorization: decide public single-corpus demo versus private multi-user; if multi-user, per-user repo access checks (F33).
6. Process supervision in `start.sh` (readiness wait, exit if API dies) and Compose healthchecks; pinned images (F34).
7. Optional background full-history mode with explicit scope and resumable budgets.
8. Second-repository demonstration with measured transfer, time, memory.
9. Update README/PRD to match what is verified; remove unproven claims.

**Exit gate:** clean install, Compose up, ingest, query, restart, concurrent queries all verified; PRD targets published.

## Dependencies and sequencing

```
0 → 2A → 2B → 2C → 3 → 4 → 5 → 6
```

- 2B (repo identity) must precede Phase 3, since scope needs it.
- 2C tagging feeds 3.2; 3.4 (comments) feeds 3.5 (graph).
- Phase 4 stabilizes the answer schema that Phase 5 evaluates.

## Risks

- Provider quotas (Gemini 429s) and no local Ollama can block embedding tests: keep fakes for unit tests and a small live smoke set.
- The `.agent/worktrees/future_scope` copy may hold partial implementations; review before reusing so work is not duplicated or diverges.
- Tree-sitter language coverage can balloon scope; start with Python plus two languages.
- Eval targets may not be reachable with free-tier models; plan to report measured numbers honestly.
