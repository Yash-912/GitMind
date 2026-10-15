# GitMind project audit

Audit date: 2026-10-06. Source of truth: the checked-out implementation. Product intent: `gitmind_prd.md`, with `INGESTION_DESIGN.md` and `IMPLEMENTATION_PLAN.md` used to distinguish deliberately planned features from implemented behavior.

## Assessment

GitMind has a substantial history-ingestion foundation and individually implemented RAG components. Its current executable workflow is a history-based prototype, not yet the current-code memory layer described in the updated PRD. The largest correctness problems are inconsistent index updates, ineffective retrieval scope, incompatible embedding fallback, incomplete evidence provenance, and silent failure handling. These need attention before answer quality or deployment readiness can be assessed credibly.

The existing plan correctly marks delivery phases 2–6 pending. The older script names `run_phase2.py` and `run_phase3.py` mean parsing/chunking and embedding/indexing; they do not mean the revised delivery phases have been completed.

## What the implementation actually does

1. `scripts/ingest.py` resolves a local repository or clones a GitHub repository. It defaults to full cloning. `--max-commits` limits collected commit records, not clone transfer or GitHub records.
2. `ingestion/pipeline.py` collects commits reachable from local HEAD, first-parent diffs, all parent identities, GitHub PRs/issues, releases, workflow metadata, and merged-PR closing-reference metadata. Raw documents and collection cursors use repository namespaces and transactional persistence. Run manifests describe raw-source coverage.
3. `scripts/run_phase2.py` selects one repository, parses its raw records, extracts prose entities, stores links in a SQLite graph, and overwrites a shared `data/chunks.jsonl`. The source-code AST parser exists but is not invoked by this workflow.
4. `scripts/run_phase3.py` embeds the entire chunk file, upserts Qdrant, rebuilds BM25, and adds FTS records. There is no coordinated publication of an index generation.
5. The API decomposes a question, resolves entities, searches dense/sparse/FTS indices, expands graph neighbors, reranks, builds context, and generates one of four answer modes. Streamlit calls this API; CLI scripts contain similar orchestration.
6. Evaluation builds synthetic questions and runs a separate retrieval/generation pipeline. Custom metrics are simple heuristics; MLflow is a standalone wrapper.

Useful existing work includes explicit raw-document repository identities, forward diffs, merge-parent preservation, atomic record/checkpoint saves, raw-source coverage manifests, repository selection during parsing, and acceptance tests using temporary Git repositories. Their presence does not establish isolation or correctness downstream.

## Verification performed

| Check | Result | Scope |
|---|---|---|
| `python -m pytest -q` | Failed during collection: four module collection errors | Missing `sqlmodel` and `qdrant_client` in the active Python 3.12.4 environment; no full-suite pass claimed |
| `python -m pytest tests/test_phase2.py tests/test_generation.py -q -p no:cacheprovider` | **43 passed**, one datetime deprecation warning | Parsing, chunking, entities and generation helpers; model generation is faked in tests |
| `python docs/audit_checks.py` | **11 defect checks confirmed** | Actual local implementation; in-memory SQLite/cache, simulated provider outages, fake retrieval backends for fusion; no live providers |
| Temporary dependency installation | Could not complete | Configured package source returned no installable `sqlmodel`; active environment was not repaired |
| Docker engine access | Unavailable: access denied to Docker Desktop engine pipe | No Docker build, container startup or deployment pass claimed |
| Code tracing | Completed across ingestion, parsing, entities, chunks, embeddings, indices, retrieval, generation, evaluation, API/UI/CLI and deployment files | Code-level findings and planned gaps below |

The earlier 106-test result in `PHASE1.md` is historical evidence from another environment, not a result reproduced by this audit. Live GitHub collection, Gemini/Ollama calls, Qdrant Cloud, PostgreSQL, restart durability, concurrency and end-to-end answer quality remain unverified here. No production corpus was rebuilt, migrated or re-ingested.

The reproduction script asserts current defects, not desired behavior. It bypasses eager package imports only to test independent leaf modules without unrelated missing dependencies. Its temporary Qdrant import stand-in is limited to the hybrid-fusion check; it does not test Qdrant itself. Turn these cases into normal regression tests with corrected expectations when implementing fixes.

## High-priority correctness findings

### F01 — Rebuilding chunks creates a new identity for unchanged content

**Confirmed by reproduction.** `chunking/chunk_metadata.py:13` defaults to random UUIDs. The chunkers do not supply deterministic IDs. Repeating chunking over identical inputs produces different IDs. Qdrant's UUID conversion is deterministic only relative to those already-random IDs.

`scripts/run_phase3.py:79` upserts vectors, `:87` replaces BM25, and `:93` adds FTS records. There is no obsolete-ID deletion. The reproduction shows the same logical commit accumulating two FTS entries. Repeated vector indexing can similarly retain old points. Stale entries consume retrieval positions before `ChunkStore` discards IDs absent from the latest chunk file, reducing useful results.

**Fix:** repository-scoped deterministic identities including processing versions; explicit obsolete-content removal; staged index generations; publish chunks, BM25, FTS and vectors as one consistent generation. Verify repeat, edit, rename, deletion and interrupted publication.

### F02 — Splitting a release section duplicates its chunk ID

**Confirmed by reproduction.** `chunking/changelog_chunker.py:117` deep-copies metadata for every piece of a long section, including its UUID. The reproduction generates multiple pieces with one shared ID. `ChunkStore` dictionary entries and FTS replacement overwrite earlier pieces; Qdrant upserts target the same point identity.

**Fix:** derive a unique identity for each section piece and preserve its source section/offset. Verify all pieces survive lookup and indexing.

### F03 — Repository isolation stops before graph and retrieval

**Confirmed by code tracing.** Raw documents have `(repo, doc_type, doc_id)` identities, but `entities/temporal_graph.py:73` stores edges without repository identity. Nodes such as `pr_42` and `issue_7` collide across repositories. The entity registry also stores document references as bare strings, without repository or document type.

`api/schemas.py:12` has no repository/snapshot selector. `indexing/qdrant_store.py:216` cannot build a repository filter. `scripts/run_phase2.py:228` overwrites one shared chunk file with the selected repository while graph, entity, vector and FTS data can retain other repositories. This combination can silently change the corpus being answered and create unrelated graph paths. Raw-store isolation does not guarantee an isolated answer.

**Fix:** namespace documents, graph nodes/edges, entities, chunks and index generations consistently; require a repository scope on queries. Verify two repositories with identical PR/issue numbers and overlapping entities.

### F04 — Filters apply only to dense retrieval

**Confirmed by reproduction with fake backends exercising the real fusion code.** `retrieval/hybrid_retriever.py:50` constructs a Qdrant filter, but `:70` calls BM25 and FTS without equivalent filters. It does not filter candidates after fusion. `retrieval/graph_walker.py:16` expands without time/module/author/repository rules.

An excluded sparse result still reaches final candidates despite conflicting document type, author, module and date. Historical questions can therefore include evidence outside the requested period.

**Fix:** one validated scope applied across every retrieval branch and graph expansion. Filter before each branch's limit/ranking where practical; retain a final exclusion check. Test excluded records independently in dense, sparse, FTS and graph paths.

### F05 — Entity-driven module filters do not match stored metadata

**Confirmed by code tracing and chunk reproduction.** `scripts/run_phase2.py` stores extracted entities in the registry but never stamps canonical entities/module tags onto generated chunks. `module_tags` remain empty throughout the current chunkers. Commit/diff entity tags are empty; PR/issue tags are labels or comment markers.

`api/main.py:151` treats all resolved query entities as module tags, including technologies and people. A question naming JWT can therefore eliminate dense matches against empty module tags, leaving the unfiltered sparse branches to supply answers.

**Fix:** model module, symbol, person and technology identities separately; populate canonical chunk metadata and use appropriate query filters. Do not interpret every entity as a module.

### F06 — Embedding failures create apparently successful, cached zero vectors

**Confirmed by reproduction.** `embedding/models.py:101` returns 768 zeros after all providers fail. `embedding/embedder.py:121` caches the returned vectors. These nonempty lists are accepted by the upsert path. `run_phase3.py` can report successful indexing despite unusable embeddings, and later retries can reuse cached failure output.

**Fix:** typed provider failures; reject zero/nonfinite/wrong-dimension vectors and incorrect batch cardinality; cache only validated successful results. Never publish an index as ready after failed embedding. Test partial and total batch failures and recovery.

### F07 — Automatic embedding fallback mixes incompatible vector spaces

**Confirmed by code tracing; semantic quality not benchmarked.** `embedding/models.py:84` and `:146` fall back from Ollama to Gemini, while `embedding_cache.py:34` keys by requested model and text. Actual provider/model identity is not recorded. Gemini results can be cached under the Nomic name, and queries choose whichever provider is available later. Equal vector dimensions do not make representations from different models compatible.

**Fix:** persist actual provider/model/version/dimension for each collection and cache entry. Queries must use the collection's embedding space. A provider change requires an explicit new generation or rebuild.

### F08 — Local Docker ingestion and serving select conflicting Qdrant modes

**Confirmed configuration/code mismatch; Docker execution unavailable.** Compose sets `QDRANT_URL=http://qdrant:6333` and an empty API key. `indexing/qdrant_store.py:63` uses remote mode only if both URL and key are truthy. It therefore chooses local file storage instead of the configured Qdrant service.

The API opens that local store at startup; its ingestion subprocess later attempts to open the same path. This creates a local-store ownership conflict. The external Qdrant service is unused by this configuration.

**Fix:** select remote mode whenever a URL exists, with authentication optional for the local service; avoid concurrent processes owning one local store. Verify API startup followed by API-triggered ingestion in the actual Compose stack.

### F09 — API ingestion leaves BM25 stale and can expose partial chunk files

**Confirmed by code tracing.** `api/main.py:32` loads BM25 once. After ingestion, `:143` reloads chunks but uses the startup BM25 object. New chunk IDs do not match the old BM25 corpus. `run_phase2.py:228` writes the live file directly; a query during writing can observe incomplete content or invalid JSON.

**Fix:** a generation manager that atomically swaps all reader resources after successful ingestion. Never overwrite files currently used by readers. Verify ingestion while serving queries and failure before publication.

### F10 — Reusing a cloned GitHub repository does not fetch new commits

**Confirmed by code tracing.** `ingestion/repository.py:65` validates an existing clone and returns it without fetching or advancing the selected ref. Repeating `--github-repo owner/repo` can refresh GitHub PR/issue records while retaining stale local HEAD history. `--refresh` rereads that same local history.

**Fix:** explicit ref resolution/fetch/update followed by commit pinning; distinguish local repository mode from managed remote-clone updates. Verify a new remote commit is ingested on the second run and a failed fetch does not misstate coverage.

## Answer quality and provenance findings

| ID | Finding and evidence | Required improvement |
|---|---|---|
| F11 | **Subqueries and intent are unused.** `QueryDecomposer` produces them; API/answer/retrieval entry points retrieve the original question once. | Execute and fuse scoped subqueries; route intent explicitly; test a question requiring separate evidence searches. |
| F12 | **Temporal graph is a relationship graph without temporal path evidence.** Edges have no event timestamps/provenance; walker traverses both directions and returns flat neighbors. `ContextAssembler` preserves rerank order rather than chronology. | Preserve evidenced paths, edge direction and event times; distinguish mentions, ancestry and causal claims. A sorted timeline alone does not prove causality. |
| F13 | **Discussion timestamps and authors are lost.** `github_collector.py:141` and `:173` keep only comment bodies. PR review comments use PR author/creation time; issue comment windows use issue opener/creation time. | Preserve comment IDs, authors, created/updated times, thread relationships and source URLs. Otherwise ownership and date filters can attribute later comments to the wrong person/time. |
| F14 | **GraphQL closing references are not usable evidence.** `LogChunker` ignores `closing_issues`; its nodes are `pr_graphql_<n>`, while raw links use `pr_<n>`. `GraphExpander` splits node type at the first underscore. | Merge GraphQL enrichment into canonical PR records/edges, preserving source completeness; avoid ambiguous node-ID parsing. Omission from chunk text is reproduced. |
| F15 | **Context can be empty despite usable later candidates.** `retrieval/context_assembler.py:42` breaks at the first oversized chunk. Separators are not counted in its token budget. | Skip or split oversized chunks, count complete rendered context, and preserve included IDs. Oversized-first behavior is reproduced. |
| F16 | **Returned evidence is not a record of what the model saw or cited.** `api/main.py:204` lists every reranked candidate even if token budgeting omitted it. `DirectQAGenerator` extracts source headers from context, not validated claims from the answer. | Return separate retrieved/included/cited sources; validate cited IDs against included sources; preserve exact immutable source locations. |
| F17 | **No citation validation or structured direct answer guarantee.** Prompts request evidence/confidence/caveats, but direct answers remain arbitrary text. API memo generation sets `strict_json=False`; ownership/risk outputs are also text only. | Define mode schemas, validate outputs, verify citations and retry/fail visibly on malformed output. |
| F18 | **Risk/ownership reports lack independent analysis.** The generators just render prompts with top-ranked chunks. No complete scoped churn/ownership/co-change computation runs. A retrieved sample cannot establish repository-wide statistics. | Compute statistics from declared complete scopes; distinguish counts from narrative explanations; report denominator, date range and missing coverage. |
| F19 | **Total generation outage returns an empty answer as success.** `generation/llm_client.py:71` returns an empty `LLMResponse`; API does not distinguish outage from an answer. No deterministic no-evidence response exists before generation. | Separate unavailable provider, insufficient evidence and successful answer; map failures to API status/error codes and UI states. Outage return is reproduced. |
| F20 | **Reranker construction is repeated per API request.** `api/main.py:170` creates a new cross-encoder/cache. Loading errors are swallowed; fallback returns the first candidates, so appended zero-score graph evidence can be excluded entirely. Reported scores remain retrieval scores after reranking. | Load one persistent model, expose availability, manage concurrency/cache invalidation, and label retrieval versus rerank scores. Benchmark graph evidence with and without reranking. |
| F21 | **Commit parsing drops information.** `parsing/multi_schema_parser.py:113` always starts body at line three, losing line two if there is no blank separator. | Preserve all message lines; reproduced with a three-line message. |
| F22 | **Chunking silently truncates important source text.** Commit messages, issue bodies/windows and unsplit release notes slice text; individual PR review comments can exceed configured chunk size. Split diff chunks have no precise per-piece source lines in metadata. | Split without losing text, use measured token limits, retain exact source ranges and report any intentional exclusion. |
| F23 | **PR issue-reference parsing can create foreign links.** Parser regex `#(\d+)` also matches `other/repo#7`; graph building converts it into a local issue edge when local issue 7 exists. Raw cross-referencing has a stricter pattern, but phase-two graph construction reintroduces the error. | Use one repository-aware reference parser, preserve destination namespace, and test foreign refs, URLs and code examples. |

## Operational and evaluation findings

| ID | Finding and evidence | Required improvement |
|---|---|---|
| F24 | **Background jobs are not durable or coordinated.** `_tasks` is process memory; API jobs have no queue bounds, cancellation, progress, time budget or writer coordination. Multiple jobs share chunks/index files. Streamlit displays the task ID but never polls status. | Persist jobs, serialize publication per scope, bound work, expose progress/failure/cancel/resume, and poll in the UI. Do not treat these FastAPI tasks as the planned full-history job system. |
| F25 | **Health is not readiness.** `/health` calls Qdrant count and can return `ok` for an empty corpus or unusable BM25. No active commit/generation, embedding identity, coverage or generator availability is checked. | Keep liveness separate from query readiness; return active scope/generation and explicit degraded components. |
| F26 | **Cloud persistence is only partial.** DocumentStore uses PostgreSQL when configured, but graph/FTS remain SQLite and chunks/BM25/cache remain local. EntityRegistry redirects to hard-coded `data/entities.db` for non-SQLite URLs. README's graph-in-PostgreSQL and BM25-rebuilt-on-startup descriptions do not match this implementation. | Choose durable storage for every necessary artifact and verify restart reconstruction; honor configured paths. Existing external document/vector stores alone cannot reconstruct all current query state. |
| F27 | **RAGAS judge and correctness metric are missing.** `ragas_evaluator.py:69` calls evaluate without an explicit judge/embedding configuration. Configured Gemini generation settings are not passed to it. Answer correctness is omitted from the selected metrics. Errors during imports become a misleading 'ragas is not installed' message. | Pin dependencies, configure the judge explicitly, include required metrics, surface actual failures, and run reviewed known-answer cases. Actual RAGAS execution is unverified. |
| F28 | **Custom metrics reward incorrect answers.** Temporal score requires any matching year, even if other years are unsupported; entity score requires any overlap, even with invented entities. Multi-hop score measures retrieved type diversity, not linked/cited reasoning. | Penalize unsupported claims, associate dates with events, evaluate paths/citations, and define applicability per question. Perfect scores for answers containing unsupported years/entities are reproduced. |
| F29 | **Synthetic evaluation data is not a reliable held-out benchmark.** Multi-hop questions use one chunk, not a linked cluster. Temporal prompts omit source timestamps. Adversarial questions are unanswerable only within one sampled chunk and may be answerable elsewhere in the corpus. Sampling has no seed and fallback answers can be blank or truncated text. | Build reviewed linked clusters and corpus-wide unsupported cases; persist source scopes/seeds, split evaluation sets and reject failed generation. |
| F30 | **MLflow exists but is not integrated.** Evaluation/CLI do not call `MLflowLogger`; logger silently disables itself on errors and logs no datasets/artifacts. | Log configuration, provider identities, datasets, per-question results, metrics and resource costs; reproduce five comparisons and PRD targets. |
| F31 | **Request/resource failure handling is incomplete.** Heavy calls are synchronous, no total query deadline exists, Gemini client does not receive the configured HTTP timeout, and resources are closed only along success paths. Each request reloads the entire chunk file. Phase two/three hold multiple complete corpus copies in memory. | Add total deadlines, `try/finally` cleanup and bounded batches; persist shared read resources; measure peak memory, transfer and latency. |
| F32 | **API input validation is late/inconsistent.** Answer mode is unrestricted text and invalid values are rejected after retrieval. Whitespace-only questions pass length checks. Negative `max_commits` is accepted by the API and rejected only by its subprocess. | Use enums and normalized query validation; reject invalid budgets before starting work. |
| F33 | **Public UI has no per-user repository/access scope.** Backend `API_KEY` is shared by Streamlit, which forwards it for every visitor. It does not distinguish users or authorize repository access. Combined with missing query scope, this cannot support isolated private-repository use. | Define whether the deployment is a public single-corpus demo or a private multi-user service; implement authorization and repository access filters for the latter. Shared API authentication is not per-user access control. |
| F34 | **Deployment is not reproducible or supervised yet.** Most dependency versions and Qdrant image are unpinned. `start.sh` starts API in background, waits three seconds and execs Streamlit without monitoring API failure. Answer/retrieval scripts do not bootstrap project imports consistently with ingestion scripts. | Lock dependencies/images, add readiness/process supervision, standardize package/module entry points, and verify fresh installs plus API/UI/Docker starts. |

## Missing target capabilities

These are planned product gaps, not claims that existing implementations failed a runtime test:

- Selective current-code inventory and content retrieval at a pinned immutable commit.
- Include/exclude rules enforced before content transfer; detection of incomplete inventories.
- Actual snapshot/code chunks with symbols and exact commit/path/line citations. Only Python AST grammar is bundled; additional languages have partial regex support or no effective parser.
- Incremental add/change/rename/delete handling, processing-version invalidation and safe index-generation replacement.
- On-demand bounded history acquisition and separation of current and historical versions.
- Resource budgets for transfer, file size, document count, duration and workers.
- Query-visible source coverage, selected paths, active commit and readiness.
- Complete discussion/review provenance and relevant CI log content; existing workflow ingestion is metadata, not logs.
- Independently verified structured reports and citations, reviewed held-out evaluation and published PRD target scores.
- Durable, cancellable full-history background collection and measured second-repository performance.

`CODE_MODEL` is currently also `nomic-embed-text` (`embedding/models.py:21`), so the current dual vectors do not provide a distinct code embedding model. Query embedding is likewise the prose model. Ingestion constructs the embedding client at its localhost default instead of honoring `settings.ollama_base_url`, while queries honor the setting. Fix configuration and measure whether a distinct code embedding path adds useful retrieval quality before claiming that feature.

## Recommended implementation order

1. **Establish a repeatable verification environment.** Install a locked dependency set, rerun the full suite, keep provider tests offline by default, and add actual startup/index-readiness checks. Existing passing helper tests should remain passing.
2. **Repair data identity and index publication.** Address F01–F03, F06–F10 and release-piece loss first. A repeated ingestion must preserve useful data, remove obsolete active content and leave readers on a consistent generation after failure.
3. **Implement the PRD's current-code foundation.** Inventory/scope selection, pinned content, parser/source provenance, deterministic code chunks, incremental changes and measured transfer/memory limits. This supplies the missing substrate for current-code answers.
4. **Make retrieval scope and evidence correct.** Address filters/entities, comment provenance, reference parsing, executed subqueries and evidenced graph paths. Include current/history separation and two-repository exclusion tests.
5. **Make answer outcomes verifiable.** Fix context budgeting, citation accounting, structured schemas, provider-outage/no-evidence outcomes and independently computed statistics. Review known-answer and unsupported questions.
6. **Prove quality and operation.** Repair evaluation, integrate MLflow, compare baselines, then verify persistence, concurrency, UI job tracking and deployment on a second repository.

Adding prompt detail or changing the generation model will not resolve lost source text, incompatible embeddings, stale indices or incorrect scope. Those are the first changes that will make later quality measurements meaningful.
