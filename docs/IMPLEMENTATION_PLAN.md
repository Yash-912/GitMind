# GitMind implementation and acceptance plan

Every completed claim must have executable evidence. A passing unit test does not establish deployment readiness or answer quality.

The ingestion direction is selective current-code collection followed by incremental indexing. History expands when required by the question. Full-history collection is an optional background mode. See [INGESTION_DESIGN.md](INGESTION_DESIGN.md) for scope, budgets and acceptance tests.

This is the target architecture. Phase 1 code still defaults to full cloning and broad requested-source collection. Existing checkpoints provide collection resume, not incremental code indexing.

| Phase | Scope | Acceptance evidence | State |
|---|---|---|---|
| 1 | Ingestion foundation: correct git changes, parent records, repository identities, checkpoints, explicit collection coverage | Real git/SQLite/CLI acceptance tests and the regression suite | Implemented; verification recorded in PHASE1.md |
| 2 | Selective current-code ingestion, entities and deterministic chunks; changed-file updates, safe index replacement and model identity | Scope/transfer limits, exact source citations, zero embedding calls for unchanged snapshots, change/rename/delete and index-switch tests | Pending; next phase |
| 3 | Filtered hybrid retrieval, question routing, bounded history expansion, subqueries, persistent reranking and temporal graph paths | Current-code/history separation, expansion budgets, filter exclusion, chronological path and multi-hop tests; baseline comparison | Pending |
| 4 | Structured answer modes, verified citations, computed report statistics and reliable fallback | Known-answer/unsupported cases, provider failures and citation validation | Pending |
| 5 | Explicit RAGAS judge, valid custom metrics, reviewed held-out data and integrated MLflow | PRD metric targets, five configurations, reproducible experiment report | Pending |
| 6 | Verified API/UI/Docker deployment, persistence, coverage reporting and background full-history collection | Fresh installation, restart, concurrency, measured resource budgets and a second-repository demonstration | Pending |

## Claim acceptance map

| Requirement | Implementation phases | Required proof |
|---|---|---|
| Git history and diff correctness | 1 | Root additions, forward changes, renames/deletions, binary files, merge parents |
| Selective current-code ingestion | 2 | Pin one commit, exclude unwanted files before content transfer, preserve paths/lines and detect incomplete inventories |
| Incremental code indexing | 2 | Reuse unchanged content; handle additions, changes, renames, deletions and version changes without stale indices |
| Bounded history expansion | 3 | Collect relevant paths and linked documents only as needed; report history gaps and budget stops |
| Optional full-history mode | 6 | Explicit scope, background execution, resumable budgets and measured resource use |
| Repository isolation and resume | 1–3 | Duplicate PR numbers remain separate; record/cursor transactions roll back together; graph and indices enforce repository boundaries |
| Code, PR, issue, release and CI content | 1–3 | Source manifests distinguish snapshots from history, metadata from content, limits, unavailable data and failures; exhaustive collection is not required by default |
| Entity aliases and module tags | 2 | Canonical entities and module filters match real chunk metadata |
| Dense/code embeddings and caching | 2 | Provider/model identity, valid vectors, matching query models and measured code-search benefit |
| Hybrid retrieval and reranking | 3 | Consistent filters, executed subqueries, loaded reranker, quality comparison |
| Temporal and multi-hop retrieval | 3 | Timestamped, evidenced relationship paths; explicit time rules and unsupported-causality behavior |
| Direct answers, memos, ownership and risk | 4 | Valid schemas, checked citations, independent statistical calculations, missing-evidence responses |
| Gemini/Ollama fallback | 4 | Primary outage, fallback outage, total outage and request duration limits |
| RAGAS and custom metrics | 5 | Correct judge configuration, reviewed test cases, incorrect answers penalized, published target scores |
| MLflow | 5 | Evaluation writes parameters, artifacts and metrics for at least five comparisons |
| API, Streamlit and Docker | 6 | Clean deployment, readiness, durable state, visible errors and concurrent requests |

The production claim remains pending until all applicable acceptance requirements pass. Any requirement that cannot be met must be removed or limited in the PRD and README.
