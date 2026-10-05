# Selective and incremental ingestion

Status: approved design direction. Selective ingestion and incremental code indexing are not implemented yet. Phase 1 currently defaults to full cloning and broad collection of the requested sources.

## Objective

Make current code searchable without first downloading and processing all repository history. Collect additional historical evidence when a question requires it. Keep every collection scope and limit visible.

Cloning and indexing are separate costs. A smaller clone does not guarantee less processing. Reading every historical diff can also download content deferred by a partial clone.

## Collection modes

| Mode | Intended use | Scope | Default |
|---|---|---|---|
| Current code | Explain current behavior and locate relevant code | Selected source files, tests, documentation and configuration at one commit | Yes, after implementation |
| Selected history | Explain a change or decision | Relevant paths, bounded commit history, and linked PRs/issues | Requested by a history question or explicit scope |
| Full history | Detailed study or archive | Explicit repository, refs, source types and resource budgets | No; background job only |

Full history means complete within the declared scope and accessible sources. It does not imply every branch, deleted GitHub record, inaccessible discussion or unavailable CI log.

## Initial current-code collection

1. Resolve the selected branch or tag to an immutable commit identifier.
2. Read the file inventory without downloading every file body.
3. Apply included paths, excluded paths and supported file types.
4. Select source files, relevant tests, documentation and configuration.
5. Exclude dependency folders, generated files, binaries and build outputs. Exclude secret files.
6. Retrieve only selected file content. Prefer partial clone with sparse checkout where supported. Reuse existing local repositories when appropriate.
7. Parse supported languages into functions, classes and modules. Preserve the source location.
8. Create deterministic chunks. Store the repository, commit identifier, file path, line range, symbol, content identifier and parser version.
9. Process chunks in bounded batches. Store valid embeddings and prepare replacement indices.
10. Publish an index generation only when its selected scope is ready. Save the coverage manifest with that generation.

A GitHub API file inventory is an alternative to cloning. Detect truncated tree responses and request subtrees separately. Respect API rate limits and record inaccessible files. Partial clone, sparse checkout and GitHub APIs are implementation choices, not unconditional scale guarantees.

Large or unsupported files must be reported as skipped. A selected scope is complete only when every included item is processed or its exclusion is recorded explicitly. Excluded content is outside the searchable coverage.

## Historical evidence expansion

1. Retrieve relevant code from the indexed snapshot.
2. Identify the paths and symbols required by the question.
3. Search their history within the configured date, depth and byte limits.
4. Retrieve relevant commit messages and diffs. Fetch linked PRs/issues and their required discussion content.
5. Store reusable evidence under its repository and historical version.
6. Keep historical evidence distinct from current code. Removed code must not be presented as the current implementation.
7. Return an answer only with sufficient support. Report incomplete history when the budget prevents further collection.

Questions about deleted symbols must also be supported through an explicit path, historical search or existing history metadata. They must not require the symbol to exist in the current snapshot.

Unlinked historical search can expand within a declared limit when initial evidence is insufficient. Do not download all PRs and issues before serving the first current-code question.

## Incremental updates

Record the last indexed commit and each selected file's content identifier. Compare the indexed snapshot with the next selected commit.

| Change | Required action |
|---|---|
| Added file | Retrieve, parse, chunk and embed its content |
| Changed file | Replace its obsolete chunks and process changed content |
| Renamed file | Update source locations; reuse content processing where valid |
| Deleted file | Remove it from current-code indices; retain explicitly stored historical evidence |
| Unchanged file | Reuse chunks and embeddings when processing versions match |
| Parser, chunker or embedding model changed | Reprocess affected records; never reuse incompatible cached results |

Use repository-scoped deterministic identities. Include the actual provider/model/version in embedding cache keys. Prepare a new index generation and switch readers together. Do not mix old BM25 data with new chunks or vectors.

A force push, non-ancestor update or branch change requires a snapshot comparison or explicit rebuild. Do not treat it as a normal forward commit range.

Phase 1 collection checkpoints support resume. They do not implement incremental code indexing or safe index-generation replacement. These are separate requirements.

## Budgets and coverage

Configure included paths, excluded paths, file types, maximum file size, transfer bytes, document count, history range/depth, batch size, worker count and job duration. These are design requirements; they are not existing CLI flags.

Use bounded queues and batches. Do not load the complete repository corpus into memory. A budget stop must preserve resume state and identify pending work.

Record at least:

- Repository and selected ref/commit.
- Collection mode, paths, source types and configured limits.
- Indexed, excluded, skipped, failed and pending items.
- Available history range and missing relationships.
- Index generation and parser/chunker/embedding versions.
- Downloaded bytes, processed documents, elapsed time and failure counts.

The API and UI must distinguish source coverage, index readiness and answer confidence. A complete code snapshot does not establish complete history. An answer must state when missing coverage affects its conclusions.

## Implementation order

| Step | Work | Evidence required |
|---|---|---|
| Phase 2A | File inventory, scope selection and current-code retrieval | Initial collection avoids unselected bodies and historical diffs; source locations match the pinned commit |
| Phase 2B | Deterministic parsing/chunks and incremental updates | Unchanged files avoid embedding calls; changes, renames and deletions produce a correct replacement index |
| Phase 3 | Question routing and bounded history expansion | Current-code questions use the snapshot; historical questions obtain relevant evidence without full-corpus collection |
| Phase 6 | Background full-history mode and operational budgets | Load, resume, memory, transfer and resource-limit tests within declared scopes |

## Acceptance tests

- Initial current-code collection does not scan all historical diffs or fetch all PR/issue bodies.
- A repository with large excluded assets does not transfer those file bodies through the selected-content path.
- Truncated API inventories are detected and completed through subtree requests, or marked partial.
- Included files have reproducible commit, path and line citations.
- Repeating an unchanged snapshot performs no new embedding calls.
- Additions, changes, renames and deletions update all current-code indices consistently.
- A processing-version change invalidates affected caches.
- Historical queries expand only within the declared scope and budget.
- Budget exhaustion, provider failure and interruption remain visible and resumable.
- Tests measure transfer bytes, peak memory, processing time and embedding calls. Publish the repository sizes and configured limits with the results.

No scalability target is considered achieved until these measurements exist. Full ingestion is never inferred from a successful clone or one completed batch.
