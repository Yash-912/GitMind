# Phase 1 — ingestion foundation

This phase corrects raw collection and storage. It does not establish answer quality or production deployment.

Design update: full cloning and broad collection are the current implementation, not the preferred default for large repositories. Phase 2 will introduce selective current-code ingestion and incremental indexing. Phase 3 will add bounded history expansion. See [INGESTION_DESIGN.md](INGESTION_DESIGN.md).

## Changes

- Collect diffs from the parent tree to the commit tree. Retain unified-diff file headers for downstream parsing.
- Preserve every merge parent, the commit timestamp, and the first-parent diff base. Root changes contain additions.
- Preserve deleted and renamed paths. Correct diff parsing for paths beginning with `b` or `a`.
- Reject incomplete diffs by default. Permit shallow collection only through an explicit option. Record unavailable diffs without calling them successful.
- The Phase 1 implementation clones full history by default. `--max-commits` limits the current collection batch, not clone depth. This default is scheduled to change under the selective-ingestion design.
- Resume commits in parent-before-child order. Skip saved commits so small batches progress across merge branches.
- Commit each document and its checkpoint in one database transaction.
- Store documents under `(repo, doc_type, doc_id)` with database-enforced uniqueness and atomic upsert.
- Validate a local clone against a requested GitHub identity. Prevent combining this project's commits with another project's PRs.
- Keep access tokens out of clone URLs and command arguments. Failed cloning does not remove unrelated files.
- Stream PR/issue updates in timestamp order. Replay timestamp ties on resume.
- Page through merged GraphQL PRs. Remove implicit 50-PR and 200-workflow limits. Detect GraphQL errors and stalled pagination.
- Mark nested issue-reference truncation explicitly. The current GraphQL collector retrieves up to 100 closing references per PR.
- Save collection manifests with coverage, limits, failures and source status.
- Preserve commit-parent and commit–PR links, issue references and merged PR closing statements as raw link records.
- Select one repository for parsing. The API passes that selection to phase two.
- Remove the unused `NamedVector` import that prevented startup with Qdrant client 1.19.1.

## Supported scope

Git collection covers commits reachable from `HEAD`. Diffs use the first parent; all parent identities are retained. Unmerged branches are outside this collection scope.

GitHub collection covers API-visible PRs, issues/comments, review comments, releases, workflow metadata, and merged PR closing-reference metadata. Phase 2 will add selected current-code snapshots. Additional historical discussion and available CI log content will be collected within declared scopes when required. A source marked complete means complete for its stated collection scope, not all historical GitHub data, including deleted or inaccessible records.

Release, workflow and GraphQL batches can be repeated safely after failure. Commit, PR and issue resume cursors are persisted per record. An interrupted GraphQL/workflow batch currently restarts that source.

A failed or limited run is recorded as failed or partial. Nested GraphQL truncation also makes the source partial. Sources not requested are identified separately.

## Existing database behavior

Opening an old document store adds a repository column and uniqueness index. Existing rows remain in the empty legacy namespace. They are not assigned a repository without evidence.

The current corpus was not opened for writing or rebuilt during this phase. The migration was tested against a temporary database with the old schema.

Re-ingest the correct repository before rebuilding its chunks. Old diffs cannot be corrected reliably from stored text alone. Rebuilding vector/keyword indices is deferred until Phase 2 corrects deterministic chunk identities and index replacement.

`run_phase2.py` requires an explicit repository if multiple repositories exist. Old unscoped data requires the explicit `--repo legacy` option. Repository boundaries inside the graph and retrieval indices remain Phase 3 work.

## Commands

These commands describe the current broad-collection implementation. They are not examples of the planned selective mode. Do not assume that `--max-commits` prevents a full clone or limits PR/issue collection.

Collect a local repository. Explicit local paths do not automatically select `GITHUB_REPO` from `.env`:

```powershell
python scripts/ingest.py --repo-path C:/repos/example --skip-github
```

Clone and collect a GitHub repository. Set `GITHUB_TOKEN` in the environment or `.env`:

```powershell
python scripts/ingest.py --github-repo owner/repo
```

Collect 100 commits, then resume the remaining commits:

```powershell
python scripts/ingest.py --github-repo owner/repo --max-commits 100
python scripts/ingest.py --github-repo owner/repo
```

Recollect data after a history rewrite or after correcting previously incomplete records:

```powershell
python scripts/ingest.py --github-repo owner/repo --refresh
```

An existing shallow clone is not automatically converted into a full clone. Fetch its missing history before a full collection, or explicitly permit partial history:

```powershell
python scripts/ingest.py --repo-path C:/repos/example --skip-github --allow-incomplete-history
```

Explicit source limits remain visible in the run manifest:

```powershell
python scripts/ingest.py --github-repo owner/repo --workflow-limit 100 --graphql-limit 50
```

Select a repository for parsing after ingestion:

```powershell
python scripts/run_phase2.py --repo owner/repo
```

## Verification

`tests/test_ingestion.py` contains 30 new acceptance cases, including parameterized identity cases. They exercise real temporary git repositories, SQLite migration/transactions and the CLI. GitHub pagination and errors use deterministic HTTP responses; a complete live GitHub collection was not performed.

API tests also check automatic cloning for the default path and explicit repository selection for parsing. PostgreSQL execution remains unverified in this phase; production verification belongs to Phase 6.

Run the phase tests with the installed project dependencies:

```powershell
python -m pytest tests/test_ingestion.py -q -p no:cacheprovider
```

The initial full run used Qdrant client 1.15.1 and passed 106 tests. After removing the obsolete import, the final suite with Qdrant client 1.19.1 passed **106 tests with 5 warnings in 46.64 seconds**. These are previously observed results. The temporary audit dependencies, scripts, and raw logs were removed at the user's request. Run the tests above to produce new verification evidence in the development environment. A complete deployment dependency lock remains Phase 6 work.
