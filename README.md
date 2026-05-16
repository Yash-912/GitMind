# GitMind (Phase 1)

Phase 1 implements the data infrastructure: local git commit ingestion, GitHub PR/issue ingestion, basic cross-referencing, and SQLite document storage.

## Setup

```bash
pip install -r requirements.txt
```

Copy and edit `.env.example` to `.env`.

## Run ingestion

```bash
python scripts/ingest.py --repo-path . --github-repo <owner/repo>
```

If `GITHUB_TOKEN` is set in `.env`, GitHub PRs and issues will be ingested.
