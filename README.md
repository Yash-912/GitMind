# GitMind

GitMind is a codebase archaeology tool that reconstructs architectural decisions from a repository's history. It ingests commits, diffs, PRs, issues, and changelogs; builds a temporal graph; and answers "why" questions with evidence-backed retrieval.

## What It Does

- Ingests local git history and GitHub PR/issue data into SQLite.
- Parses documents into typed records and extracts entities.
- Chunks and embeds code, prose, and diffs for hybrid retrieval.
- Expands results via a temporal graph of causality.
- Generates direct answers or decision memos with citations.

## Architecture (At a Glance)

```
Sources (git + GitHub)
	-> Ingestion (SQLite)
		-> Parsing + Entities
			-> Chunking + Embedding
				-> Indexing (Qdrant + BM25 + SQLite FTS5)
					-> Retrieval + Rerank
						-> Generation + Evaluation
```

## Requirements

- Python 3.10+ recommended
- Optional: GitHub token for PR/issue ingestion
- Optional: Qdrant (local or server) for vector indexing

## Setup

```bash
pip install -r requirements.txt
```

Copy and edit `.env.example` to `.env`.

## Quickstart (Phases 1-5)

1) Ingest repository and GitHub data

```bash
python scripts/ingest.py --repo-path . --github-repo <owner/repo>
```

If `GITHUB_TOKEN` is set in `.env`, GitHub PRs and issues will be ingested.

2) Parse + entities + chunking

```bash
python scripts/run_phase2.py
```

3) Embedding + indexing

```bash
python scripts/run_phase3.py
```

4) Retrieval demo

```bash
python scripts/run_retrieval.py "why was auth changed"
```

The retrieval demo uses LLM-based query decomposition and entity resolution when a Gemini key is available, with heuristic fallbacks otherwise.

5) End-to-end answer demo

```bash
python scripts/run_answer.py "why was auth changed" --mode direct
```

## Evaluation (Phase 6)

Build a QA dataset:

```bash
python -m interface.cli build-dataset data/chunks.jsonl data/qa_pairs.jsonl --limit 50 --dry-run
```

Run RAGAS evaluation:

```bash
python -m interface.cli evaluate data/qa_pairs.jsonl --limit 20
```

## UI (Phase 7)

```bash
streamlit run interface/streamlit_app.py
```

## Configuration

- App settings: [config/settings.py](config/settings.py)
- Environment variables: `.env`

Common toggles include API keys for LLM providers and embedding backends.

## Data Stores

- SQLite: document store, ingestion checkpoints, entity registry, temporal graph, FTS index
- Qdrant: vector index payloads for retrieval
- BM25: lightweight keyword index (local files)

## Tests

```bash
pytest -q
```

Expected warning (local Qdrant payload indexes):

```
UserWarning: Payload indexes have no effect in the local Qdrant.
```

## Project Notes

- Full product spec and architecture: [gitmind_prd.md](gitmind_prd.md)
- Build log: [BUILD_LOG.md](BUILD_LOG.md)

## Roadmap Ideas

- Better cross-repo linking for monorepos
- Web UI with timelines and evidence graphs
- Pluggable vector backends

## License

TBD
