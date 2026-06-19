---
title: GitMind
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Ask "why" questions about any codebase history using RAG + temporal graphs
---

# GitMind

GitMind is a codebase archaeology tool that reconstructs architectural decisions from a repository's history. It ingests commits, diffs, PRs, issues, and changelogs; builds a temporal graph; and answers "why" questions with evidence-backed retrieval.

## What It Does

- Ingests local git history and GitHub PR/issue data.
- Parses documents into typed records and extracts entities.
- Chunks and embeds code, prose, and diffs for hybrid retrieval.
- Expands results via a temporal graph of causality.
- Generates direct answers or decision memos with citations.

## Architecture (At a Glance)

```
Sources (git + GitHub)
	-> Ingestion (SQLite / PostgreSQL)
		-> Parsing + Entities
			-> Chunking + Embedding (Ollama / nomic-embed-text)
				-> Indexing (Qdrant Cloud + BM25 + FTS5)
					-> Retrieval + Rerank (cross-encoder)
						-> Generation (Gemini 1.5 Flash)
```

## Production Deployment (Hugging Face Spaces)

This project is deployed as a Docker Space on Hugging Face Spaces.

### Required Secrets (set in HF Space Settings → Repository secrets)

| Secret | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `DATABASE_URL` | PostgreSQL connection string (Neon / Supabase) |
| `GITHUB_TOKEN` | GitHub Personal Access Token (for PR/issue ingestion) |
| `API_KEY` | Optional secret to restrict access to the REST API |

### Local Development

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Start all services with Docker Compose
docker compose up --build
```

Open:
- **Streamlit UI**: http://localhost:7860
- **FastAPI docs**: http://localhost:8000/docs

### Manual Local Run (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run ingestion
python scripts/ingest.py --repo-path . --github-repo <owner/repo>
python scripts/run_phase2.py
python scripts/run_phase3.py

# Start API server
uvicorn api.main:app --port 8000 --reload

# Start Streamlit (in another terminal)
streamlit run interface/streamlit_app.py
```

## Configuration

All settings live in `config/settings.py` and are read from environment variables or `.env`.

## Data Stores

| Store | Local | Production |
|---|---|---|
| Document / graph | SQLite | PostgreSQL (Neon / Supabase) |
| Vector index | Qdrant (local file) | Qdrant Cloud |
| Keyword index | BM25 (local file) | BM25 (rebuilt on startup) |

## Tests

```bash
pytest -q
```

## Project Notes

- Full product spec: [gitmind_prd.md](gitmind_prd.md)
- Build log: [BUILD_LOG.md](BUILD_LOG.md)

## License

TBD
