---
title: GitMind
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Ask "why" questions about codebase history using RAG
---

# GitMind

Implementation and verification are tracked phase by phase in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md). Phase 1 establishes ingestion correctness and resume behavior. Production readiness and evaluation targets remain pending.

GitMind is a codebase archaeology prototype. The target design answers current-code questions from a selected snapshot and expands historical evidence when required. The existing pipeline collects git and GitHub records for retrieval and generation.

## Ingestion Direction

The planned default is selective current-code ingestion, followed by incremental indexing. Download and process selected source files, tests, documentation and configuration at a pinned commit. Update only changed content. Collect relevant commits, PRs and issues when a history question requires them. Full-history collection becomes an explicit background mode with resource limits.

**Current status:** these selective modes are not implemented yet. Phase 1 still defaults to full cloning and broad requested-source collection. Its checkpoints support collection resume; they do not yet update code indices incrementally. See [docs/INGESTION_DESIGN.md](docs/INGESTION_DESIGN.md).

## What It Does

- Ingests local git history and GitHub PR/issue data.
- Parses documents into typed records and extracts entities.
- Chunks and embeds collected prose and diffs for hybrid retrieval. Current-code file indexing is planned.
- Expands results through stored document relationships. Temporal and causal reasoning remain acceptance requirements.
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

# Current broad-collection workflow; selective ingestion is planned
python scripts/ingest.py --github-repo <owner/repo>
python scripts/run_phase2.py --repo <owner/repo>
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

- Implementation status and acceptance requirements: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- Selective ingestion, incremental indexing, and collection budgets: [docs/INGESTION_DESIGN.md](docs/INGESTION_DESIGN.md)
- Phase 1 changes, collection scope, and resume commands: [docs/PHASE1.md](docs/PHASE1.md)
- Full product spec: [gitmind_prd.md](gitmind_prd.md)
- Build log: [BUILD_LOG.md](BUILD_LOG.md)

## License

TBD
