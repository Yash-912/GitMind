# GitMind (Phases 1-7)

Phase 1 implements the data infrastructure: local git commit ingestion, GitHub PR/issue ingestion, basic cross-referencing, and SQLite document storage. Phase 2 adds parsing, entities, chunking, and a temporal graph. Phase 3 embeds and indexes chunks. Phase 4-7 provide retrieval, generation, evaluation, and interfaces.

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

## Run Phase 2 (parse + entities + chunking)

```bash
python scripts/run_phase2.py
```

## Run Phase 3 (embedding + indexing)

```bash
python scripts/run_phase3.py
```

## Run Retrieval Demo (Phase 4)

```bash
python scripts/run_retrieval.py "why was auth changed"
```

The retrieval demo uses LLM-based query decomposition and entity resolution when a Gemini key is available, with heuristic fallbacks otherwise.

## Run End-to-End Answer Demo (Phase 5)

```bash
python scripts/run_answer.py "why was auth changed" --mode direct
```

## Build Evaluation Dataset (Phase 6)

```bash
python -m interface.cli build-dataset data/chunks.jsonl data/qa_pairs.jsonl --limit 50 --dry-run
```

## Run RAGAS Evaluation (Phase 6)

```bash
python -m interface.cli evaluate data/qa_pairs.jsonl --limit 20
```

## Run UI (Phase 7)

```bash
streamlit run interface/streamlit_app.py
```

## Build Log

See [BUILD_LOG.md](BUILD_LOG.md) for a detailed, chronological record of post-learning implementation changes.
