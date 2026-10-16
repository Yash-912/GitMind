# ─────────────────────────────────────────────────────────────────────────────
#  GitMind — Production Docker Image
#  Targets Hugging Face Spaces (Docker SDK, port 7860)
# ─────────────────────────────────────────────────────────────────────────────
# Pinned to an exact Python patch + Debian release. (Not pinned by digest: no
# registry access when this was written. Add `@sha256:...` for fully
# reproducible builds.)
FROM python:3.11.9-slim-bookworm

# System dependencies (git for GitPython, build tools for tree-sitter + psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# requirements.txt gives the ~= pins; requirements.lock constrains every package
# it covers to the exact versions the tests ran against.
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt -c requirements.lock

# Download spaCy model (en_core_web_sm ~12 MB)
RUN python -m spacy download en_core_web_sm

# ── Application code ─────────────────────────────────────────────────────────
COPY --chown=user . .

# Persistent data dirs (local Qdrant fallback, BM25, SQLite, index generations, jobs)
RUN mkdir -p data/.qdrant data/bm25_index data/.diskcache/embeddings data/generations \
 && chown -R user:user data

USER user

# ── Startup script ───────────────────────────────────────────────────────────
# HF Spaces exposes port 7860.  We run:
#   - FastAPI  on 8000 (internal, called by Streamlit)
#   - Streamlit on 7860 (public-facing UI)
# start.sh waits for the API to be live before starting Streamlit and exits
# non-zero if either process dies, so the container is restarted.
COPY --chown=user start.sh .
RUN chmod +x start.sh

EXPOSE 7860

# Liveness only (the process is up); corpus readiness is GET /ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["./start.sh"]
