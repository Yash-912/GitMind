# ─────────────────────────────────────────────────────────────────────────────
#  GitMind — Production Docker Image
#  Targets Hugging Face Spaces (Docker SDK, port 7860)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download spaCy model (en_core_web_sm ~12 MB)
RUN python -m spacy download en_core_web_sm

# ── Application code ─────────────────────────────────────────────────────────
COPY --chown=user . .

# Create persistent data dirs (Qdrant local fallback, BM25, SQLite)
RUN mkdir -p data/.qdrant data/bm25_index data/.diskcache/embeddings \
 && chown -R user:user data

USER user

# ── Startup script ───────────────────────────────────────────────────────────
# HF Spaces exposes port 7860.  We run:
#   - FastAPI  on 8000 (internal, called by Streamlit)
#   - Streamlit on 7860 (public-facing UI)
COPY --chown=user start.sh .
RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
