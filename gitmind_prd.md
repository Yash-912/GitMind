# GitMind — Codebase Archaeology Tool
### Complete Project Document: PRD + Technical Architecture + Execution Plan

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Target Users & Use Cases](#2-target-users--use-cases)
3. [What Makes This Architecturally Non-Trivial](#3-what-makes-this-architecturally-non-trivial)
4. [Full Technical Architecture](#4-full-technical-architecture)
5. [Tech Stack — Every Layer Specified](#5-tech-stack--every-layer-specified)
6. [Data Ingestion Pipeline](#6-data-ingestion-pipeline)
7. [Chunking Strategy](#7-chunking-strategy)
8. [Embedding & Indexing Layer](#8-embedding--indexing-layer)
9. [Retrieval Layer](#9-retrieval-layer)
10. [Generation Layer](#10-generation-layer)
11. [Evaluation with RAGAS](#11-evaluation-with-ragas)
12. [Project Phases & Execution Plan](#12-project-phases--execution-plan)
13. [Directory Structure](#13-directory-structure)
14. [What This Demonstrates to Recruiters](#14-what-this-demonstrates-to-recruiters)
15. [Resume One-Liner](#15-resume-one-liner)

---

## 1. Vision & Problem Statement

GitMind is an intelligent codebase memory layer that transforms a repository's entire evolutionary history into a queryable knowledge base. It doesn't just search code — it understands **why** the code is the way it is, **who** shaped it, **what broke it**, and **what tradeoffs** were consciously made by the team over time.

### The Core Problem

Every engineering team suffers from **institutional amnesia**:

- A senior engineer leaves and takes with them the context of 200 architectural decisions
- A new hire touches the payments module and breaks something that was broken before — for a reason nobody remembered
- A tech lead wants to understand why the team moved away from GraphQL in 2022, but that context lives scattered across a Slack thread, a PR description, and three closed GitHub issues nobody linked together
- A new contributor to an open-source project has no way to understand *why* a design choice was made without reading thousands of issues

### Why Existing Tools Fail

| Tool | What It Does | Why It Fails Here |
|---|---|---|
| `git log` / `git blame` | Shows commit history | No semantic search, no causality, no natural language |
| GitHub Search | Keyword search across PRs and issues | No cross-document reasoning, no temporal awareness |
| ChatGPT on codebase | Answers questions about code | No history, no causality, stateless |
| Standard RAG on code | Semantic search over code files | Retrieves *what*, not *why* |
| Confluence / Notion | Documentation | Manually written, always out of date, never comprehensive |

GitMind closes this gap entirely by performing **temporally-aware, multi-hop retrieval across heterogeneous data sources** — commits, diffs, PRs, issues, and changelogs — to reconstruct institutional knowledge on demand.

---

## 2. Target Users & Use Cases

| User | Pain Point | What GitMind Gives Them |
|---|---|---|
| New engineer onboarding | No context on why things are built a certain way | 10 years of institutional knowledge on day 1 |
| Senior engineer / tech lead | Spends hours in archaeology before refactors | Instant decision history for any module |
| Engineering manager | Needs to understand team velocity and churn patterns | Module risk maps, ownership timelines |
| Open-source contributor | Doesn't understand why a design choice was made | PR + issue context surfaced alongside the code |
| SRE / DevOps engineer | Needs to know what changed before an incident | Causality chains across deploys and commits |

### Example Queries GitMind Can Answer

- *"Why does the auth module use JWT instead of session cookies?"*
- *"What broke every time someone touched the payments service?"*
- *"Who made the decision to move to async architecture in 2022 and why?"*
- *"What modules have the highest churn rate and what are the common reasons?"*
- *"What was the original intent behind the current retry logic?"*
- *"Are there any decisions made in 2019 that were later contradicted?"*

---

## 3. What Makes This Architecturally Non-Trivial

This is not a chatbot over a codebase. The complexity lives in five distinct challenges that most RAG systems never face.

### 3.1 Heterogeneous Data Fusion
You are simultaneously ingesting and linking: raw code files, git commit messages, git diffs (structured patches), PR titles + bodies + review comments, issue titles + body + labels + close reason, changelogs, and CI/CD failure logs. Each has a different schema, different semantic register, and a different relationship to time.

### 3.2 Temporal Retrieval
Standard RAG is stateless — it doesn't know that document A came before document B. GitMind must understand that a commit in March 2021 *caused* an issue opened in April 2021, which *caused* a PR merged in May 2021, which *introduced* a pattern still visible today. Retrieval must be temporally aware and causality-sensitive.

### 3.3 Multi-Hop Reasoning
A question like *"why does the auth module use JWT?"* cannot be answered from a single document. It requires: finding where JWT was introduced (commit), finding the PR that introduced it (PR body), finding the issue or discussion that motivated the PR (issue thread), and possibly finding a prior commit where sessions were removed (diff). The system must chain these hops.

### 3.4 Entity-Centric Indexing
The system must understand that "auth module", "authentication", "auth.py", "login flow", and "JWT middleware" are all the same conceptual entity across different documents. This requires building an entity graph on top of the vector index — not just embed-and-retrieve.

### 3.5 Structured + Unstructured in One Pipeline
Git diffs are structured (old line / new line / file path / hunk). Issues have metadata (labels, assignees, open/close state, timestamps). PRs have structured review states. Code has AST structure. Yet all of this must live in a retrieval pipeline alongside free-form prose from commit messages and issue comments. Fusing these is a genuine systems challenge.

---

## 4. Full Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                   │
│  Git Clone  │  GitHub REST API  │  GitHub GraphQL API  │  Local Files   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE                                │
│                                                                         │
│  GitPython Collector ──► Multi-Schema Parser ──► Entity Extractor       │
│                                    │                      │             │
│                                    ▼                      ▼             │
│                          Temporal Graph Builder    Entity Registry      │
│                                    │                      │             │
│                                    └──────────┬───────────┘             │
│                                               ▼                         │
│                                     Document Store (SQLite)             │
└───────────────────────────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CHUNKING & EMBEDDING                                │
│                                                                         │
│  Per-Type Chunker ──► Dual Embedding                                    │
│  (code / prose /        Code: text-embedding-3-small (OpenAI)           │
│   diff / issue)         Prose: same model, different prompt prefix      │
│                                    │                                    │
│                                    ▼                                    │
│              Qdrant Vector Store (metadata-rich payload)                │
│              + BM25 Keyword Index (bm25s)                               │
│              + Entity Index (SQLite FTS5)                               │
└───────────────────────────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL LAYER                                  │
│                                                                         │
│  Query ──► Query Decomposer (LLM)                                       │
│                │                                                        │
│                ├──► Entity Resolver (FTS5 lookup)                       │
│                ├──► Time Range Extractor (regex + LLM)                  │
│                └──► Intent Classifier (decision / bug / ownership)      │
│                                                                         │
│  Hybrid Retriever:                                                      │
│  Dense (Qdrant) + Sparse (BM25) ──► RRF Fusion ──► Metadata Filter     │
│                                                                         │
│  Graph Walker ──► Temporal Expansion of Retrieved Nodes                 │
│                                                                         │
│  Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)                       │
└───────────────────────────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       GENERATION LAYER                                  │
│                                                                         │
│  Context Assembler ──► Prompt Builder ──► LLM (GPT-4o / Claude)        │
│                                                    │                    │
│                            ┌───────────────────────┤                    │
│                            │                       │                    │
│                            ▼                       ▼                    │
│                    Direct Answer         Decision Memo Generator        │
│                    + Evidence Chain      + Blame Map Generator          │
│                    + Confidence Score    + Change Frequency Report      │
└─────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION (RAGAS)                                │
│                                                                         │
│  Faithfulness │ Answer Relevancy │ Context Precision │ Context Recall   │
│  + Custom Metrics: Temporal Accuracy, Multi-hop Score, Entity F1        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Tech Stack — Every Layer Specified

### 5.1 Data Collection & Parsing

| Component | Technology | Why |
|---|---|---|
| Git history extraction | `GitPython` | Full programmatic access to commits, diffs, branches, tags |
| GitHub API (PRs, Issues) | `PyGitHub` + `httpx` | REST for structured data, async for bulk pulls |
| Diff parsing | `unidiff` | Parses unified diff format into structured hunks |
| Code AST parsing | `tree-sitter` (Python bindings) | Language-aware code parsing for function/class extraction |
| HTML/Markdown cleaning | `markdownify` + `BeautifulSoup4` | Clean issue/PR bodies from raw Markdown or HTML |
| Rate limit handling | `tenacity` | Exponential backoff for GitHub API rate limits |

### 5.2 Entity Extraction & NLP

| Component | Technology | Why |
|---|---|---|
| NER on prose (commit/issue text) | `spaCy` (en_core_web_trf) | Extracts people, technologies, module names |
| Code entity extraction | `tree-sitter` | Functions, classes, imports from diffs |
| Entity normalization | Custom registry + fuzzy matching via `rapidfuzz` | Resolves "auth", "authentication.py", "login middleware" to same entity |
| Keyword extraction | `KeyBERT` | Unsupervised key phrase extraction from longer issue threads |

### 5.3 Chunking

| Component | Technology | Why |
|---|---|---|
| Semantic chunking for prose | `langchain_text_splitters.SemanticChunker` | Splits on meaning boundaries, not character count |
| Code-aware chunking | Custom splitter using `tree-sitter` AST | Respects function/class boundaries in diffs |
| Diff chunking | Custom hunk-aware splitter on top of `unidiff` | Keeps file path + hunk context intact |
| Chunk metadata attachment | Custom `ChunkMetadata` dataclass | Stamps every chunk with type, timestamp, author, entity tags, graph node ID |

### 5.4 Embedding

| Component | Technology | Why |
|---|---|---|
| Primary embedding model | `text-embedding-3-small` (OpenAI API) | Strong performance, cheap at scale, 1536 dims |
| Local fallback / offline | `sentence-transformers/all-MiniLM-L6-v2` | No API cost, good baseline, runs on CPU |
| Code-specific embedding | `voyage-code-2` (VoyageAI API) or `nomic-embed-code` | Trained on code, outperforms general models on diffs |
| Embedding cache | `diskcache` | Avoids re-embedding unchanged chunks across re-ingestions |
| Batch embedding | Custom async batcher with `asyncio` + `httpx` | Efficient bulk embedding with rate limit handling |

### 5.5 Vector Store & Indexes

| Component | Technology | Why |
|---|---|---|
| Vector store | `Qdrant` (local mode, no server needed) | Persistent local mode, rich metadata filtering, hybrid search support |
| Sparse / keyword index | `bm25s` | Pure Python BM25, no Elasticsearch needed, fast |
| Entity + metadata index | `SQLite` with FTS5 extension | Built into Python, full-text search, stores the temporal graph as adjacency list |
| Fusion algorithm | Reciprocal Rank Fusion (RRF) — custom implementation | Merges dense + sparse ranked lists without a learned ranker |

### 5.6 Retrieval

| Component | Technology | Why |
|---|---|---|
| Query decomposition | `GPT-4o-mini` with structured output (JSON mode) | Cheap, fast, good at extraction tasks |
| Hybrid retrieval orchestration | Custom `HybridRetriever` class | Coordinates Qdrant + BM25 + metadata filter |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` | Best open-source reranker, runs locally, massive quality improvement |
| Graph walking | Custom `TemporalGraphWalker` on SQLite adjacency list | Expands retrieved nodes to causally linked documents |
| Context window management | Custom `ContextAssembler` with token counting via `tiktoken` | Ensures context fits model limits with priority-based truncation |

### 5.7 Generation (LLM)

| Component | Technology | Why |
|---|---|---|
| Primary LLM | `GPT-4o` via OpenAI API | Best reasoning for multi-hop, citation-aware generation |
| Fallback / local LLM | `Ollama` + `mistral` or `llama3` | Zero-cost local option, works offline |
| LLM orchestration | `LangChain` LCEL (minimal usage) or raw API calls | Keeps the pipeline transparent and debuggable |
| Structured output | Pydantic models + OpenAI JSON mode | Guarantees parseable responses for Decision Memos and Blame Maps |
| Prompt management | `Jinja2` templates | Separates prompt logic from Python code, version-controllable |

### 5.8 Storage & State

| Component | Technology | Why |
|---|---|---|
| Raw document store | `SQLite` via `SQLModel` | Lightweight, no server, queryable |
| Ingestion state / checkpointing | `SQLite` + custom `IngestionCheckpoint` | Resume long ingestions without re-fetching |
| Embedding cache | `diskcache` on local filesystem | Persistent across runs |
| Vector store persistence | `Qdrant` local storage (`.qdrant/` folder) | Survives restarts, no Docker |
| Config management | `pydantic-settings` + `.env` file | Type-safe config, secrets out of code |

### 5.9 Evaluation

| Component | Technology | Why |
|---|---|---|
| Core RAG evaluation | `RAGAS` | Industry standard: Faithfulness, Answer Relevancy, Context Precision/Recall |
| LLM judge for evaluation | `GPT-4o` (via RAGAS config) | RAGAS uses LLM-as-judge internally |
| Custom metric framework | `RAGAS` custom metrics API | For temporal accuracy, multi-hop score, entity F1 |
| Evaluation dataset creation | Semi-synthetic: real repo + GPT-4o generated Q&A | Realistic ground truth without manual labeling |
| Experiment tracking | `MLflow` (local) | Log metric runs, compare chunking/retrieval configs |

### 5.10 Interface & Developer Experience

| Component | Technology | Why |
|---|---|---|
| Primary interface | `Streamlit` | Fast to build, looks good in demos, no frontend skill needed |
| CLI interface | `Typer` | Clean CLI for ingestion commands and batch queries |
| Logging | `loguru` | Better than stdlib logging, structured output |
| Progress bars | `rich` + `tqdm` | Visual feedback during long ingestions |
| Testing | `pytest` + `pytest-asyncio` | Unit and integration tests across pipeline |

---

## 6. Data Ingestion Pipeline

### 6.1 What Gets Ingested

```
Repository
├── Git Commits
│   ├── Hash, author, timestamp, message
│   └── Full diff (parsed into file-level hunks)
├── Pull Requests (via GitHub API)
│   ├── Title, body, labels, state, merge status
│   ├── Review comments (threaded)
│   └── Linked issues (extracted from body + API)
├── Issues (via GitHub API)
│   ├── Title, body, labels, assignees, state
│   ├── Comment thread
│   └── Close reason / linked PR
├── Releases / Tags
│   ├── Tag name, timestamp
│   └── Release notes body (Markdown)
└── CHANGELOG.md (if present)
    └── Parsed per-version sections
```

### 6.2 Ingestion Flow

```
Step 1: Git Clone → extract all commits with diffs via GitPython
Step 2: GitHub API pull → paginate all PRs, issues, reviews in parallel
Step 3: Cross-reference linking → match commit SHA → PR, PR → issue
Step 4: Entity extraction → run spaCy + tree-sitter over all documents
Step 5: Entity normalization → build entity registry, resolve aliases
Step 6: Temporal graph construction → build SQLite adjacency list
Step 7: Per-document chunking → apply type-specific chunker
Step 8: Embedding → batch embed all chunks, cache to disk
Step 9: Index population → upsert into Qdrant + BM25 + FTS5
Step 10: Checkpoint save → mark ingestion complete for this repo @ HEAD
```

### 6.3 Incremental Re-ingestion

On subsequent runs, GitMind checks the checkpoint and only ingests:
- Commits after the last ingested SHA
- PRs and issues updated after the last ingestion timestamp

This makes re-ingestion fast for active repos.

---

## 7. Chunking Strategy

Chunking is the most impactful design decision in the pipeline. GitMind uses **per-document-type chunking**, not generic character-count splitting.

### 7.1 Commit Messages
- Kept whole as a single chunk
- Short context window: 256 tokens max
- Metadata: hash, author, timestamp, files touched, entity tags

### 7.2 Git Diffs
- Chunked **per file per hunk** using `unidiff`
- Each chunk preserves: file path, old line range, new line range, surrounding context lines (±5)
- Chunk size: typically 400–800 tokens depending on hunk size
- Oversized hunks split at logical boundaries using tree-sitter AST (function boundaries)

### 7.3 PR Bodies
- Kept whole if under 1000 tokens
- Split at Markdown section boundaries if longer
- Review comments: each top-level comment + its thread as one chunk

### 7.4 Issue Bodies + Comments
- Issue body: whole chunk
- Comment thread: sliding window of 3 consecutive comments with 1-comment overlap
- Labels and close-reason appended as metadata, not embedded text

### 7.5 Release Notes / CHANGELOG
- Split per version section (e.g., `## v2.3.0 — 2022-04-10`)
- Each version section is one chunk with version + date in metadata

### 7.6 Chunk Metadata Schema

Every chunk carries:

```python
@dataclass
class ChunkMetadata:
    chunk_id: str           # UUID
    doc_type: str           # "commit" | "diff" | "pr" | "issue" | "release"
    doc_id: str             # Commit SHA, PR number, issue number, etc.
    timestamp: datetime     # When this event happened in the repo
    author: str             # GitHub username or git author
    module_tags: list[str]  # Entity-resolved module names
    entity_tags: list[str]  # Tech names, people, concepts
    graph_node_id: str      # ID in temporal graph
    file_paths: list[str]   # Files this chunk relates to (if applicable)
    repo: str               # Repo slug
```

---

## 8. Embedding & Indexing Layer

### 8.1 Dual Embedding Strategy

Each chunk gets embedded twice:
- **Semantic embedding**: `text-embedding-3-small` with a prose-optimized prefix for commit messages, issue text, PR bodies
- **Code embedding**: `voyage-code-2` for diff chunks and code file chunks

Both vectors are stored in Qdrant as named vectors on the same point. Retrieval queries both and fuses results.

### 8.2 Qdrant Collection Schema

```python
# Each point in Qdrant
{
  "id": "<chunk_uuid>",
  "vectors": {
    "semantic": [...],   # 1536-dim from text-embedding-3-small
    "code": [...]        # 1024-dim from voyage-code-2
  },
  "payload": {
    # Full ChunkMetadata fields
    "doc_type": "commit",
    "timestamp": "2021-03-15T14:22:00Z",
    "author": "gvanrossum",
    "module_tags": ["parser", "ast"],
    "entity_tags": ["PEG parser", "tokenizer"],
    "graph_node_id": "commit_abc123",
    "file_paths": ["Lib/ast.py", "Parser/parser.c"],
    "text": "<the actual chunk text>"
  }
}
```

### 8.3 Metadata Filtering

Qdrant's payload filter is applied *before* ANN search (not post-filtering), which makes it fast:

```python
# Example: only search commits touching auth module after 2021
filter = Filter(must=[
    FieldCondition(key="module_tags", match=MatchAny(any=["auth"])),
    FieldCondition(key="timestamp", range=DatetimeRange(gte="2021-01-01"))
])
```

### 8.4 BM25 Index

`bm25s` is initialized over the same corpus. Stored as a serialized index file. Used in parallel with Qdrant for hybrid retrieval.

### 8.5 SQLite FTS5 Entity Index

```sql
CREATE VIRTUAL TABLE entity_index USING fts5(
    entity_name,
    aliases,
    doc_ids,         -- comma-separated list of doc IDs containing this entity
    entity_type      -- "module" | "tech" | "person" | "concept"
);
```

This enables fast exact-match + fuzzy entity resolution before semantic search.

---

## 9. Retrieval Layer

### 9.1 Query Processing Pipeline

```
Raw Query
    │
    ▼
Query Decomposer (GPT-4o-mini, JSON output)
    ├── extracted_entities: ["auth module", "JWT"]
    ├── time_range: {"start": "2021-01-01", "end": "2022-12-31"}
    ├── intent: "decision_archaeology"
    └── sub_queries: [
            "JWT introduction commit",
            "session cookies removal",
            "auth architecture discussion"
        ]
    │
    ▼
Entity Resolver (FTS5 lookup)
    └── "auth module" → module_tag: "authentication"
        "JWT" → entity_tag: "JWT", "jsonwebtoken"
    │
    ▼
Hybrid Retriever (per sub-query)
    ├── Qdrant dense search (filtered by entity + time)
    ├── BM25 sparse search (same query)
    └── RRF fusion → top-40 candidates
    │
    ▼
Graph Walker
    └── For each retrieved node, fetch:
        ├── Parent commit of this PR
        ├── Issues linked to this PR
        └── Follow-up commits referencing same issue
    │
    ▼
Merged Candidate Pool (deduplicated, ~60-80 chunks)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
    └── Top-12 chunks selected
    │
    ▼
Context Assembler
    └── Ordered by timestamp, metadata injected, token budget managed
    │
    ▼
LLM Generation
```

### 9.2 RRF Fusion

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = defaultdict(float)
    for rank, doc in enumerate(dense_results):
        scores[doc.id] += 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 9.3 Graph Walker Logic

The temporal graph in SQLite stores edges:

```
commit_X → pr_42          (this commit closed this PR)
pr_42 → issue_17          (this PR was opened for this issue)
issue_17 → commit_Y       (this commit was referenced in the issue)
commit_Y → commit_X       (parent commit)
```

After initial retrieval, the walker performs 1-hop expansion by default (configurable to 2-hop). This means if a PR is retrieved, its linked issues and merge commits are also included.

---

## 10. Generation Layer

### 10.1 Answer Modes

**Mode 1: Direct Q&A**
Returns: direct answer, evidence chain (list of source docs with metadata), confidence level (high/medium/low based on retrieval scores), suggested follow-up questions.

**Mode 2: Decision Memo Generator**
Given a query about a design decision, generates a structured output:
```
DECISION: [what was decided]
DATE: [when it was decided, reconstructed from evidence]
DECISION MAKERS: [authors of key commits / PR reviewers]
CONTEXT: [what problem prompted this decision]
ALTERNATIVES CONSIDERED: [other approaches mentioned in PRs/issues]
RATIONALE: [why this approach was chosen, from PR discussions]
CONSEQUENCES: [subsequent commits/issues that reference this decision]
EVIDENCE: [list of source documents]
```

**Mode 3: Module Blame Map**
Given a module name, generates a chronological ownership + decision timeline showing who made what decisions about this module and when.

**Mode 4: Risk Report**
Given a module, produces: commit frequency over time, number of distinct authors, bug-to-feature ratio in issue history, modules with highest coupling (co-changed files).

### 10.2 Prompt Template Structure

```
System: You are GitMind, a codebase historian. You answer questions about 
        architectural decisions using evidence from git history, pull requests, 
        and issue discussions. Always cite your sources by referencing the 
        doc_type and doc_id of the evidence you use.

Context:
[ASSEMBLED CHUNKS — ordered by timestamp, formatted as:]
---
[{doc_type}] [{doc_id}] [{timestamp}] [{author}]
{chunk_text}
---

Question: {user_query}

Answer in this format:
- Direct answer
- Evidence chain: list each source document you used
- Confidence: high/medium/low
- Caveats: any gaps or uncertainties in the evidence
```

---

## 11. Evaluation with RAGAS

### 11.1 Why RAGAS

RAGAS (Retrieval Augmented Generation Assessment) is the standard open-source framework for evaluating RAG pipelines. It measures what matters: not just whether the answer sounds good, but whether it is **faithful to the retrieved context** and whether the **right context was retrieved**.

### 11.2 Core RAGAS Metrics Used

| Metric | What It Measures | Target Score |
|---|---|---|
| `Faithfulness` | Does the answer only make claims supported by the retrieved context? | > 0.85 |
| `Answer Relevancy` | Is the answer actually relevant to the question asked? | > 0.80 |
| `Context Precision` | Are the retrieved chunks actually useful for answering the question? | > 0.75 |
| `Context Recall` | Does the retrieved context contain all information needed to answer? | > 0.70 |
| `Answer Correctness` | End-to-end: is the final answer factually correct? (requires ground truth) | > 0.75 |

### 11.3 Custom RAGAS Metrics

RAGAS allows defining custom metrics. GitMind adds three domain-specific metrics:

**Temporal Accuracy Score**
Measures whether the answer correctly identifies *when* events happened. Evaluated by checking if dates/timeframes mentioned in the answer match timestamps in the source documents.

```python
class TemporalAccuracyMetric(Metric):
    name = "temporal_accuracy"
    # Extracts date references from answer, checks against source metadata
```

**Multi-hop Coverage Score**
Measures whether the answer draws from multiple document types (e.g., both a PR and its linked issue). Single-source answers on multi-hop questions are penalized.

```python
class MultiHopCoverageMetric(Metric):
    name = "multihop_coverage"
    # Checks doc_type diversity in the retrieved + cited sources
```

**Entity Consistency Score**
Measures whether the entity references in the answer (module names, author names, technologies) are consistent with the entity tags in the retrieved chunks.

```python
class EntityConsistencyMetric(Metric):
    name = "entity_consistency"
    # Cross-checks named entities in answer vs entity_tags in chunk metadata
```

### 11.4 Evaluation Dataset Construction

Creating a ground-truth evaluation dataset without manual labeling:

**Step 1: Synthetic QA Generation**
For each high-confidence retrieved cluster (a PR + its linked commits + issues), use GPT-4o to generate realistic questions a developer might ask, with reference answers derived strictly from the cluster.

**Step 2: Adversarial Questions**
Generate questions where the answer is *not* in the corpus (to test hallucination resistance). Correct behavior: "I couldn't find clear evidence for this in the repository history."

**Step 3: Temporal Questions**
Questions specifically requiring temporal reasoning: "What happened *after* the JWT PR was merged?" Tests the graph walker and temporal ordering.

**Step 4: Multi-hop Questions**
Questions that require linking 3+ documents: "What user complaint ultimately led to the introduction of the rate limiter?" Tests the full chain from issue → PR → commit.

### 11.5 Experiment Tracking with MLflow

```python
import mlflow

with mlflow.start_run(run_name="hybrid_retrieval_v3"):
    mlflow.log_params({
        "embedding_model": "text-embedding-3-small",
        "reranker": "ms-marco-MiniLM-L-6-v2",
        "top_k": 12,
        "graph_hop_depth": 1,
        "chunk_size": 512
    })
    
    results = ragas.evaluate(dataset, metrics=[...])
    
    mlflow.log_metrics({
        "faithfulness": results["faithfulness"],
        "context_precision": results["context_precision"],
        "context_recall": results["context_recall"],
        "answer_relevancy": results["answer_relevancy"],
        "temporal_accuracy": results["temporal_accuracy"],
        "multihop_coverage": results["multihop_coverage"]
    })
```

---

## 12. Project Phases & Execution Plan

### Phase 1: Data Infrastructure (Week 1–2)

- [ ] Set up project repository with directory structure
- [ ] Implement `GitCollector` using GitPython — pull all commits with diffs
- [ ] Implement `GitHubAPICollector` — pull PRs, issues, review comments
- [ ] Implement `CrossReferenceLinker` — link commits → PRs → issues
- [ ] Build `DocumentStore` in SQLite using SQLModel
- [ ] Implement ingestion checkpointing for resume capability
- [ ] Write unit tests for all collectors
- [ ] Test on FastAPI repository (manageable size, rich history)

**Milestone**: All raw data for FastAPI repo ingested and stored in SQLite.

---

### Phase 2: Parsing, Entities & Chunking (Week 3–4)

- [ ] Implement `MultiSchemaParser` — per-type structured extraction
- [ ] Integrate `unidiff` for structured diff parsing
- [ ] Integrate `tree-sitter` for code entity extraction
- [ ] Build `EntityExtractor` using spaCy
- [ ] Build `EntityRegistry` with `rapidfuzz` for alias resolution
- [ ] Implement `TemporalGraphBuilder` — construct SQLite adjacency list
- [ ] Implement per-type chunkers (commit, diff, PR, issue, changelog)
- [ ] Attach `ChunkMetadata` to every chunk
- [ ] Write tests for chunking edge cases (empty diffs, giant PRs, etc.)

**Milestone**: All documents chunked with rich metadata. Entity graph built.

---

### Phase 3: Embedding & Index (Week 5)

- [ ] Set up Qdrant in local persistence mode
- [ ] Implement dual embedding pipeline (semantic + code vectors)
- [ ] Implement `diskcache`-based embedding cache
- [ ] Implement async batch embedder with rate limit handling
- [ ] Build BM25 index using `bm25s`
- [ ] Build SQLite FTS5 entity index
- [ ] Upsert all chunks into all three indexes
- [ ] Test retrieval sanity (basic keyword and semantic searches)

**Milestone**: All chunks embedded and indexed. Basic retrieval working.

---

### Phase 4: Retrieval Pipeline (Week 6–7)

- [ ] Implement `QueryDecomposer` using GPT-4o-mini with JSON output
- [ ] Implement `EntityResolver` using FTS5 index
- [ ] Implement `HybridRetriever` with RRF fusion
- [ ] Implement `TemporalGraphWalker`
- [ ] Integrate cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- [ ] Implement `ContextAssembler` with `tiktoken` token budgeting
- [ ] End-to-end retrieval test: complex multi-hop queries
- [ ] Tune retrieval hyperparameters (top_k, graph_hop_depth, RRF k)

**Milestone**: Full retrieval pipeline working. Multi-hop queries returning relevant context.

---

### Phase 5: Generation & Output Modes (Week 8)

- [ ] Build Jinja2 prompt templates for each answer mode
- [ ] Implement `DirectQAGenerator`
- [ ] Implement `DecisionMemoGenerator` with Pydantic structured output
- [ ] Implement `BlameMapGenerator`
- [ ] Implement `RiskReportGenerator`
- [ ] Add Ollama fallback for local LLM usage
- [ ] End-to-end demo on FastAPI repo: real questions, real answers

**Milestone**: All output modes working. Demo-ready on FastAPI.

---

### Phase 6: Evaluation (Week 9–10)

- [ ] Build synthetic evaluation dataset (100 QA pairs) using GPT-4o
- [ ] Include adversarial, temporal, and multi-hop question types
- [ ] Set up RAGAS evaluation pipeline
- [ ] Implement three custom RAGAS metrics
- [ ] Run baseline evaluation (no reranker, no graph walk)
- [ ] Run full pipeline evaluation
- [ ] Set up MLflow experiment tracking
- [ ] Log 5+ experiment configurations, compare metrics
- [ ] Document findings: which components improve which metrics
- [ ] Write evaluation report

**Milestone**: RAGAS evaluation complete with documented metric comparisons across configurations.

---

### Phase 7: Interface & Polish (Week 11)

- [ ] Build Streamlit interface with:
  - Repo ingestion form
  - Query input with answer mode selector
  - Answer display with expandable evidence chain
  - Timeline visualization of retrieved documents
  - Module selector for Blame Map and Risk Report
- [ ] Build `Typer` CLI for ingestion and batch querying
- [ ] Write comprehensive README with demo GIF
- [ ] Add second repo: Django or CPython subsystem
- [ ] Final end-to-end demo pass

**Milestone**: Project demo-ready for interviews and GitHub showcase.

---

## 13. Directory Structure

```
gitmind/
├── README.md
├── pyproject.toml              # Dependencies managed by uv or poetry
├── .env.example                # API keys template
│
├── config/
│   └── settings.py             # pydantic-settings config
│
├── ingestion/
│   ├── git_collector.py        # GitPython-based commit + diff extraction
│   ├── github_collector.py     # PyGitHub PR, issue, review extraction
│   ├── cross_referencer.py     # Links commits → PRs → issues
│   ├── checkpoint.py           # Ingestion state management
│   └── document_store.py       # SQLite storage via SQLModel
│
├── parsing/
│   ├── multi_schema_parser.py  # Per-type structured extraction
│   ├── diff_parser.py          # unidiff-based hunk parsing
│   ├── code_parser.py          # tree-sitter AST extraction
│   └── text_cleaner.py         # Markdown/HTML normalization
│
├── entities/
│   ├── entity_extractor.py     # spaCy + tree-sitter NER
│   ├── entity_registry.py      # Alias resolution via rapidfuzz
│   └── temporal_graph.py       # SQLite adjacency list builder + walker
│
├── chunking/
│   ├── base_chunker.py         # Abstract base class
│   ├── commit_chunker.py
│   ├── diff_chunker.py
│   ├── pr_chunker.py
│   ├── issue_chunker.py
│   ├── changelog_chunker.py
│   └── chunk_metadata.py       # ChunkMetadata dataclass
│
├── embedding/
│   ├── embedder.py             # Dual embedding with async batching
│   ├── embedding_cache.py      # diskcache wrapper
│   └── models.py               # Model client abstractions
│
├── indexing/
│   ├── qdrant_store.py         # Qdrant collection management
│   ├── bm25_index.py           # bm25s wrapper
│   └── fts_index.py            # SQLite FTS5 entity index
│
├── retrieval/
│   ├── query_decomposer.py     # GPT-4o-mini structured query decomposition
│   ├── entity_resolver.py      # FTS5-based entity lookup
│   ├── hybrid_retriever.py     # Dense + sparse + RRF fusion
│   ├── graph_walker.py         # Temporal graph expansion
│   ├── reranker.py             # Cross-encoder reranker
│   └── context_assembler.py    # Token-budget-aware context builder
│
├── generation/
│   ├── llm_client.py           # OpenAI + Ollama abstraction
│   ├── prompt_templates/
│   │   ├── direct_qa.j2
│   │   ├── decision_memo.j2
│   │   ├── blame_map.j2
│   │   └── risk_report.j2
│   ├── direct_qa.py
│   ├── decision_memo.py
│   ├── blame_map.py
│   └── risk_report.py
│
├── evaluation/
│   ├── dataset_builder.py      # Synthetic QA dataset generation
│   ├── ragas_evaluator.py      # RAGAS pipeline setup and execution
│   ├── custom_metrics/
│   │   ├── temporal_accuracy.py
│   │   ├── multihop_coverage.py
│   │   └── entity_consistency.py
│   ├── mlflow_logger.py        # Experiment tracking
│   └── eval_datasets/          # Stored evaluation datasets (JSON)
│
├── interface/
│   ├── streamlit_app.py        # Main Streamlit UI
│   └── cli.py                  # Typer CLI
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── notebooks/
    ├── 01_ingestion_exploration.ipynb
    ├── 02_chunking_analysis.ipynb
    ├── 03_retrieval_experiments.ipynb
    └── 04_ragas_results.ipynb
```

---

## 14. What This Demonstrates to Recruiters

| Technical Concept | Where It Appears |
|---|---|
| End-to-end RAG pipeline design | Full architecture |
| Hybrid retrieval (dense + sparse + RRF) | `hybrid_retriever.py` |
| Multi-hop retrieval over a knowledge graph | `graph_walker.py` |
| Heterogeneous data ingestion | Ingestion pipeline |
| Per-type chunking strategy | `chunking/` module |
| Metadata-filtered vector search | `qdrant_store.py` |
| Cross-encoder reranking | `reranker.py` |
| Temporal reasoning in retrieval | `temporal_graph.py` + `context_assembler.py` |
| Entity resolution | `entity_registry.py` |
| Structured LLM output with Pydantic | `decision_memo.py` |
| Dual embedding (semantic + code-specific) | `embedder.py` |
| RAG evaluation with RAGAS | `evaluation/` module |
| Custom RAGAS metrics | `custom_metrics/` |
| Experiment tracking with MLflow | `mlflow_logger.py` |
| Async data pipelines | `github_collector.py`, `embedder.py` |
| Prompt engineering | `prompt_templates/` |
| Systems thinking | End-to-end design |

---

## 15. Resume One-Liner

> *"Built GitMind, an LLM-powered codebase archaeology system performing multi-hop, temporally-aware retrieval across git history, PRs, and issue threads using hybrid dense-sparse retrieval with cross-encoder reranking, evaluated end-to-end with RAGAS and custom temporal accuracy metrics."*

---

*Document version: 1.0 | Project: GitMind | Status: Pre-build*
