"""entity_extractor.py — Extract named entities from text using spaCy + tree-sitter."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedEntity:
    """A single entity occurrence found in a document."""

    text: str               # surface form as it appears in the document
    label: str              # "PERSON" | "ORG" | "TECH" | "MODULE" | "CONCEPT" | etc.
    source_doc_id: str      # which document this was found in
    doc_type: str           # "commit" | "diff" | "pr" | "issue" | "release"
    start_char: int = 0
    end_char: int = 0


# Common tech-stack keywords to catch when spaCy misses them
_TECH_PATTERNS = re.compile(
    r"\b("
    r"JWT|OAuth|GraphQL|REST|gRPC|WebSocket|CORS|CSRF|SQL|NoSQL|"
    r"Kafka|Redis|RabbitMQ|Celery|Django|FastAPI|Flask|Express|"
    r"React|Vue|Angular|Next\.js|Nuxt|Svelte|"
    r"Postgres(?:QL)?|MySQL|MongoDB|SQLite|Qdrant|Elasticsearch|Pinecone|"
    r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Terraform|Ansible|"
    r"Python|JavaScript|TypeScript|Go(?:lang)?|Rust|Java|C\+\+|Ruby|"
    r"GitHub Actions|CI/CD|pytest|Jest|mypy|ruff|black|"
    r"asyncio|aiohttp|httpx|requests|pydantic|SQLModel|SQLAlchemy|"
    r"LangChain|LlamaIndex|OpenAI|Anthropic|Ollama|Mistral|"
    r"RAGAS|MLflow|Weights\s?&\s?Biases|W&B"
    r")\b",
    re.IGNORECASE,
)


class EntityExtractor:
    """Extract entities from text.

    Uses spaCy (en_core_web_trf or en_core_web_sm) when available,
    supplemented by a regex pass for tech-stack keywords.
    Falls back to regex-only when spaCy is not installed.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._nlp = self._load_spacy(spacy_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        doc_id: str,
        doc_type: str,
    ) -> list[ExtractedEntity]:
        """Return all entities found in *text*."""
        entities: list[ExtractedEntity] = []

        # spaCy pass
        if self._nlp is not None:
            entities.extend(self._extract_spacy(text, doc_id, doc_type))

        # Regex tech-keyword pass (deduplicated later)
        entities.extend(self._extract_tech_keywords(text, doc_id, doc_type))

        # Deduplicate by (text_lower, label)
        seen: set[tuple[str, str]] = set()
        deduped: list[ExtractedEntity] = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        return deduped

    def extract_code_entities(
        self,
        source: str,
        file_path: str,
        doc_id: str,
        doc_type: str,
    ) -> list[ExtractedEntity]:
        """Extract module/class/function names from source using CodeParser."""
        from parsing.code_parser import CodeParser

        parser = CodeParser()
        code_entities = parser.parse(source, file_path=file_path)
        return [
            ExtractedEntity(
                text=ce.name,
                label="MODULE" if ce.entity_type == "class" else "FUNCTION",
                source_doc_id=doc_id,
                doc_type=doc_type,
            )
            for ce in code_entities
            if ce.name and ce.entity_type in ("function", "class", "method")
        ]

    # ------------------------------------------------------------------
    # spaCy helpers
    # ------------------------------------------------------------------

    def _load_spacy(self, model: str):
        try:
            import spacy  # type: ignore

            try:
                return spacy.load(model)
            except OSError:
                # Try smaller fallback
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    return None
        except ImportError:
            return None

    def _extract_spacy(
        self, text: str, doc_id: str, doc_type: str
    ) -> list[ExtractedEntity]:
        doc = self._nlp(text[:100_000])  # spaCy has a per-doc limit
        entities: list[ExtractedEntity] = []
        spacy_label_map = {
            "PERSON": "PERSON",
            "ORG": "ORG",
            "PRODUCT": "TECH",
            "GPE": "PLACE",
            "LOC": "PLACE",
            "EVENT": "CONCEPT",
            "WORK_OF_ART": "CONCEPT",
            "LAW": "CONCEPT",
        }
        for ent in doc.ents:
            mapped = spacy_label_map.get(ent.label_, ent.label_)
            entities.append(
                ExtractedEntity(
                    text=ent.text.strip(),
                    label=mapped,
                    source_doc_id=doc_id,
                    doc_type=doc_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Regex helpers
    # ------------------------------------------------------------------

    def _extract_tech_keywords(
        self, text: str, doc_id: str, doc_type: str
    ) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        for m in _TECH_PATTERNS.finditer(text):
            entities.append(
                ExtractedEntity(
                    text=m.group(0),
                    label="TECH",
                    source_doc_id=doc_id,
                    doc_type=doc_type,
                    start_char=m.start(),
                    end_char=m.end(),
                )
            )
        return entities
