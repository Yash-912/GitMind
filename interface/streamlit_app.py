from __future__ import annotations

import streamlit as st

from config.settings import settings
from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex
from retrieval import (
    ChunkStore,
    HybridRetriever,
    GraphExpander,
    CrossEncoderReranker,
    ContextAssembler,
    QueryDecomposer,
    EntityResolver,
)
from generation import (
    DirectQAGenerator,
    DecisionMemoGenerator,
    BlameMapGenerator,
    RiskReportGenerator,
)


st.set_page_config(page_title="GitMind", page_icon="🧠", layout="wide")
st.title("GitMind — Codebase Archaeology")

query = st.text_input("Ask a question about the code history")
mode = st.selectbox("Mode", ["direct", "memo", "blame", "risk"], index=0)

if st.button("Answer") and query:
    try:
        chunk_store = None
        qdrant = None
        bm25 = None
        fts = None
        decomposer = None
        resolver = None
        retriever = None
        expander = None
        generator = None
        try:
            chunk_store = ChunkStore(str(st.session_state.get("chunks_path", settings.data_dir + "/chunks.jsonl")))
            qdrant = QdrantStore(path=settings.qdrant_path)
            bm25 = BM25Index(index_dir=settings.bm25_index_dir)
            fts = FTSIndex(db_path=settings.db_path)

            decomposer = QueryDecomposer()
            plan = decomposer.decompose(query)
            resolver = EntityResolver(settings.db_path)
            module_tags = resolver.resolve(plan.entities)

            retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, fts=fts, chunk_store=chunk_store)
            candidates = retriever.retrieve(
                query,
                limit=40,
                filters={
                    "time_start": plan.time_start,
                    "time_end": plan.time_end,
                    "module_tags": module_tags,
                },
            )

            expander = GraphExpander(db_path=settings.db_path, chunk_store=chunk_store, hop_depth=1)
            expanded = expander.expand(candidates)

            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(query, expanded, top_k=12)

            assembler = ContextAssembler(max_tokens=3000)
            context = assembler.assemble(reranked)

            if mode == "direct":
                generator = DirectQAGenerator()
                result = generator.generate(query, context)
                answer = result.answer
                model = result.model
            elif mode == "memo":
                generator = DecisionMemoGenerator()
                result = generator.generate(query, context)
                answer = result.raw_text
                model = result.model
            elif mode == "blame":
                generator = BlameMapGenerator()
                result = generator.generate(query, context)
                answer = result.raw_text
                model = result.model
            else:
                generator = RiskReportGenerator()
                result = generator.generate(query, context)
                answer = result.raw_text
                model = result.model

            st.subheader("Answer")
            st.caption(f"Model: {model}")
            st.write(answer)

            st.subheader("Evidence")
            st.dataframe(
                [
                    {
                        "doc_type": c.metadata.get("doc_type", ""),
                        "doc_id": c.metadata.get("doc_id", ""),
                        "score": c.score,
                        "source": c.source,
                    }
                    for c in reranked
                ]
            )

            with st.expander("Context"):
                st.text(context)
        except Exception as exc:
            st.error(f"Failed to generate answer: {exc}")
        finally:
            if expander is not None:
                expander.close()
            if retriever is not None:
                retriever.close()
            if decomposer is not None:
                decomposer.close()
            if resolver is not None:
                resolver.close()
            if qdrant is not None:
                qdrant.close()
            if fts is not None:
                fts.close()
            if generator is not None:
                generator.close()
