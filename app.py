from pathlib import Path
from typing import Any
import streamlit as st

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    PDF_PATH,
    RRF_K,
)
from src.extractor import chunk_pages, extract_pages
from src.generator import GroqRAGGenerator
from src.retriever import HybridSOPRetriever

st.set_page_config(
    page_title="LeadGen Hybrid RAG Assistant",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource
def get_retriever() -> HybridSOPRetriever:
    """Cache the ChromaDB + SentenceTransformer + BM25 retriever across reruns."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return HybridSOPRetriever(
        persist_dir=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )


def index_pdf(pdf_file: Path, retriever: HybridSOPRetriever) -> tuple[int, int]:
    """Extract, chunk, and index a PDF file."""
    pages = extract_pages(pdf_file)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    retriever.index_chunks(chunks)
    return len(pages), len(chunks)


def run_search(
    retriever: HybridSOPRetriever,
    query: str,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Execute semantic, BM25, or hybrid RRF search."""
    if mode == "semantic":
        return retriever.semantic_search(query, top_k=top_k)
    if mode == "bm25":
        return retriever.bm25_search(query, top_k=top_k)
    return retriever.hybrid_search(query, top_k=top_k, rrf_k=RRF_K)


def main() -> None:
    st.title("🎯 Lead Generation & Outreach Hybrid RAG")
    st.caption(f"ChromaDB Semantic + BM25 Lexical Search + Groq (`{GROQ_MODEL_NAME}`)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    retriever = get_retriever()

    with st.sidebar:
        st.header("Streamlit & RAG Settings")
        ui_api_key = st.text_input(
            "Groq API Key (optional if in Secrets / .env)",
            type="password",
        )
        mode = st.selectbox("Search Mode", ["hybrid", "semantic", "bm25"], index=0)
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=4)

        st.divider()
        st.subheader("SOP + CV Index")
        st.write(f"Indexed chunks: **{retriever.doc_count()}**")

        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_pdf is not None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = DATA_DIR / uploaded_pdf.name
            saved_path.write_bytes(uploaded_pdf.getvalue())
            if st.button("Index Uploaded PDF", use_container_width=True):
                try:
                    p_cnt, c_cnt = index_pdf(saved_path, retriever)
                    st.success(f"Indexed {c_cnt} chunks from {p_cnt} pages.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        if PDF_PATH.exists():
            if st.button("Index Default PDF", use_container_width=True):
                try:
                    p_cnt, c_cnt = index_pdf(PDF_PATH, retriever)
                    st.success(f"Indexed {c_cnt} chunks from {p_cnt} pages.")
                except Exception as exc:
                    st.error(f"Error: {exc}")
        elif retriever.doc_count() == 0:
            st.info(f"Upload your PDF or place it at `{PDF_PATH}`.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Retrieved Context ({len(message['sources'])} chunks)"):
                    for idx, src in enumerate(message["sources"], start=1):
                        page = src.get("metadata", {}).get("page", "?")
                        score_text = (
                            f"rrf={src['rrf_score']}"
                            if "rrf_score" in src
                            else f"score={src.get('score')}"
                        )
                        st.markdown(
                            f"**[{idx}] Page {page}** (`{src['id']}`, `{score_text}`)

"
                            f"{src['text']}"
                        )

    prompt = st.chat_input("Ask about SOP steps, audit tools, outreach templates, or CV...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if retriever.doc_count() == 0:
                msg = "No chunks indexed yet. Upload or index your PDF in the sidebar first."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

            try:
                generator = GroqRAGGenerator(
                    model_name=GROQ_MODEL_NAME,
                    ui_api_key=ui_api_key or None,
                    secrets_obj=st.secrets,
                )
                with st.spinner(f"Running {mode} search & generating answer..."):
                    hits = run_search(retriever, prompt, mode=mode, top_k=top_k)
                    answer = generator.generate(prompt, hits)

                st.markdown(answer)
                with st.expander(f"Retrieved Context ({len(hits)} chunks)"):
                    for idx, src in enumerate(hits, start=1):
                        page = src.get("metadata", {}).get("page", "?")
                        score_text = (
                            f"rrf={src['rrf_score']}"
                            if "rrf_score" in src
                            else f"score={src.get('score')}"
                        )
                        st.markdown(
                            f"**[{idx}] Page {page}** (`{src['id']}`, `{score_text}`)

"
                            f"{src['text']}"
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": hits}
                )
            except Exception as exc:
                err = f"Error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
