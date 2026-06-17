"""Streamlit UI for the Hybrid Search RAG Assistant."""

import os
import logging
import tempfile
import warnings

import streamlit as st

from config.config import RAGConfig
from database.database import init_db, store_chunks
from processing.document_processor import extract_pdf_text, create_chunks
from processing.embeddings import generate_embeddings
from retrieval.vector_store import store_in_faiss, rebuild_faiss_from_db
from retrieval.bm25_store import build_bm25_index
from retrieval.search import perform_search
from generation.llm import ollama_generate
from generation.fallback import handle_fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")


# ---------------------------------------------------------------------------
# Orchestration: Document Processing
# ---------------------------------------------------------------------------

def process_document(file_path: str, doc_name: str, config: RAGConfig) -> bool:
    """Full document processing pipeline: extract → chunk → embed → index.

    Args:
        file_path: Path to the PDF file.
        doc_name: Original filename of the document.
        config: RAG configuration.

    Returns:
        True if processing succeeded, False otherwise.
    """
    try:
        # 1. Extract text from PDF
        logger.info(f"Extracting text from: {doc_name}")
        text = extract_pdf_text(file_path)
        if not text.strip():
            logger.warning(f"No text extracted from {doc_name}")
            return False

        # 2. Create chunks
        chunks = create_chunks(text, max_size=config.chunk_max_size, overlap=config.chunk_overlap)
        if not chunks:
            logger.warning(f"No chunks created from {doc_name}")
            return False

        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = generate_embeddings(chunks, config)

        # 4. Store in PostgreSQL first (dedup via content_hash)
        store_chunks(chunks, embeddings, doc_name, config)

        # 5. Rebuild FAISS from DB (mirrors only deduplicated rows)
        rebuild_faiss_from_db(config)

        # 6. Rebuild BM25 index (includes all documents)
        build_bm25_index(config)

        logger.info(f"Successfully processed: {doc_name}")
        return True

    except Exception as e:
        logger.error(f"Error processing document '{doc_name}': {str(e)}")
        return False


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Hybrid Search RAG Assistant", layout="wide")

    for state_var in [
        'chat_history',
        'documents_loaded',
        'my_config',
        'user_env',
        'processed_docs',
        'faiss_index',
        'bm25_index',
        'all_chunks'
    ]:
        if state_var not in st.session_state:
            if state_var == 'chat_history':
                st.session_state[state_var] = []
            elif state_var == 'documents_loaded':
                st.session_state[state_var] = False
            elif state_var == 'processed_docs':
                st.session_state[state_var] = set()
            elif state_var == 'all_chunks':
                st.session_state[state_var] = []
            elif state_var in ('my_config', 'faiss_index', 'bm25_index'):
                st.session_state[state_var] = None
            else:
                st.session_state[state_var] = {}

    with st.sidebar:
        st.title("Configuration")
        ollama_host = st.text_input("Ollama Host", value=st.session_state.get('ollama_host', 'http://localhost:11434'))
        llm_model = st.selectbox("Model", ["qwen3:8b", "llama3.1:8b", "gemma3:4b"], index=0)
        db_url = st.text_input("Database URL", value=st.session_state.get('db_url', 'postgresql://postgres:postgresql@localhost:5432/ragdb'), placeholder="postgresql://postgres:postgresql@localhost:5432/ragdb")

        if st.button("Save Configuration"):
            try:
                if not all([ollama_host, db_url]):
                    st.error("Ollama Host and Database URL are required!")
                    return

                for key, value in {'ollama_host': ollama_host, 'llm_model': llm_model, 'db_url': db_url}.items():
                    st.session_state[key] = value

                config = RAGConfig(
                    db_url=db_url,
                    ollama_host=ollama_host,
                    llm_model=llm_model
                )

                # Initialize PostgreSQL schema
                init_db(config)

                # Rebuild in-memory indexes from existing DB data
                build_bm25_index(config)

                # Rebuild FAISS index from existing embeddings in DB
                rebuild_faiss_from_db(config)

                st.session_state.my_config = config
                st.session_state.user_env = {
                    "OLLAMA_HOST": ollama_host
                }
                st.success("Configuration saved successfully!")

                # If there are already chunks in the DB, mark documents as loaded
                if st.session_state.all_chunks:
                    st.session_state.documents_loaded = True

            except Exception as e:
                st.error(f"Configuration error: {str(e)}")

    st.title("📚 Hybrid Search RAG Assistant")

    if st.session_state.my_config:
        uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

        if uploaded_files:
            success = False
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if uploaded_file.name not in st.session_state.processed_docs:

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_path = tmp.name

                        try:
                            if process_document(temp_path, uploaded_file.name, st.session_state.my_config):
                                st.success(f"Successfully processed: {uploaded_file.name}")
                                st.session_state.processed_docs.add(uploaded_file.name)
                                success = True
                            else:
                                st.error(f"Failed to process: {uploaded_file.name}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    else:
                        st.info(f"{uploaded_file.name} already exists.")

            if success:
                st.session_state.documents_loaded = True
                st.success("Documents are ready! You can now ask questions about them.")

    if st.session_state.documents_loaded:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"): st.write(msg[0])
            with st.chat_message("assistant"): st.write(msg[1])

        user_input = st.chat_input("Ask a question about the documents...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    reranked_chunks = perform_search(user_input, st.session_state.my_config)

                    if not reranked_chunks:
                        logger.info("No relevant documents found.")
                        st.info("No relevant documents found. Using general knowledge.")
                        full_response = handle_fallback(user_input, st.session_state.my_config)
                        message_placeholder.markdown(full_response)

                    else:

                        top_score = float(reranked_chunks[0].get("score", 0))

                        if top_score < 0.3:
                            logger.info("Low confidence retrieval.")
                            st.info("Low confidence retrieval. Using general knowledge.")
                            full_response = handle_fallback(user_input, st.session_state.my_config)
                            message_placeholder.markdown(full_response)

                        else:
                            full_response = ""
                            for chunk in ollama_generate(
                                query=user_input,
                                context_chunks=reranked_chunks,
                                chat_history=st.session_state.chat_history,
                                config=st.session_state.my_config
                            ):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append((user_input, full_response))
                except Exception as e:
                    logger.exception("Application error")
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Please configure your API keys and upload documents to get started." if not st.session_state.my_config else "Please upload some documents to get started.")


if __name__ == "__main__":
    main()