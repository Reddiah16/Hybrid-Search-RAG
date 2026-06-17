"""FAISS vector store operations."""

import logging
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st

from config.config import RAGConfig
from database.database import load_all_embeddings
from processing.embeddings import generate_embeddings

logger = logging.getLogger(__name__)


def store_in_faiss(embeddings: np.ndarray, config: RAGConfig) -> faiss.IndexFlatIP:
    """Adds embeddings to the FAISS index (creates or appends).

    Uses Inner Product (IP) similarity since embeddings are normalized.

    Args:
        embeddings: NumPy array of embeddings to add.
        config: RAG configuration.

    Returns:
        The updated FAISS index.
    """
    # Normalize embeddings for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    if 'faiss_index' not in st.session_state or st.session_state.faiss_index is None:
        index = faiss.IndexFlatIP(config.embedding_dim)
        st.session_state.faiss_index = index
    else:
        index = st.session_state.faiss_index

    index.add(embeddings)
    logger.info(f"FAISS index now contains {index.ntotal} vectors")
    return index


def rebuild_faiss_from_db(config: RAGConfig):
    """Rebuilds the FAISS index from embeddings stored in PostgreSQL.

    Called on startup to restore the FAISS index from persisted data.

    Args:
        config: RAG configuration.
    """
    embeddings = load_all_embeddings(config)

    if not embeddings:
        st.session_state.faiss_index = None
        return

    embeddings_array = np.vstack(embeddings)
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(config.embedding_dim)
    index.add(embeddings_array)
    st.session_state.faiss_index = index
    logger.info(f"Rebuilt FAISS index with {index.ntotal} vectors from DB")


def faiss_search(query: str, config: RAGConfig, k: int = 20) -> List[Tuple[int, float]]:
    """Performs vector similarity search using FAISS.

    Args:
        query: The search query.
        config: RAG configuration.
        k: Number of top results to return.

    Returns:
        List of (chunk_index, score) tuples, sorted by descending score.
    """
    if 'faiss_index' not in st.session_state or st.session_state.faiss_index is None:
        return []

    index = st.session_state.faiss_index
    if index.ntotal == 0:
        return []

    # Embed the query
    query_embedding = generate_embeddings([query], config)
    faiss.normalize_L2(query_embedding)

    # Search
    actual_k = min(k, index.ntotal)
    scores, indices = index.search(query_embedding, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:  # FAISS returns -1 for invalid results
            results.append((int(idx), float(score)))

    return results
