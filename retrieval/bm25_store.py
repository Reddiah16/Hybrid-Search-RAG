"""BM25 keyword search index."""

import logging
from typing import List, Tuple

import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi

from config.config import RAGConfig
from database.database import load_all_chunks

logger = logging.getLogger(__name__)


def build_bm25_index(config: RAGConfig):
    """Builds/rebuilds the BM25 index from all chunks stored in PostgreSQL.

    Loads all chunk content from the database, tokenizes them, and creates
    a BM25Okapi index. Also stores the chunk metadata for retrieval.

    Args:
        config: RAG configuration.
    """
    chunks_data = load_all_chunks(config)

    if not chunks_data:
        st.session_state.bm25_index = None
        st.session_state.all_chunks = []
        return

    # Tokenize for BM25
    tokenized_corpus = [chunk["content"].lower().split() for chunk in chunks_data]
    bm25 = BM25Okapi(tokenized_corpus)

    st.session_state.bm25_index = bm25
    st.session_state.all_chunks = chunks_data
    logger.info(f"Built BM25 index with {len(chunks_data)} chunks")


def bm25_search(query: str, config: RAGConfig, k: int = 20) -> List[Tuple[int, float]]:
    """Performs keyword search using BM25.

    Args:
        query: The search query.
        config: RAG configuration.
        k: Number of top results to return.

    Returns:
        List of (chunk_index, score) tuples, sorted by descending score.
    """
    if 'bm25_index' not in st.session_state or st.session_state.bm25_index is None:
        return []

    bm25 = st.session_state.bm25_index
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices sorted by score
    top_indices = np.argsort(scores)[::-1][:k]
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    return results
