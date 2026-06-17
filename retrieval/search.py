"""Hybrid search combining FAISS and BM25 with Reciprocal Rank Fusion."""

import logging
import time
from typing import Dict, List

import streamlit as st

from config.config import RAGConfig
from retrieval.vector_store import faiss_search
from retrieval.bm25_store import bm25_search

logger = logging.getLogger(__name__)


def hybrid_search(query: str, config: RAGConfig, k: int = 5) -> List[Dict]:
    """Combines FAISS and BM25 search results using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1 / (rrf_k + rank(d))
    where rrf_k = 60 (standard constant).

    Args:
        query: The search query.
        config: RAG configuration.
        k: Number of results to return.

    Returns:
        List of chunk dicts with 'id', 'content', 'doc_name', 'score'.
    """
    all_chunks = st.session_state.get('all_chunks', [])
    if not all_chunks:
        return []

    # Get results from both search methods
    faiss_results = faiss_search(query, config, k=k)
    bm25_results = bm25_search(query, config, k=k)

    # Reciprocal Rank Fusion
    rrf_k = 60
    rrf_scores = {}

    for rank, (idx, _score) in enumerate(faiss_results):
        if idx < len(all_chunks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    for rank, (idx, _score) in enumerate(bm25_results):
        if idx < len(all_chunks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    # Sort by RRF score
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

    results = []
    for idx in sorted_indices:
        chunk = all_chunks[idx]
        results.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "doc_name": chunk["doc_name"],
            "score": rrf_scores[idx]
        })

    logger.info(f"Hybrid search returned {len(results)} results (FAISS: {len(faiss_results)}, BM25: {len(bm25_results)})")
    return results


def perform_search(query: str, config: RAGConfig) -> List[Dict]:
    """Full retrieval pipeline: hybrid search (FAISS + BM25 with RRF) → Top 5.

    Args:
        query: The search query.
        config: RAG configuration.

    Returns:
        List of top chunk dicts ranked by RRF score.
    """
    try:
        start = time.time()
        search_results = hybrid_search(query, config, k=5)
        print(f"Hybrid search time: {time.time()-start:.2f}s")
        return search_results[:5]

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []
