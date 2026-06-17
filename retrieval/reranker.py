"""Local CrossEncoder reranking for search results."""

import logging
from typing import Dict, List

import streamlit as st
from sentence_transformers import CrossEncoder

from config.config import RAGConfig

logger = logging.getLogger(__name__)


@st.cache_resource
def get_rerank_model(model_name: str) -> CrossEncoder:
    """Returns a cached CrossEncoder model instance."""
    logger.info(f"Loading local rerank model: {model_name}")
    return CrossEncoder(model_name)


def cross_encoder_rerank(query: str, chunks: List[Dict], config: RAGConfig, top_n: int = 5) -> List[Dict]:
    """Reranks chunks using local CrossEncoder.

    Args:
        query: The search query.
        chunks: List of chunk dicts from hybrid search.
        config: RAG configuration.
        top_n: Number of top results to return after reranking.

    Returns:
        List of chunk dicts with updated 'score' from CrossEncoder, sorted by relevance.
    """
    if not chunks:
        return []

    try:
        model = get_rerank_model(config.rerank_model)
        # Limit chunks to avoid scoring too many pairs
        chunks = chunks[:10]
        # Pair format: [query, doc]
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = model.predict(pairs)

        reranked = []
        for idx, score in enumerate(scores):
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            reranked.append(chunk)

        # Sort by score in descending order
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        logger.info(f"CrossEncoder reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked[:top_n]

    except Exception as e:
        logger.error(f"CrossEncoder rerank error: {str(e)}")
        # Fallback: return original chunks unchanged
        return chunks[:top_n]
