"""Local SentenceTransformer embeddings generation."""

import logging
from typing import List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config.config import RAGConfig

logger = logging.getLogger(__name__)


@st.cache_resource
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Returns a cached SentenceTransformer model instance."""
    logger.info(f"Loading local embedding model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(texts: List[str], config: RAGConfig) -> np.ndarray:
    """Generates embeddings for a list of texts using local SentenceTransformer.

    Args:
        texts: List of text strings to embed.
        config: RAG configuration containing the embedding model name.

    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    model = get_embedding_model(config.embedding_model)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)
