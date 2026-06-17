"""Configuration for the Hybrid Search RAG pipeline."""

from dataclasses import dataclass


RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip() 


@dataclass
class RAGConfig:
    """Configuration for the custom RAG pipeline (replaces RAGLiteConfig)."""
    db_url: str
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    llm_model: str = "qwen3:4b"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_max_size: int = 800
    chunk_overlap: int = 100
    