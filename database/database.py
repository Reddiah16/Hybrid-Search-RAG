"""PostgreSQL database functions for chunk storage and retrieval."""

import hashlib
import logging 

import numpy as np
import psycopg2
import psycopg2.extras

from config.config import RAGConfig

logger = logging.getLogger(__name__)


def get_db_connection(config: RAGConfig):
    """Returns a psycopg2 connection to PostgreSQL."""
    return psycopg2.connect(config.db_url)


def init_db(config: RAGConfig):
    """Creates the chunks table if it doesn't exist."""
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks(doc_name);
            """)
        conn.commit()
    finally:
        conn.close()


def store_chunks(chunks, embeddings: np.ndarray, doc_name: str, config: RAGConfig):
    """Stores chunk text and embeddings in PostgreSQL.

    Args:
        chunks: List of chunk text strings.
        embeddings: NumPy array of embeddings corresponding to chunks.
        doc_name: Name of the source document.
        config: RAG configuration.
    """
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                content_hash = hashlib.sha256(chunk.encode()).hexdigest()
                embedding_bytes = embedding.tobytes()
                cur.execute(
                    """INSERT INTO chunks (doc_name, chunk_index, content, content_hash, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING""",
                    (doc_name, i, chunk, content_hash, psycopg2.Binary(embedding_bytes))
                )
        conn.commit()
        logger.info(f"Stored {len(chunks)} chunks for '{doc_name}' in PostgreSQL")
    finally:
        conn.close()


def load_all_chunks(config: RAGConfig):
    """Loads all chunk metadata from the database.

    Returns:
        List of dicts with 'id', 'doc_name', 'chunk_index', 'content'.
    """
    conn = get_db_connection(config)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, doc_name, chunk_index, content FROM chunks ORDER BY id")
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": row["id"], "content": row["content"], "doc_name": row["doc_name"], "chunk_index": row["chunk_index"]}
        for row in rows
    ]


def load_all_embeddings(config: RAGConfig):
    """Loads all embeddings from the database.

    Returns:
        List of numpy arrays, one per chunk (in id order).
    """
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM chunks ORDER BY id")
            rows = cur.fetchall()
    finally:
        conn.close()

    embeddings = []
    for row in rows:
        if row[0] is not None:
            emb = np.frombuffer(bytes(row[0]), dtype=np.float32)
            if len(emb) == config.embedding_dim:
                embeddings.append(emb)

    return embeddings
