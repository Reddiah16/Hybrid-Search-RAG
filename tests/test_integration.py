import sys
import os 
import numpy as np

# Add workspace directory to python path to resolve absolute module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from config.config import RAGConfig
from database.database import init_db, store_chunks, load_all_chunks, load_all_embeddings
from retrieval.vector_store import store_in_faiss, rebuild_faiss_from_db, faiss_search
from retrieval.bm25_store import build_bm25_index, bm25_search
from retrieval.search import hybrid_search

def test_pipeline():
    print("Starting integration test...")
    # 1. Setup mock session state for Streamlit (as streamlit is imported)
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'bm25_index' not in st.session_state:
        st.session_state.bm25_index = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []

    # 2. Config
    config = RAGConfig(
        db_url="postgresql://postgres:postgresql@localhost:5432/ragdb",
        ollama_host="http://localhost:11434"
    )

    # 3. Init Database
    print("Initializing database...")
    init_db(config)

    # 4. Truncate/Clean Table for clean test
    import psycopg2
    conn = psycopg2.connect(config.db_url)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")
    conn.commit()
    conn.close()

    # 5. Generate and store some chunks
    chunks = [
        "Python is an interpreted high-level general-purpose programming language.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "BM25 is a ranking function used by search engines to estimate the relevance of documents."
    ]
    # We use embedding dimension 384. Generate some distinct dummy vectors.
    embeddings = np.zeros((len(chunks), 384), dtype=np.float32)
    # Give them some distinct values so similarity search behaves predictably
    embeddings[0, 0] = 1.0  # Python vector
    embeddings[1, 1] = 1.0  # FAISS vector
    embeddings[2, 2] = 1.0  # BM25 vector

    print("Storing chunks in DB...")
    store_chunks(chunks, embeddings, "test_doc.pdf", config)

    # 6. Load chunks and embeddings from database
    print("Loading chunks from DB...")
    loaded_chunks = load_all_chunks(config)
    assert len(loaded_chunks) == len(chunks), f"Expected {len(chunks)} chunks, got {len(loaded_chunks)}"
    assert loaded_chunks[0]["content"] == chunks[0]

    print("Loading embeddings from DB...")
    loaded_embs = load_all_embeddings(config)
    assert len(loaded_embs) == len(embeddings), f"Expected {len(embeddings)} embeddings, got {len(loaded_embs)}"
    assert np.allclose(loaded_embs[0], embeddings[0])

    # 7. Rebuild in-memory indices
    print("Rebuilding FAISS from DB...")
    rebuild_faiss_from_db(config)
    assert st.session_state.faiss_index is not None
    assert st.session_state.faiss_index.ntotal == len(chunks)

    print("Building BM25 from DB...")
    build_bm25_index(config)
    assert st.session_state.bm25_index is not None
    assert len(st.session_state.all_chunks) == len(chunks)

    # 8. Test search components directly
    # Vector search: query close to FAISS vector
    query_emb = np.zeros((1, 384), dtype=np.float32)
    query_emb[0, 1] = 1.0  # matches FAISS vector

    # We mock generate_embeddings in vector_store to return query_emb
    from retrieval import vector_store
    old_gen_embeddings = vector_store.generate_embeddings
    vector_store.generate_embeddings = lambda q, cfg: query_emb

    print("Testing FAISS search...")
    faiss_res = faiss_search("FAISS", config, k=3)
    # The best match should be chunk 1 (FAISS)
    assert faiss_res[0][0] == 1, f"Expected best vector match to be chunk index 1, got {faiss_res[0][0]}"

    # BM25 search
    print("Testing BM25 search...")
    bm25_res = bm25_search("programming", config, k=3)
    # Best match should be chunk 0 (Python) which has "programming"
    assert bm25_res[0][0] == 0, f"Expected best BM25 match to be chunk index 0, got {bm25_res[0][0]}"

    # Hybrid Search
    print("Testing Hybrid search...")
    hybrid_res = hybrid_search("FAISS library", config, k=3)
    # Should contain relevant chunks
    print("Hybrid Search Results:")
    for res in hybrid_res:
        print(f" - ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content'][:50]}...")

    print("All integration tests passed successfully!")

    # Restore mock
    vector_store.generate_embeddings = old_gen_embeddings

if __name__ == "__main__":
    test_pipeline()
