# 🔍 Hybrid Search RAG Assistant

> A fully local AI-powered document intelligence system that combines semantic understanding and keyword retrieval to answer questions from PDF documents.

---

## 🌟 Motivation

Traditional search systems often struggle to capture meaning, while vector-only retrieval may miss important keywords. This project bridges both approaches by combining dense retrieval and sparse retrieval into a unified hybrid search pipeline.

The goal is to build a cost-efficient and production-style RAG system without relying on external LLM APIs.

---

## ⚡ What Happens Behind the Scenes?

```text
📄 PDF Upload
      ↓
📚 Text Extraction
      ↓
✂ Chunking
      ↓
🧠 Embedding Generation
      ↓
🗄 PostgreSQL Storage
      ↓
🔍 FAISS Semantic Search
+
🔎 BM25 Keyword Search
      ↓
⚖ Reciprocal Rank Fusion
      ↓
📌 Top Relevant Context
      ↓
🤖 Qwen3 via Ollama
      ↓
💬 Final Response
```

---

## 🚀 Highlights

✨ Fully Local Architecture

✨ No API Keys Required

✨ No Usage Limits

✨ Hybrid Search (FAISS + BM25)

✨ PostgreSQL Persistence

✨ Streamlit Interface

✨ Modular Codebase

✨ Production-Style Pipeline

---

## 🛠 Technology Stack

| Category | Technologies |
|------------|-------------|
| Language | Python |
| Database | PostgreSQL |
| Interface | Streamlit |
| Document Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25 |
| Fusion Strategy | Reciprocal Rank Fusion |
| LLM | Ollama + Qwen3 |
| Core Libraries | PyTorch, Transformers, NumPy, Pandas |

---

## 📂 Repository Layout

```text
Hybrid_search_RAG/
│
├── config/
├── database/
├── processing/
├── retrieval/
├── generation/
├── tests/
├── screenshots/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .env.example
└── .gitignore
```

---

## 🔄 End-to-End Pipeline

### 1️⃣ Document Ingestion

PDF files are uploaded and processed using PyMuPDF.

### 2️⃣ Chunk Creation

Large documents are divided into smaller overlapping chunks.

### 3️⃣ Embedding Generation

Sentence Transformers convert chunks into dense vectors.

### 4️⃣ Storage Layer

Chunks and embeddings are persisted in PostgreSQL.

### 5️⃣ Hybrid Retrieval

Relevant information is retrieved using:

- FAISS (semantic similarity)
- BM25 (keyword matching)

Results are combined using Reciprocal Rank Fusion.

### 6️⃣ Response Generation

Qwen3 running locally through Ollama generates context-aware answers.

---

## 📸 Application Preview
 
### Home Page

```text
screenshots/home.png
```

### Document Upload

```text
screenshots/upload.png
```

### Chat Interface

```text
screenshots/chat.png
```

---

## 📈 Performance Snapshot

| Stage | Average Time |
|---------|-------------:|
| Hybrid Search | ~0.05 sec |
| Retrieval | < 0.1 sec |
| Response Generation | 5–20 sec |
| End-to-End | 10–30 sec |

---

## 🎓 Concepts Explored

This project provided hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Sparse Retrieval
- Hybrid Search
- Vector Embeddings
- FAISS Indexing
- BM25 Ranking
- Local LLMs
- PostgreSQL Integration
- Prompt Engineering
- Performance Optimization
- Modular Software Design

---

## 🔮 Possible Extensions

- Source Citations
- Metadata Filtering
- Query Rewriting
- Agentic RAG
- LangGraph Integration
- Docker Deployment
- FastAPI API Layer
- Multi-modal RAG

---

## 📜 License

Distributed under the MIT License.

---

### ⭐ If this project interests you, feel free to star the repository.