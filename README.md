# 🔍 Hybrid Search RAG Assistant

> A fully local AI-powered document intelligence system that combines semantic understanding and keyword retrieval to answer questions from PDF documents.

---

## 🌟 Motivation

Traditional keyword-based search systems often fail to understand semantic meaning, while vector-only retrieval may overlook important keywords. This project bridges both approaches by combining dense retrieval and sparse retrieval into a unified hybrid search pipeline.

The objective was to build a production-style Retrieval-Augmented Generation (RAG) system using entirely local components without relying on external LLM APIs.

---

## 🎯 Project Goal

The goal of this project was to understand the architecture behind modern RAG systems and implement a complete document question-answering pipeline using:

* Local LLM inference
* Hybrid retrieval
* Persistent storage
* Modular architecture

The system is designed for learning, experimentation, and production-style workflows.

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

| Category            | Technologies                         |
| ------------------- | ------------------------------------ |
| Language            | Python                               |
| Database            | PostgreSQL                           |
| Interface           | Streamlit                            |
| Document Processing | PyMuPDF                              |
| Embeddings          | Sentence Transformers                |
| Dense Retrieval     | FAISS                                |
| Sparse Retrieval    | BM25                                 |
| Fusion Strategy     | Reciprocal Rank Fusion               |
| LLM                 | Ollama + Qwen3                       |
| Core Libraries      | PyTorch, Transformers, NumPy, Pandas |

---

## 🏗 System Architecture

```text
              PDF Documents
                     │
                     ▼
                PyMuPDF
                     │
                     ▼
                Chunking
                     │
                     ▼
      SentenceTransformer Embeddings
                     │
                     ▼
                PostgreSQL
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       FAISS                 BM25
          │                     │
          └──────────┬──────────┘
                     ▼
          Reciprocal Rank Fusion
                     │
                     ▼
              Top Relevant Chunks
                     │
                     ▼
             Ollama (Qwen3)
                     │
                     ▼
               Streamlit UI
```

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

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Hybrid_search_RAG.git

cd Hybrid_search_RAG
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Install Ollama

Download Ollama:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

Pull the model:

```bash
ollama pull qwen3:8b
```

Test:

```bash
ollama run qwen3:8b
```

---

## 🗄 PostgreSQL Setup

Create database:

```sql
CREATE DATABASE ragdb;
```

Example configuration:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/ragdb
OLLAMA_HOST=http://localhost:11434
```

---

## ▶ Running the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🔄 End-to-End Pipeline

### 1️⃣ Document Ingestion

PDF files are uploaded and processed using PyMuPDF.

### 2️⃣ Chunk Creation

Documents are divided into overlapping chunks.

### 3️⃣ Embedding Generation

Sentence Transformers generate dense vector embeddings.

### 4️⃣ Storage Layer

Chunks and embeddings are stored in PostgreSQL.

### 5️⃣ Hybrid Retrieval

Relevant information is retrieved using:

* FAISS (semantic similarity)
* BM25 (keyword matching)

Both results are combined using Reciprocal Rank Fusion.

### 6️⃣ Response Generation

Qwen3 running locally through Ollama generates context-aware responses.

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

### Terminal Logs

```text
screenshots/logs.png
```

### Architecture Diagram

```text
screenshots/architecture.png
```

---

## 📈 Performance Snapshot

| Stage               | Average Time |
| ------------------- | -----------: |
| Hybrid Search       |    ~0.05 sec |
| Retrieval           |    < 0.1 sec |
| Response Generation |     5–20 sec |
| End-to-End          |    10–30 sec |

---

## 💡 Key Learnings

This project provided hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Dense Retrieval
* Sparse Retrieval
* Hybrid Search
* Vector Embeddings
* FAISS Indexing
* BM25 Ranking
* Reciprocal Rank Fusion (RRF)
* Local LLMs
* PostgreSQL Integration
* Prompt Engineering
* Performance Optimization
* Modular Software Design

---

## 🔮 Future Improvements

* Source citations
* Metadata filtering
* Query rewriting
* Agentic RAG
* LangGraph integration
* FastAPI REST API
* Docker deployment
* Multi-modal RAG

---

## 📜 License

Distributed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving the repository a star.

---

Built with Python, PostgreSQL, FAISS, BM25, Ollama, and Qwen3.
