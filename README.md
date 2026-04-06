# 🧪 RAG Playground — Visualization & Experimentation Platform

An interactive platform for building, configuring, and **visualizing** RAG (Retrieval-Augmented Generation) pipelines in 3D.

![Phase](https://img.shields.io/badge/Phase-1%20MVP-6366f1)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-Angular%2019-dd0031)
![3D](https://img.shields.io/badge/3D-Three.js-000000)

## ✨ Features (Phase 1)

- **Document Ingestion** — Upload PDF/TXT files with configurable chunking (fixed or overlapping)
- **Vector Retrieval** — FAISS-based search with Flat (exact) and HNSW (approximate) indices
- **3D Embedding Visualization** — Interactive Three.js scene showing document vectors, query points, and nearest neighbor connections
- **Pipeline Configuration** — Adjustable parameters (index type, top_k, HNSW M/efSearch, temperature)
- **Stage Metrics** — Per-stage latency breakdown (embedding → retrieval → generation → visualization)
- **Dark Theme UI** — Premium 3-panel layout with glassmorphism design

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────────────────────┐    ┌────────────┐
│  Angular UI │◄──►│  FastAPI Backend (port 8000)      │    │   FAISS    │
│  (port 4200)│    │                                    │◄──►│  Indices   │
│             │    │  ┌──────────┐ ┌───────────────┐   │    └────────────┘
│ Three.js 3D │    │  │Ingestion │ │SentenceTransf.│   │
│ Document Up.│    │  │Service   │ │  Embeddings   │   │
│ Pipeline Cfg│    │  └──────────┘ └───────────────┘   │
│ Query Panel │    │  ┌──────────┐ ┌───────────────┐   │
│ Metrics     │    │  │Retrieval │ │  Generation   │   │
│             │    │  │(Flat/HNSW│ │  (Template)   │   │
└─────────────┘    │  └──────────┘ └───────────────┘   │
                   └──────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload --port 8000
```

> ℹ️ First startup downloads the embedding model (~80MB). API docs available at http://localhost:8000/docs

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies (already done if scaffolded)
npm install

# Start dev server
npx ng serve --open
```

Opens at http://localhost:4200

### 3. Try It Out

1. **Upload** `backend/data/sample.txt` via the Documents panel
2. **Configure** the pipeline (try Flat vs HNSW)
3. **Query**: "What is RAG?" or "How does HNSW work?"
4. **Explore** the 3D visualization — orbit, zoom, hover over points

## 📁 Project Structure

```
RAG Playground/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py             # Configuration
│   ├── models/               # Pydantic schemas
│   ├── services/             # Core business logic
│   │   ├── ingestion.py      #   Document parsing + chunking
│   │   ├── embedding.py      #   SentenceTransformer wrapper
│   │   ├── retrieval.py      #   FAISS (Flat + HNSW)
│   │   ├── generation.py     #   Template-based generation
│   │   └── pipeline.py       #   Pipeline orchestrator
│   ├── api/                  # REST route handlers
│   └── data/                 # Sample dataset
└── frontend/                 # Angular 19 app
    └── src/app/
        ├── components/
        │   ├── document-upload/
        │   ├── pipeline-config/
        │   ├── query-interface/
        │   └── visualization/   # Three.js 3D scene
        ├── services/
        └── models/
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload and process document |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/{id}` | Delete a document |
| GET | `/api/pipeline/config` | Get pipeline configuration |
| POST | `/api/pipeline/configure` | Update pipeline config |
| GET | `/api/pipeline/types` | List available index types |
| POST | `/api/query` | Execute RAG query |

## 🗺️ Roadmap

- **Phase 2**: IVF/PQ indices, cross-encoder re-ranking, enhanced visualizations
- **Phase 3**: Hybrid (BM25 + vector), agentic RAG, graph retrieval
- **Phase 4**: Experiment comparison dashboard, evaluation metrics (Recall@K, MRR)

## 📄 License

MIT
