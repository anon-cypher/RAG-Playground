# RAG Playground — Visualization & Experimentation

Build and **visualize** RAG (Retrieval-Augmented Generation) pipelines: chunking, **OpenRouter** embeddings, FAISS retrieval, prompt assembly, and LLM answers—with a **node-based canvas** (v2) and optional **3D embedding** projection.

![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-646cff)
![Models](https://img.shields.io/badge/Models-OpenRouter-5a67d8)

## Features

- **OpenRouter** for embeddings and chat (`https://openrouter.ai/api/v1`) — bring your own API key in the UI (`frontend_v2.0`).
- **Document upload** (PDF/TXT) with configurable chunking and preview.
- **FAISS** retrieval (flat / HNSW) with `top_k` and optional document filters.
- **Per-node execution**, full-graph runs, retrieval preview, and **PCA 3D** embedding view.
- **Export** last run as JSON (API keys redacted).

## Architecture

- **Backend** (`backend/`): FastAPI, ingestion, embedding via OpenRouter, FAISS, pipeline orchestration.
- **Frontend v2** (`frontend_v2.0/`): React, XYFlow canvas, Three.js 3D view, settings bar for API key + localStorage persistence.

## OpenRouter setup

1. Create an account at [OpenRouter](https://openrouter.ai/) and create an API key.
2. Start the backend and open the v2 UI (see below).
3. In **Settings** (top of the app), paste the key → **Save key** (persists in the browser and syncs to the server session without rebuilding the index). Use **Test key** to verify.
4. Choose **embedding** and **LLM** models in the **Embedder** / **LLM** nodes (defaults align with common OpenRouter IDs, e.g. `openai/text-embedding-3-small`, `openai/gpt-4o-mini`).
5. **Embedding dimension** must match the model (e.g. 1536 for `text-embedding-3-small`). If you change model/dim, rebuild the index; the API returns a clear error if dimensions mismatch.

**Cost & latency**: Every embedding and chat call is billed by OpenRouter. The **visualization-only** query mode skips LLM generation for cheaper 3D experiments.

**Security**: The API key is stored in `localStorage` for convenience and held in server memory for the session. For shared machines, prefer a private browser profile or a future backend-only secret. Exported JSON never includes the raw key.

## Quick start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### Frontend v2

```bash
cd frontend_v2.0
npm install
npm run dev
```

Open the printed URL (typically `http://localhost:5173`). Ensure the client can reach `http://localhost:8000` (CORS is configured for common dev ports).

### Try it

1. Add your OpenRouter key in **Settings**.
2. Use **Document Loader** to upload a `.txt` / `.pdf`.
3. Set **Embedder** model/dim, then **Build embeddings & index**.
4. Enter a query on the **Query** node, **Run pipeline** or use **Query neighbors (PCA)** in the 3D section.

## Project layout

```
RAG Playground/
├── backend/           # FastAPI app, services, models
├── frontend_v2.0/     # React + Vite + XYFlow + R3F
├── plan.md            # Phased product / implementation checklist
└── README.md
```

## API highlights

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/credentials` | Set API key without index rebuild |
| POST | `/api/pipeline/verify-key` | Validate OpenRouter key |
| POST | `/api/pipeline/configure` | Full pipeline config (+ optional index rebuild) |
| POST | `/api/pipeline/execute-graph` | Run visual graph |
| POST | `/api/query` | Full RAG query (+ optional `visualization_only`) |
| GET | `/api/pipeline/embedding-space` | Corpus PCA/UMAP projection |

## Roadmap

See [`plan.md`](plan.md) for phased checklist (glossary, export, polish, optional compare/streaming).

## License

MIT
