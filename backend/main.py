"""RAG Playground Backend — FastAPI application."""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api import documents, pipeline, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — loads the embedding model on startup."""
    print("🚀 RAG Playground Backend starting...")
    print(f"   Embedding model: {config.EMBEDDING_MODEL}")
    print(f"   Storage: {config.STORAGE_DIR}")

    # Use API for embedding model (no local preload)
    print("✅ Using API for embeddings.")

    yield

    print("👋 RAG Playground Backend shutting down.")


app = FastAPI(
    title="RAG Playground",
    description="RAG Visualization & Experimentation Platform API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(documents.router)
app.include_router(pipeline.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {
        "name": "RAG Playground API",
        "version": "1.0.0",
        "status": "ready",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
