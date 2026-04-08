import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from models.pipeline import (
    PipelineConfig,
    PipelineTypesResponse,
    PipelineGraph,
    GraphExecutionResult,
    ExecuteNodeRequest,
    ExecuteNodeResponse,
    PreviewRetrievalRequest,
    PreviewRetrievalResponse,
    VerifyOpenRouterKeyRequest,
    CredentialsUpdate,
)
from models.query import VisualizationData
from services import pipeline as pipeline_service

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("/config", response_model=PipelineConfig)
async def get_pipeline_config():
    """Get current pipeline configuration."""
    return pipeline_service.get_config()


@router.post("/credentials", response_model=PipelineConfig)
async def set_credentials(body: CredentialsUpdate):
    """
    Update OpenRouter API key (and optionally other secrets) in memory only — does **not** rebuild the index.
    Use this for UI key persistence; run Build index / pipeline configure when models change.
    """
    cfg = pipeline_service.get_config()
    data = cfg.model_dump()
    if body.openrouter_api_key is not None:
        data["openrouter_api_key"] = body.openrouter_api_key or None
    updated = PipelineConfig(**data)
    return pipeline_service.set_config(updated)


@router.post("/verify-key")
async def verify_openrouter_key(body: VerifyOpenRouterKeyRequest):
    """Check that an OpenRouter API key is accepted (does not store the key)."""
    if not (body.api_key or "").strip():
        raise HTTPException(400, "API key is required.")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {body.api_key.strip()}"},
            timeout=20.0,
        )
    if resp.status_code == 401 or resp.status_code == 403:
        raise HTTPException(401, "Invalid or unauthorized OpenRouter API key.")
    if resp.status_code == 429:
        raise HTTPException(429, "OpenRouter rate limit — try again shortly.")
    if resp.status_code != 200:
        raise HTTPException(502, "Could not verify key with OpenRouter.")
    return {"ok": True}


@router.post("/configure", response_model=PipelineConfig)
async def configure_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    """Update pipeline configuration and rebuild index in the background."""
    if config.openrouter_api_key:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {config.openrouter_api_key}"}
            )
            if resp.status_code != 200:
                raise HTTPException(400, "Invalid OpenRouter API Key or unable to verify.")

    cfg = pipeline_service.set_config(config)
    background_tasks.add_task(pipeline_service.build_index)
    return cfg


@router.get("/types", response_model=PipelineTypesResponse)
async def get_pipeline_types():
    """Get available pipeline types and retrieval strategies."""
    return PipelineTypesResponse(
        index_types=[
            {
                "value": "flat",
                "label": "Flat (Exact Search)",
                "description": "Exact L2 distance search. Best accuracy, slower for large datasets.",
                "params": []
            },
            {
                "value": "hnsw",
                "label": "HNSW (Approximate)",
                "description": "Graph-based approximate nearest neighbor search. Fast with high recall.",
                "params": [
                    {"name": "hnsw_m", "label": "M (connections)", "default": 32, "min": 4, "max": 128},
                    {"name": "hnsw_ef_construction", "label": "efConstruction", "default": 200, "min": 50, "max": 1000},
                    {"name": "hnsw_ef_search", "label": "efSearch", "default": 64, "min": 10, "max": 500},
                ]
            }
        ],
        retrieval_strategies=[
            {
                "value": "vector",
                "label": "Vector Retrieval",
                "description": "Semantic similarity search using vector embeddings."
            },
            {
                "value": "graph",
                "label": "Knowledge Graph",
                "description": "Bipartite Chunk-Keyword NetworkX subgraph."
            }
        ]
    )


@router.post("/build-index")
async def build_index():
    """Manually rebuild the FAISS index."""
    result = pipeline_service.build_index()
    return result


@router.post("/execute-graph", response_model=GraphExecutionResult)
async def execute_graph(graph: PipelineGraph):
    """
    Execute a visual pipeline graph (Phase 1).

    The backend currently performs pure DAG validation + ordering and
    returns per-node timing metadata so the frontend can animate execution.
    Later phases can plug actual RAG behavior into this hook.
    """
    return pipeline_service.execute_graph(graph)


@router.post("/execute-node", response_model=ExecuteNodeResponse)
async def execute_node(req: ExecuteNodeRequest):
    """Execute a single node in isolation (optional inputs for experiments)."""
    return pipeline_service.execute_node(req)


@router.post("/preview-retrieval", response_model=PreviewRetrievalResponse)
async def preview_retrieval(req: PreviewRetrievalRequest):
    """Debounced vector search preview without sending the full graph."""
    return pipeline_service.preview_retrieval(
        req.query, req.top_k, selected_documents=req.selected_documents
    )


@router.get("/embedding-space", response_model=VisualizationData)
async def get_embedding_space(
    reduction_method: str = "pca",
    selected_documents: str | None = None,
):
    """
    Phase 3: return a standalone embedding-space projection for all chunks.

    Query params:
    - reduction_method: "pca" or "umap"
    - selected_documents: comma-separated document IDs (optional)
    """
    docs = [d for d in (selected_documents or "").split(",") if d.strip()]
    return pipeline_service.get_embedding_space(
        reduction_method=reduction_method,
        selected_documents=docs or None,
    )
