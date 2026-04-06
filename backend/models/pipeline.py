"""Pipeline configuration Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"
    PQ = "pq"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class RagType(str, Enum):
    NAIVE = "naive"
    ADVANCED = "advanced"
    ITERATIVE = "iterative"
    AGENTIC = "agentic"
    HYBRID = "hybrid"
    VECTORLESS = "vectorless"
    GRAPH = "graph"


class PipelineConfig(BaseModel):
    rag_type: RagType = RagType.NAIVE
    index_type: IndexType = IndexType.FLAT
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    top_k: int = Field(default=5, ge=1, le=50)
    # HNSW params
    hnsw_m: int = Field(default=32, ge=4, le=128)
    hnsw_ef_construction: int = Field(default=200, ge=50, le=1000)
    hnsw_ef_search: int = Field(default=64, ge=10, le=500)
    # IVF params
    ivf_nlist: int = Field(default=10, ge=1, le=100)
    # PQ params
    pq_m: int = Field(default=8, ge=2, le=64)
    pq_nbits: int = Field(default=8, ge=4, le=8)
    # Reranking (Phase 2)
    enable_reranking: bool = False
    rerank_top_n: int = Field(default=10, ge=1, le=50)
    enable_agentic: bool = False
    # Generation
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=50, le=4096)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API Key")
    llm_model: str = Field(default="meta-llama/llama-3.2-3b-instruct:free", description="OpenRouter Model ID")
    embedding_model: str = Field(default="qwen/qwen3-embedding-8b", description="Embedding Model ID")
    embedding_dim: int = Field(default=4096, description="Embedding Dimension")


class StageMetrics(BaseModel):
    stage_name: str
    latency_ms: float
    details: dict = Field(default_factory=dict)


class PipelineTypesResponse(BaseModel):
    index_types: list[dict]
    retrieval_strategies: list[dict]


class NodeKind(str, Enum):
    """Canonical node types for the visual RAG pipeline builder."""

    DOCUMENT_LOADER = "document_loader"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    INDEXER = "indexer"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    LLM = "llm"
    OUTPUT = "output"


class NodePortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class PipelineNode(BaseModel):
    """Single node instance on the visual canvas."""

    id: str
    kind: NodeKind
    label: str
    x: float = 0.0
    y: float = 0.0
    config: dict = Field(default_factory=dict)


class PipelineEdge(BaseModel):
    """Directed edge between two node ports."""

    id: str
    source_id: str
    source_port: str = "out"
    target_id: str
    target_port: str = "in"


class PipelineGraph(BaseModel):
    """Full DAG description sent from the frontend."""

    nodes: list[PipelineNode]
    edges: list[PipelineEdge]


class NodeExecutionResult(BaseModel):
    """Per-node execution metadata returned to the frontend."""

    node_id: str
    kind: NodeKind
    started_at_ms: float
    finished_at_ms: float
    latency_ms: float
    output_summary: dict = Field(default_factory=dict)


class GraphExecutionResult(BaseModel):
    """Execution summary for a single graph run."""

    execution_order: list[str]
    node_results: list[NodeExecutionResult]
