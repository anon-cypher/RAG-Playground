"""Query-related Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from .pipeline import StageMetrics, PipelineConfig


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = None  # Override pipeline config if set
    index_type: Optional[str] = None  # Override pipeline config if set
    override_config: Optional[PipelineConfig] = None
    selected_documents: Optional[list[str]] = None
    visualization_only: bool = Field(
        default=False,
        description="Skip LLM generation; return retrieval + 3D visualization only (faster, cheaper).",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict = Field(default_factory=dict)


class Point3D(BaseModel):
    x: float
    y: float
    z: float
    label: str = ""
    chunk_id: str = ""
    is_query: bool = False
    is_neighbor: bool = False
    score: float = 0.0
    document_id: str = ""
    text_preview: str = ""


class VisualizationData(BaseModel):
    points: list[Point3D]
    query_point: Optional[Point3D] = None
    neighbor_indices: list[int] = Field(default_factory=list)
    edges: list[list[int]] = Field(default_factory=list)
    rag_mode: str = ""
    caption: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    visualization: VisualizationData
    stage_metrics: list[StageMetrics]
    total_latency_ms: float
    pipeline_config: dict = Field(default_factory=dict)
