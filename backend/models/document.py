"""Document-related Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    OVERLAPPING = "overlapping"


class ChunkConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.OVERLAPPING
    chunk_size: int = Field(default=500, ge=50, le=5000)
    overlap: int = Field(default=50, ge=0, le=500)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    index: int
    metadata: dict = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    total_chunks: int
    total_characters: int
    chunk_config: ChunkConfig


class ProcessedDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    chunks: list[Chunk]
    raw_text: str = Field(default="")


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_chunks: int
    total_characters: int
    created_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int
