"""Document management API routes."""
import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from models.document import (
    ChunkConfig, ChunkingStrategy,
    DocumentListResponse, DocumentSummary
)
from services import ingestion, retrieval

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _invalidate_search_indices() -> None:
    """After chunks change without re-embedding, drop vector indices so retrieval cannot use stale vectors."""
    retrieval.clear_indices()


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(default="overlapping"),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    """Upload a document: parse and chunk only. Run Embedder (build index) separately for vectors."""
    # Validate file type
    if file.filename is None:
        raise HTTPException(400, "Filename is required.")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("txt", "pdf"):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Supported: .txt, .pdf")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    # Build chunk config
    try:
        strategy = ChunkingStrategy(chunk_strategy)
    except ValueError:
        strategy = ChunkingStrategy.OVERLAPPING

    chunk_config = ChunkConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=chunk_overlap if strategy == ChunkingStrategy.OVERLAPPING else 0,
    )

    # Process document (parse + chunk + store)
    try:
        doc = ingestion.process_document(content, file.filename, chunk_config)
    except Exception as e:
        raise HTTPException(500, f"Error processing document: {str(e)}")

    _invalidate_search_indices()

    return {
        "status": "success",
        "document_id": doc.document_id,
        "filename": doc.metadata.filename,
        "total_chunks": doc.metadata.total_chunks,
        "total_characters": doc.metadata.total_characters,
        "message": f"Document '{file.filename}' stored: {doc.metadata.total_chunks} chunks. Use Embedder to build embeddings and index.",
    }


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/upload-stream")
async def upload_document_stream(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(default="overlapping"),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    """Upload with SSE: parse and chunk only (no embedding)."""
    if file.filename is None:
        raise HTTPException(400, "Filename is required.")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("txt", "pdf"):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Supported: .txt, .pdf")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    try:
        strategy = ChunkingStrategy(chunk_strategy)
    except ValueError:
        strategy = ChunkingStrategy.OVERLAPPING

    chunk_config = ChunkConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=chunk_overlap if strategy == ChunkingStrategy.OVERLAPPING else 0,
    )

    filename = file.filename

    async def event_generator():
        # Step 1 — Parse
        yield _sse({"step": "parse", "progress": 5, "message": f"Reading '{filename}'...", "detail": f"{len(content):,} bytes"})
        await asyncio.sleep(0)

        try:
            text = ingestion.parse_file(content, filename)
        except Exception as e:
            yield _sse({"step": "error", "progress": 0, "message": str(e)})
            return

        yield _sse({"step": "parse", "progress": 25, "message": "Text extracted.", "detail": f"{len(text):,} characters"})
        await asyncio.sleep(0)

        # Step 2 — Chunk
        yield _sse({"step": "chunk", "progress": 35, "message": "Splitting into chunks...", "detail": f"Strategy: {strategy.value}, size: {chunk_size}, overlap: {chunk_overlap}"})
        await asyncio.sleep(0)

        try:
            doc = ingestion.process_document(content, filename, chunk_config)
        except Exception as e:
            yield _sse({"step": "error", "progress": 0, "message": f"Chunking failed: {str(e)}"})
            return

        yield _sse({"step": "chunk", "progress": 75, "message": f"Created {doc.metadata.total_chunks} chunks.", "detail": f"Avg ~{chunk_size} chars each"})
        await asyncio.sleep(0)

        _invalidate_search_indices()

        yield _sse({
            "step": "done",
            "progress": 100,
            "message": f"'{filename}' stored.",
            "detail": f"{doc.metadata.total_chunks} chunks — run Embedder to build embeddings and index.",
            "document_id": doc.document_id,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/reprocess", response_model=dict)
async def reprocess_documents(config: ChunkConfig):
    """Re-chunk all documents. Does not embed; run pipeline build-index from the Embedder UI."""
    try:
        total_chunks = ingestion.reprocess_all_documents(config)
        _invalidate_search_indices()
        return {
            "status": "success",
            "message": f"Reprocessed into {total_chunks} chunks. Run Embedder to rebuild embeddings and index.",
            "total_chunks": total_chunks,
        }
    except Exception as e:
        raise HTTPException(500, f"Reprocessing failed: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    docs = ingestion.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get a specific document with its chunks."""
    doc = ingestion.get_document(document_id)
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")
    return doc


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document. Clears vector indices; run Embedder to rebuild after further uploads."""
    success = ingestion.delete_document(document_id)
    if not success:
        raise HTTPException(404, f"Document {document_id} not found.")

    _invalidate_search_indices()

    return {"status": "success", "message": f"Document {document_id} deleted. Run Embedder to rebuild index if needed."}
