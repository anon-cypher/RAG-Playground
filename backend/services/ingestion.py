"""Document ingestion service — parsing, chunking, and metadata extraction."""
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader

from models.document import (
    Chunk, ChunkConfig, ChunkingStrategy,
    DocumentMetadata, ProcessedDocument, DocumentSummary
)


# In-memory document store
_documents: dict[str, ProcessedDocument] = {}
_document_timestamps: dict[str, str] = {}


def parse_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file."""
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        return file_content.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        return _parse_pdf(file_content)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf")


def _parse_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    import io
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    """Split text into chunks based on the configured strategy."""
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if not text:
        return []

    if config.strategy == ChunkingStrategy.FIXED:
        return _fixed_chunk(text, config.chunk_size)
    elif config.strategy == ChunkingStrategy.OVERLAPPING:
        return _overlapping_chunk(text, config.chunk_size, config.overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def _fixed_chunk(text: str, chunk_size: int) -> list[str]:
    """Split text into fixed-size chunks at word boundaries."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current else 0)
        if current_len + word_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _overlapping_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks at word boundaries."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        # Build chunk of approximately chunk_size characters
        end = start
        current_len = 0
        while end < len(words) and current_len + len(words[end]) + (1 if end > start else 0) <= chunk_size:
            current_len += len(words[end]) + (1 if end > start else 0)
            end += 1

        if end == start:
            end = start + 1  # At least one word

        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        # Calculate overlap in words (approximate)
        overlap_words = 0
        overlap_len = 0
        for i in range(end - 1, start - 1, -1):
            if overlap_len + len(words[i]) > overlap:
                break
            overlap_len += len(words[i]) + 1
            overlap_words += 1

        old_start = start
        start = end - overlap_words
        
        # Prevent infinite loops
        if start <= old_start:
            start = old_start + 1

    return chunks


def process_document(
    file_content: bytes,
    filename: str,
    chunk_config: ChunkConfig | None = None
) -> ProcessedDocument:
    """Parse, chunk, and store a document."""
    if chunk_config is None:
        chunk_config = ChunkConfig()

    doc_id = str(uuid.uuid4())
    text = parse_file(file_content, filename)
    chunk_texts = chunk_text(text, chunk_config)

    chunks = [
        Chunk(
            chunk_id=f"{doc_id}_chunk_{i}",
            document_id=doc_id,
            text=ct,
            index=i,
            metadata={
                "filename": filename,
                "chunk_index": i,
                "char_count": len(ct),
            }
        )
        for i, ct in enumerate(chunk_texts)
    ]

    metadata = DocumentMetadata(
        filename=filename,
        file_type=Path(filename).suffix.lower(),
        total_chunks=len(chunks),
        total_characters=len(text),
        chunk_config=chunk_config,
    )

    doc = ProcessedDocument(
        document_id=doc_id,
        metadata=metadata,
        chunks=chunks,
        raw_text=text,
    )

    _documents[doc_id] = doc
    _document_timestamps[doc_id] = datetime.now(timezone.utc).isoformat()
    return doc

def reprocess_all_documents(config: ChunkConfig) -> int:
    """Re-chunk all existing documents using a new configuration."""
    total_rechunked = 0
    for doc in _documents.values():
        chunk_texts = chunk_text(doc.raw_text, config)
        doc.chunks = [
            Chunk(
                chunk_id=f"{doc.document_id}_chunk_{i}",
                document_id=doc.document_id,
                text=ct,
                index=i,
                metadata={
                    "filename": doc.metadata.filename,
                    "chunk_index": i,
                    "char_count": len(ct),
                }
            )
            for i, ct in enumerate(chunk_texts)
        ]
        doc.metadata.chunk_config = config
        doc.metadata.total_chunks = len(doc.chunks)
        total_rechunked += len(doc.chunks)
    return total_rechunked


def get_document(doc_id: str) -> ProcessedDocument | None:
    """Get a document by ID."""
    return _documents.get(doc_id)


def list_documents() -> list[DocumentSummary]:
    """List all ingested documents."""
    return [
        DocumentSummary(
            document_id=doc.document_id,
            filename=doc.metadata.filename,
            file_type=doc.metadata.file_type,
            total_chunks=doc.metadata.total_chunks,
            total_characters=doc.metadata.total_characters,
            created_at=_document_timestamps.get(doc.document_id),
        )
        for doc in _documents.values()
    ]


def delete_document(doc_id: str) -> bool:
    """Delete a document by ID."""
    if doc_id in _documents:
        del _documents[doc_id]
        _document_timestamps.pop(doc_id, None)
        return True
    return False


def get_all_chunks() -> list[Chunk]:
    """Get all chunks across all documents."""
    chunks = []
    for doc in _documents.values():
        chunks.extend(doc.chunks)
    return chunks


def get_chunk_texts() -> list[str]:
    """Get all chunk texts for embedding."""
    return [chunk.text for chunk in get_all_chunks()]
