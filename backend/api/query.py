"""Query API routes."""
from fastapi import APIRouter, HTTPException

from models.query import QueryRequest, QueryResponse
from services import pipeline as pipeline_service

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a RAG query through the configured pipeline."""
    try:
        response = pipeline_service.execute_query(request)
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {str(e)}")
