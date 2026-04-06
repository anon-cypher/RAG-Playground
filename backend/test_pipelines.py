import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from main import app
from models.pipeline import RagType
from models.document import Chunk, ProcessedDocument, DocumentMetadata, ChunkConfig
import services.ingestion as ingestion
from services.retrieval import build_all_indices
from services.sparse_retrieval import sparse_retriever

client = TestClient(app)

# Common payload structure
base_payload = {
    "query": "What is semantic search?",
    "top_k": 3,
    "override_config": {
        "index_type": "flat",
        "retrieval_strategy": "vector",
        "openrouter_api_key": "dummy_key_for_testing",
        "llm_model": "meta-llama/llama-3.2-3b-instruct:free",
    }
}

@pytest.fixture(autouse=True)
def mock_documents():
    doc1 = ProcessedDocument(
        document_id="doc1",
        metadata=DocumentMetadata(filename="test.txt", file_type=".txt", total_chunks=1, total_characters=10, chunk_config=ChunkConfig()),
        chunks=[Chunk(text="Semantic search is cool", index=0, chunk_id="1", document_id="doc1", metadata={"source": "test"})]
    )
    doc2 = ProcessedDocument(
        document_id="doc2",
        metadata=DocumentMetadata(filename="test2.txt", file_type=".txt", total_chunks=1, total_characters=10, chunk_config=ChunkConfig()),
        chunks=[Chunk(text="Another informative chunk", index=1, chunk_id="2", document_id="doc2", metadata={"source": "test"})]
    )
    ingestion._documents = {"doc1": doc1, "doc2": doc2}
    sparse_retriever.build_index([c.text for c in ingestion.get_all_chunks()])
    build_all_indices(np.random.rand(2, 4096).astype(np.float32))
    yield
    ingestion._documents = {}
    sparse_retriever.is_built = False

@pytest.fixture(autouse=True)
def mock_external_apis():
    # Mock embeddings to return a random 4096-dim vector
    with patch("services.embedding.embed_query") as mock_embed_query, \
         patch("services.embedding.embed_texts") as mock_embed_texts, \
         patch("services.generation.OpenAI") as mock_gen_openai, \
         patch("services.agent.OpenAI") as mock_agent_openai, \
         patch("services.iterative.OpenAI") as mock_iterative_openai:
         
        mock_embed_query.return_value = np.random.rand(1, 4096).astype(np.float32)
        mock_embed_texts.return_value = np.random.rand(10, 4096).astype(np.float32)
        
        # Mock standard LLM generation
        mock_msg = MagicMock()
        mock_msg.message.content = "Mocked LLM generation string."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_msg]
        mock_resp.usage.total_tokens = 42
        
        mock_gen_inst = MagicMock()
        mock_gen_inst.chat.completions.create.return_value = mock_resp
        mock_gen_openai.return_value = mock_gen_inst
        
        # Mock iterative eval
        mock_iterative_msg = MagicMock()
        mock_iterative_msg.message.content = '{"is_sufficient": true, "suggested_next_query": ""}'
        mock_iter_resp = MagicMock()
        mock_iter_resp.choices = [mock_iterative_msg]
        mock_iter_resp.usage.total_tokens = 10
        
        mock_iter_client_inst = MagicMock()
        mock_iter_client_inst.chat.completions.create.side_effect = [mock_iter_resp, mock_resp]
        mock_iterative_openai.return_value = mock_iter_client_inst
        
        # Mock agent loop
        mock_agent_msg = MagicMock()
        mock_agent_msg.message.content = "[SYNTHESIZE: Mocked agentic response]"
        mock_agent_resp = MagicMock()
        mock_agent_resp.choices = [mock_agent_msg]
        mock_agent_resp.usage.total_tokens = 55
        
        mock_agent_client_inst = MagicMock()
        mock_agent_client_inst.chat.completions.create.return_value = mock_agent_resp
        mock_agent_openai.return_value = mock_agent_client_inst

        yield


def test_rag_naive():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "naive"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pipeline_config"]["rag_type"] == "naive"
    assert "retrieved_chunks" in data
    assert "answer" in data

def test_rag_advanced():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "advanced"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pipeline_config"]["rag_type"] == "advanced"

def test_rag_iterative():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "iterative"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_config"]["rag_type"] == "iterative"

def test_rag_agentic():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "agentic"
    payload["override_config"]["enable_agentic"] = True
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_config"]["rag_type"] == "agentic"

def test_rag_hybrid():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "hybrid"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_config"]["rag_type"] == "hybrid"
    
def test_rag_vectorless():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "vectorless"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_config"]["rag_type"] == "vectorless"

def test_rag_graph():
    payload = base_payload.copy()
    payload["override_config"]["rag_type"] = "graph"
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_config"]["rag_type"] == "graph"
