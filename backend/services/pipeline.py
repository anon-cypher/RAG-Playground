"""Pipeline orchestrator — ties together all services into a configurable RAG pipeline."""
import time
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

import config
from models.pipeline import (
    PipelineConfig,
    StageMetrics,
    RetrievalStrategy,
    PipelineGraph,
    GraphExecutionResult,
    NodeExecutionResult,
)
from models.query import (
    QueryRequest, QueryResponse, RetrievedChunk,
    Point3D, VisualizationData
)
from services import ingestion, embedding, retrieval, generation


# Current pipeline configuration
_current_config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get current pipeline configuration."""
    return _current_config


def set_config(cfg: PipelineConfig) -> PipelineConfig:
    """Update pipeline configuration."""
    global _current_config
    _current_config = cfg
    return _current_config


_is_building = False


def build_index() -> dict:
    """Build/rebuild the FAISS index from all ingested documents."""
    global _is_building
    if _is_building:
        return {"status": "in_progress", "message": "Index compilation already running."}
    
    _is_building = True
    try:
        chunks = ingestion.get_all_chunks()
        if not chunks:
            return {"status": "no_documents", "message": "No documents ingested yet."}

        texts = [c.text for c in chunks]

        # Embed all chunks
        t0 = time.perf_counter()
        embeddings = embedding.embed_texts(texts)
        embed_time = (time.perf_counter() - t0) * 1000

        # Build indices
        t0 = time.perf_counter()
        retrieval.build_all_indices(embeddings)

        # Also build the specific configured index type
        retrieval.get_or_create_retriever(
            index_type=_current_config.index_type.value,
            hnsw_m=_current_config.hnsw_m,
            hnsw_ef_construction=_current_config.hnsw_ef_construction,
            hnsw_ef_search=_current_config.hnsw_ef_search,
        )
        if not retrieval._retrievers.get(_current_config.index_type.value, None) or \
           not retrieval._retrievers[_current_config.index_type.value].is_built:
            ret = retrieval.get_or_create_retriever(_current_config.index_type.value)
            ret.build_index(embeddings)

        from services.sparse_retrieval import sparse_retriever
        from services.graph import knowledge_graph
        sparse_retriever.build_index(texts)
        knowledge_graph.build_graph(chunks)

        index_time = (time.perf_counter() - t0) * 1000

        return {
            "status": "success",
            "total_chunks": len(chunks),
            "embedding_time_ms": round(embed_time, 2),
            "index_time_ms": round(index_time, 2),
        }
    finally:
        _is_building = False


def execute_query(request: QueryRequest) -> QueryResponse:
    """Execute a full RAG pipeline strictly according to the selected RAG Architecture."""
    from models.pipeline import RagType
    
    cfg = request.override_config if request.override_config else _current_config
    stages: list[StageMetrics] = []
    total_start = time.perf_counter()

    top_k = request.top_k or cfg.top_k
    index_type = request.index_type or cfg.index_type.value
    rag_type = getattr(cfg, 'rag_type', RagType.NAIVE)

    # Stage 1: Embed query
    query_vec = None
    if rag_type != RagType.VECTORLESS:
        t0 = time.perf_counter()
        query_vec = embedding.embed_query(request.query)
        embed_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageMetrics(
            stage_name="embedding",
            latency_ms=round(embed_ms, 2),
            details={"model": cfg.embedding_model, "dim": cfg.embedding_dim}
        ))

    # Stage 2: Retrieve setup
    t0 = time.perf_counter()
    ret = retrieval.get_or_create_retriever(
        index_type=index_type, hnsw_m=cfg.hnsw_m,
        hnsw_ef_construction=cfg.hnsw_ef_construction, hnsw_ef_search=cfg.hnsw_ef_search,
    )

    stored = retrieval.get_stored_embeddings()
    if stored is None or not ret.is_built:
        build_index()
        stored = retrieval.get_stored_embeddings()
        ret = retrieval.get_or_create_retriever(index_type=index_type)

    try:
        if stored is None or ret.total_vectors == 0:
            raise ValueError("No documents indexed. Upload documents first.")
    except Exception as e:
        import traceback; traceback.print_exc()
        raise ValueError(f"Pipeline error: {str(e)}")

    all_chunks = ingestion.get_all_chunks()
    retrieved_chunks: list[RetrievedChunk] = []
    context_texts: list[str] = []
    selected_doc_set = set(request.selected_documents) if request.selected_documents else None

    final_indices = []
    final_scores = []
    retrieve_details = {"strategy": rag_type.value, "top_k": top_k}

    if rag_type == RagType.VECTORLESS:
        from services.sparse_retrieval import sparse_retriever
        
        k_sparse = max(top_k * 5, 200)
        s_scores, s_indices = sparse_retriever.search(request.query, k_sparse)
        
        for sc, idx in zip(s_scores, s_indices):
            if selected_doc_set and all_chunks[idx].document_id not in selected_doc_set: continue
            final_indices.append(idx)
            final_scores.append(sc)
            if len(final_indices) >= top_k: break

    elif rag_type == RagType.GRAPH:
        from services.graph import knowledge_graph
        graph_ids = knowledge_graph.retrieve_subgraph(request.query, top_k)
        for i, cid in enumerate(graph_ids):
            for idx, c in enumerate(all_chunks):
                if c.chunk_id == cid:
                    if selected_doc_set and c.document_id not in selected_doc_set: continue
                    final_indices.append(idx)
                    final_scores.append(1.0 / (i + 1.0))
                    break
            if len(final_indices) >= top_k: break

    elif rag_type in [RagType.ADVANCED, RagType.HYBRID]:
        # Advanced (pool -> rerank) & Hybrid (merge exact + semantic) map to RRF structurally
        dense_k = cfg.rerank_top_n if rag_type == RagType.ADVANCED else top_k * 2
        distances, indices = ret.search(query_vec, len(all_chunks))
        
        dense_indices = []
        dense_scores = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = all_chunks[idx]
            if selected_doc_set and chunk.document_id not in selected_doc_set: continue
            dense_indices.append(idx.item())
            dense_scores.append(dist.item())
            if len(dense_indices) >= dense_k: break
            
        from services.sparse_retrieval import sparse_retriever
        from services.reranking import reciprocal_rank_fusion
        
        s_scores, s_indices = sparse_retriever.search(request.query, dense_k * 3)
        sparse_indices = []
        sparse_scores = []
        for sc, idx in zip(s_scores, s_indices):
            if selected_doc_set and all_chunks[idx].document_id not in selected_doc_set: continue
            sparse_indices.append(idx)
            sparse_scores.append(sc)
        
        final_indices, final_scores = reciprocal_rank_fusion(
            dense_indices=dense_indices, dense_scores=dense_scores,
            sparse_indices=sparse_indices, sparse_scores=sparse_scores,
            k=60, top_k=top_k
        )
        retrieve_details["reranked"] = True

    else:
        # NAIVE, ITERATIVE, AGENTIC
        distances, indices = ret.search(query_vec, len(all_chunks))
        for dist, idx in zip(distances[0], indices[0]):
            chunk = all_chunks[idx]
            if selected_doc_set and chunk.document_id not in selected_doc_set: continue
            final_indices.append(idx.item())
            final_scores.append(float(1.0 / (1.0 + dist.item())))
            if len(final_indices) >= top_k: break

    retrieve_ms = (time.perf_counter() - t0) * 1000
    stages.append(StageMetrics(
        stage_name="retrieval",
        latency_ms=round(retrieve_ms, 2),
        details=retrieve_details
    ))

    for rank, (idx, score) in enumerate(zip(final_indices, final_scores)):
        if idx < 0 or idx >= len(all_chunks): continue
        chunk = all_chunks[idx]
        retrieved_chunks.append(RetrievedChunk(
            chunk_id=chunk.chunk_id, document_id=chunk.document_id,
            text=chunk.text, score=round(score, 4), rank=rank + 1, metadata=chunk.metadata,
        ))
        context_texts.append(chunk.text)

    # Stage 3: Generation / Agentic
    t0 = time.perf_counter()
    def retriever_callback(sq):
        sq_v = embedding.embed_query(sq)
        dists, indexes = ret.search(sq_v, len(all_chunks))
        sq_chunks = []
        for idx in indexes[0]:
            if selected_doc_set and all_chunks[idx].document_id not in selected_doc_set: continue
            sq_chunks.append(all_chunks[idx].text)
            if len(sq_chunks) >= top_k: break
        return sq_chunks

    if rag_type == RagType.AGENTIC:
        from services.agent import run_agentic_loop
        answer, logs, usage = run_agentic_loop(
            query=request.query, initial_context=context_texts,
            openrouter_api_key=cfg.openrouter_api_key, llm_model=cfg.llm_model, retriever_func=retriever_callback
        )
        gen_details = {"type": "agentic", "logs": logs, "usage": usage}

    elif rag_type == RagType.ITERATIVE:
        from services.iterative import run_iterative_loop
        answer, logs, usage = run_iterative_loop(
            query=request.query, retriever_func=retriever_callback,
            openrouter_api_key=cfg.openrouter_api_key, llm_model=cfg.llm_model
        )
        gen_details = {"type": "iterative", "logs": logs, "usage": usage}

    else:
        answer, usage = generation.generate_answer(
            query=request.query, context_chunks=context_texts,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            openrouter_api_key=cfg.openrouter_api_key, llm_model=cfg.llm_model,
        )
        gen_details = {"type": "standard", "model": cfg.llm_model, "usage": usage, "logs": [{"action": "Standard Synthesis Complete"}]}

    avg_score = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0
    gen_details["context_relevance_score"] = round(avg_score, 4)
    # rough token estimate: total chars of context + query / 4
    context_chars = sum(len(t) for t in context_texts)
    gen_details["prompt_token_count"] = (context_chars + len(request.query)) // 4
    gen_details["context_chunks_used"] = len(context_texts)

    gen_ms = (time.perf_counter() - t0) * 1000
    stages.append(StageMetrics(
        stage_name="generation", latency_ms=round(gen_ms, 2),
        details=gen_details
    ))

    # Optional 3D viz point coords mapping (we still pass valid indices to keep format stable)
    query_vec_viz = query_vec if query_vec is not None else np.zeros((1, config.EMBEDDING_DIM), dtype=np.float32)
    
    extra_edges = []
    if rag_type == RagType.GRAPH:
        from services.graph import knowledge_graph
        graph_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(all_chunks)}
        for cid in graph_chunk_ids:
            if f"chunk_{cid}" in knowledge_graph.graph:
                for neighbor in knowledge_graph.graph.neighbors(f"chunk_{cid}"):
                    if neighbor.startswith("chunk_"):
                        neighbor_id = neighbor.replace("chunk_", "")
                        if neighbor_id in chunk_id_to_idx:
                            edge = [chunk_id_to_idx[cid], chunk_id_to_idx[neighbor_id]]
                            if edge not in extra_edges and [edge[1], edge[0]] not in extra_edges:
                                extra_edges.append(edge)

    viz = _build_visualization_data(
        stored_embeddings=stored, query_vec=query_vec_viz,
        neighbor_indices=final_indices, neighbor_distances=final_scores, all_chunks=all_chunks,
        extra_edges=extra_edges
    )

    # Attach educational metadata to visualization
    RAG_CAPTIONS = {
        RagType.NAIVE:      "Finding closest N vectors in embedding space by L2 distance",
        RagType.ADVANCED:   "Retrieve a large pool, re-rank by relevance, extract best K",
        RagType.HYBRID:     "Fusing dense vector results with sparse keyword (BM25) signals",
        RagType.GRAPH:      "Traversing a knowledge graph 1-hop from matched keyword entities",
        RagType.ITERATIVE:  "LLM re-queries the index until its confidence threshold is met",
        RagType.AGENTIC:    "LLM autonomously routes tool calls to gather partial contexts",
        RagType.VECTORLESS: "No embeddings — pure keyword / TF-IDF text matching",
    }
    viz.rag_mode = rag_type.value
    viz.caption  = RAG_CAPTIONS.get(rag_type, "")

    total_ms = (time.perf_counter() - total_start) * 1000

    return QueryResponse(
        query=request.query, answer=answer, retrieved_chunks=retrieved_chunks,
        visualization=viz, stage_metrics=stages, total_latency_ms=round(total_ms, 2),
        pipeline_config=cfg.model_dump(),
    )


def _build_visualization_data(
    stored_embeddings: np.ndarray,
    query_vec: np.ndarray,
    neighbor_indices: list[int],
    neighbor_distances: list[float],
    all_chunks,
    extra_edges: list[list[int]] = None
) -> VisualizationData:
    """Project embeddings to 3D using PCA for visualization."""
    # Combine document embeddings + query embedding
    all_vecs = np.vstack([stored_embeddings, query_vec])

    # PCA to 3D
    n_components = min(3, all_vecs.shape[0], all_vecs.shape[1])
    if n_components < 3:
        # Pad with zeros if fewer than 3 components
        coords_3d = np.zeros((all_vecs.shape[0], 3), dtype=np.float32)
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(all_vecs)
        coords_3d[:, :n_components] = coords
    else:
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(all_vecs)

    # Scale for better visualization
    scale = 10.0
    coords_3d = coords_3d * scale

    valid_neighbor_set = set(
        idx for idx in neighbor_indices if 0 <= idx < len(all_chunks)
    )

    # Build 3D points for document chunks
    points: list[Point3D] = []
    for i in range(len(all_chunks)):
        chunk = all_chunks[i]
        is_nb = i in valid_neighbor_set
        # Calculate score for neighbors
        score = 0.0
        if is_nb:
            pos = neighbor_indices.index(i)
            # Scores passed are already formatted via final_scores (RRF or inverted L2)
            score = neighbor_distances[pos]

        points.append(Point3D(
            x=float(coords_3d[i, 0]),
            y=float(coords_3d[i, 1]),
            z=float(coords_3d[i, 2]),
            label=f"Chunk {chunk.index}",
            chunk_id=chunk.chunk_id,
            is_query=False,
            is_neighbor=is_nb,
            score=round(score, 4),
            document_id=chunk.document_id,
            text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
        ))

    # Query point (last in the array)
    query_idx = len(all_chunks)
    query_point = Point3D(
        x=float(coords_3d[query_idx, 0]),
        y=float(coords_3d[query_idx, 1]),
        z=float(coords_3d[query_idx, 2]),
        label="Query",
        is_query=True,
        text_preview="",
    )

    # Build edges from query to neighbors
    edges = []
    neighbor_point_indices = []
    for idx in neighbor_indices:
        if 0 <= idx < len(points):
            edges.append([len(points), idx])  # query_point_index -> neighbor
            neighbor_point_indices.append(idx)

    # Add extra graph edges if any
    if extra_edges:
        for edge in extra_edges:
            edges.append(edge)

    return VisualizationData(
        points=points,
        query_point=query_point,
        neighbor_indices=neighbor_point_indices,
        edges=edges,
    )


def execute_graph(graph: PipelineGraph) -> GraphExecutionResult:
    """
    Execute a visual pipeline graph as a pure DAG.

    Phase 1 keeps execution intentionally lightweight:
    - Validate acyclicity.
    - Compute a topological order.
    - Emit per-node timing metadata and a small structural summary.
    """
    # Build adjacency + in-degree for Kahn topological sort
    in_degree: dict[str, int] = {n.id: 0 for n in graph.nodes}
    adjacency: dict[str, list[str]] = {n.id: [] for n in graph.nodes}

    for edge in graph.edges:
        if edge.source_id not in adjacency or edge.target_id not in in_degree:
            # Ignore dangling edges for now; frontend should prevent them.
            continue
        adjacency[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] += 1

    # Initial frontier: all nodes with no incoming edges
    frontier = [nid for nid, deg in in_degree.items() if deg == 0]
    order: list[str] = []

    while frontier:
        nid = frontier.pop(0)
        order.append(nid)
        for neighbor in adjacency.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                frontier.append(neighbor)

    # If we could not visit every node, the graph has a cycle; keep behavior explicit.
    if len(order) != len(graph.nodes):
        # Fall back to original insertion order but still return something useful.
        order = [n.id for n in graph.nodes]

    # Build quick lookup for convenience
    node_by_id = {n.id: n for n in graph.nodes}

    results: list[NodeExecutionResult] = []
    current_time_ms = lambda: time.perf_counter() * 1000.0

    # Simulate per-node execution; later phases can plug real RAG stages here.
    for nid in order:
        node = node_by_id[nid]
        started = current_time_ms()
        # Minimal synthetic latency to make the animation meaningful on the UI.
        # We avoid real sleeps here to keep the API snappy; frontend can animate.
        finished = current_time_ms()
        latency = finished - started

        # Structural summary: which upstream nodes feed into this one.
        upstream_ids = [
            e.source_id
            for e in graph.edges
            if e.target_id == nid
        ]

        results.append(
            NodeExecutionResult(
                node_id=nid,
                kind=node.kind,
                started_at_ms=started,
                finished_at_ms=finished,
                latency_ms=latency,
                output_summary={
                    "label": node.label,
                    "config": node.config,
                    "upstream_nodes": upstream_ids,
                },
            )
        )

    return GraphExecutionResult(
        execution_order=order,
        node_results=results,
    )


def get_embedding_space(reduction_method: str = "pca", selected_documents: Optional[list[str]] = None) -> VisualizationData:
    """
    Build a standalone embedding-space visualization for Phase 3.

    - Supports PCA and UMAP projections into 3D.
    - Returns points only (no query point / edges) for interactive exploration.
    """
    chunks = ingestion.get_all_chunks()
    if not chunks:
        return VisualizationData(
            points=[],
            query_point=None,
            neighbor_indices=[],
            edges=[],
            rag_mode="embedding-space",
            caption="No documents available. Upload content to visualize embeddings.",
        )

    # Ensure embeddings exist
    stored = retrieval.get_stored_embeddings()
    if stored is None or stored.shape[0] != len(chunks):
        build_index()
        stored = retrieval.get_stored_embeddings()
        if stored is None:
            return VisualizationData(
                points=[],
                query_point=None,
                neighbor_indices=[],
                edges=[],
                rag_mode="embedding-space",
                caption="Failed to build embeddings. Please retry ingestion.",
            )

    selected_set = set(selected_documents) if selected_documents else None
    indices = [
        i for i, c in enumerate(chunks)
        if selected_set is None or c.document_id in selected_set
    ]

    if not indices:
        return VisualizationData(
            points=[],
            query_point=None,
            neighbor_indices=[],
            edges=[],
            rag_mode="embedding-space",
            caption="No chunks match the selected document filter.",
        )

    vectors = stored[indices]
    method_used = reduction_method.lower()

    # Default to PCA for stability; switch to UMAP if requested and available.
    if method_used == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=max(5, min(30, len(indices) - 1)),
                min_dist=0.1,
            )
            coords_3d = reducer.fit_transform(vectors)
        except Exception:
            method_used = "pca"

    if method_used != "umap":
        n_components = min(3, vectors.shape[0], vectors.shape[1])
        coords_3d = np.zeros((vectors.shape[0], 3), dtype=np.float32)
        if n_components > 0:
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(vectors)
            coords_3d[:, :n_components] = reduced

    coords_3d = coords_3d * 10.0

    points: list[Point3D] = []
    for local_i, global_idx in enumerate(indices):
        chunk = chunks[global_idx]
        points.append(
            Point3D(
                x=float(coords_3d[local_i, 0]),
                y=float(coords_3d[local_i, 1]),
                z=float(coords_3d[local_i, 2]),
                label=f"Chunk {chunk.index}",
                chunk_id=chunk.chunk_id,
                is_query=False,
                is_neighbor=False,
                score=0.0,
                document_id=chunk.document_id,
                text_preview=chunk.text[:160] + ("..." if len(chunk.text) > 160 else ""),
            )
        )

    return VisualizationData(
        points=points,
        query_point=None,
        neighbor_indices=[],
        edges=[],
        rag_mode="embedding-space",
        caption=f"Embedding projection in 3D using {method_used.upper()}. Click points to inspect chunk text.",
    )
