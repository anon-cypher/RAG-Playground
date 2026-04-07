/** Pipeline graph types — aligned with FastAPI backend */

export type NodeKind =
  | 'document_loader'
  | 'chunker'
  | 'embedder'
  | 'indexer'
  | 'vector_store'
  | 'query'
  | 'retriever'
  | 'reranker'
  | 'prompt_augment'
  | 'llm'
  | 'output';

export interface PipelineNode {
  id: string;
  kind: NodeKind;
  label: string;
  x: number;
  y: number;
  config: Record<string, unknown>;
}

export interface PipelineEdge {
  id: string;
  source_id: string;
  source_port: string;
  target_id: string;
  target_port: string;
}

export interface PipelineGraph {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}

export interface NodeExecutionResult {
  node_id: string;
  kind: NodeKind;
  started_at_ms: number;
  finished_at_ms: number;
  latency_ms: number;
  output_summary: Record<string, unknown>;
}

export interface GraphExecutionResult {
  execution_order: string[];
  node_results: NodeExecutionResult[];
}

export interface ExecuteNodeResponse {
  kind: NodeKind;
  latency_ms: number;
  output_summary: Record<string, unknown>;
}

export interface PreviewRetrievalResponse {
  chunks: Array<Record<string, unknown>>;
  query_used: string;
  top_k: number;
  error: string | null;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  file_type: string;
  total_chunks: number;
  total_characters: number;
  created_at?: string;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
  total: number;
}

export interface Chunk {
  chunk_id: string;
  document_id: string;
  text: string;
  index: number;
  metadata: Record<string, unknown>;
}

export interface DocumentMetadata {
  filename: string;
  file_type: string;
  total_chunks: number;
  total_characters: number;
  chunk_config: Record<string, unknown>;
}

export interface ProcessedDocument {
  document_id: string;
  metadata: DocumentMetadata;
  chunks: Chunk[];
  raw_text?: string;
}

/** 3D embedding visualization (PCA projection) — aligns with backend `models/query.py`. */
export interface Point3D {
  x: number;
  y: number;
  z: number;
  label: string;
  chunk_id: string;
  is_query: boolean;
  is_neighbor: boolean;
  score: number;
  document_id: string;
  text_preview: string;
}

export interface VisualizationData {
  points: Point3D[];
  query_point: Point3D | null;
  neighbor_indices: number[];
  edges: number[][];
  rag_mode: string;
  caption: string;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  rank: number;
  metadata: Record<string, unknown>;
}

export interface StageMetrics {
  stage_name: string;
  latency_ms: number;
  details: Record<string, unknown>;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  index_type?: string;
  override_config?: Partial<PipelineConfig>;
  selected_documents?: string[];
  visualization_only?: boolean;
}

export interface QueryResponse {
  query: string;
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  visualization: VisualizationData;
  stage_metrics: StageMetrics[];
  total_latency_ms: number;
  pipeline_config: Record<string, unknown>;
}

export interface PipelineConfig {
  rag_type: string;
  index_type: string;
  retrieval_strategy: string;
  top_k: number;
  hnsw_m: number;
  hnsw_ef_construction: number;
  hnsw_ef_search: number;
  ivf_nlist?: number;
  pq_m?: number;
  pq_nbits?: number;
  enable_reranking: boolean;
  rerank_top_n: number;
  enable_agentic: boolean;
  temperature: number;
  max_tokens: number;
  openrouter_api_key?: string;
  llm_model?: string;
  embedding_model?: string;
  embedding_dim?: number;
}
