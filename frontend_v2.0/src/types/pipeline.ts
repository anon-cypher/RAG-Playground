/** Pipeline graph types — aligned with FastAPI backend */

export type NodeKind =
  | 'document_loader'
  | 'chunker'
  | 'embedder'
  | 'indexer'
  | 'retriever'
  | 'reranker'
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
