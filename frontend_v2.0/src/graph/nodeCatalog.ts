import type { NodeKind } from '../types/pipeline';
import type { PipelineGraph } from '../types/pipeline';

export interface NodeTypeDef {
  kind: NodeKind;
  label: string;
  description: string;
  defaultConfig: Record<string, unknown>;
}

export const NODE_TYPES: NodeTypeDef[] = [
  {
    kind: 'document_loader',
    label: 'Document Loader',
    description: 'Entry point: reads documents from the knowledge base.',
    defaultConfig: {},
  },
  {
    kind: 'chunker',
    label: 'Chunker',
    description: 'Splits documents into chunks.',
    defaultConfig: { chunk_strategy: 'overlapping', chunk_size: 500, overlap: 50 },
  },
  {
    kind: 'embedder',
    label: 'Embedder',
    description: 'Turns chunks into dense vectors.',
    defaultConfig: {
      openrouter_api_key: '',
      embedding_model: 'qwen/qwen3-embedding-8b',
      embedding_dim: 4096,
    },
  },
  {
    kind: 'indexer',
    label: 'Indexer',
    description: 'Builds the vector index.',
    defaultConfig: { index_type: 'flat' },
  },
  {
    kind: 'retriever',
    label: 'Retriever',
    description: 'Fetches nearest chunks for a query.',
    defaultConfig: { top_k: 5 },
  },
  {
    kind: 'reranker',
    label: 'Re-ranker',
    description: 'Re-ranks retrieved chunks.',
    defaultConfig: { enable: false },
  },
  {
    kind: 'llm',
    label: 'LLM',
    description: 'Generates the final answer.',
    defaultConfig: { temperature: 0.7 },
  },
  {
    kind: 'output',
    label: 'Output',
    description: 'Terminal node.',
    defaultConfig: {},
  },
];

function nextId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Linear pipeline matching v1 bootstrap for a quick start. */
export function createBootstrapGraph(): PipelineGraph {
  const baseY = 120;
  const spacingX = 200;
  const x0 = 40;

  const makeNode = (
    kind: NodeKind,
    label: string,
    x: number,
    y: number,
    config: Record<string, unknown> = {},
  ) => {
    const def = NODE_TYPES.find((t) => t.kind === kind)!;
    return {
      id: nextId(kind),
      kind,
      label,
      x,
      y,
      config: { ...def.defaultConfig, ...config },
    };
  };

  const n1 = makeNode('document_loader', 'Documents', x0, baseY);
  const n2 = makeNode('chunker', 'Chunker', x0 + spacingX, baseY);
  const n3 = makeNode('embedder', 'Embedder', x0 + spacingX * 2, baseY);
  const n4 = makeNode('indexer', 'Indexer', x0 + spacingX * 3, baseY);
  const n5 = makeNode('retriever', 'Retriever', x0 + spacingX * 4, baseY);
  const n6 = makeNode('llm', 'LLM', x0 + spacingX * 5, baseY);
  const n7 = makeNode('output', 'Answer', x0 + spacingX * 6, baseY);

  const nodes = [n1, n2, n3, n4, n5, n6, n7];
  const edges: PipelineGraph['edges'] = [
    { id: nextId('edge'), source_id: n1.id, source_port: 'out', target_id: n2.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n2.id, source_port: 'out', target_id: n3.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n3.id, source_port: 'out', target_id: n4.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n4.id, source_port: 'out', target_id: n5.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n5.id, source_port: 'out', target_id: n6.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n6.id, source_port: 'out', target_id: n7.id, target_port: 'in' },
  ];

  return { nodes, edges };
}
