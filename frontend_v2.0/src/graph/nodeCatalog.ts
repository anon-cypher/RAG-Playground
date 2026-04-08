import type { Edge, Node } from '@xyflow/react';
import type { NodeKind } from '../types/pipeline';
import type { PipelineGraph } from '../types/pipeline';
import type { RagNodeData } from './adapters';
import { toReactFlow } from './adapters';

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
      embedding_model: 'openai/text-embedding-3-small',
      embedding_dim: 1536,
    },
  },
  {
    kind: 'indexer',
    label: 'Indexer',
    description: 'Builds the vector index.',
    defaultConfig: { index_type: 'flat' },
  },
  {
    kind: 'vector_store',
    label: 'Vector Store',
    description: 'Summarizes indexed vectors (health / browse). Built by Indexer.',
    defaultConfig: { index_type: 'flat' },
  },
  {
    kind: 'query',
    label: 'Query',
    description: 'User question for retrieval and generation.',
    defaultConfig: { query_text: 'What is the main topic of the uploaded documents?' },
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
    kind: 'prompt_augment',
    label: 'Prompt augment',
    description: 'Assembles system + user prompt for the LLM.',
    defaultConfig: {
      system_prompt:
        'You are a helpful assistant. Use the numbered context blocks [1], [2], … When you use facts from a block, cite it in your answer with the same bracket number, e.g. [1].',
      user_template: 'Context:\n{context}\n\nQuestion:\n{query}',
      context_separator: '\n\n---\n\n',
    },
  },
  {
    kind: 'llm',
    label: 'LLM',
    description: 'Generates the final answer.',
    defaultConfig: { temperature: 0.7, llm_model: 'openai/gpt-4o-mini' },
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

/** Diagram-like linear chain: indexing → query path → augment → LLM → output. */
export function createBootstrapGraph(): PipelineGraph {
  const baseY = 130;
  const spacingX = 150;
  const x0 = 24;

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
  const n5 = makeNode('vector_store', 'Vector Store', x0 + spacingX * 4, baseY);
  const n6 = makeNode('query', 'Query', x0 + spacingX * 5, baseY);
  const n7 = makeNode('retriever', 'Retriever', x0 + spacingX * 6, baseY);
  const n8 = makeNode('reranker', 'Re-rank', x0 + spacingX * 7, baseY);
  const n9 = makeNode('prompt_augment', 'Augment', x0 + spacingX * 8, baseY);
  const n10 = makeNode('llm', 'LLM', x0 + spacingX * 9, baseY);
  const n11 = makeNode('output', 'Response', x0 + spacingX * 10, baseY);

  const nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11];
  const edges: PipelineGraph['edges'] = [
    { id: nextId('edge'), source_id: n1.id, source_port: 'out', target_id: n2.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n2.id, source_port: 'out', target_id: n3.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n3.id, source_port: 'out', target_id: n4.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n4.id, source_port: 'out', target_id: n5.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n5.id, source_port: 'out', target_id: n6.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n6.id, source_port: 'out', target_id: n7.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n7.id, source_port: 'out', target_id: n8.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n8.id, source_port: 'out', target_id: n9.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n9.id, source_port: 'out', target_id: n10.id, target_port: 'in' },
    { id: nextId('edge'), source_id: n10.id, source_port: 'out', target_id: n11.id, target_port: 'in' },
  ];

  return { nodes, edges };
}

const LANE_INDEX = 'lane-indexing';
const LANE_QUERY = 'lane-query';
const LANE_GEN = 'lane-generation';

const INDEX_KINDS = new Set<NodeKind>([
  'document_loader',
  'chunker',
  'embedder',
  'indexer',
  'vector_store',
]);
const QUERY_KINDS = new Set<NodeKind>(['query', 'retriever']);
const GEN_KINDS = new Set<NodeKind>(['reranker', 'prompt_augment', 'llm', 'output']);

/** Bootstrap graph with XYFlow group lanes (indexing / query / generation). */
export function createBootstrapReactFlow(): { nodes: Node[]; edges: Edge[] } {
  const graph = createBootstrapGraph();
  const { nodes: flat, edges } = toReactFlow(graph);

  const groupNodes: Node[] = [
    {
      id: LANE_INDEX,
      type: 'group',
      position: { x: 8, y: 12 },
      style: { width: 828, height: 208 },
      data: { label: 'Indexing' },
      draggable: false,
      selectable: false,
      zIndex: -1,
    },
    {
      id: LANE_QUERY,
      type: 'group',
      position: { x: 8, y: 236 },
      style: { width: 328, height: 200 },
      data: { label: 'Query path' },
      draggable: false,
      selectable: false,
      zIndex: -1,
    },
    {
      id: LANE_GEN,
      type: 'group',
      position: { x: 352, y: 236 },
      style: { width: 648, height: 200 },
      data: { label: 'Generation' },
      draggable: false,
      selectable: false,
      zIndex: -1,
    },
  ];

  let iIdx = 0;
  let iQ = 0;
  let iG = 0;
  const col = (i: number) => 20 + i * 148;

  const ragNodes = flat.map((n) => {
    const kind = n.data.kind;
    if (INDEX_KINDS.has(kind)) {
      const node: Node<RagNodeData> = {
        ...n,
        parentId: LANE_INDEX,
        position: { x: col(iIdx++), y: 58 },
        extent: 'parent',
      };
      return node;
    }
    if (QUERY_KINDS.has(kind)) {
      const node: Node<RagNodeData> = {
        ...n,
        parentId: LANE_QUERY,
        position: { x: col(iQ++), y: 58 },
        extent: 'parent',
      };
      return node;
    }
    if (GEN_KINDS.has(kind)) {
      const node: Node<RagNodeData> = {
        ...n,
        parentId: LANE_GEN,
        position: { x: col(iG++), y: 58 },
        extent: 'parent',
      };
      return node;
    }
    return n;
  });

  return { nodes: [...groupNodes, ...ragNodes], edges };
}
