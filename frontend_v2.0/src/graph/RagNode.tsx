import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import type { RagNodeData } from './adapters';

type RagFlowNode = Node<RagNodeData>;

const LANE: Record<string, string> = {
  document_loader: 'indexing',
  chunker: 'indexing',
  embedder: 'indexing',
  indexer: 'indexing',
  vector_store: 'indexing',
  query: 'query',
  retriever: 'query',
  reranker: 'gen',
  prompt_augment: 'gen',
  llm: 'gen',
  output: 'gen',
};

export function RagNode({ data, selected }: NodeProps<RagFlowNode>) {
  const lane = LANE[data.kind] ?? 'gen';
  const runState = data.runState ?? 'idle';
  return (
    <div
      className={`rag-node rag-node--lane-${lane} rag-node--run-${runState}${selected ? ' rag-node--selected' : ''}`}
    >
      <Handle className="rag-handle rag-handle--in" type="target" position={Position.Left} id="in" />
      <div className="rag-node__body">
        <div className="rag-node__title">{data.label}</div>
        <div className="rag-node__kind">{data.kind}</div>
      </div>
      {runState === 'done' ? <span className="rag-node__tick" aria-label="step completed">OK</span> : null}
      <Handle className="rag-handle rag-handle--out" type="source" position={Position.Right} id="out" />
    </div>
  );
}
