import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import type { RagNodeData } from './adapters';

type RagFlowNode = Node<RagNodeData>;

export function RagNode({ data, selected }: NodeProps<RagFlowNode>) {
  return (
    <div className={`rag-node${selected ? ' rag-node--selected' : ''}`}>
      <Handle className="rag-handle rag-handle--in" type="target" position={Position.Left} id="in" />
      <div className="rag-node__body">
        <div className="rag-node__title">{data.label}</div>
        <div className="rag-node__kind">{data.kind}</div>
      </div>
      <Handle className="rag-handle rag-handle--out" type="source" position={Position.Right} id="out" />
    </div>
  );
}
