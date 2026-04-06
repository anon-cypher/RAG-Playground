import { useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  type Connection,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type { RagNodeData } from './adapters';
import { HANDLE_IN, HANDLE_OUT } from './adapters';
import { RagNode } from './RagNode';

const nodeTypes = { ragNode: RagNode };

function nextId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

export type FlowCanvasProps = {
  nodes: Node<RagNodeData>[];
  edges: Edge[];
  onNodesChange: import('@xyflow/react').OnNodesChange<Node<RagNodeData>>;
  onEdgesChange: import('@xyflow/react').OnEdgesChange;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  onDrop: (e: React.DragEvent) => void;
  onSelectionChange: (ids: { nodes: string[]; edges: string[] }) => void;
};

export function FlowCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  setEdges,
  onDrop,
  onSelectionChange,
}: FlowCanvasProps) {
  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            id: nextId('edge'),
            type: 'smoothstep',
            sourceHandle: params.sourceHandle ?? HANDLE_OUT,
            targetHandle: params.targetHandle ?? HANDLE_IN,
          },
          eds,
        ),
      ),
    [setEdges],
  );

  const isValidConnection = useCallback((c: Edge | Connection) => c.source !== c.target, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      isValidConnection={isValidConnection}
      nodeTypes={nodeTypes}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onSelectionChange={({ nodes: ns, edges: es }) => {
        onSelectionChange({
          nodes: ns.map((n) => n.id),
          edges: es.map((e) => e.id),
        });
      }}
      fitView
      minZoom={0.15}
      maxZoom={1.8}
      defaultEdgeOptions={{ type: 'smoothstep' }}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={16} size={1} color="rgba(148, 163, 184, 0.35)" />
      <Controls />
      <MiniMap zoomable pannable className="rag-minimap" />
    </ReactFlow>
  );
}
