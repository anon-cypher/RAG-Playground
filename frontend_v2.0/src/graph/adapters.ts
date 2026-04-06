import type { Edge, Node } from '@xyflow/react';
import type { NodeKind, PipelineEdge, PipelineGraph, PipelineNode } from '../types/pipeline';

export type RagNodeData = {
  kind: NodeKind;
  label: string;
  config: Record<string, unknown>;
};

export const HANDLE_IN = 'in';
export const HANDLE_OUT = 'out';

export function toReactFlow(graph: PipelineGraph): { nodes: Node<RagNodeData>[]; edges: Edge[] } {
  const nodes: Node<RagNodeData>[] = graph.nodes.map((n) => ({
    id: n.id,
    type: 'ragNode',
    position: { x: n.x, y: n.y },
    data: { kind: n.kind, label: n.label, config: { ...n.config } },
  }));
  const edges: Edge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.source_id,
    target: e.target_id,
    sourceHandle: e.source_port,
    targetHandle: e.target_port,
    type: 'smoothstep',
  }));
  return { nodes, edges };
}

export function reactNodeToPipeline(n: Node<RagNodeData>): PipelineNode {
  return {
    id: n.id,
    kind: n.data.kind,
    label: n.data.label,
    x: n.position.x,
    y: n.position.y,
    config: { ...n.data.config },
  };
}

export function fromReactFlow(nodes: Node[], edges: Edge[]): PipelineGraph {
  const pipelineNodes: PipelineNode[] = nodes
    .filter((n): n is Node<RagNodeData> => n.type === 'ragNode')
    .map((n) => ({
      id: n.id,
      kind: n.data.kind,
      label: n.data.label,
      x: n.position.x,
      y: n.position.y,
      config: { ...n.data.config },
    }));
  const pipelineEdges: PipelineEdge[] = edges.map((e) => ({
    id: e.id,
    source_id: e.source,
    target_id: e.target,
    source_port: (e.sourceHandle as string) || HANDLE_OUT,
    target_port: (e.targetHandle as string) || HANDLE_IN,
  }));
  return { nodes: pipelineNodes, edges: pipelineEdges };
}
