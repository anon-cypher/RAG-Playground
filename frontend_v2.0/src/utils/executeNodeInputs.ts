import type { Node } from '@xyflow/react';
import type { RagNodeData } from '../graph/adapters';

/** Build optional inputs for POST /execute-node from canvas + last run cache. */
export function buildExecuteNodeInputs(
  _target: Node<RagNodeData>,
  allNodes: Node[],
  outputCache: Record<string, Record<string, unknown>>,
): Record<string, unknown> {
  const inputs: Record<string, unknown> = {};
  const rag = allNodes.filter((n): n is Node<RagNodeData> => n.type === 'ragNode');

  const qNode = rag.find((n) => n.data.kind === 'query');
  if (qNode) {
    inputs['query_text'] = String(qNode.data.config['query_text'] ?? '');
  }

  const retrieverNode = rag.find((n) => n.data.kind === 'retriever');
  const retOut = retrieverNode ? outputCache[retrieverNode.id] : undefined;
  const chunks = retOut?.['chunks'] as Array<{ text_preview?: string }> | undefined;
  if (chunks?.length) {
    inputs['context_texts'] = chunks.map((c) => String(c.text_preview ?? ''));
  }

  const augNode = rag.find((n) => n.data.kind === 'prompt_augment');
  if (augNode && outputCache[augNode.id]?.['assembled_prompt']) {
    inputs['assembled_prompt'] = outputCache[augNode.id]['assembled_prompt'];
  }

  return inputs;
}
