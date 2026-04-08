import type { Node } from '@xyflow/react';
import type { RagNodeData } from '../graph/adapters';

/** Read query + retriever settings from the canvas for `/api/query` and 3D viz. */
export function extractQueryRetrievalParams(
  nodes: Node[],
  selectedDocumentIds: Set<string>,
): { queryText: string; topK: number; selected_documents: string[] | undefined } {
  const qNode = nodes.find((n) => n.type === 'ragNode' && n.data.kind === 'query');
  const rNode = nodes.find((n) => n.type === 'ragNode' && n.data.kind === 'retriever');
  const qd = qNode?.data as RagNodeData | undefined;
  const rd = rNode?.data as RagNodeData | undefined;
  const queryText = String(qd?.config?.['query_text'] ?? '');
  const topK = Number(rd?.config?.['top_k'] ?? 5);
  const selected_documents =
    selectedDocumentIds.size > 0 ? Array.from(selectedDocumentIds) : undefined;
  return { queryText, topK, selected_documents };
}
