import type {
  DocumentListResponse,
  ExecuteNodeResponse,
  GraphExecutionResult,
  PipelineConfig,
  PipelineGraph,
  PipelineNode,
  PreviewRetrievalResponse,
  ProcessedDocument,
  QueryRequest,
  QueryResponse,
  VisualizationData,
} from '../types/pipeline';

const BASE = 'http://localhost:8000/api';

function friendlyHttpError(status: number, detail: string): string {
  const trimmed = detail.trim().replace(/\s+/g, ' ');
  const short = trimmed.length > 400 ? `${trimmed.slice(0, 400)}…` : trimmed;
  if (status === 401 || status === 403) {
    return short || 'Unauthorized — check your OpenRouter API key.';
  }
  if (status === 429) return short || 'Rate limited — try again shortly.';
  if (status === 502 || status === 503 || status === 504) {
    return short || 'Service temporarily unavailable — try again later.';
  }
  return short || `HTTP ${status}`;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (j?.detail !== undefined) detail = String(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(friendlyHttpError(res.status, detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  listDocuments(): Promise<DocumentListResponse> {
    return fetch(`${BASE}/documents`).then((r) => json<DocumentListResponse>(r));
  },

  getDocument(id: string): Promise<ProcessedDocument> {
    return fetch(`${BASE}/documents/${id}`).then((r) => json<ProcessedDocument>(r));
  },

  async deleteDocument(id: string): Promise<void> {
    const r = await fetch(`${BASE}/documents/${id}`, { method: 'DELETE' });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || `HTTP ${r.status}`);
    }
  },

  executeGraph(graph: PipelineGraph): Promise<GraphExecutionResult> {
    return fetch(`${BASE}/pipeline/execute-graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graph),
    }).then((r) => json<GraphExecutionResult>(r));
  },

  executeNode(node: PipelineNode, inputs: Record<string, unknown> = {}): Promise<ExecuteNodeResponse> {
    return fetch(`${BASE}/pipeline/execute-node`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node, inputs }),
    }).then((r) => json<ExecuteNodeResponse>(r));
  },

  previewRetrieval(
    query: string,
    topK: number,
    selectedDocuments?: string[],
  ): Promise<PreviewRetrievalResponse> {
    return fetch(`${BASE}/pipeline/preview-retrieval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        selected_documents:
          selectedDocuments && selectedDocuments.length > 0 ? selectedDocuments : undefined,
      }),
    }).then((r) => json<PreviewRetrievalResponse>(r));
  },

  getPipelineConfig(): Promise<PipelineConfig> {
    return fetch(`${BASE}/pipeline/config`).then((r) => json<PipelineConfig>(r));
  },

  setPipelineConfig(config: PipelineConfig): Promise<PipelineConfig> {
    return fetch(`${BASE}/pipeline/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }).then((r) => json<PipelineConfig>(r));
  },

  /** Set API key on server without triggering index rebuild (see plan.md Phase 1). */
  setCredentials(openrouterApiKey: string | null): Promise<PipelineConfig> {
    return fetch(`${BASE}/pipeline/credentials`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ openrouter_api_key: openrouterApiKey ?? '' }),
    }).then((r) => json<PipelineConfig>(r));
  },

  /** Verify OpenRouter key; does not persist it beyond optional server in-memory config. */
  verifyOpenRouterKey(apiKey: string): Promise<{ ok: boolean }> {
    return fetch(`${BASE}/pipeline/verify-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    }).then((r) => json<{ ok: boolean }>(r));
  },

  buildIndex(): Promise<unknown> {
    return fetch(`${BASE}/pipeline/build-index`, { method: 'POST' }).then((r) => json(r));
  },

  /** Full RAG query (includes optional 3D visualization from PCA). */
  query(body: QueryRequest): Promise<QueryResponse> {
    return fetch(`${BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<QueryResponse>(r));
  },

  /** Corpus-only 3D projection (no query neighbors). */
  getEmbeddingSpace(params?: {
    reduction_method?: 'pca' | 'umap';
    selected_documents?: string[];
  }): Promise<VisualizationData> {
    const q = new URLSearchParams();
    if (params?.reduction_method) q.set('reduction_method', params.reduction_method);
    if (params?.selected_documents?.length) {
      q.set('selected_documents', params.selected_documents.join(','));
    }
    const qs = q.toString();
    return fetch(`${BASE}/pipeline/embedding-space${qs ? `?${qs}` : ''}`).then((r) =>
      json<VisualizationData>(r),
    );
  },
};

export function uploadDocumentStream(
  file: File,
  chunkStrategy: string,
  chunkSize: number,
  chunkOverlap: number,
  onEvent: (evt: { progress?: number; message?: string; step?: string }) => void,
): Promise<void> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('chunk_strategy', chunkStrategy);
  formData.append('chunk_size', String(chunkSize));
  formData.append('chunk_overlap', String(chunkOverlap));

  return fetch(`${BASE}/documents/upload-stream`, {
    method: 'POST',
    body: formData,
  }).then(async (response) => {
    if (!response.ok || !response.body) {
      throw new Error(`Upload failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6)) as Record<string, unknown>;
          onEvent({
            progress: evt.progress as number | undefined,
            message: evt.message as string | undefined,
            step: evt.step as string | undefined,
          });
        } catch {
          /* ignore */
        }
      }
    }
  });
}
