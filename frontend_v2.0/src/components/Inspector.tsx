import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Node } from '@xyflow/react';
import type { RagNodeData } from '../graph/adapters';
import { api, uploadDocumentStream } from '../api/client';
import type { DocumentSummary, PipelineConfig } from '../types/pipeline';
import { previewChunks, countBoundaryOverlapTokens } from '../utils/chunk-preview';

type InspectorProps = {
  selected: Node<RagNodeData> | null;
  allNodes: Node<RagNodeData>[];
  setNodes: React.Dispatch<React.SetStateAction<Node<RagNodeData>[]>>;
  documents: DocumentSummary[];
  onRefreshDocuments: () => void;
  selectedDocumentIds: Set<string>;
  toggleDocumentSelection: (id: string) => void;
};

function patchData(
  setNodes: React.Dispatch<React.SetStateAction<Node<RagNodeData>[]>>,
  id: string,
  fn: (d: RagNodeData) => RagNodeData,
) {
  setNodes((nds) =>
    nds.map((n) => (n.id === id ? { ...n, data: fn(n.data as RagNodeData) } : n)),
  );
}

export function Inspector({
  selected,
  allNodes,
  setNodes,
  documents,
  onRefreshDocuments,
  selectedDocumentIds,
  toggleDocumentSelection,
}: InspectorProps) {
  const [runError, setRunError] = useState<string | null>(null);

  const chunkerParams = useMemo(() => {
    const c = allNodes.find((n) => n.data.kind === 'chunker');
    const cfg = (c?.data.config ?? {}) as Record<string, unknown>;
    return {
      strategy: String(cfg['chunk_strategy'] ?? 'overlapping'),
      size: Number(cfg['chunk_size'] ?? 500),
      overlap: Number(cfg['overlap'] ?? 50),
    };
  }, [allNodes]);

  if (!selected) {
    return (
      <aside className="inspector">
        <h3 className="inspector__title">Node config</h3>
        <p className="inspector__hint">Select a node on the canvas.</p>
      </aside>
    );
  }

  const id = selected.id;
  const data = selected.data as RagNodeData;
  const cfg = data.config;

  const updateConfig = (key: string, value: unknown) => {
    patchData(setNodes, id, (d) => ({
      ...d,
      config: { ...d.config, [key]: value },
    }));
  };

  return (
    <aside className="inspector">
      <h3 className="inspector__title">Node config</h3>
      <div className="inspector__meta">
        <span className="inspector__label">{data.label}</span>
        <span className="inspector__kind">{data.kind}</span>
      </div>

      {data.kind === 'document_loader' && (
        <DocumentLoaderPanel
          chunkerParams={chunkerParams}
          documents={documents}
          onRefreshDocuments={onRefreshDocuments}
          selectedDocumentIds={selectedDocumentIds}
          toggleDocumentSelection={toggleDocumentSelection}
        />
      )}

      {data.kind === 'chunker' && (
        <ChunkerPanel
          nodeId={id}
          cfg={cfg}
          updateConfig={updateConfig}
          documents={documents}
        />
      )}

      {data.kind === 'embedder' && (
        <EmbedderPanel cfg={cfg} updateConfig={updateConfig} setRunError={setRunError} />
      )}

      {data.kind !== 'document_loader' && data.kind !== 'chunker' && data.kind !== 'embedder' && (
        <GenericConfig cfg={cfg} updateConfig={updateConfig} />
      )}

      {runError && <p className="inspector__error">{runError}</p>}
    </aside>
  );
}

function DocumentLoaderPanel({
  chunkerParams,
  documents,
  onRefreshDocuments,
  selectedDocumentIds,
  toggleDocumentSelection,
}: {
  chunkerParams: { strategy: string; size: number; overlap: number };
  documents: DocumentSummary[];
  onRefreshDocuments: () => void;
  selectedDocumentIds: Set<string>;
  toggleDocumentSelection: (id: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [err, setErr] = useState<string | null>(null);

  const upload = () => {
    if (!file) return;
    setErr(null);
    setUploading(true);
    setProgress(0);
    setMessage('Starting…');
    uploadDocumentStream(
      file,
      chunkerParams.strategy,
      chunkerParams.size,
      chunkerParams.overlap,
      (evt) => {
        if (evt.progress !== undefined) setProgress(evt.progress);
        if (evt.message) setMessage(evt.message);
        if (evt.step === 'done') {
          setUploading(false);
          setFile(null);
          onRefreshDocuments();
        }
        if (evt.step === 'error') {
          setUploading(false);
          setErr(evt.message || 'Upload failed');
        }
      },
    ).catch((e: Error) => {
      setUploading(false);
      setErr(e.message);
    });
  };

  return (
    <>
      <div className="field">
        <label>Upload</label>
        <input
          type="file"
          accept=".txt,.pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="button" className="btn btn--primary" disabled={!file || uploading} onClick={upload}>
          {uploading ? 'Uploading…' : 'Upload'}
        </button>
      </div>
      {(uploading || progress > 0) && (
        <div className="progress">
          <div className="progress__bar" style={{ width: `${progress}%` }} />
          <div className="progress__txt">
            {progress}% · {message}
          </div>
        </div>
      )}
      {err && <p className="inspector__error">{err}</p>}
      <div className="doc-list">
        {documents.map((doc) => (
          <div key={doc.document_id} className="doc-row">
            <input
              type="checkbox"
              checked={selectedDocumentIds.has(doc.document_id)}
              onChange={() => toggleDocumentSelection(doc.document_id)}
            />
            <span className="doc-row__name">{doc.filename}</span>
            <button
              type="button"
              className="btn btn--danger"
              onClick={() =>
                api.deleteDocument(doc.document_id).then(onRefreshDocuments).catch(() => {})
              }
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </>
  );
}

const CHUNK_SIZE_SLIDER = { min: 100, max: 2000, step: 50 };
const OVERLAP_SLIDER = { min: 0, max: 400, step: 10 };

function ChunkerPanel({
  nodeId,
  cfg,
  updateConfig,
  documents,
}: {
  nodeId: string;
  cfg: Record<string, unknown>;
  updateConfig: (k: string, v: unknown) => void;
  documents: DocumentSummary[];
}) {
  const strategy = String(cfg['chunk_strategy'] ?? 'overlapping');
  const chunkSize = Number(cfg['chunk_size'] ?? 500);
  const overlap = Number(cfg['overlap'] ?? 50);

  const [previewDocId, setPreviewDocId] = useState<string | null>(documents[0]?.document_id ?? null);
  const [rawText, setRawText] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!previewDocId) {
      setRawText('');
      return;
    }
    setLoading(true);
    api
      .getDocument(previewDocId)
      .then((doc) => {
        const raw = doc.raw_text ?? doc.chunks?.map((c) => c.text).join('\n\n') ?? '';
        setRawText(raw);
      })
      .catch(() => setRawText(''))
      .finally(() => setLoading(false));
  }, [previewDocId]);

  const chunks = useMemo(() => {
    if (!rawText) return [];
    return previewChunks(
      rawText,
      strategy === 'fixed' ? 'fixed' : 'overlapping',
      chunkSize,
      overlap,
    );
  }, [rawText, strategy, chunkSize, overlap]);

  const overlapTokens = useCallback((ci: number) => {
    const cur = chunks[ci];
    const next = chunks[ci + 1];
    if (!cur) return [];
    const tokens = cur.split(/\s+/).filter(Boolean);
    if (!next) return tokens.map((t) => ({ text: t, overlap: false }));
    const n = countBoundaryOverlapTokens(cur, next);
    const start = tokens.length - n;
    return tokens.map((t, i) => ({ text: t, overlap: n > 0 && i >= start }));
  }, [chunks]);

  return (
    <>
      <div className="field">
        <label>Chunk strategy</label>
        <select
          value={strategy}
          onChange={(e) => updateConfig('chunk_strategy', e.target.value)}
        >
          <option value="fixed">fixed</option>
          <option value="overlapping">overlapping</option>
        </select>
      </div>
      <div className="field">
        <label>Preview document</label>
        <select
          value={previewDocId ?? ''}
          onChange={(e) => setPreviewDocId(e.target.value || null)}
        >
          <option value="">— none —</option>
          {documents.map((d) => (
            <option key={d.document_id} value={d.document_id}>
              {d.filename}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>Chunk size · {chunkSize}</label>
        <input
          type="range"
          min={CHUNK_SIZE_SLIDER.min}
          max={CHUNK_SIZE_SLIDER.max}
          step={CHUNK_SIZE_SLIDER.step}
          value={chunkSize}
          onChange={(e) => updateConfig('chunk_size', +e.target.value)}
        />
      </div>
      <div className="field">
        <label>Overlap · {overlap}</label>
        <input
          type="range"
          min={OVERLAP_SLIDER.min}
          max={OVERLAP_SLIDER.max}
          step={OVERLAP_SLIDER.step}
          value={overlap}
          onChange={(e) => updateConfig('overlap', +e.target.value)}
        />
      </div>
      {loading && <p className="inspector__hint">Loading preview…</p>}
      {!loading && rawText && chunks.length > 0 && (
        <div className="chunkviz">
          {chunks.map((_, ci) => (
            <div key={`${nodeId}-c-${ci}`} className="chunkviz__block">
              <div className="chunkviz__head">Chunk {ci + 1}</div>
              <div className="chunkviz__tokens">
                {overlapTokens(ci).map((tok, ti) => (
                  <span key={ti} className={tok.overlap ? 'chunkviz__tok chunkviz__tok--overlap' : 'chunkviz__tok'}>
                    {tok.text}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function EmbedderPanel({
  cfg,
  updateConfig,
  setRunError,
}: {
  cfg: Record<string, unknown>;
  updateConfig: (k: string, v: unknown) => void;
  setRunError: (s: string | null) => void;
}) {
  const [busy, setBusy] = useState(false);

  const build = () => {
    setRunError(null);
    setBusy(true);
    api
      .getPipelineConfig()
      .then((base: PipelineConfig) => {
        const merged: PipelineConfig = {
          ...base,
          openrouter_api_key: String(cfg['openrouter_api_key'] ?? '').trim() || base.openrouter_api_key,
          embedding_model: (cfg['embedding_model'] as string) || base.embedding_model,
          embedding_dim: Number(cfg['embedding_dim'] ?? base.embedding_dim),
        };
        return api.setPipelineConfig(merged).then(() => api.buildIndex());
      })
      .then(() => setBusy(false))
      .catch((e: Error) => {
        setBusy(false);
        setRunError(e.message);
      });
  };

  return (
    <>
      <p className="inspector__hint">Configure embeddings and build the FAISS index.</p>
      <div className="field">
        <label>OpenRouter API key</label>
        <input
          type="password"
          autoComplete="off"
          value={String(cfg['openrouter_api_key'] ?? '')}
          onChange={(e) => updateConfig('openrouter_api_key', e.target.value)}
        />
      </div>
      <div className="field">
        <label>Embedding model</label>
        <input
          type="text"
          value={String(cfg['embedding_model'] ?? 'qwen/qwen3-embedding-8b')}
          onChange={(e) => updateConfig('embedding_model', e.target.value)}
        />
      </div>
      <div className="field">
        <label>Embedding dim</label>
        <input
          type="number"
          value={Number(cfg['embedding_dim'] ?? 4096)}
          onChange={(e) => updateConfig('embedding_dim', +e.target.value)}
        />
      </div>
      <button type="button" className="btn btn--primary" disabled={busy} onClick={build}>
        {busy ? 'Building…' : 'Build embeddings & index'}
      </button>
    </>
  );
}

function GenericConfig({
  cfg,
  updateConfig,
}: {
  cfg: Record<string, unknown>;
  updateConfig: (k: string, v: unknown) => void;
}) {
  const entries = Object.entries(cfg);
  if (entries.length === 0) {
    return <p className="inspector__hint">No configurable fields.</p>;
  }
  return (
    <>
      {entries.map(([key, value]) => (
        <div key={key} className="field">
          <label>{key}</label>
          {typeof value === 'number' ? (
            <input
              type="number"
              value={value}
              onChange={(e) => updateConfig(key, +e.target.value)}
            />
          ) : typeof value === 'boolean' ? (
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => updateConfig(key, e.target.checked)}
            />
          ) : (
            <input
              type="text"
              value={String(value ?? '')}
              onChange={(e) => updateConfig(key, e.target.value)}
            />
          )}
        </div>
      ))}
    </>
  );
}
