import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ReactFlowProvider, useEdgesState, useNodesState, useReactFlow } from '@xyflow/react';
import type { Node } from '@xyflow/react';

import { api } from './api/client';
import { Palette } from './components/Palette';
import { Inspector } from './components/Inspector';
import { ExecutionOutputs } from './components/ExecutionOutputs';
import { SettingsBar } from './components/SettingsBar';
import { GlossaryModal } from './components/GlossaryModal';
import { PipelineOrderHint } from './components/PipelineOrderHint';
import { EmbeddingSpace3D, type VizSelection } from './components/EmbeddingSpace3D';
import { FlowCanvas } from './graph/FlowCanvas';
import type { RagNodeData } from './graph/adapters';
import { fromReactFlow } from './graph/adapters';
import { createBootstrapReactFlow, NODE_TYPES } from './graph/nodeCatalog';
import type {
  DocumentSummary,
  GraphExecutionResult,
  NodeKind,
  VisualizationData,
} from './types/pipeline';
import { extractQueryRetrievalParams } from './utils/extractQueryParams';
import { buildRunExportPayload, downloadJson } from './utils/exportRun';
import { loadStoredApiKey } from './utils/openrouterStorage';

import './App.css';

function nextId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function AppContent() {
  const initial = useMemo(() => createBootstrapReactFlow(), []);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<Set<string>>(() => new Set());
  const [exec, setExec] = useState<GraphExecutionResult | null>(null);
  const [runErr, setRunErr] = useState<string | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [nodeOutputCache, setNodeOutputCache] = useState<Record<string, Record<string, unknown>>>({});
  const [vizData, setVizData] = useState<VisualizationData | null>(null);
  const [vizErr, setVizErr] = useState<string | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizSelection, setVizSelection] = useState<VizSelection>(null);
  const [runAt, setRunAt] = useState<string | null>(null);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [hasStoredKey, setHasStoredKey] = useState(() => Boolean(loadStoredApiKey().trim()));
  const runTimersRef = useRef<number[]>([]);

  const onNodeOutput = useCallback((nodeId: string, summary: Record<string, unknown>) => {
    setNodeOutputCache((p) => ({ ...p, [nodeId]: summary }));
  }, []);

  const { screenToFlowPosition } = useReactFlow();

  const refreshDocuments = useCallback(() => {
    api.listDocuments().then((r) => setDocuments(r.documents)).catch(() => setDocuments([]));
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  const toggleDocumentSelection = useCallback((id: string) => {
    setSelectedDocumentIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectedNode = useMemo(
    () =>
      selectedId
        ? ((nodes.find((n) => n.id === selectedId && n.type === 'ragNode') as Node<RagNodeData> | undefined) ??
          null)
        : null,
    [nodes, selectedId],
  );

  const onPaletteDragStart = useCallback((kind: NodeKind, e: React.DragEvent) => {
    e.dataTransfer.setData('application/rag-kind', kind);
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const addNodeAt = useCallback(
    (kind: NodeKind, position: { x: number; y: number }) => {
      const def = NODE_TYPES.find((t) => t.kind === kind);
      if (!def) return;
      const id = nextId(kind);
      const newNode: Node<RagNodeData> = {
        id,
        type: 'ragNode',
        position,
        data: {
          kind,
          label: def.label,
          config: { ...def.defaultConfig },
          runState: 'idle',
        },
      };
      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const kind = e.dataTransfer.getData('application/rag-kind') as NodeKind;
      if (!kind) return;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addNodeAt(kind, position);
    },
    [screenToFlowPosition, addNodeAt],
  );

  const onAddClick = useCallback(
    (kind: NodeKind) => {
      const position = screenToFlowPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
      });
      addNodeAt(kind, position);
    },
    [screenToFlowPosition, addNodeAt],
  );

  const onSelectionChange = useCallback((ids: { nodes: string[]; edges: string[] }) => {
    setSelectedId(ids.nodes[0] ?? null);
  }, []);

  const clearRunTimers = useCallback(() => {
    for (const t of runTimersRef.current) window.clearTimeout(t);
    runTimersRef.current = [];
  }, []);

  const setAllNodeRunState = useCallback(
    (state: RagNodeData['runState']) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.type === 'ragNode'
            ? {
                ...n,
                data: { ...(n.data as RagNodeData), runState: state },
              }
            : n,
        ),
      );
    },
    [setNodes],
  );

  const setSingleNodeRunState = useCallback(
    (nodeId: string, state: RagNodeData['runState']) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.type === 'ragNode' && n.id === nodeId
            ? {
                ...n,
                data: { ...(n.data as RagNodeData), runState: state },
              }
            : n,
        ),
      );
    },
    [setNodes],
  );

  const animateRunExecution = useCallback(
    (order: string[]) => {
      clearRunTimers();
      setAllNodeRunState('idle');
      const stepMs = 460;
      const runningSliceMs = 260;
      order.forEach((nodeId, idx) => {
        const tRun = window.setTimeout(() => {
          setSingleNodeRunState(nodeId, 'running');
        }, idx * stepMs);
        const tDone = window.setTimeout(() => {
          setSingleNodeRunState(nodeId, 'done');
        }, idx * stepMs + runningSliceMs);
        runTimersRef.current.push(tRun, tDone);
      });
    },
    [clearRunTimers, setAllNodeRunState, setSingleNodeRunState],
  );

  const runGraph = useCallback(() => {
    setRunErr(null);
    setRunBusy(true);
    clearRunTimers();
    setAllNodeRunState('idle');
    const graph = fromReactFlow(nodes, edges);
    api
      .executeGraph(graph)
      .then((r) => {
        setExec(r);
        setRunAt(new Date().toISOString());
        animateRunExecution(r.execution_order);
        setNodeOutputCache((prev) => {
          const next = { ...prev };
          for (const nr of r.node_results) {
            next[nr.node_id] = nr.output_summary as Record<string, unknown>;
          }
          return next;
        });
        setRunBusy(false);
      })
      .catch((e: Error) => {
        setRunErr(e.message);
        setRunBusy(false);
      });
  }, [nodes, edges, clearRunTimers, setAllNodeRunState, animateRunExecution]);

  useEffect(() => {
    return () => clearRunTimers();
  }, [clearRunTimers]);

  const deleteSelected = useCallback(() => {
    if (!selectedId) return;
    setNodes((nds) => nds.filter((n) => n.id !== selectedId));
    setEdges((eds) => eds.filter((e) => e.source !== selectedId && e.target !== selectedId));
    setSelectedId(null);
  }, [selectedId, setNodes, setEdges]);

  const loadViz3dNeighbors = useCallback(() => {
    const { queryText, topK, selected_documents } = extractQueryRetrievalParams(
      nodes,
      selectedDocumentIds,
    );
    if (!queryText.trim()) {
      setVizErr('Add text to the Query node before loading the 3D view.');
      return;
    }
    setVizErr(null);
    setVizLoading(true);
    api
      .query({
        query: queryText,
        top_k: topK,
        selected_documents,
        visualization_only: true,
      })
      .then((r) => {
        setVizData(r.visualization);
        setVizSelection(null);
      })
      .catch((e: Error) => {
        setVizErr(e.message);
        setVizData(null);
      })
      .finally(() => setVizLoading(false));
  }, [nodes, selectedDocumentIds]);

  const loadCorpusEmbeddingSpace = useCallback(() => {
    setVizErr(null);
    setVizLoading(true);
    const selected_documents =
      selectedDocumentIds.size > 0 ? Array.from(selectedDocumentIds) : undefined;
    api
      .getEmbeddingSpace({ reduction_method: 'pca', selected_documents })
      .then((v) => {
        setVizData(v);
        setVizSelection(null);
      })
      .catch((e: Error) => {
        setVizErr(e.message);
        setVizData(null);
      })
      .finally(() => setVizLoading(false));
  }, [selectedDocumentIds]);

  const exportLastRun = useCallback(() => {
    if (!exec) return;
    const graph = fromReactFlow(nodes, edges);
    const ts = new Date().toISOString();
    api
      .getPipelineConfig()
      .then((cfg) => {
        downloadJson(
          `rag-run-${Date.now()}.json`,
          buildRunExportPayload({
            exportedAt: ts,
            graph,
            result: exec,
            pipelineConfigSnapshot: cfg as unknown as Record<string, unknown>,
          }),
        );
      })
      .catch(() => {
        downloadJson(
          `rag-run-${Date.now()}.json`,
          buildRunExportPayload({ exportedAt: ts, graph, result: exec }),
        );
      });
  }, [exec, nodes, edges]);

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>RAG Visual Playground</h1>
          <p className="app__sub">v2 workflow canvas · OpenRouter · learn &amp; experiment</p>
        </div>
        <div className="app__header__actions">
          <button type="button" className="btn btn--ghost" onClick={() => setGlossaryOpen(true)}>
            Glossary
          </button>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="app__link">
            Backend API
          </a>
        </div>
      </header>

      <section className="app__settings" aria-label="OpenRouter credentials">
        <SettingsBar
          onCredentialsSynced={() => {
            setHasStoredKey(Boolean(loadStoredApiKey().trim()));
          }}
        />
      </section>

      {!hasStoredKey && (
        <div className="app__banner app__banner--warn" role="status">
          Add your <strong>OpenRouter API key</strong> below for embeddings and LLM. Keys stay in this browser;
          use <strong>Save key</strong> to sync to the server session.
        </div>
      )}

      {documents.length === 0 && (
        <div className="app__banner app__banner--info" role="status">
          <strong>No documents yet.</strong> Select the <strong>Document Loader</strong> node, upload a file, then
          configure the <strong>Embedder</strong> and build the index.
        </div>
      )}

      <GlossaryModal open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />

      <div className="app__workspace">
        <div className="app__palette-col">
          <Palette onDragStart={onPaletteDragStart} onAddClick={onAddClick} />
          <PipelineOrderHint />
        </div>

        <div className="app__canvas">
          <FlowCanvas
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            setEdges={setEdges}
            onDrop={onDrop}
            onSelectionChange={onSelectionChange}
          />
        </div>

        <div className="app__right">
          <Inspector
            selected={selectedNode}
            allNodes={nodes}
            setNodes={setNodes}
            documents={documents}
            onRefreshDocuments={refreshDocuments}
            selectedDocumentIds={selectedDocumentIds}
            toggleDocumentSelection={toggleDocumentSelection}
            nodeOutputCache={nodeOutputCache}
            onNodeOutput={onNodeOutput}
          />

          <div className="run-panel">
            <button type="button" className="btn btn--run" disabled={runBusy} onClick={runGraph}>
              {runBusy ? 'Running…' : 'Run pipeline'}
            </button>
            <button type="button" className="btn" onClick={deleteSelected} disabled={!selectedId}>
              Delete node
            </button>
            <button
              type="button"
              className="btn"
              disabled={!exec}
              onClick={exportLastRun}
              aria-label="Export last pipeline run as JSON"
            >
              Export run (JSON)
            </button>
            {runErr && <p className="inspector__error">{runErr}</p>}
            {exec && (
              <>
                <div className="run-panel__order">
                  <strong>Topological order</strong>
                  <div className="run-panel__order-chain">{exec.execution_order.join(' → ')}</div>
                </div>
                <ExecutionOutputs result={exec} runAt={runAt} />
              </>
            )}
          </div>
        </div>
      </div>

      <section className="app__embedding" aria-label="Embedding space visualization">
        <div className="app__embedding__head">
          <h2 className="app__embedding__title">3D embedding view</h2>
          <div className="app__embedding__actions">
            <button
              type="button"
              className="btn btn--primary"
              disabled={vizLoading}
              onClick={loadViz3dNeighbors}
            >
              {vizLoading ? 'Loading…' : 'Query neighbors (PCA)'}
            </button>
            <button
              type="button"
              className="btn"
              disabled={vizLoading}
              onClick={loadCorpusEmbeddingSpace}
            >
              Corpus only
            </button>
          </div>
        </div>
        <p className="app__embedding__note">
          Neighbors mode uses your Query + Retriever <code>top_k</code> and optional document checkboxes.
          Corpus mode projects all chunks without a query point. Requires a built index and embeddings.
        </p>
        {vizErr && <p className="inspector__error">{vizErr}</p>}
        <div className="app__embedding__body">
          <div className="app__embedding__canvas">
            <EmbeddingSpace3D data={vizData} onSelect={setVizSelection} />
          </div>
          <aside className="app__embedding__detail">
            {vizData && (
              <>
                <div className="app__embedding__caption">
                  <strong>{vizData.rag_mode || 'embedding'}</strong>
                  {vizData.caption && <p>{vizData.caption}</p>}
                </div>
                {vizSelection?.type === 'query' && vizData.query_point && (
                  <div className="app__embedding__pick">
                    <h3>Query</h3>
                    <p>PCA position of the embedded query vector (gold).</p>
                  </div>
                )}
                {vizSelection?.type === 'chunk' && vizData.points[vizSelection.index] && (
                  <div className="app__embedding__pick">
                    <h3>Chunk</h3>
                    <p className="app__embedding__meta">
                      <code>{vizData.points[vizSelection.index].chunk_id}</code>
                      {vizData.points[vizSelection.index].is_neighbor && (
                        <span className="app__embedding__badge">neighbor</span>
                      )}
                    </p>
                    <p className="app__embedding__score">
                      score: {vizData.points[vizSelection.index].score}
                    </p>
                    <p className="app__embedding__text">{vizData.points[vizSelection.index].text_preview}</p>
                  </div>
                )}
                {!vizSelection && (
                  <p className="inspector__hint">Click a point in the scene for details.</p>
                )}
              </>
            )}
          </aside>
        </div>
      </section>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppContent />
    </ReactFlowProvider>
  );
}
