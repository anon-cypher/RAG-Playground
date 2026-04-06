import { useCallback, useEffect, useMemo, useState } from 'react';
import { ReactFlowProvider, useEdgesState, useNodesState, useReactFlow } from '@xyflow/react';
import type { Node } from '@xyflow/react';

import { api } from './api/client';
import { Palette } from './components/Palette';
import { Inspector } from './components/Inspector';
import { ExecutionOutputs } from './components/ExecutionOutputs';
import { FlowCanvas } from './graph/FlowCanvas';
import type { RagNodeData } from './graph/adapters';
import { fromReactFlow } from './graph/adapters';
import { createBootstrapReactFlow, NODE_TYPES } from './graph/nodeCatalog';
import type { DocumentSummary, GraphExecutionResult, NodeKind } from './types/pipeline';

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

  const runGraph = useCallback(() => {
    setRunErr(null);
    setRunBusy(true);
    const graph = fromReactFlow(nodes, edges);
    api
      .executeGraph(graph)
      .then((r) => {
        setExec(r);
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
  }, [nodes, edges]);

  const deleteSelected = useCallback(() => {
    if (!selectedId) return;
    setNodes((nds) => nds.filter((n) => n.id !== selectedId));
    setEdges((eds) => eds.filter((e) => e.source !== selectedId && e.target !== selectedId));
    setSelectedId(null);
  }, [selectedId, setNodes, setEdges]);

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>RAG Visual Playground</h1>
          <p className="app__sub">v2 workflow canvas · XYFlow</p>
        </div>
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="app__link">
          Backend API
        </a>
      </header>

      <div className="app__workspace">
        <Palette onDragStart={onPaletteDragStart} onAddClick={onAddClick} />

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
            {runErr && <p className="inspector__error">{runErr}</p>}
            {exec && (
              <>
                <div className="run-panel__order">
                  <strong>Topological order</strong>
                  <div className="run-panel__order-chain">{exec.execution_order.join(' → ')}</div>
                </div>
                <ExecutionOutputs result={exec} />
              </>
            )}
          </div>
        </div>
      </div>
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
