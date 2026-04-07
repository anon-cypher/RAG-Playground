import type { GraphExecutionResult, PipelineGraph } from '../types/pipeline';

/** Remove secrets from exported artifacts. */
export function redactConfig(config: Record<string, unknown>): Record<string, unknown> {
  const c = { ...config };
  if ('openrouter_api_key' in c && typeof c['openrouter_api_key'] === 'string') {
    const v = c['openrouter_api_key'] as string;
    c['openrouter_api_key'] = v ? '***redacted***' : '';
  }
  return c;
}

export function sanitizeGraphForExport(graph: PipelineGraph): PipelineGraph {
  return {
    nodes: graph.nodes.map((n) => ({
      ...n,
      config: redactConfig(n.config),
    })),
    edges: graph.edges,
  };
}

export function buildRunExportPayload(args: {
  exportedAt: string;
  graph: PipelineGraph;
  result: GraphExecutionResult;
  pipelineConfigSnapshot?: Record<string, unknown>;
}): Record<string, unknown> {
  return {
    exportedAt: args.exportedAt,
    version: 1,
    graph: sanitizeGraphForExport(args.graph),
    execution_order: args.result.execution_order,
    node_results: args.result.node_results.map((nr) => ({
      node_id: nr.node_id,
      kind: nr.kind,
      latency_ms: nr.latency_ms,
      output_summary: nr.output_summary,
    })),
    pipeline_config: args.pipelineConfigSnapshot
      ? redactConfig(args.pipelineConfigSnapshot as Record<string, unknown>)
      : undefined,
  };
}

export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
