import type { GraphExecutionResult } from '../types/pipeline';
import { approxTokensFromChars } from '../utils/tokenEstimate';

type RunNarrativeProps = {
  result: GraphExecutionResult | null;
  runAt: string | null;
  onJumpToNode?: (nodeId: string) => void;
};

export function RunNarrative({ result, runAt, onJumpToNode }: RunNarrativeProps) {
  if (!result?.node_results?.length) return null;

  let retrieved = 0;
  let approxPromptTok: number | null = null;
  let genUsageTotal: number | null = null;

  for (const nr of result.node_results) {
    const s = nr.output_summary as Record<string, unknown>;
    if (String(s.viz) === 'retrieved_chunks' && Array.isArray(s.chunks)) {
      retrieved = Math.max(retrieved, (s.chunks as unknown[]).length);
    }
    if (String(s.viz) === 'prompt_augment') {
      const at = s.approx_input_tokens;
      if (at != null) approxPromptTok = Number(at);
      else if (s.char_count != null) approxPromptTok = approxTokensFromChars(Number(s.char_count));
    }
    if (String(s.viz) === 'llm_answer' && s.usage && typeof s.usage === 'object') {
      const u = s.usage as Record<string, unknown>;
      if (u.total_tokens != null) genUsageTotal = Number(u.total_tokens);
    }
  }

  const totalMs = result.node_results.reduce((a, b) => a + b.latency_ms, 0);

  return (
    <div className="run-narrative">
      <h4 className="run-narrative__title">Run narrative (teaching)</h4>
      <p className="run-narrative__sub">
        Ordered stages with timings. Approximate counts help explain retrieval → prompt → generation.
        {runAt && (
          <>
            {' '}
            <time dateTime={runAt}>{runAt}</time>
          </>
        )}
      </p>
      <ul className="run-narrative__stages">
        {result.node_results.map((nr) => (
          <li key={nr.node_id} className="run-narrative__stage">
            <button
              type="button"
              className="run-narrative__jump"
              onClick={() => onJumpToNode?.(nr.node_id)}
              disabled={!onJumpToNode}
              title={onJumpToNode ? 'Scroll to this node output' : undefined}
            >
              {nr.kind}
            </button>
            <span className="run-narrative__label">
              {String((nr.output_summary as Record<string, unknown>)?.label ?? nr.node_id)}
            </span>
            <span className="run-narrative__ms">{nr.latency_ms.toFixed(1)} ms</span>
          </li>
        ))}
      </ul>
      <dl className="run-narrative__kpis">
        <div>
          <dt>Wall time (sum of node latencies)</dt>
          <dd>{totalMs.toFixed(1)} ms</dd>
        </div>
        <div>
          <dt>Retriever hits (last list length)</dt>
          <dd>{retrieved}</dd>
        </div>
        <div>
          <dt>Approx prompt tokens (augment node)</dt>
          <dd>{approxPromptTok != null ? approxPromptTok.toLocaleString() : '—'}</dd>
        </div>
        <div>
          <dt>LLM usage.total_tokens (if reported)</dt>
          <dd>{genUsageTotal != null ? genUsageTotal.toLocaleString() : '—'}</dd>
        </div>
      </dl>
    </div>
  );
}
