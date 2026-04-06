import type { GraphExecutionResult, NodeExecutionResult } from '../types/pipeline';

type Summary = Record<string, unknown>;

export function ExecutionOutputs({ result }: { result: GraphExecutionResult | null }) {
  if (!result?.node_results?.length) return null;

  return (
    <div className="exec-outputs">
      <h4 className="exec-outputs__title">Pipeline output (per node)</h4>
      <p className="exec-outputs__sub">
        Results follow the graph order. Fix errors shown in red before expecting a full answer.
      </p>
      <div className="exec-outputs__list">
        {result.node_results.map((nr) => (
          <NodeOutputCard key={nr.node_id} nr={nr} />
        ))}
      </div>
    </div>
  );
}

function NodeOutputCard({ nr }: { nr: NodeExecutionResult }) {
  const s = nr.output_summary as Summary;
  const viz = String(s.viz ?? 'generic');
  const title = `${nr.kind} · ${String(s.label ?? nr.node_id)}`;

  return (
    <article className="node-out">
      <header className="node-out__head">
        <span className="node-out__title">{title}</span>
        <span className="node-out__latency">{nr.latency_ms.toFixed(1)} ms</span>
      </header>
      <div className="node-out__body">
        <NodeViz viz={viz} summary={s} />
      </div>
    </article>
  );
}

function NodeViz({ viz, summary }: { viz: string; summary: Summary }) {
  if (summary.error) {
    return <p className="node-out__err">{String(summary.error)}</p>;
  }

  switch (viz) {
    case 'documents_table':
      return <VizDocuments summary={summary} />;
    case 'chunk_stats':
      return <VizChunkStats summary={summary} />;
    case 'embedding_stats':
      return <VizEmbedding summary={summary} />;
    case 'index_stats':
      return <VizIndex summary={summary} />;
    case 'query_text':
      return <VizQuery summary={summary} />;
    case 'retrieved_chunks':
      return <VizRetrieved summary={summary} />;
    case 'vector_store':
      return <VizVectorStore summary={summary} />;
    case 'prompt_augment':
      return <VizPromptAugment summary={summary} />;
    case 'reranker':
      return <VizReranker summary={summary} />;
    case 'llm_answer':
      return <VizLlm summary={summary} />;
    case 'final_output':
      return <VizFinal summary={summary} />;
    default:
      return <VizGeneric summary={summary} />;
  }
}

function VizDocuments({ summary }: { summary: Summary }) {
  const docs = (summary.documents as Array<Record<string, unknown>>) ?? [];
  return (
    <div className="viz-table-wrap">
      <div className="viz-kpis">
        <span className="viz-kpi">
          <strong>{Number(summary.total_documents ?? 0)}</strong> docs
        </span>
        <span className="viz-kpi">
          <strong>{Number(summary.total_chunks ?? 0)}</strong> chunks
        </span>
      </div>
      <table className="viz-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Chunks</th>
            <th>Chars</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={String(d.document_id)}>
              <td>{String(d.filename)}</td>
              <td>{Number(d.chunks)}</td>
              <td>{Number(d.characters)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VizChunkStats({ summary }: { summary: Summary }) {
  return (
    <ul className="viz-list">
      <li>
        <strong>{Number(summary.total_chunks)}</strong> chunks in store
      </li>
      <li>
        <strong>{Number(summary.total_characters)}</strong> total characters
      </li>
      {summary.sample_preview ? (
        <li className="viz-preview">
          <span className="viz-muted">Sample</span>
          <p>{String(summary.sample_preview)}</p>
        </li>
      ) : null}
    </ul>
  );
}

function VizEmbedding({ summary }: { summary: Summary }) {
  return (
    <ul className="viz-list">
      <li>
        Model <code>{String(summary.embedding_model)}</code>
      </li>
      <li>
        Dim <strong>{Number(summary.embedding_dim)}</strong> · Vectors{' '}
        <strong>{Number(summary.total_vectors)}</strong>
      </li>
      <li className="viz-muted">Status: {String(summary.status)}</li>
      {summary.hint ? <li className="viz-hint">{String(summary.hint)}</li> : null}
    </ul>
  );
}

function VizIndex({ summary }: { summary: Summary }) {
  return (
    <ul className="viz-list">
      <li>
        Type <strong>{String(summary.index_type)}</strong>
      </li>
      <li>
        Vectors <strong>{Number(summary.total_vectors)}</strong> · Built{' '}
        <strong>{summary.is_built ? 'yes' : 'no'}</strong>
      </li>
    </ul>
  );
}

function VizVectorStore({ summary }: { summary: Summary }) {
  return (
    <ul className="viz-list">
      <li>
        Index <strong>{String(summary.index_type)}</strong> · vectors{' '}
        <strong>{Number(summary.total_vectors)}</strong> · chunks{' '}
        <strong>{Number(summary.total_chunks)}</strong>
      </li>
      <li>
        Built: <strong>{summary.is_built ? 'yes' : 'no'}</strong>
      </li>
      {Array.isArray(summary.sample_chunk_ids) && summary.sample_chunk_ids.length > 0 ? (
        <li className="viz-muted">Sample IDs: {(summary.sample_chunk_ids as string[]).slice(0, 6).join(', ')}</li>
      ) : null}
      {summary.note ? <li className="viz-hint">{String(summary.note)}</li> : null}
    </ul>
  );
}

function VizPromptAugment({ summary }: { summary: Summary }) {
  const ap = String(summary.assembled_prompt ?? '');
  return (
    <div className="viz-prompt-aug">
      <p className="viz-muted">{summary.char_count ? `${summary.char_count} characters` : ''}</p>
      <pre className="viz-prompt-aug__pre">{ap || '—'}</pre>
    </div>
  );
}

function VizQuery({ summary }: { summary: Summary }) {
  const q = String(summary.query_text ?? '');
  return (
    <div className="viz-query">
      <p className="viz-muted">{summary.char_count ? `${summary.char_count} characters` : 'Empty query'}</p>
      <blockquote className="viz-query__text">{q || '—'}</blockquote>
    </div>
  );
}

function VizRetrieved({ summary }: { summary: Summary }) {
  const chunks = (summary.chunks as Array<Record<string, unknown>>) ?? [];
  const maxScore = Math.max(0.001, ...chunks.map((c) => Number(c.score)));
  return (
    <div className="viz-retrieved">
      {summary.query_used ? (
        <p className="viz-muted">
          Query: <em>{String(summary.query_used)}</em> · top_k {Number(summary.top_k)}
        </p>
      ) : null}
      <p className="viz-muted">Retrieved {Number(summary.retrieved_count ?? chunks.length)} chunks</p>
      <ul className="viz-chunks">
        {chunks.map((c) => {
          const sc = Number(c.score);
          const pct = Math.min(100, (sc / maxScore) * 100);
          return (
            <li key={`${String(c.chunk_id)}-${String(c.rank)}`} className="viz-chunk">
              <div className="viz-chunk__head">
                <span className="viz-chunk__rank">#{Number(c.rank)}</span>
                <span className="viz-chunk__score">{sc.toFixed(4)}</span>
              </div>
              <div className="viz-bar">
                <div className="viz-bar__fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="viz-chunk__text">{String(c.text_preview)}</p>
              <span className="viz-chunk__meta">{String(c.document_id)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function VizReranker({ summary }: { summary: Summary }) {
  return (
    <ul className="viz-list">
      <li>
        Enabled: <strong>{summary.enabled ? 'yes' : 'no'}</strong>
      </li>
      <li>Context chunks: {Number(summary.upstream_context_chunks ?? 0)}</li>
      <li className="viz-hint">{String(summary.note ?? '')}</li>
    </ul>
  );
}

function VizLlm({ summary }: { summary: Summary }) {
  const ans = summary.answer != null ? String(summary.answer) : null;
  return (
    <div className="viz-llm">
      {summary.model ? (
        <p className="viz-muted">
          Model <code>{String(summary.model)}</code>
        </p>
      ) : null}
      {ans ? (
        <div className="viz-answer">{ans}</div>
      ) : (
        <p className="node-out__err">{String(summary.error ?? 'No answer')}</p>
      )}
      {summary.usage ? (
        <pre className="viz-json">{JSON.stringify(summary.usage, null, 2)}</pre>
      ) : null}
    </div>
  );
}

function VizFinal({ summary }: { summary: Summary }) {
  const fa = summary.final_answer != null ? String(summary.final_answer) : null;
  return (
    <div className="viz-final">
      {summary.query ? (
        <p className="viz-muted">
          Q: <em>{String(summary.query)}</em>
        </p>
      ) : null}
      <div className="viz-final__answer">{fa ?? '—'}</div>
    </div>
  );
}

function VizGeneric({ summary }: { summary: Summary }) {
  const upstream = summary.upstream_nodes as string[] | undefined;
  return (
    <div className="viz-generic">
      {upstream?.length ? (
        <p className="viz-muted">Upstream: {upstream.join(', ')}</p>
      ) : null}
      <pre className="viz-json">{JSON.stringify(summary, null, 2)}</pre>
    </div>
  );
}
