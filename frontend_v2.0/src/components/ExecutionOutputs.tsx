import { useEffect, useState } from 'react';
import type { GraphExecutionResult, NodeExecutionResult } from '../types/pipeline';
import {
  approxTokensFromChars,
  DEFAULT_TEACHING_CONTEXT_LIMIT,
} from '../utils/tokenEstimate';

type Summary = Record<string, unknown>;

export function ExecutionOutputs({
  result,
  runAt,
}: {
  result: GraphExecutionResult | null;
  /** ISO timestamp when this run finished (for logs / export). */
  runAt?: string | null;
}) {
  const [citationFocus, setCitationFocus] = useState<number | null>(null);

  useEffect(() => {
    if (citationFocus == null) return;
    window.setTimeout(() => {
      document.getElementById(`citation-chunk-${citationFocus}`)?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }, 50);
  }, [citationFocus]);

  if (!result?.node_results?.length) return null;

  return (
    <div className="exec-outputs">
      <h4 className="exec-outputs__title">Pipeline output (per node)</h4>
      {runAt && (
        <p className="exec-outputs__runat" title="Run log timestamp">
          Run at <time dateTime={runAt}>{runAt}</time>
        </p>
      )}
      <p className="exec-outputs__sub">
        Results follow the graph order. Fix errors shown in red before expecting a full answer. Use{' '}
        <strong>Sources</strong> below the LLM answer to highlight matching chunks by rank.
      </p>
      <div className="exec-outputs__list">
        {result.node_results.map((nr) => (
          <NodeOutputCard
            key={nr.node_id}
            nr={nr}
            citationFocus={citationFocus}
            onPickCitation={setCitationFocus}
          />
        ))}
      </div>
    </div>
  );
}

function NodeOutputCard({
  nr,
  citationFocus,
  onPickCitation,
}: {
  nr: NodeExecutionResult;
  citationFocus: number | null;
  onPickCitation: (rank: number | null) => void;
}) {
  const s = nr.output_summary as Summary;
  const viz = String(s.viz ?? 'generic');
  const title = `${nr.kind} · ${String(s.label ?? nr.node_id)}`;

  return (
    <article className="node-out" id={`node-out-${nr.node_id}`}>
      <header className="node-out__head">
        <span className="node-out__title">{title}</span>
        <span className="node-out__latency">{nr.latency_ms.toFixed(1)} ms</span>
      </header>
      <div className="node-out__body">
        <NodeViz
          viz={viz}
          summary={s}
          citationFocus={citationFocus}
          onPickCitation={onPickCitation}
        />
      </div>
    </article>
  );
}

function ContextBudgetStrip({ summary }: { summary: Summary }) {
  const chars = Number(summary.char_count ?? 0);
  const approxTok =
    summary.approx_input_tokens != null
      ? Number(summary.approx_input_tokens)
      : approxTokensFromChars(chars);
  const pct = Math.min(100, (approxTok / DEFAULT_TEACHING_CONTEXT_LIMIT) * 100);
  const warn = approxTok > DEFAULT_TEACHING_CONTEXT_LIMIT * 0.85;
  return (
    <div className="context-budget">
      <div className="context-budget__head">
        <strong>Context budget (teaching estimate)</strong>
        <span className="context-budget__nums">
          ~{approxTok.toLocaleString()} tokens · {chars.toLocaleString()} chars · cap{' '}
          {DEFAULT_TEACHING_CONTEXT_LIMIT.toLocaleString()} (typical modern chat model)
        </span>
      </div>
      <div className="context-budget__bar" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className="context-budget__fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="context-budget__note">
        Uses ~4 chars per token (rough). Real tokenizers differ; long prompts may truncate or cost more.
        {warn ? ' You are approaching a range where truncation or quality loss is common.' : ''}
      </p>
    </div>
  );
}

function scoreMatchHint(score: number, maxScore: number): string {
  if (!(maxScore > 0) || !Number.isFinite(score)) return '';
  const r = score / maxScore;
  if (r >= 0.85) return 'Strong match (relative to this hit list)';
  if (r >= 0.55) return 'Moderate';
  if (r >= 0.3) return 'Weak';
  return 'Marginal — verify in corpus';
}

function SourceStrip({
  chunks,
  citationFocus,
  onPickCitation,
}: {
  chunks: Array<Record<string, unknown>>;
  citationFocus: number | null;
  onPickCitation: (rank: number | null) => void;
}) {
  if (!chunks.length) return null;
  return (
    <div className="citation-sources">
      <span className="citation-sources__label">Sources (click to highlight in Retriever card)</span>
      <div className="citation-sources__btns">
        {chunks.map((c) => {
          const rank = Number(c.rank);
          const active = citationFocus === rank;
          return (
            <button
              key={`${rank}-${String(c.chunk_id)}`}
              type="button"
              className={`citation-sources__btn${active ? ' citation-sources__btn--active' : ''}`}
              onClick={() => onPickCitation(active ? null : rank)}
            >
              [{rank}]
            </button>
          );
        })}
      </div>
    </div>
  );
}

function NodeViz({
  viz,
  summary,
  citationFocus,
  onPickCitation,
}: {
  viz: string;
  summary: Summary;
  citationFocus: number | null;
  onPickCitation: (rank: number | null) => void;
}) {
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
      return (
        <VizRetrieved
          summary={summary}
          citationFocus={citationFocus}
        />
      );
    case 'vector_store':
      return <VizVectorStore summary={summary} />;
    case 'prompt_augment':
      return <VizPromptAugment summary={summary} />;
    case 'reranker':
      return <VizReranker summary={summary} />;
    case 'llm_answer':
      return (
        <VizLlm
          summary={summary}
          citationFocus={citationFocus}
          onPickCitation={onPickCitation}
        />
      );
    case 'final_output':
      return (
        <VizFinal
          summary={summary}
          citationFocus={citationFocus}
          onPickCitation={onPickCitation}
        />
      );
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
      <ContextBudgetStrip summary={summary} />
      <p className="viz-muted">
        Context blocks use <code>[1]</code>, <code>[2]</code>, … so the model can cite sources.
      </p>
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

function VizRetrieved({
  summary,
  citationFocus,
}: {
  summary: Summary;
  citationFocus: number | null;
}) {
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
          const rank = Number(c.rank);
          const hint = scoreMatchHint(sc, maxScore);
          const meta = c.metadata as Record<string, unknown> | undefined;
          const rerankNote =
            meta && meta['rerank_score'] != null ? ` · rerank score ${meta['rerank_score']}` : '';
          const focused = citationFocus != null && citationFocus === rank;
          return (
            <li
              key={`${String(c.chunk_id)}-${String(c.rank)}`}
              className={`viz-chunk${focused ? ' viz-chunk--focus' : ''}`}
              id={`citation-chunk-${rank}`}
            >
              <div className="viz-chunk__head">
                <span className="viz-chunk__rank">[{rank}]</span>
                <span className="viz-chunk__score">{sc.toFixed(4)}</span>
                {hint ? <span className="viz-chunk__hint">{hint}{rerankNote}</span> : null}
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

function VizLlm({
  summary,
  citationFocus,
  onPickCitation,
}: {
  summary: Summary;
  citationFocus: number | null;
  onPickCitation: (rank: number | null) => void;
}) {
  const ans = summary.answer != null ? String(summary.answer) : null;
  const rawSources = summary.source_chunks as Array<Record<string, unknown>> | undefined;
  const sourceChunks = Array.isArray(rawSources) ? rawSources : [];
  return (
    <div className="viz-llm">
      <p className="viz-hint">
        Generation via OpenRouter chat completions. Raw HTTP messages are not shown; exports omit API keys.
        Citation buttons link to the Retriever card when ranks match.
      </p>
      {summary.model ? (
        <p className="viz-muted">
          Model <code>{String(summary.model)}</code>
        </p>
      ) : null}
      <SourceStrip chunks={sourceChunks} citationFocus={citationFocus} onPickCitation={onPickCitation} />
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

function VizFinal({
  summary,
  citationFocus,
  onPickCitation,
}: {
  summary: Summary;
  citationFocus: number | null;
  onPickCitation: (rank: number | null) => void;
}) {
  const fa = summary.final_answer != null ? String(summary.final_answer) : null;
  const rawSources = summary.source_chunks as Array<Record<string, unknown>> | undefined;
  const sourceChunks = Array.isArray(rawSources) ? rawSources : [];
  return (
    <div className="viz-final">
      {summary.query ? (
        <p className="viz-muted">
          Q: <em>{String(summary.query)}</em>
        </p>
      ) : null}
      <SourceStrip chunks={sourceChunks} citationFocus={citationFocus} onPickCitation={onPickCitation} />
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
