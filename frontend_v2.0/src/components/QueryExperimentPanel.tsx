import { useCallback, useMemo, useState } from 'react';
import type { Node } from '@xyflow/react';
import { api } from '../api/client';
import type { PreviewRetrievalResponse } from '../types/pipeline';
import { extractQueryRetrievalParams } from '../utils/extractQueryParams';

type QueryExperimentPanelProps = {
  nodes: Node[];
  selectedDocumentIds: Set<string>;
};

function chunkOverlap(a: PreviewRetrievalResponse | null, b: PreviewRetrievalResponse | null): string {
  if (!a?.chunks?.length || !b?.chunks?.length) return '';
  const idsA = new Set(a.chunks.map((c) => String(c['chunk_id'] ?? '')));
  let n = 0;
  for (const c of b.chunks) {
    if (idsA.has(String(c['chunk_id'] ?? ''))) n++;
  }
  return `${n} chunk(s) appear in both lists`;
}

export function QueryExperimentPanel({ nodes, selectedDocumentIds }: QueryExperimentPanelProps) {
  const { queryText, topK } = extractQueryRetrievalParams(nodes, selectedDocumentIds);
  const [variantB, setVariantB] = useState('');
  const [left, setLeft] = useState<PreviewRetrievalResponse | null>(null);
  const [right, setRight] = useState<PreviewRetrievalResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const docFilter = useMemo(
    () => (selectedDocumentIds.size > 0 ? Array.from(selectedDocumentIds) : undefined),
    [selectedDocumentIds],
  );

  const runCompare = useCallback(() => {
    const qa = queryText.trim();
    const qb = variantB.trim();
    if (!qa) {
      setErr('Set a query on the Query node first.');
      return;
    }
    if (!qb) {
      setErr('Enter an alternate query (B) — e.g. shorter keywords or a paraphrase.');
      return;
    }
    setErr(null);
    setBusy(true);
    Promise.all([
      api.previewRetrieval(qa, topK, docFilter),
      api.previewRetrieval(qb, topK, docFilter),
    ])
      .then(([ra, rb]) => {
        setLeft(ra);
        setRight(rb);
        if (ra.error) setErr(ra.error);
        else if (rb.error) setErr(rb.error);
      })
      .catch((e: Error) => {
        setErr(e.message);
        setLeft(null);
        setRight(null);
      })
      .finally(() => setBusy(false));
  }, [queryText, variantB, topK, docFilter]);

  return (
    <div className="query-exp">
      <h4 className="query-exp__title">Query experiment (A / B)</h4>
      <p className="query-exp__hint">
        Compare dense retrieval for the <strong>Query node</strong> text (A) vs a second phrasing (B). Uses the
        same <code>top_k</code> and document filter as the Retriever preview.
      </p>
      <div className="query-exp__field">
        <label>A (from Query node)</label>
        <textarea className="field__textarea" rows={2} readOnly value={queryText} />
      </div>
      <div className="query-exp__field">
        <label htmlFor="query-b">B (alternate query)</label>
        <textarea
          id="query-b"
          className="field__textarea"
          rows={3}
          value={variantB}
          placeholder="Try keywords only, or a shorter question…"
          onChange={(e) => setVariantB(e.target.value)}
        />
      </div>
      <button type="button" className="btn btn--primary" disabled={busy} onClick={() => void runCompare()}>
        {busy ? 'Comparing…' : 'Run A / B retrieval'}
      </button>
      {err && <p className="inspector__error">{err}</p>}
      {left && right && !left.error && !right.error && (
        <p className="query-exp__overlap">{chunkOverlap(left, right)}</p>
      )}
      <div className="query-exp__grid">
        <div className="query-exp__col">
          <div className="query-exp__col-head">A · top {left?.top_k ?? topK}</div>
          <ul className="query-exp__list">
            {(left?.chunks ?? []).map((c) => (
              <li key={`a-${String(c['chunk_id'])}-${String(c['rank'])}`} className="query-exp__item">
                <span className="query-exp__rank">{String(c['rank'])}</span>
                <span className="query-exp__score">{Number(c['score']).toFixed(4)}</span>
                <p>{String(c['text_preview'] ?? '')}</p>
              </li>
            ))}
          </ul>
        </div>
        <div className="query-exp__col">
          <div className="query-exp__col-head">B · top {right?.top_k ?? topK}</div>
          <ul className="query-exp__list">
            {(right?.chunks ?? []).map((c) => (
              <li key={`b-${String(c['chunk_id'])}-${String(c['rank'])}`} className="query-exp__item">
                <span className="query-exp__rank">{String(c['rank'])}</span>
                <span className="query-exp__score">{Number(c['score']).toFixed(4)}</span>
                <p>{String(c['text_preview'] ?? '')}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
