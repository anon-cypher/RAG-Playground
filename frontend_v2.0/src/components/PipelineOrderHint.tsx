/** Canonical pipeline order for learning (matches plan.md Phase 2). */
const STAGES = [
  'Document loader — load corpus',
  'Chunker — split text',
  'Embedder — OpenRouter embeddings',
  'Indexer — build FAISS index',
  'Vector store — index health',
  'Query — user question',
  'Retriever — nearest chunks',
  'Re-ranker — optional reorder',
  'Prompt augment — assemble prompt',
  'LLM — OpenRouter chat',
  'Output — final answer',
];

export function PipelineOrderHint() {
  return (
    <details className="pipeline-hint">
      <summary className="pipeline-hint__summary">Typical pipeline order</summary>
      <ol className="pipeline-hint__list">
        {STAGES.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
    </details>
  );
}
