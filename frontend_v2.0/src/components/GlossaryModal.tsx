import { useEffect, useId } from 'react';

const TERMS: Array<{ term: string; def: string }> = [
  { term: 'Chunk', def: 'A slice of document text used for embedding and retrieval.' },
  { term: 'Embedding', def: 'A dense vector representing text meaning; similar texts map closer in vector space.' },
  { term: 'top-k', def: 'How many nearest chunks to retrieve for a query.' },
  { term: 'Indexer', def: 'Builds the FAISS vector index from embeddings.' },
  { term: 'Vector store', def: 'Health / stats view over the index (does not replace indexing).' },
  { term: 'Reranker', def: 'Optional second stage to reorder retrieved chunks.' },
  { term: 'Prompt augment', def: 'Builds the final prompt from templates, query, and retrieved context.' },
  {
    term: 'PCA projection',
    def: '3D plot compresses high-dimensional vectors; distances in the plot are approximate, not identical to ranking distance.',
  },
];

type GlossaryModalProps = {
  open: boolean;
  onClose: () => void;
};

export function GlossaryModal({ open, onClose }: GlossaryModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="glossary-backdrop" role="presentation" onClick={onClose}>
      <div
        className="glossary-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="glossary-modal__head">
          <h2 id={titleId}>Glossary</h2>
          <button type="button" className="btn" onClick={onClose} aria-label="Close glossary">
            Close
          </button>
        </header>
        <dl className="glossary-modal__list">
          {TERMS.map((t) => (
            <div key={t.term} className="glossary-modal__row">
              <dt>{t.term}</dt>
              <dd>{t.def}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
