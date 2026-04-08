import type { NodeKind } from '../types/pipeline';

export type WhyBlock = {
  purpose: string;
  failures: string;
  verify: string;
};

export const NODE_WHY: Record<NodeKind, WhyBlock> = {
  document_loader: {
    purpose: 'Brings raw files into the corpus and applies your chunker settings at upload time.',
    failures: 'Wrong file type, huge PDFs timing out, or forgetting to re-upload after chunker changes.',
    verify: 'Confirm listed documents, chunk counts, and that selected checkboxes match what you expect to retrieve.',
  },
  chunker: {
    purpose: 'Splits text into segments that fit embedding models and balance recall vs. noise.',
    failures: 'Chunks too large (dilute relevance) or too small (lose context); overlap too low → broken sentences at boundaries.',
    verify: 'Watch token estimates and overlap highlighting in the preview before paying for embeddings.',
  },
  embedder: {
    purpose: 'Turns each chunk into a dense vector so “similar meaning” becomes “close in space”.',
    failures: 'Wrong model vs. index dim, missing API key, or stale index after new uploads.',
    verify: 'Dim matches server config; build step succeeds; vector count equals chunk count.',
  },
  indexer: {
    purpose: 'Structures vectors for fast approximate or exact nearest-neighbor search (FAISS).',
    failures: 'Index not built, HNSW parameters too aggressive for tiny corpora, or index type mismatch.',
    verify: '“Built: yes” and non-zero vectors in downstream vector store stats.',
  },
  vector_store: {
    purpose: 'Summarizes what is persisted: vectors, chunk linkage, and index health.',
    failures: 'Empty store when upload/embed steps were skipped; dimension mismatch after model swap.',
    verify: 'Chunk IDs and vector totals line up with embedder + indexer outputs.',
  },
  query: {
    purpose: 'Holds the user question that retrieval and generation will condition on.',
    failures: 'Empty query, copy-paste clutter, or jargon that does not appear in the corpus.',
    verify: 'One clear question; try the Query experiment panel to see paraphrases change rankings.',
  },
  retriever: {
    purpose: 'Finds top-k chunks whose embeddings are closest to the query embedding (dense search).',
    failures: 'top_k too small, wrong documents selected, or index stale vs. latest chunks.',
    verify: 'Scores decrease down the list; previews match full pipeline retriever output.',
  },
  reranker: {
    purpose: 'Optional second stage that re-orders a wider candidate pool (cross-encoder in many systems).',
    failures: 'On this canvas, the graph run still uses dense order from the Retriever; rerank is mainly documented here.',
    verify: 'Read the node note; compare ranked scores in the Query API when you add full reranking.',
  },
  prompt_augment: {
    purpose: 'Merges system instructions, numbered context, and the question into one LLM-facing prompt.',
    failures: 'Template missing {context} or {query}; separator breaks parsing; context over budget.',
    verify: 'Context budget meter; live preview matches augmented node output after a run.',
  },
  llm: {
    purpose: 'Produces natural-language answers conditioned on the assembled prompt (retrieval + question).',
    failures: 'Missing key, rate limits, temperature too high for factual tasks, or context truncated.',
    verify: 'Citations [1], [2] line up with retriever ranks; usage block shows prompt/completion split if returned.',
  },
  output: {
    purpose: 'Final sink that echoes the answer for exports and teaching narratives.',
    failures: 'Upstream LLM error not fixed — output only reflects last successful generation.',
    verify: 'source list matches retriever; answer matches LLM node when the graph completes cleanly.',
  },
};
