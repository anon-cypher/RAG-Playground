# RAG Playground — phased implementation plan

This document turns the product vision into **implementable phases**. Stack assumption: **OpenRouter** for embeddings and chat (`https://openrouter.ai/api/v1`), OpenAI-compatible APIs. Users **enter their OpenRouter API key in the UI** (see [API key handling](#api-key-handling)).

**Implementation notes (repo):** Backend `POST /api/pipeline/credentials` and `POST /api/pipeline/verify-key`; frontend [`SettingsBar`](frontend_v2.0/src/components/SettingsBar.tsx), [`openrouterStorage`](frontend_v2.0/src/utils/openrouterStorage.ts), [`exportRun`](frontend_v2.0/src/utils/exportRun.ts); [`README.md`](README.md) documents OpenRouter setup.

---

## Principles (apply in every phase)

- **Teach honestly**: Each visualization states what it shows and what it distorts (e.g. PCA/UMAP ≠ true distance in embedding space).
- **Separate concerns**: Retrieval results, assembled prompt, and model answer must be inspectable independently where possible.
- **OpenRouter-only (MVP)**: No required local model path; configurable model IDs (e.g. `openai/text-embedding-3-small`, `openai/gpt-4o-mini`).
- **Never log or expose keys**: Keys are not written to server logs, analytics, or shared exports.

---

## API key handling

| Approach | Use when |
|----------|----------|
| **Client → OpenRouter** | Personal / local dev; key visible in browser (network tab). |
| **Backend proxy (BFF)** | Shared deployments; key sent only to your backend, forwarded to OpenRouter, not persisted. |

**Current implementation:** Keys are stored in **localStorage** and synced to the FastAPI process via **`POST /api/pipeline/credentials`** (no automatic index rebuild). Documented in [README.md](README.md).

---

## Phase 1 — Foundation (auth + config + safety)

| ID | Task | Done |
|----|------|------|
| 1.1 | Settings / inspector: OpenRouter API key field (password input); optional “Test key” action (minimal completion or models list). | [x] |
| 1.2 | Persist key: session or `localStorage` with clear UX; clear-on-logout if applicable. | [x] |
| 1.3 | Central config: `base_url`, default embedding model + dim, default LLM model; validate dim matches chosen embedding model. | [x] |
| 1.4 | User-facing errors for 401 / 429 / rate limits; never echo raw provider bodies that might contain secrets. | [x] |

**Exit criteria:** User can save a key, see success/failure, and run one embedding + one chat call through OpenRouter.

**Where:** [`SettingsBar`](frontend_v2.0/src/components/SettingsBar.tsx), [`api/client.ts`](frontend_v2.0/src/api/client.ts) (`friendlyHttpError`, `setCredentials`, `verifyOpenRouterKey`), [`backend/api/pipeline.py`](backend/api/pipeline.py) (`/credentials`, `/verify-key`), [`build_index` dim check](backend/services/pipeline.py).

---

## Phase 2 — Pipeline clarity (mental model)

| ID | Task | Done |
|----|------|------|
| 2.1 | Visual DAG or ordered stages: loader → chunk → embed → index → query → retrieve → augment → LLM → output, with short copy per stage. | [x] |
| 2.2 | Per-step I/O in UI: counts, short previews, latency per stage (where available). | [x] |
| 2.3 | Glossary or tooltips: chunk, embedding, top-k, rerank, prompt augment, etc. | [x] |

**Exit criteria:** A new user can name each stage and see what data passed through.

**Where:** XYFlow canvas + group lanes; [`PipelineOrderHint`](frontend_v2.0/src/components/PipelineOrderHint.tsx); [`GlossaryModal`](frontend_v2.0/src/components/GlossaryModal.tsx); [`ExecutionOutputs`](frontend_v2.0/src/components/ExecutionOutputs.tsx) + per-node latency; Palette `title` tooltips.

---

## Phase 3 — Chunking & corpus

| ID | Task | Done |
|----|------|------|
| 3.1 | Chunk preview: live boundaries / overlap on sample or uploaded text. | [x] |
| 3.2 | Document ingestion: upload or select docs; list corpus with metadata. | [x] |
| 3.3 | Optional filters: restrict retrieval to selected document IDs. | [x] |

**Exit criteria:** Changing chunk settings visibly changes chunks before any embedding cost.

**Where:** [`Inspector` chunker](frontend_v2.0/src/components/Inspector.tsx), document loader, `selected_documents` in query / preview paths.

---

## Phase 4 — Embeddings & index (OpenRouter)

| ID | Task | Done |
|----|------|------|
| 4.1 | Embed texts via OpenRouter embeddings API; batching + error surfacing. | [x] |
| 4.2 | Vector index (e.g. FAISS): build/rebuild; block or warn on dimension mismatch. | [x] |
| 4.3 | Health panel: chunk count, vector count, index type, last build / status. | [x] |

**Exit criteria:** After upload, user builds index and sees non-zero vectors with correct dim.

**Where:** [`embedding.py`](backend/services/embedding.py), [`build_index`](backend/services/pipeline.py) + `dimension_mismatch` response, vector_store / index stats in graph output.

---

## Phase 5 — Retrieval experiments

| ID | Task | Done |
|----|------|------|
| 5.1 | Query + `top_k` controls; debounced retrieval preview where useful. | [x] |
| 5.2 | Ranked results: score, document id, text preview; optional side-by-side compare of two settings. | [x] |
| 5.3 | (Stretch) Hybrid / reranker: if present, show “what changed” vs dense-only. | [ ] |

**Notes:** 5.2 satisfied by ranked list + scores in pipeline output; **side-by-side compare UI** not implemented (optional future).

**Where:** Retriever inspector preview, [`VizRetrieved`](frontend_v2.0/src/components/ExecutionOutputs.tsx).

---

## Phase 6 — Embedding-space visualization

| ID | Task | Done |
|----|------|------|
| 6.1 | Projection: PCA default; optional UMAP later if deps/runtime acceptable. | [x] |
| 6.2 | Visual encoding: query vs corpus vs top-k neighbors; optional edges query→neighbor. | [x] |
| 6.3 | Performance: cap or sample points for large corpora; full neighbor list remains in table. | [x] |
| 6.4 | Disclaimer copy: 3D/2D plot is intuitive only; not identical to ranking geometry. | [x] |

**Exit criteria:** User sees neighbors in space and in a ranked list, with disclaimer visible.

**Where:** [`EmbeddingSpace3D`](frontend_v2.0/src/components/EmbeddingSpace3D.tsx), edge cap, [`get_embedding_space`](backend/services/pipeline.py) (UMAP optional).

---

## Phase 7 — Prompt & generation (OpenRouter)

| ID | Task | Done |
|----|------|------|
| 7.1 | Prompt augment: templates with `{query}` / `{context}`; live assembled prompt. | [x] |
| 7.2 | LLM: chat completions via OpenRouter; model picker, temperature, max tokens. | [x] |
| 7.3 | Show final message(s) or prompt sent to API (redact key). | [x] |
| 7.4 | (Stretch) Streaming responses for long answers. | [ ] |

**Notes:** 7.3 — assembled prompt in augment viz; LLM panel notes OpenRouter; raw HTTP message bodies not duplicated (key never exported).

---

## Phase 8 — Evaluation & reproducibility

| ID | Task | Done |
|----|------|------|
| 8.1 | Run log: timestamp, config snapshot, per-stage latency. | [x] |
| 8.2 | Export: JSON (or similar) of run config + results **without** API key. | [x] |
| 8.3 | (Stretch) Compare two runs (same query, different settings). | [ ] |

**Where:** [`ExecutionOutputs`](frontend_v2.0/src/components/ExecutionOutputs.tsx) `runAt`; [`exportRun.ts`](frontend_v2.0/src/utils/exportRun.ts) + **Export run** button in [`App.tsx`](frontend_v2.0/src/App.tsx).

---

## Phase 9 — Polish & trust

| ID | Task | Done |
|----|------|------|
| 9.1 | Empty states: no key, no docs, no index — each with next steps. | [x] |
| 9.2 | Accessibility: not color-only; keyboard for main actions. | [x] |
| 9.3 | README: OpenRouter setup, model IDs, cost/latency expectations. | [x] |

**Notes:** 9.2 — glossary modal closes with Escape; main buttons have `aria-label` where added; further pass possible.

**Where:** Banners in [`App.tsx`](frontend_v2.0/src/App.tsx); [README.md](README.md).

---

## Suggested implementation order

```text
Phase 1 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
         ↘ Phase 2 can start early in parallel (copy + DAG)
Phase 8 after core path works
Phase 9 ongoing + hardening before “release”
```

---

## Mapping to this repository (optional)

As you complete phases, tick items above and add short notes under each phase (e.g. file paths, PR links). Existing pieces in this repo may already cover parts of 4, 5, 6, 7 — reconcile rather than duplicate.

---

## Revision

Update this file when scope changes (e.g. add reranking as first-class, or move to BFF-only keys).
