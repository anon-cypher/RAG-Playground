# RAG Playground — phased implementation plan

This document turns the product vision into **implementable phases**. Stack assumption: **OpenRouter** for embeddings and chat (`https://openrouter.ai/api/v1`), OpenAI-compatible APIs. Users **enter their OpenRouter API key in the UI** (see [API key handling](#api-key-handling)).

**Implementation notes (repo):** Backend `POST /api/pipeline/credentials` and `POST /api/pipeline/verify-key`; frontend [`SettingsBar`](frontend_v2.0/src/components/SettingsBar.tsx), [`openrouterStorage`](frontend_v2.0/src/utils/openrouterStorage.ts), [`exportRun`](frontend_v2.0/src/utils/exportRun.ts); [`README.md`](README.md) documents OpenRouter setup.

**This release (next):** [Phase 11 — Resizable workspace layout](#phase-11--resizable-workspace-layout) — remove fixed `vh`/band heights so panels **flow and resize** (auto + optional drag); stop overlap with the embedding section and run panel.

**Shipped:** [Phase 10 — Advanced RAG & LLM pedagogy](#phase-10--advanced-rag--llm-pedagogy-this-release).

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

## Phase 10 — Advanced RAG & LLM pedagogy (this release)

**Goal:** A newcomer can use only the UI to explain **what filled the context window**, **which chunks grounded the answer**, **why two query formulations retrieve different sets**, and **how candidate chunks were ranked**.

| ID | Task | Done |
|----|------|------|
| 10.1 | **Context budget meter** — Approximate token usage for instructions + retrieved chunks + user query vs a nominal limit; brief copy on truncation and why it matters. | [x] |
| 10.2 | **Citations / grounding** — Number retrieved chunks in the assembled prompt; show the same indices next to the model answer; optional **citation inspector** (click a ref → scroll/highlight chunk). | [x] |
| 10.3 | **Score breakdown** — Per hit: similarity score, rerank score when reranking is used, and a short qualitative hint (e.g. strong vs marginal match). | [x] |
| 10.4 | **Chunking & cost intuition** — Extend existing chunk preview (Phase 3) with **token estimates** (or clear “approx tokens per chunk”) so overlap/size tradeoffs are visible before embed cost. | [x] |
| 10.5 | **Query experiment (A/B)** — One concrete alternate path (e.g. query rewrite or second retrieval variant); **side-by-side** ranked lists or answers vs baseline so users see retrieval-driven differences. | [x] |
| 10.6 | **Per-node “Why this step?”** — Short panel or expander per node kind: purpose, common failures, what to verify; complements [`GlossaryModal`](frontend_v2.0/src/components/GlossaryModal.tsx). | [x] |
| 10.7 | **Run narrative** — Single place after a full run: ordered stages, timings, chunk/top-k/token **counts**, with links into inspector sections (builds on Phase 2.2 / 8.1 but framed for teaching). | [x] |

**Where (Phase 10):** [`tokenEstimate.ts`](frontend_v2.0/src/utils/tokenEstimate.ts), [`nodeWhy.ts`](frontend_v2.0/src/utils/nodeWhy.ts), [`assemblePrompt.ts`](frontend_v2.0/src/utils/assemblePrompt.ts) · [`ExecutionOutputs.tsx`](frontend_v2.0/src/components/ExecutionOutputs.tsx) (context budget, citations, score hints) · [`Inspector.tsx`](frontend_v2.0/src/components/Inspector.tsx) (Why this step?, chunk tokens, prompt live estimate, retrieval preview + doc filter) · [`QueryExperimentPanel.tsx`](frontend_v2.0/src/components/QueryExperimentPanel.tsx) · [`RunNarrative.tsx`](frontend_v2.0/src/components/RunNarrative.tsx) · [`App.tsx`](frontend_v2.0/src/App.tsx) · backend [`pipeline.py`](backend/services/pipeline.py) (numbered context, `approx_input_tokens`, `source_chunks`, `preview_retrieval` + `selected_documents`).

**Out of scope for this release:** Graph RAG, agentic multi-hop, full Self-RAG/CRAG, large families of new pipeline nodes, or new LLM providers beyond the current OpenRouter-oriented flow.

**Exit criteria:** Same as **Goal** above; screenshots or README “Teaching with this playground” blurb updated when stable.

**Suggested build order within Phase 10:** 10.1 → 10.2 → 10.3 → 10.6 (mostly UI copy + wiring) in parallel with 10.4; then 10.5 (needs clearest single A/B story); finish with 10.7 once metrics are stable.

---

## Phase 11 — Resizable workspace layout

**Problem:** The main column stack mixes **fixed `vh` caps**, **nested scroll regions**, and a **tall “3D embedding” strip**. That produces clipped inspector/run UI and bars that appear to sit **on top of** the node config panel (especially on shorter viewports).

**Goal:** One predictable page flow: **no overlapping regions**; section sizes follow **available viewport** and user preference. Prefer **automatic** distribution first, then **optional drag-to-resize** for power users.

### Design direction

1. **Single primary scroll (default)**  
   - Treat the app as a **vertical document**: header → settings → **workspace row** → embedding section.  
   - Avoid multiple nested `overflow: hidden` stacks unless each pane has a clear role (e.g. graph canvas vs sidebars).

2. **Workspace row: fractional widths, not fixed pixel-only**  
   - Keep a 3-column grid: palette \| canvas \| inspector+run.  
   - Use `minmax()` for side columns and `minmax(0, 1fr)` for the canvas so flex/grid children can shrink.  
   - **Row height:** derive from `min-height` + content, or `clamp()` / `svh` where helpful—not a pile of `50vh` caps on nested panels.

3. **Inspector vs run panel: split pane**  
   - **Auto mode:** Inspector gets natural height up to a **soft** max (`max-height` in `%` of workspace or `min(…, 40%)`), then internal scroll; run/output block uses remaining space OR follows below in the document (single outer scroll).  
   - **Manual mode (Phase 11.2):** A **horizontal drag handle** between “node config” and “run / experiment” (or a vertical handle between workspace and embedding—pick one primary handle first to limit complexity).

4. **Embedding section**  
   - Give it a **bounded height** (`min-height` + `max-height` with internal scroll for the 3D/detail split) so it does not eat the whole viewport and shove the inspector under a “footer” feel.  
   - Ensure it is **in normal document flow** below the workspace (not `position: fixed`).

### Implementation options (choose in 11.1)

| Option | Pros | Cons |
|--------|------|------|
| **A. CSS-only** (`clamp`, `fr`, one scroll, remove `vh` caps) | Zero deps, fast | No drag handles without extra UX |
| **B. `react-resizable-panels`** | Battle-tested drag, persistence API | New dependency, learn API |
| **C. Native `resize` on a wrapper** | No React lib | Clunky UX, inconsistent across browsers |

**Recommendation:** **11.1** ship **option A** (layout correctness + overlap fix). **11.2** add **option B** (or a thin custom splitter) for **persisted** inspector/run split widths/heights in `localStorage`.

### Tasks

| ID | Task | Done |
|----|------|------|
| 11.1 | **Layout audit & auto-resize** — Remove/lift rigid `vh` caps on `.inspector` / nested panes; unify overflow so embedding actions never cover inspector; workspace grid uses `minmax(0, …)`; validate short + tall viewports. | [ ] |
| 11.2 | **Drag resize (MVP)** — One resizable split (inspector height **or** right column width **or** workspace vs embedding height); persist keys in `localStorage`; keyboard-safe (optional). | [ ] |
| 11.3 | **Polish** — Min/max constraints, cursor/`aria` on splitter, reset layout control; snapshot in README or in-app tip. | [ ] |

**Exit criteria:** At 768px and 1440px widths, user can always see **Node config** and **Run pipeline** without overlap; embedding controls remain visible or scroll into view without obscuring the inspector; optional: splitter restores last size on reload.

**Where (expected):** [`App.tsx`](frontend_v2.0/src/App.tsx), [`App.css`](frontend_v2.0/src/App.css); possible new `WorkspaceSplit.tsx` or `react-resizable-panels` wrapper.

---

## Suggested implementation order

```text
Phase 1 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
         ↘ Phase 2 can start early in parallel (copy + DAG)
Phase 8 after core path works
Phase 9 ongoing + hardening before “release”
Phase 10 (pedagogy) after Phase 7 + 8 baseline; can overlap Phase 9 polish
Phase 11 (layout) after Phase 10 or in parallel with small UI fixes
```

---

## Mapping to this repository (optional)

As you complete phases, tick items above and add short notes under each phase (e.g. file paths, PR links). Existing pieces in this repo may already cover parts of 4, 5, 6, 7 — reconcile rather than duplicate.

---

## Revision

- **2026-04-08** — Added **Phase 10 (this release)** for advanced RAG / LLM pedagogy: context budget, citations, score UX, chunk token hints, one A/B query experiment, per-node guidance, unified run narrative. Stretch items remain in Phases 5.3, 7.4, 8.3.
- **2026-04-08** — Phase 10 **implemented** (numbered prompt context, citation UI, teaching meter, A/B panel, run narrative, `preview-retrieval` document filter).
- **2026-04-08** — **Phase 11** drafted: resizable/adaptive workspace (remove fixed `vh` overlap; optional drag splits + `localStorage`).
- Update this file when scope changes (e.g. add reranking as first-class, or move to BFF-only keys).
