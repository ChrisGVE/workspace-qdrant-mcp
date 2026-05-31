# RAG Quality Session — 2026-05-30 Outcomes

Continuation of the search/graph quality work. Three improvements **shipped to
`main` + `dev`**; one upgrade (embedding model) **started and paused** at a
hardware-compat gate.

> **Scope note.** This doc covers the RAG/search/graph frente only. The LSP &
> code-intelligence work that ran the same session (bundled language servers,
> protocol framing, callHierarchy CALLS edges, observability) is recorded
> separately in
> [2026-05-31-lsp-session-outcomes.md](2026-05-31-lsp-session-outcomes.md).

## Shipped (main + dev)

| Commit | Change | Verification |
|---|---|---|
| `06d916852` | **Cross-encoder reranker** — root-cause fix + jina-turbo model | live + benchmark |
| `8252330fc` | **Turbofish/artifact filter** for graph call edges | 7 unit tests |
| `158f85364` | **Name-based stub-edge resolution** + periodic background resolver | 2 unit tests + live |

### 1. Reranker (`06d916852`)
- **Root cause it fixed:** the lazy reranker never loaded — fastembed defaulted
  its ONNX cache to `./.fastembed_cache` (CWD-relative), but the daemon runs as a
  non-root user with `CWD=/`, so the path was unwritable → download failed in
  ~310ms with `Failed to retrieve model file: onnx/model.onnx`. Fix: thread
  `model_cache_dir` from `embedding_settings` → `EmbeddingServiceImpl` →
  `RerankInitOptions::with_cache_dir` (same pattern the dense/sparse providers use).
- **Model:** switched `BGERerankerBase` (278M params, 1.1GB, **~3s/query** on CPU
  — too slow, broke the benchmark timeout) → **`JINARerankerV1TurboEn`** (38M
  params, ~150MB, English, ~0.5s/query).
- **Measured** (12-query bundled benchmark, semantic, vs no-rerank baseline):
  `top1 16.7→33.3%` (2×), `top3 41.7→66.7%`, `recall@10 58.3→70.8%` (meets ≥70%
  target), `MRR 0.34→0.52`.

### 2. Graph call/import graph (`8252330fc` + `158f85364`)
- **Eval finding:** the code-relationship graph was **0% resolved** — every
  CALLS/IMPORTS/USES_TYPE edge was a dangling tree-sitter stub (empty `file_path`),
  plus 18 generic-fragment artifacts (`<String`, `_>`).
- **Artifact filter:** `clean_callee_name`/`strip_generic_args` (chunker) +
  `is_valid_symbol_name` (extractor) strip turbofish/generic fragments at the source.
- **Name-based resolution:** `GraphStore::resolve_stub_edges` repoints each dangling
  edge to a real same-name project symbol (same-file → unique-in-tenant → skip
  ambiguous; recomputes edge_id; prunes orphaned stubs). Driven by a periodic
  background task (120s) — **heals the existing graph in place, no reindex.**
- **Verified live:** 109 edges repointed, correct intra-project mappings
  (`applyPathRelevanceBoost → queryContentTokens`, `index_points_by_label →
  extract_str`). Reaches the **resolvable ceiling** (~53 CALLS / 43 IMPORTS / 44
  USES_TYPE for the dev tenant); the rest are stdlib (`clone`/`map`/`Ok`) and are
  correctly excluded.

### 3. Graph MCP tool (#2 of this session's asks)
- Already wired+committed last session; this session confirmed the running MCP
  **advertises `graph`** via a live `tools/list` (10 tools total). A client-side
  MCP reconnect is needed for an existing session to see/call it.

### 4. Reembed collection-schema fix (`recreator.rs` + `upsert.rs`) — post-incident
Clearing the last 2 turbofish artifacts (in `grammar.rs`) via `wqm admin reembed`
surfaced a latent bug that **took vector search down**:
- **Bug:** `TriggerReembed`'s drop-and-recreate path rebuilt the 4 canonical
  collections with a single **unnamed** 384d vector (`create_collection`), not the
  named `dense` + named `sparse` vectors hybrid search requires. Qdrant then
  declined every upsert (`collection_updater: Update operation declined: Not
  existing vector name error: dense` / `sparse`) — **silently**, because batch
  upserts run `wait=false`, so the daemon logged "N successful, 0 failed" while all
  collections sat at 0 points. No snapshots existed to restore from.
- **Fix:** recreate via `create_multi_tenant_collection` (the SAME method the
  create-on-index path uses via `shared::ensure_collection`). Belt-and-suspenders:
  `finalize_batch_result` no longer reports acknowledged-but-unconfirmed points as
  "successful" under `wait=false` (now "submitted … apply not confirmed"), so a
  future schema mismatch can't hide. Regression test
  `reembed_recreate_uses_named_dense_sparse_schema`. **DEPLOYED 2026-05-31**
  (`main` tip `f36468078`, PR #71; payload-index backfill `753506f09`; clippy
  cleanup PR #72): the running daemon was recreated from the post-fix image and
  search verified — so the recreate path is now safe in code, not just patched live.
- **Recovery (used live, for the still-running pre-fix daemon):** delete the 4
  empty collections via the Qdrant API, then `docker restart wqm-memexd` → the
  create-on-index path rebuilds the correct named-vector schema and startup
  reconcile re-enqueues every source → collections repopulate. Verified: search
  restored (`projects` climbed back past 5.7k points), graph garbage still **0**.
- **De-risks the embedding upgrade:** the migration plan below (step 3) calls
  `TriggerReembed` to recreate collections at 1024d — it would have hit this exact
  bug. Now safe once the fix is deployed.

## Negative results (do NOT retry)
- **`bge-reranker-base` on CPU:** ~3s/query — too slow for interactive search and
  blows the benchmark timeout. Use jina-turbo.
- **Rerank-before-dedup (#3, "top3→80%"):** reranking chunks (instead of
  one-per-file) to let each file be scored by its best chunk **REGRESSED top3
  66.7→58.3%** and MRR 0.52→0.48 (recall rose 70.8→75%). `embedding-provider`
  fell rank 2→5 — a spuriously-high chunk displaced the right file. Reverted
  (was uncommitted). **top3→80% is not reachable via rerank/pool tuning** — the
  remaining misses are retrieval failures (`queue-metrics`, `search-quality-plan`
  never surface the file) and semantic ambiguity (Rust rules files vs the TS
  `rules.ts`). The real lever is retrieval-side (embeddings/query expansion).

## Embedding upgrade — STARTED, PAUSED at a gate
Goal: replace MiniLM-L6 (384d, weak) with **BGE-large-en-v1.5 (1024d)** — the real
retrieval-ceiling lever. **Not yet wired to the daemon (still on `fastembed`).**

- **GPU gate FAILED (Blackwell):** TEI `:1.7` image is compiled for Ampere
  (`compute cap 80`); the RTX 5070 Ti is Blackwell (`compute cap 120`) →
  `Could not start Candle backend: Runtime compute cap 120 is not compatible
  with compile time compute cap 80`. Docker GPU passthrough itself works.
- **CPU works:** `text-embeddings-inference:cpu-1.7` serves bge-large via the ONNX
  backend (`onnx/model.onnx`), OpenAI-compatible at `/v1/embeddings`, **dim=1024
  confirmed**. (Container `wqm-tei`.)
- **In progress at pause:** trying **Infinity** (`michaelf34/infinity`, PyTorch-based)
  on the GPU — PyTorch ≥2.7 + CUDA 12.8 supports sm_120 where TEI's Candle didn't.
  Container `wqm-infinity` was pulling when we stopped.

### Migration mechanism (when resuming)
1. Config: `WQM_EMBEDDING_PROVIDER=openai_compatible`, `WQM_EMBEDDING_BASE_URL=http://<server>:<port>`,
   `WQM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5`, `output_dim=1024`,
   `WQM_EMBEDDING_API_KEY_ENV_VAR=<dummy>` (TEI/Infinity need no auth). The OpenAI
   provider calls `base_url + /v1/embeddings`.
2. Start `memexd` with **`--bootstrap-reembed`** (suppresses the startup
   dim-mismatch guard, which otherwise aborts on 384d-collection vs 1024d-config).
3. Call **`TriggerReembed`** (DESTRUCTIVE — drops/recreates collections at 1024d).
   NOT `ReembedTenant` (non-destructive, can't change dim). **Requires the §4
   schema fix deployed** — pre-fix, the recreate built an unnamed-vector collection
   that silently declined all hybrid upserts.
4. Remove `--bootstrap-reembed`.

### ⚠️ Scope caveat
`projects` is ONE collection partitioned by `tenant_id` (ADR-001). The dim change
recreates it → **re-embeds ALL indexed projects** (workspace-qdrant-mcp,
bws-engineer, v0-bws-training, …), and **all search is degraded during the reembed
window.** On CPU that's hours; on GPU, minutes — hence the push for a working GPU
server. The reembed is **model-bound, not server-bound**: bge-large 1024d vectors
produced on CPU-TEI are identical to GPU-Infinity, so a later CPU→GPU swap needs
no re-embed.

## Lessons learned
1. **Verify the hardware/toolchain gate BEFORE big ops** (recurring theme): the
   graph LSP-resolution was blocked by missing `cargo`; the embedding upgrade by
   Blackwell-vs-TEI. Both are "looks done in code, fails at runtime" traps.
2. **The reranker reorders; it can't fix retrieval.** Once misses are retrieval
   failures, only a better embedding/query lever helps.
3. **Dedup-before-rerank beats rerank-before-dedup for top-k precision** (file
   diversity in the pool > best-chunk-per-file scoring noise).
4. **Multi-clone tenant knot has wide blast radius:** it caps the graph resolution
   ceiling (partial indexing under the legacy `local_` tenant) AND blocks LSP graph
   resolution (LSP registers under the canonical `367157a01d98`, indexing under
   `local_5288aa13ad6c` — `is_server_ready_for_file` never matches).
5. **Don't run a full reembed for a cosmetic cleanup.** Polishing 2 harmless stale
   graph edges via `wqm admin reembed` triggered a global, multi-project re-embed
   AND exposed the unnamed-vector recreate bug — a long search outage for near-zero
   benefit. The artifact filter is in the extraction code, so residual pre-fix edges
   are harmless and clear on the file's next real edit. Match the tool to the blast
   radius.
6. **`wait=false` upserts hide async declines.** Qdrant ACKs before applying; a
   wrong-schema apply is rejected later with no error on the write path, so the
   daemon happily logged "successful" at 0 points. Destructive ops (drop/recreate)
   need an explicit post-condition check (point count / a probe upsert with
   `wait=true`), not just a green write log.

## Next steps (priority order)
1. **Finish embedding upgrade** — verify the Infinity Blackwell gate; if it works,
   run the migration above (mind the global scope). If not, options: CPU-TEI now
   (quality win, ~+100ms/query, hours-long global reembed) or build TEI from source
   with sm_120. Then measure with `search_eval` (expect top1/recall to jump).
   **De-risked:** the §4 schema fix this step depended on is now **deployed**
   (`f36468078`) — `TriggerReembed` at 1024d will recreate named-vector
   collections correctly instead of silently declining all hybrid upserts.
2. **Multi-clone tenant knot** — highest-leverage infra fix: unblocks dev-branch
   search AND raises the graph resolution ceiling AND enables LSP graph resolution.
3. **Query expansion** (lighter retrieval lever, no reembed) if the embedding
   upgrade stalls.
4. Minor review follow-ups: macro-exclusion comment in `is_valid_symbol_name`;
   dirty-tracking so the graph resolver doesn't re-scan unresolvable stdlib
   danglers every 120s.

## Experimental containers (cleanup state)
`wqm-tei` (CPU bge-large, working) and `wqm-infinity` (GPU attempt) were created
this session and are **not wired to the daemon**. Cached weights persist in the
`tei_data` / `infinity_data` volumes, so restart is fast. Stop/remove to free
resources; recreate per the commands in this doc when resuming.
