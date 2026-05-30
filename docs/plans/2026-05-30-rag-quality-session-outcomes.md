# RAG Quality Session — 2026-05-30 Outcomes

Continuation of the search/graph quality work. Three improvements **shipped to
`main` + `dev`**; one upgrade (embedding model) **started and paused** at a
hardware-compat gate.

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
   NOT `ReembedTenant` (non-destructive, can't change dim).
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

## Next steps (priority order)
1. **Finish embedding upgrade** — verify the Infinity Blackwell gate; if it works,
   run the migration above (mind the global scope). If not, options: CPU-TEI now
   (quality win, ~+100ms/query, hours-long global reembed) or build TEI from source
   with sm_120. Then measure with `search_eval` (expect top1/recall to jump).
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
