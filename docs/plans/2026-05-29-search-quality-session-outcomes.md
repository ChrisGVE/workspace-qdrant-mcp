# Search Quality — Session Outcomes & Next Steps (2026-05-29)

A working session focused on validating and improving semantic/keyword search.
Records what shipped, what we confirmed about the system, and the prioritized
next steps. Companion docs:
[benchmarking guide](../testing/semantic-search-benchmarking.md),
[embedding/rerank plan](2026-05-25-search-quality-next-steps.md),
[tree-sitter roadmap](../specs/21-tree-sitter-roadmap.md).

## Shipped this session (branch `dev`)

| Commit | What |
|---|---|
| `c6a8272` | `fix(search)` — `branch="*"` now means "any branch" in the FTS paths (`search exact=true` + `grep`); it was filtered literally and matched nothing. |
| `d3bb646` | `chore(index)` — root `.wqmignore` excludes the search eval artifacts (`reports/semantic-search.json`, `scripts/benchmark-data/`) that were polluting results (data leakage). |
| `e0bd75f` | `feat(search)` — same-file chunk dedup + wider candidate pool (`limit*5`) + path-relevance re-rank (`PATH_BOOST_ALPHA=0.8`); dropped the redundant `precision@10` verdict gate. |
| `66645a1` | `docs` — benchmarking guide + tree-sitter roadmap status refresh. |
| `59d8896` | `feat(mcp)` — `search_eval` MCP tool: run the benchmark harness in-process (real index), inline `cases` or bundled dataset. |
| `61e3a75` | `docs(testing)` — document `search_eval`. |

### Measured effect (semantic mode, 12-query known-item benchmark)

| metric | before | after |
|---|---|---|
| top-3 | 33.3% | 50.0% |
| top-10 | 41.7% | 83.3% |
| recall@10 | 20.8% | **70.8%** (clears the 0.70 gate) |
| MRR | 0.22 | 0.39 |
| duplicate rate | 24.2% | 0% |

Verdict POOR → MIXED. The only failing gate left is `top-3 ≥ 80%`.

## What we confirmed about the system

- **Tenant mismatch, not a search bug.** The `.wqm-fork` registry mapped the
  project to an empty tenant; the data lives under `local_5288aa13ad6c`. `grep`
  (FTS5, keyed by path) worked while `search` (Qdrant, keyed by tenant) returned
  0. The MCP detector resolves the correct tenant from `cwd`, so default search
  works; only an explicit wrong `projectId` returned nothing.
- **`hybrid` underperforms `semantic` on natural-language queries** — the
  sparse/BM25 leg amplifies content-term "magnets" (a 2 357-line config YAML,
  the `keyword_extraction/*` subsystem). `exact` mode is 0% for NL by design.
- **`precision@10` is not a meaningful gate** for a ≤2-relevant-doc known-item
  benchmark (caps at ~0.19, ≈ `recall × 0.19`).
- **Tree-sitter roadmap top items already shipped:** Gap #1 (real tokenizer) and
  Gap #2 (symbols → tagging) are done; the doc was stale and is now corrected.
  Gap #3 (LSP `definition`/`kind`) and Gap #4 (incremental parse + diff re-embed)
  remain open.

### Embedding model upgrade — confirmed mechanism (for a stronger model / GPU)

Validated against the code so a future session can act on it directly:

- The embedding layer is **provider-driven** (`config/embedding.rs`,
  `embedding/provider/{fastembed,openai}.rs`): `fastembed` (local ONNX) or
  `openai_compatible` (remote, hits `{base_url}/v1/embeddings`).
- **`fastembed` is pinned to `AllMiniLM-L6-v2` / 384d** in this fork — you cannot
  point it at a bigger local model, and the ONNX build is static-CPU. So a
  stronger model / GPU is **not** done by upgrading fastembed.
- **The supported upgrade path is `provider: openai_compatible` → a GPU embedding
  server** (HuggingFace TEI exposes the OpenAI `/v1/embeddings` API). The compose
  already scaffolds an `embeddings` (TEI) service; switch it to a CUDA image with
  a GPU reservation and a strong model.
- Edit `state/memexd/config.yaml` `embedding:` (`provider`, `base_url`, `model`,
  `output_dim`, `api_key_env_var`). **`model` and `output_dim` are config-file
  only** — env overrides cover only `provider`/`base_url`/`api_key_env_var`. The
  compose currently forces `WQM_EMBEDDING_PROVIDER=fastembed` — remove/adjust.
- **Reindex is one command:** `wqm admin reembed --confirm` →
  `AdminWriteService.TriggerReembed` drops & recreates the 4 canonical Qdrant
  collections **at the configured `output_dim`** and re-enqueues every source.
- **Model candidates:** `gte-large-en-v1.5` / `bge-large-en-v1.5` (1024d),
  `multilingual-e5-large` (1024d, if PT queries matter),
  `jina-embeddings-v2-base-code` (768d, code-specialized).
- **GPU caveat (RTX 5070 Ti = Blackwell sm_120):** needs a recent TEI CUDA image
  (CUDA 12.8+); verify the service starts and serves a test embed on the GPU
  **before** reembedding (else ingestion fails). GPU passthrough requires Docker
  Desktop WSL2 + NVIDIA Container Toolkit. Same TEI infra also serves the
  second-stage reranker.

## Next steps (prioritized)

1. **Embedding upgrade (GPU, biggest expected quality lift).** Stand up TEI on
   the RTX 5070 Ti as the `openai_compatible` backend, validate health + a test
   embed, switch `config.yaml`, `wqm admin reembed --confirm`, then re-run
   `search_eval` / the benchmark before↔after. Pick model+dim (recommend
   `gte-large` 1024d, or `multilingual-e5-large` if PT matters).
2. ~~**Second-stage reranker (Phase 3 of the embedding/rerank plan).**~~
   **SHIPPED 2026-05-30** (`06d916852`) — daemon-side cross-encoder, but with
   **`JINARerankerV1TurboEn`**, not `bge-reranker-base` (~3s/query on CPU was too
   slow). Measured: top1 16.7→33.3%, top3 41.7→66.7%, recall@10 70.8%. **It did
   NOT reach `top-3 ≥ 80%`** — see [2026-05-30 outcomes](2026-05-30-rag-quality-session-outcomes.md):
   the remaining misses are *retrieval* failures, so the real lever is the
   embedding upgrade (BGE-large 1024d), not more rerank tuning. **Still TODO:**
   grow the dataset to >30 queries before trusting `PATH_BOOST_ALPHA=0.8`.
3. **Admin-UI benchmark panel (Phase B).** A panel over the `search_eval` path —
   edit dataset, tweak params, re-run, view per-query drill-down + verdict.
4. **Tree-sitter Gap #3 / #4.** LSP `definition` + `kind` precision and incremental
   parse + tree-diff re-embed (Large; `parse_incremental` exists but has no
   production caller). **Update (2026-05-31):** the Gap #3 blocker — "needs a live
   LSP server" — is now **removed** (servers bundled + protocol framing fixed, see
   [LSP outcomes](2026-05-31-lsp-session-outcomes.md)); `accc6db24` also added
   resolved CALLS edges via callHierarchy. The specific `definition: None` /
   substring-`kind` gap in `enrichment.rs` is still open but now host-validatable.
5. **Multi-clone tenant registration fix.** The registry/tenant knot that made
   `search` resolve an empty tenant — see the existing spawned task.

## Watch-outs / debt

- `PATH_BOOST_ALPHA=0.8` is tuned on 12 queries — revisit when the dataset grows.
- The committed `reports/semantic-search.json` is index-excluded; regenerate it
  from the host runner after meaningful changes (it is the tracked baseline).
- `search_eval` is an `internalTool` — a client must reconnect to see it.
