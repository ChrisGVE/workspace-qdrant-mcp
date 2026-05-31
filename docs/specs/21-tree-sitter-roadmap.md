## Tree-sitter Evolution Roadmap

This document tracks concrete improvements identified for the tree-sitter
subsystem after auditing the current implementation against the spec and
upstream feature set. Items are validated against the codebase (not
speculation) and ordered for sequential E2E delivery.

Companion specs: [07-code-intelligence.md](07-code-intelligence.md),
[11-grammar-runtime.md](11-grammar-runtime.md), [15-language-registry.md](15-language-registry.md).

### Context

Tree-sitter is the always-on code-intelligence baseline. Implementation lives
in `src/rust/daemon/core/src/tree_sitter/`:

- `chunker/` — `SemanticChunker` + `GenericExtractor` (pattern-driven walker)
- `grammar_*` — dynamic grammar download, cache, load, version check
- `parser/`, `languages/` — runtime parser wrappers
- `types.rs` — `SemanticChunk` payload type

The chunker is fed via `extract_chunks` / `extract_chunks_with_provider` and
its output is consumed downstream by:

- `strategies/processing/file/chunk_embed/` — embed → upsert to Qdrant
- `strategies/processing/file/lsp_payload.rs` — LSP enrichment merged into Qdrant payload
- `graph/extractor` — relationship edges (CALLS, CONTAINS, …) into the code graph
- `keyword_extraction/` and `tagging/` — keyword and tag pipelines

### Items considered and discarded

#### LSP enrichment fields on `SemanticChunk` — already done

Initial hypothesis was that the spec promises `references` / `type_info` on the
chunk payload but the code doesn't deliver. Verification:

- [enrichment.rs:36-162](../../src/rust/daemon/core/src/lsp/project_manager/enrichment.rs)
  computes `LspEnrichment { references, type_info, resolved_imports, definition, … }`
- [lsp_payload.rs:11-35](../../src/rust/daemon/core/src/strategies/processing/file/lsp_payload.rs)
  serializes all fields onto the Qdrant payload (`lsp_references`,
  `lsp_type_signature`, `lsp_imports`, `lsp_definition`, `lsp_enrichment_status`, …)

The architecture intentionally keeps `SemanticChunk` decoupled from LSP — the
two are merged at write time in the queue processor. **No gap. Item dropped.**

The only narrow gap inside this area is item #3 below (`definition` query
never issued, `kind` classifier uses naive string match on hover text).

### Validated gaps (execution order)

> **Status update — 2026-05-31 (re-audited against the code).** Items #1 and #2
> are SHIPPED; the per-item text below is kept for historical context. Items #3
> and #4 remain open, but #3's runtime blocker is now removed (see below).
>
> - **#1 Real tokenizer — DONE.** `tokenizer::estimated_token_count(text, Option<&ModelTokenizer>)`
>   drives the oversize gate in `chunker/splitting.rs::handle_oversized_chunks`
>   (with a post-split sanity warning); `SemanticChunker` carries an
>   `Option<Arc<ModelTokenizer>>` with a `with_tokenizer` builder; and the
>   production path wires the real tokenizer in via `default_tokenizer()`
>   (`document_processor/mod.rs:373`, `tree_sitter/mod.rs:243`). Covered by
>   `chunker/tests.rs` (with/without-tokenizer split cases). The legacy
>   `SemanticChunk::estimated_tokens()` (`len/4`) is now only referenced by a test.
> - **#2 Symbols → tagging — DONE.** `keyword_extraction/symbol_candidates.rs`
>   consumes `SemanticChunk` symbol names into the candidate merge
>   (`keyword_extraction/pipeline.rs`); see also `cooccurrence_graph.rs`.
> - **⚠️ Prerequisite regression fixed (`d59175369`, 2026-05-31).** Independent of
>   the numbered items: `ensure_grammar_available` was short-circuiting for EVERY
>   registry language (it checked `is_language_supported()` as a proxy for "has a
>   static grammar", but the v0.1.3 dynamic-grammar refactor emptied the static
>   set), so the chunker silently **text-chunked all code** and never reached
>   dynamic grammar loading. The code-relationship graph therefore got no symbol
>   nodes/edges. Now the real static predicate is checked → dynamic loading always
>   reached. This sits upstream of everything in this roadmap.
> - **#3 LSP `definition` + `kind` precision — OPEN, but unblocked.** `enrichment.rs`
>   still hardcodes `definition: None` and still classifies `kind` by
>   `type_signature.contains("fn ")…`. The historical caveat ("validating a fix
>   needs a live LSP server, not a host-only drop-in") is now **resolved**: the
>   2026-05-30/31 LSP frente bundled the servers and fixed the protocol framing, so
>   a live server is available to validate against (see
>   [../plans/2026-05-31-lsp-session-outcomes.md](../plans/2026-05-31-lsp-session-outcomes.md)).
>   `accc6db24` also landed resolved CALLS edges via callHierarchy — a related but
>   separate win (it does not populate the `definition`/`kind`/`container` fields).
> - **#4 Incremental parse + tree-diff re-embed — OPEN (groundwork only).**
>   `parser::parse_incremental(old_tree)` exists with a unit test but has NO
>   production caller; the watcher/`document_processor` still full-reparses and
>   re-embeds every chunk. Large effort (old-tree cache + changed-range → changed-
>   chunk-only re-embed).

#### 1. Real tokenizer in place of `content.len() / 4`

**Current state.** `SemanticChunk::estimated_tokens()`
([types.rs:224-226](../../src/rust/daemon/core/src/tree_sitter/types.rs))
returns `content.len() / 4`. Called once in production at
[splitting.rs:36](../../src/rust/daemon/core/src/tree_sitter/chunker/splitting.rs)
as the gate that decides whether a chunk must be fragmented.

A `ModelTokenizer` already exists at
[tokenizer.rs](../../src/rust/daemon/core/src/tokenizer.rs) wrapping the
HuggingFace `tokenizers` crate against the actual `tokenizer.json` from
all-MiniLM-L6-v2's FastEmbed cache. It is unused in the chunker pipeline.

**Gap.** The 4-chars-per-token heuristic underestimates real token count for:

- Identifier-heavy code (long camelCase / snake_case names → many subwords)
- Languages with frequent operators / punctuation
- Non-ASCII content (UTF-8 byte length ≠ character count ≠ token count)

Result: chunks pass the `< max_chunk_size` gate but actually exceed the
embedding model's window. The embedder either truncates silently or splits
sub-optimally, losing information.

**Plan.**

1. Add optional `tokenizer: Option<Arc<ModelTokenizer>>` field on `SemanticChunker`.
2. Builder method `with_tokenizer(tokenizer)`.
3. Thread tokenizer into `splitting::handle_oversized_chunks` so the gate uses
   real token count when available.
4. Keep `len/4` as fallback when no tokenizer is loaded (CI without HF cache,
   test environments).
5. Post-split sanity: if a produced fragment still exceeds `max_chunk_size`
   under the real tokenizer, log a warning with file/symbol/token-count.
6. Plumb tokenizer through the production entry path (`document_processor`
   → `process_file_content_with_provider`) so live ingestion uses it.

**Acceptance.**

- New unit test: a synthetic chunk with `content.len()/4 < max` but real
  token count `> max` is correctly fragmented.
- Existing tests pass (with `len/4` fallback when tokenizer unavailable).
- One end-to-end test indexing a real source file produces no warning about
  oversize fragments under the real tokenizer.

**Effort.** S. ~half-day.

#### 2. Tree-sitter symbols feed the auto-tagging pipeline

**Current state.** `tagging/` derives signal from:

- Path components ([tier1.rs](../../src/rust/daemon/core/src/tagging/tier1.rs):
  `extract_path_tags`)
- PDF metadata (`extract_pdf_metadata_tags`)
- Dependency manifests ([concepts.rs](../../src/rust/daemon/core/src/tagging/concepts.rs))
- Embedding taxonomy ([tier2.rs](../../src/rust/daemon/core/src/tagging/tier2.rs))
- Optional LLM ([tier3.rs](../../src/rust/daemon/core/src/tagging/tier3.rs))

`keyword_extraction/` adds:

- TF-IDF lexical candidates ([lexical_candidates/](../../src/rust/daemon/core/src/keyword_extraction/lexical_candidates))
- LSP-derived candidates from imports ([lsp_candidates/](../../src/rust/daemon/core/src/keyword_extraction/lsp_candidates))
- Structural tags from substring matching against framework lists ([structural_tags.rs](../../src/rust/daemon/core/src/keyword_extraction/structural_tags.rs))

`grep` confirms: **no file in `tagging/` or `keyword_extraction/` consumes
`SemanticChunk`, `symbol_name`, or `chunk_type`.** Tree-sitter symbols are
dropped on the floor by tagging.

**Gap.** Tree-sitter already extracts the cleanest possible concept signal —
public type names, class names, trait names, top-level function names — and
the tagging pipeline never sees it. TF-IDF gives raw terms (`tokio`, `serde`,
`reqwest`); symbol names give domain concepts (`AuthService`, `RetryPolicy`,
`HttpClient`) that map cleanly to the existing
[concepts.rs](../../src/rust/daemon/core/src/tagging/concepts.rs) normalization
dictionary.

The future-development spec
([14-future-development.md:420](14-future-development.md))
already flags this gap: *"TF-IDF produces terms, not concepts."*

**Plan.**

1. New module `keyword_extraction/symbol_candidates/` consuming `Vec<SemanticChunk>`
   (analog of `lsp_candidates/`).
2. Extract candidate signals from chunks: `symbol_name` for `Class`, `Struct`,
   `Trait`, `Interface`, top-level `Function` (not nested methods).
3. Normalize via `tagging/concepts.rs` (CamelCase → kebab, drop common suffixes
   like `Service` / `Manager` / `Handler` if configured, dedupe against path
   tags).
4. Wire output into `keyword_extraction/pipeline.rs` candidate merge step.
5. Add a config flag to disable the source (rollback safety).

**Acceptance.**

- A Rust file defining `struct AuthService` produces a candidate tag
  `auth-service` from the new module.
- The pipeline's final tag set on a sample project includes at least one
  symbol-derived tag absent from a baseline run with the source disabled.
- Concept normalization aligns symbol-derived tags with dependency-derived
  tags (e.g., `struct HttpClient` and `dependency: reqwest` both tagged
  `http-client`).

**Effort.** M. ~2 days.

#### 3. LSP `definition` + `kind` precision

**Current state.** In
[enrichment.rs:158](../../src/rust/daemon/core/src/lsp/project_manager/enrichment.rs)
and `:332`, the `LspEnrichment` returned by `perform_enrichment` and
`get_type_info` hardcodes `definition: None`. The `textDocument/definition`
LSP request is never issued.

In `parse_hover_response`
([enrichment.rs:419-431](../../src/rust/daemon/core/src/lsp/project_manager/enrichment.rs)),
the `kind` field is classified by string-matching the hover signature:

```rust
let kind = if type_signature.contains("fn ") || type_signature.contains("function") {
    "function"
} else if type_signature.contains("struct ") || type_signature.contains("class") {
    "class"
} // …
```

This is brittle: a docstring containing "function" misclassifies; rust-analyzer
hover may format types in ways that miss the heuristics.

`TypeInfo.container` (the enclosing class/module) is also always `None`.

**Gap.** Three fields advertised in the payload (`lsp_definition`, the `kind`
discriminator, `container`) are either always-null or noisy. Downstream
consumers (graph extractor, search ranking) can't rely on them.

**Plan.**

1. Issue `textDocument/definition` in `perform_enrichment`. Cache like
   references / hover already do.
2. Replace `kind` heuristic with `textDocument/documentSymbol` lookup at file
   open. Cache per-file. Use LSP `SymbolKind` enum mapped to our `kind`
   strings.
3. Populate `container` from the document-symbol parent chain.
4. Update payload tests in [lsp_payload.rs](../../src/rust/daemon/core/src/strategies/processing/file/lsp_payload.rs).

**Acceptance.**

- Indexed Rust file: at least one chunk has non-null `lsp_definition`.
- A function named "deconstruct" no longer gets classified as `"function"`
  via substring match on the docstring; it goes through `SymbolKind::Function`.
- `container` populated for methods (e.g., `AuthService` for `login`).

**Effort.** S–M. ~1 day.

#### 4. Incremental parsing + tree-diff re-embed

**Current state.** `grep` confirms no use of `Parser::parse_with(old_tree, …)`,
`Tree::changed_ranges`, or `set_included_ranges` anywhere in the codebase.
Every file change re-parses from scratch and (downstream) re-embeds every
chunk in the file.

**Gap.** Two compounding costs:

1. *Re-parse cost.* Tree-sitter's killer feature is `O(edit_size)` re-parsing
   when an `old_tree` is provided. Today every save = full reparse.
2. *Re-embed cost.* When one function changes in a 50-function file, the
   pipeline re-embeds all 50 chunks. Embedding is the most expensive step
   ([04-write-path.md] flow).

Live-editing latency and CPU/IO on large repos both suffer.

**Plan.** (Larger; may be staged.)

Stage 4a — incremental parsing:

1. Cache `Tree` per `(tenant_id, branch, relative_path, file_hash)` in
   `ProcessingContext` (LRU, bounded).
2. On file change, call `parse_with(old_tree, new_source, ...)` if the cache
   has the prior tree; otherwise full parse.
3. Cache invalidation on branch switch / file rename / cache eviction.

Stage 4b — tree-diff re-embed:

1. Compute `old_tree.changed_ranges(&new_tree)` and convert to affected
   `(start_line, end_line)` ranges.
2. Re-extract chunks only for affected ranges.
3. For unaffected chunks, reuse the existing Qdrant point ID (skip embedding,
   skip upsert).
4. Update graph edges only for symbols whose chunk changed.

**Acceptance.**

- New bench: 100KB Rust file with 1-line edit. Re-parse latency drops ≥5×
  versus full parse.
- Pipeline integration test: editing one function in a multi-function file
  re-embeds only the affected chunk (count via metrics).
- No regression in full-file ingestion benchmark.

**Effort.** L. ~1 week, can be staged.

#### 5. `.scm` queries replacing the imperative walker

**Current state.** The `GenericExtractor` walks the AST in
[walker.rs](../../src/rust/daemon/core/src/tree_sitter/chunker/generic_extractor/walker.rs)
with a cascade of `matches_any(kind, &patterns.X.node_types)` checks
(`classify_node` at lines 22-59) plus dedicated handlers for decorated nodes
and Elixir-style `call` nodes.

The bundled YAML (`language_registry.yaml`) lists node types per chunk
category, but the dispatch logic is in Rust.

**Gap.** Three:

1. *Maintenance.* Every quirk per language (Elixir's `call` shape, Python's
   `decorated_definition`, …) lives in `walker.rs` as imperative special-cases.
   Adding a new language is "YAML + maybe a walker tweak."
2. *Free upstream queries.* `nvim-treesitter` ships `.scm` query files per
   grammar covering `highlights.scm` (syntax categories) and `locals.scm`
   (lexical scope: definitions, references, free identifiers). These are
   curated by language maintainers.
3. *Lexical scope for free.* `locals.scm` resolves intra-file identifier
   scoping without LSP. This means: better `calls` extraction (distinguish
   real function calls from type annotations or strings), better keyword
   extraction (skip local variable names), better symbol-graph edges.

**Plan.**

1. Extend the YAML schema with an optional `queries:` block per language
   pointing to `.scm` file paths (bundled alongside the YAML, or fetched
   from nvim-treesitter on grammar download).
2. New `QueryExtractor` implementing `ChunkExtractor` using
   `tree_sitter::Query` with captures named `@function.def`, `@class.def`,
   `@local.scope`, etc.
3. For each bundled language, port the existing imperative dispatch into a
   query. Validate parity via existing chunker tests.
4. Use `locals.scm` to improve `calls` extraction (helpers.rs).
5. Keep `GenericExtractor` walker as fallback for languages without queries.

**Acceptance.**

- 5 representative languages (Rust, Python, TypeScript, Go, Elixir) ported to
  queries; chunker tests still pass.
- New test: `calls` field for a Rust function with a local variable named the
  same as a free function excludes the local (via `locals.scm`).
- Walker code reduced; YAML `semantic_patterns` block can be `null` when
  queries are provided.

**Effort.** L. ~1 week.

### Sequencing rationale

1. **#1 (tokenizer)** first because it is small, contained, fixes a real bug
   (silent embedding-window violations), and the `ModelTokenizer` infra
   already exists.
2. **#2 (symbols → tagging)** second because it is a self-contained additive
   module (new `symbol_candidates/` analogous to `lsp_candidates/`) and
   directly closes the gap that the future-development spec already calls out.
3. **#3 (LSP refinements)** third because it is small, lives in already-tested
   `enrichment.rs`, and unblocks better signal for the graph extractor.
4. **#4 (incremental + tree-diff)** fourth because it is the highest leverage
   on long-term CPU/latency but the largest refactor — needs the prior items
   landed and stable.
5. **#5 (`.scm` queries)** last because it is a structural refactor with the
   widest blast radius and benefits most from the lexical-scope foundation
   for the items above.

### Out of scope (covered elsewhere)

- Co-occurrence graph for concept extraction — deferred to LadybugDB v2, see
  [14-future-development.md:446](14-future-development.md).
- Per-project `.wqmconfig.yaml` overrides for grammar / patterns — see
  [15-language-registry.md:66](15-language-registry.md).
- New language support — handled by the language registry YAML workflow.

### Tracking

Each item is tracked as a separate task in the harness. Status, acceptance
test runs, and follow-up findings are recorded against the task; this
document is the agreed-upon plan, not the live status.

---

**Version.** 1.0.0
**Created.** 2026-05-26
