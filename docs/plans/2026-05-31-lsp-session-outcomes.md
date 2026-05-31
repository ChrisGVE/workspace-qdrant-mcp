# LSP & Code-Intelligence Session — 2026-05-30/31 Outcomes

The code-intelligence frente that ran alongside the RAG-quality work (recorded
separately in [2026-05-30-rag-quality-session-outcomes.md](2026-05-30-rag-quality-session-outcomes.md)).
LSP went from **"wired in code but a no-op at runtime"** (servers never spawned in
the container, protocol framing broken, grammars silently not loading) to
**bundled, framed correctly, observable, and verified green per language**.

Companion docs: [07-code-intelligence.md](../specs/07-code-intelligence.md),
[21-tree-sitter-roadmap.md](../specs/21-tree-sitter-roadmap.md),
[project memory `project_lsp_spawn_on_register`].

## Shipped (main)

| Commit | Change | Verification |
|---|---|---|
| `d59175369` | **Grammar-download fix** — registry languages now reach dynamic loading instead of silently text-chunking | reasoned + graph repopulates |
| `ba2413d51` | **Bundle dart/R/C/Java/Python LSP servers** in the memexd image + fix server launch args | dashboard |
| `accc6db24` | **callHierarchy resolved CALLS edges** (#3) + enrichment tracing/instrumentation | `lsp_real_trace` (real rust-analyzer) |
| `4e67aa099` | **LSP Prometheus metrics + Grafana dashboard** | live panels |
| `e3ccd5347` | **Config tolerance** — minimal/partial YAML no longer crash-loops the daemon | regression tests |
| `2c9c15a90` | **R languageserver compiled in a dedicated build stage** (was silently absent) | build hard-fails if not importable |
| `56306a3f6` | **Dashboard fix** — LSP panel grey = *idle*, not stopped/error | panel description + v2 |
| `6982277bf` | **LSP header-framing fix** + prefer `pyright-langserver` | `read_message` unit tests (2 passed) |
| `9cfe426b7` | **Working jdtls (Java LSP)** — Temurin 21 JRE + writable config + hard-fail install | java goes green |
| `6e161c394` | **Auto-register the self-repo for LSP** via `WQM_REPO_DIR` | fresh session → 7 servers green |

### 1. The chunking regression that broke the whole graph (`d59175369`)
`ensure_grammar_available` used `is_language_supported()` (true for any
*downloadable* registry language) as a proxy for "has a statically compiled
grammar". The v0.1.3 dynamic-grammar refactor **emptied the static set**
(`get_static_language` always returns `None`), so this short-circuited for
**every** registry language: the chunker ran with no grammar and silently
text-chunked all code, and the dynamic download below was unreachable dead code.
**Net effect:** the code-relationship graph never received symbol nodes/edges.
Fixed by checking the real static predicate so dynamic loading is always reached.
This is the root cause behind much of the graph being empty — it sits *upstream*
of the stub-edge resolution work from the RAG session.

### 2. Bundled language servers + launch args (`ba2413d51`)
The container shipped no language servers, so every LSP server sat grey/red
regardless of registration. Bundled dart, R, C (clangd), Java (jdtls), Python
(pyright-langserver) into the memexd image and corrected the per-server launch
arguments. (R and Java needed follow-ups — see `2c9c15a90` and `9cfe426b7`.)

### 3. callHierarchy → resolved CALLS edges (`accc6db24`)
Adds `resolved_outgoing_calls` (`prepareCallHierarchy` + `outgoingCalls`) and a
pure `resolved_call_edges` helper that builds CALLS edges to the callee's **real**
`node_id` (relativized vs project root) instead of name-only tree-sitter stubs.
Wired into `ingest_graph_edges` behind an **LSP-ready gate** (no-op on cold index).
Also adds tracing spans (`lsp.enrich_chunk/references/type_info/outgoing_calls`)
for OTLP/Tempo and fixes the enrich query column that was always 0 (→ empty
enrichment). This is the *resolution-side* counterpart to the name-based stub
resolver: where the daemon has a live LSP server, calls resolve precisely; where
it doesn't, the 120s background name resolver still heals stubs heuristically.

### 4. Protocol framing — the bug that kept jdtls/dart/pyright red (`6982277bf`)
Servers that frame responses with a `Content-Type` header **alongside**
`Content-Length` (jdtls, dart, pyright) broke the stdout reader: it assumed a
single `Content-Length` header line, so the `Content-Type` line was consumed as
the header/body separator. `read_exact` then started two bytes into the real
separator and truncated the body — on the large `initialize` response this
surfaced as `EOF while parsing an object` + a spurious `Timeout: LSP initialize`,
leaving those servers permanently red. Fix: `read_message()` loops the header
block until the blank separator (any number/order of headers tolerated; only
`Content-Length` parsed, case-insensitively), then reads exactly that many body
bytes. Generic over the reader, so framing is unit-testable without spawning a
process. Also dropped bare `pyright` from Python detection — it is the CLI
type-checker; only `pyright-langserver` speaks LSP over stdio.

### 5. jdtls actually runs now (`9cfe426b7`)
Three compounding problems left Java permanently not-running: (a) the pinned
milestone tarball now 404s, so the best-effort download silently shipped an
**empty** `/opt/jdtls`; (b) current JDT bundles require Java 21 while the image
shipped Java 17; (c) the wrapper pointed Eclipse at a root-owned read-only
config with no `-data` workspace dir. Fixed by downloading from the always-present
`latest` snapshot **and hard-failing** if the launcher JAR / `config_linux` are
missing, shipping a Temurin 21 JRE, and a `jdtls-wrapper.sh` that uses a writable
per-user config copy + per-workspace `-data` dir. Verified: registering a Java
project spawns jdtls and it reports "started successfully" within the init timeout.

### 6. Self-repo auto-registration (`6e161c394`)
The containerized MCP couldn't detect its own checkout (runs from `/app`;
`WQM_PROJECT_ROOT` points at the bind-mounted dev *root*, which has no project
marker), so its own LSP servers never spawned until a manual `store(type="project")`.
`ensureSelfRepoRegistered()` registers the checkout on each session init via
`WQM_REPO_DIR` (its exact container path) with `register_if_new` — idempotent,
best-effort, independent of the connecting client's project. Verified: a fresh
session brings up 7 servers (c, cpp, go, javascript, python, rust, typescript) green.

## Observability
- New `docker/grafana/dashboards/lsp.json`: server-state-per-language, enrichment
  throughput by status, success-rate panels.
- New collectors: `memexd_lsp_{server_state,enrichments_total,available_languages,active_servers}`,
  refreshed by a 30s background task; wired into shutdown.
- **Reading the dashboard:** grey = *idle* (binary installed, no active project
  uses that language) — **not** broken. Green = a registered project is using it.
  A memexd restart clears all LSP state → all grey until projects re-register; the
  file reconcile does NOT spawn servers, only `register_project` does.

## What this unblocks / still remains
- **Tree-sitter roadmap Gap #3 (`definition`/`kind` precision):** the historical
  blocker — "validating a fix needs a live LSP server" — is now **removed** (servers
  are bundled, framed correctly, and verified green). The specific gap (the
  `definition: None` hardcode and the substring `kind` heuristic in `enrichment.rs`)
  is still open, but is now a host-validatable drop-in rather than blocked.
- **Multi-clone tenant knot still caps LSP graph resolution:** LSP registers under
  the canonical `367157a01d98` while indexing runs under the legacy
  `local_5288aa13ad6c`, so `is_server_ready_for_file` never matches for the dev
  tenant. The resolved-call-edge path is correct but only fires where registration
  and indexing share a tenant. This is the same knot that caps the RAG graph ceiling.
- **Dirty-tracking for the stub resolver** (so it doesn't re-scan unresolvable
  stdlib danglers every 120s) — carried from the RAG session's follow-ups.

## Lessons learned
1. **"Silently absent" is the recurring failure mode.** The grammar short-circuit,
   the empty jdtls install, and the missing R package all *passed* (returned
   None / shipped an empty dir / detected the binary) while delivering nothing.
   The fixes all added a **hard-fail** so a broken install can't pass silently again.
2. **Verify the runtime gate before declaring LSP "done."** The code was correct
   for months; it never ran because of framing, missing binaries, and a Java
   version mismatch — none visible without a live server. (Same theme as the
   embedding Blackwell gate and the missing `cargo` for rust-analyzer.)
3. **Grey ≠ broken.** Encode operational meaning into dashboards or every restart
   reads as an outage.
