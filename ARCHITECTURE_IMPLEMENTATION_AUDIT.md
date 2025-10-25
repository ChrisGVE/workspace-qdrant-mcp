# Workspace Qdrant MCP – Architecture & Implementation Audit

## Scope & Method

- Reviewed the mixed Python/Rust codebase under `/Users/chris/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp`, focusing on architectural alignment, component integration, and implementation maturity.
- Read primary documentation (`README.md`, `docs/ARCHITECTURE.md`, protocol and configuration specs) to understand intended design.
- Inspected key Python modules (`workspace_qdrant_mcp`, `common`, `wqm_cli`), the Rust daemon sources (`rust-engine-legacy`, `src/rust/daemon` workspace), and configuration assets.
- Did not execute or modify code; the assessment is based on static analysis only.

## Architectural Overview

- The project targets a four-component architecture (MCP server, Rust daemon “memexd”, CLI, shared storage) as documented in `docs/ARCHITECTURE.md`.
- Python side provides the MCP entry point (`workspace_qdrant_mcp/server.py`) with four tools (store/search/manage/retrieve) that should delegate writes to the daemon (`store` docstring at `src/python/workspace_qdrant_mcp/server.py:368`).
- Rust side currently has two parallel implementations: `rust-engine-legacy` (legacy, built into the `memexd` binary) and a newer modular workspace under `src/rust/daemon` (`workspace-qdrant-core`, `workspace-qdrant-grpc`, Python bindings).
- Configuration is centralized through `common/core/config.py`, using the comprehensive `assets/default_configuration.yaml`.

## Component Findings

### MCP Server (`workspace_qdrant_mcp/server.py`)

- Global initialization of Qdrant client, embedding model, and daemon client happens inside `initialize_components()`. The Qdrant client is synchronous; its methods (`search`, `scroll`, `upsert`) are invoked directly inside `async` tool handlers (`src/python/workspace_qdrant_mcp/server.py:607-664`, `src/python/workspace_qdrant_mcp/server.py:482-509`), which can block the asyncio event loop and degrade tool responsiveness.
- The `store` tool is designed to call `DaemonClient.ingest_text` first (`src/python/workspace_qdrant_mcp/server.py:456-475`). When the daemon is unavailable, it silently falls back to direct Qdrant writes (`src/python/workspace_qdrant_mcp/server.py:482-519`), undermining the “daemon-only writes” principle and yielding divergent metadata handling (no chunking, embeddings generated inline).
- For project collections the code sets `collection_basename = ""` while keeping the tenant hash as the suffix (`src/python/workspace_qdrant_mcp/server.py:439-446`). The new Rust `DocumentService` explicitly rejects empty basenames (`src/rust/daemon/grpc/src/services/document_service.rs:109-123`), so the nominal “daemon-first” path would fail even if the service were running—forcing every project write into fallback mode.
- Hybrid search combines vector similarity with a manual substring scan over scroll results (`src/python/workspace_qdrant_mcp/server.py:614-666`). This provides limited ranking quality (no explicit RRF, simple term frequency) and is computationally heavy due to full payload scans; sparse vector support is absent.

### Shared Python Core (`src/python/common`)

- `common/core/daemon_client.py` still depends on legacy ingestion stubs while partially adopting the new `workspace_daemon` protocol. Service discovery is currently disabled and always falls back to static host/port (`src/python/common/core/daemon_client.py:154-185`).
- `DaemonClient.ingest_text` expects a functioning `DocumentService` (`src/python/common/core/daemon_client.py:306-365`), but the Rust daemon compiled from `rust-engine-legacy` does not expose that service, leading to the fallback discussed above.
- There are two gRPC client implementations: `common/core/daemon_client.py` and a richer `common/grpc/daemon_client.py` with connection pooling (`src/python/common/grpc/daemon_client.py:1-128`). The duplication increases maintenance cost and risks divergent behaviour (exceptions, retries).
- `common/core/pure_daemon_client.py` imports a `DaemonError` symbol that is not defined in the referenced module (`src/python/common/core/pure_daemon_client.py:16` vs. `src/python/common/core/daemon_client.py`), so that helper cannot be imported without raising `ImportError`.
- Positive: `common/core/config.py` implements a robust, dictionary-based configuration loader with unit conversions (`src/python/common/core/config.py:200-270`), backed by the exhaustive defaults file (`assets/default_configuration.yaml:1-120`).

### `wqm` CLI (`src/python/wqm_cli`)

- The CLI exposes a large surface area (admin, ingestion, lsp, observability, etc.) with consistent logging suppression for MCP environments.
- Service management assumes a `memexd` binary located under user paths; `wqm service install --build` compiles from `rust-engine-legacy` (`src/python/wqm_cli/cli/commands/service.py:49-210`). The binary produced by `rust-engine-legacy` is `workspace-qdrant-daemon` (`rust-engine-legacy/Cargo.toml`), so the default build path does not match the expected filename, leading to install failures unless renamed.
- Several command groups depend on daemon methods that are only stubbed in the Rust implementation (e.g., queue and tool monitoring).

### Rust Daemon – Legacy (`rust-engine-legacy`)

- Only the `SystemService` gRPC endpoint is registered; other services remain commented out (`rust-engine-legacy/src/grpc/services/mod.rs:3-13`). Consequently, Python clients cannot reach collection or document operations.
- `SystemService` returns canned metrics instead of live health data (`rust-engine-legacy/src/grpc/services/system_service.rs:42-118`), limiting observability.
- Document processing builds placeholder embeddings and writes everything into a fixed collection (`rust-engine-legacy/src/daemon/processing.rs:143-218`). There is no integration with a real embedding model, no chunking control, and collection naming ignores project metadata.
- File-watcher integration is scaffolded but not wired to ingestion queues (`rust-engine-legacy/src/daemon/watcher.rs`), and queue-related modules exist but are unused.
- Runtime initialization wires auto-watch creation from configuration (`rust-engine-legacy/src/daemon/mod.rs:134-188`), yet failure paths only log and proceed; there is no backoff strategy or health accounting.

### Rust Daemon – Modular Workspace (`src/rust/daemon`)

- `workspace-qdrant-core` provides richer primitives for storage, queueing, and embedding (e.g., configurable Qdrant client with HTTP/gRPC switching and HTTP/2 tuning in `src/rust/daemon/core/src/storage.rs:301-399`).
- `workspace-qdrant-grpc` implements the full `DocumentService` with chunking, metadata enforcement, and deterministic embeddings (`src/rust/daemon/grpc/src/services/document_service.rs:1-200`), and mirrors the planned `CollectionService`/`SystemService`.
- Python bindings exist (`src/rust/daemon/python-bindings`), but no packaging script ties them into the Python runtime.
- This newer workspace is not referenced by the CLI or the Python MCP components; the build tooling still targets `rust-engine-legacy`, so the mature modules remain unused.

## Integration & Data Flow Observations

- There is a fundamental protocol mismatch: Python expects the new gRPC services while the deployed daemon exposes only the legacy interface. This causes immediate fallback to direct Qdrant writes for MCP operations.
- Even with the new DocumentService, the parameter contract differs (empty basenames from Python vs. required non-empty basenames in Rust), so ingestion would still fail without additional mapping logic.
- Multiple Python clients (`common/core`, `common/grpc`) and the HTTP server (`src/python/workspace_qdrant_mcp/http_server.py`) reference different daemon interfaces, risking inconsistent behaviour and duplicated retry logic.
- The “single writer” policy is compromised by direct `qdrant_client` fallbacks and by the Rust DocumentProcessor writing to a hard-coded collection, which can introduce data fragmentation during daemon outages.

## Configuration & Deployment

- Configuration discovery is thorough, but defaults assume production-grade installation paths. Development mode requires toggling `deployment.develop` or setting `WQM_TEST_MODE`; otherwise, asset discovery falls back to system directories (`assets/default_configuration.yaml`).
- CLI build/deploy scripts point at `rust-engine-legacy`, while the new Rust workspace has separate tooling (`src/rust/daemon/justfile`, benches). Aligning these build pipelines is necessary before packaging.
- Service management scripts presume launchd/systemd integration; Windows service handling is stubbed out.

## Testing & Quality Signals

- The repository contains an extensive test suite (`tests/`), with dedicated folders for CLI, daemon, MCP server, security, etc. However, many tests rely on gRPC behaviours that are not yet implemented in the running daemon (e.g., DocumentService in `tests/mcp_server/test_mcp_transaction_atomicity.py`).
- Coverage artefacts (`coverage_baseline.lcov`, `coverage.xml`) suggest automated runs, but given the implementation gaps, several tests likely skip or stub critical flows.
- Rust crates include numerous unit tests, yet the placeholder logic (e.g., `DocumentProcessor`) indicates they primarily validate scaffolding rather than production ingestion.

## Security & Observability

- LLM access control is thoughtfully implemented (`src/python/common/core/llm_access_control.py`), preventing accidental collection mutations by the MCP.
- Observability hooks (log suppression in MCP mode, metadata on writes, queue metrics) are planned but not fully wired; the SystemService currently returns static health data, and no metrics export is available yet.
- Service discovery infrastructure is planned but disabled, so multi-project isolation depends on manual configuration.

## Key Risks

- **Protocol divergence** – Python components rely on gRPC services that the compiled daemon does not provide, making the “daemon-first” architecture non-functional today.
- **Inconsistent ingestion paths** – Direct Qdrant fallbacks and mismatched collection naming risk data duplication and untracked writes.
- **Fragmented Rust code bases** – Maintaining both `rust-engine-legacy` and the newer `workspace-qdrant-*` crates dilutes effort and confuses build tooling.
- **Blocking operations in async tools** – Synchronous Qdrant calls inside async endpoints can stall the MCP event loop under load.
- **Broken helper modules** – Import errors in `pure_daemon_client` and duplicated gRPC clients will surprise downstream integrators.
- **Deployment mismatch** – CLI expects a `memexd` binary while the build emits `workspace-qdrant-daemon`, preventing out-of-the-box service management.

## Recommendations & Next Steps

1. **Unify the daemon implementation**: Decide between `rust-engine-legacy` and the `src/rust/daemon` workspace. Expose the full `workspace_daemon` protocol (DocumentService, CollectionService) and align the build outputs with CLI expectations.
2. **Fix ingestion contract**: Update Python’s collection naming logic to pass a meaningful basename (e.g., “project”) or adjust the Rust service to accept project-only identifiers. Add integration tests validating end-to-end ingestion over gRPC.
3. **Eliminate direct writes**: Once the daemon path is stable, remove or severely gate the direct Qdrant fallback. Introduce explicit degradation modes and telemetry so the team is alerted when writes bypass the daemon.
4. **Async-friendly Qdrant access**: Wrap Qdrant operations in executors or adopt the async client to avoid blocking the FastMCP event loop during searches and upserts.
5. **Consolidate daemon clients**: Merge `common/core/daemon_client.py` and `common/grpc/daemon_client.py`, provide a single error taxonomy, and ensure helper modules import correctly.
6. **Align packaging & naming**: Ensure the Rust build outputs `memexd` (or update CLI expectations), and wire the new Rust workspace into the packaging pipeline if it replaces `rust-engine-legacy`.
7. **Complete observability**: Implement real health metrics in SystemService, and surface queue/ingestion statistics through both CLI and MCP manage tools.
8. **Tighten tests**: Add cross-language tests that exercise the actual gRPC contract, and mark placeholder implementations (e.g., hash-based embeddings) with TODOs or feature flags to avoid misleading coverage metrics.
