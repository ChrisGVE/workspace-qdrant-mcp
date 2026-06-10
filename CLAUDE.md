# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

workspace-qdrant-mcp (v0.1.3) is a Model Context Protocol (MCP) server providing project-scoped Qdrant vector database operations with hybrid search capabilities.

**Core Architecture:**
- **MCP Server**: TypeScript
- **Daemon**: Rust (high-performance file watching and processing)
- **CLI**: Rust (high-performance command-line interface)

**Core Features:**
- 10 MCP tools: store, search, rules, retrieve, grep, list (core) + graph, embedding, search_eval, workspace_index
- Hybrid search combining dense (semantic) and sparse (keyword) vectors
- Automatic project detection with Git integration
- Behavioral rules via persistent rules collection
- Rust daemon for high-performance file watching and processing

**v0.1.3 Features:**
- Dynamic Language Registry: 44 bundled languages defined in YAML, zero hardcoded language support. GenericExtractor replaces 25 per-language extractors. Upstream providers (Linguist, nvim-treesitter, mason-registry, tree-sitter-grammars org) for metadata refresh.
- Tree-sitter semantic code chunking by function/class/method (dynamic grammar auto-download)
- LSP integration for code intelligence (per-project, active projects only)
- Branch lifecycle management (create, delete, rename, default tracking)
- Project ID disambiguation for multi-clone repositories
- Enhanced folder move detection with notify-debouncer-full
- Path validation for orphaned project cleanup
- Code relationship graph (SQLite CTEs) with algorithms: PageRank, community detection, betweenness centrality
- Graph CLI (`wqm graph`) with 7 subcommands: query, impact, stats, pagerank, communities, betweenness, migrate

**Critical Design Files:**
1. **FIRST-PRINCIPLES.md** - Core architectural principles (in project root)
2. **docs/specs/** - Modular specifications (16 spec files, including 15-language-registry.md)
3. **research/languages/** - Language research (LSP, Tree-sitter, 500+ languages)
4. **assets/default_configuration.yaml** - Default system configuration
5. **.taskmaster/docs/** - Additional PRD documents for specific features

Backward compatibility is not necessary as this project is a work in progress and has not been released.

## Refactoring Scope

When modifying files that exceed the code size limits (see global CLAUDE.md), refactor gradually by extracting the section being changed into its own module.

To identify oversized files, use `LargeCode --root <project root> --output <project root>/tmp/largecode.csv` or `wc -l` on specific files. The limits are defined in the global CLAUDE.md (Rust: 500 lines/file, 80 lines/function).

All Critical (>2000) and High (1000-2000) files have been split across arch-refactor rounds 1-3. Remaining oversized files are in the 500-760 line range.

## Development Commands

### Build and Test

**This fork is container-first.** Every build runs *inside Docker* — you do NOT
need a local Rust/cargo toolchain, a local ONNX Runtime, or a host `npm` build.
`docker/Dockerfile.memexd` builds the daemon (Rust + statically-linked ONNX) and
the root `Dockerfile` compiles the TypeScript MCP server and the Rust node addon.

Use the Makefile that matches your shell:

- **Linux / WSL** → `make <target>` (top-level `Makefile`, bash + docker compose)
- **Windows / PowerShell** → `make -f Makefile.win <target>`

```bash
# ── Linux / WSL (run from inside the WSL distro, native ext4) ──
make first-time        # from scratch: create db volume + build + up + hooks + status
make redeploy          # after code changes / git pull: rebuild + recreate mcp+memexd
make stack-status      # compose ps + ping admin/qdrant/daemon
make stack-logs        # tail mcp + memexd logs
make reindex           # force-reindex every watched project (admin API)
make reindex-status    # per-project indexing progress
make help              # all targets

# Under the hood `redeploy` is just:
docker compose --env-file docker/.env -f docker-compose.yml build mcp memexd
docker compose --env-file docker/.env -f docker-compose.yml up -d --force-recreate mcp memexd
```

The daemon's watch root (the projects it indexes) is `WQM_DEV_ROOT` in
`docker/.env`. On WSL point it at a native ext4 path; on Windows use a host path
— see the "Watch root" notes in `docker/.env.example`.

**Advanced — native Rust build (optional, not required for normal dev/deploy).**
Only needed if you are iterating on the daemon outside Docker and have a local
Rust toolchain + ONNX Runtime set up (see "ONNX Runtime Build Requirements"
below). The macOS/Homebrew `launchctl` deploy from upstream does not apply to the
container flow.

```bash
# Run the Rust unit tests against a local toolchain (ORT_LIB_LOCATION points at
# your static ONNX libs — path is machine-specific, not the upstream default):
ORT_LIB_LOCATION=<your-onnxruntime-static>/lib cargo test --manifest-path src/rust/Cargo.toml --package workspace-qdrant-core
# Filters: git_integration, watching, tree_sitter, keyword, tag, graph, schema_version, etc.
```

### CLI Quick Reference

Use `wqm --help` and `wqm <subcommand> --help` for full usage. Key commands:

```bash
wqm service status                # Check daemon status
wqm status health                 # Health diagnostics
wqm queue list / stats            # Queue monitoring
wqm project watch pause / resume  # File watcher control
wqm admin perf                    # Pipeline performance stats
wqm admin collections list        # Collection management
wqm admin rebuild all             # Rebuild all indexes
wqm graph stats --tenant <t>      # Graph subsystem
wqm init completions zsh          # Shell completions
wqm init man install              # Install man pages
wqm init hooks install            # Install Claude Code hooks
```

## Code Architecture

### Project Structure

Use `workspace-qdrant` MCP's `list` tool with `format: "summary"` to explore the current project structure. The auto-detected Cargo components are:

| Component | Path | Description |
|-----------|------|-------------|
| `cli` | `src/rust/cli` | Rust CLI binary |
| `common` | `src/rust/common` | Shared types (wqm-common crate) |
| `common-node` | `src/rust/common-node` | Node.js native bridge |
| `daemon.core` | `src/rust/daemon/core` | Core library |
| `daemon.grpc` | `src/rust/daemon/grpc` | gRPC service layer |
| `daemon.memexd` | `src/rust/daemon/memexd` | Daemon binary |
| `daemon.shared-test-utils` | `src/rust/daemon/shared-test-utils` | Test utilities |

TypeScript MCP server: `src/typescript/mcp-server/src/`

### Key Design Invariants

These are **guardrails** — consult `docs/specs/` for full design details.

- **Daemon owns all persistent state**: Qdrant writes and SQLite schema are daemon-only (see `docs/specs/04-write-path.md`, ADR-003)
- **All DB operations within transactions**: No bare queries outside a transaction
- **4 canonical collections** (ADR-001): `projects` (by `tenant_id`), `libraries` (by `library_name`), `rules`, `scratchpad`
- **Enqueue-only gRPC pattern**: gRPC handlers enqueue to unified queue; queue processor handles mutations (see `docs/specs/04-write-path.md`)
- **Idempotency key**: `SHA256(item_type|op|tenant_id|collection|payload_json)[:32]` — same algorithm in TypeScript, Rust daemon, and Rust CLI
- **MCP tools**: store, search, rules, retrieve, grep, list (see `docs/specs/08-api-reference.md`)
- **Hybrid search**: Dense (FastEmbed) + Sparse (BM25/IDF persisted in SQLite) + RRF fusion

### ONNX Runtime Build Requirements

> **Not needed for the container-first flow.** The Docker images already vendor a
> statically-linked ONNX Runtime, so `make redeploy` / `docker compose build`
> require none of the steps below. This section applies **only** if you opt into
> the advanced native Rust build on the host.

**Option 1: Pre-built static library (Recommended)**

```bash
mkdir -p ~/.onnxruntime-static
curl -L "https://github.com/supertone-inc/onnxruntime-build/releases/download/v1.23.2/onnxruntime-osx-universal2-static_lib-1.23.2.tgz" \
  -o ~/.onnxruntime-static/ort.tgz
tar xzf ~/.onnxruntime-static/ort.tgz -C ~/.onnxruntime-static
rm ~/.onnxruntime-static/ort.tgz
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release
```

**Option 2: Homebrew (dynamic, dev only)**

```bash
brew install onnxruntime
ORT_LIB_LOCATION=/usr/local/Cellar/onnxruntime/1.23.2_2 ORT_PREFER_DYNAMIC_LINK=1 cargo build --release
```

See https://ort.pyke.io/setup/linking for detailed instructions.

### gRPC Services

7 services defined in `src/rust/daemon/proto/workspace_daemon.proto`:
SystemService, CollectionService, DocumentService, EmbeddingService, ProjectService, TextSearchService, GraphService.

## Environment Configuration

**Required for Qdrant:**
- `QDRANT_URL` - Server URL (default: http://localhost:6333)
- `QDRANT_API_KEY` - API key (required for Qdrant Cloud)

**Embeddings (full reference: `docs/deployment/embeddings.md`):**
- The reference deployment uses `WQM_EMBEDDING_PROVIDER=openai_compatible`
  with `intfloat/multilingual-e5-large` (1024d) served by in-stack backends:
  GPU (Infinity, preferred) + CPU (TEI, warm standby), selected via
  `COMPOSE_PROFILES=embeddings-cpu[,embeddings-gpu]` in `docker/.env` with
  automatic daemon-side failover (`embedding.fallback_base_url`).
- The GPU backend requires the NVIDIA Container Toolkit on the Docker
  engine (containers cannot see the GPU otherwise — `nvidia-smi` working in
  WSL is not sufficient). Switching CPU↔GPU never requires a reembed —
  vectors are model-bound. Changing the model/dim always does.
- `model`/`output_dim`/prefixes are config-file only (`state/memexd/config.yaml`);
  memexd must be started with `--config /etc/wqm/config.yaml` (the compose
  command does this) or the file is silently ignored.
- `WQM_EMBEDDING_PROVIDER=fastembed` remains the zero-dependency fallback
  (in-process, pinned to all-MiniLM-L6-v2 384d).

**Optional:**
- `WQM_DATABASE_PATH` - Override database path
- `WQM_LOG_LEVEL` - Log level (DEBUG, INFO, WARN, ERROR)

## Task Master AI Instructions

**Import Task Master's development workflow commands and guidelines.**
@./.taskmaster/CLAUDE.md

## Task Master Workflow Preferences

- **Task Expansion**: Always expand tasks with `research=false` to avoid unnecessary API calls and delays
- **Task Execution**: Execute plans continuously, only stopping for disambiguation or non-obvious design decisions
- **Design Decisions**: Make obvious design decisions that align with project plan and requirements autonomously
- **Agent Execution Mode**: Execute agents SEQUENTIALLY (one at a time), not in parallel, to conserve API usage

## Critical Development Rules

**NO MIGRATION EFFORT**: This project requires NO migration effort of any kind. The project is a work in progress and has not been released - there are no users to migrate.

- **DO NOT CHANGE THE MCP CONFIGURATION**: Never modify MCP server configuration without explicit permission
- **NO .mcp.json FILE**: The MCP server is already installed system-wide. Do not create `.mcp.json` in this project. If the server needs reconnecting, the user will handle it manually.
- **Server Stability**: Always verify server starts without crashes after changes
- **Git Discipline**: Follow strict temporary file naming conventions (YYYYMMDD-HHMM_name format)
- **Atomic Commits**: Make focused, single-purpose commits with clear messages
- When even a small error is detected, or a compilation warning is showing up, immediately address their root cause and do not silence them
- **Fix all failing tests**: When you discover a test that fails, fix it immediately - even if the failure is not caused by your current work. Do not skip or defer pre-existing test failures.
- Do not create backup folders to maintain "old" code. Make modifications directly.

## Future Enhancements

See `docs/specs/14-future-development.md` for the full parking lot of planned features (OCR, multimodal embeddings, rule deduplication, etc.).
