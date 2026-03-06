# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

workspace-qdrant-mcp (v0.1.0) is a Model Context Protocol (MCP) server providing project-scoped Qdrant vector database operations with hybrid search capabilities.

**Core Architecture:**
- **MCP Server**: TypeScript (in development)
- **Daemon**: Rust (high-performance file watching and processing)
- **CLI**: Rust (high-performance command-line interface)

**Core Features:**
- 6 MCP tools: store, search, rules, retrieve, grep, list
- Hybrid search combining dense (semantic) and sparse (keyword) vectors
- Automatic project detection with Git integration
- Behavioral rules via persistent rules collection
- Rust daemon for high-performance file watching and processing

**v0.1.0 Features:**
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
2. **docs/specs/** - Modular specifications (15 spec files)
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

```bash
# Rust daemon (release build with ONNX Runtime)
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib cargo build --release --manifest-path src/rust/Cargo.toml --package memexd

# Rust CLI
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib cargo build --release --manifest-path src/rust/Cargo.toml --package wqm-cli

# Rust tests
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib cargo test --manifest-path src/rust/Cargo.toml --package workspace-qdrant-core

# Test specific features: cargo test --package workspace-qdrant-core -- <filter>
# Filters: git_integration, watching, tree_sitter, keyword, tag, graph, schema_version, etc.

# TypeScript MCP Server
cd src/typescript/mcp-server && npm install && npm run build && npm test

# Deploy (after successful build)
cp src/rust/target/release/memexd ~/.local/bin/memexd
cp src/rust/target/release/wqm ~/.local/bin/wqm
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
```

### CLI Quick Reference

Use `wqm --help` and `wqm <subcommand> --help` for full usage. Key commands:

```bash
wqm service status              # Check daemon status
wqm admin health                # Health diagnostics
wqm queue list / stats          # Queue monitoring
wqm watch pause / resume        # File watcher control
wqm graph stats --tenant <t>    # Graph subsystem
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

**Optional:**
- `FASTEMBED_MODEL` - Embedding model (default: all-MiniLM-L6-v2)
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
