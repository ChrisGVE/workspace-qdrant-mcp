# Documentation Index

## Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start](quick-start.md) | Get running in 5 minutes |
| [User Manual](user-manual.md) | Complete usage guide (installation through troubleshooting) |

## Reference

| Document | Description |
|----------|-------------|
| [Installation](reference/installation.md) | Detailed installation for all platforms |
| [Windows Installation](reference/windows-installation.md) | Windows-specific setup guide |
| [Configuration](reference/configuration.md) | All config options, env vars, and defaults |
| [CLI Reference](reference/cli.md) | Every `wqm` command with flags and examples |
| [MCP Tools](reference/mcp-tools.md) | Tool parameters, schemas, and example calls |
| [LLM Integration](reference/mcp-best-practices.md) | Best practices for Claude and other AI assistants |
| [Architecture](reference/architecture.md) | Component overview and data flow |
| [Admin UI](ADMIN_UI.md) | Browser dashboard at `/admin/` — project discovery, real-time index info, CRUD over watch folders |
| [Admin · Behavioral Rules Handoff](design/admin-behavioral-rules-handoff.md) | Developer handoff spec for the `/admin/` rules-management section — tokens, states, REST contract, a11y |

## Guides & Operations

| Document | Description |
|----------|-------------|
| [LSP Integration](LSP_INTEGRATION.md) | Per-project LSP setup and code-intelligence guide |
| [Claude Code Hooks](CLAUDE_CODE_HOOKS.md) | CLI-based session-lifecycle hooks |
| [Metrics](METRICS.md) | Prometheus metrics catalog and alerting rules |
| [Troubleshooting](TROUBLESHOOTING.md) | Diagnostics and common issues |
| [Backup & Restore](BACKUP_RESTORE.md) | Snapshot and restore procedures |
| [OS Compatibility](OS_COMPATIBILITY.md) | Per-platform support matrix |
| [Qdrant Cheatsheet](qdrant_cheatsheet.md) | Common Qdrant point operations |
| [Deployment](deployment/docker.md) | Docker / [API keys](deployment/api-keys.md) / [reliability](deployment/reliability.md) reference-compose ops |
| [Collection Naming Migration](migration/collection-naming-migration.md) | `_projects` → `projects` data migration (ADR-001) |
| [Path Abstraction Upgrade Notes](upgrade-notes/0.1.x-path-abstraction.md) | schema v37 migration notes |

## Runbooks

Operational playbooks for recovering from specific incident classes.

| Document | Description |
|----------|-------------|
| [Qdrant Corruption Recovery](runbooks/qdrant-corruption.md) | Unloadable collections after a dirty shutdown — auto-quarantine wrapper, manual fallback, drift cleanup |
| [Self-Watch Loop Recovery](runbooks/self-watch-loop.md) | Docker / `memexd_db` recovery for the self-watch loop |

## Specifications

Technical specifications for developers and contributors.

| Document | Description |
|----------|-------------|
| [Overview](specs/00-overview.md) | Specification index |
| [Architecture](specs/01-architecture.md) | System design and component diagram |
| [Collection Architecture](specs/02-collection-architecture.md) | Multi-tenant collection model |
| [Document Taxonomy](specs/03-document-taxonomy.md) | Document types and processing |
| [Write Path](specs/04-write-path.md) | Queue architecture and write flow |
| [Rules System](specs/05-rules-system.md) | Behavioral rules design |
| [File Watching](specs/06-file-watching.md) | File watcher and debouncing |
| [Code Intelligence](specs/07-code-intelligence.md) | Tree-sitter, LSP, and code graph |
| [API Reference](specs/08-api-reference.md) | MCP tools and gRPC services |
| [Search Instrumentation](specs/09-search-instrumentation.md) | Analytics and metrics |
| [Keyword/Tag Extraction](specs/10-keyword-tag-extraction.md) | Extraction pipeline |
| [Grammar Runtime](specs/11-grammar-runtime.md) | ONNX Runtime and tree-sitter |
| [Configuration](specs/12-configuration.md) | Configuration specification |
| [Deployment](specs/13-deployment.md) | Build, packaging, and distribution |
| [Future Development](specs/14-future-development.md) | Roadmap and parking lot |
| [Language Registry](specs/15-language-registry.md) | Dynamic language registry and GenericExtractor |
| [Path Abstraction](specs/16-path-abstraction.md) | Root-relative path model |
| [Deferred Reprocessing](specs/17-deferred-reprocessing.md) | Deferred reprocessing pipeline |
| [Error Handling](specs/18-error-handling-resilience.md) | Error handling and resilience patterns |
| [Branch/Worktree Audit](specs/19-branch-worktree-audit.md) | Branch lifecycle and worktree management |
| [Token Economy Instrumentation](specs/20-token-economy-instrumentation.md) | Token-economy metrics |
| [Tree-sitter Roadmap](specs/21-tree-sitter-roadmap.md) | Tree-sitter chunking roadmap |
| [Cross-branch Dedup](specs/21-cross-branch-dedup.md) | Cross-branch chunk deduplication |

## Architecture Decision Records

| Document | Description |
|----------|-------------|
| [ADR-001](adr/ADR-001-canonical-collection-architecture.md) | Canonical collection names |
| [ADR-002](adr/ADR-002-daemon-only-write-policy.md) | Daemon-only Qdrant writes |
| [ADR-003](adr/ADR-003-daemon-owns-sqlite.md) | Daemon owns SQLite |

## Plans

| Document | Description |
|----------|-------------|
| [Search Quality Next Steps](plans/2026-05-25-search-quality-next-steps.md) | Configurable roadmap for embedding profiles, rerank, diversity, GPU, and benchmarking |
| [CLI Redesign](plans/2026-02-04-cli-redesign.md) | Completed CLI command restructure (historical record) |

## Reading Paths

**New user:** [Quick Start](quick-start.md) → [User Manual](user-manual.md) → [LLM Integration](reference/mcp-best-practices.md)

**Advanced user:** [CLI Reference](reference/cli.md) + [MCP Tools](reference/mcp-tools.md) + [Configuration](reference/configuration.md)

**Developer/Contributor:** [Architecture](reference/architecture.md) → [Specifications](specs/00-overview.md) → [ADRs](adr/)
