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

## Architecture Decision Records

| Document | Description |
|----------|-------------|
| [ADR-001](adr/ADR-001-canonical-collection-architecture.md) | Canonical collection names |
| [ADR-002](adr/ADR-002-daemon-only-write-policy.md) | Daemon-only Qdrant writes |
| [ADR-003](adr/ADR-003-daemon-owns-sqlite.md) | Daemon owns SQLite |

## Reading Paths

**New user:** [Quick Start](quick-start.md) → [User Manual](user-manual.md) → [LLM Integration](reference/mcp-best-practices.md)

**Advanced user:** [CLI Reference](reference/cli.md) + [MCP Tools](reference/mcp-tools.md) + [Configuration](reference/configuration.md)

**Developer/Contributor:** [Architecture](reference/architecture.md) → [Specifications](specs/00-overview.md) → [ADRs](adr/)
