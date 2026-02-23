# workspace-qdrant-mcp Specification

**Version:** 1.9.0
**Date:** 2026-02-23
**Status:** Authoritative Specification
**Supersedes:** CONSOLIDATED_PRD_V2.md, PRDv3.txt, PRDv3-snapshot1.txt

---

This specification is organized into modular files under [`docs/specs/`](./docs/specs/). Each section is self-contained with cross-references between files.

## Table of Contents

| # | Section | File | Description |
|---|---------|------|-------------|
| 0 | [Overview and Vision](./docs/specs/00-overview.md) | `00-overview.md` | Purpose, design philosophy, core principles |
| 1 | [Architecture](./docs/specs/01-architecture.md) | `01-architecture.md` | Two-process architecture, component responsibilities, SQLite ownership |
| 2 | [Collection Architecture](./docs/specs/02-collection-architecture.md) | `02-collection-architecture.md` | Canonical collections, multi-tenancy, project ID generation, point identity, payload schemas |
| 3 | [Document Type Taxonomy](./docs/specs/03-document-taxonomy.md) | `03-document-taxonomy.md` | Document families, title extraction, chunking strategy |
| 4 | [Write Path Architecture](./docs/specs/04-write-path.md) | `04-write-path.md` | Write/read flows, unified queue, adaptive resources, daemon processing phases |
| 5 | [Memory System](./docs/specs/05-memory-system.md) | `05-memory-system.md` | Rule schema, scope, context injection, management |
| 6 | [File Watching and Ingestion](./docs/specs/06-file-watching.md) | `06-file-watching.md` | Two-layer watching, file type allowlist, git submodules, ingestion pipeline |
| 7 | [Code Intelligence](./docs/specs/07-code-intelligence.md) | `07-code-intelligence.md` | Tree-sitter baseline, LSP enhancement, semantic code chunking |
| 8 | [API Reference](./docs/specs/08-api-reference.md) | `08-api-reference.md` | MCP tools, session lifecycle, gRPC services |
| 9 | [Search Instrumentation](./docs/specs/09-search-instrumentation.md) | `09-search-instrumentation.md` | Search events, resolution events, analytics queries |
| 10 | [Keyword and Tag Extraction](./docs/specs/10-keyword-tag-extraction.md) | `10-keyword-tag-extraction.md` | Extraction pipeline, canonical tag hierarchy, query expansion |
| 11 | [Grammar and Runtime Management](./docs/specs/11-grammar-runtime.md) | `11-grammar-runtime.md` | Cache locations, grammar management, CI integration |
| 12 | [Configuration Reference](./docs/specs/12-configuration.md) | `12-configuration.md` | Configuration file, environment variables, SQLite database, disaster recovery |
| 13 | [Deployment and Installation](./docs/specs/13-deployment.md) | `13-deployment.md` | Platform support, installation methods, Docker, service management, CI/CD |
| 14 | [Future Development](./docs/specs/14-future-development.md) | `14-future-development.md` | Graph RAG, cross-project search, library management, related documents |

## Related Documents

| Document | Purpose |
|----------|---------|
| [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md) | Architectural philosophy |
| [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md) | Collection architecture decision |
| [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md) | Write policy decision |
| [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md) | SQLite ownership decision |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Visual architecture diagrams |
| [docs/LSP_INTEGRATION.md](./docs/LSP_INTEGRATION.md) | LSP integration guide |
| [README.md](./README.md) | User documentation |

---

## Changelog

- v1.9.0 (2026-02-23): Split monolithic spec into modular files under `docs/specs/`
- v1.8.0 (2026-02-18): Added grep-searcher fallback, FTS5 query optimization
- v1.7.0: Architecture decision — single daemon with embedded graph (replacing dual-daemon graphd)
- v1.6.0: Added keyword/tag extraction pipeline, canonical tag hierarchy, search query expansion
- v1.5.0: Major queue and daemon architecture update
- v1.4.0: Clarified 4 MCP tools, memory multi-tenancy, graceful degradation
- v1.3.0: Major API redesign (manage → memory/health/session tools)
- v1.2.0: Updated API Reference, pattern configuration docs
- v1.1.0: Added comprehensive Project ID specification
