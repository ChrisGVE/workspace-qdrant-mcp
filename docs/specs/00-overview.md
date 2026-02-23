## Overview and Vision

### Purpose

workspace-qdrant-mcp is a Model Context Protocol (MCP) server providing project-scoped Qdrant vector database operations with hybrid search capabilities. It enables LLM agents to:

- Store and retrieve project-specific knowledge
- Search across code, documentation, and notes using semantic similarity
- Maintain behavioral rules and preferences through persistent memory
- Index reference documentation libraries for cross-project search

### Design Philosophy

The system optimizes for:

1. **Conversational Memory**: Natural rule updates over configuration management
2. **Project Context**: Automatic workspace awareness over explicit collection selection
3. **Semantic Discovery**: Cross-content-type search over format-specific queries
4. **Behavioral Persistence**: Consistent LLM behavior over session configuration
5. **Intelligent Processing**: LSP-enhanced code understanding over text-only search

### Core Principles

See [FIRST-PRINCIPLES.md](../../FIRST-PRINCIPLES.md) for the complete architectural philosophy. Key principles:

- **Test Driven Development**: Unit tests written immediately after code
- **Memory-Driven Behavioral Persistence**: Rules stored in memory collection
- **Project-Scoped Semantic Context**: Automatic project detection and filtering
- **Daemon-Only Writes**: Single writer to Qdrant for consistency (see [ADR-002](../adr/ADR-002-daemon-only-write-policy.md))
- **Four Collections Only**: Exactly `projects`, `libraries`, `memory`, `scratchpad` (see [ADR-001](../adr/ADR-001-canonical-collection-architecture.md))

---
