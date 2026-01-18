# Multi-Tenant Collection Schema (v2.0)

**Version:** 2.0
**Date:** 2026-01-18
**Status:** Implementation Specification
**Supersedes:** multitenancy_architecture.md (v1.0)

---

## Table of Contents

1. [Overview](#overview)
2. [Schema Design](#schema-design)
3. [Projects Collection](#projects-collection)
4. [Libraries Collection](#libraries-collection)
5. [Memory Collection](#memory-collection)
6. [Indexing Strategy](#indexing-strategy)
7. [Protobuf Definitions](#protobuf-definitions)
8. [JSON Schema Validation](#json-schema-validation)
9. [Migration from v1.0](#migration-from-v10)
10. [Performance Considerations](#performance-considerations)

---

## Overview

### Architecture Change

This document specifies the **v2.0 multi-tenant collection schema**, which consolidates the previous per-project collection model into **3 unified collections**:

| Collection | Purpose | Multi-Tenant Key | Description |
|------------|---------|------------------|-------------|
| `projects` | All project content | `project_id` | Code, docs, tests, configs, notes, artifacts |
| `libraries` | Reference documentation | `library_name` | Books, papers, manuals, documentation |
| `memory` | Behavioral rules | N/A | LLM preferences, rules (unchanged) |

### Design Rationale

**Why consolidate?**

1. **Qdrant Performance**: Fewer, larger collections perform better than many small ones
2. **Simplified Management**: 3 collections vs. potentially hundreds
3. **Cross-Project Search**: Trivial (remove `project_id` filter) instead of complex
4. **Mental Model**: "Everything about project X" in one place

**Key Design Decisions:**

- **Drop `_` prefix**: Cleaner naming with intentional collection set
- **Unified content types**: Code and artifacts in single `projects` collection
- **Agent-managed projects**: MCP server handles registration, not CLI
- **User-managed libraries**: CLI for adding reference documentation folders
- **Priority-based ingestion**: Active agent sessions get HIGH priority

---

## Schema Design

### Vector Configuration

All collections use the same vector configuration:

```yaml
Vector Configuration:
  model: FastEmbed all-MiniLM-L6-v2
  dimensions: 384
  distance: Cosine

Sparse Vector Configuration:
  name: text
  modifier: idf

HNSW Parameters:
  m: 16
  ef_construct: 100
  on_disk: false  # Memory-mapped for performance
```

### Common Payload Fields

Fields present across all collections:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chunk_index` | integer | Yes | Position in chunked document (0-indexed) |
| `total_chunks` | integer | No | Total chunks for document |
| `created_at` | timestamp | Yes | ISO 8601 creation timestamp |
| `updated_at` | timestamp | No | ISO 8601 last update timestamp |
| `content_hash` | string | No | SHA256 of content for deduplication |

---

## Projects Collection

### Purpose

Single collection holding **all project content** (code, documentation, tests, configurations, notes, artifacts) with isolation via `project_id` metadata filter.

### Collection Name

```
Collection: projects
```

### Payload Schema

```json
{
  "project_id": {
    "type": "string",
    "description": "12-character hex hash identifier (required, indexed)",
    "pattern": "^[a-f0-9]{12}$",
    "example": "a1b2c3d4e5f6"
  },
  "project_name": {
    "type": "string",
    "description": "Human-readable project name",
    "example": "workspace-qdrant-mcp"
  },
  "file_path": {
    "type": "string",
    "description": "Relative path from project root",
    "example": "src/python/common/core/client.py"
  },
  "file_absolute_path": {
    "type": "string",
    "description": "Full absolute path (reference only, may become stale)",
    "example": "/Users/chris/dev/project/src/client.py"
  },
  "file_type": {
    "type": "string",
    "enum": ["code", "doc", "test", "config", "note", "artifact"],
    "description": "Content classification"
  },
  "language": {
    "type": "string",
    "description": "Programming language or file format",
    "example": "python"
  },
  "branch": {
    "type": "string",
    "description": "Git branch name",
    "default": "main",
    "example": "feature/auth"
  },
  "symbols": {
    "type": "array",
    "items": {"type": "string"},
    "description": "LSP-extracted symbols (functions, classes, variables)",
    "example": ["QdrantClient", "search", "ingest_document"]
  },
  "symbols_defined": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Symbols defined in this chunk"
  },
  "symbols_used": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Symbols referenced/imported in this chunk"
  },
  "title": {
    "type": "string",
    "description": "Document/file title or first line summary"
  },
  "source": {
    "type": "string",
    "enum": ["file", "user_input", "web", "chat", "generated"],
    "description": "Content origin",
    "default": "file"
  },
  "lsp_metadata": {
    "type": "object",
    "description": "Rich metadata from LSP server",
    "properties": {
      "definitions": {"type": "array"},
      "references": {"type": "array"},
      "hover_info": {"type": "string"}
    }
  },
  "chunk_index": {
    "type": "integer",
    "description": "Position in chunked document",
    "minimum": 0
  },
  "total_chunks": {
    "type": "integer",
    "description": "Total chunks for this document",
    "minimum": 1
  },
  "created_at": {
    "type": "string",
    "format": "date-time",
    "description": "ISO 8601 timestamp"
  }
}
```

### Required Fields

| Field | Indexed | Notes |
|-------|---------|-------|
| `project_id` | **Yes** | Primary filter key |
| `file_path` | No | For display/navigation |
| `file_type` | Yes | For type-based queries |
| `chunk_index` | No | For document reconstruction |
| `created_at` | Yes | For temporal queries |

### Query Patterns

**Project-scoped search (default):**
```python
search(
    collection="projects",
    query_vector=embed("authentication implementation"),
    filter={
        "must": [
            {"key": "project_id", "match": {"value": "a1b2c3d4e5f6"}}
        ]
    },
    limit=10
)
```

**Cross-project search (global scope):**
```python
search(
    collection="projects",
    query_vector=embed("authentication implementation"),
    # No project_id filter
    limit=10
)
```

**Symbol search:**
```python
search(
    collection="projects",
    query_vector=embed("QdrantClient class definition"),
    filter={
        "must": [
            {"key": "project_id", "match": {"value": "a1b2c3d4e5f6"}},
            {"key": "symbols_defined", "match": {"any": ["QdrantClient"]}}
        ]
    },
    limit=5
)
```

---

## Libraries Collection

### Purpose

Single collection holding **all reference documentation** (books, papers, manuals, articles) with isolation via `library_name` metadata filter.

### Collection Name

```
Collection: libraries
```

### Payload Schema

```json
{
  "library_name": {
    "type": "string",
    "description": "Library identifier (folder root name, required, indexed)",
    "pattern": "^[a-z0-9][a-z0-9_-]*$",
    "example": "color-science"
  },
  "source_file": {
    "type": "string",
    "description": "Original file path within library",
    "example": "/Volumes/Reference/color-science/gamma-correction.pdf"
  },
  "file_type": {
    "type": "string",
    "enum": ["pdf", "epub", "md", "txt", "html", "rst", "doc", "docx"],
    "description": "Document format"
  },
  "title": {
    "type": "string",
    "description": "Extracted or filename-derived title",
    "example": "Gamma Correction in Digital Imaging"
  },
  "author": {
    "type": "string",
    "description": "Document author(s) if extractable"
  },
  "topics": {
    "type": "array",
    "items": {"type": "string"},
    "description": "User-provided or extracted topics/tags",
    "example": ["color", "gamma", "imaging", "calibration"]
  },
  "folder": {
    "type": "string",
    "description": "Subfolder within library for optional filtering",
    "example": "fundamentals"
  },
  "library_version": {
    "type": "string",
    "description": "Version if library has versioning",
    "example": "2.0"
  },
  "page_number": {
    "type": "integer",
    "description": "Page number for PDFs",
    "minimum": 1
  },
  "chunk_index": {
    "type": "integer",
    "minimum": 0
  },
  "total_chunks": {
    "type": "integer",
    "minimum": 1
  },
  "created_at": {
    "type": "string",
    "format": "date-time"
  }
}
```

### Required Fields

| Field | Indexed | Notes |
|-------|---------|-------|
| `library_name` | **Yes** | Primary filter key |
| `source_file` | No | For reference |
| `file_type` | Yes | For format filtering |
| `chunk_index` | No | For document reconstruction |
| `created_at` | Yes | For temporal queries |

### Query Patterns

**Library-scoped search:**
```python
search(
    collection="libraries",
    query_vector=embed("gamma correction calibration"),
    filter={
        "must": [
            {"key": "library_name", "match": {"value": "color-science"}}
        ]
    },
    limit=10
)
```

**All libraries search:**
```python
search(
    collection="libraries",
    query_vector=embed("gamma correction"),
    # No library_name filter
    limit=10
)
```

**Topic-filtered search:**
```python
search(
    collection="libraries",
    query_vector=embed("color management"),
    filter={
        "must": [
            {"key": "topics", "match": {"any": ["color", "calibration"]}}
        ]
    },
    limit=10
)
```

---

## Memory Collection

### Purpose

Global collection for **LLM behavioral rules** and user preferences. Unchanged from v1.0.

### Collection Name

```
Collection: memory
```

### Payload Schema

```json
{
  "rule_id": {
    "type": "string",
    "description": "Unique rule identifier",
    "pattern": "^[a-z0-9][a-z0-9_-]*$"
  },
  "rule_type": {
    "type": "string",
    "enum": ["preference", "behavior", "constraint", "pattern"],
    "description": "Rule classification"
  },
  "content": {
    "type": "string",
    "description": "Rule text content"
  },
  "priority": {
    "type": "integer",
    "description": "Rule priority (1-10, higher = more important)",
    "minimum": 1,
    "maximum": 10,
    "default": 5
  },
  "scope": {
    "type": "string",
    "enum": ["global", "project", "language"],
    "description": "Rule applicability scope",
    "default": "global"
  },
  "project_id": {
    "type": "string",
    "description": "Optional project scope (if scope=project)"
  },
  "language": {
    "type": "string",
    "description": "Optional language scope (if scope=language)"
  },
  "enabled": {
    "type": "boolean",
    "default": true
  },
  "created_at": {
    "type": "string",
    "format": "date-time"
  },
  "updated_at": {
    "type": "string",
    "format": "date-time"
  }
}
```

### Query Patterns

**Fetch active rules:**
```python
search(
    collection="memory",
    query_vector=embed("coding style preferences"),
    filter={
        "must": [
            {"key": "enabled", "match": {"value": True}},
            {"key": "scope", "match": {"any": ["global", "project"]}}
        ]
    },
    limit=20
)
```

---

## Indexing Strategy

### Payload Index Configuration

```python
# Projects collection indexes
projects_indexes = {
    "project_id": {
        "type": "keyword",
        "is_tenant": True  # Qdrant optimization hint
    },
    "file_type": {
        "type": "keyword"
    },
    "language": {
        "type": "keyword"
    },
    "branch": {
        "type": "keyword"
    },
    "symbols_defined": {
        "type": "keyword",
        "is_array": True
    },
    "created_at": {
        "type": "datetime"
    }
}

# Libraries collection indexes
libraries_indexes = {
    "library_name": {
        "type": "keyword",
        "is_tenant": True
    },
    "file_type": {
        "type": "keyword"
    },
    "topics": {
        "type": "keyword",
        "is_array": True
    },
    "created_at": {
        "type": "datetime"
    }
}

# Memory collection indexes
memory_indexes = {
    "rule_type": {
        "type": "keyword"
    },
    "scope": {
        "type": "keyword"
    },
    "enabled": {
        "type": "bool"
    },
    "priority": {
        "type": "integer"
    }
}
```

### Index Creation Commands

```python
from qdrant_client import models

# Create payload index for tenant filtering
await client.create_payload_index(
    collection_name="projects",
    field_name="project_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    is_tenant=True  # Enables optimized tenant filtering
)
```

---

## Protobuf Definitions

### Collection Schema Messages

```protobuf
// Collection payload schemas for gRPC consistency

message ProjectPayload {
    string project_id = 1;           // 12-char hex (required)
    optional string project_name = 2;
    string file_path = 3;            // Relative path (required)
    optional string file_absolute_path = 4;
    FileType file_type = 5;
    optional string language = 6;
    string branch = 7;               // Default: "main"
    repeated string symbols = 8;
    repeated string symbols_defined = 9;
    repeated string symbols_used = 10;
    optional string title = 11;
    ContentSource source = 12;
    optional LspMetadata lsp_metadata = 13;
    int32 chunk_index = 14;
    optional int32 total_chunks = 15;
    google.protobuf.Timestamp created_at = 16;
}

message LibraryPayload {
    string library_name = 1;         // Required, indexed
    string source_file = 2;
    LibraryFileType file_type = 3;
    optional string title = 4;
    optional string author = 5;
    repeated string topics = 6;
    optional string folder = 7;
    optional string library_version = 8;
    optional int32 page_number = 9;
    int32 chunk_index = 10;
    optional int32 total_chunks = 11;
    google.protobuf.Timestamp created_at = 12;
}

message MemoryPayload {
    string rule_id = 1;
    RuleType rule_type = 2;
    string content = 3;
    int32 priority = 4;              // 1-10
    RuleScope scope = 5;
    optional string project_id = 6;
    optional string language = 7;
    bool enabled = 8;
    google.protobuf.Timestamp created_at = 9;
    optional google.protobuf.Timestamp updated_at = 10;
}

// Enums
enum FileType {
    FILE_TYPE_UNSPECIFIED = 0;
    FILE_TYPE_CODE = 1;
    FILE_TYPE_DOC = 2;
    FILE_TYPE_TEST = 3;
    FILE_TYPE_CONFIG = 4;
    FILE_TYPE_NOTE = 5;
    FILE_TYPE_ARTIFACT = 6;
}

enum LibraryFileType {
    LIBRARY_FILE_TYPE_UNSPECIFIED = 0;
    LIBRARY_FILE_TYPE_PDF = 1;
    LIBRARY_FILE_TYPE_EPUB = 2;
    LIBRARY_FILE_TYPE_MD = 3;
    LIBRARY_FILE_TYPE_TXT = 4;
    LIBRARY_FILE_TYPE_HTML = 5;
    LIBRARY_FILE_TYPE_RST = 6;
    LIBRARY_FILE_TYPE_DOC = 7;
    LIBRARY_FILE_TYPE_DOCX = 8;
}

enum ContentSource {
    CONTENT_SOURCE_UNSPECIFIED = 0;
    CONTENT_SOURCE_FILE = 1;
    CONTENT_SOURCE_USER_INPUT = 2;
    CONTENT_SOURCE_WEB = 3;
    CONTENT_SOURCE_CHAT = 4;
    CONTENT_SOURCE_GENERATED = 5;
}

enum RuleType {
    RULE_TYPE_UNSPECIFIED = 0;
    RULE_TYPE_PREFERENCE = 1;
    RULE_TYPE_BEHAVIOR = 2;
    RULE_TYPE_CONSTRAINT = 3;
    RULE_TYPE_PATTERN = 4;
}

enum RuleScope {
    RULE_SCOPE_UNSPECIFIED = 0;
    RULE_SCOPE_GLOBAL = 1;
    RULE_SCOPE_PROJECT = 2;
    RULE_SCOPE_LANGUAGE = 3;
}

message LspMetadata {
    repeated SymbolDefinition definitions = 1;
    repeated SymbolReference references = 2;
    optional string hover_info = 3;
}

message SymbolDefinition {
    string name = 1;
    string kind = 2;                 // function, class, variable, etc.
    int32 line = 3;
    int32 column = 4;
}

message SymbolReference {
    string name = 1;
    string file_path = 2;
    int32 line = 3;
}
```

---

## JSON Schema Validation

### Projects Collection Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "projects-payload-schema",
  "title": "Projects Collection Payload",
  "type": "object",
  "required": ["project_id", "file_path", "chunk_index", "created_at"],
  "properties": {
    "project_id": {
      "type": "string",
      "pattern": "^[a-f0-9]{12}$",
      "description": "12-character hex identifier"
    },
    "file_path": {
      "type": "string",
      "minLength": 1
    },
    "file_type": {
      "type": "string",
      "enum": ["code", "doc", "test", "config", "note", "artifact"]
    },
    "language": {
      "type": "string"
    },
    "branch": {
      "type": "string",
      "default": "main"
    },
    "symbols": {
      "type": "array",
      "items": {"type": "string"}
    },
    "chunk_index": {
      "type": "integer",
      "minimum": 0
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

### Libraries Collection Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "libraries-payload-schema",
  "title": "Libraries Collection Payload",
  "type": "object",
  "required": ["library_name", "source_file", "chunk_index", "created_at"],
  "properties": {
    "library_name": {
      "type": "string",
      "pattern": "^[a-z0-9][a-z0-9_-]*$"
    },
    "source_file": {
      "type": "string",
      "minLength": 1
    },
    "file_type": {
      "type": "string",
      "enum": ["pdf", "epub", "md", "txt", "html", "rst", "doc", "docx"]
    },
    "title": {
      "type": "string"
    },
    "topics": {
      "type": "array",
      "items": {"type": "string"}
    },
    "chunk_index": {
      "type": "integer",
      "minimum": 0
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

---

## Migration from v1.0

### Collection Mapping

| v1.0 Collection | v2.0 Collection | Mapping |
|-----------------|-----------------|---------|
| `_{project_id}` | `projects` | Add `project_id` to payload |
| `_{library_name}` | `libraries` | Add `library_name` to payload |
| `{basename}-{type}` | `projects` | Add `project_id`, set `source=user_input` |
| `memory` | `memory` | Unchanged |

### Migration Script

```python
async def migrate_to_v2(
    qdrant_client,
    old_collection: str,
    new_collection: str,
    tenant_field: str,
    tenant_value: str
):
    """
    Migrate documents from per-project collection to unified collection.

    Args:
        old_collection: e.g., "_a1b2c3d4e5f6"
        new_collection: "projects" or "libraries"
        tenant_field: "project_id" or "library_name"
        tenant_value: The tenant identifier
    """
    offset = None
    batch_size = 100

    while True:
        # Scroll through old collection
        records, offset = await qdrant_client.scroll(
            collection_name=old_collection,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )

        if not records:
            break

        # Transform and upsert to new collection
        points = []
        for record in records:
            payload = record.payload.copy()
            payload[tenant_field] = tenant_value

            points.append(models.PointStruct(
                id=record.id,
                vector=record.vector,
                payload=payload
            ))

        await qdrant_client.upsert(
            collection_name=new_collection,
            points=points
        )

        if offset is None:
            break

    logger.info(f"Migrated {old_collection} to {new_collection}")
```

---

## Performance Considerations

### Projected Scale

| Metric | v1.0 (per-project) | v2.0 (unified) |
|--------|-------------------|----------------|
| Collections | ~150 | 3 |
| Documents per collection | 1K-50K | 100K-10M |
| Query scope (filtered) | 1K-50K | 1K-50K |
| Memory overhead | ~13GB | ~5GB |

### Performance Benefits

1. **Reduced Collection Overhead**: 3 collections vs. potentially hundreds
2. **Efficient Tenant Filtering**: Qdrant `is_tenant=True` optimization
3. **Better HNSW Indexing**: Larger collections build more efficient indexes
4. **Simpler Query Logic**: No dynamic collection name resolution

### Benchmarks to Validate

1. **Query latency**: Compare v1.0 vs v2.0 with same data volume
2. **Ingestion throughput**: Documents/second with concurrent projects
3. **Memory usage**: Per-collection vs. unified with payload indexing
4. **Cross-tenant query**: Performance of global search without filters

### Success Criteria

- Query latency < 100ms for p95 (with tenant filter)
- Cross-project query < 200ms for p95
- Ingestion throughput > 100 docs/sec
- Memory usage < 10GB for 1M documents

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | 2026-01-18 | Architecture Design | New unified 3-collection schema |
| 1.0 | 2025-01-02 | Design Session | Original per-project collection model |

---

**End of Multi-Tenant Collection Schema Specification v2.0**
