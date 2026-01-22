# Protocol Migration Guide: IngestService to workspace_daemon

**Status:** Active Migration
**Target Removal:** Next Major Version

## Overview

The workspace-qdrant-mcp daemon is transitioning from the legacy `IngestService` protocol (`ingestion.proto`) to the new modular `workspace_daemon` protocol (`workspace_daemon.proto`).

All legacy methods now emit deprecation warnings and will be removed in the next major version.

## Protocol Comparison

### New Protocol Structure

The new protocol (`workspace_daemon.proto`) provides three focused services:

| Service | RPCs | Purpose |
|---------|------|---------|
| **SystemService** | 7 | Health, status, metrics, lifecycle |
| **CollectionService** | 5 | Collection and alias management |
| **DocumentService** | 3 | Text ingestion, update, deletion |

### Legacy Protocol

The legacy `IngestService` provided 27+ RPCs in a monolithic service:
- Document processing
- Folder watching
- Search operations
- Collection management
- Memory rule management
- Configuration management
- Status and monitoring

## Migration Mapping

### Document Operations

| Legacy Method | New Method | Notes |
|--------------|------------|-------|
| `process_document()` | `ingest_text()` | Use DocumentService.IngestText |
| `process_folder()` | SQLite watch config | Configure via SQLiteStateManager |
| `list_documents()` | `list_documents_new()` | Use DocumentService.ListDocuments |
| `get_document()` | Use new protocol | Use DocumentService.GetDocument |
| `delete_document()` | `delete_document_new()` | Use DocumentService.DeleteDocument |

### Collection Operations

| Legacy Method | New Method | Notes |
|--------------|------------|-------|
| `create_collection_legacy()` | `create_collection()` | Use CollectionService.CreateCollection |
| `delete_collection_legacy()` | `delete_collection()` | Use CollectionService.DeleteCollection |
| `list_collections()` | `list_collections_new()` | Use CollectionService.ListCollections |
| `get_collection_info()` | Use new protocol | Use CollectionService.GetCollectionInfo |
| `collection_exists()` | Use new protocol | Check via CollectionService.GetCollectionInfo |

### Watch Operations

| Legacy Method | New Approach | Notes |
|--------------|--------------|-------|
| `start_watching()` | `SQLiteStateManager.save_watch_folder_config()` | SQLite-driven watching |
| `stop_watching()` | `SQLiteStateManager.remove_watch_folder_config()` | Remove from SQLite |
| `list_watches()` | `SQLiteStateManager.list_watch_folders()` | Query SQLite |
| `configure_watch()` | `SQLiteStateManager.save_watch_folder_config()` | Update SQLite config |

### Search Operations

| Legacy Method | New Method | Notes |
|--------------|------------|-------|
| `execute_query()` | MCP `search()` tool | Use hybrid search |

### Memory Operations

| Legacy Method | New Approach | Notes |
|--------------|--------------|-------|
| `add_memory_rule()` | `ingest_text(collection="memory")` | Store in canonical 'memory' collection |
| `list_memory_rules()` | Search 'memory' collection | Query with metadata filters |
| `delete_memory_rule()` | `delete_document()` on 'memory' | Delete by document ID |
| `search_memory_rules()` | Search 'memory' collection | Semantic search |

### Status Operations

| Legacy Method | New Method | Notes |
|--------------|------------|-------|
| `get_stats()` | `get_status()` | Use SystemService.GetStatus |
| `get_processing_status()` | `get_status()` | Use SystemService.GetStatus |
| `get_system_status()` | `get_status()` | Use SystemService.GetStatus |
| `health_check_legacy()` | `health_check()` | Use SystemService.HealthCheck |

### Configuration Operations

| Legacy Method | New Approach | Notes |
|--------------|--------------|-------|
| `load_configuration()` | Local config files | Use configuration manager |
| `save_configuration()` | Local config files | Use configuration manager |
| `validate_configuration()` | Local validation | Use configuration manager |

## Code Migration Examples

### Before (Legacy)

```python
# Process a document
await client.process_document(
    file_path="/path/to/file.py",
    collection="_myproject",
    metadata={"type": "code"}
)

# Search
results = await client.execute_query(
    query="authentication",
    collections=["_myproject"],
    limit=10
)

# Add memory rule
await client.add_memory_rule(
    category="behavior",
    name="formatting",
    rule_text="Always use black for Python formatting"
)
```

### After (New Protocol)

```python
# Process a document - use ingest_text
await client.ingest_text(
    content=file_content,
    collection_basename="code",  # Goes to 'projects' collection
    tenant_id="myproject",
    metadata={"file_type": "python", "path": "/path/to/file.py"}
)

# Search - use MCP tool
# The MCP server's search() tool uses hybrid search

# Add memory rule - store in 'memory' collection
await client.ingest_text(
    content="Always use black for Python formatting",
    collection_basename="memory",  # Goes to 'memory' collection
    metadata={"category": "behavior", "name": "formatting"}
)
```

## Canonical Collections (ADR-001)

The new architecture uses only three canonical collections:

| Collection | Purpose | Tenant Isolation |
|------------|---------|------------------|
| `memory` | Behavioral rules | Optional project_id tag |
| `projects` | All project data | project_id + branch |
| `libraries` | Documentation | library_name |

**Deprecated patterns:**
- `_projects`, `_libraries`, `_memory` (underscore prefix)
- `_{project_id}` (per-project collections)
- `__memory`, `__system` (double underscore)

## Timeline

1. **Current:** All legacy methods emit deprecation warnings
2. **Next Minor:** Legacy methods may be marked for removal
3. **Next Major:** Legacy methods and `ingestion.proto` removed

## Migration Checklist

- [ ] Update all `process_document()` calls to `ingest_text()`
- [ ] Replace watch operations with SQLiteStateManager
- [ ] Update collection operations to new service methods
- [ ] Replace memory rule operations with 'memory' collection storage
- [ ] Update status checks to use `get_status()`
- [ ] Test all migrations with new canonical collections

## References

- [ADR-001: Multi-Tenant Architecture](decisions/ADR-001-multi-tenant-architecture.md)
- [workspace_daemon.proto](../src/rust/daemon/proto/workspace_daemon.proto)
- [DaemonClient API](../src/python/common/grpc/daemon_client.py)
