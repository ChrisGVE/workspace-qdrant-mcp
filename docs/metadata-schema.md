# Multi-Tenant Metadata Schema

This document provides comprehensive guidance on the multi-tenant metadata schema system for workspace-qdrant-mcp. The schema enables project isolation through metadata-based filtering while maintaining backward compatibility with existing collection naming conventions.

## Overview

The multi-tenant metadata schema provides:

- **Project Isolation**: Efficient filtering via `project_id` metadata field
- **Collection Classification**: Systematic categorization of memory, library, project, and global collections
- **Reserved Naming Validation**: Support for `__` (system) and `_` (library) prefix collections
- **Backward Compatibility**: Seamless migration from suffix-based collection naming
- **Performance Optimization**: Indexed metadata fields for efficient queries

## Core Architecture

### Collection Categories

| Category | Prefix | MCP Access | CLI Access | Global Search | Example |
|----------|---------|------------|------------|---------------|---------|
| **System** | `__` | Read-only | Read-Write | No | `__user_preferences` |
| **Library** | `_` | Read-only | Read-Write | Yes | `_code_references` |
| **Project** | None | Read-Write | Read-Write | Conditional | `my-project-docs` |
| **Global** | None | Read-Write | Read-Write | Yes | `algorithms` |

### Metadata Fields

#### Core Tenant Isolation (Required)
- `project_id`: 12-character hash for efficient filtering
- `project_name`: Human-readable project name
- `tenant_namespace`: Format `{project_name}.{collection_type}`

#### Collection Classification (Required)
- `collection_type`: Functional type (docs, notes, memory, etc.)
- `collection_category`: System classification (system/library/project/global)
- `workspace_scope`: Accessibility scope (project/shared/global/library)

#### Access Control
- `access_level`: Permission level (public/private/shared/readonly)
- `mcp_readonly`: Boolean flag for MCP write restrictions
- `cli_writable`: Boolean flag for CLI write access
- `created_by`: Creator/origin tracking

#### Migration Support
- `migration_source`: How collection was created/migrated
- `legacy_collection_name`: Original name for reference
- `compatibility_version`: Schema version for future updates

## Usage Examples

### Creating Project Collection

```python
from common.core.metadata_schema import MultiTenantMetadataSchema, AccessLevel

# Create metadata for project collection
metadata = MultiTenantMetadataSchema.create_for_project(
    project_name="workspace_qdrant_mcp",
    collection_type="docs",
    created_by="user",
    access_level=AccessLevel.PRIVATE,
    tags=["documentation", "user-guide"],
    priority=4
)

# Convert to Qdrant payload
payload = metadata.to_qdrant_payload()
```

### Creating System Collection

```python
# Create metadata for system collection (CLI-writable, LLM-readable)
system_metadata = MultiTenantMetadataSchema.create_for_system(
    collection_name="__user_preferences",
    collection_type="memory_collection",
    created_by="system"
)

# System collections automatically configured as:
# - mcp_readonly=False (CLI can write)
# - access_level=AccessLevel.PRIVATE
# - workspace_scope=WorkspaceScope.GLOBAL
# - not globally searchable
```

### Creating Library Collection

```python
# Create metadata for library collection (MCP read-only)
library_metadata = MultiTenantMetadataSchema.create_for_library(
    collection_name="_code_references",
    collection_type="code_collection",
    created_by="cli"
)

# Library collections automatically configured as:
# - mcp_readonly=True (MCP cannot write)
# - access_level=AccessLevel.SHARED
# - workspace_scope=WorkspaceScope.LIBRARY
# - globally searchable
```

### Creating Global Collection

```python
# Create metadata for global collection
global_metadata = MultiTenantMetadataSchema.create_for_global(
    collection_name="algorithms",
    collection_type="global",
    created_by="system"
)

# Global collections automatically configured as:
# - access_level=AccessLevel.PUBLIC
# - workspace_scope=WorkspaceScope.GLOBAL
# - globally searchable
```

## Validation and Constraints

### Field Validation Rules

| Field | Type | Constraints | Example |
|-------|------|-------------|---------|
| `project_id` | string | Exactly 12 hex chars | `a1b2c3d4e5f6` |
| `project_name` | string | Max 128 chars, alphanumeric + underscore/hyphen | `workspace_qdrant_mcp` |
| `tenant_namespace` | string | Format `{project}.{type}`, max 192 chars | `myproject.docs` |
| `collection_type` | string | Max 64 chars, letters/numbers/underscore | `docs`, `memory_collection` |
| `priority` | integer | Range 1-5 (1=lowest, 5=highest) | `3` |

### Business Rules

1. **System Collections**: Must start with `__`, set `mcp_readonly=false`, `cli_writable=true`
2. **Library Collections**: Must start with `_` (not `__`), set `mcp_readonly=true`
3. **Project Collections**: Use `{project}-{suffix}` pattern or metadata-based naming
4. **Global Collections**: Must be in predefined list or properly documented

### Consistency Rules

- `tenant_namespace` must equal `{project_name}.{collection_type}`
- `is_reserved_name` must be `true` for system and library collections
- Library collections must have `mcp_readonly=true`
- System collections should have `access_level=private`

## Migration Guide

### Existing Collection Migration

The system supports non-destructive migration of existing collections:

```python
from common.core.backward_compatibility import BackwardCompatibilityManager

# Initialize migration manager
manager = BackwardCompatibilityManager(qdrant_client, config)

# Analyze existing collections
analysis = await manager.analyze_existing_collections()

# Migrate collections (adds metadata without changing names)
results = await manager.migrate_collections(analysis)

# Validate migration
validation = await manager.validate_migration(results)
```

### Migration Strategies

1. **Additive Migration**: Add metadata to existing collections without renaming
2. **Gradual Migration**: Support both old and new filtering methods during transition
3. **Rollback Support**: Ability to remove metadata if needed
4. **Validation**: Comprehensive testing of migrated collections

### Collection Name Mapping

| Original Pattern | New Metadata | Notes |
|------------------|--------------|-------|
| `project-docs` | `project_id=hash(project)`, `collection_type=docs` | Project collection |
| `__user_prefs` | `collection_category=system`, `mcp_readonly=false` | System collection |
| `_library_docs` | `collection_category=library`, `mcp_readonly=true` | Library collection |
| `algorithms` | `collection_category=global`, `access_level=public` | Global collection |

## Performance Optimization

### Indexed Fields

The following metadata fields should be indexed for optimal query performance:

```python
indexed_fields = [
    "project_id",           # Primary filtering field
    "tenant_namespace",     # Hierarchical filtering
    "collection_type",      # Type-based filtering
    "collection_category",  # Category-based filtering
    "workspace_scope",      # Scope-based filtering
    "access_level",         # Permission filtering
    "mcp_readonly",         # Access control filtering
    "created_by"            # Origin filtering
]
```

### Query Patterns

```python
# Single tenant filtering (most efficient)
filter_conditions = models.Filter(
    must=[
        models.FieldCondition(
            key="project_id",
            match=models.MatchValue(value="a1b2c3d4e5f6")
        )
    ]
)

# Cross-tenant shared resources
filter_conditions = models.Filter(
    must=[
        models.FieldCondition(
            key="workspace_scope",
            match=models.MatchValue(value="shared")
        ),
        models.FieldCondition(
            key="access_level",
            match=models.MatchAny(any=["public", "shared"])
        )
    ]
)
```

## Integration with Existing Systems

### Collection Manager Integration

```python
from common.core.collections import WorkspaceCollectionManager
from common.core.metadata_schema import MultiTenantMetadataSchema

class EnhancedCollectionManager(WorkspaceCollectionManager):
    async def create_collection_with_metadata(
        self,
        collection_name: str,
        metadata: MultiTenantMetadataSchema
    ):
        # Create collection with standard configuration
        await self._ensure_collection_exists(collection_config)

        # Add metadata indexing
        await self._optimize_metadata_indexing([collection_config])
```

### Hybrid Search Integration

```python
from common.core.hybrid_search import HybridSearchEngine

# Search with project context
search_results = await search_engine.hybrid_search(
    collection_name="docs",
    query_embeddings=embeddings,
    project_context={
        "project_id": "a1b2c3d4e5f6",
        "tenant_namespace": "myproject.docs",
        "workspace_scope": "project"
    },
    auto_inject_metadata=True
)
```

## Best Practices

### Schema Design

1. **Use Factory Methods**: Always use `create_for_*()` methods for consistency
2. **Validate Early**: Use `MetadataValidator` before storing metadata
3. **Index Strategically**: Index frequently filtered fields
4. **Document Collections**: Use descriptive `category` and `tags` fields

### Performance

1. **Primary Filtering**: Always filter by `project_id` when possible
2. **Minimize Cross-Tenant**: Limit cross-tenant searches for performance
3. **Batch Operations**: Use batch operations for metadata updates
4. **Monitor Queries**: Track query performance and optimize as needed

### Security

1. **Access Control**: Properly set `access_level` and `mcp_readonly` flags
2. **Validate Input**: Always validate metadata before storage
3. **Audit Trail**: Use `created_by` and temporal fields for auditing
4. **Isolation**: Ensure proper tenant isolation in queries

## Troubleshooting

### Common Issues

1. **Invalid Project ID**: Ensure 12-character hexadecimal format
2. **Inconsistent Namespace**: Verify `tenant_namespace` matches `project_name.collection_type`
3. **Access Denied**: Check `mcp_readonly` and `access_level` settings
4. **Migration Failures**: Review validation errors and fix schema issues

### Debugging Tools

```python
from common.core.metadata_validator import MetadataValidator

# Validate metadata
validator = MetadataValidator()
result = validator.validate_metadata(metadata)

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.message}")

# Get suggestions for fixes
suggestions = validator.suggest_fixes(metadata)
for field, suggestion in suggestions.items():
    print(f"{field}: {suggestion}")
```

## API Reference

### Core Classes

- `MultiTenantMetadataSchema`: Main schema class with factory methods
- `MetadataValidator`: Validation and constraint checking
- `BackwardCompatibilityManager`: Migration support
- `SchemaDocumentation`: Complete field specifications

### Enums

- `CollectionCategory`: SYSTEM, LIBRARY, PROJECT, GLOBAL
- `WorkspaceScope`: PROJECT, SHARED, GLOBAL, LIBRARY
- `AccessLevel`: PUBLIC, PRIVATE, SHARED, READONLY

### Factory Methods

- `create_for_project()`: Project-scoped collections
- `create_for_system()`: System collections with `__` prefix
- `create_for_library()`: Library collections with `_` prefix
- `create_for_global()`: Global system-wide collections

## Future Enhancements

### Planned Features

1. **Dynamic Schema Evolution**: Support for schema versioning and migration
2. **Advanced Access Control**: Role-based access control (RBAC)
3. **Audit Logging**: Comprehensive audit trail for metadata changes
4. **Performance Analytics**: Built-in performance monitoring and optimization

### Migration Roadmap

1. **Phase 1**: Additive metadata support (current)
2. **Phase 2**: Gradual migration of existing collections
3. **Phase 3**: Full metadata-based filtering
4. **Phase 4**: Legacy naming pattern deprecation

This metadata schema provides a robust foundation for multi-tenant project isolation while maintaining backward compatibility and performance optimization.