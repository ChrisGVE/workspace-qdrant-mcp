# Multi-Tenant Architecture Migration Guide

This guide helps existing MCP clients migrate to the new multi-tenant architecture in workspace-qdrant-mcp. The changes enable project-scoped workspace collections while maintaining backward compatibility where possible.

## Overview of Changes

The workspace-qdrant-mcp server now supports multi-tenant workspace collections that provide project-level isolation and organization. This update introduces 6 new MCP tools and enhances existing functionality.

## New Multi-Tenant Tools

### Collection Management Tools

1. **`create_workspace_collection`** - Create project-specific workspace collections
2. **`initialize_project_workspace_collections`** - Set up complete workspace for a project

### Search and Retrieval Tools

3. **`search_workspace_by_project`** - Search with automatic project context filtering
4. **`search_workspace_metadata_by_project`** - Metadata-based search with project filtering
5. **`list_workspace_collections_by_project`** - List collections for specific project

### Document Management Tools

6. **`add_document_with_project_context`** - Add documents with automatic project metadata injection
7. **`get_workspace_collection_info`** - Get comprehensive workspace information

## Migration Examples

### Example 1: Basic Document Search Migration

**Before (Legacy API):**
```python
# Old approach - search all collections
result = await search_workspace(
    query="authentication implementation",
    collections=["my-project"],
    mode="hybrid",
    limit=10
)
```

**After (Multi-Tenant API):**
```python
# New approach - project-aware search with auto-detection
result = await search_workspace_by_project(
    query="authentication implementation",
    workspace_types=["notes", "docs"],  # Search specific workspace types
    mode="hybrid",
    limit=10,
    include_shared=True  # Include shared collections if needed
)

# Or with explicit project context
result = await search_workspace_by_project(
    query="authentication implementation",
    project_name="backend-service",
    workspace_types=["notes", "docs"],
    mode="hybrid",
    limit=10
)
```

### Example 2: Collection Creation Migration

**Before (Legacy API):**
```python
# Old approach - manual collection creation
# (Required direct Qdrant client interaction)
```

**After (Multi-Tenant API):**
```python
# New approach - structured workspace collection creation
result = await create_workspace_collection(
    project_name="my-project",
    collection_type="notes",
    enable_metadata_indexing=True
)

# Initialize complete workspace
result = await initialize_project_workspace_collections(
    project_name="my-new-project",
    workspace_types=["notes", "docs", "scratchbook"],
    subprojects=["frontend", "backend"]
)
```

### Example 3: Document Addition Migration

**Before (Legacy API):**
```python
# Old approach - basic document addition
result = await add_document(
    content="Implementation notes for OAuth flow",
    collection="my-project",
    metadata={"category": "security"}
)
```

**After (Multi-Tenant API):**
```python
# New approach - automatic project metadata enrichment
result = await add_document_with_project_context(
    content="Implementation notes for OAuth flow",
    collection="my-project-notes",
    workspace_type="notes",
    metadata={"category": "security", "priority": 4},
    creator="dev-team"
)
```

## Backward Compatibility

### Fully Compatible Tools

These existing tools continue to work without changes:

- `search_workspace` - Still available for multi-collection search
- `add_document` - Still works for basic document addition
- `get_document` - Unchanged document retrieval
- `list_collections` - Still lists all collections
- `workspace_status` - Provides system status
- `get_server_info` - Server capabilities info

### Enhanced Tools

These tools have been enhanced but remain backward compatible:

- `search_workspace` - Now supports optional `enable_project_isolation` parameter
- `search_memory_collections` - Can use multi-tenant search when `enable_project_isolation=true`

### Breaking Changes

**None** - All existing APIs maintain backward compatibility.

## Migration Strategies

### Strategy 1: Gradual Migration

1. **Phase 1**: Start using new multi-tenant tools alongside existing ones
2. **Phase 2**: Migrate search operations to project-aware tools
3. **Phase 3**: Migrate document management to project-context tools
4. **Phase 4**: Optimize collection structure using workspace types

### Strategy 2: New Project Setup

For new projects, use the multi-tenant tools from the start:

```python
# 1. Initialize project workspace
await initialize_project_workspace_collections(
    project_name="new-project",
    workspace_types=["notes", "docs", "knowledge"]
)

# 2. Add documents with project context
await add_document_with_project_context(
    content=documentation,
    collection="new-project-docs",
    workspace_type="docs",
    metadata={"version": "1.0", "author": "team"}
)

# 3. Search within project context
results = await search_workspace_by_project(
    query="API endpoints",
    project_name="new-project"
)
```

## Configuration Changes

### Environment Variables

No new environment variables are required. Existing configuration works:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key  # Optional for local Qdrant
FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Collection Naming Conventions

The multi-tenant architecture introduces structured naming:

- **Project Collections**: `{project-name}-{workspace-type}` (e.g., `backend-api-notes`)
- **Shared Collections**: `{workspace-type}` (e.g., `scratchbook`)
- **Legacy Collections**: Existing collections remain unchanged

## Testing Migration

### Test Script Example

```python
async def test_migration():
    """Test both legacy and new multi-tenant APIs."""

    # Test 1: Legacy search still works
    legacy_result = await search_workspace(
        query="test query",
        collections=["existing-collection"]
    )

    # Test 2: New project-aware search
    new_result = await search_workspace_by_project(
        query="test query",
        project_name="test-project"
    )

    # Test 3: Document addition with project context
    doc_result = await add_document_with_project_context(
        content="Test document",
        collection="test-project-notes",
        workspace_type="notes"
    )

    # Verify all operations succeeded
    assert legacy_result.get("success") is not False
    assert new_result.get("success") is not False
    assert doc_result.get("success") is True

    print("Migration compatibility verified!")
```

## Performance Considerations

### Multi-Tenant Benefits

- **Better Isolation**: Project-specific collections reduce search scope
- **Improved Performance**: Smaller collections = faster queries
- **Enhanced Metadata**: Richer context for better search results
- **Structured Organization**: Clear workspace types and project boundaries

### Migration Tips

1. **Gradual Migration**: Don't migrate everything at once
2. **Test Thoroughly**: Verify functionality before full migration
3. **Monitor Performance**: Multi-tenant collections should improve performance
4. **Use Workspace Types**: Leverage structured workspace organization

## Support and Troubleshooting

### Common Issues

**Q: Can I use both legacy and new APIs together?**
A: Yes, they're fully compatible and can be used simultaneously.

**Q: Do I need to migrate existing collections?**
A: No, existing collections continue to work. New multi-tenant features enhance but don't replace existing functionality.

**Q: How do I know which project context is being used?**
A: Multi-tenant tools auto-detect project context from the workspace client's project information. You can also explicitly specify `project_name`.

**Q: What if I don't want project isolation?**
A: Continue using existing tools. The new tools are additive, not mandatory.

### Error Handling

Multi-tenant tools provide clear error messages:

```python
{
    "error": "Invalid workspace type 'invalid'. Valid types: ['notes', 'docs', 'scratchbook', 'knowledge', 'context', 'memory']"
}
```

### Logging

Enhanced logging helps with migration:

```
INFO - Multi-tenant tools registered: create_workspace_collection, search_workspace_by_project, ...
INFO - Project-aware search completed: project_name=backend-api, results_count=15
INFO - Workspace collection creation succeeded: my-project-notes
```

## Next Steps

1. **Read the Documentation**: Review updated tool docstrings for detailed parameter information
2. **Start Small**: Try new tools with test data first
3. **Plan Migration**: Choose gradual or new-project migration strategy
4. **Monitor**: Use logging to verify multi-tenant features work as expected
5. **Provide Feedback**: Report any issues or suggestions for improvement

For questions or issues, refer to the main project documentation or create an issue in the project repository.