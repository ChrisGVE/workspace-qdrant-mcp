# Tool Set Migration Guide

**Date**: 2025-09-08  
**Version**: Tool Simplification v1.0  
**Audience**: Existing workspace-qdrant-mcp users  

## Overview

workspace-qdrant-mcp has been updated with a simplified tool interface to improve usability and compatibility with reference MCP implementations. This guide helps you migrate from the 30+ tool interface to the streamlined 2-5 tool interface.

## What Changed

### Before (30+ tools):
- 30+ individual MCP tools
- Complex parameter sets
- Overlapping functionality
- Steep learning curve

### After (2-5 core tools):
- **2-4 essential tools** covering all functionality
- **Reference implementation compatible** (qdrant_store, qdrant_find)
- **Consolidated parameters** with intelligent defaults
- **Backward compatibility** maintained

## Migration Modes

### Environment Variable Configuration

Set `QDRANT_MCP_MODE` to control tool availability:

```bash
# Basic mode (2 tools) - Reference implementation compatible
export QDRANT_MCP_MODE=basic

# Standard mode (4 tools) - Recommended for most users  
export QDRANT_MCP_MODE=standard

# Full mode (30+ tools) - Legacy behavior for existing workflows
export QDRANT_MCP_MODE=full

# Compatible mode (2 tools) - Strict reference compatibility
export QDRANT_MCP_MODE=compatible
```

### Configuration File Support

Add to your YAML configuration:

```yaml
mcp:
  mode: "standard"  # "basic", "standard", "full", "compatible"
  tools:
    enable_advanced_search: true
    enable_watch_management: true
    enable_grpc_tools: false
```

## New Simplified Tools

### 1. `qdrant_store` - Universal Document Storage

**Purpose**: Store any type of information (documents, notes, scratchbook entries)

**Parameters**:
- `information` (required): Content to store
- `collection`: Target collection (auto-detected if None)  
- `metadata`: Document metadata dictionary
- `document_id`: Custom ID (UUID generated if None)
- `note_type`: "document", "note", "scratchbook", "code"
- `tags`: List of tags for organization
- `chunk_text`: Auto-chunking for large content (auto-detected if None)
- `title`: Title for scratchbook notes

**Examples**:
```python
# Basic usage (reference compatible)
await qdrant_store(
    information="Important meeting notes",
    collection="my-project"
)

# Advanced usage with metadata
await qdrant_store(
    information=file_content,
    collection="documentation", 
    metadata={"file_path": "/api/auth.py", "file_type": "python"},
    note_type="document",
    tags=["api", "authentication"],
    chunk_text=True
)

# Scratchbook note
await qdrant_store(
    information="Remember to update the API docs",
    note_type="scratchbook",
    tags=["todo", "documentation"],
    title="Documentation Update Reminder"
)
```

### 2. `qdrant_find` - Universal Search & Retrieval  

**Purpose**: Search across all content types with advanced filtering

**Parameters**:
- `query` (required): Search query
- `collection`: Specific collection (searches all if None)
- `limit`: Max results (default 10)
- `score_threshold`: Minimum relevance (default 0.7)
- `search_mode`: "hybrid", "semantic", "keyword", "exact"
- `filters`: Metadata filters dictionary
- `note_types`: Filter by content types
- `tags`: Filter by tags
- `include_relationships`: Include related documents

**Examples**:
```python
# Basic search (reference compatible)
results = await qdrant_find(
    query="authentication implementation"
)

# Advanced search with filtering
results = await qdrant_find(
    query="API documentation",
    collection="my-project",
    search_mode="hybrid",
    tags=["api", "docs"],
    note_types=["document"],
    limit=5,
    score_threshold=0.8
)

# Scratchbook search
notes = await qdrant_find(
    query="meeting notes",
    note_types=["scratchbook", "note"],
    tags=["meetings"],
    search_mode="semantic"
)
```

### 3. `qdrant_manage` - Workspace & Collection Management

**Purpose**: Handle workspace operations, document retrieval, and management tasks

**Parameters**:
- `action` (required): "status", "collections", "get", "list_notes", "delete"
- `collection`: Target collection
- `document_id`: Document identifier
- `note_id`: Note identifier  
- `include_vectors`: Include embeddings in response
- `note_type`: Filter by note type
- `tags`: Filter by tags
- `limit`: Max results for listing

**Examples**:
```python
# Get workspace status
status = await qdrant_manage(action="status")

# List available collections
collections = await qdrant_manage(action="collections")

# Get specific document
doc = await qdrant_manage(
    action="get",
    collection="my-project", 
    document_id="doc-123",
    include_vectors=True
)

# List scratchbook notes
notes = await qdrant_manage(
    action="list_notes",
    note_type="scratchbook",
    tags=["important"],
    limit=20
)

# Delete a note
result = await qdrant_manage(
    action="delete",
    note_id="note-456"
)
```

### 4. `qdrant_watch` - File Monitoring & Auto-Ingestion

**Purpose**: Manage folder watching for automatic document processing

**Parameters**:
- `action` (required): "add", "remove", "list", "status", "configure", "validate"
- `path`: Directory path to watch
- `collection`: Target collection for ingested files
- `watch_id`: Watch identifier
- `patterns`: File patterns to include
- `auto_ingest`: Enable automatic ingestion
- `recursive`: Watch subdirectories  
- `debounce_seconds`: Processing delay

**Examples**:
```python
# Add folder watch
result = await qdrant_watch(
    action="add",
    path="/home/user/documents",
    collection="my-project",
    patterns=["*.pdf", "*.txt", "*.md"],
    recursive=True,
    debounce_seconds=10
)

# List all watches
watches = await qdrant_watch(action="list")

# Get watch status  
status = await qdrant_watch(
    action="status",
    watch_id="my-watch"
)

# Configure existing watch
result = await qdrant_watch(
    action="configure", 
    watch_id="my-watch",
    patterns=["*.pdf", "*.docx"],
    auto_ingest=True
)

# Validate path before watching
validation = await qdrant_watch(
    action="validate",
    path="/path/to/watch"
)
```

## Tool Migration Mapping

### Document Operations

| Old Tool | New Tool | Migration Notes |
|----------|----------|------------------|
| `add_document_tool` | `qdrant_store` | Use `note_type="document"` |
| `update_scratchbook_tool` | `qdrant_store` | Use `note_type="scratchbook"` |
| `process_document_via_grpc_tool` | `qdrant_store` | File reading handled automatically |

### Search Operations

| Old Tool | New Tool | Migration Notes |
|----------|----------|------------------|
| `search_workspace_tool` | `qdrant_find` | Mode parameter mapping: dense→semantic, sparse→keyword |
| `search_scratchbook_tool` | `qdrant_find` | Use `note_types=["scratchbook"]` |
| `research_workspace` | `qdrant_find` | Use `include_relationships=True` |
| `hybrid_search_advanced_tool` | `qdrant_find` | Use `search_mode="hybrid"` |
| `search_by_metadata_tool` | `qdrant_find` | Use `filters` parameter with `search_mode="exact"` |
| `search_via_grpc_tool` | `qdrant_find` | Automatic routing to best available engine |

### Management Operations

| Old Tool | New Tool | Migration Notes |
|----------|----------|------------------|
| `workspace_status` | `qdrant_manage` | Use `action="status"` |
| `list_workspace_collections` | `qdrant_manage` | Use `action="collections"` |
| `get_document_tool` | `qdrant_manage` | Use `action="get"` |
| `list_scratchbook_notes_tool` | `qdrant_manage` | Use `action="list_notes"` |
| `delete_scratchbook_note_tool` | `qdrant_manage` | Use `action="delete"` |

### Watch Operations

| Old Tool | New Tool | Migration Notes |
|----------|----------|------------------|
| `add_watch_folder` | `qdrant_watch` | Use `action="add"` |
| `remove_watch_folder` | `qdrant_watch` | Use `action="remove"` |
| `list_watched_folders` | `qdrant_watch` | Use `action="list"` |
| `configure_watch_settings` | `qdrant_watch` | Use `action="configure"` |
| `get_watch_status` | `qdrant_watch` | Use `action="status"` |
| `validate_watch_path` | `qdrant_watch` | Use `action="validate"` |

## Backward Compatibility

### Automatic Tool Routing

When running in simplified mode, old tool calls are automatically routed to the new interface with deprecation warnings:

```bash
# Set mode to maintain compatibility while migrating
export QDRANT_MCP_MODE=standard

# Old tool calls still work but log migration suggestions
2025-09-08T16:30:15Z WARNING Deprecated tool 'add_document_tool' called - consider migrating to 'qdrant_store'
```

### Legacy Support

For critical workflows that cannot be immediately migrated:

```bash
# Temporarily revert to full tool set
export QDRANT_MCP_MODE=full

# Or enable advanced tools in standard mode
export QDRANT_MCP_ENABLE_ADVANCED=true
```

## Migration Strategy

### Phase 1: Assessment (Week 1)
1. **Audit current usage**: Review which tools your workflows use
2. **Set compatibility mode**: `export QDRANT_MCP_MODE=standard`
3. **Monitor deprecation warnings**: Check logs for tool migration suggestions
4. **Test basic functionality**: Ensure core workflows continue working

### Phase 2: Tool-by-Tool Migration (Week 2-3)
1. **Start with high-usage tools**: Migrate most frequently used tools first
2. **Update tool calls**: Replace old tools with simplified interface
3. **Test equivalency**: Verify results match previous behavior
4. **Update documentation**: Update internal docs and scripts

### Phase 3: Optimization (Week 4)
1. **Remove old tool references**: Clean up deprecated tool calls
2. **Optimize parameters**: Take advantage of intelligent defaults
3. **Enable basic mode**: `export QDRANT_MCP_MODE=basic` for maximum simplicity
4. **Performance validation**: Confirm performance improvements

## Code Examples

### Before/After Comparison

#### Document Storage
```python
# OLD: Multiple tools for different content types
await add_document_tool(
    content=doc_content,
    collection="my-project", 
    metadata={"type": "code"},
    document_id="file-123",
    chunk_text=True
)

await update_scratchbook_tool(
    content=note_content,
    note_id="note-456",
    title="Meeting Notes",
    tags=["meeting", "planning"]
)

# NEW: Single unified tool
await qdrant_store(
    information=doc_content,
    collection="my-project",
    metadata={"type": "code"},
    document_id="file-123", 
    note_type="document",
    chunk_text=True
)

await qdrant_store(
    information=note_content,
    document_id="note-456",
    title="Meeting Notes", 
    tags=["meeting", "planning"],
    note_type="scratchbook"
)
```

#### Search Operations
```python
# OLD: Different tools for different search types
workspace_results = await search_workspace_tool(
    query="authentication",
    collections=["my-project"],
    mode="hybrid", 
    limit=10
)

scratchbook_results = await search_scratchbook_tool(
    query="meeting notes",
    note_types=["note"],
    tags=["meetings"],
    limit=5
)

# NEW: Single unified search
all_results = await qdrant_find(
    query="authentication",
    collection="my-project",
    search_mode="hybrid",
    limit=10
)

note_results = await qdrant_find(
    query="meeting notes", 
    note_types=["note"],
    tags=["meetings"],
    limit=5
)
```

## Troubleshooting

### Common Migration Issues

#### 1. Parameter Name Changes
**Error**: `TypeError: unexpected keyword argument 'content'`
**Solution**: Use `information` instead of `content` in `qdrant_store`

#### 2. Collection List to String
**Error**: First collection not being selected correctly
**Solution**: Pass single collection string instead of list to simplified tools

#### 3. Mode Parameter Mapping
**Error**: Search mode not recognized
**Solution**: Use public modes ("hybrid", "semantic", "keyword") instead of internal modes

#### 4. Missing Tool in Simplified Mode
**Error**: `Tool 'advanced_tool_name' not available`
**Solution**: Either use equivalent simplified tool or enable advanced tools:
```bash
export QDRANT_MCP_ENABLE_ADVANCED=true
```

### Performance Considerations

#### Expected Performance Improvements:
- **Reduced memory usage**: Fewer registered tools and handlers
- **Faster initialization**: Simplified tool registration 
- **Better caching**: Consolidated parameter validation
- **Improved error handling**: Unified error recovery strategies

#### Monitoring Performance:
```python
# Enable performance monitoring
import os
os.environ["LOG_LEVEL"] = "DEBUG"

# Check tool registration time in logs
# Look for "Tool registration completed" with timing info
```

## Advanced Configuration

### Fine-Grained Tool Control

For organizations needing specific tool combinations:

```yaml
# config.yaml
mcp:
  mode: "custom"
  tools:
    # Core tools (always enabled)
    qdrant_store: true
    qdrant_find: true
    
    # Optional tools  
    qdrant_manage: true
    qdrant_watch: false  # Disable if not using file monitoring
    
    # Advanced features
    enable_grpc_tools: false
    enable_error_stats: false
    enable_advanced_watch: false
```

### Multi-Environment Configuration

```bash
# Development environment - full tools for debugging
export QDRANT_MCP_MODE=full
export QDRANT_MCP_ENABLE_ADVANCED=true

# Staging environment - standard tools  
export QDRANT_MCP_MODE=standard

# Production environment - minimal tools for performance
export QDRANT_MCP_MODE=basic
```

## Support and Feedback

### Getting Help

1. **Check deprecation warnings** in logs for specific migration guidance
2. **Review tool mapping table** for parameter translation
3. **Test in compatibility mode** before full migration
4. **Report issues** with specific tool combinations that don't work

### Feedback Channels

- **GitHub Issues**: Report bugs or migration problems
- **Documentation Requests**: Suggest improvements to migration guide
- **Feature Requests**: Request additional simplified tool functionality

### Migration Timeline

- **2025-09-15**: Simplified tools available in beta
- **2025-10-01**: Simplified tools become default for new installations
- **2025-11-01**: Full tool set marked as legacy (still supported)
- **2026-01-01**: Deprecation warnings added to full tool set
- **2026-06-01**: Full tool set moved to compatibility mode only

---

## Quick Reference Card

### Essential Commands
```bash
# Enable simplified mode
export QDRANT_MCP_MODE=standard

# Basic document storage
qdrant_store(information="content", collection="project")

# Basic search
qdrant_find(query="search term") 

# Get status
qdrant_manage(action="status")

# List collections
qdrant_manage(action="collections")
```

### Migration Checklist
- [ ] Set `QDRANT_MCP_MODE=standard` 
- [ ] Update tool calls to simplified interface
- [ ] Test core workflows
- [ ] Monitor deprecation warnings
- [ ] Update documentation and scripts
- [ ] Enable basic mode for production

This completes your migration to the simplified workspace-qdrant-mcp tool interface. The new system provides the same functionality with a much more intuitive and maintainable interface.