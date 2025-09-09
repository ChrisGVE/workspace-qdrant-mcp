# API Documentation

## Workspace Qdrant MCP Server API

This document describes the MCP (Model Context Protocol) server API for workspace-qdrant-mcp.

### Overview

The workspace-qdrant-mcp server provides intelligent vector database operations with automatic project detection, hybrid search capabilities, and configurable collection management.

### Server Configuration

The server can be configured via environment variables or a YAML configuration file:

```bash
export PYTHONPATH=src
export QDRANT_URL=http://localhost:6333
export OPENAI_API_KEY=your_api_key_here
python -m workspace_qdrant_mcp.server
```

Or with a configuration file:
```bash
workspace-qdrant-mcp --config-file ~/.workspace-qdrant-mcp/config/default.yaml
```

### MCP Tools

The server exposes the following MCP tools for interaction with Claude and other MCP clients:

#### Document Management

**`ingest_files`**
- **Description**: Ingest files into a Qdrant collection
- **Parameters**:
  - `file_paths` (list[str]): List of file paths to ingest
  - `collection_name` (str, optional): Target collection name
  - `overwrite` (bool, optional): Whether to overwrite existing documents
- **Returns**: Ingestion results with document counts and status

**`delete_documents`**
- **Description**: Delete documents from a collection
- **Parameters**:
  - `collection_name` (str): Collection to delete from
  - `document_ids` (list[str], optional): Specific document IDs to delete
  - `file_paths` (list[str], optional): Delete documents by file path
- **Returns**: Deletion status and count of removed documents

#### Search Operations

**`search`**
- **Description**: Semantic search across collections
- **Parameters**:
  - `query` (str): Search query text
  - `collection_names` (list[str], optional): Collections to search
  - `limit` (int, optional): Maximum number of results (default: 10)
  - `score_threshold` (float, optional): Minimum similarity score
- **Returns**: List of search results with content, metadata, and scores

**`hybrid_search`**
- **Description**: Combined semantic and keyword search
- **Parameters**:
  - `query` (str): Search query text
  - `collection_names` (list[str], optional): Collections to search
  - `limit` (int, optional): Maximum number of results
  - `semantic_weight` (float, optional): Weight for semantic search (0.0-1.0)
- **Returns**: Ranked search results using reciprocal rank fusion

**`advanced_search`**
- **Description**: Advanced search with metadata filtering
- **Parameters**:
  - `query` (str): Search query text
  - `collection_names` (list[str], optional): Collections to search
  - `filters` (dict, optional): Metadata filters
  - `limit` (int, optional): Maximum number of results
  - `score_threshold` (float, optional): Minimum similarity score
- **Returns**: Filtered search results

#### Collection Management

**`list_collections`**
- **Description**: List all available collections
- **Parameters**: None
- **Returns**: List of collection names and metadata

**`create_collection`**
- **Description**: Create a new collection
- **Parameters**:
  - `collection_name` (str): Name of the collection to create
  - `vector_size` (int, optional): Vector dimension size
  - `distance_metric` (str, optional): Distance metric for similarity
- **Returns**: Collection creation status

**`delete_collection`**
- **Description**: Delete an entire collection
- **Parameters**:
  - `collection_name` (str): Name of the collection to delete
  - `confirm` (bool): Confirmation flag (must be true)
- **Returns**: Deletion status

**`get_collection_info`**
- **Description**: Get detailed information about a collection
- **Parameters**:
  - `collection_name` (str): Collection name
- **Returns**: Collection metadata, document count, and statistics

#### Project and Workspace Management

**`detect_project`**
- **Description**: Detect project information from current directory
- **Parameters**: None
- **Returns**: Project metadata, detected collections, and configuration

**`create_project_collections`**
- **Description**: Create standard project collections
- **Parameters**:
  - `project_name` (str, optional): Override detected project name
  - `collection_types` (list[str], optional): Types of collections to create
- **Returns**: Created collection names and status

**`setup_auto_ingestion`**
- **Description**: Configure automatic file ingestion for the project
- **Parameters**:
  - `project_path` (str, optional): Path to project directory
  - `watch_patterns` (list[str], optional): File patterns to watch
  - `enabled` (bool, optional): Enable/disable auto-ingestion
- **Returns**: Auto-ingestion configuration status

#### Scratchbook Operations

**`add_scratchbook_entry`**
- **Description**: Add an entry to the project scratchbook
- **Parameters**:
  - `content` (str): Entry content
  - `title` (str, optional): Entry title
  - `tags` (list[str], optional): Associated tags
- **Returns**: Entry ID and storage status

**`search_scratchbook`**
- **Description**: Search scratchbook entries
- **Parameters**:
  - `query` (str): Search query
  - `limit` (int, optional): Maximum results
  - `tags` (list[str], optional): Filter by tags
- **Returns**: Matching scratchbook entries

**`list_scratchbook_entries`**
- **Description**: List all scratchbook entries
- **Parameters**:
  - `limit` (int, optional): Maximum entries to return
  - `offset` (int, optional): Number of entries to skip
- **Returns**: List of scratchbook entries with metadata

### Error Handling

All tools return structured error information when operations fail:

```json
{
  "error": "CollectionNotFound",
  "message": "Collection 'nonexistent' does not exist",
  "details": {
    "available_collections": ["project-main", "project-docs"]
  }
}
```

Common error types:
- `CollectionNotFound`: Specified collection doesn't exist
- `InvalidParameters`: Invalid or missing required parameters
- `EmbeddingError`: Issues with text embedding generation
- `QdrantConnectionError`: Database connection problems
- `FileNotFound`: Specified file path doesn't exist
- `PermissionError`: Insufficient permissions for file operations

### Configuration Reference

See [configuration.md](configuration.md) for detailed configuration options including:
- Qdrant connection settings
- Embedding model configuration  
- Auto-ingestion patterns and rules
- Performance tuning parameters
- Security and access control

### Examples

#### Basic Search Example
```python
# Search across all collections
results = await mcp_client.call_tool("search", {
    "query": "authentication implementation",
    "limit": 5
})
```

#### Project Setup Example
```python
# Detect and set up project
project_info = await mcp_client.call_tool("detect_project")
collections = await mcp_client.call_tool("create_project_collections", {
    "project_name": project_info["name"]
})
```

#### File Ingestion Example
```python
# Ingest documentation files
result = await mcp_client.call_tool("ingest_files", {
    "file_paths": ["README.md", "docs/api.md", "CONTRIBUTING.md"],
    "collection_name": "project-docs"
})
```

For more examples and usage patterns, see the [examples/](../examples/) directory.

### Performance Considerations

- **Batch Operations**: Use batch ingestion for multiple files
- **Collection Scoping**: Limit searches to relevant collections
- **Caching**: Results are cached for improved performance
- **Resource Limits**: Configure memory and processing limits appropriately

### Security Notes

- API keys should be stored securely as environment variables
- File access is limited to configured project directories
- Collection names follow project-scoped naming conventions
- Web ingestion can be disabled via configuration for security

### Support and Troubleshooting

For issues and support:
1. Check the [troubleshooting guide](CONTAINER_TROUBLESHOOTING.md)
2. Review server logs for detailed error information
3. Verify Qdrant service is running and accessible
4. Check file permissions for ingestion operations

See also:
- [Installation Guide](../INSTALLATION.md)
- [Configuration Guide](configuration.md)
- [Developer Guide](../CONTRIBUTING.md)