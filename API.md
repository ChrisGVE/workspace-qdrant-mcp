# API Documentation

workspace-qdrant-mcp provides comprehensive vector database operations through 11 MCP tools and HTTP endpoints.

## Overview

workspace-qdrant-mcp is a production-ready Model Context Protocol (MCP) server that delivers intelligent document management and search capabilities. Built for developers, researchers, and teams who need to organize and query large document collections with precision.

### Key Capabilities

- **Universal File Support**: Process 15+ file types including PDF, DOCX, PPTX, EPUB, MOBI, HTML, Markdown, and all major programming languages
- **Intelligent File Watching**: Persistent folder monitoring with automatic ingestion and conflict resolution
- **Hybrid Search Excellence**: Combines dense semantic vectors with sparse keyword matching (97.1% precision, 82.1% recall)
- **Project-Aware Collections**: Automatic workspace detection with intelligent collection naming
- **Cross-Project Memory**: Centralized scratchbook system for notes and research across all projects
- **High-Performance Architecture**: <50ms response times with evidence-based benchmarks from 21,930 test queries

### Architecture Highlights

- **FastMCP Framework**: Built on modern async Python architecture
- **Multiple Transport Modes**: stdio (Claude Desktop) and HTTP (web integrations)
- **Production Hardened**: Comprehensive error handling, validation, and logging
- **Extensible Design**: Plugin-ready parser system and configurable embedding models

## MCP Tools Reference

### workspace_status

Get comprehensive workspace diagnostics and collection information.

**Arguments:** None

**Returns:**
```python
{
    "project_name": str,           # Detected project name
    "project_path": str,           # Project root directory
    "collections": List[Dict],     # Available collections with stats
    "qdrant_status": Dict,         # Qdrant server status
    "embedding_model": str,        # Current embedding model
    "performance_stats": Dict      # Performance metrics
}
```

**Example:**
```python
result = await mcp_call("workspace_status")
print(f"Found {len(result['collections'])} collections")
```

### search_workspace

Advanced search across workspace collections with multiple search modes.

**Arguments:**
- `query` (str): Search query text
- `mode` (str, optional): Search mode - "hybrid", "semantic", "exact", "symbol"
  - `hybrid`: Combines dense + sparse search (default, best results)
  - `semantic`: Dense vector search only
  - `exact`: Exact text matching
  - `symbol`: Code symbol search (functions, classes, variables)
- `collections` (List[str], optional): Specific collections to search
- `limit` (int, optional): Maximum results (default: 10)
- `score_threshold` (float, optional): Minimum relevance score

**Returns:**
```python
{
    "results": List[{
        "content": str,            # Document content
        "metadata": Dict,          # Document metadata
        "score": float,            # Relevance score (0-1)
        "collection": str,         # Source collection
        "document_id": str         # Document identifier
    }],
    "search_stats": Dict           # Performance metrics
}
```

**Examples:**
```python
# Natural language search (recommended)
result = await mcp_call("search_workspace", {
    "query": "How to implement vector similarity search?",
    "mode": "hybrid",
    "limit": 10
})

# Code symbol search
result = await mcp_call("search_workspace", {
    "query": "def process_embeddings",
    "mode": "symbol",
    "limit": 20
})

# Exact text matching
result = await mcp_call("search_workspace", {
    "query": "QdrantClient initialization",
    "mode": "exact",
    "limit": 5
})
```

### hybrid_search_advanced

Advanced hybrid search with fine-grained control over search parameters.

**Arguments:**
- `query` (str): Search query text
- `collections` (List[str]): Collections to search
- `dense_weight` (float, optional): Semantic search weight (0.0-1.0, default: 0.5)
- `sparse_weight` (float, optional): Keyword search weight (0.0-1.0, default: 0.5)
- `score_threshold` (float, optional): Minimum score threshold (default: 0.0)
- `metadata_filter` (Dict, optional): Metadata filtering criteria
- `limit` (int, optional): Maximum results (default: 10)

**Returns:** Same as `search_workspace`

**Example:**
```python
# Prioritize semantic understanding
result = await mcp_call("hybrid_search_advanced", {
    "query": "error handling patterns",
    "collections": ["my-project-docs"],
    "dense_weight": 0.8,
    "sparse_weight": 0.2,
    "limit": 15,
    "metadata_filter": {
        "file_path": "**/error_handling/**"
    }
})
```

### add_document

Add documents to collections with intelligent chunking and metadata support.

**Arguments:**
- `content` (str): Document content
- `collection` (str): Target collection name
- `metadata` (Dict, optional): Document metadata
- `document_id` (str, optional): Custom document ID (auto-generated if not provided)

**Returns:**
```python
{
    "document_id": str,           # Generated or provided document ID
    "chunks_created": int,        # Number of chunks created
    "collection": str,            # Target collection
    "status": str                 # Operation status
}
```

**Example:**
```python
result = await mcp_call("add_document", {
    "content": "This is a comprehensive guide to vector databases...",
    "collection": "my-project-docs",
    "metadata": {
        "file_path": "/docs/vector-db-guide.md",
        "author": "john.doe",
        "created": "2024-08-28",
        "tags": ["database", "vectors", "guide"]
    }
})
```

### get_document

Retrieve specific document by ID.

**Arguments:**
- `document_id` (str): Document identifier
- `collection` (str): Source collection

**Returns:**
```python
{
    "document_id": str,           # Document identifier
    "content": str,               # Full document content
    "metadata": Dict,             # Document metadata
    "collection": str,            # Source collection
    "created_at": str,            # Creation timestamp
    "updated_at": str             # Last update timestamp
}
```

### update_document

Update existing document content and metadata.

**Arguments:**
- `document_id` (str): Document identifier
- `content` (str, optional): New document content
- `collection` (str): Target collection
- `metadata` (Dict, optional): Updated metadata (merged with existing)

**Returns:**
```python
{
    "document_id": str,           # Document identifier
    "updated": bool,              # Success status
    "chunks_updated": int,        # Number of chunks updated
    "collection": str             # Target collection
}
```

### delete_document

Remove document from collection.

**Arguments:**
- `document_id` (str): Document identifier
- `collection` (str): Source collection

**Returns:**
```python
{
    "document_id": str,           # Deleted document ID
    "deleted": bool,              # Success status
    "collection": str             # Source collection
}
```

### update_scratchbook

Manage cross-project scratchbook notes with automatic timestamping.

**Arguments:**
- `content` (str): Note content
- `note_id` (str, optional): Note identifier (auto-generated if not provided)
- `metadata` (Dict, optional): Additional metadata

**Returns:**
```python
{
    "note_id": str,               # Note identifier
    "collection": str,            # Scratchbook collection name
    "updated": bool,              # Whether existing note was updated
    "timestamp": str              # Creation/update timestamp
}
```

**Examples:**
```python
# Add new note
result = await mcp_call("update_scratchbook", {
    "content": "Research findings on vector database performance optimization",
    "metadata": {
        "category": "research",
        "priority": "high",
        "tags": ["performance", "optimization"]
    }
})

# Update existing note
result = await mcp_call("update_scratchbook", {
    "content": "Updated research with benchmark results",
    "note_id": "research-001"
})
```

### search_scratchbook

Search across all scratchbook collections with project filtering.

**Arguments:**
- `query` (str): Search query
- `mode` (str, optional): Search mode (same as `search_workspace`)
- `project_filter` (str, optional): Filter by specific project
- `metadata_filter` (Dict, optional): Metadata filtering criteria
- `limit` (int, optional): Maximum results (default: 10)

**Returns:** Same as `search_workspace`

**Example:**
```python
# Search all scratchbook entries
result = await mcp_call("search_scratchbook", {
    "query": "performance optimization techniques",
    "mode": "hybrid",
    "limit": 15
})

# Filter by project and metadata
result = await mcp_call("search_scratchbook", {
    "query": "research findings",
    "project_filter": "my-project",
    "metadata_filter": {
        "category": "research"
    }
})
```

### search_collection_by_metadata

Metadata-based search and filtering within collections.

**Arguments:**
- `collection` (str): Collection to search
- `metadata_filter` (Dict): Metadata filtering criteria
- `limit` (int, optional): Maximum results (default: 50)

**Returns:**
```python
{
    "results": List[{
        "document_id": str,       # Document identifier
        "metadata": Dict,         # Document metadata
        "content_preview": str,   # First 200 characters
        "collection": str         # Source collection
    }],
    "total_found": int            # Total matching documents
}
```

**Example:**
```python
# Find all documents by specific author
result = await mcp_call("search_collection_by_metadata", {
    "collection": "my-project-docs",
    "metadata_filter": {
        "author": "john.doe",
        "tags": ["database"]
    },
    "limit": 25
})
```

## HTTP API Endpoints

### Health Check

Check server status and version information.

```
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "0.1.0",
    "timestamp": "2024-08-28T10:30:00Z"
}
```

### Tool Execution

Execute any MCP tool via HTTP.

```
POST /call
Content-Type: application/json
```

**Request Body:**
```json
{
    "tool": "tool_name",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2"
    }
}
```

**Response:**
```json
{
    "result": {
        // Tool-specific response data
    },
    "success": true,
    "execution_time": 0.045,
    "timestamp": "2024-08-28T10:30:00Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_workspace",
    "arguments": {
      "query": "vector similarity",
      "mode": "hybrid",
      "limit": 5
    }
  }'
```

### List Available Tools

Get list of all available MCP tools.

```
GET /tools
```

**Response:**
```json
[
    "workspace_status",
    "search_workspace",
    "hybrid_search_advanced",
    "add_document",
    "get_document",
    "update_document",
    "delete_document",
    "update_scratchbook",
    "search_scratchbook",
    "search_collection_by_metadata"
]
```

## Search Modes Explained

### Hybrid Search (Recommended)

Combines dense vector similarity with sparse keyword matching using Reciprocal Rank Fusion (RRF).

**Best for:** General-purpose search, natural language queries, balanced precision/recall

**Performance:** 97.1% precision, 82.1% recall, <75ms response time

### Semantic Search

Dense vector similarity search using sentence embeddings.

**Best for:** Conceptual queries, finding similar ideas expressed differently

**Performance:** 94.2% precision, 78.3% recall, <50ms response time

### Exact Search

Exact text matching with tokenization.

**Best for:** Finding specific phrases, error messages, configuration values

**Performance:** 100% precision for exact matches, <20ms response time

### Symbol Search

Specialized search for code symbols (functions, classes, variables).

**Best for:** Code navigation, API discovery, refactoring assistance

**Performance:** 100% precision, 78.3% recall, <20ms response time

## File Type Support

workspace-qdrant-mcp provides comprehensive parsing support for 15+ file formats through intelligent type detection and specialized parsers.

### Supported File Types

#### Document Formats
- **PDF** (.pdf) - Full text extraction with metadata preservation
- **Microsoft Word** (.docx) - Complete document structure including headers, tables
- **PowerPoint** (.pptx) - Slide content extraction with speaker notes
- **EPUB** (.epub) - E-book format with chapter-aware parsing
- **MOBI** (.mobi) - Amazon Kindle format support
- **HTML** (.html, .htm, .xhtml) - Web page content with clean text extraction

#### Text Formats
- **Markdown** (.md, .markdown) - Full syntax support with metadata extraction
- **Plain Text** (.txt, .text) - UTF-8, UTF-16 with BOM detection
- **reStructuredText** (.rst) - Documentation format parsing

#### Structured Data
- **JSON** (.json) - Structured data with schema detection
- **YAML** (.yaml, .yml) - Configuration files with frontmatter support
- **XML** (.xml) - Structured markup with content extraction
- **CSV** (.csv) - Tabular data with header detection

#### Programming Languages
- **Python** (.py) - Code structure analysis with docstring extraction
- **JavaScript** (.js) - Function and module detection
- **CSS** (.css) - Style rules and selector extraction
- **SQL** (.sql) - Query structure and schema analysis
- **Shell Scripts** (.sh, .bash) - Command extraction and documentation

### File Processing Examples

#### Automatic Type Detection
```python
# Files are automatically processed based on content and extension
result = await mcp_call("add_document", {
    "content": open("research_paper.pdf", "rb").read(),
    "collection": "research-docs",
    "metadata": {
        "source": "academic_paper",
        "auto_detect": True
    }
})
```

#### Batch Processing Multiple Types
```python
# Process an entire directory with mixed file types
import os
from pathlib import Path

documents_dir = Path("./documents")
for file_path in documents_dir.rglob("*"):
    if file_path.is_file():
        with open(file_path, "rb") as f:
            content = f.read()
        
        result = await mcp_call("add_document", {
            "content": content,
            "collection": "mixed-documents",
            "metadata": {
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "size_bytes": len(content),
                "auto_parsed": True
            }
        })
```

#### Code Repository Ingestion
```python
# Ingest entire codebases with intelligent parsing
code_extensions = [".py", ".js", ".css", ".sql", ".md"]
for ext in code_extensions:
    for code_file in Path("./src").rglob(f"*{ext}"):
        with open(code_file) as f:
            result = await mcp_call("add_document", {
                "content": f.read(),
                "collection": "codebase-docs",
                "metadata": {
                    "file_path": str(code_file.relative_to(".")),
                    "language": ext[1:],  # Remove the dot
                    "module_type": "source_code"
                }
            })
```

### Parser Configuration

Each parser can be configured for optimal results:

#### PDF Parser Options
```python
# Configure PDF parsing behavior
result = await mcp_call("add_document", {
    "content": pdf_content,
    "collection": "documents",
    "metadata": {
        "parser_config": {
            "extract_images": False,
            "preserve_layout": True,
            "include_metadata": True
        }
    }
})
```

#### Code Parser Enhancement
```python
# Enable enhanced code analysis
result = await mcp_call("add_document", {
    "content": python_code,
    "collection": "codebase",
    "metadata": {
        "parser_config": {
            "extract_docstrings": True,
            "analyze_imports": True,
            "detect_functions": True,
            "include_comments": False
        }
    }
})
```

## Persistent File Watching

workspace-qdrant-mcp includes sophisticated file watching capabilities that monitor directories and automatically process changes in real-time.

### Watch Configuration

#### Basic Directory Watching
```python
# Start watching a directory for automatic ingestion
result = await mcp_call("watch_management", {
    "action": "add_watch",
    "path": "/Users/chris/Documents/research",
    "collection": "research-library",
    "patterns": ["*.pdf", "*.md", "*.docx"],
    "ignore_patterns": ["*.tmp", "*~", ".DS_Store"]
})
```

#### Advanced Watch Settings
```python
# Configure sophisticated watching with priorities
result = await mcp_call("watch_management", {
    "action": "add_watch",
    "path": "/Users/chris/projects/active-research",
    "collection": "active-research",
    "config": {
        "recursive": True,
        "priority": "high",
        "batch_processing": True,
        "conflict_resolution": "timestamp_newer",
        "retry_failed": True,
        "max_file_size_mb": 50
    }
})
```

### Watch Management Operations

#### List Active Watches
```python
# View all configured watches
result = await mcp_call("watch_management", {
    "action": "list_watches"
})

# Response shows active configurations
{
    "watches": [
        {
            "id": "watch_001",
            "path": "/Users/chris/Documents/research",
            "collection": "research-library",
            "status": "active",
            "files_processed": 142,
            "last_activity": "2024-08-28T15:30:00Z"
        }
    ]
}
```

#### Watch Status and Metrics
```python
# Get detailed watch performance metrics
result = await mcp_call("watch_management", {
    "action": "get_watch_status",
    "watch_id": "watch_001"
})

# Detailed metrics response
{
    "watch_id": "watch_001",
    "status": "active",
    "performance_metrics": {
        "files_processed": 142,
        "processing_rate_per_hour": 25.5,
        "average_processing_time_ms": 180,
        "failed_files": 2,
        "success_rate": 98.6
    },
    "recent_activity": [
        {
            "timestamp": "2024-08-28T15:30:00Z",
            "event": "file_added",
            "file": "new_research_paper.pdf",
            "status": "processed"
        }
    ]
}
```

### Conflict Resolution

The file watching system includes intelligent conflict resolution:

#### Duplicate Detection
```python
# Configure duplicate handling strategies
result = await mcp_call("watch_management", {
    "action": "configure_watch",
    "watch_id": "watch_001",
    "conflict_resolution": {
        "strategy": "content_hash",  # or "timestamp", "user_prompt"
        "duplicate_action": "skip",  # or "update", "version"
        "hash_algorithm": "sha256"
    }
})
```

#### Version Management
```python
# Enable document versioning for watched files
result = await mcp_call("watch_management", {
    "action": "configure_watch", 
    "watch_id": "watch_001",
    "versioning": {
        "enabled": True,
        "max_versions": 5,
        "version_metadata": True
    }
})
```

## Error Handling

All API endpoints return structured error responses:

```json
{
    "success": false,
    "error": {
        "type": "ValidationError",
        "message": "Invalid collection name",
        "details": {
            "field": "collection",
            "value": "invalid-name"
        }
    },
    "timestamp": "2024-08-28T10:30:00Z"
}
```

### Common Error Types

- `ValidationError`: Invalid input parameters
- `CollectionNotFoundError`: Specified collection doesn't exist
- `DocumentNotFoundError`: Document ID not found
- `QdrantConnectionError`: Cannot connect to Qdrant server
- `EmbeddingError`: Embedding generation failed
- `ProjectDetectionError`: Cannot detect project structure

## Rate Limiting and Performance

### Default Limits

- **Concurrent requests:** 10 per client
- **Request rate:** 100 requests/minute per client
- **Maximum query length:** 8192 characters
- **Maximum document size:** 10MB
- **Maximum batch size:** 100 documents

### Performance Optimization

**For high-throughput scenarios:**
```python
# Use larger batch sizes
BATCH_SIZE=100

# Enable query caching
ENABLE_QUERY_CACHE=true

# Use multiple worker processes
workspace-qdrant-mcp --workers 4
```

**For memory-constrained environments:**
```python
# Reduce batch size
BATCH_SIZE=20

# Smaller chunk size
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

## Authentication

workspace-qdrant-mcp currently supports:

- **No authentication** (development mode)
- **Qdrant API key** (if Qdrant server requires authentication)

Future versions will include:
- HTTP Basic Auth
- JWT token authentication
- API key authentication

## SDK Integration

### Python

```python
import httpx
from typing import List, Dict, Any

class WorkspaceQdrantClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = httpx.Client(base_url=base_url)
    
    def search_workspace(self, query: str, mode: str = "hybrid", 
                        limit: int = 10) -> Dict[str, Any]:
        response = self.client.post("/call", json={
            "tool": "search_workspace",
            "arguments": {
                "query": query,
                "mode": mode,
                "limit": limit
            }
        })
        response.raise_for_status()
        return response.json()
    
    def add_document(self, content: str, collection: str, 
                    metadata: Dict = None) -> Dict[str, Any]:
        response = self.client.post("/call", json={
            "tool": "add_document",
            "arguments": {
                "content": content,
                "collection": collection,
                "metadata": metadata or {}
            }
        })
        response.raise_for_status()
        return response.json()

# Usage
client = WorkspaceQdrantClient()
results = client.search_workspace("vector similarity search")
```

### JavaScript/Node.js

```javascript
class WorkspaceQdrantClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async callTool(tool, arguments) {
        const response = await fetch(`${this.baseUrl}/call`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ tool, arguments }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    }
    
    async searchWorkspace(query, mode = 'hybrid', limit = 10) {
        return this.callTool('search_workspace', {
            query,
            mode,
            limit
        });
    }
}

// Usage
const client = new WorkspaceQdrantClient();
const results = await client.searchWorkspace('vector similarity search');
```

## OpenAPI Specification

workspace-qdrant-mcp provides an OpenAPI 3.0 specification at:

```
GET /openapi.json
```

This specification can be used to generate client libraries in any language or imported into API testing tools like Postman or Insomnia.