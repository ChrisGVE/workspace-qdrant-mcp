# workspace-qdrant-mcp PRD

**Version:** 1.0  
**Date:** 2025-08-27  
**Status:** Draft

## Executive Summary

The workspace-qdrant-mcp is a project-scoped Qdrant MCP server for workspace collections with scratchbook functionality. This Python implementation is a port of the claude-qdrant-mcp (TypeScript) with FastEmbed integration and project-aware collection management, designed to work alongside memexd daemon for comprehensive code indexing.

## Project Background

Following comprehensive community analysis of existing Qdrant MCP servers, the workspace-qdrant-mcp addresses the need for:

1. **Project-scoped R/W operations** on predefined collections
2. **FastEmbed integration** with all-MiniLM-L6-v2 embeddings  
3. **Python+FastMCP consistency** across the MCP ecosystem
4. **Scratchbook collections** for interactive workspace notes
5. **GitHub user-aware** project and submodule detection

## Architecture Overview

### Source Reference
- **Base Implementation**: [marlian/claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp)
- **Transposition**: TypeScript → Python using FastMCP framework
- **Integration**: Works with memexd daemon and qdrant-retrieve-mcp for complete system

### Collection Strategy

**Per Project/Subproject**:
- `{project-name}-scratchbook` - Interactive notes, ideas, context
- `{project-name}-docs` - Documentation (not code - reserved for memexd)

**Global Collections** (configurable):
- `docs` - Shared documentation
- `references` - Reference materials  
- `standards` - Coding standards, guidelines

**Exclusions**:
- NO `-code` collections (reserved for memexd daemon integration)
- NO destructive collection operations

## Core Features

### 1. Project Detection Logic

**Project Name Resolution**:
```python
def get_project_name(path: str, github_user: str = None) -> str:
    git_dir = find_git_root(path)
    if not git_dir:
        return os.path.basename(path)
    
    remote_url = get_git_remote_url(git_dir)
    if github_user and belongs_to_user(remote_url, github_user):
        return extract_repo_name_from_remote(remote_url)
    else:
        return os.path.basename(git_dir)
```

**Submodule Detection**:
- Scan git submodules in current project
- Filter by GitHub user ownership (if specified)
- Apply same naming logic as main project

### 2. Collection Management

**Startup Collection Discovery**:
1. Detect current project name and subprojects
2. Generate collection names using naming convention
3. Create collections if they don't exist
4. Initialize FastEmbed with all-MiniLM-L6-v2

**Collection Naming Convention**:
- Project collections: `{project-name}-scratchbook`, `{project-name}-docs`
- Global collections: As configured in GLOBAL_COLLECTIONS env var

### 3. FastEmbed Integration

**Embedding Configuration**:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Sparse Vectors**: BM25 implementation for hybrid search
- **Performance**: ONNX Runtime for fast inference

**Vector Storage**:
- Dense vectors: 384-dimensional embeddings
- Sparse vectors: BM25 term weights
- Named vectors in Qdrant collections

### 4. MCP Tools Interface

**Read Operations**:
```python
@mcp.tool()
async def search_workspace(query: str, collections: List[str] = None, 
                          mode: str = "hybrid", limit: int = 10):
    """Search across workspace collections with hybrid search"""

@mcp.tool()
async def list_workspace_collections():
    """List all available workspace collections"""
```

**Write Operations**:
```python  
@mcp.tool()
async def add_document(content: str, collection: str, 
                      metadata: Dict[str, Any] = None):
    """Add document to specified collection"""

@mcp.tool()  
async def update_scratchbook(content: str, note_id: str = None):
    """Add/update scratchbook note in current project collection"""
```

**Management Operations**:
```python
@mcp.tool()
async def workspace_status():
    """Get workspace and collection status information"""
```

## Configuration

### Environment Variables

```bash
# Collection Configuration
GLOBAL_COLLECTIONS=docs,references,standards
GITHUB_USER=chris  # Optional - for project name detection

# Qdrant Configuration  
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional

# Embedding Configuration
FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_SPARSE_VECTORS=true

# Performance Tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=50
```

### Startup Workflow

1. **Load configuration** from environment/config file
2. **Detect current project** using git directory analysis
3. **Scan submodules** and filter by GitHub user
4. **Initialize FastEmbed** with specified model
5. **Create collections** for detected projects (if missing)
6. **Register MCP tools** with FastMCP framework

## Implementation Phases

### Phase 1: Core Port (Days 1-5)
- [ ] Set up Python project structure with FastMCP
- [ ] Port TypeScript collection management to Python
- [ ] Implement project detection logic
- [ ] Integrate FastEmbed for embedding generation
- [ ] Basic search and document management tools

### Phase 2: Enhancement (Days 5-8)  
- [ ] Add BM25 sparse vector support
- [ ] Implement hybrid search with RRF fusion
- [ ] Add scratchbook-specific functionality
- [ ] GitHub user-aware submodule detection
- [ ] Configuration management and validation

### Phase 3: Integration (Days 8-10)
- [ ] Test integration with existing Qdrant instance
- [ ] Validate against memexd daemon compatibility
- [ ] Performance optimization and tuning
- [ ] Documentation and usage examples

## Success Criteria

1. **Functional Parity**: All claude-qdrant-mcp seeding/search capabilities preserved
2. **Python Migration**: Clean FastMCP implementation with proper tool interfaces  
3. **Project Awareness**: Automatic project/subproject collection detection
4. **FastEmbed Integration**: Fast embedding generation with all-MiniLM-L6-v2
5. **Workspace Focus**: Effective scratchbook and documentation management

## Technical Specifications

### Dependencies
- FastMCP (MCP server framework)
- FastEmbed (embedding generation) 
- qdrant-client (vector database client)
- GitPython (git repository analysis)

### Performance Targets
- Collection detection: < 1 second for typical projects
- Embedding generation: > 100 docs/second on CPU
- Search latency: < 200ms for workspace queries
- Memory usage: < 150MB RSS when active

### Security & Safety
- No collection deletion capabilities
- Project scope limitation (no system-wide access)
- GitHub user filtering for submodule access
- Environment variable configuration (no hardcoded credentials)

## Future Evolution

### Phase 2: Daemon Integration
- Remove embedding generation from MCP
- Delegate to memexd daemon for all processing
- MCP becomes thin search/management interface

### Enhancements
- Multi-project workspace support
- Advanced scratchbook organization
- Integration with IDE plugins
- Real-time collaboration features

## Conclusion

The workspace-qdrant-mcp provides a focused, safe, and efficient solution for project-scoped vector operations while maintaining compatibility with the broader Qdrant ecosystem. By porting proven TypeScript functionality to Python with modern embedding capabilities, it fills a critical gap in the unified Qdrant architecture.