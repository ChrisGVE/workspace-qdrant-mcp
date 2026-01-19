# Multi-Tenant Search Examples

Comprehensive examples demonstrating search scenarios in the unified multi-tenant collection architecture.

## Table of Contents

- [Understanding Search Scope](#understanding-search-scope)
- [Basic Search Scenarios](#basic-search-scenarios)
- [Library Search Scenarios](#library-search-scenarios)
- [Cross-Project Search](#cross-project-search)
- [Filtered Search Scenarios](#filtered-search-scenarios)
- [Common Pitfalls and Empty Results](#common-pitfalls-and-empty-results)
- [Advanced Scenarios](#advanced-scenarios)

## Understanding Search Scope

The unified multi-tenant architecture uses 4 collection types:

| Collection | Contents | Tenant Isolation |
|------------|----------|------------------|
| `_projects` | ALL project code and docs | `tenant_id` (12-char hex) |
| `_libraries` | ALL library documentation | `library_name` |
| `{user}-{type}` | User notes (e.g., `work-notes`) | Optional `project_id` |
| `_memory` | System rules and preferences | N/A |

**Search scopes:**
- `"project"` (default): Current project only via `tenant_id` filter
- `"global"`: Global collections (user collections, memory)
- `"all"`: All projects in `_projects` collection

## Basic Search Scenarios

### 1. Search Current Project Only (Default)

Search within the current project, excluding libraries and other projects.

```python
# MCP Tool
search(query="authentication middleware")

# Equivalent with explicit parameters
search(query="authentication middleware", scope="project", include_libraries=False)

# CLI
wqm search "authentication middleware"
wqm search "authentication middleware" --scope project
```

**Expected Results:**
- Documents from current project's code and documentation
- Filtered by current project's `tenant_id`
- Libraries NOT included

### 2. Search Current Project with Libraries

Include reference documentation from libraries alongside project code.

```python
# MCP Tool
search(query="FastAPI dependency injection", include_libraries=True)

# CLI
wqm search "FastAPI dependency injection" --include-libraries
```

**Expected Results:**
- Current project documents matching the query
- Library documentation from `_libraries` collection (FastAPI docs, tutorials)
- Useful for finding both implementation examples AND reference material

### 3. Search Global Collections

Search user notes and memory collections, not project content.

```python
# MCP Tool
search(query="meeting notes API design", scope="global")

# CLI
wqm search "meeting notes API design" --scope global
```

**Expected Results:**
- Documents from user collections (e.g., `work-notes`, `myapp-scratchbook`)
- Memory rules if matching
- Project code NOT included

## Library Search Scenarios

### 4. Search All Libraries

Search across all library documentation.

```python
# MCP Tool
search(query="async patterns", scope="project", include_libraries=True)

# The include_libraries parameter adds _libraries to the search
# Library results are interleaved with project results via RRF ranking
```

**Expected Results:**
- Library documentation about async patterns from any library
- Project code also included if scope="project" or scope="all"

### 5. Search Specific Library

Filter library search to a specific library name.

```python
# MCP Tool - Use filter parameter
search(
    query="routing patterns",
    include_libraries=True,
    filter={"library_name": "fastapi"}
)

# CLI - Search specific library
wqm library search fastapi "routing patterns"
```

**Expected Results:**
- Only FastAPI library documentation about routing
- Other libraries (React, NumPy, etc.) excluded

### 6. Compare Implementations Across Libraries

Search for a concept across multiple libraries.

```python
# MCP Tool
search(
    query="state management patterns",
    scope="all",
    include_libraries=True
)
```

**Expected Results:**
- React's state management docs (useState, Redux)
- Vue's state management docs (Pinia, Vuex)
- Project implementations of state management
- Results ranked by relevance via RRF

## Cross-Project Search

### 7. Search All Projects

Discover implementations across your entire codebase.

```python
# MCP Tool
search(query="rate limiting implementation", scope="all")

# CLI
wqm search "rate limiting implementation" --scope all
```

**Expected Results:**
- Rate limiting code from ALL projects
- Each result includes `tenant_id` indicating source project
- Useful for finding reusable patterns

### 8. Search All Projects with Libraries

Most comprehensive search - everything searchable.

```python
# MCP Tool
search(
    query="JWT authentication",
    scope="all",
    include_libraries=True
)

# CLI
wqm search "JWT authentication" --scope all --include-libraries
```

**Expected Results:**
- JWT implementations from all projects
- JWT documentation from libraries (auth0, jose, etc.)
- Results ranked by semantic relevance

## Filtered Search Scenarios

### 9. Branch-Specific Search

Search within a specific git branch.

```python
# MCP Tool - Search specific branch
search(query="new feature", branch="feature/auth")

# Search all branches
search(query="deprecated function", branch="*")

# CLI
wqm search "new feature" --branch feature/auth
wqm search "deprecated function" --branch "*"
```

**Expected Results:**
- `branch="feature/auth"`: Only documents from that branch
- `branch="*"`: Documents from all branches (useful for finding removed code)

### 10. File Type Filtering

Filter by content type (code, docs, tests, config).

```python
# MCP Tool - Code only
search(query="database connection", file_type="code")

# Tests only
search(query="authentication", file_type="test")

# Documentation only
search(query="API reference", file_type="doc")

# CLI
wqm search "database connection" --file-type code
wqm search "authentication" --file-type test
```

**Expected Results:**
- Results filtered to specified file type
- `code`: .py, .js, .ts, .rs, etc.
- `test`: test_*.py, *.test.js, etc.
- `doc`: .md, .rst, .txt documentation
- `config`: .yaml, .json, .toml configuration

### 11. Combined Filters

Use multiple filters together.

```python
# MCP Tool - Code on specific branch in all projects
search(
    query="async handler",
    scope="all",
    branch="main",
    file_type="code"
)

# CLI
wqm search "async handler" --scope all --branch main --file-type code
```

**Expected Results:**
- Only production code (main branch)
- Only source code files (not tests/docs)
- From all projects

## Common Pitfalls and Empty Results

### 12. Missing Libraries (Empty Results)

**Scenario:** Searching for library documentation but getting no results.

```python
# Query
search(query="pandas DataFrame", scope="project")

# Expected: Empty or minimal results
# Reason: Libraries NOT included by default
```

**Solution:**
```python
# Add include_libraries parameter
search(query="pandas DataFrame", include_libraries=True)
```

### 13. Wrong Scope (Empty Results)

**Scenario:** Searching for content that exists but isn't found.

```python
# Query - Searching for code from another project
search(query="auth-service middleware", scope="project")

# Expected: Empty results
# Reason: Scope is "project" but content is in different project
```

**Solution:**
```python
# Use scope="all" for cross-project search
search(query="auth-service middleware", scope="all")
```

### 14. Branch Mismatch (Empty Results)

**Scenario:** Searching for code that was deleted from current branch.

```python
# Query - Code exists on feature branch but not main
search(query="experimental feature")

# Expected: Empty results if on main branch
# Reason: Default branch filter is current branch
```

**Solution:**
```python
# Search all branches
search(query="experimental feature", branch="*")

# Or search specific branch
search(query="experimental feature", branch="feature/experimental")
```

### 15. Project Not Registered

**Scenario:** New project content not appearing in search.

```python
# Query
search(query="new project code", scope="all")

# Expected: Empty results
# Reason: Project not registered with daemon, no tenant_id assigned
```

**Diagnosis:**
```bash
# Check project registration
wqm admin status

# List registered projects
wqm admin collections --verbose
```

**Solution:**
```bash
# Ensure MCP server has connected from project directory
# Or manually register via CLI
wqm watch add /path/to/new-project --collection new-project-code
```

## Advanced Scenarios

### 16. Semantic vs Exact Search

Compare search modes for different use cases.

```python
# Semantic search - Finds conceptually similar content
search(query="user authentication flow", mode="semantic")
# Results: Auth code, login handlers, session management, OAuth implementations

# Exact search - Finds literal matches
search(query="def authenticate_user", mode="exact")
# Results: Only functions literally named authenticate_user

# Hybrid search (default) - Best of both
search(query="authenticate_user", mode="hybrid")
# Results: Exact matches ranked higher, semantic matches included
```

### 17. Research Mode (CLI)

Deep research across codebase with analysis.

```bash
# CLI research mode
wqm search research "microservices communication patterns"

# Output includes:
# - Matched documents with context
# - Pattern analysis across projects
# - Related concepts detected
# - Suggested next searches
```

### 18. Memory Search

Search behavioral rules and preferences.

```bash
# CLI memory search
wqm search memory "coding style preferences"

# Returns:
# - User preferences (e.g., "always use type hints")
# - Project rules (e.g., "this project uses pytest")
# - Agent behaviors (e.g., "prefer async over threading")
```

### 19. Pagination for Large Results

Handle many results efficiently.

```python
# MCP Tool - Limit results
search(query="import", scope="all", limit=50)

# CLI with pagination
wqm search "import" --scope all --limit 50
```

### 20. Response Format

Understanding search response structure.

```python
response = search(query="authentication", scope="all", include_libraries=True)

# Response structure:
{
    "results": [
        {
            "document_id": "abc123",
            "content": "...",
            "score": 0.89,
            "metadata": {
                "tenant_id": "github_com_user_repo",  # Project identifier
                "file_path": "/src/auth/handler.py",
                "branch": "main",
                "file_type": "code",
                "library_name": null  # null for project content
            }
        },
        {
            "document_id": "lib456",
            "content": "...",
            "score": 0.85,
            "metadata": {
                "tenant_id": null,  # null for library content
                "library_name": "fastapi",
                "source_file": "security.md",
                "file_type": "doc"
            }
        }
    ],
    "total_results": 42,
    "collections_searched": ["_projects", "_libraries"],
    "query": "authentication",
    "scope": "all"
}
```

---

## Quick Reference

| Scenario | Parameters |
|----------|------------|
| Current project only | `scope="project"` (default) |
| Include libraries | `include_libraries=True` |
| All projects | `scope="all"` |
| Everything searchable | `scope="all", include_libraries=True` |
| Specific branch | `branch="feature/xyz"` |
| All branches | `branch="*"` |
| Code only | `file_type="code"` |
| Tests only | `file_type="test"` |
| Semantic only | `mode="semantic"` |
| Exact matches | `mode="exact"` |

---

**See Also:**
- [API.md](../API.md) - MCP tools API reference
- [CLI.md](../CLI.md) - CLI command reference
- [GRPC_API.md](GRPC_API.md) - gRPC protocol reference
