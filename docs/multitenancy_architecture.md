# Workspace-Qdrant-MCP Multi-Tenancy Architecture

**Version:** 1.0
**Date:** 2025-01-02
**Status:** Design Specification

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Collection Architecture](#collection-architecture)
4. [Metadata Schema](#metadata-schema)
5. [Query Patterns](#query-patterns)
6. [Real-Life Scenarios](#real-life-scenarios)
7. [Configuration Changes](#configuration-changes)
8. [Implementation Guidelines](#implementation-guidelines)
9. [Performance Considerations](#performance-considerations)

---

## Overview

This document specifies the multi-tenancy architecture for workspace-qdrant-mcp, defining how projects, libraries, and user collections are organized within Qdrant vector database.

### Design Context

**Qdrant Limitations:**
- Qdrant Cloud limit: 1,000 collections per cluster
- Recommendation: Single large collection with payload filtering
- Reality: ~150 collections projected (well under limit)

**Our Decision:** Collection-per-project architecture
- Balances Qdrant best practices with use case requirements
- Provides hard database-level isolation
- Aligns collection boundaries with conceptual boundaries
- Optimizes for symbol search (primary use case)

---

## Design Principles

### 1. Use Case Driven Architecture

**Project Content:**
- **Primary use case:** Symbol search (definitions, usages, references)
- **Secondary use case:** Structural understanding of codebase
- **Query pattern:** Highly scoped (one project, one branch typically)
- **Optimization:** Collection-per-project for natural scoping

**Libraries:**
- **Primary use case:** Information mining (documentation, papers, manuals)
- **Query pattern:** Semantic search across documents
- **Optimization:** Collection-per-library for isolation

### 2. Hard Isolation Over Soft Isolation

- **Database-level isolation** via collection boundaries
- **Not application-level** via metadata filtering
- Prevents accidental data leakage between projects
- Simplifies query logic (less filtering required)

### 3. Collection Count is Acceptable

- Projected: ~150 collections (50 projects + 100 libraries)
- Qdrant limit: 1,000 collections
- Overhead: Acceptable for clarity and safety

### 4. Single User Context

- Design for single-user projects (no team collaboration)
- Defer multi-user scenarios to future phase
- Simplifies tenant isolation logic

---

## Collection Architecture

### Collection Types

#### 1. Project Collections: `_{project_id}`

One collection per project containing all code and non-code files from that project.

**Collection Naming:**
```
Format: _{project_id}

project_id calculation:
1. If git remote exists: sanitized remote URL
   Example: github.com/user/repo → github_com_user_repo
2. If no remote: SHA256 hash of absolute path (first 16 chars)
   Example: /Users/chris/dev/myapp → path_abc123def456789a

Collection names:
- _github_com_anthropics_claude
- _github_com_user_myrepo
- _path_abc123def456789a
```

**Content:**
- ALL files from the project (code and non-code)
- Follows inclusion/exclusion patterns from configuration
- Multi-branch support via metadata

**Rationale:**
- Symbol search is naturally project-scoped
- Hard isolation: collection boundary = project boundary
- Easy project deletion: drop collection
- Query optimization: smaller collection per query (1K docs vs 10K docs)

#### 2. Library Collections: `_{library_name}`

One collection per library containing documentation, papers, manuals, etc.

**Collection Naming:**
```
Format: _{library_name}

Examples:
- _react
- _python_docs
- _rust_book
- _research_papers
```

**Content:**
- Library documentation files
- Folder structure preserved in metadata (optional filtering)
- No complex multi-tenancy (semantic search handles topic discovery)

**Rationale:**
- Libraries are naturally isolated (React ≠ Python docs)
- Semantic search handles topic discovery without folder-based tenancy
- Rare to search across libraries
- One collection per library is clean and simple

#### 3. User Collections: `{basename}-{type}`

User-defined collections specified in configuration.

**Collection Naming:**
```
Format: {collection_basename}-{type}

Configuration:
  workspace:
    collection_basename: "myapp"
    collection_types: ["notes", "docs", "ideas"]

Results in collections:
- myapp-notes
- myapp-docs
- myapp-ideas
```

**Content:**
- Manual user content (NOT automatic file ingestion)
- User-defined organization
- Optional multi-tenancy via metadata

**Rationale:**
- User-controlled collections for custom workflows
- Separate from automatic project ingestion
- Flexible for various use cases

#### 4. Memory Collection: `memory`

Global collection for user preferences and LLM behavioral rules.

**Collection Naming:**
```
Collection: memory (fixed name from config)
```

**Content:**
- User preferences
- LLM behavioral rules
- Conversational memory
- Cross-project learnings

**Rationale:**
- NOT readonly (previous design was wrong)
- NOT system-reserved (regular r/w collection)
- Global across all projects
- Multi-tenant via metadata if needed

#### 5. Reserved Patterns

**`__*` pattern:** Placeholder for system collections (currently UNUSED)
- We decided to make `memory` r/w instead of creating separate system collections
- Reserved for future use if needed

**Deprecated patterns:**
- ❌ `_projects_content` - NOT USED (was previous mega-collection design)
- ❌ `{project_name}-{type}` per project - WRONG pattern

---

## Metadata Schema

### Project Collection Metadata

Each document in a project collection (`_{project_id}`) contains:

```json
{
  "project_id": "github_com_user_repo",
  "project_name": "user/repo",
  "branch": "main",
  "file_path": "/src/parser.py",
  "file_absolute_path": "/Users/chris/dev/repo/src/parser.py",
  "language": "python",
  "file_type": "code",
  "last_modified": "2025-01-02T10:30:00Z",

  "symbols_defined": ["parse_ast", "Token", "Lexer"],
  "symbols_used": ["ast", "typing", "re"],
  "imports": ["ast", "typing", "re"],
  "exports": ["parse_ast", "Token"],

  "lsp_metadata": {
    "symbols": [...],
    "definitions": [...],
    "references": [...]
  },

  "tree_sitter_ast": {
    "root_node": "module",
    "functions": [...],
    "classes": [...]
  },

  "chunk_index": 0,
  "total_chunks": 5
}
```

**Key fields:**

- **project_id**: Calculated tenant identifier (git remote or path hash)
- **branch**: Git branch name (enables multi-branch support)
- **file_path**: Relative path from project root
- **file_absolute_path**: Full absolute path (for reference, may become stale)
- **symbols_defined**: Symbols defined in this file (for symbol search)
- **symbols_used**: Symbols imported/used (for reference tracking)
- **lsp_metadata**: Rich metadata from LSP server
- **tree_sitter_ast**: Parsed AST from tree-sitter

### Library Collection Metadata

Each document in a library collection (`_{library_name}`) contains:

```json
{
  "library_name": "react",
  "file_path": "/hooks/useState/index.md",
  "folder": "hooks",
  "subfolder": "useState",
  "document_type": "documentation",
  "library_version": "18.0",
  "last_modified": "2025-01-02T10:30:00Z",

  "title": "useState Hook",
  "topics": ["hooks", "state", "reactivity"],

  "chunk_index": 0,
  "total_chunks": 3
}
```

**Key fields:**

- **library_name**: Library identifier
- **file_path**: Path within library
- **folder/subfolder**: Optional for filtering (but semantic search usually sufficient)
- **topics**: Extracted topics for categorization
- **library_version**: Track version if library has versioning

---

## Query Patterns

### Symbol Search (Primary Use Case)

**"Where is `parse_ast` defined?"**

```python
search(
  collection="_github_com_user_repo",  # Project collection (already scoped)
  query_vector=embed("parse_ast definition python"),
  filter={
    "must": [
      {"key": "branch", "match": {"value": "main"}},
      {"key": "symbols_defined", "match": {"any": ["parse_ast"]}}
    ]
  },
  limit=10
)
```

**Query scope:** ~1,000 documents (one project-branch after filtering)

**"Where is `parse_ast` used?"**

```python
search(
  collection="_github_com_user_repo",
  query_vector=embed("parse_ast usage references"),
  filter={
    "must": [
      {"key": "branch", "match": {"value": "main"}},
      {"key": "symbols_used", "match": {"any": ["parse_ast"]}}
    ]
  },
  limit=100  # May have many usage sites
)
```

### Library Search

**"Explain React hooks"**

```python
search(
  collection="_react",  # Library collection
  query_vector=embed("react hooks explanation"),
  limit=20
)
```

**"Search only hooks documentation"** (optional folder filtering)

```python
search(
  collection="_react",
  query_vector=embed("useState hook usage"),
  filter={
    "key": "folder",
    "match": {"value": "hooks"}
  },
  limit=20
)
```

### Cross-Branch Search

**Search across multiple branches:**

```python
search(
  collection="_github_com_user_repo",
  query_vector=embed("authentication implementation"),
  filter={
    "key": "branch",
    "match": {
      "any": ["main", "develop", "feature-auth"]
    }
  },
  limit=50
)
```

### Tenant-Aware Search Wrapper (Safety)

To prevent accidental queries without tenant filtering:

```python
class ProjectAwareSearch:
    """Ensures all queries are properly scoped to project/branch."""

    def __init__(self, project_id: str, branch: str = "main"):
        self.collection = f"_{project_id}"
        self.branch = branch

    def search(self, query: str, additional_filters: dict = None, **kwargs):
        """Always injects project and branch filters."""
        filters = {
            "must": [
                {"key": "branch", "match": {"value": self.branch}}
            ]
        }

        if additional_filters:
            filters["must"].append(additional_filters)

        return qdrant.search(
            collection=self.collection,
            query_vector=embed(query),
            filter=filters,
            **kwargs
        )
```

---

## Real-Life Scenarios

### 1. Local Project → Remote Repository

**Situation:** User initializes git remote on previously local project

```
Before: project_id = "path_abc123def456789a"
After:  project_id = "github_com_user_repo"
```

**Solution: Collection Alias (Zero Downtime)**

```bash
# CLI command
wqm project update-remote \
  --old-id path_abc123def456789a \
  --new-id github_com_user_repo

# Implementation:
1. Create Qdrant collection alias:
   alias "github_com_user_repo" → collection "path_abc123def456789a"

2. Update SQLite state:
   - projects table: update project_id
   - ingestion_queue: update project_id filter
   - watch_folders: update project_id

3. Queries work immediately using new alias

4. Optional background migration:
   - Create new collection "_github_com_user_repo"
   - Copy/migrate data from old collection
   - Update alias to point to new collection
   - Drop old collection
```

**Benefits:**
- Zero downtime
- Queries work during transition
- Optional data migration (alias can stay indefinitely)

### 2. Project Moves on Storage

**Situation:** User moves project to different directory

```
Before: /Users/chris/dev/oldpath/myproject
After:  /Users/chris/dev/newpath/myproject
```

**Solution: Update Watch Configuration (No Qdrant Changes)**

```bash
# CLI command
wqm project move \
  --project github_com_user_repo \
  --from /Users/chris/dev/oldpath/myproject \
  --to /Users/chris/dev/newpath/myproject

# Implementation:
1. Update SQLite:
   - projects table: update project_root
   - watch_folders: update folder paths

2. Qdrant: NO changes needed
   - Collection name based on git remote, not path
   - file_absolute_path metadata becomes stale (acceptable)
   - file_path (relative) remains valid

3. Resume watching new location
```

**Note:** `file_absolute_path` in metadata is for reference only and may become stale. The definitive path is `project_root + file_path`.

### 3. Branch Operations

#### Delete Branch

```bash
wqm branch delete \
  --project github_com_user_repo \
  --branch feature-x

# Implementation:
DELETE FROM collection "_github_com_user_repo"
WHERE branch = 'feature-x'
```

#### Rename Branch

```bash
wqm branch rename \
  --project github_com_user_repo \
  --from old-name \
  --to new-name

# Implementation:
UPDATE collection "_github_com_user_repo"
SET branch = 'new-name'
WHERE branch = 'old-name'
```

#### Merge and Delete

```bash
wqm branch merge \
  --project github_com_user_repo \
  --from feature-x \
  --into main \
  --delete-source

# Implementation:
1. Optional: Keep feature-x documents for history
2. Or: DELETE WHERE branch = 'feature-x'
3. Main branch unchanged (already has merged code)
```

### 4. Dynamic Library Move

**Situation:** Library folder relocates on disk

```bash
wqm library move \
  --library react \
  --from /old/docs/react \
  --to /new/docs/react

# Implementation:
1. Update SQLite watch_folders table
2. Option A: Re-ingest from new location
3. Option B: Update metadata file_absolute_path
```

**Optimization:** Provide move command to prevent full re-ingestion

---

## Configuration Changes

### Current Configuration (Incorrect)

```yaml
# assets/default_configuration.yaml (BEFORE)
workspace:
  collection_basename: null
  collection_types: ["code", "docs", "tests"]

  auto_create_collections: true
  memory_collection_name: "memory"

  auto_ingestion:
    project_collection: "projects_content"  # ← WRONG: implies single collection
```

### New Configuration (Correct)

```yaml
# assets/default_configuration.yaml (AFTER)
workspace:
  # User-defined collections (manual content, not auto-ingestion)
  # Format: {collection_basename}-{type}
  # Example: "myapp-notes", "myapp-docs"
  # Set to null to disable user collections
  collection_basename: null
  collection_types: ["notes", "docs", "ideas"]

  # Memory collection for user preferences and LLM rules
  # Global collection, read/write access
  memory_collection_name: "memory"

  auto_ingestion:
    # Project collection naming: _{project_id}
    # One collection per project (automatically created)
    # project_id derived from git remote or path hash
    # NO LONGER A SINGLE SHARED COLLECTION
    # This setting is REMOVED
    # project_collection: "projects_content"  # ← REMOVED

    # Auto-create project collections on detection
    auto_create_project_collections: true
```

### Configuration Migration

**Changes required:**

1. **Remove** `workspace.auto_ingestion.project_collection` setting
2. **Add** `workspace.auto_ingestion.auto_create_project_collections` setting
3. **Clarify** `collection_basename` is for USER collections, not project content
4. **Document** that project collections are named `_{project_id}` automatically

---

## Implementation Guidelines

### Collection Creation

```python
async def create_project_collection(project_root: Path) -> str:
    """
    Create a new project collection.

    Returns:
        Collection name (e.g., "_github_com_user_repo")
    """
    # Calculate project_id
    project_id = await calculate_tenant_id(project_root)
    collection_name = f"_{project_id}"

    # Check if collection exists
    if await qdrant.collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return collection_name

    # Create collection with proper configuration
    await qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": 384,  # FastEmbed all-MiniLM-L6-v2
            "distance": "Cosine"
        },
        # Sparse vectors for hybrid search
        sparse_vectors_config={
            "text": {
                "modifier": "idf"
            }
        }
    )

    logger.info(f"Created project collection: {collection_name}")
    return collection_name
```

### Tenant ID Calculation

```python
async def calculate_tenant_id(project_root: Path) -> str:
    """
    Calculate consistent tenant ID for a project.

    Priority:
    1. Git remote URL (sanitized) if exists
    2. SHA256 hash of absolute path (first 16 chars)

    Examples:
    - https://github.com/user/repo → github_com_user_repo
    - /Users/chris/dev/myapp → path_abc123def456789a
    """
    try:
        # Try to get git remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            remote_url = result.stdout.strip()
            # Sanitize URL
            sanitized = re.sub(r'^(https?://|git@|ssh://)', '', remote_url)
            sanitized = re.sub(r'[:/\.]+', '_', sanitized)
            sanitized = re.sub(r'_git$', '', sanitized)
            tenant_id = sanitized.lower().strip('_')
            logger.debug(f"Tenant ID from git remote: {tenant_id}")
            return tenant_id

    except Exception as e:
        logger.debug(f"Could not get git remote: {e}")

    # Fallback: hash of absolute path
    path_str = str(project_root.resolve())
    path_hash = hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:16]
    tenant_id = f"path_{path_hash}"
    logger.debug(f"Tenant ID from path hash: {tenant_id}")
    return tenant_id
```

### Document Ingestion

```python
async def ingest_file_to_project(
    file_path: Path,
    project_root: Path,
    branch: str = "main"
):
    """Ingest a file into its project collection."""

    # Calculate collection name
    project_id = await calculate_tenant_id(project_root)
    collection_name = f"_{project_id}"

    # Ensure collection exists
    if not await qdrant.collection_exists(collection_name):
        await create_project_collection(project_root)

    # Process file
    chunks = await process_file(file_path)

    # Generate metadata
    relative_path = file_path.relative_to(project_root)
    metadata = {
        "project_id": project_id,
        "project_name": project_root.name,
        "branch": branch,
        "file_path": str(relative_path),
        "file_absolute_path": str(file_path),
        "language": detect_language(file_path),
        # ... LSP metadata, symbols, etc.
    }

    # Ingest chunks with embeddings
    await ingest_chunks(
        collection=collection_name,
        chunks=chunks,
        metadata=metadata
    )
```

---

## Performance Considerations

### Collection Count vs Query Performance

**Projected scale:**
- 50 projects × 1,000 files/project = 50K documents per project collection
- 100 libraries × 500 docs/library = 50K documents per library collection
- Total: ~150 collections, ~7.5M documents

**Performance profile:**
- Query scope: 1K-50K documents per collection (after filtering)
- HNSW index: O(log N) complexity
- 1K vs 10K documents: marginal difference (one extra HNSW level)
- 150 collections: manageable overhead (well under 1,000 limit)

### Memory Overhead

**Per-collection overhead:**
- HNSW index structures: ~10MB base + ~1KB per point
- Metadata storage: ~500 bytes per document
- Collection metadata: ~1MB

**Total overhead estimate:**
- 150 collections × 10MB base = 1.5GB base
- 7.5M documents × 1.5KB = ~11GB data
- **Total: ~13GB memory** (reasonable for modern systems)

### Benchmarking Plan

**Metrics to track:**
1. Query latency by collection size (1K, 10K, 50K docs)
2. Ingestion throughput (docs/sec) per collection
3. Memory usage vs collection count
4. Cross-branch query performance

**Test scenarios:**
1. Baseline: Single project, single branch (1K docs)
2. Multi-branch: Single project, 10 branches (10K docs)
3. Multi-project: 50 projects, 50K docs each (2.5M total)
4. Stress test: 100 projects, 100K docs each (10M total)

**Success criteria:**
- Query latency < 100ms for p95
- Ingestion throughput > 100 docs/sec
- Memory usage < 20GB for 150 collections

---

## Appendix: Comparison with Alternative Approaches

### Single Mega-Collection Approach (Rejected)

```
Collection: _projects_content (single collection for ALL projects)

Pros:
✅ Follows Qdrant "best practice" recommendation
✅ Most resource-efficient
✅ Single collection to manage

Cons:
❌ Query scope: 10K docs after filtering (all projects → one project)
❌ Application-level isolation only (filter-based)
❌ Must ALWAYS add tenant filter (risk of leakage)
❌ Complex query logic (multiple filter levels)
```

**Why rejected:** Application-level isolation is riskier, and query scope difference (1K vs 10K) is marginal given Qdrant's performance characteristics.

### Collection-per-Branch Approach (Rejected)

```
Collection: _{project_id}_{branch} (one per branch)

Pros:
✅ Smallest query scope (no branch filtering)
✅ Natural branch isolation

Cons:
❌ Collection explosion (every feature branch = new collection)
❌ Cross-branch queries become complex
❌ Violates "bounded collection count" principle
```

**Why rejected:** Unbounded collection count (user can create unlimited branches), and cross-branch queries are important.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-02 | Design Session | Initial specification |

---

**End of Multi-Tenancy Architecture Specification**
