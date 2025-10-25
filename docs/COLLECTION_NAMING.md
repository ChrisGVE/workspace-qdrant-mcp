# Collection Naming and Basename Requirements

Comprehensive guide to collection naming conventions, type classification, and basename requirements for workspace-qdrant-mcp.

## Overview

workspace-qdrant-mcp uses a strict collection naming system with **four collection types**, each with specific naming patterns and basename requirements. All collections MUST have non-empty basenames to ensure proper Rust daemon validation and metadata enrichment.

## Quick Reference

| Type | Naming Pattern | Basename | Example | Use Case |
|------|----------------|----------|---------|----------|
| **PROJECT** | `_{project_id}` | `"code"` | `_a1b2c3d4e5f6` | Auto-created for file watching |
| **USER** | `{basename}-{type}` | `"notes"` (default) | `myapp-notes` | User-created collections |
| **LIBRARY** | `_{library_name}` | `"lib"` | `_numpy` | External library documentation |
| **MEMORY** | `_memory`, `_agent_memory` | `"memory"` | `_memory` | Meta-level agent data |

## Collection Types

### 1. PROJECT Collections

**Purpose**: Auto-created collections for project file watching and code ingestion.

**Naming Pattern**: `_{project_id}`
- `project_id`: 12-character hexadecimal hash
- Generated via `calculate_tenant_id(project_path)`
- Exactly 13 characters total (underscore + 12-char hash)

**Basename**: `"code"` (fixed, required for daemon validation)

**Examples**:
```python
# Project at /Users/chris/workspace-qdrant-mcp
project_id = calculate_tenant_id("/Users/chris/workspace-qdrant-mcp")
# → "a1b2c3d4e5f6" (12 chars)

collection_name = f"_{project_id}"
# → "_a1b2c3d4e5f6" (13 chars total)

basename = "code"  # Required for Rust daemon validation
```

**Creation**:
- Automatically created by Rust daemon when watching project directories
- Manual creation via `manage(action="init_project")`
- Single collection per project (no type suffixes)

**Content Types**:
All content types stored in single collection, differentiated by metadata:
- Code files: `file_type="code"`
- Documentation: `file_type="docs"`
- Tests: `file_type="test"`
- Configuration: `file_type="config"`

**Metadata Requirements**:
```python
{
    "project_id": "a1b2c3d4e5f6",  # 12-char hash
    "file_type": "code|docs|test|config",
    "branch": "main",  # Git branch
    "file_path": "/path/to/file.py",
    "basename": "code"  # Required
}
```

**Validation Rules**:
- ✅ Name starts with single underscore `_`
- ✅ Followed by exactly 12 hexadecimal characters
- ✅ Total length exactly 13 characters
- ❌ Cannot have type suffixes (e.g., `_a1b2c3d4e5f6-docs`)
- ✅ Basename must be `"code"`

### 2. USER Collections

**Purpose**: User-created collections for notes, scratchbooks, and custom content.

**Naming Pattern**: `{basename}-{type}`
- `basename`: Descriptive project/app identifier (e.g., "myapp", "work", "personal")
- `type`: Content category (e.g., "notes", "scratchbook", "bookmarks")
- Separator: hyphen `-`

**Basename**: `"notes"` (default, can be customized)

**Examples**:
```python
# Development notes for myapp project
collection_name = "myapp-notes"
basename = "notes"

# Work scratchbook
collection_name = "work-scratchbook"
basename = "notes"

# Research bookmarks
collection_name = "research-bookmarks"
basename = "notes"

# Personal code snippets
collection_name = "personal-snippets"
basename = "notes"
```

**Creation**:
```python
# MCP Server - auto-enriched with project_id
store(content="my note", collection="myapp-notes")

# CLI - explicit project specification required
wqm add --collection myapp-notes --project /path/to/project "my note"
```

**Auto-Enrichment**:
- **MCP Server**: Daemon automatically adds `project_id` from current project context
- **CLI**: User must explicitly specify `--project` flag
- Enables filtering: `search(query="notes", project=current_project)`

**Metadata Requirements**:
```python
{
    "content": "Note text",
    "source": "scratchbook",
    "project_id": "a1b2c3d4e5f6",  # Auto-added by daemon (MCP only)
    "basename": "notes"  # Required
}
```

**Validation Rules**:
- ✅ Name contains hyphen separator `-`
- ✅ Does NOT start with underscore `_`
- ✅ Basename must be non-empty string
- ✅ Type must be non-empty string
- ❌ Cannot be reserved names (`_memory`, `_agent_memory`)

### 3. LIBRARY Collections

**Purpose**: External library and framework documentation collections.

**Naming Pattern**: `_{library_name}`
- `library_name`: Library identifier (e.g., "numpy", "fastapi", "react")
- Length: More than 13 characters (distinguishes from PROJECT collections)
- Single underscore prefix

**Basename**: `"lib"` (fixed, required for daemon validation)

**Examples**:
```python
# Python libraries
collection_name = "_numpy"
basename = "lib"

collection_name = "_pandas"
basename = "lib"

collection_name = "_fastapi"
basename = "lib"

# JavaScript libraries
collection_name = "_react"
basename = "lib"

collection_name = "_next_js"
basename = "lib"
```

**Creation**:
```bash
# CLI - add library documentation
wqm library add numpy --source /path/to/numpy/docs

# Stores in _numpy collection with basename="lib"
```

**Metadata Requirements**:
```python
{
    "library_name": "numpy",
    "version": "1.24.0",
    "language": "python",
    "doc_type": "api_reference",
    "basename": "lib"  # Required
}
```

**Validation Rules**:
- ✅ Name starts with single underscore `_`
- ✅ Length is NOT exactly 13 characters (distinguishes from PROJECT)
- ✅ Typically recognizable library names
- ✅ Basename must be `"lib"`

**Disambiguation from PROJECT**:
```python
def get_collection_type(collection_name: str) -> str:
    if collection_name.startswith("_"):
        if len(collection_name) == 13:  # _{12-char-hash}
            return "project"
        else:
            return "library"
```

### 4. MEMORY Collections

**Purpose**: Meta-level data for agent memory and context management.

**Naming Pattern**: Fixed names only
- `_memory`: User memory and context
- `_agent_memory`: Agent-specific memory storage

**Basename**: `"memory"` (fixed, required for daemon validation)

**Examples**:
```python
# User memory collection
collection_name = "_memory"
basename = "memory"

# Agent memory collection
collection_name = "_agent_memory"
basename = "memory"
```

**Special Characteristics**:
- **Exception to daemon-only writes**: Direct writes allowed (meta-level data)
- Fixed collection names (no variations permitted)
- Global scope (not project-specific)

**Metadata Requirements**:
```python
{
    "memory_type": "conversation|preference|context",
    "created_at": "2025-10-25T10:00:00Z",
    "basename": "memory"  # Required
}
```

**Validation Rules**:
- ✅ Name must be exactly `_memory` or `_agent_memory`
- ✅ Basename must be `"memory"`
- ❌ No other memory collection names allowed

## Basename Requirements

### BASENAME_MAP

Defined in `src/python/workspace_qdrant_mcp/server.py`:

```python
# Collection basename mapping for Rust daemon validation
# Maps collection types to valid basenames (non-empty strings)
BASENAME_MAP = {
    "project": "code",      # PROJECT collections: _{project_id}
    "user": "notes",        # USER collections: {basename}-{type}
    "library": "lib",       # LIBRARY collections: _{library_name}
    "memory": "memory",     # MEMORY collections: _memory, _agent_memory
}
```

### Why Basenames are Required

1. **Rust Daemon Validation**: Daemon validates collection metadata before processing
2. **Protocol Compliance**: Non-empty basenames prevent gRPC protocol errors
3. **Metadata Enrichment**: Daemon uses basename to route and enrich content
4. **Type Detection**: Basename helps identify collection purpose

### Basename Validation Rules

**All basenames MUST**:
- Be non-empty strings
- Match expected value for collection type
- Be included in all daemon write operations
- Remain consistent across collection lifecycle

**Validation Failure Scenarios**:
```python
# ❌ Empty basename - protocol error
basename = ""
# Error: "basename cannot be empty string"

# ❌ Missing basename - validation failure
metadata = {"content": "test"}
# Error: "basename field required for daemon routing"

# ❌ Wrong basename for type - type mismatch
collection_type = "project"
basename = "notes"  # Should be "code"
# Error: "basename 'notes' invalid for PROJECT collection"
```

## Collection Naming Validation

### Complete Validation Rules

**Length**:
- Collection name: 3-255 characters
- Minimum 3 chars (prevents collisions and typos)
- Maximum 255 chars (Qdrant limit)

**Character Set**:
- Alphanumeric: `a-z`, `A-Z`, `0-9`
- Allowed special chars: underscore `_`, hyphen `-`
- Case-sensitive

**Pattern Constraints**:
- Cannot start with a number
- Cannot be empty string
- Must match one of the four type patterns

### Validation Examples

**Valid Collection Names**:
```python
# PROJECT collections
"_a1b2c3d4e5f6"  # ✅ 12-char hash
"_f9e8d7c6b5a4"  # ✅ Different project

# USER collections
"myapp-notes"         # ✅ Standard pattern
"work-scratchbook"    # ✅ Descriptive
"personal-bookmarks"  # ✅ Clear purpose

# LIBRARY collections
"_numpy"       # ✅ Python library
"_react"       # ✅ JS framework
"_fastapi"     # ✅ Web framework

# MEMORY collections
"_memory"        # ✅ User memory
"_agent_memory"  # ✅ Agent memory
```

**Invalid Collection Names**:
```python
# Too short
"_a"           # ❌ Only 2 chars
"ab"           # ❌ Only 2 chars

# Wrong patterns
"_a1b2c3d4e5"  # ❌ PROJECT but only 11 chars (needs 12)
"myapp_notes"  # ❌ USER but underscore instead of hyphen
"__system"     # ❌ Double underscore (not a valid type)

# Invalid characters
"my app-notes" # ❌ Contains space
"my@app-notes" # ❌ Contains @
"my.app-notes" # ❌ Contains .

# Starts with number
"9myapp-notes" # ❌ Cannot start with number

# Empty
""             # ❌ Empty string
```

## Usage Examples

### PROJECT Collection Operations

```python
# Get project collection name
from common.utils.project_detection import calculate_tenant_id
from workspace_qdrant_mcp.server import build_project_collection_name

project_path = "/Users/chris/workspace-qdrant-mcp"
project_id = calculate_tenant_id(project_path)  # "a1b2c3d4e5f6"
collection_name = build_project_collection_name(project_id)  # "_a1b2c3d4e5f6"

# Store code with proper basename
store(
    content="def hello(): pass",
    file_path="main.py",
    # Daemon auto-sets basename="code"
)
```

### USER Collection Operations

```python
# Create user collection via MCP
store(
    content="Important architecture decision...",
    source="scratchbook",
    collection="myapp-notes"  # Auto-enriched with project_id
    # Daemon sets basename="notes"
)

# Search user collection
search(
    query="architecture decision",
    collection="myapp-notes",
    project=current_project  # Filter by project
)
```

### LIBRARY Collection Operations

```bash
# Add library documentation via CLI
wqm library add numpy --source /path/to/numpy/docs
# Creates _numpy collection with basename="lib"

# Search library docs
wqm search "array operations" --collection _numpy
```

### MEMORY Collection Operations

```python
# Store to memory collection
store(
    content="User prefers verbose logging",
    source="memory",
    collection="_memory"
    # Daemon sets basename="memory"
)
```

## Implementation Reference

### Type Detection Function

From `src/python/workspace_qdrant_mcp/server.py`:

```python
def get_collection_type(collection_name: str) -> str:
    """Determine collection type from collection name.

    Args:
        collection_name: Collection name to analyze

    Returns:
        One of: "project", "user", "library", "memory"
    """
    if collection_name in ("_memory", "_agent_memory"):
        return "memory"
    elif collection_name.startswith("_"):
        # Could be project or library - check for library patterns
        # Libraries typically have recognizable names (e.g., _numpy, _pandas)
        # Projects are hex hashes (e.g., _a1b2c3d4e5f6)
        if len(collection_name) == 13:  # _{12-char-hash}
            return "project"
        else:
            return "library"
    else:
        # No underscore prefix = user collection
        return "user"
```

### Project ID Calculation

```python
from common.utils.project_detection import calculate_tenant_id

# Calculate project_id from path
project_id = calculate_tenant_id("/Users/chris/workspace-qdrant-mcp")
# Returns: 12-character hexadecimal hash (e.g., "a1b2c3d4e5f6")

# Build collection name
collection_name = f"_{project_id}"
# Returns: "_a1b2c3d4e5f6"
```

### Basename Assignment

```python
# Get collection type
collection_type = get_collection_type(target_collection)

# Get appropriate basename
collection_basename = BASENAME_MAP[collection_type]

# Basename is included in all daemon write operations
daemon_client.ingest_text(
    content=content,
    collection=target_collection,
    basename=collection_basename,  # Required
    tenant_id=tenant_id,
    metadata=metadata
)
```

## Common Patterns

### Multi-Tenant Project Isolation

```python
# Each project gets its own collection
project_a = "_a1b2c3d4e5f6"  # Project A
project_b = "_f9e8d7c6b5a4"  # Project B

# Search only Project A
search(query="authentication", collection=project_a)

# Search only Project B
search(query="authentication", collection=project_b)
```

### Cross-Project USER Collections

```python
# USER collections span projects
"myapp-notes" in project_a  # project_id=a1b2c3d4e5f6 in metadata
"myapp-notes" in project_b  # project_id=f9e8d7c6b5a4 in metadata

# Search all notes across projects
search(query="notes", collection="myapp-notes")

# Search notes for specific project
search(query="notes", collection="myapp-notes", project="a1b2c3d4e5f6")
```

### Library Documentation Sharing

```python
# LIBRARY collections are global (shared across projects)
"_numpy" collection  # Accessible from any project
"_react" collection  # Accessible from any project

# Search across all libraries
search(query="array slice", collection="_numpy")
```

## Best Practices

### 1. Choose the Right Collection Type

**Use PROJECT collections for**:
- Source code files
- Project documentation
- Configuration files
- Any file-watched content

**Use USER collections for**:
- Development notes and journals
- Scratchbooks and TODOs
- Bookmarks and references
- Cross-project snippets

**Use LIBRARY collections for**:
- External library documentation
- Framework references
- Language standard libraries
- Third-party API docs

**Use MEMORY collections for**:
- Agent conversation history
- User preferences and context
- System-level memory

### 2. Follow Naming Conventions

**PROJECT**: Always use generated project_id
```python
# ✅ Correct
collection_name = f"_{calculate_tenant_id(project_path)}"

# ❌ Wrong
collection_name = "_myproject"  # Not a valid hash
```

**USER**: Use descriptive basename-type pattern
```python
# ✅ Correct
collection_name = "myapp-notes"
collection_name = "work-scratchbook"

# ❌ Wrong
collection_name = "myapp_notes"  # Underscore instead of hyphen
collection_name = "notes"        # Missing basename
```

**LIBRARY**: Use clear library identifiers
```python
# ✅ Correct
collection_name = "_numpy"
collection_name = "_fastapi"

# ❌ Wrong
collection_name = "_np"          # Too abbreviated
collection_name = "_a1b2c3d4e5f6"  # Looks like PROJECT
```

### 3. Always Provide Basenames

```python
# ✅ Correct - basename specified
daemon_client.ingest_text(
    content=content,
    collection=collection,
    basename=BASENAME_MAP[get_collection_type(collection)],
    metadata=metadata
)

# ❌ Wrong - missing basename
daemon_client.ingest_text(
    content=content,
    collection=collection,
    metadata=metadata  # Protocol error!
)
```

### 4. Validate Before Creation

```python
def validate_collection_name(name: str) -> tuple[bool, str]:
    """Validate collection name against rules."""
    # Length check
    if not (3 <= len(name) <= 255):
        return False, "Name must be 3-255 characters"

    # Pattern check
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
        return False, "Invalid characters or starts with number"

    # Type-specific validation
    collection_type = get_collection_type(name)
    if collection_type == "project" and len(name) != 13:
        return False, "PROJECT collections must be exactly 13 chars"

    return True, "Valid"
```

## Troubleshooting

### Common Errors

**Error**: `"basename cannot be empty string"`
```python
# Problem
basename = ""

# Solution
collection_type = get_collection_type(collection_name)
basename = BASENAME_MAP[collection_type]
```

**Error**: `"Collection name length invalid"`
```python
# Problem
collection_name = "_a"  # Only 2 chars

# Solution
# For PROJECT: Use full 12-char hash
collection_name = "_a1b2c3d4e5f6"  # 13 chars total
```

**Error**: `"basename 'notes' invalid for PROJECT collection"`
```python
# Problem
collection_type = "project"
basename = "notes"  # Wrong basename

# Solution
basename = BASENAME_MAP["project"]  # "code"
```

### Debugging Collection Type Detection

```python
def debug_collection_type(collection_name: str):
    """Debug helper for type detection."""
    print(f"Collection name: {collection_name}")
    print(f"Length: {len(collection_name)}")
    print(f"Starts with '_': {collection_name.startswith('_')}")

    collection_type = get_collection_type(collection_name)
    print(f"Detected type: {collection_type}")
    print(f"Expected basename: {BASENAME_MAP[collection_type]}")

    # Type-specific checks
    if collection_type == "project":
        print(f"Valid PROJECT: {len(collection_name) == 13}")
    elif collection_type == "user":
        print(f"Contains hyphen: {'-' in collection_name}")
    elif collection_type == "library":
        print(f"Length != 13: {len(collection_name) != 13}")

# Example usage
debug_collection_type("_a1b2c3d4e5f6")
# Collection name: _a1b2c3d4e5f6
# Length: 13
# Starts with '_': True
# Detected type: project
# Expected basename: code
# Valid PROJECT: True
```

## Migration Notes

### Task 385 Changes

**Before Task 385** (empty basenames allowed):
```python
# Old code - no basename validation
daemon_client.ingest_text(
    content=content,
    collection=collection,
    metadata=metadata
)
```

**After Task 385** (basenames required):
```python
# New code - basename required
collection_type = get_collection_type(collection)
basename = BASENAME_MAP[collection_type]

daemon_client.ingest_text(
    content=content,
    collection=collection,
    basename=basename,  # Now required
    metadata=metadata
)
```

**Impact**: All code paths updated to include basenames, preventing protocol validation failures.

## See Also

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview
- **[API.md](../API.md)** - MCP tools documentation
- **[metadata-schema.md](metadata-schema.md)** - Complete metadata reference
- **[multitenancy_architecture.md](multitenancy_architecture.md)** - Multi-tenant design
- **Task 385**: Implementation of BASENAME_MAP and validation
- **Task 374.6**: Single collection per project architecture

---

**Version**: 0.3.0
**Last Updated**: 2025-10-25
**Status**: Stable
**Related Tasks**: 385 (Basename Requirements), 374.6 (Single Collection Architecture)
