# workspace-qdrant-admin CLI

Administrative command-line interface for managing workspace-qdrant-mcp collections safely.

## Overview

The `workspace-qdrant-admin` CLI provides safe administrative operations for Qdrant collections, separate from the MCP server. It includes built-in safety features like project scoping, protected collection identification, and confirmation prompts.

## Installation

The CLI is automatically installed when you install the `workspace-qdrant-mcp` package:

```bash
# Install the package
pip install -e .

# The admin CLI is now available
workspace-qdrant-admin --help
```

## Safety Features

### Project Scoping
- Only operates on collections within the current project scope
- Identifies main project and subproject collections
- Respects global collections configured in your settings

### Protected Collections
- Automatically identifies and protects memexd daemon collections (ending with `-code`)
- Prevents accidental deletion of critical system collections

### Confirmation Prompts
- Interactive confirmation before destructive operations
- Shows collection details before deletion
- Can be bypassed with `--force` flag for automation

### Dry-Run Mode
- Test operations without making changes
- Preview what would be deleted
- Safe way to verify commands before execution

## Commands

### list-collections

List collections in the Qdrant instance.

```bash
# List workspace collections only
workspace-qdrant-admin list-collections

# List all collections
workspace-qdrant-admin list-collections --all

# Show detailed information
workspace-qdrant-admin list-collections --verbose
```

**Options:**
- `--all` / `-a`: Show all collections, not just workspace collections
- `--verbose` / `-v`: Show verbose output with collection details

**Example output:**
```
Found 3 collections:
[P] [*] claude-code-cfg-scratchbook (50 points)
[P]     claude-code-cfg-docs (125 points)  
[G]     global-references (200 points)
```

Icons:
- [P] = Project-scoped collection
- [G] = Global collection
- [*] = Protected collection

### delete-collection

Delete a collection with safety checks.

```bash
# Delete with confirmation prompt
workspace-qdrant-admin delete-collection my-project-docs

# Delete without confirmation (use with caution)
workspace-qdrant-admin delete-collection my-project-docs --force

# Preview what would be deleted (safe)
workspace-qdrant-admin delete-collection my-project-docs --dry-run
```

**Arguments:**
- `collection_name`: Name of collection to delete (required)

**Options:**
- `--force` / `-f`: Skip confirmation prompts
- `--dry-run`: Show what would be deleted without actually deleting

**Safety Checks:**
1. Collection must exist
2. Collection must not be protected (no `-code` suffix)
3. Collection must be within current project scope
4. User confirmation required (unless `--force` or `--dry-run`)

**Example interaction:**
```bash
$ workspace-qdrant-admin delete-collection test-project-scratchbook

Collection: test-project-scratchbook
Points: 45
Project scoped: Yes
Protected: No

Delete collection 'test-project-scratchbook' with 45 points? [y/N]: y
Collection deleted successfully
```

### collection-info

Show detailed information about collections.

```bash
# Show info for specific collection
workspace-qdrant-admin collection-info my-collection

# Show info for all workspace collections
workspace-qdrant-admin collection-info
```

**Arguments:**
- `collection_name`: Specific collection name (optional)

**Example output for single collection:**
```
Collection: claude-code-cfg-scratchbook
Points: 125
Vectors: 125
Status: green
Project scoped: Yes
Protected: No
Vector size: 384
Distance metric: Cosine
```

**Example output for all collections:**
```
Workspace Collections (3 total):
• claude-code-cfg-scratchbook: 125 points (green)
• claude-code-cfg-docs: 67 points (green)
• global-references: 200 points (green)
```

## Global Options

Available for all commands:

- `--config`: Path to custom config file
- `--verbose` / `-v`: Enable verbose logging

## Configuration

The CLI uses the same configuration as the main MCP server. It reads from:

1. Environment variables
2. `.env` file in current directory
3. Config file specified with `--config`

### Required Configuration

```bash
# .env file
QDRANT_URL=http://localhost:6333
```

### Optional Configuration

```bash
# GitHub user for project detection
GITHUB_USER=your-username

# Global collections (comma-separated)
GLOBAL_COLLECTIONS=references,standards,docs
```

## Project Detection

The CLI automatically detects your project context:

- **Git repositories**: Uses repository name and detects submodules/subprojects
- **Non-git directories**: Uses directory name
- **GitHub integration**: Enhanced detection with `GITHUB_USER` setting

## Error Handling

The CLI provides clear error messages for common issues:

### Collection Not Found
```
Cannot delete collection: Collection 'nonexistent' does not exist
```

### Protected Collection
```
Cannot delete collection: Collection 'memexd-main-code' is protected (memexd daemon collection)
```

### Out of Project Scope
```
Cannot delete collection: Collection 'other-project-docs' is outside current project scope
```

### Connection Issues
```
Error listing collections: HTTPConnectionPool(host='localhost', port=6333): 
Connection refused. Is Qdrant running?
```

## Automation and Scripting

### Batch Operations

Delete multiple collections:
```bash
# Get list of collections to delete
collections=$(workspace-qdrant-admin list-collections | grep "old-project" | cut -d' ' -f2)

# Delete each with force flag
for collection in $collections; do
    workspace-qdrant-admin delete-collection "$collection" --force
done
```

### CI/CD Integration

Use dry-run mode for validation:
```bash
# Validate collections exist before deployment
workspace-qdrant-admin collection-info required-collection
if [ $? -ne 0 ]; then
    echo "Required collection missing!"
    exit 1
fi
```

### JSON Output (Future Enhancement)

Planned feature for machine-readable output:
```bash
workspace-qdrant-admin list-collections --format json
workspace-qdrant-admin collection-info --format json
```

## Security Considerations

### Network Security
- CLI connects directly to Qdrant instance
- Supports API keys and authentication
- Use secure connections in production

### Access Control
- Project scoping prevents cross-project interference
- Protected collections cannot be deleted
- Confirmation prompts prevent accidental operations

### Audit Trail
- All operations are logged with timestamps
- Use verbose mode for detailed operation logs
- Consider centralized logging for production use

## Troubleshooting

### Common Issues

**CLI not found after installation:**
```bash
# Reinstall in editable mode
pip install -e .
# Or check if it's in your PATH
which workspace-qdrant-admin
```

**Cannot connect to Qdrant:**
```bash
# Check if Qdrant is running
curl http://localhost:6333
# Verify configuration
workspace-qdrant-validate
```

**Permission denied errors:**
```bash
# Check file permissions
ls -la ~/.env
# Ensure Qdrant is accessible
telnet localhost 6333
```

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
workspace-qdrant-admin --verbose list-collections
```

## Migration from Manual Operations

If you were previously deleting collections manually:

### Old Manual Method
```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
client.delete_collection("collection-name")  # No safety checks!
```

### New Safe Method
```bash
# Interactive with safety checks
workspace-qdrant-admin delete-collection collection-name

# Or automated with explicit confirmation
workspace-qdrant-admin delete-collection collection-name --force
```

## Contributing

To extend the CLI functionality:

1. Add new commands in `src/workspace_qdrant_mcp/utils/admin_cli.py`
2. Follow the existing safety pattern (validate → confirm → execute)
3. Add comprehensive tests in `tests/test_admin_cli.py`
4. Update this documentation

## Changelog

### v0.1.0
- Initial release with basic collection management
- Safety features: project scoping, protected collections, confirmations
- Commands: list-collections, delete-collection, collection-info
- Dry-run mode and force options
- Comprehensive error handling and logging