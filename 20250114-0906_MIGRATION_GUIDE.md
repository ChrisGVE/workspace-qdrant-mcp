# Multi-Tenant Collection Architecture Migration Guide

## Overview

This guide documents the migration from individual collection-based architecture to a multi-tenant collection system with metadata-based project isolation. This migration removes deprecated fields and implements a more scalable, efficient collection management system.

## Changes Made

### Deprecated Fields Removed

#### `collection_prefix` (String field)
- **Previous Purpose**: Added a prefix to all collection names for organizational purposes
- **Replacement**: Multi-tenant collections with metadata-based project filtering
- **Migration Impact**: Existing prefixes are no longer needed; project isolation is handled via metadata

#### `max_collections` (Integer field)
- **Previous Purpose**: Limited the total number of collections per workspace
- **Replacement**: Metadata-based tenant isolation with efficient resource management
- **Migration Impact**: No explicit limits needed; multi-tenant architecture scales efficiently

### New Multi-Tenant Architecture

#### Key Concepts

1. **Shared Collections**: Single collections shared across multiple projects
2. **Metadata Filtering**: Projects isolated via `project_id` metadata field
3. **Collection Types**: Configurable types like 'docs', 'notes', 'scratchbook'
4. **Automatic Project Detection**: Git-based project identification

#### Benefits

- **Performance**: Fewer collections means faster queries and reduced overhead
- **Scalability**: Metadata filtering scales better than multiple collections
- **Resource Efficiency**: Reduced memory and storage footprint
- **Simplified Management**: No complex naming schemes or prefix management

## Migration Process

### Automatic Migration

The migration system automatically:

1. **Detects** existing configurations with deprecated fields
2. **Backs up** current configuration files
3. **Removes** deprecated field references from configs
4. **Updates** environment variables to remove deprecated settings
5. **Preserves** all functional configuration options

### Manual Migration Steps

#### 1. Run Migration Analysis

```bash
python 20250114-0906_config_migration.py --analyze
```

This analyzes current usage of deprecated fields and estimates migration complexity.

#### 2. Backup Current Configuration

```bash
python 20250114-0906_config_migration.py --migrate --dry-run
```

Review what changes would be made before proceeding.

#### 3. Perform Migration

```bash
python 20250114-0906_config_migration.py --migrate
```

This performs the actual migration with automatic backup.

#### 4. Update Configuration

Configure the new multi-tenant system:

```yaml
workspace:
  collection_types: ["docs", "notes", "scratchbook"]  # Define your collection types
  global_collections: ["shared", "references"]        # Cross-project collections
  github_user: "your-username"                        # For project detection
  auto_create_collections: true                       # Enable automatic creation
```

### Environment Variables

#### Removed Variables
- `WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX`
- `WORKSPACE_QDRANT_WORKSPACE__MAX_COLLECTIONS`

#### New/Updated Variables
- `WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES` - Define collection types
- `WORKSPACE_QDRANT_WORKSPACE__AUTO_CREATE_COLLECTIONS` - Control automatic creation

## Post-Migration Configuration

### Collection Types Setup

Define the types of collections your workspace will use:

```yaml
workspace:
  collection_types:
    - "docs"        # Documentation and guides
    - "notes"       # General notes and thoughts
    - "scratchbook" # Temporary and experimental content
    - "code"        # Code snippets and examples
```

### Project Detection Configuration

Enable automatic project detection:

```yaml
workspace:
  github_user: "your-github-username"  # Filter to your repositories
  auto_create_collections: true       # Automatically create collections
```

### Global Collections

Configure collections shared across projects:

```yaml
workspace:
  global_collections:
    - "references"   # Shared reference materials
    - "standards"    # Coding standards and guidelines
    - "templates"    # Reusable templates
```

## Data Migration

### Existing Collections

Existing collections will continue to work but should be migrated to the new metadata-based system:

1. **Project-specific collections** → Consolidated into multi-tenant collections with project metadata
2. **Prefixed collections** → Metadata filtering replaces prefix-based organization
3. **Multiple collections per project** → Single collections with type-based metadata

### Collection Naming

#### Before (Prefix-based)
```
myproject_docs
myproject_notes
myproject_scratchbook
otherproject_docs
otherproject_notes
```

#### After (Multi-tenant)
```
docs (with project_id metadata)
notes (with project_id metadata)
scratchbook (with project_id metadata)
```

### Metadata Structure

Each document now includes project identification:

```json
{
  "project_id": "workspace-qdrant-mcp",
  "project_type": "git-repository",
  "collection_type": "docs",
  "content": "Document content...",
  "source_file": "/path/to/file.md"
}
```

## Testing Migration

Run the migration test suite to verify everything works:

```bash
python 20250114-0906_test_migration.py
```

This validates:
- Deprecated fields are properly removed
- Configuration loading works correctly
- YAML export excludes deprecated fields
- Migration script functions properly

## Rollback Procedure

If issues arise, rollback is available:

```bash
python 20250114-0906_config_migration.py --rollback
```

This restores the backed-up configuration files.

## Troubleshooting

### Common Issues

#### "Config validation failed"
- **Cause**: Missing required configuration after migration
- **Solution**: Ensure `collection_types` are properly configured

#### "Auto-ingestion target not found"
- **Cause**: `target_collection_suffix` references removed collection type
- **Solution**: Update to use new collection types or enable auto-creation

#### "Project detection not working"
- **Cause**: Missing `github_user` configuration
- **Solution**: Configure `github_user` for automatic project detection

### Migration Verification

Check that migration completed successfully:

```bash
# Verify no deprecated fields in config
python -c "from common.core.config import Config; c=Config(); print('✅ Migration OK' if not hasattr(c.workspace, 'collection_prefix') else '❌ Migration incomplete')"

# Test configuration loading
python -c "from common.core.config import Config; Config().validate_config(); print('✅ Config valid')"
```

## Support

### Getting Help

1. **Check Migration Log**: Review `20250114-0906_migration.log` for detailed migration steps
2. **Validate Configuration**: Use built-in validation to identify issues
3. **Review Documentation**: Check updated configuration documentation
4. **Test Environment**: Use migration test script to verify functionality

### Reporting Issues

If you encounter problems:

1. **Include Migration Log**: Attach the migration log file
2. **Configuration Details**: Share sanitized configuration (remove API keys)
3. **Error Messages**: Include full error messages and stack traces
4. **Environment Info**: Operating system, Python version, package versions

## Next Steps

After successful migration:

1. **Update Documentation**: Update any project-specific documentation referencing old collection naming
2. **Review Automation**: Update scripts that reference deprecated fields
3. **Optimize Configuration**: Review collection types and global collections for your workflow
4. **Monitor Performance**: Verify improved performance with multi-tenant architecture

The multi-tenant architecture provides a more scalable, efficient foundation for document management while maintaining full backward compatibility for existing functionality.