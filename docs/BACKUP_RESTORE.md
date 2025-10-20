# Backup and Restore Operations

This guide covers backup and restore operations for workspace-qdrant-mcp, including version compatibility requirements, best practices, and troubleshooting.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Version Compatibility](#version-compatibility)
4. [Backup Operations](#backup-operations)
5. [Restore Operations](#restore-operations)
6. [Configuration Options](#configuration-options)
7. [Version Migration Framework](#version-migration-framework)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

The backup and restore system provides data protection and disaster recovery capabilities for workspace-qdrant-mcp. It supports:

- Complete system state backups (Qdrant collections, SQLite state, configuration)
- Partial backups (selected collections)
- Version compatibility validation
- Configurable retention policies
- Optional compression
- Integrity verification
- Version migration framework (future)

### Key Features

- **Semantic Versioning**: Uses semantic versioning (MAJOR.MINOR.PATCH) for compatibility checks
- **Safe Defaults**: Prevents data corruption with strict version checking
- **Flexible Options**: Configurable validation levels for different scenarios
- **Clear Errors**: Detailed, actionable error messages guide resolution
- **CLI Integration**: Full command-line interface for all operations

## Quick Start

### Creating a Backup

```bash
# Basic backup (auto-detects collections)
wqm backup create /path/to/backup

# Backup with description
wqm backup create /path/to/backup --description "Pre-migration backup"

# Backup specific collections
wqm backup create /path/to/backup --collections "collection1,collection2"

# Force overwrite existing backup
wqm backup create /path/to/backup --force
```

### Restoring from Backup

```bash
# Restore from backup (with confirmation prompt)
wqm backup restore /path/to/backup

# Restore without confirmation
wqm backup restore /path/to/backup --force

# Dry-run to see what would be restored
wqm backup restore /path/to/backup --dry-run

# Allow downgrade from newer patch version
wqm backup restore /path/to/backup --allow-downgrade

# Verbose output
wqm backup restore /path/to/backup --verbose
```

### Backup Information

```bash
# View backup details
wqm backup info /path/to/backup

# JSON output
wqm backup info /path/to/backup --json

# List all backups in directory
wqm backup list /path/to/backups-dir

# List with sorting
wqm backup list /path/to/backups-dir --sort timestamp

# Validate backup structure
wqm backup validate /path/to/backup

# Validate with file checking
wqm backup validate /path/to/backup --check-files
```

## Version Compatibility

### Semantic Versioning

workspace-qdrant-mcp uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes, incompatible API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

### Compatibility Rules

| Backup Version | Current Version | Compatibility | Requires Flag |
|---------------|----------------|---------------|---------------|
| 0.2.1         | 0.2.1          | ✅ COMPATIBLE | No            |
| 0.2.0         | 0.2.1          | ✅ COMPATIBLE | No            |
| 0.2.2         | 0.2.1          | ⚠️ DOWNGRADE  | --allow-downgrade |
| 0.3.0         | 0.2.1          | ❌ INCOMPATIBLE | Not allowed   |
| 1.0.0         | 0.2.1          | ❌ INCOMPATIBLE | Not allowed   |

### Compatibility Status Definitions

#### COMPATIBLE
- **Definition**: Versions have same MAJOR.MINOR, backup PATCH ≤ current PATCH
- **Risk**: None
- **Action**: Restore allowed without flags

#### UPGRADE_AVAILABLE
- **Definition**: Same as COMPATIBLE (backup from older patch)
- **Risk**: None
- **Action**: Restore allowed, note about older patch version

#### DOWNGRADE
- **Definition**: Same MAJOR.MINOR, backup PATCH > current PATCH
- **Risk**: Newer patch may have data format changes
- **Action**: Requires `--allow-downgrade` flag
- **Safety**: Create backup of current system first

#### INCOMPATIBLE
- **Definition**: Different MAJOR or MINOR versions
- **Risk**: High - data corruption or loss
- **Action**: Not allowed, must use matching version
- **Alternatives**: Version migration (future) or system upgrade/downgrade

### Development Versions

Development versions (e.g., `0.2.1dev1`) are handled specially:

- Treated as patch version (0.2.1)
- Same compatibility rules apply
- Can be allowed via configuration: `backup.validation.allow_dev_versions = true`

## Backup Operations

### Backup Structure

A backup directory contains:

```
backup/
├── metadata/
│   └── manifest.json          # Backup metadata and version
├── collections/
│   ├── collection1.snapshot   # Qdrant collection snapshots
│   └── collection2.snapshot
└── sqlite/
    └── state.db              # SQLite state database
```

### Creating Backups

#### Complete System Backup

```bash
wqm backup create /backups/full-backup-$(date +%Y%m%d)
```

Creates backup of:
- All Qdrant collections
- SQLite state database
- System configuration metadata

#### Partial Backup

```bash
# Backup specific collections
wqm backup create /backups/partial --collections "project-docs,user-notes"
```

Partial backups:
- Only specified collections
- Marked as `partial_backup: true` in manifest
- Restore only restores included collections

#### Backup with Metadata

```bash
wqm backup create /backups/migration \
  --description "Pre-migration to v0.3.0" \
  --force
```

Metadata includes:
- Description
- Timestamp
- Version information
- Collection statistics
- System information (Python, Qdrant versions)

### Backup Validation

Validate backup structure and integrity:

```bash
# Basic structure validation
wqm backup validate /backups/my-backup

# Deep validation with file checks
wqm backup validate /backups/my-backup --check-files --verbose
```

Validation checks:
- Directory structure exists
- manifest.json is valid
- Required subdirectories present
- Version format is valid
- Collection snapshots exist (with --check-files)
- Checksums verify (if enabled)

## Restore Operations

### Before Restoring

**IMPORTANT**: Always create a safety backup before restore:

```bash
# Create safety backup
wqm backup create /backups/safety-$(date +%Y%m%d%H%M%S)

# Then restore
wqm backup restore /backups/target-backup
```

### Restore Workflow

1. **Validation Phase**
   - Verify backup structure
   - Check version compatibility
   - Validate manifest

2. **Confirmation Phase**
   - Display restore plan
   - Show version information
   - Prompt for confirmation (unless --force)

3. **Restore Phase**
   - Stop daemon (if running)
   - Restore collections
   - Restore state database
   - Start daemon

4. **Verification Phase**
   - Verify collections exist
   - Check point counts
   - Validate state

### Dry-Run Mode

Preview restore without making changes:

```bash
wqm backup restore /backups/my-backup --dry-run
```

Dry-run shows:
- Backup information
- Version compatibility
- Collections to be restored
- Estimated restore time
- Potential issues

### Restoring with Version Mismatch

#### Patch Version Downgrade

```bash
# Backup: 0.2.2, Current: 0.2.1
wqm backup restore /backups/v0.2.2 --allow-downgrade
```

**Safety Steps:**
1. Create current system backup first
2. Review release notes for changes in 0.2.2
3. Use `--dry-run` first
4. Monitor restore carefully

#### Major/Minor Version Incompatibility

```
# Backup: 0.3.0, Current: 0.2.1
wqm backup restore /backups/v0.3.0
```

**Error:** Incompatible versions (cannot restore)

**Resolution Options:**
1. Find backup from 0.2.x series
2. Upgrade system to 0.3.0 first
3. Wait for version migration framework (future)

### Partial Restore

Restoring partial backups:

```bash
wqm backup restore /backups/partial-backup
```

**Behavior:**
- Only restores collections in backup
- Other collections unchanged
- State database merged (not replaced)
- Manifest indicates partial restore

## Configuration Options

Configure backup/restore behavior in `.config/workspace-qdrant/config.yaml`:

### Backup Settings

```yaml
backup:
  # Master control
  enabled: true

  # Auto-backup before restore
  auto_backup_before_restore: true

  # Default backup location
  default_backup_directory: "/var/backups/wqm"

  # Retention policy (days)
  retention_days: 30

  # Compression
  compression: true
```

### Version Validation

```yaml
backup:
  validation:
    # Strict version checking
    strict_version_check: true

    # Allow development versions
    allow_dev_versions: false

    # Allow patch downgrades
    allow_patch_downgrade: false

    # Allow minor downgrades (dangerous!)
    allow_minor_downgrade: false

    # Warning threshold for old backups (days)
    version_warning_threshold: 90

    # Schema compatibility checking
    check_schema_compatibility: true
```

### Verification Settings

```yaml
backup:
  verification:
    # Verify after backup creation
    verify_after_backup: true

    # Verify before restore
    verify_before_restore: true

    # Checksum algorithm
    checksum_algorithm: "xxhash64"  # or "sha256", "md5", "none"
```

### Metadata Settings

```yaml
backup:
  metadata:
    # Include collection statistics
    include_collection_stats: true

    # Include system information
    include_system_info: true

    # Custom metadata
    custom_metadata:
      environment: "production"
      backup_type: "scheduled"
```

## Version Migration Framework

### Overview

The version migration framework (v0.3.0+) provides:
- Automated data transformations between versions
- Migration validation and warnings
- Reversible migrations (where possible)
- Migration path discovery

### Defining Migrations

```python
from common.core.version_migration import (
    BaseMigration,
    BackupData,
    register_migration
)

@register_migration(from_version="0.2.0", to_version="0.3.0")
class MigrateTo030(BaseMigration):
    """Migration from 0.2.0 to 0.3.0."""

    description = "Add schema_version field to collections"
    reversible = False

    def migrate(self, backup_data: BackupData) -> BackupData:
        """Apply migration transformations."""
        # Add schema_version to all collections
        for coll_name, coll_data in backup_data.collections.items():
            if isinstance(coll_data, dict):
                coll_data["schema_version"] = "2.0"

        backup_data.version = "0.3.0"
        return backup_data

    def get_warnings(self, backup_data: BackupData) -> List[str]:
        return ["Schema version field added - verify collection schemas"]
```

### Using Migrations

Migrations are automatically discovered and applied:

```bash
# Migrations applied transparently during restore
wqm backup restore /backups/v0.2.0-backup
# Automatically applies 0.2.0 → 0.3.0 migration if available
```

### Listing Available Migrations

```python
from common.core.version_migration import MigrationManager

manager = MigrationManager()
migrations = manager.list_available_migrations(from_version="0.2.0")

for from_ver, to_ver, name in migrations:
    print(f"{from_ver} → {to_ver}: {name}")
```

## Troubleshooting

### Version Compatibility Errors

#### Error: "Backup version X.Y.Z is incompatible"

**Cause:** Major or minor version mismatch

**Solution:**
```bash
# Option 1: Find compatible backup
wqm backup list /backups | grep "$(wqm --version | cut -d. -f1-2)"

# Option 2: Upgrade/downgrade system
# Upgrade to match backup version
pip install workspace-qdrant-mcp==X.Y.Z

# Option 3: Wait for migration framework
# (Future enhancement)
```

#### Error: "Cannot restore from newer backup version"

**Cause:** Backup patch version is newer than current system

**Solution:**
```bash
# Option 1: Create safety backup first
wqm backup create /backups/safety-$(date +%Y%m%d%H%M%S)

# Then restore with --allow-downgrade
wqm backup restore /backups/newer-backup --allow-downgrade

# Option 2: Upgrade system to match backup
pip install workspace-qdrant-mcp --upgrade
```

### Backup Structure Errors

#### Error: "Invalid backup directory structure"

**Cause:** Missing required directories or manifest

**Solution:**
```bash
# Validate backup structure
wqm backup validate /backups/my-backup --verbose

# Check for required components:
ls -la /backups/my-backup/metadata/manifest.json
ls -la /backups/my-backup/collections/
ls -la /backups/my-backup/sqlite/
```

#### Error: "Backup manifest is missing or invalid"

**Cause:** Corrupted or missing manifest.json

**Solution:**
```bash
# Try to read manifest manually
cat /backups/my-backup/metadata/manifest.json | jq .

# If corrupted, restore from earlier backup
wqm backup list /backups --sort timestamp
```

### Restore Failures

#### Error: "Restore failed: Collection X already exists"

**Cause:** Target collections already exist

**Solution:**
```bash
# Option 1: Backup and delete existing collections
wqm backup create /backups/before-clean
wqm admin collections delete collection-name

# Option 2: Use different collection names (partial restore)
# Edit manifest to change collection names before restore
```

#### Error: "Daemon connection failed during restore"

**Cause:** Daemon not running or not responding

**Solution:**
```bash
# Check daemon status
wqm service status

# Restart daemon
wqm service restart

# Retry restore
wqm backup restore /backups/my-backup --force
```

## Best Practices

### Backup Strategy

1. **Regular Backups**
   ```bash
   # Daily automated backup
   0 2 * * * wqm backup create /backups/daily/backup-$(date +%Y%m%d) --force
   ```

2. **Pre-Upgrade Backups**
   ```bash
   # Before system upgrade
   wqm backup create /backups/pre-upgrade-$(date +%Y%m%d%H%M%S) \
     --description "Before upgrade to v0.3.0"
   ```

3. **Multiple Retention Periods**
   ```bash
   # Daily (7 days), Weekly (4 weeks), Monthly (12 months)
   /backups/
   ├── daily/    # 7-day retention
   ├── weekly/   # 28-day retention
   └── monthly/  # 365-day retention
   ```

### Restore Strategy

1. **Always Test First**
   ```bash
   # Dry-run before actual restore
   wqm backup restore /backups/my-backup --dry-run
   ```

2. **Safety Backup**
   ```bash
   # Create safety backup before restore
   wqm backup create /backups/safety-$(date +%Y%m%d%H%M%S)
   ```

3. **Verify After Restore**
   ```bash
   # Check system status
   wqm admin status

   # Verify collections
   wqm admin collections list

   # Test search functionality
   wqm search project "test query"
   ```

### Version Management

1. **Track System Version**
   ```bash
   # Record version in backup description
   wqm backup create /backups/backup-$(wqm --version) \
     --description "System backup at v$(wqm --version)"
   ```

2. **Compatibility Checks**
   ```bash
   # Always check compatibility before restore
   wqm backup info /backups/my-backup
   ```

3. **Upgrade Path Planning**
   ```bash
   # Plan upgrades with backup/restore in mind
   # 0.2.0 → 0.2.5 → 0.3.0 (staged upgrades)
   ```

### Monitoring and Maintenance

1. **Regular Validation**
   ```bash
   # Weekly backup validation
   find /backups -type d -name "backup-*" | \
     xargs -I {} wqm backup validate {}
   ```

2. **Storage Management**
   ```bash
   # Check backup disk usage
   du -sh /backups/*

   # Clean old backups (respecting retention policy)
   find /backups/daily -type d -mtime +7 -exec rm -rf {} \;
   ```

3. **Backup Inventory**
   ```bash
   # Maintain backup catalog
   wqm backup list /backups --json > /backups/catalog.json
   ```

## CLI Reference

### Backup Commands

```bash
# Create backup
wqm backup create <PATH> [OPTIONS]
  --description TEXT        Backup description
  --collections TEXT        Comma-separated collection names
  --force                   Overwrite existing backup
  --verbose                 Show detailed output

# View backup info
wqm backup info <PATH> [OPTIONS]
  --json                    Output as JSON
  --verbose                 Show detailed information

# List backups
wqm backup list <DIRECTORY> [OPTIONS]
  --sort [name|timestamp|size]  Sort order
  --json                    Output as JSON

# Validate backup
wqm backup validate <PATH> [OPTIONS]
  --check-files            Verify file integrity
  --verbose                Show detailed validation
```

### Restore Commands

```bash
# Restore from backup
wqm backup restore <PATH> [OPTIONS]
  --dry-run                Show restore plan without executing
  --force                  Skip confirmation prompt
  --allow-downgrade        Allow restore from newer patch version
  --verbose                Show detailed progress
```

## Additional Resources

- [Architecture Overview](ARCHITECTURE.md)
- [Configuration Guide](CONFIGURATION.md)
- [CLI Reference](CLI.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Version Migration Guide](VERSION_MIGRATION.md) (future)

## Support

For issues or questions:
- GitHub Issues: https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
- Documentation: https://github.com/ChrisGVE/workspace-qdrant-mcp/docs
- Community: [Link to community forum/chat]
