#!/usr/bin/env python3
"""
Configuration Migration Script for WorkspaceConfig

This temporary script implements migration logic to remove deprecated fields
(collection_prefix and max_collections) from WorkspaceConfig and migrate
existing configurations to the new multi-tenant collection architecture.

Key migration steps:
1. Detect existing configurations with deprecated fields
2. Create migration mapping for existing collections
3. Update configuration files to remove deprecated fields
4. Ensure backward compatibility during transition
5. Provide rollback capability if needed

Usage:
    python 20250114-0906_config_migration.py --analyze     # Analyze current usage
    python 20250114-0906_config_migration.py --migrate     # Perform migration
    python 20250114-0906_config_migration.py --rollback    # Rollback if needed
"""

import argparse
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from datetime import datetime

# Import the current config system
import sys
sys.path.append('src/python')

from common.core.config import Config, WorkspaceConfig
from common.utils.project_detection import ProjectDetector


class ConfigMigration:
    """Handles migration of deprecated WorkspaceConfig fields."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.backup_dir = Path(f"20250114-0906_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.migration_log = []

    def analyze_current_usage(self) -> Dict[str, Any]:
        """Analyze current usage of deprecated fields."""
        print("Analyzing current configuration usage...")

        analysis = {
            "deprecated_fields_found": [],
            "config_files_with_deprecated_fields": [],
            "environment_variables_with_deprecated_fields": [],
            "code_files_using_deprecated_fields": [],
            "migration_complexity": "low",
            "estimated_impact": []
        }

        # Check environment variables
        deprecated_env_vars = [
            "WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX",
            "WORKSPACE_QDRANT_WORKSPACE__MAX_COLLECTIONS"
        ]

        for env_var in deprecated_env_vars:
            if os.getenv(env_var):
                analysis["environment_variables_with_deprecated_fields"].append({
                    "name": env_var,
                    "value": os.getenv(env_var)
                })
                analysis["deprecated_fields_found"].append(env_var.split("__")[-1].lower())

        # Check current config
        try:
            config = Config()
            if config.workspace.collection_prefix:
                analysis["deprecated_fields_found"].append("collection_prefix")
                analysis["estimated_impact"].append(
                    f"Active collection_prefix '{config.workspace.collection_prefix}' needs migration"
                )

            if config.workspace.max_collections != 100:  # Default value
                analysis["deprecated_fields_found"].append("max_collections")
                analysis["estimated_impact"].append(
                    f"Custom max_collections '{config.workspace.max_collections}' needs review"
                )
        except Exception as e:
            analysis["estimated_impact"].append(f"Config loading error: {e}")

        # Estimate complexity
        if len(analysis["deprecated_fields_found"]) > 0:
            analysis["migration_complexity"] = "medium"
        if len(analysis["estimated_impact"]) > 2:
            analysis["migration_complexity"] = "high"

        return analysis

    def create_backup(self) -> bool:
        """Create backup of current configuration files."""
        if self.dry_run:
            print(f"[DRY RUN] Would create backup in {self.backup_dir}")
            return True

        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup config files
            config_files = [
                Path(".env"),
                Path("workspace_qdrant_config.yaml"),
                Path("workspace_qdrant_config.yml"),
                Path(".workspace-qdrant.yaml"),
                Path(".workspace-qdrant.yml"),
            ]

            for config_file in config_files:
                if config_file.exists():
                    backup_path = self.backup_dir / config_file.name
                    shutil.copy2(config_file, backup_path)
                    self.migration_log.append(f"Backed up {config_file} to {backup_path}")

            # Backup XDG config directories
            config = Config()
            xdg_dirs = config._get_xdg_config_dirs()

            for xdg_dir in xdg_dirs:
                if xdg_dir.exists():
                    for config_file in xdg_dir.glob("*.yaml"):
                        backup_path = self.backup_dir / f"xdg_{config_file.name}"
                        shutil.copy2(config_file, backup_path)
                        self.migration_log.append(f"Backed up XDG config {config_file} to {backup_path}")

            return True

        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def migrate_yaml_config(self, yaml_path: Path) -> bool:
        """Migrate a YAML configuration file."""
        if not yaml_path.exists():
            return True

        try:
            with yaml_path.open('r') as f:
                config_data = yaml.safe_load(f) or {}

            if 'workspace' not in config_data:
                return True  # Nothing to migrate

            workspace_config = config_data['workspace']
            migration_needed = False
            migration_notes = []

            # Handle collection_prefix
            if 'collection_prefix' in workspace_config:
                prefix = workspace_config['collection_prefix']
                del workspace_config['collection_prefix']
                migration_needed = True

                if prefix:
                    migration_notes.append(
                        f"collection_prefix '{prefix}' was removed. "
                        f"Use multi-tenant architecture with metadata filtering instead."
                    )

            # Handle max_collections
            if 'max_collections' in workspace_config:
                max_cols = workspace_config['max_collections']
                del workspace_config['max_collections']
                migration_needed = True

                migration_notes.append(
                    f"max_collections '{max_cols}' was removed. "
                    f"Multi-tenant architecture uses metadata filtering for limits."
                )

            if migration_needed:
                if self.dry_run:
                    print(f"[DRY RUN] Would migrate {yaml_path}")
                    for note in migration_notes:
                        print(f"  - {note}")
                else:
                    # Add migration comment
                    if migration_notes:
                        config_data['_migration_notes'] = {
                            'timestamp': datetime.now().isoformat(),
                            'changes': migration_notes
                        }

                    with yaml_path.open('w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

                    self.migration_log.append(f"Migrated YAML config: {yaml_path}")
                    for note in migration_notes:
                        self.migration_log.append(f"  - {note}")

            return True

        except Exception as e:
            print(f"Failed to migrate {yaml_path}: {e}")
            return False

    def migrate_env_file(self, env_path: Path) -> bool:
        """Migrate a .env file."""
        if not env_path.exists():
            return True

        try:
            lines = []
            migration_needed = False
            migration_notes = []

            with env_path.open('r') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX='):
                        value = line.split('=', 1)[1] if '=' in line else ''
                        migration_needed = True
                        migration_notes.append(
                            f"# MIGRATED: collection_prefix '{value}' removed - use multi-tenant metadata filtering"
                        )
                        lines.append(f"# {line}")  # Comment out

                    elif line.startswith('WORKSPACE_QDRANT_WORKSPACE__MAX_COLLECTIONS='):
                        value = line.split('=', 1)[1] if '=' in line else ''
                        migration_needed = True
                        migration_notes.append(
                            f"# MIGRATED: max_collections '{value}' removed - use multi-tenant architecture"
                        )
                        lines.append(f"# {line}")  # Comment out

                    else:
                        lines.append(line)

            if migration_needed:
                if self.dry_run:
                    print(f"[DRY RUN] Would migrate {env_path}")
                    for note in migration_notes:
                        print(f"  - {note}")
                else:
                    # Add migration header
                    migration_header = [
                        "",
                        "# Configuration Migration Notes",
                        f"# Migration performed: {datetime.now().isoformat()}",
                        "# Deprecated fields collection_prefix and max_collections removed",
                        "# Use multi-tenant collection architecture with metadata filtering",
                        ""
                    ]

                    with env_path.open('w') as f:
                        f.write('\n'.join(lines))
                        f.write('\n'.join(migration_header))
                        f.write('\n'.join(migration_notes))
                        f.write('\n')

                    self.migration_log.append(f"Migrated .env file: {env_path}")
                    for note in migration_notes:
                        self.migration_log.append(f"  - {note}")

            return True

        except Exception as e:
            print(f"Failed to migrate {env_path}: {e}")
            return False

    def create_migration_guide(self) -> bool:
        """Create a migration guide for users."""
        guide_content = """
# Configuration Migration Guide

## Deprecated Fields Removed

The following fields have been removed from WorkspaceConfig as part of the multi-tenant collection architecture:

### collection_prefix
- **Purpose**: Added prefix to all collection names
- **Replacement**: Use multi-tenant collections with metadata filtering
- **Migration**: Collections now use project-specific metadata instead of name prefixes

### max_collections
- **Purpose**: Limited total number of collections per workspace
- **Replacement**: Multi-tenant architecture with efficient metadata filtering
- **Migration**: Limits now handled through metadata-based tenant isolation

## New Multi-Tenant Architecture

Instead of separate collections per project, the new architecture uses:

1. **Shared Collections**: Single collections shared across projects
2. **Metadata Filtering**: Projects isolated via metadata filters
3. **Collection Types**: Configurable collection types (e.g., 'docs', 'notes', 'scratchbook')
4. **Efficient Storage**: Reduced collection overhead and better resource usage

## Migration Steps

1. **Automatic Migration**: Deprecated fields automatically removed from configs
2. **Collection Types**: Configure `collection_types` in workspace config
3. **Project Detection**: Automatic project detection via Git repository scanning
4. **Metadata Isolation**: Projects isolated via `project_id` metadata field

## Example Configuration

```yaml
workspace:
  collection_types: ["docs", "notes", "scratchbook"]  # Instead of collection_prefix
  global_collections: ["shared", "references"]        # Cross-project collections
  github_user: "username"                             # For project detection
  auto_create_collections: true                       # Automatic collection creation
```

## Benefits

- **Better Performance**: Fewer collections, more efficient queries
- **Simpler Management**: No complex prefix/suffix naming schemes
- **Scalability**: Metadata filtering scales better than multiple collections
- **Flexibility**: Dynamic project detection and isolation

## Support

If you encounter issues with the migration, check:

1. Configuration files have been backed up
2. New collection_types are properly configured
3. Project detection is working correctly
4. Existing data is accessible via new metadata filtering

Contact support if manual intervention is needed.
"""

        guide_path = Path("20250114-0906_MIGRATION_GUIDE.md")

        if self.dry_run:
            print(f"[DRY RUN] Would create migration guide: {guide_path}")
            return True

        try:
            with guide_path.open('w') as f:
                f.write(guide_content)

            self.migration_log.append(f"Created migration guide: {guide_path}")
            return True

        except Exception as e:
            print(f"Failed to create migration guide: {e}")
            return False

    def perform_migration(self) -> bool:
        """Perform the complete migration process."""
        print("Starting configuration migration...")

        # Create backup
        if not self.create_backup():
            print("Backup failed, aborting migration")
            return False

        success = True

        # Migrate local config files
        config_files = [
            Path(".env"),
            Path("workspace_qdrant_config.yaml"),
            Path("workspace_qdrant_config.yml"),
            Path(".workspace-qdrant.yaml"),
            Path(".workspace-qdrant.yml"),
        ]

        for config_file in config_files:
            if config_file.exists():
                if config_file.suffix in ['.yaml', '.yml']:
                    success &= self.migrate_yaml_config(config_file)
                elif config_file.name == '.env':
                    success &= self.migrate_env_file(config_file)

        # Migrate XDG config files
        try:
            config = Config()
            xdg_dirs = config._get_xdg_config_dirs()

            for xdg_dir in xdg_dirs:
                if xdg_dir.exists():
                    for config_file in xdg_dir.glob("*.yaml"):
                        success &= self.migrate_yaml_config(config_file)
        except Exception as e:
            print(f"XDG config migration failed: {e}")
            success = False

        # Create migration guide
        success &= self.create_migration_guide()

        # Save migration log
        if not self.dry_run:
            log_path = Path("20250114-0906_migration.log")
            with log_path.open('w') as f:
                f.write(f"Migration completed: {datetime.now().isoformat()}\n\n")
                for entry in self.migration_log:
                    f.write(f"{entry}\n")
            print(f"Migration log saved to: {log_path}")

        return success

    def rollback_migration(self) -> bool:
        """Rollback the migration if needed."""
        if not self.backup_dir.exists():
            print("No backup directory found, cannot rollback")
            return False

        try:
            # Restore backed up files
            for backup_file in self.backup_dir.iterdir():
                if backup_file.is_file():
                    if backup_file.name.startswith('xdg_'):
                        # Restore XDG config
                        original_name = backup_file.name[4:]  # Remove 'xdg_' prefix
                        config = Config()
                        xdg_dirs = config._get_xdg_config_dirs()

                        for xdg_dir in xdg_dirs:
                            if xdg_dir.exists():
                                restore_path = xdg_dir / original_name
                                shutil.copy2(backup_file, restore_path)
                                print(f"Restored XDG config: {restore_path}")
                                break
                    else:
                        # Restore local config
                        restore_path = Path(backup_file.name)
                        shutil.copy2(backup_file, restore_path)
                        print(f"Restored: {restore_path}")

            print(f"Rollback completed. Backup preserved in: {self.backup_dir}")
            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="WorkspaceConfig Migration Tool")
    parser.add_argument("--analyze", action="store_true", help="Analyze current configuration usage")
    parser.add_argument("--migrate", action="store_true", help="Perform migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")

    args = parser.parse_args()

    migration = ConfigMigration(dry_run=args.dry_run)

    if args.analyze:
        analysis = migration.analyze_current_usage()
        print("\n=== Configuration Analysis ===")
        print(json.dumps(analysis, indent=2))

        if analysis["deprecated_fields_found"]:
            print(f"\n⚠️  Migration recommended for deprecated fields: {analysis['deprecated_fields_found']}")
        else:
            print("\n✅ No deprecated fields found - no migration needed")

    elif args.migrate:
        if migration.perform_migration():
            print("\n✅ Migration completed successfully")
        else:
            print("\n❌ Migration failed")

    elif args.rollback:
        if migration.rollback_migration():
            print("\n✅ Rollback completed successfully")
        else:
            print("\n❌ Rollback failed")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()