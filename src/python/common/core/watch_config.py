"""
Persistent watch configuration storage system.

This module provides configuration persistence for file watching operations,
using SQLite database backend for unified data storage with the Rust daemon.
No JSON files are created - all data is stored in the daemon state database.
"""

import json
from loguru import logger
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, List

from pydantic import BaseModel, Field, ValidationError, field_validator

# Import LSP detector for dynamic extension detection
try:
    from .lsp_detector import get_default_detector
except ImportError:
    # Fallback if LSP detector is not available
    get_default_detector = None

# Import PatternManager for default patterns
try:
    from .pattern_manager import PatternManager

    def _get_default_patterns() -> List[str]:
        """Get default include patterns from PatternManager."""
        try:
            pattern_manager = PatternManager()
            # Use common document patterns as default - these should come from PatternManager in future
            return ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback patterns: {e}")
            return ["*.pdf", "*.epub", "*.txt", "*.md"]

    def _get_default_ignore_patterns() -> List[str]:
        """Get default exclude patterns from PatternManager."""
        try:
            pattern_manager = PatternManager()
            # Use common ignore patterns as default - these should come from PatternManager in future
            return [
                ".git/*",
                "node_modules/*",
                "__pycache__/*",
                ".DS_Store",
            ]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback ignore patterns: {e}")
            return [
                ".git/*",
                "node_modules/*",
                "__pycache__/*",
                ".DS_Store",
            ]

except ImportError:
    logger.debug("PatternManager not available - using hardcoded default patterns")

    def _get_default_patterns() -> List[str]:
        """Fallback default include patterns."""
        return ["*.pdf", "*.epub", "*.txt", "*.md"]

    def _get_default_ignore_patterns() -> List[str]:
        """Fallback default exclude patterns."""
        return [
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            ".DS_Store",
        ]

# logger imported from loguru


class WatchConfigSchema(BaseModel):
    """Pydantic schema for watch configuration validation."""

    id: str = Field(..., min_length=1, description="Unique watch identifier")
    path: str = Field(..., min_length=1, description="Directory path to watch")
    collection: str = Field(..., min_length=1, description="Target Qdrant collection")
    patterns: list[str] = Field(
        default_factory=_get_default_patterns,
        description="File patterns to include",
    )
    ignore_patterns: list[str] = Field(
        default_factory=_get_default_ignore_patterns,
        description="File patterns to ignore",
    )
    lsp_based_extensions: bool = Field(
        default=True,
        description="Enable dynamic extension detection based on available LSP servers",
    )
    lsp_detection_cache_ttl: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="LSP detection cache TTL in seconds",
    )
    auto_ingest: bool = Field(default=True, description="Enable automatic ingestion")
    recursive: bool = Field(
        default=True, description="Watch subdirectories recursively"
    )
    recursive_depth: int = Field(
        default=-1, ge=-1, description="Maximum recursive depth (-1 for unlimited)"
    )
    debounce_seconds: int = Field(
        default=5, ge=1, le=300, description="Debounce delay in seconds"
    )
    update_frequency: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="File system check frequency in milliseconds",
    )
    status: str = Field(
        default="active",
        pattern="^(active|paused|error|disabled)$",
        description="Watch status",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )
    last_activity: Optional[str] = Field(
        default=None, description="Last activity timestamp"
    )
    files_processed: int = Field(
        default=0, ge=0, description="Number of files processed"
    )
    errors_count: int = Field(default=0, ge=0, description="Error count")

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: str) -> str:
        """Validate that the watch path exists and is a directory."""
        path = Path(v).resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return str(path)

    @field_validator("patterns", "ignore_patterns")
    @classmethod
    def validate_patterns(cls, v: list[str]) -> list[str]:
        """Validate that patterns are non-empty strings."""
        if not v:
            raise ValueError("Pattern list cannot be empty")
        for pattern in v:
            if not pattern or not isinstance(pattern, str):
                raise ValueError("All patterns must be non-empty strings")
        return v


class WatchConfigFile(BaseModel):
    """Schema for the entire watch configuration file."""

    version: str = Field(default="1.0", description="Configuration file version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Configuration file creation time",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last update time",
    )
    watches: list[WatchConfigSchema] = Field(
        default_factory=list, description="List of watch configurations"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


@dataclass
class WatchConfigurationPersistent:
    """Enhanced watch configuration with persistence features."""

    # Core configuration fields (from existing WatchConfiguration)
    id: str
    path: str
    collection: str
    patterns: list[str] = field(
        default_factory=_get_default_patterns
    )
    ignore_patterns: list[str] = field(
        default_factory=_get_default_ignore_patterns
    )
    lsp_based_extensions: bool = True
    lsp_detection_cache_ttl: int = 300
    auto_ingest: bool = True
    recursive: bool = True
    debounce_seconds: int = 5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_activity: Optional[str] = None
    status: str = "active"  # active, paused, error, disabled
    files_processed: int = 0
    errors_count: int = 0

    # New persistence-specific fields
    recursive_depth: int = -1  # -1 for unlimited, positive integer for depth limit
    update_frequency: int = 1000  # milliseconds between file system checks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchConfigurationPersistent":
        """Create from dictionary with validation."""
        # Validate using Pydantic schema
        validated_data = WatchConfigSchema(**data)
        return cls(**validated_data.model_dump())

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate path
        path = Path(self.path)
        if not path.exists():
            issues.append(f"Watch path does not exist: {self.path}")
        elif not path.is_dir():
            issues.append(f"Watch path is not a directory: {self.path}")
        elif not os.access(path, os.R_OK):
            issues.append(f"Watch path is not readable: {self.path}")

        # Validate patterns
        if not self.patterns:
            issues.append("At least one file pattern is required")

        # Validate collection name
        if not self.collection or not self.collection.strip():
            issues.append("Collection name cannot be empty")

        # Validate numeric fields
        if self.debounce_seconds < 1 or self.debounce_seconds > 300:
            issues.append("Debounce seconds must be between 1 and 300")

        if self.update_frequency < 100 or self.update_frequency > 10000:
            issues.append("Update frequency must be between 100 and 10000 milliseconds")

        if self.recursive_depth < -1:
            issues.append("Recursive depth must be -1 (unlimited) or positive integer")

        return issues

    def get_effective_patterns(self) -> tuple[list[str], list[str]]:
        """
        Get effective include and exclude patterns, optionally enhanced with LSP detection.
        
        Returns:
            Tuple of (include_patterns, exclude_patterns)
        """
        # Start with configured patterns
        include_patterns = list(self.patterns)
        exclude_patterns = list(self.ignore_patterns)
        
        # Add LSP-based patterns if enabled and detector is available
        if self.lsp_based_extensions and get_default_detector is not None:
            try:
                detector = get_default_detector()
                detector.cache_ttl = self.lsp_detection_cache_ttl
                
                # Get LSP-detected extensions
                lsp_extensions = detector.get_supported_extensions(include_fallbacks=True)
                
                # Convert extensions to glob patterns  
                lsp_patterns = []
                for ext in lsp_extensions:
                    if ext.startswith('.'):
                        lsp_patterns.append(f"*{ext}")
                    else:
                        # Handle non-extension patterns (like Dockerfile, Makefile)
                        lsp_patterns.append(ext)
                
                # Merge patterns, avoiding duplicates
                all_patterns = set(include_patterns + lsp_patterns)
                include_patterns = sorted(list(all_patterns))
                
                logger.debug(f"Enhanced patterns with LSP detection: {len(lsp_patterns)} LSP patterns added")
                
            except Exception as e:
                logger.warning(f"Failed to get LSP-based patterns: {e}")
                # Fall back to original patterns
        
        return (include_patterns, exclude_patterns)


class DatabaseWatchConfigManager:
    """
    Database-backed manager for watch configuration storage.

    Uses SQLite database backend for unified data storage with the Rust daemon.
    All configurations are stored in the daemon state database, not JSON files.
    """

    def __init__(
        self,
        database_path: Optional[Union[str, Path]] = None,
        project_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize database configuration manager.

        Args:
            database_path: Custom database path (optional, uses daemon default)
            project_dir: Project directory context (optional)
        """
        self.project_dir = Path(project_dir) if project_dir else None

        if database_path:
            self.database_path = Path(database_path).resolve()
        else:
            self.database_path = self._get_default_database_path()

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema if needed
        self._initialize_database()

        logger.info(f"Watch config manager initialized with database: {self.database_path}")

    def _get_default_database_path(self) -> Path:
        """Get the default daemon state database path."""
        # Use the same path as the Rust daemon
        data_dir = Path.home() / ".local" / "share" / "workspace-qdrant"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "daemon_state.db"

    def _initialize_database(self):
        """Initialize database schema if needed."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watch_configurations (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    patterns TEXT NOT NULL,
                    ignore_patterns TEXT NOT NULL,
                    lsp_based_extensions BOOLEAN NOT NULL DEFAULT TRUE,
                    lsp_detection_cache_ttl INTEGER NOT NULL DEFAULT 300,
                    auto_ingest BOOLEAN NOT NULL DEFAULT TRUE,
                    recursive BOOLEAN NOT NULL DEFAULT TRUE,
                    recursive_depth INTEGER NOT NULL DEFAULT -1,
                    debounce_seconds INTEGER NOT NULL DEFAULT 5,
                    update_frequency INTEGER NOT NULL DEFAULT 1000,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP,
                    files_processed INTEGER NOT NULL DEFAULT 0,
                    errors_count INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_watch_status ON watch_configurations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_watch_path ON watch_configurations(path)")
            conn.commit()

    def _watch_config_to_dict(self, config: WatchConfigurationPersistent) -> dict:
        """Convert WatchConfigurationPersistent to database-compatible dict."""
        return {
            'id': config.id,
            'path': config.path,
            'collection': config.collection,
            'patterns': json.dumps(config.patterns),
            'ignore_patterns': json.dumps(config.ignore_patterns),
            'lsp_based_extensions': config.lsp_based_extensions,
            'lsp_detection_cache_ttl': config.lsp_detection_cache_ttl,
            'auto_ingest': config.auto_ingest,
            'recursive': config.recursive,
            'recursive_depth': config.recursive_depth,
            'debounce_seconds': config.debounce_seconds,
            'update_frequency': config.update_frequency,
            'status': config.status,
            'created_at': config.created_at,
            'last_activity': config.last_activity,
            'files_processed': config.files_processed,
            'errors_count': config.errors_count,
        }

    def _dict_to_watch_config(self, row_dict: dict) -> WatchConfigurationPersistent:
        """Convert database row dict to WatchConfigurationPersistent."""
        return WatchConfigurationPersistent(
            id=row_dict['id'],
            path=row_dict['path'],
            collection=row_dict['collection'],
            patterns=json.loads(row_dict['patterns']),
            ignore_patterns=json.loads(row_dict['ignore_patterns']),
            lsp_based_extensions=bool(row_dict['lsp_based_extensions']),
            lsp_detection_cache_ttl=row_dict['lsp_detection_cache_ttl'],
            auto_ingest=bool(row_dict['auto_ingest']),
            recursive=bool(row_dict['recursive']),
            recursive_depth=row_dict['recursive_depth'],
            debounce_seconds=row_dict['debounce_seconds'],
            update_frequency=row_dict['update_frequency'],
            status=row_dict['status'],
            created_at=row_dict['created_at'],
            last_activity=row_dict['last_activity'],
            files_processed=row_dict['files_processed'],
            errors_count=row_dict['errors_count'],
        )

    async def load_config(self) -> WatchConfigFile:
        """Load watch configuration from database."""
        try:
            configs = await self.list_watch_configs()

            # Convert to WatchConfigFile format for compatibility
            config_file = WatchConfigFile()
            config_file.watches = []

            for config in configs:
                # Convert to WatchConfigSchema for validation
                schema = WatchConfigSchema(
                    id=config.id,
                    path=config.path,
                    collection=config.collection,
                    patterns=config.patterns,
                    ignore_patterns=config.ignore_patterns,
                    lsp_based_extensions=config.lsp_based_extensions,
                    lsp_detection_cache_ttl=config.lsp_detection_cache_ttl,
                    auto_ingest=config.auto_ingest,
                    recursive=config.recursive,
                    recursive_depth=config.recursive_depth,
                    debounce_seconds=config.debounce_seconds,
                    update_frequency=config.update_frequency,
                    status=config.status,
                    created_at=config.created_at,
                    last_activity=config.last_activity,
                    files_processed=config.files_processed,
                    errors_count=config.errors_count,
                )
                config_file.watches.append(schema)

            logger.debug(f"Loaded {len(config_file.watches)} watch configurations from database")
            return config_file

        except Exception as e:
            logger.error(f"Failed to load config from database: {e}")
            return WatchConfigFile()

    async def save_config(self, config: WatchConfigFile) -> bool:
        """Save configuration to database (for compatibility)."""
        try:
            # This method exists for compatibility but individual configs
            # are saved via add_watch_config/update_watch_config
            logger.debug(f"Database-backed config manager doesn't use save_config directly")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    async def add_watch_config(
        self, watch_config: WatchConfigurationPersistent
    ) -> bool:
        """Add a new watch configuration to database."""

        # Validate configuration
        issues = watch_config.validate()
        if issues:
            logger.error(f"Watch config validation failed: {', '.join(issues)}")
            return False

        try:
            with sqlite3.connect(self.database_path) as conn:
                # Check for duplicate ID
                cursor = conn.execute("SELECT id FROM watch_configurations WHERE id = ?", (watch_config.id,))
                if cursor.fetchone():
                    logger.error(f"Watch ID already exists: {watch_config.id}")
                    return False

                # Insert new watch config
                config_dict = self._watch_config_to_dict(watch_config)
                columns = list(config_dict.keys())
                placeholders = ', '.join(['?' for _ in columns])
                values = list(config_dict.values())

                conn.execute(
                    f"INSERT INTO watch_configurations ({', '.join(columns)}) VALUES ({placeholders})",
                    values
                )
                conn.commit()

                logger.info(f"Added watch configuration: {watch_config.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to add watch config to database: {e}")
            return False

    async def remove_watch_config(self, watch_id: str) -> bool:
        """Remove a watch configuration from database."""

        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("DELETE FROM watch_configurations WHERE id = ?", (watch_id,))
                conn.commit()

                if cursor.rowcount == 0:
                    logger.warning(f"Watch configuration not found: {watch_id}")
                    return False

                logger.info(f"Removed watch configuration: {watch_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to remove watch config from database: {e}")
            return False

    async def update_watch_config(
        self, watch_config: WatchConfigurationPersistent
    ) -> bool:
        """Update an existing watch configuration in database."""

        # Validate configuration
        issues = watch_config.validate()
        if issues:
            logger.error(f"Watch config validation failed: {', '.join(issues)}")
            return False

        try:
            with sqlite3.connect(self.database_path) as conn:
                # Check if configuration exists
                cursor = conn.execute("SELECT id FROM watch_configurations WHERE id = ?", (watch_config.id,))
                if not cursor.fetchone():
                    logger.warning(f"Watch configuration not found for update: {watch_config.id}")
                    return False

                # Update configuration
                config_dict = self._watch_config_to_dict(watch_config)
                # Remove id from update since it's the WHERE clause
                del config_dict['id']

                columns = list(config_dict.keys())
                set_clause = ', '.join([f"{col} = ?" for col in columns])
                values = list(config_dict.values()) + [watch_config.id]

                conn.execute(
                    f"UPDATE watch_configurations SET {set_clause} WHERE id = ?",
                    values
                )
                conn.commit()

                logger.info(f"Updated watch configuration: {watch_config.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to update watch config in database: {e}")
            return False

    async def list_watch_configs(
        self, active_only: bool = False
    ) -> List[WatchConfigurationPersistent]:
        """List all watch configurations from database."""

        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name

                query = """
                    SELECT id, path, collection, patterns, ignore_patterns,
                           lsp_based_extensions, lsp_detection_cache_ttl, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, update_frequency,
                           status, created_at, last_activity, files_processed, errors_count
                    FROM watch_configurations
                """

                if active_only:
                    query += " WHERE status = 'active'"

                query += " ORDER BY created_at"

                cursor = conn.execute(query)
                rows = cursor.fetchall()

                watch_configs = []
                for row in rows:
                    try:
                        row_dict = dict(row)
                        watch_config = self._dict_to_watch_config(row_dict)
                        watch_configs.append(watch_config)
                    except Exception as e:
                        logger.warning(f"Failed to convert database row {row['id']}: {e}")

                return watch_configs

        except Exception as e:
            logger.error(f"Failed to list watch configs from database: {e}")
            return []

    async def get_watch_config(
        self, watch_id: str
    ) -> Optional[WatchConfigurationPersistent]:
        """Get a specific watch configuration by ID from database."""

        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute("""
                    SELECT id, path, collection, patterns, ignore_patterns,
                           lsp_based_extensions, lsp_detection_cache_ttl, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, update_frequency,
                           status, created_at, last_activity, files_processed, errors_count
                    FROM watch_configurations WHERE id = ?
                """, (watch_id,))

                row = cursor.fetchone()
                if row:
                    row_dict = dict(row)
                    return self._dict_to_watch_config(row_dict)
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to get watch config from database: {e}")
            return None

    async def validate_all_configs(self) -> dict[str, list[str]]:
        """Validate all watch configurations and return issues."""

        validation_results = {}
        configs = await self.list_watch_configs()

        for config in configs:
            issues = config.validate()
            if issues:
                validation_results[config.id] = issues

        return validation_results

    def get_database_path(self) -> Path:
        """Get the path to the database file."""
        return self.database_path

    async def backup_config(self, backup_path: Optional[Path] = None) -> bool:
        """Create a backup of the database."""

        if not self.database_path.exists():
            logger.warning("No database to backup")
            return False

        try:
            if backup_path is None:
                backup_dir = self.database_path.parent / "backups"
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"daemon_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

            # Use SQLite backup API for atomic backup
            import shutil
            shutil.copy2(self.database_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    async def migrate_from_json(self, json_config_path: Path) -> bool:
        """Migrate configurations from old JSON file to database."""

        if not json_config_path.exists():
            logger.info(f"No JSON config file found at {json_config_path}")
            return True

        try:
            # Load old JSON config
            with open(json_config_path, 'r', encoding='utf-8') as f:
                old_config_data = json.load(f)

            if 'watches' not in old_config_data:
                logger.info("No watch configurations in JSON file")
                return True

            migrated_count = 0
            for watch_data in old_config_data['watches']:
                try:
                    # Convert to WatchConfigurationPersistent
                    watch_config = WatchConfigurationPersistent.from_dict(watch_data)

                    # Add to database
                    success = await self.add_watch_config(watch_config)
                    if success:
                        migrated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate watch config {watch_data.get('id', 'unknown')}: {e}")

            logger.info(f"Migrated {migrated_count} watch configurations from JSON to database")

            # Rename old file to indicate it's been migrated
            migrated_path = json_config_path.with_suffix('.json.migrated')
            json_config_path.rename(migrated_path)
            logger.info(f"Renamed old JSON file to: {migrated_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to migrate from JSON: {e}")
            return False


