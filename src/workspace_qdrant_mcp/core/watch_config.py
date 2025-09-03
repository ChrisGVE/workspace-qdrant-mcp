"""
Persistent watch configuration storage system.

This module provides configuration persistence for file watching operations,
including schema validation, atomic file operations, and backup recovery.
Configurations are stored in JSON format with proper versioning and validation.
"""

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class WatchConfigSchema(BaseModel):
    """Pydantic schema for watch configuration validation."""

    id: str = Field(..., min_length=1, description="Unique watch identifier")
    path: str = Field(..., min_length=1, description="Directory path to watch")
    collection: str = Field(..., min_length=1, description="Target Qdrant collection")
    patterns: list[str] = Field(
        default_factory=lambda: ["*.pdf", "*.epub", "*.txt", "*.md"],
        description="File patterns to include",
    )
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            ".DS_Store",
        ],
        description="File patterns to ignore",
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
    metadata: Dict[str, Any] = Field(
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
        default_factory=lambda: ["*.pdf", "*.epub", "*.txt", "*.md"]
    )
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            ".DS_Store",
        ]
    )
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchConfigurationPersistent":
        """Create from dictionary with validation."""
        # Validate using Pydantic schema
        validated_data = WatchConfigSchema(**data)
        return cls(**validated_data.dict())

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


class PersistentWatchConfigManager:
    """
    Manager for persistent watch configuration storage.

    Handles configuration file operations including validation, atomic writes,
    backup creation, and error recovery for watch folder configurations.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        project_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            config_file: Custom config file path (optional)
            project_dir: Project directory for project-specific config (optional)
        """
        self.project_dir = Path(project_dir) if project_dir else None

        if config_file:
            self.config_file = Path(config_file).resolve()
        else:
            self.config_file = self._determine_config_location()

        self.backup_file = self.config_file.with_suffix(".json.backup")
        self.temp_file = self.config_file.with_suffix(".json.tmp")

        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Watch config manager initialized: {self.config_file}")

    def _determine_config_location(self) -> Path:
        """Determine the best location for configuration file."""

        # Option 1: Project-specific config (if project_dir provided)
        if self.project_dir:
            project_config_dir = self.project_dir / ".workspace-qdrant"
            project_config_file = project_config_dir / "watch-config.json"
            if project_config_dir.exists() or self.project_dir.is_dir():
                return project_config_file

        # Option 2: User-specific config directory
        user_config_dir = Path.home() / ".config" / "workspace-qdrant"
        user_config_file = user_config_dir / "watch-config.json"

        # Option 3: Fallback to home directory
        fallback_config_dir = Path.home() / ".workspace-qdrant"
        fallback_config_file = fallback_config_dir / "watch-config.json"

        # Use user config by default
        return user_config_file

    async def load_config(self) -> WatchConfigFile:
        """Load watch configuration from file with error recovery."""

        if not self.config_file.exists():
            logger.info("No watch configuration file found, creating new one")
            return WatchConfigFile()

        try:
            # Try to load main config file
            config_data = self._load_json_file(self.config_file)
            config = WatchConfigFile(**config_data)
            logger.debug(f"Loaded {len(config.watches)} watch configurations")
            return config

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to load config file: {e}")

            # Try to load from backup
            if self.backup_file.exists():
                logger.info("Attempting recovery from backup file")
                try:
                    backup_data = self._load_json_file(self.backup_file)
                    config = WatchConfigFile(**backup_data)
                    logger.info("Successfully recovered configuration from backup")

                    # Save recovered config as main config
                    await self.save_config(config)
                    return config

                except Exception as backup_error:
                    logger.error(f"Backup recovery failed: {backup_error}")

            # Create fresh config if recovery fails
            logger.warning("Creating fresh configuration file")
            return WatchConfigFile()

        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            return WatchConfigFile()

    def _load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load and parse JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def save_config(self, config: WatchConfigFile) -> bool:
        """Save configuration with atomic write and backup creation."""

        try:
            # Update timestamp
            config.updated_at = datetime.now(timezone.utc).isoformat()

            # Create backup of existing config
            if self.config_file.exists():
                shutil.copy2(self.config_file, self.backup_file)
                logger.debug("Created configuration backup")

            # Write to temporary file first (atomic operation)
            with open(self.temp_file, "w", encoding="utf-8") as f:
                json.dump(config.dict(), f, indent=2, ensure_ascii=False)

            # Atomically move temp file to main config
            self.temp_file.replace(self.config_file)

            logger.debug(f"Saved {len(config.watches)} watch configurations")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

            # Clean up temp file if it exists
            if self.temp_file.exists():
                self.temp_file.unlink()

            return False

    async def add_watch_config(
        self, watch_config: WatchConfigurationPersistent
    ) -> bool:
        """Add a new watch configuration."""

        # Validate configuration
        issues = watch_config.validate()
        if issues:
            logger.error(f"Watch config validation failed: {', '.join(issues)}")
            return False

        try:
            config = await self.load_config()

            # Check for duplicate ID
            existing_ids = [w.id for w in config.watches]
            if watch_config.id in existing_ids:
                logger.error(f"Watch ID already exists: {watch_config.id}")
                return False

            # Add new watch config
            validated_config = WatchConfigSchema(**watch_config.to_dict())
            config.watches.append(validated_config)

            # Save configuration
            success = await self.save_config(config)
            if success:
                logger.info(f"Added watch configuration: {watch_config.id}")
            return success

        except Exception as e:
            logger.error(f"Failed to add watch config: {e}")
            return False

    async def remove_watch_config(self, watch_id: str) -> bool:
        """Remove a watch configuration."""

        try:
            config = await self.load_config()

            # Find and remove configuration
            original_count = len(config.watches)
            config.watches = [w for w in config.watches if w.id != watch_id]

            if len(config.watches) == original_count:
                logger.warning(f"Watch configuration not found: {watch_id}")
                return False

            # Save updated configuration
            success = await self.save_config(config)
            if success:
                logger.info(f"Removed watch configuration: {watch_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to remove watch config: {e}")
            return False

    async def update_watch_config(
        self, watch_config: WatchConfigurationPersistent
    ) -> bool:
        """Update an existing watch configuration."""

        # Validate configuration
        issues = watch_config.validate()
        if issues:
            logger.error(f"Watch config validation failed: {', '.join(issues)}")
            return False

        try:
            config = await self.load_config()

            # Find and update configuration
            updated = False
            for i, existing_config in enumerate(config.watches):
                if existing_config.id == watch_config.id:
                    validated_config = WatchConfigSchema(**watch_config.to_dict())
                    config.watches[i] = validated_config
                    updated = True
                    break

            if not updated:
                logger.warning(
                    f"Watch configuration not found for update: {watch_config.id}"
                )
                return False

            # Save updated configuration
            success = await self.save_config(config)
            if success:
                logger.info(f"Updated watch configuration: {watch_config.id}")
            return success

        except Exception as e:
            logger.error(f"Failed to update watch config: {e}")
            return False

    async def list_watch_configs(
        self, active_only: bool = False
    ) -> list[WatchConfigurationPersistent]:
        """List all watch configurations."""

        try:
            config = await self.load_config()

            # Convert to WatchConfigurationPersistent objects
            watch_configs = []
            for config_data in config.watches:
                try:
                    watch_config = WatchConfigurationPersistent.from_dict(
                        config_data.dict()
                    )
                    if active_only and watch_config.status != "active":
                        continue
                    watch_configs.append(watch_config)
                except Exception as e:
                    logger.warning(f"Failed to convert config {config_data.id}: {e}")

            return sorted(watch_configs, key=lambda x: x.created_at)

        except Exception as e:
            logger.error(f"Failed to list watch configs: {e}")
            return []

    async def get_watch_config(
        self, watch_id: str
    ) -> Optional[WatchConfigurationPersistent]:
        """Get a specific watch configuration by ID."""

        configs = await self.list_watch_configs()
        for config in configs:
            if config.id == watch_id:
                return config
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

    def get_config_file_path(self) -> Path:
        """Get the path to the configuration file."""
        return self.config_file

    async def backup_config(self, backup_path: Optional[Path] = None) -> bool:
        """Create a backup of the current configuration."""

        if not self.config_file.exists():
            logger.warning("No configuration file to backup")
            return False

        try:
            backup_path = backup_path or self.config_file.with_suffix(
                f".json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            shutil.copy2(self.config_file, backup_path)
            logger.info(f"Configuration backed up to: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return False
