"""Version Migration Framework for Backup/Restore Operations.

This module provides a foundation for handling version migrations when restoring
backups from different versions of the system. It allows for:

- Defining migration strategies for specific version transitions
- Registering and discovering available migrations
- Applying migrations during restore operations
- Validating migration compatibility and safety

The framework is designed to be extensible, allowing future versions to add
migration strategies without modifying core backup/restore logic.

Usage:
    # Define a migration
    @register_migration(from_version="0.2.0", to_version="0.3.0")
    class MigrateTo030(BaseMigration):
        def migrate(self, backup_data: BackupData) -> BackupData:
            # Migration logic here
            return backup_data

    # Apply migrations during restore
    manager = MigrationManager()
    migrated_data = manager.apply_migrations(
        backup_data,
        from_version="0.2.0",
        to_version="0.3.0"
    )
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackupData:
    """Container for backup data during migration.

    This class encapsulates all data from a backup that might need to be
    transformed during version migrations.

    Attributes:
        version: Original backup version
        manifest: Backup manifest data
        collections: Collection metadata and configurations
        state_db: State database content (if applicable)
        metadata: Additional backup metadata
    """

    version: str
    manifest: dict[str, Any]
    collections: dict[str, Any] = field(default_factory=dict)
    state_db: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationResult:
    """Result of a migration operation.

    Attributes:
        success: Whether migration succeeded
        backup_data: Migrated backup data
        applied_migrations: List of migrations applied
        warnings: Any warnings generated during migration
        errors: Any errors encountered (if success=False)
    """

    success: bool
    backup_data: BackupData | None = None
    applied_migrations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseMigration(ABC):
    """Base class for version migrations.

    All version migrations must inherit from this class and implement
    the migrate() method to transform backup data from one version to another.

    Attributes:
        from_version: Source version this migration applies to
        to_version: Target version after migration
        description: Human-readable description of what this migration does
        reversible: Whether this migration can be reversed
    """

    from_version: str
    to_version: str
    description: str = "No description provided"
    reversible: bool = False

    @abstractmethod
    def migrate(self, backup_data: BackupData) -> BackupData:
        """Apply migration to backup data.

        Args:
            backup_data: Backup data at from_version

        Returns:
            BackupData: Transformed backup data at to_version

        Raises:
            MigrationError: If migration fails
        """
        pass

    def reverse(self, backup_data: BackupData) -> BackupData:
        """Reverse this migration (if reversible).

        Args:
            backup_data: Backup data at to_version

        Returns:
            BackupData: Transformed backup data back to from_version

        Raises:
            NotImplementedError: If migration is not reversible
        """
        if not self.reversible:
            raise NotImplementedError(
                f"Migration {self.__class__.__name__} is not reversible"
            )
        raise NotImplementedError("Reverse migration not implemented")

    def validate(self, backup_data: BackupData) -> bool:
        """Validate that this migration can be safely applied.

        Args:
            backup_data: Backup data to validate

        Returns:
            bool: True if migration can be safely applied
        """
        # Default implementation - can be overridden by subclasses
        return backup_data.version == self.from_version

    def get_warnings(self, backup_data: BackupData) -> list[str]:
        """Get any warnings about applying this migration.

        Args:
            backup_data: Backup data to check

        Returns:
            List[str]: List of warning messages (empty if none)
        """
        # Default implementation - can be overridden by subclasses
        return []


class MigrationRegistry:
    """Registry for available version migrations.

    Maintains a mapping of version transitions to migration classes,
    allowing the migration manager to discover and apply appropriate
    migrations during restore operations.
    """

    def __init__(self):
        """Initialize empty migration registry."""
        self._migrations: dict[tuple[str, str], type[BaseMigration]] = {}

    def register(
        self,
        from_version: str,
        to_version: str,
        migration_class: type[BaseMigration],
    ) -> None:
        """Register a migration for a specific version transition.

        Args:
            from_version: Source version
            to_version: Target version
            migration_class: Migration class to handle this transition

        Raises:
            ValueError: If migration already registered for this transition
        """
        key = (from_version, to_version)
        if key in self._migrations:
            existing = self._migrations[key]
            logger.warning(
                f"Migration for {from_version} -> {to_version} already registered "
                f"({existing.__name__}), replacing with {migration_class.__name__}"
            )
        self._migrations[key] = migration_class
        logger.info(
            f"Registered migration: {migration_class.__name__} "
            f"({from_version} -> {to_version})"
        )

    def get_migration(
        self, from_version: str, to_version: str
    ) -> type[BaseMigration] | None:
        """Get migration class for a specific version transition.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Optional[type[BaseMigration]]: Migration class if registered, None otherwise
        """
        return self._migrations.get((from_version, to_version))

    def list_migrations(self) -> list[tuple[str, str, str]]:
        """List all registered migrations.

        Returns:
            List of tuples: (from_version, to_version, migration_name)
        """
        return [
            (from_ver, to_ver, cls.__name__)
            for (from_ver, to_ver), cls in self._migrations.items()
        ]

    def has_migration(self, from_version: str, to_version: str) -> bool:
        """Check if migration exists for version transition.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            bool: True if migration registered
        """
        return (from_version, to_version) in self._migrations


# Global migration registry instance
_registry = MigrationRegistry()


def register_migration(
    from_version: str, to_version: str
) -> Callable[[type[BaseMigration]], type[BaseMigration]]:
    """Decorator to register a migration class.

    Args:
        from_version: Source version this migration applies to
        to_version: Target version after migration

    Returns:
        Decorator function that registers the migration class

    Example:
        @register_migration(from_version="0.2.0", to_version="0.3.0")
        class MigrateTo030(BaseMigration):
            def migrate(self, backup_data):
                # Migration logic
                return backup_data
    """

    def decorator(cls: type[BaseMigration]) -> type[BaseMigration]:
        # Set version attributes on class
        cls.from_version = from_version
        cls.to_version = to_version
        # Register in global registry
        _registry.register(from_version, to_version, cls)
        return cls

    return decorator


def get_registry() -> MigrationRegistry:
    """Get the global migration registry.

    Returns:
        MigrationRegistry: Global registry instance
    """
    return _registry


class MigrationManager:
    """Manager for applying version migrations.

    Handles finding and applying appropriate migrations when restoring
    backups from different versions.
    """

    def __init__(self, registry: MigrationRegistry | None = None):
        """Initialize migration manager.

        Args:
            registry: Migration registry to use (uses global if None)
        """
        self.registry = registry or get_registry()

    def apply_migrations(
        self, backup_data: BackupData, from_version: str, to_version: str
    ) -> MigrationResult:
        """Apply migrations to transform backup data between versions.

        Args:
            backup_data: Original backup data
            from_version: Version of the backup
            to_version: Target version to migrate to

        Returns:
            MigrationResult: Result of migration operation
        """
        # Check if direct migration exists
        migration_class = self.registry.get_migration(from_version, to_version)

        if migration_class is not None:
            return self._apply_single_migration(
                backup_data, migration_class, from_version, to_version
            )

        # TODO: In future, implement multi-hop migrations
        # For now, only support direct migrations
        logger.warning(
            f"No direct migration from {from_version} to {to_version}. "
            "Multi-hop migrations not yet supported."
        )

        return MigrationResult(
            success=False,
            errors=[
                f"No migration available from {from_version} to {to_version}",
                "Multi-hop migrations not yet supported",
            ],
        )

    def _apply_single_migration(
        self,
        backup_data: BackupData,
        migration_class: type[BaseMigration],
        from_version: str,
        to_version: str,
    ) -> MigrationResult:
        """Apply a single migration.

        Args:
            backup_data: Backup data to migrate
            migration_class: Migration class to apply
            from_version: Source version
            to_version: Target version

        Returns:
            MigrationResult: Result of migration
        """
        try:
            # Instantiate migration
            migration = migration_class()

            # Validate migration can be applied
            if not migration.validate(backup_data):
                return MigrationResult(
                    success=False,
                    errors=[
                        f"Migration {migration_class.__name__} validation failed"
                    ],
                )

            # Get any warnings
            warnings = migration.get_warnings(backup_data)

            # Apply migration
            logger.info(
                f"Applying migration {migration_class.__name__}: "
                f"{from_version} -> {to_version}"
            )
            migrated_data = migration.migrate(backup_data)

            return MigrationResult(
                success=True,
                backup_data=migrated_data,
                applied_migrations=[migration_class.__name__],
                warnings=warnings,
            )

        except Exception as e:
            logger.error(
                f"Migration {migration_class.__name__} failed: {e}", exc_info=True
            )
            return MigrationResult(
                success=False,
                errors=[f"Migration failed: {str(e)}"],
            )

    def get_migration_path(
        self, from_version: str, to_version: str
    ) -> list[tuple[str, str]] | None:
        """Find path of migrations between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Optional[List[Tuple[str, str]]]: List of (from, to) version pairs
                representing migration path, or None if no path exists

        Note:
            Current implementation only supports direct migrations.
            Future versions will implement graph-based path finding.
        """
        # Check for direct migration
        if self.registry.has_migration(from_version, to_version):
            return [(from_version, to_version)]

        # TODO: Implement multi-hop migration path finding
        # This would use graph algorithms (e.g., BFS) to find a path
        # through multiple migrations
        return None

    def list_available_migrations(
        self, from_version: str | None = None
    ) -> list[tuple[str, str, str]]:
        """List available migrations.

        Args:
            from_version: Filter to migrations from this version (optional)

        Returns:
            List of tuples: (from_version, to_version, migration_name)
        """
        all_migrations = self.registry.list_migrations()

        if from_version is not None:
            return [
                (from_ver, to_ver, name)
                for from_ver, to_ver, name in all_migrations
                if from_ver == from_version
            ]

        return all_migrations
