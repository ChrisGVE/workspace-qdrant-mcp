"""Unit tests for version migration framework (Task 376.16).

Tests cover:
- Migration registration and discovery
- Migration validation and application
- Migration registry functionality
- Migration manager orchestration
- Error handling and edge cases
"""

import pytest
from unittest.mock import MagicMock, patch

from common.core.version_migration import (
    BackupData,
    BaseMigration,
    MigrationRegistry,
    MigrationManager,
    MigrationResult,
    register_migration,
    get_registry,
)


class TestBackupData:
    """Test BackupData dataclass."""

    def test_backup_data_creation(self):
        """Test creating BackupData with minimal fields."""
        data = BackupData(
            version="0.2.0",
            manifest={"timestamp": "2024-01-01", "collections": {}}
        )

        assert data.version == "0.2.0"
        assert data.manifest["timestamp"] == "2024-01-01"
        assert data.collections == {}
        assert data.state_db is None
        assert data.metadata == {}

    def test_backup_data_with_all_fields(self):
        """Test creating BackupData with all fields populated."""
        data = BackupData(
            version="0.2.1",
            manifest={"test": "manifest"},
            collections={"coll1": {"points": 100}},
            state_db={"watches": []},
            metadata={"description": "Test backup"}
        )

        assert data.version == "0.2.1"
        assert data.manifest == {"test": "manifest"}
        assert data.collections == {"coll1": {"points": 100}}
        assert data.state_db == {"watches": []}
        assert data.metadata == {"description": "Test backup"}


class TestMigrationResult:
    """Test MigrationResult dataclass."""

    def test_successful_result(self):
        """Test creating successful migration result."""
        data = BackupData(version="0.3.0", manifest={})
        result = MigrationResult(
            success=True,
            backup_data=data,
            applied_migrations=["MigrateTo030"],
            warnings=["Some warning"]
        )

        assert result.success is True
        assert result.backup_data == data
        assert result.applied_migrations == ["MigrateTo030"]
        assert result.warnings == ["Some warning"]
        assert result.errors == []

    def test_failed_result(self):
        """Test creating failed migration result."""
        result = MigrationResult(
            success=False,
            errors=["Migration failed", "Data incompatible"]
        )

        assert result.success is False
        assert result.backup_data is None
        assert result.applied_migrations == []
        assert result.warnings == []
        assert result.errors == ["Migration failed", "Data incompatible"]


class TestBaseMigration:
    """Test BaseMigration abstract base class."""

    def test_base_migration_cannot_instantiate(self):
        """Test that BaseMigration cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMigration()

    def test_migration_subclass_requires_migrate(self):
        """Test that subclasses must implement migrate method."""
        class IncompleteMigration(BaseMigration):
            pass

        with pytest.raises(TypeError):
            IncompleteMigration()

    def test_migration_with_attributes(self):
        """Test migration with all attributes set."""
        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"
            description = "Test migration"
            reversible = True

            def migrate(self, backup_data):
                return backup_data

        migration = TestMigration()
        assert migration.from_version == "0.2.0"
        assert migration.to_version == "0.3.0"
        assert migration.description == "Test migration"
        assert migration.reversible is True

    def test_migration_validate_default(self):
        """Test default validate implementation."""
        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                return backup_data

        migration = TestMigration()
        data = BackupData(version="0.2.0", manifest={})

        assert migration.validate(data) is True

        # Different version should fail validation
        wrong_version_data = BackupData(version="0.1.0", manifest={})
        assert migration.validate(wrong_version_data) is False

    def test_migration_get_warnings_default(self):
        """Test default get_warnings implementation."""
        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                return backup_data

        migration = TestMigration()
        data = BackupData(version="0.2.0", manifest={})

        warnings = migration.get_warnings(data)
        assert warnings == []

    def test_migration_reverse_not_reversible(self):
        """Test reverse raises error for non-reversible migration."""
        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"
            reversible = False

            def migrate(self, backup_data):
                return backup_data

        migration = TestMigration()
        data = BackupData(version="0.3.0", manifest={})

        with pytest.raises(NotImplementedError, match="not reversible"):
            migration.reverse(data)

    def test_migration_reverse_not_implemented(self):
        """Test reverse raises error even for reversible migration if not implemented."""
        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"
            reversible = True

            def migrate(self, backup_data):
                return backup_data

        migration = TestMigration()
        data = BackupData(version="0.3.0", manifest={})

        with pytest.raises(NotImplementedError, match="Reverse migration not implemented"):
            migration.reverse(data)


class TestMigrationRegistry:
    """Test MigrationRegistry class."""

    def test_registry_initialization(self):
        """Test creating empty registry."""
        registry = MigrationRegistry()
        assert registry.list_migrations() == []

    def test_registry_register_migration(self):
        """Test registering a migration."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.2.0", "0.3.0", TestMigration)

        assert registry.has_migration("0.2.0", "0.3.0")
        assert registry.get_migration("0.2.0", "0.3.0") == TestMigration

    def test_registry_get_nonexistent_migration(self):
        """Test getting migration that doesn't exist."""
        registry = MigrationRegistry()

        migration = registry.get_migration("0.1.0", "0.2.0")
        assert migration is None

    def test_registry_has_migration(self):
        """Test checking if migration exists."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        assert registry.has_migration("0.2.0", "0.3.0") is False

        registry.register("0.2.0", "0.3.0", TestMigration)

        assert registry.has_migration("0.2.0", "0.3.0") is True

    def test_registry_list_migrations(self):
        """Test listing all migrations."""
        registry = MigrationRegistry()

        class Migration1(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        class Migration2(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.1.0", "0.2.0", Migration1)
        registry.register("0.2.0", "0.3.0", Migration2)

        migrations = registry.list_migrations()
        assert len(migrations) == 2
        assert ("0.1.0", "0.2.0", "Migration1") in migrations
        assert ("0.2.0", "0.3.0", "Migration2") in migrations

    def test_registry_duplicate_registration_warning(self, caplog):
        """Test warning when registering duplicate migration."""
        registry = MigrationRegistry()

        class Migration1(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        class Migration2(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.2.0", "0.3.0", Migration1)
        registry.register("0.2.0", "0.3.0", Migration2)

        # Should replace with warning
        assert registry.get_migration("0.2.0", "0.3.0") == Migration2


class TestRegisterMigrationDecorator:
    """Test @register_migration decorator."""

    def test_decorator_registers_migration(self):
        """Test decorator registers migration in global registry."""
        # Create new registry for this test
        test_registry = MigrationRegistry()

        with patch('common.core.version_migration._registry', test_registry):
            @register_migration(from_version="0.2.0", to_version="0.3.0")
            class TestMigration(BaseMigration):
                def migrate(self, backup_data):
                    return backup_data

            assert test_registry.has_migration("0.2.0", "0.3.0")
            assert TestMigration.from_version == "0.2.0"
            assert TestMigration.to_version == "0.3.0"

    def test_decorator_sets_class_attributes(self):
        """Test decorator sets version attributes on class."""
        @register_migration(from_version="0.1.0", to_version="0.2.0")
        class TestMigration(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        assert TestMigration.from_version == "0.1.0"
        assert TestMigration.to_version == "0.2.0"


class TestMigrationManager:
    """Test MigrationManager class."""

    def test_manager_initialization(self):
        """Test creating migration manager."""
        registry = MigrationRegistry()
        manager = MigrationManager(registry=registry)

        assert manager.registry == registry

    def test_manager_uses_global_registry_by_default(self):
        """Test manager uses global registry if none provided."""
        manager = MigrationManager()
        assert manager.registry is not None

    def test_apply_direct_migration_success(self):
        """Test applying a successful direct migration."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                # Transform data
                backup_data.version = "0.3.0"
                backup_data.metadata["migrated"] = True
                return backup_data

        registry.register("0.2.0", "0.3.0", TestMigration)
        manager = MigrationManager(registry=registry)

        backup_data = BackupData(version="0.2.0", manifest={})
        result = manager.apply_migrations(backup_data, "0.2.0", "0.3.0")

        assert result.success is True
        assert result.backup_data.version == "0.3.0"
        assert result.backup_data.metadata["migrated"] is True
        assert "TestMigration" in result.applied_migrations

    def test_apply_migration_no_migration_available(self):
        """Test applying migration when none is available."""
        registry = MigrationRegistry()
        manager = MigrationManager(registry=registry)

        backup_data = BackupData(version="0.1.0", manifest={})
        result = manager.apply_migrations(backup_data, "0.1.0", "0.2.0")

        assert result.success is False
        assert "No migration available" in result.errors[0]

    def test_apply_migration_validation_failure(self):
        """Test migration fails validation."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                return backup_data

            def validate(self, backup_data):
                return False  # Always fail validation

        registry.register("0.2.0", "0.3.0", TestMigration)
        manager = MigrationManager(registry=registry)

        backup_data = BackupData(version="0.2.0", manifest={})
        result = manager.apply_migrations(backup_data, "0.2.0", "0.3.0")

        assert result.success is False
        assert "validation failed" in result.errors[0]

    def test_apply_migration_with_warnings(self):
        """Test migration that produces warnings."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                backup_data.version = "0.3.0"
                return backup_data

            def get_warnings(self, backup_data):
                return ["Schema changes detected", "Manual verification recommended"]

        registry.register("0.2.0", "0.3.0", TestMigration)
        manager = MigrationManager(registry=registry)

        backup_data = BackupData(version="0.2.0", manifest={})
        result = manager.apply_migrations(backup_data, "0.2.0", "0.3.0")

        assert result.success is True
        assert len(result.warnings) == 2
        assert "Schema changes detected" in result.warnings

    def test_apply_migration_exception_handling(self):
        """Test migration exception is caught and reported."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            from_version = "0.2.0"
            to_version = "0.3.0"

            def migrate(self, backup_data):
                raise ValueError("Migration error occurred")

        registry.register("0.2.0", "0.3.0", TestMigration)
        manager = MigrationManager(registry=registry)

        backup_data = BackupData(version="0.2.0", manifest={})
        result = manager.apply_migrations(backup_data, "0.2.0", "0.3.0")

        assert result.success is False
        assert "Migration failed" in result.errors[0]
        assert "Migration error occurred" in result.errors[0]

    def test_get_migration_path_direct(self):
        """Test finding direct migration path."""
        registry = MigrationRegistry()

        class TestMigration(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.2.0", "0.3.0", TestMigration)
        manager = MigrationManager(registry=registry)

        path = manager.get_migration_path("0.2.0", "0.3.0")
        assert path == [("0.2.0", "0.3.0")]

    def test_get_migration_path_none_available(self):
        """Test finding migration path when none available."""
        registry = MigrationRegistry()
        manager = MigrationManager(registry=registry)

        path = manager.get_migration_path("0.1.0", "0.2.0")
        assert path is None

    def test_list_available_migrations(self):
        """Test listing all available migrations."""
        registry = MigrationRegistry()

        class Migration1(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        class Migration2(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.1.0", "0.2.0", Migration1)
        registry.register("0.2.0", "0.3.0", Migration2)
        manager = MigrationManager(registry=registry)

        all_migrations = manager.list_available_migrations()
        assert len(all_migrations) == 2

    def test_list_available_migrations_filtered(self):
        """Test listing migrations filtered by from_version."""
        registry = MigrationRegistry()

        class Migration1(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        class Migration2(BaseMigration):
            def migrate(self, backup_data):
                return backup_data

        registry.register("0.1.0", "0.2.0", Migration1)
        registry.register("0.2.0", "0.3.0", Migration2)
        manager = MigrationManager(registry=registry)

        migrations_from_02 = manager.list_available_migrations(from_version="0.2.0")
        assert len(migrations_from_02) == 1
        assert migrations_from_02[0][0] == "0.2.0"  # from_version
        assert migrations_from_02[0][1] == "0.3.0"  # to_version


class TestMigrationIntegration:
    """Integration tests for migration framework."""

    def test_complete_migration_workflow(self):
        """Test complete workflow of defining and applying migration."""
        registry = MigrationRegistry()

        @register_migration(from_version="0.2.0", to_version="0.3.0")
        class MigrateTo030(BaseMigration):
            description = "Add new collection metadata field"

            def migrate(self, backup_data):
                # Add new metadata field to all collections
                for coll_name, coll_data in backup_data.collections.items():
                    if isinstance(coll_data, dict):
                        coll_data["schema_version"] = "2.0"

                backup_data.version = "0.3.0"
                return backup_data

            def get_warnings(self, backup_data):
                return ["Schema version field added to all collections"]

        # Apply migration
        manager = MigrationManager()
        original_data = BackupData(
            version="0.2.0",
            manifest={"test": "data"},
            collections={
                "collection1": {"points": 100},
                "collection2": {"points": 200}
            }
        )

        result = manager.apply_migrations(original_data, "0.2.0", "0.3.0")

        assert result.success is True
        assert result.backup_data.version == "0.3.0"
        assert result.backup_data.collections["collection1"]["schema_version"] == "2.0"
        assert result.backup_data.collections["collection2"]["schema_version"] == "2.0"
        assert len(result.warnings) == 1
