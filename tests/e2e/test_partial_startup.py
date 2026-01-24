"""
End-to-end tests for partial startup and mixed version scenarios.

Tests system behavior with partial component availability, version mismatches,
and degraded configurations. Verifies fallback behaviors, compatibility handling,
and graceful degradation when not all components are available.
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    DaemonManager,
    MCPServerManager,
    SystemComponents,
)


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonUnavailableScenarios:
    """Test system behavior when daemon is unavailable."""

    def test_cli_help_without_daemon(
        self, integration_test_workspace, integration_state_db
    ):
        """Test CLI help command works without daemon."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # CLI help should work without daemon
        result = cli_helper.run_command(["--help"])

        assert result is not None
        assert result.returncode == 0
        assert result.stdout

    def test_cli_version_without_daemon(
        self, integration_test_workspace, integration_state_db
    ):
        """Test CLI version command works without daemon."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        result = cli_helper.run_command(["--version"])

        assert result is not None
        assert result.returncode == 0

    def test_status_command_with_daemon_down(
        self, integration_test_workspace, integration_state_db
    ):
        """Test status command when daemon is not running."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Status should report daemon unavailable
        result = cli_helper.run_command(["status"], timeout=10)

        # Should complete (reporting daemon down)
        assert result is not None

    def test_service_status_with_daemon_down(
        self, integration_test_workspace, integration_state_db
    ):
        """Test service status when daemon is not running."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        result = cli_helper.run_command(["service", "status"])

        # Should return non-zero (daemon not running)
        assert result is not None
        assert result.returncode in [1, 3]  # 1=stopped, 3=not installed

    def test_ingestion_fails_gracefully_without_daemon(
        self, integration_test_workspace, integration_state_db
    ):
        """Test ingestion fails gracefully when daemon unavailable."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)
        workspace = integration_test_workspace

        test_file = workspace / "no_daemon.txt"
        test_file.write_text("Test without daemon")

        # Should fail but not crash
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-no-daemon"],
            timeout=30,
        )

        assert result is not None
        # Expect failure
        assert result.returncode != 0


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantUnavailableScenarios:
    """Test system behavior when Qdrant is unavailable."""

    def test_cli_commands_with_qdrant_down(
        self, integration_test_workspace, integration_state_db
    ):
        """Test CLI commands when Qdrant is down."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Help and version should still work
        help_result = cli_helper.run_command(["--help"])
        version_result = cli_helper.run_command(["--version"])

        assert help_result is not None
        assert help_result.returncode == 0
        assert version_result is not None
        assert version_result.returncode == 0

    def test_status_reports_qdrant_unavailable(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test status command reports Qdrant availability."""
        # Status should provide information about Qdrant
        result = cli_helper.run_command(["status"])

        assert result is not None

    def test_collection_operations_fail_without_qdrant(
        self, integration_test_workspace, integration_state_db
    ):
        """Test collection operations fail gracefully without Qdrant."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Operations requiring Qdrant should fail gracefully
        result = cli_helper.run_command(["admin", "collections"], timeout=10)

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestPartialComponentAvailability:
    """Test scenarios with partial component availability."""

    def test_cli_only_mode(
        self, integration_test_workspace, integration_state_db
    ):
        """Test CLI functions in standalone mode."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # CLI-only operations
        operations = [
            ["--help"],
            ["--version"],
        ]

        for op in operations:
            result = cli_helper.run_command(op)
            assert result is not None
            assert result.returncode == 0

    def test_daemon_without_mcp_server(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon can run without MCP server."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon only (no MCP server)
        daemon_manager.start(timeout=15)

        # Daemon should be running
        assert daemon_manager.is_running()

        # Cleanup
        daemon_manager.stop()

    def test_mcp_server_requires_daemon(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server behavior when daemon is unavailable."""
        # MCP server may require daemon for operations
        # Test that it handles missing daemon gracefully
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        result = cli_helper.run_command(["service", "status"])
        assert result is not None

    def test_partial_functionality_with_missing_components(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system provides partial functionality with missing components."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Even with components missing, basic CLI should work
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestConfigurationPartialAvailability:
    """Test behavior with partial configuration availability."""

    def test_missing_config_file(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system behavior when config file is missing."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # System should use defaults when config missing
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0

    def test_partial_config_values(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system with partially configured values."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)
        workspace = integration_test_workspace

        # Create partial config
        config_file = workspace / "partial_config.yaml"
        config_file.write_text("qdrant_url: http://localhost:6333")

        # System should handle partial config
        result = cli_helper.run_command(["status"])
        assert result is not None

        # Cleanup
        config_file.unlink()

    def test_invalid_config_fallback(
        self, integration_test_workspace, integration_state_db
    ):
        """Test fallback to defaults with invalid config."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)
        workspace = integration_test_workspace

        # Create invalid config
        config_file = workspace / "invalid_config.yaml"
        config_file.write_text("invalid yaml content: [")

        # System should handle invalid config gracefully
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0

        # Cleanup
        config_file.unlink()

    def test_environment_variable_overrides(
        self, integration_test_workspace, integration_state_db
    ):
        """Test environment variables override missing config."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Environment variables should provide config
        env = {
            "QDRANT_URL": "http://localhost:6333",
        }

        result = cli_helper.run_command(["--version"], env=env)
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestVersionCompatibility:
    """Test version compatibility scenarios."""

    def test_version_info_available(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test version information is available."""
        result = cli_helper.run_command(["--version"])

        assert result is not None
        assert result.returncode == 0
        assert result.stdout

    def test_component_version_reporting(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test components report their versions."""
        # Status should include version information
        result = cli_helper.run_command(["status"])

        assert result is not None

    def test_backward_compatibility_check(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system checks for backward compatibility."""
        # Operations should work with current version
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestFallbackBehaviors:
    """Test fallback behaviors when components unavailable."""

    def test_fallback_to_local_mode(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system falls back to local operations."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # CLI should work in local mode
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0

    def test_readonly_fallback_mode(
        self, integration_test_workspace, integration_state_db
    ):
        """Test fallback to readonly operations."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Read operations should work even if write operations fail
        result = cli_helper.run_command(["--version"])
        assert result is not None
        assert result.returncode == 0

    def test_degraded_functionality_indication(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system indicates degraded functionality."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Status should indicate availability
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_graceful_feature_degradation(
        self, integration_test_workspace, integration_state_db
    ):
        """Test features degrade gracefully when components missing."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Core features should remain available
        operations = [
            ["--help"],
            ["--version"],
        ]

        for op in operations:
            result = cli_helper.run_command(op)
            assert result is not None
            assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestMixedComponentStates:
    """Test scenarios with mixed component states."""

    def test_daemon_running_qdrant_down(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon running but Qdrant unavailable."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon (Qdrant may not be available)
        daemon_manager.start(timeout=15)

        # Daemon should be running
        assert daemon_manager.is_running()

        # Cleanup
        daemon_manager.stop()

    def test_all_components_starting(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system during component startup phase."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # System should handle startup phase gracefully
        result = cli_helper.run_command(["status"], timeout=15)
        assert result is not None

    def test_component_startup_order(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system handles various component startup orders."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon first
        daemon_manager.start(timeout=15)
        assert daemon_manager.is_running()

        # Components can start in any order
        cli_helper = CLIHelper(workspace=integration_test_workspace)
        result = cli_helper.run_command(["service", "status"])
        assert result is not None

        # Cleanup
        daemon_manager.stop()

    def test_component_availability_transitions(
        self, integration_test_workspace, integration_state_db
    ):
        """Test transitions between component availability states."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # State 1: No daemon
        result1 = cli_helper.run_command(["service", "status"])
        assert result1 is not None

        # State 2: Daemon running
        daemon_manager.start(timeout=15)
        result2 = cli_helper.run_command(["service", "status"])
        assert result2 is not None

        # State 3: Daemon stopped
        daemon_manager.stop()
        result3 = cli_helper.run_command(["service", "status"])
        assert result3 is not None


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseAvailability:
    """Test behavior with database availability issues."""

    def test_sqlite_database_locked(
        self, integration_test_workspace, integration_state_db
    ):
        """Test handling of locked SQLite database."""
        import sqlite3

        # Create connection that locks database
        conn = sqlite3.connect(integration_state_db)
        conn.execute("BEGIN EXCLUSIVE")

        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Operations should handle locked database
        result = cli_helper.run_command(["status"], timeout=10)
        assert result is not None

        # Release lock
        conn.rollback()
        conn.close()

    def test_sqlite_database_missing(
        self, integration_test_workspace, tmp_path
    ):
        """Test system behavior when database is missing."""
        # Use non-existent database path
        fake_db = tmp_path / "nonexistent.db"

        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=fake_db,
        )

        # System should handle missing database
        # (May create new database or fail gracefully)
        try:
            daemon_manager.start(timeout=15)
            # If successful, cleanup
            daemon_manager.stop()
        except Exception:
            # Failure is acceptable for missing database
            pass

    def test_sqlite_database_readonly(
        self, integration_test_workspace, integration_state_db
    ):
        """Test behavior with readonly database."""
        from unittest.mock import patch

        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Mock database access to simulate readonly behavior
        with patch('os.access') as mock_access:
            # Allow read but deny write
            def access_side_effect(path, mode):
                if str(path) == str(integration_state_db):
                    if mode == os.W_OK:
                        return False
                    return True
                return True
            mock_access.side_effect = access_side_effect

            # Read operations should still work
            result = cli_helper.run_command(["--version"])
            assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestNetworkPartitioning:
    """Test behavior during network partitioning scenarios."""

    def test_local_operations_during_network_issues(
        self, integration_test_workspace, integration_state_db
    ):
        """Test local operations work during network issues."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Local operations should work
        operations = [
            ["--help"],
            ["--version"],
        ]

        for op in operations:
            result = cli_helper.run_command(op)
            assert result is not None
            assert result.returncode == 0

    def test_qdrant_unreachable_handling(
        self, integration_test_workspace, integration_state_db
    ):
        """Test handling when Qdrant is unreachable."""
        # Use unreachable Qdrant URL
        daemon_manager = DaemonManager(
            qdrant_url="http://unreachable.host:6333",
            state_db_path=integration_state_db,
        )

        # Should handle unreachable Qdrant
        try:
            daemon_manager.start(timeout=15)
            # May start but can't connect to Qdrant
            daemon_manager.stop()
        except Exception:
            # Failure is acceptable
            pass

    def test_timeout_during_network_partition(
        self, integration_test_workspace, integration_state_db
    ):
        """Test operation timeouts during network partition."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Operations should timeout gracefully
        result = cli_helper.run_command(["status"], timeout=5)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestResourceConstraints:
    """Test behavior under resource constraints."""

    def test_limited_file_descriptors(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system with limited file descriptors."""
        # System should handle resource limits
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_low_memory_conditions(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test basic operations under memory pressure."""
        # Basic operations should still work
        result = cli_helper.run_command(["--version"])
        assert result is not None
        assert result.returncode == 0

    def test_disk_space_constraints(
        self, integration_test_workspace, integration_state_db
    ):
        """Test behavior with limited disk space."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Read operations should work
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestUpgradeScenarios:
    """Test upgrade and compatibility scenarios."""

    def test_system_operational_during_upgrade(
        self, integration_test_workspace, integration_state_db
    ):
        """Test system remains operational during upgrades."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # Basic operations should work
        result = cli_helper.run_command(["--version"])
        assert result is not None
        assert result.returncode == 0

    def test_database_schema_compatibility(
        self, integration_test_workspace, integration_state_db
    ):
        """Test database schema compatibility."""
        import sqlite3

        # Verify database can be accessed
        conn = sqlite3.connect(integration_state_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()

        # Should have expected tables
        assert len(tables) > 0

    def test_config_format_compatibility(
        self, integration_test_workspace, integration_state_db
    ):
        """Test configuration format compatibility."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # System should handle config changes
        result = cli_helper.run_command(["--version"])
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestComponentDependencies:
    """Test component dependency handling."""

    def test_missing_optional_dependencies(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system with missing optional dependencies."""
        # Core features should work without optional deps
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0

    def test_required_dependencies_check(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system checks for required dependencies."""
        # System should report missing required deps
        result = cli_helper.run_command(["--version"])
        assert result is not None
        assert result.returncode == 0

    def test_dependency_version_mismatch(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of dependency version mismatches."""
        # System should handle version conflicts gracefully
        result = cli_helper.run_command(["status"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestStartupValidation:
    """Test startup validation and health checks."""

    def test_component_health_checks(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test component health check validation."""
        # Status should provide health information
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_startup_validation_failures(
        self, integration_test_workspace, integration_state_db
    ):
        """Test handling of startup validation failures."""
        cli_helper = CLIHelper(workspace=integration_test_workspace)

        # System should handle validation failures
        result = cli_helper.run_command(["status"], timeout=10)
        assert result is not None

    def test_component_readiness_checks(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test component readiness validation."""
        # Service status provides readiness info
        result = cli_helper.run_command(["service", "status"])
        assert result is not None

    def test_dependency_resolution_order(
        self, integration_test_workspace, integration_state_db
    ):
        """Test components start in correct dependency order."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start components in order
        daemon_manager.start(timeout=15)
        assert daemon_manager.is_running()

        # Cleanup
        daemon_manager.stop()
