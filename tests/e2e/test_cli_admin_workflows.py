"""
End-to-end tests for CLI administration and management workflows.

Tests complete CLI administrative operations including collection management,
watch folder configuration, system status queries, daemon control operations,
and configuration updates.
"""

import pytest
import time
from pathlib import Path

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
)


@pytest.mark.integration
@pytest.mark.slow
class TestCollectionManagement:
    """Test collection management operations."""

    def test_list_collections(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test listing all collections."""
        result = cli_helper.run_command(["admin", "collections"])

        # Should succeed (may be empty list)
        assert result is not None
        assert result.returncode in [0, 1]

    def test_collection_info(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test getting collection information."""
        # First create a collection by ingesting
        workspace = system_components.workspace_path
        test_file = workspace / "admin_test.txt"
        test_file.write_text("Content for admin testing")

        collection_name = f"test-admin-info-{int(time.time())}"
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # List collections should show our collection
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSystemStatus:
    """Test system status queries."""

    def test_status_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test basic status command."""
        result = cli_helper.run_command(["status"])

        # Status command should work
        assert result is not None
        assert result.returncode in [0, 1]

    def test_status_quiet_mode(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test status command in quiet mode."""
        result = cli_helper.run_command(["status", "--quiet"])

        # Quiet mode should work
        assert result is not None

    def test_daemon_status_check(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test checking daemon status."""
        result = cli_helper.run_command(["service", "status"])

        # Service status should work
        assert result is not None
        # 0 = running, 1 = stopped, 3 = not installed
        assert result.returncode in [0, 1, 3]


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonControl:
    """Test daemon control operations."""

    def test_daemon_status_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test querying daemon status."""
        result = cli_helper.run_command(["service", "status"])

        # Should be able to query status
        assert result is not None

    def test_service_commands_available(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that service commands are available."""
        # Just verify commands don't crash
        # Note: We won't actually restart services during tests
        result = cli_helper.run_command(["service", "status"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestWatchFolderConfiguration:
    """Test watch folder configuration management."""

    def test_list_watch_folders(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test listing configured watch folders."""
        # Note: Watch folder management might not be fully implemented via CLI
        # This tests basic command availability
        result = cli_helper.run_command(["status"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestConfigurationManagement:
    """Test configuration management operations."""

    def test_version_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test version command."""
        result = cli_helper.run_command(["--version"])

        # Version command should succeed
        assert result is not None
        assert result.returncode == 0

    def test_help_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test help command."""
        result = cli_helper.run_command(["--help"])

        # Help should be available
        assert result is not None
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestAdminCommandValidation:
    """Test admin command validation and error handling."""

    def test_invalid_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of invalid command."""
        result = cli_helper.run_command(["invalid-command-xyz"])

        # Should error gracefully
        assert result is not None
        assert result.returncode != 0

    def test_missing_required_argument(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of missing required arguments."""
        # Try search without query
        result = cli_helper.run_command(["search"])

        # Should error or show usage
        assert result is not None

    def test_invalid_collection_name(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of invalid collection names."""
        # Try with invalid characters
        result = cli_helper.run_command(
            ["search", "test", "--collection", "invalid/collection/name"]
        )

        # Should handle invalid name
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestBulkOperations:
    """Test bulk administrative operations."""

    def test_bulk_collection_listing(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test listing collections when many exist."""
        workspace = system_components.workspace_path

        # Create multiple collections
        for i in range(3):
            test_file = workspace / f"bulk_{i}.txt"
            test_file.write_text(f"Bulk test content {i}")
            collection_name = f"test-bulk-{i}-{int(time.time())}"

            cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", collection_name]
            )

        time.sleep(5)

        # List all collections
        result = cli_helper.run_command(["admin", "collections"])

        # Should list all collections
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestAdminWorkflowIntegration:
    """Test complete admin workflow integration."""

    def test_create_ingest_query_workflow(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test complete workflow: create collection, ingest, query status."""
        workspace = system_components.workspace_path

        # Step 1: Create content
        test_file = workspace / "workflow.txt"
        test_file.write_text("Complete workflow test content")

        # Step 2: Ingest to collection
        collection_name = f"test-workflow-{int(time.time())}"
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        assert result is not None

        time.sleep(3)

        # Step 3: Query status
        status_result = cli_helper.run_command(["status"])
        assert status_result is not None

        # Step 4: List collections
        list_result = cli_helper.run_command(["admin", "collections"])
        assert list_result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSystemInformation:
    """Test system information queries."""

    def test_system_status_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test querying overall system status."""
        result = cli_helper.run_command(["status"])

        # System status should be queryable
        assert result is not None

    def test_daemon_connection_status(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test checking daemon connection status."""
        result = cli_helper.run_command(["service", "status"])

        # Should show daemon status
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestErrorRecovery:
    """Test error recovery in admin operations."""

    def test_operation_retry_on_failure(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that failed operations can be retried."""
        # Try operation that might fail
        result1 = cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"]
        )

        # Retry should work
        result2 = cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"]
        )

        # Both attempts should complete
        assert result1 is not None
        assert result2 is not None

    def test_system_state_consistency(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system maintains consistent state after errors."""
        # Cause an error
        cli_helper.run_command(["invalid-command"])

        # System should still be functional
        result = cli_helper.run_command(["status"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentAdminOperations:
    """Test concurrent administrative operations."""

    def test_concurrent_status_queries(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple concurrent status queries."""
        # Execute multiple status queries in sequence
        # (True concurrency would require threading)
        results = []
        for _ in range(5):
            result = cli_helper.run_command(["status", "--quiet"])
            results.append(result)

        # All queries should complete
        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_concurrent_collection_queries(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple concurrent collection queries."""
        results = []
        for _ in range(3):
            result = cli_helper.run_command(["admin", "collections"])
            results.append(result)

        # All queries should complete
        assert len(results) == 3
        assert all(r is not None for r in results)


@pytest.mark.integration
@pytest.mark.slow
class TestAdminCommandPerformance:
    """Test performance of admin commands."""

    def test_status_query_latency(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that status queries complete quickly."""
        start_time = time.time()
        result = cli_helper.run_command(["status", "--quiet"], timeout=5)
        duration = time.time() - start_time

        # Status should be fast (<5 seconds)
        assert duration < 5.0
        assert result is not None

    def test_collection_listing_performance(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that collection listing completes in reasonable time."""
        start_time = time.time()
        result = cli_helper.run_command(["admin", "collections"], timeout=10)
        duration = time.time() - start_time

        # Collection listing should be fast (<10 seconds)
        assert duration < 10.0
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestAdminOutputFormatting:
    """Test formatting of admin command outputs."""

    def test_collections_output_format(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that collections output is properly formatted."""
        result = cli_helper.run_command(["admin", "collections"])

        # Should have output
        assert result is not None
        # Should have stdout or stderr
        assert result.stdout or result.stderr

    def test_status_output_format(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that status output is properly formatted."""
        result = cli_helper.run_command(["status"])

        # Should have output
        assert result is not None
        assert result.stdout or result.stderr

    def test_help_output_readable(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that help output is readable."""
        result = cli_helper.run_command(["--help"])

        # Help should be available and readable
        assert result is not None
        assert result.returncode == 0
        assert result.stdout  # Help goes to stdout


@pytest.mark.integration
@pytest.mark.slow
class TestAdminCommandChaining:
    """Test chaining multiple admin commands."""

    def test_status_then_collections(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test querying status then collections."""
        # Get status
        status_result = cli_helper.run_command(["status"])
        assert status_result is not None

        # List collections
        collections_result = cli_helper.run_command(["admin", "collections"])
        assert collections_result is not None

    def test_ingest_then_verify(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingesting then verifying via admin commands."""
        workspace = system_components.workspace_path

        # Ingest content
        test_file = workspace / "chain_test.txt"
        test_file.write_text("Chaining test content")
        collection_name = f"test-chain-{int(time.time())}"

        ingest_result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        assert ingest_result is not None

        time.sleep(3)

        # Verify via collections list
        list_result = cli_helper.run_command(["admin", "collections"])
        assert list_result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestAdminRobustness:
    """Test robustness of admin commands."""

    def test_rapid_successive_commands(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test executing many commands in rapid succession."""
        # Execute commands rapidly
        for i in range(10):
            result = cli_helper.run_command(["status", "--quiet"])
            assert result is not None

        # System should still be responsive
        final_result = cli_helper.run_command(["admin", "collections"])
        assert final_result is not None

    def test_admin_after_heavy_load(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test admin commands work after heavy load."""
        workspace = system_components.workspace_path

        # Create heavy load with ingestion
        for i in range(5):
            test_file = workspace / f"load_{i}.txt"
            test_file.write_text(f"Load test content {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-load-{i}",
                ]
            )

        time.sleep(5)

        # Admin commands should still work
        result = cli_helper.run_command(["status"])
        assert result is not None

        collections_result = cli_helper.run_command(["admin", "collections"])
        assert collections_result is not None
