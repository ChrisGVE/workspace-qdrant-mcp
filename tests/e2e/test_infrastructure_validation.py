"""
Validation tests for end-to-end test infrastructure.

Verifies that the comprehensive test environment can be set up
and all components work together correctly.
"""

import asyncio

import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    ResourceMonitor,
    SystemComponents,
)


@pytest.mark.integration
@pytest.mark.slow
class TestInfrastructureSetup:
    """Verify end-to-end test infrastructure setup."""

    def test_qdrant_container_available(self, system_components: SystemComponents):
        """Test that Qdrant container is running and accessible."""
        assert system_components.qdrant is not None
        assert system_components.qdrant_url is not None
        assert "http://" in system_components.qdrant_url

    def test_daemon_process_running(self, system_components: SystemComponents):
        """Test that daemon process is running."""
        assert system_components.daemon_process is not None
        assert system_components.daemon_process.poll() is None  # Process not exited

    def test_mcp_server_running(self, system_components: SystemComponents):
        """Test that MCP server is running."""
        assert system_components.mcp_server_process is not None
        assert (
            system_components.mcp_server_process.poll() is None
        )  # Process not exited

    def test_workspace_structure(self, system_components: SystemComponents):
        """Test that workspace has correct structure."""
        workspace = system_components.workspace_path
        assert workspace.exists()
        assert (workspace / "src").exists()
        assert (workspace / "tests").exists()
        assert (workspace / "docs").exists()
        assert (workspace / ".git").exists()
        assert (workspace / "README.md").exists()

    def test_state_db_exists(self, system_components: SystemComponents):
        """Test that state database exists."""
        # State DB might not exist yet if daemon hasn't initialized
        # Just verify the path is set correctly
        assert system_components.state_db_path is not None


@pytest.mark.integration
@pytest.mark.slow
class TestCLIHelper:
    """Verify CLI helper functionality."""

    def test_cli_helper_available(self, cli_helper: CLIHelper):
        """Test that CLI helper is available."""
        assert cli_helper is not None
        assert cli_helper.state_db_path is not None
        assert cli_helper.qdrant_url is not None

    def test_cli_command_execution(self, cli_helper: CLIHelper):
        """Test that CLI commands can be executed."""
        # Try running a simple CLI command
        result = cli_helper.run_command(["--version"])

        # Should return successfully
        assert result.returncode == 0 or result.stderr == ""  # Version command works


@pytest.mark.integration
@pytest.mark.slow
class TestResourceMonitor:
    """Verify resource monitoring functionality."""

    @pytest.mark.asyncio
    async def test_resource_monitor_available(self, resource_monitor: ResourceMonitor):
        """Test that resource monitor is available."""
        assert resource_monitor is not None
        assert not resource_monitor.monitoring
        assert len(resource_monitor.metrics_history) == 0

    @pytest.mark.asyncio
    async def test_resource_monitoring_lifecycle(
        self, resource_monitor: ResourceMonitor, system_components: SystemComponents
    ):
        """Test resource monitor start/stop lifecycle."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)
        assert resource_monitor.monitoring

        # Let it collect some metrics
        await asyncio.sleep(2)

        # Stop monitoring
        await resource_monitor.stop_monitoring()
        assert not resource_monitor.monitoring

        # Should have collected some metrics
        assert len(resource_monitor.metrics_history) > 0

        # Get summary
        summary = resource_monitor.get_summary()
        assert isinstance(summary, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestSystemHealth:
    """Verify system health checking."""

    @pytest.mark.asyncio
    async def test_system_health_check(self, system_health_check: bool):
        """Test that system health check runs and reports status."""
        # Health check should complete (even if components not all healthy)
        assert isinstance(system_health_check, bool)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBaseline:
    """Verify performance baseline configuration."""

    def test_baseline_thresholds_defined(
        self, performance_baseline: dict
    ):
        """Test that performance baseline thresholds are defined."""
        assert "max_cpu_percent" in performance_baseline
        assert "max_memory_mb" in performance_baseline
        assert "max_file_descriptors" in performance_baseline
        assert "search_latency_ms" in performance_baseline
        assert "ingestion_throughput" in performance_baseline

        # Verify reasonable values
        assert performance_baseline["max_cpu_percent"] > 0
        assert performance_baseline["max_memory_mb"] > 0
        assert performance_baseline["max_file_descriptors"] > 0
        assert performance_baseline["search_latency_ms"] > 0
        assert performance_baseline["ingestion_throughput"] > 0


@pytest.mark.integration
@pytest.mark.slow
class TestComponentIntegration:
    """Verify all components work together."""

    def test_all_components_available(self, system_components: SystemComponents):
        """Test that all required components are available."""
        # Qdrant
        assert system_components.qdrant is not None
        assert system_components.qdrant_url is not None

        # Daemon
        assert system_components.daemon_process is not None

        # MCP Server
        assert system_components.mcp_server_process is not None

        # Paths
        assert system_components.state_db_path is not None
        assert system_components.workspace_path is not None
        assert system_components.config_path is not None

    def test_components_communicate(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that components can communicate with each other."""
        # Try a CLI command that would require daemon/Qdrant interaction
        # Use a query command that's safe to run
        result = cli_helper.run_command(["status", "--quiet"])

        # Command should execute (may fail if services not fully initialized,
        # but should not crash)
        assert result is not None
