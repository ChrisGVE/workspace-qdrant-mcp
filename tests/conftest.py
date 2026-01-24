"""
Root pytest configuration and shared fixtures.

This conftest provides:
- Path configuration for imports
- Pytest configuration hooks
- Integration of shared fixtures from tests/shared/
- Test categorization and marker management
- Permission diagnostic utilities
"""

import os
import stat
import sys
from pathlib import Path

import pytest


# Permission diagnostic constants (read-only, for diagnostics only)
_PROJECT_ROOT = Path(__file__).parent.parent
_EXPECTED_DIR_MODE = 0o755  # drwxr-xr-x
_ORIGINAL_PERMISSIONS: dict[str, int] = {}


def _get_permission_mode(path: Path) -> int | None:
    """Get the permission mode of a path, or None if it doesn't exist."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except (OSError, PermissionError):
        return None


def _check_and_restore_project_permissions() -> bool:
    """
    Check project root permissions for diagnostic purposes only.

    NOTE: This function NO LONGER modifies permissions. It only logs status.
    Returns True if permissions are OK, False if they appear incorrect.
    """
    # Check project root
    current_mode = _get_permission_mode(_PROJECT_ROOT)

    if current_mode is None:
        print(f"DIAGNOSTIC: Cannot access project root: {_PROJECT_ROOT}")
        return False

    # Check if permissions match expected or original
    original_mode = _ORIGINAL_PERMISSIONS.get("project_root", _EXPECTED_DIR_MODE)

    if current_mode != original_mode and current_mode != _EXPECTED_DIR_MODE:
        print(f"DIAGNOSTIC: Project root permissions differ from expected!")
        print(f"  Current: {oct(current_mode)} ({stat.filemode(current_mode | stat.S_IFDIR)})")
        print(f"  Expected: {oct(original_mode)} ({stat.filemode(original_mode | stat.S_IFDIR)})")
        print(f"  NOTE: No automatic restoration - permissions left as-is")
        return False

    return True

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Quarantined tests - files with import errors or missing dependencies
# These need to be fixed before being re-enabled
# Quarantined on 2026-01-18 during Phase 1 stabilization
collect_ignore = [
    # CRITICAL: Tests causing permission issues or infinite hangs (2026-01-24)
    # These tests blindly execute CLI commands and async functions without proper mocking
    # causing os.chmod calls that break project folder permissions or infinite blocking
    "unit/test_complete_100_percent_coverage.py",
    "unit/test_final_100_percent_push.py",
    "unit/test_final_coverage_push.py",
    "unit/test_final_100_percent_comprehensive.py",
    "unit/test_maximum_coverage_push.py",
    # Tests referencing missing workspace_qdrant_mcp.utils module
    "unit/test_config_validator.py",
    "unit/test_config_validator_comprehensive.py",
    "unit/test_project_detection.py",
    "unit/test_project_collection_validator.py",
    "unit/test_os_directories.py",
    # Tests referencing missing workspace_qdrant_mcp.web module
    "unit/test_web_cache.py",
    "unit/test_web_crawler.py",
    "unit/test_web_extractor.py",
    "unit/test_web_integration.py",
    "unit/test_web_links.py",
    # Tests referencing missing analytics module
    "unit/test_analytics_collector.py",
    "unit/test_analytics_dashboard.py",
    "unit/test_analytics_privacy.py",
    "unit/test_analytics_storage.py",
    # Tests referencing missing ML modules
    "unit/test_ml_config.py",
    "unit/test_ml_system_integration.py",
    "unit/test_model_monitor.py",
    "unit/test_model_registry.py",
    "unit/test_training_pipeline.py",
    # Tests with missing common.core imports (Config, ChunkingStrategy, etc.)
    "unit/test_common_collections_comprehensive.py",
    "unit/test_common_core_client_comprehensive.py",
    "unit/test_common_core_hybrid_search_comprehensive.py",
    "unit/test_common_core_memory_comprehensive.py",
    "unit/test_common_memory_manager_comprehensive.py",
    "unit/test_core_embeddings.py",
    "unit/test_embeddings_comprehensive.py",
    "unit/test_memory_tools_comprehensive.py",
    "unit/test_tools_memory.py",
    "unit/test_mcp_error_handling_edge_cases.py",
    "unit/test_grpc_performance_validation.py",
    # Tests with missing modules
    "unit/test_consistency_checker.py",
    "unit/test_cross_reference_validator.py",
    "unit/test_daemon_identifier.py",
    "unit/test_deployment_manager.py",
    "unit/test_deployment_pipeline.py",
    "unit/test_documentation_ast_parser.py",
    "unit/test_documentation_coverage_analyzer.py",
    "unit/test_versioning.py",
    "unit/test_admin_cli_comprehensive.py",
    # Testing framework tests
    "unit/testing/",
    # Integration/E2E tests with missing dependencies
    "cli/test_comprehensive_cli.py",
    "e2e/test_24hour_stability.py",
    "e2e/test_performance_regression.py",
    "examples/test_mcp_ai_evaluation_example.py",
    "functional/test_mcp_protocol_compliance.py",
    "functional/test_daemon_lifecycle_integration.py",
    "legacy/",
    "test_collection_naming.py",
    "test_collection_types.py",
    "test_config.py",
    "test_llm_access_control.py",
    "test_monitoring_integration.py",
    "test_production_deployment.py",
    "test_testcontainers_integration.py",
    "test_testcontainers_setup.py",
]

# Import shared fixtures to make them available to all tests
pytest_plugins = [
    "tests.shared.fixtures",
]


def pytest_configure(config):
    """
    Configure pytest for workspace-qdrant-mcp testing.

    Registers custom markers and sets up test environment.
    Records original permissions for diagnostic purposes only.
    """
    # Record original project root permissions for diagnostic purposes
    original_mode = _get_permission_mode(_PROJECT_ROOT)
    if original_mode is not None:
        _ORIGINAL_PERMISSIONS["project_root"] = original_mode
        print(f"\nDIAGNOSTIC: Project root permissions: {oct(original_mode)}")
    else:
        print(f"\nDIAGNOSTIC: Could not read project root permissions")

    # Set environment variable to indicate testing mode
    os.environ["WQM_TESTING"] = "true"
    # Note: PYTEST_CURRENT_TEST is managed by pytest internally, don't set it manually

    # Register additional markers (beyond those in pyproject.toml)
    config.addinivalue_line(
        "markers",
        "isolated_container: Test uses isolated Qdrant container",
    )
    config.addinivalue_line(
        "markers",
        "shared_container: Test uses shared Qdrant container",
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to apply markers automatically.

    This hook automatically applies markers based on:
    - Test file location (daemon/, mcp_server/, cli/)
    - Test category subdirectory (nominal/, edge/, stress/)
    - Test name patterns
    """
    for item in items:
        # Get test file path relative to tests directory
        test_path = Path(item.fspath).relative_to(Path(__file__).parent)
        parts = test_path.parts

        # Apply domain markers based on directory
        if len(parts) > 0:
            domain = parts[0]
            if domain == "daemon":
                item.add_marker(pytest.mark.daemon)
            elif domain == "mcp_server":
                item.add_marker(pytest.mark.mcp_server)
            elif domain == "cli":
                item.add_marker(pytest.mark.cli)

        # Apply category markers based on subdirectory
        if len(parts) > 1:
            category = parts[1]
            if category == "nominal":
                item.add_marker(pytest.mark.nominal)
            elif category == "edge":
                item.add_marker(pytest.mark.edge)
            elif category == "stress":
                item.add_marker(pytest.mark.stress)
                item.add_marker(pytest.mark.slow)  # Stress tests are slow

        # Apply markers based on test name patterns
        test_name = item.name.lower()

        if "stress" in test_name or "load" in test_name:
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.slow)

        if "edge" in test_name or "invalid" in test_name or "error" in test_name:
            item.add_marker(pytest.mark.edge)

        if (
            "container" in test_name
            or "qdrant" in test_name
            or "isolated" in item.fixturenames
        ):
            item.add_marker(pytest.mark.requires_qdrant)

        if "daemon" in test_name or "grpc" in test_name:
            item.add_marker(pytest.mark.requires_rust)


def pytest_runtest_teardown(item, nextitem):
    """
    Teardown hook called after each test.

    Performs cleanup tasks but does NOT modify filesystem permissions.
    """
    # NOTE: Permission restoration logic removed - tests should not modify permissions
    pass


def pytest_sessionfinish(session, exitstatus):
    """
    Session finish hook called at the end of the test session.

    Logs diagnostic information but does NOT modify permissions.
    """
    print("\n" + "-" * 70)
    print("Test session complete - permission diagnostic check...")

    # Check (but don't restore) project root permissions
    _check_and_restore_project_permissions()

    print("Diagnostic check complete.")
    print("-" * 70)


def pytest_runtest_setup(item):
    """
    Setup hook called before each test.

    Handles conditional test skipping based on requirements.
    """
    # Skip tests requiring Docker if Docker is not available
    if item.get_closest_marker("requires_docker"):
        try:
            import docker

            client = docker.from_env()
            client.ping()
        except Exception:
            pytest.skip("Docker not available")

    # Skip tests requiring Rust daemon if not available
    if item.get_closest_marker("requires_rust"):
        # Check if daemon service is available
        try:
            import subprocess
            result = subprocess.run(
                ["uv", "run", "wqm", "service", "status"],
                capture_output=True,
                timeout=10
            )
            # If service status command works, daemon is available
            # (Even if daemon is not running, the service management exists)
            if result.returncode not in [0, 1, 3]:
                # Error codes 0, 1, 3 typically mean service management works
                # Other codes suggest service management not available
                pytest.skip("Daemon service not available")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pytest.skip("Daemon service not available")


@pytest.fixture(scope="session", autouse=True)
def test_environment_setup():
    """
    Session-wide test environment setup.

    Runs once at the start of the test session to prepare the environment.
    Logs permission status for diagnostics but does NOT modify permissions.
    """
    print("\n" + "=" * 70)
    print("Starting workspace-qdrant-mcp test session")
    print("=" * 70)

    # Log project root permissions at session start (diagnostic only)
    current_mode = _get_permission_mode(_PROJECT_ROOT)
    if current_mode is not None:
        print(f"DIAGNOSTIC: Project root permissions at session start: {oct(current_mode)}")
        # Store in _ORIGINAL_PERMISSIONS if not already set
        if "project_root" not in _ORIGINAL_PERMISSIONS:
            _ORIGINAL_PERMISSIONS["project_root"] = current_mode

        # Log if permissions differ from expected (but don't fix)
        if current_mode != _EXPECTED_DIR_MODE:
            print(f"DIAGNOSTIC: Project root permissions differ from expected {oct(_EXPECTED_DIR_MODE)}")
            print(f"            No automatic modification - leaving as-is")

    # Set additional environment variables for testing
    os.environ["WQM_LOG_LEVEL"] = os.getenv("WQM_LOG_LEVEL", "WARNING")
    os.environ["FASTEMBED_CACHE_DIR"] = str(Path.home() / ".cache" / "fastembed_test")

    yield

    print("\n" + "=" * 70)
    print("Cleaning up test session")
    print("=" * 70)

    # Cleanup environment
    os.environ.pop("WQM_TESTING", None)
    # Note: PYTEST_CURRENT_TEST is managed by pytest internally, don't remove it manually

    # Final permission check (diagnostic only, no restoration)
    _check_and_restore_project_permissions()


@pytest.fixture
def project_root() -> Path:
    """Provide path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def tests_root() -> Path:
    """Provide path to tests directory."""
    return Path(__file__).parent


@pytest.fixture
def src_root() -> Path:
    """Provide path to source directory."""
    return Path(__file__).parent.parent / "src" / "python"


# NOTE: permission_guard and safe_chmod fixtures removed
# Tests should not modify filesystem permissions
# Use mocking or temporary directories if permission testing is needed
