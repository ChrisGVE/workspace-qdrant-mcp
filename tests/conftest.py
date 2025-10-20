"""
Root pytest configuration and shared fixtures.

This conftest provides:
- Path configuration for imports
- Pytest configuration hooks
- Integration of shared fixtures from tests/shared/
- Test categorization and marker management
"""

import sys
import os
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import shared fixtures to make them available to all tests
pytest_plugins = [
    "tests.shared.fixtures",
]


def pytest_configure(config):
    """
    Configure pytest for workspace-qdrant-mcp testing.

    Registers custom markers and sets up test environment.
    """
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


def pytest_runtest_setup(item):
    """
    Setup hook called before each test.

    Can be used for conditional test skipping based on environment.
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
    """
    print("\n" + "=" * 70)
    print("Starting workspace-qdrant-mcp test session")
    print("=" * 70)

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