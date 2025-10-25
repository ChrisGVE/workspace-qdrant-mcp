"""
Configuration and fixtures for integration tests.

Provides shared fixtures, configuration, and utilities for running
comprehensive integration tests with isolated test environments.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from testcontainers.compose import DockerCompose

import docker

# Test environment configuration
TEST_ENVIRONMENT_CONFIG = {
    "qdrant": {
        "image": "qdrant/qdrant:v1.7.4",
        "http_port": 6333,
        "grpc_port": 6334,
        "health_check_timeout": 30,
        "startup_wait": 5
    },
    "timeout": {
        "short": 10,
        "medium": 30,
        "long": 60
    },
    "performance": {
        "baseline_ingestion_rate": 1.0,  # docs/second
        "max_search_latency": 2000,  # milliseconds
        "memory_limit_mb": 500
    }
}


@pytest.fixture(scope="session")
def docker_client():
    """Provide Docker client for container management."""
    try:
        client = docker.from_env()
        yield client
    except Exception:
        pytest.skip("Docker not available for integration tests")


@pytest.fixture(scope="session")
def integration_test_markers():
    """Define test markers for integration test categorization."""
    return {
        "smoke": "Basic functionality smoke tests",
        "integration": "Full integration tests",
        "performance": "Performance and benchmark tests",
        "slow": "Long-running tests",
        "regression": "Regression tests for bug fixes",
        "requires_docker": "Tests requiring Docker containers",
        "requires_qdrant": "Tests requiring Qdrant server"
    }


@pytest.fixture(scope="session")
def test_data_factory():
    """Factory for generating test data of various sizes and types."""

    class TestDataFactory:
        @staticmethod
        def create_text_document(size: str = "small", topic: str = "general") -> str:
            """Create text document of specified size."""
            base_content = {
                "general": "This is a test document for integration testing. ",
                "technical": "Technical documentation content with API references. ",
                "narrative": "A story about software development and testing processes. "
            }

            content = base_content.get(topic, base_content["general"])

            multipliers = {
                "tiny": 5,      # ~100 characters
                "small": 50,    # ~1KB
                "medium": 500,  # ~10KB
                "large": 5000,  # ~100KB
                "huge": 50000   # ~1MB
            }

            multiplier = multipliers.get(size, multipliers["small"])
            return content * multiplier

        @staticmethod
        def create_structured_data(complexity: str = "simple") -> dict[str, Any]:
            """Create structured data with specified complexity."""
            if complexity == "simple":
                return {
                    "title": "Test Document",
                    "author": "Test Author",
                    "tags": ["test", "integration"]
                }
            elif complexity == "complex":
                return {
                    "metadata": {
                        "title": "Complex Test Document",
                        "author": {"name": "Test Author", "id": 12345},
                        "tags": ["test", "integration", "complex"],
                        "properties": {
                            "language": "en",
                            "version": "1.0",
                            "features": ["search", "analytics", "reporting"]
                        }
                    },
                    "content_sections": [
                        {"section": "introduction", "words": 150},
                        {"section": "methodology", "words": 300},
                        {"section": "results", "words": 250}
                    ],
                    "references": list(range(1, 26)),  # 25 references
                    "embedding_vector": [0.1] * 384
                }
            else:
                return {"simple": "data"}

        @staticmethod
        def create_binary_file(file_type: str = "pdf") -> bytes:
            """Create mock binary file content."""
            headers = {
                "pdf": b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n",
                "docx": b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00",
                "image": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
            }

            header = headers.get(file_type, b"BINARY_FILE_HEADER")
            content = b"Mock binary content for testing " * 100
            return header + content

    yield TestDataFactory()


@pytest.fixture
def temp_workspace():
    """Create temporary workspace with realistic project structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create realistic project structure
        dirs = ["src", "docs", "tests", "config", "data", ".git"]
        for dir_name in dirs:
            (workspace / dir_name).mkdir()

        # Add common files
        files = {
            "README.md": "# Test Project\n\nIntegration testing workspace.",
            "pyproject.toml": "[tool.pytest]\ntestpaths = ['tests']",
            ".gitignore": "__pycache__/\n*.pyc\n.env",
            "src/main.py": "def main():\n    pass",
            "docs/api.md": "# API Documentation\n\n## Endpoints",
            "tests/test_example.py": "def test_example():\n    assert True",
            "config/settings.yaml": "debug: false\nlog_level: info"
        }

        for file_path, content in files.items():
            full_path = workspace / file_path
            full_path.write_text(content)

        yield {
            "path": workspace,
            "files": list(files.keys()),
            "total_files": len(files)
        }


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for integration tests."""
    return {
        "ingestion": {
            "small_doc_max_time_ms": 1000,
            "medium_doc_max_time_ms": 3000,
            "large_doc_max_time_ms": 10000,
            "min_throughput_docs_per_sec": 0.5
        },
        "search": {
            "simple_query_max_time_ms": 1000,
            "complex_query_max_time_ms": 5000,
            "p95_latency_ms": 3000,
            "min_relevance_score": 0.5
        },
        "system": {
            "startup_max_time_ms": 15000,
            "shutdown_max_time_ms": 5000,
            "memory_max_mb": 500,
            "max_open_files": 100
        }
    }


@pytest.fixture
def integration_config():
    """Provide integration test configuration."""
    return {
        "test_environment": TEST_ENVIRONMENT_CONFIG,
        "coverage": {
            "target_percentage": 80,
            "branch_coverage": True,
            "exclude_patterns": ["*/tests/*", "*/conftest.py"]
        },
        "retry": {
            "max_attempts": 3,
            "delay_seconds": 1.0,
            "backoff_multiplier": 2.0
        },
        "logging": {
            "level": "INFO",
            "capture": True,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def mock_external_services():
    """Provide mocks for external service dependencies."""

    class MockServices:
        def __init__(self):
            self.qdrant_available = True
            self.grpc_available = True
            self.embedding_service_available = True

        def set_qdrant_available(self, available: bool):
            self.qdrant_available = available

        def set_grpc_available(self, available: bool):
            self.grpc_available = available

        def set_embedding_service_available(self, available: bool):
            self.embedding_service_available = available

        def reset_all(self):
            self.qdrant_available = True
            self.grpc_available = True
            self.embedding_service_available = True

    yield MockServices()


@pytest.fixture
def cleanup_tracker():
    """Track resources that need cleanup after tests."""
    resources_to_cleanup = []

    def register_cleanup(resource_type: str, resource_id: str, cleanup_func):
        resources_to_cleanup.append({
            "type": resource_type,
            "id": resource_id,
            "cleanup": cleanup_func
        })

    yield register_cleanup

    # Cleanup registered resources
    for resource in resources_to_cleanup:
        try:
            resource["cleanup"]()
        except Exception as e:
            print(f"Cleanup failed for {resource['type']} {resource['id']}: {e}")


def pytest_configure(config):
    """Configure pytest for integration tests."""
    # Register custom markers
    config.addinivalue_line("markers", "smoke: Basic functionality tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "requires_docker: Tests requiring Docker")

    # Set up test environment
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent.parent.parent))
    os.environ.setdefault("TEST_ENV", "integration")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for integration test suite."""
    # Add markers based on test location and name
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests with "slow" in name or that are performance tests
        if ("slow" in item.name.lower() or
            "performance" in item.name.lower() or
            item.get_closest_marker("performance")):
            item.add_marker(pytest.mark.slow)

        # Add requires_docker marker to tests using containers
        if ("container" in item.name.lower() or
            "docker" in item.name.lower() or
            any("testcontainer" in str(dep) for dep in item.fixturenames)):
            item.add_marker(pytest.mark.requires_docker)


@pytest.fixture(autouse=True, scope="session")
def integration_test_setup():
    """Automatic setup for integration test suite."""
    print("\n" + "=" * 70)
    print("Starting workspace-qdrant-mcp test session")
    print("=" * 70)
    print("Setting up integration test environment...")

    # Verify Docker availability
    try:
        import docker
        client = docker.from_env()
        client.ping()
        docker_available = True
    except Exception:
        docker_available = False
        print("WARNING: Docker not available - some tests will be skipped")

    # Set environment variables for integration tests
    os.environ["INTEGRATION_TESTING"] = "1"
    os.environ["DOCKER_AVAILABLE"] = str(docker_available).lower()

    # Note: PYTEST_CURRENT_TEST is managed by pytest internally, don't manipulate it

    yield

    print("Tearing down integration test environment...")
    print("\n" + "=" * 70)
    print("Cleaning up test session")
    print("=" * 70)
    # Cleanup would happen here


@pytest.fixture
async def async_test_timeout():
    """Provide timeout context for async tests."""

    class AsyncTimeout:
        def __init__(self, timeout_seconds: int = 30):
            self.timeout = timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def wait_for(self, coro, timeout: int | None = None):
            """Wait for coroutine with timeout."""
            actual_timeout = timeout or self.timeout
            return await asyncio.wait_for(coro, timeout=actual_timeout)

    yield AsyncTimeout
