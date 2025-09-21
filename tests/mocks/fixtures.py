"""
Comprehensive fixture library for external dependency mocking.

Provides pytest fixtures that integrate all mock components for easy use
in test suites, with configurable error injection and realistic behavior.
"""

import pytest
from typing import Dict, Any, List, Optional

from .error_injection import ErrorModeManager
from .qdrant_mocks import create_realistic_qdrant_mock, create_failing_qdrant_mock
from .filesystem_mocks import create_filesystem_mock, create_file_watcher_mock
from .grpc_mocks import create_realistic_daemon_communication, create_failing_grpc_client
from .network_mocks import create_realistic_network_client, create_failing_network_client
from .lsp_mocks import create_realistic_metadata_extractor, create_basic_lsp_server
from .embedding_mocks import create_realistic_embedding_service, create_failing_embedding_service
from .external_service_mocks import create_basic_external_service, create_openai_api_mock


@pytest.fixture
def error_manager():
    """Global error manager for coordinating failures across components."""
    return ErrorModeManager()


@pytest.fixture
def mock_qdrant_client_enhanced():
    """Enhanced Qdrant client mock with realistic behavior."""
    return create_realistic_qdrant_mock()


@pytest.fixture
def mock_qdrant_client_failing():
    """Qdrant client mock with high failure rate for error testing."""
    return create_failing_qdrant_mock()


@pytest.fixture
def mock_filesystem():
    """Filesystem operations mock with realistic behavior."""
    return create_filesystem_mock(with_error_injection=True, error_probability=0.02)


@pytest.fixture
def mock_file_watcher():
    """File watcher mock with realistic behavior."""
    return create_file_watcher_mock(with_error_injection=True, error_probability=0.05)


@pytest.fixture
def mock_grpc_daemon():
    """gRPC daemon communication mock with realistic behavior."""
    return create_realistic_daemon_communication()


@pytest.fixture
def mock_grpc_client_failing():
    """gRPC client mock with high failure rate for error testing."""
    return create_failing_grpc_client()


@pytest.fixture
def mock_network_client():
    """Network client mock with realistic behavior."""
    return create_realistic_network_client()


@pytest.fixture
def mock_network_client_failing():
    """Network client mock with high failure rate for error testing."""
    return create_failing_network_client()


@pytest.fixture
def mock_lsp_server():
    """LSP server mock for Python development."""
    return create_basic_lsp_server("python")


@pytest.fixture
def mock_lsp_metadata_extractor():
    """LSP metadata extractor mock with realistic behavior."""
    return create_realistic_metadata_extractor()


@pytest.fixture
def mock_embedding_service():
    """Embedding service mock with realistic behavior."""
    return create_realistic_embedding_service()


@pytest.fixture
def mock_embedding_service_failing():
    """Embedding service mock with high failure rate for error testing."""
    return create_failing_embedding_service()


@pytest.fixture
def mock_external_service():
    """Generic external service mock."""
    return create_basic_external_service()


@pytest.fixture
def mock_openai_api():
    """OpenAI API mock with realistic behavior."""
    return create_openai_api_mock()


@pytest.fixture
def mock_dependency_suite(
    mock_qdrant_client_enhanced,
    mock_filesystem,
    mock_file_watcher,
    mock_grpc_daemon,
    mock_network_client,
    mock_lsp_metadata_extractor,
    mock_embedding_service,
    mock_external_service
):
    """Complete suite of external dependency mocks for integration testing."""
    return {
        "qdrant": mock_qdrant_client_enhanced,
        "filesystem": mock_filesystem,
        "file_watcher": mock_file_watcher,
        "grpc": mock_grpc_daemon,
        "network": mock_network_client,
        "lsp": mock_lsp_metadata_extractor,
        "embedding": mock_embedding_service,
        "external": mock_external_service
    }


@pytest.fixture
def mock_dependency_suite_failing(
    mock_qdrant_client_failing,
    mock_filesystem,
    mock_file_watcher,
    mock_grpc_client_failing,
    mock_network_client_failing,
    mock_lsp_metadata_extractor,
    mock_embedding_service_failing,
    mock_external_service
):
    """Complete suite of external dependency mocks with high failure rates."""
    return {
        "qdrant": mock_qdrant_client_failing,
        "filesystem": mock_filesystem,
        "file_watcher": mock_file_watcher,
        "grpc": mock_grpc_client_failing,
        "network": mock_network_client_failing,
        "lsp": mock_lsp_metadata_extractor,
        "embedding": mock_embedding_service_failing,
        "external": mock_external_service
    }


@pytest.fixture
def mock_error_scenarios(error_manager):
    """Pre-configured error scenarios for testing different failure modes."""

    def apply_scenario(scenario_name: str, components: Optional[List[str]] = None):
        """Apply a specific error scenario to components."""
        error_manager.apply_scenario(scenario_name, components)

    def reset_errors():
        """Reset all error injection."""
        error_manager.reset_all()

    def get_stats():
        """Get error injection statistics."""
        return error_manager.get_global_statistics()

    return {
        "apply_scenario": apply_scenario,
        "reset_errors": reset_errors,
        "get_stats": get_stats
    }


@pytest.fixture(scope="session")
def mock_configuration_profiles():
    """Pre-defined mock configuration profiles for different testing scenarios."""
    return {
        "unit_testing": {
            "description": "Minimal error injection for unit tests",
            "error_rates": {
                "qdrant": 0.01,
                "filesystem": 0.005,
                "grpc": 0.02,
                "network": 0.01,
                "embedding": 0.01
            }
        },
        "integration_testing": {
            "description": "Moderate error injection for integration tests",
            "error_rates": {
                "qdrant": 0.05,
                "filesystem": 0.02,
                "grpc": 0.08,
                "network": 0.05,
                "embedding": 0.03
            }
        },
        "stress_testing": {
            "description": "High error injection for stress tests",
            "error_rates": {
                "qdrant": 0.2,
                "filesystem": 0.1,
                "grpc": 0.3,
                "network": 0.25,
                "embedding": 0.15
            }
        },
        "production_simulation": {
            "description": "Realistic production-like error rates",
            "error_rates": {
                "qdrant": 0.001,
                "filesystem": 0.0005,
                "grpc": 0.002,
                "network": 0.001,
                "embedding": 0.001
            }
        }
    }


@pytest.fixture
def mock_performance_profiles():
    """Pre-defined performance profiles for testing different response times."""
    return {
        "fast": {
            "description": "Fast response times for unit tests",
            "delays": {
                "qdrant_search": 0.01,
                "qdrant_upsert": 0.005,
                "file_read": 0.001,
                "grpc_call": 0.01,
                "network_request": 0.02,
                "embedding_generation": 0.05
            }
        },
        "realistic": {
            "description": "Realistic response times",
            "delays": {
                "qdrant_search": 0.1,
                "qdrant_upsert": 0.05,
                "file_read": 0.01,
                "grpc_call": 0.05,
                "network_request": 0.2,
                "embedding_generation": 0.5
            }
        },
        "slow": {
            "description": "Slow response times for timeout testing",
            "delays": {
                "qdrant_search": 1.0,
                "qdrant_upsert": 0.5,
                "file_read": 0.1,
                "grpc_call": 0.5,
                "network_request": 2.0,
                "embedding_generation": 3.0
            }
        }
    }


@pytest.fixture
def mock_data_generators():
    """Generators for creating realistic test data."""

    class DataGenerators:
        @staticmethod
        def generate_documents(count: int = 10) -> List[Dict[str, Any]]:
            """Generate mock documents for testing."""
            documents = []
            for i in range(count):
                documents.append({
                    "id": f"doc_{i}",
                    "content": f"This is mock document {i} with some test content for searching and indexing.",
                    "metadata": {
                        "source": f"test_file_{i}.txt",
                        "created_at": "2024-01-01T12:00:00Z",
                        "language": "en",
                        "size": len(f"This is mock document {i} with some test content for searching and indexing.")
                    }
                })
            return documents

        @staticmethod
        def generate_vectors(count: int = 10, dim: int = 384) -> List[List[float]]:
            """Generate mock vectors for testing."""
            import random
            return [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(count)]

        @staticmethod
        def generate_search_queries(count: int = 5) -> List[str]:
            """Generate mock search queries for testing."""
            queries = [
                "python function definition",
                "error handling best practices",
                "database connection pooling",
                "API authentication methods",
                "async await patterns"
            ]
            return queries[:count]

        @staticmethod
        def generate_file_tree(depth: int = 3, files_per_dir: int = 5) -> Dict[str, Any]:
            """Generate mock file tree structure."""
            def create_directory(current_depth: int, path: str) -> Dict[str, Any]:
                if current_depth <= 0:
                    return {}

                directory = {"type": "directory", "children": {}}

                # Add files
                for i in range(files_per_dir):
                    file_name = f"file_{i}.py"
                    directory["children"][file_name] = {
                        "type": "file",
                        "content": f"# Mock file content for {path}/{file_name}",
                        "size": 100 + i * 50
                    }

                # Add subdirectories
                for i in range(2):
                    subdir_name = f"subdir_{i}"
                    subdir_path = f"{path}/{subdir_name}"
                    directory["children"][subdir_name] = create_directory(current_depth - 1, subdir_path)

                return directory

            return create_directory(depth, "/mock_project")

    return DataGenerators()


@pytest.fixture
def mock_validation_helpers():
    """Helpers for validating mock behavior in tests."""

    class ValidationHelpers:
        @staticmethod
        def assert_operation_history(mock_obj, expected_operations: List[str]):
            """Assert that mock performed expected operations."""
            if hasattr(mock_obj, 'get_operation_history'):
                history = mock_obj.get_operation_history()
                actual_operations = [op.get('operation', 'unknown') for op in history]
                assert set(expected_operations).issubset(set(actual_operations)), \
                    f"Expected operations {expected_operations} not found in {actual_operations}"

        @staticmethod
        def assert_error_injection_stats(mock_obj, min_errors: int = 0, max_error_rate: float = 1.0):
            """Assert error injection statistics are within expected bounds."""
            if hasattr(mock_obj, 'error_injector') and hasattr(mock_obj.error_injector, 'get_statistics'):
                stats = mock_obj.error_injector.get_statistics()
                assert stats['error_count'] >= min_errors, f"Expected at least {min_errors} errors, got {stats['error_count']}"
                assert stats['error_rate'] <= max_error_rate, f"Error rate {stats['error_rate']} exceeds maximum {max_error_rate}"

        @staticmethod
        def assert_performance_within_bounds(operation_duration: float, max_duration: float):
            """Assert operation completed within expected time bounds."""
            assert operation_duration <= max_duration, \
                f"Operation took {operation_duration}s, expected <= {max_duration}s"

        @staticmethod
        def get_mock_statistics(mock_suite: Dict[str, Any]) -> Dict[str, Any]:
            """Get comprehensive statistics from a mock suite."""
            stats = {}
            for component_name, mock_obj in mock_suite.items():
                component_stats = {"operations": 0, "errors": 0}

                if hasattr(mock_obj, 'get_operation_history'):
                    history = mock_obj.get_operation_history()
                    component_stats["operations"] = len(history)

                if hasattr(mock_obj, 'error_injector') and hasattr(mock_obj.error_injector, 'get_statistics'):
                    error_stats = mock_obj.error_injector.get_statistics()
                    component_stats["errors"] = error_stats['error_count']
                    component_stats["error_rate"] = error_stats['error_rate']

                stats[component_name] = component_stats

            return stats

    return ValidationHelpers()


# Marker-based fixtures for different test types
@pytest.fixture
def mock_for_unit_tests(mock_dependency_suite, mock_configuration_profiles):
    """Optimized mock suite for unit tests with minimal error injection."""
    profile = mock_configuration_profiles["unit_testing"]

    # Configure low error rates for stable unit testing
    for component_name, mock_obj in mock_dependency_suite.items():
        if hasattr(mock_obj, 'error_injector'):
            error_rate = profile["error_rates"].get(component_name, 0.01)
            if hasattr(mock_obj.error_injector, 'configure_connection_issues'):
                mock_obj.error_injector.configure_connection_issues(error_rate)

    return mock_dependency_suite


@pytest.fixture
def mock_for_integration_tests(mock_dependency_suite, mock_configuration_profiles):
    """Mock suite configured for integration tests with moderate error injection."""
    profile = mock_configuration_profiles["integration_testing"]

    # Configure moderate error rates for integration testing
    for component_name, mock_obj in mock_dependency_suite.items():
        if hasattr(mock_obj, 'error_injector'):
            error_rate = profile["error_rates"].get(component_name, 0.05)
            if hasattr(mock_obj.error_injector, 'configure_connection_issues'):
                mock_obj.error_injector.configure_connection_issues(error_rate)

    return mock_dependency_suite


@pytest.fixture
def mock_for_stress_tests(mock_dependency_suite_failing, mock_configuration_profiles):
    """Mock suite configured for stress tests with high error injection."""
    profile = mock_configuration_profiles["stress_testing"]

    # Configure high error rates for stress testing
    for component_name, mock_obj in mock_dependency_suite_failing.items():
        if hasattr(mock_obj, 'error_injector'):
            error_rate = profile["error_rates"].get(component_name, 0.2)
            if hasattr(mock_obj.error_injector, 'configure_connection_issues'):
                mock_obj.error_injector.configure_connection_issues(error_rate)

    return mock_dependency_suite_failing