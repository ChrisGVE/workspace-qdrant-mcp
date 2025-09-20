"""
Tests for testcontainers setup and configuration.

These tests validate the testcontainers infrastructure setup without
requiring Docker to be running, focusing on configuration and imports.
"""

import pytest
from unittest.mock import Mock, patch

from tests.utils.testcontainers_qdrant import (
    IsolatedQdrantContainer,
    QdrantContainerManager,
    get_container_manager,
    create_test_config
)


class TestTestcontainersSetup:
    """Test testcontainers setup and configuration."""

    def test_isolated_container_configuration(self):
        """Test isolated container configuration."""
        container = IsolatedQdrantContainer(
            image="qdrant/qdrant:test",
            http_port=6333,
            grpc_port=6334,
            startup_timeout=30,
            health_check_interval=0.5
        )

        assert container.image == "qdrant/qdrant:test"
        assert container.http_port == 6333
        assert container.grpc_port == 6334
        assert container.startup_timeout == 30
        assert container.health_check_interval == 0.5
        assert not container._is_started

    def test_container_manager_singleton(self):
        """Test container manager singleton behavior."""
        manager1 = get_container_manager()
        manager2 = get_container_manager()

        # Should be the same instance
        assert manager1 is manager2
        assert isinstance(manager1, QdrantContainerManager)

    def test_container_manager_operations(self):
        """Test container manager operations without Docker."""
        manager = QdrantContainerManager()

        # Test initial state
        assert manager._session_container is None
        assert len(manager._containers) == 0

    def test_test_config_creation(self):
        """Test test configuration creation."""
        # Mock container
        mock_container = Mock()
        mock_container.http_url = "http://localhost:12345"

        config = create_test_config(mock_container)

        # Test that the container URL override works
        assert config.qdrant.url == "http://localhost:12345"

        # Config will load from actual config file, so we just verify
        # that the config is created and URL is overridden correctly
        assert config is not None
        assert hasattr(config, 'qdrant')
        assert hasattr(config, 'workspace')
        assert hasattr(config, 'embedding')

    def test_container_properties_before_start(self):
        """Test container properties raise appropriate errors before start."""
        container = IsolatedQdrantContainer()

        with pytest.raises(RuntimeError, match="Container not initialized"):
            _ = container.container

        with pytest.raises(RuntimeError, match="Container not started"):
            _ = container.client

    def test_pytest_markers_available(self):
        """Test that pytest markers are properly configured."""
        # These markers should be available after pytest configuration
        expected_markers = [
            "requires_docker",
            "requires_qdrant_container",
            "isolated_container",
            "shared_container"
        ]

        for marker_name in expected_markers:
            marker = getattr(pytest.mark, marker_name, None)
            assert marker is not None, f"Marker {marker_name} not available"

    def test_fixture_imports(self):
        """Test that all fixtures can be imported."""
        from tests.conftest import (
            qdrant_container_manager,
            session_qdrant_container,
            isolated_qdrant_container,
            shared_qdrant_container,
            isolated_qdrant_client,
            shared_qdrant_client,
            test_config,
            containerized_qdrant_instance
        )

        # All fixtures should be callable (they're fixture functions)
        fixtures = [
            qdrant_container_manager,
            session_qdrant_container,
            isolated_qdrant_container,
            shared_qdrant_container,
            isolated_qdrant_client,
            shared_qdrant_client,
            test_config,
            containerized_qdrant_instance
        ]

        for fixture in fixtures:
            assert callable(fixture)
            # Check that they have pytest fixture attributes
            assert hasattr(fixture, '_pytestfixturefunction') or hasattr(fixture, '__wrapped__')

    @patch('tests.utils.testcontainers_qdrant.QdrantContainer')
    def test_container_context_manager_interface(self, mock_qdrant_container):
        """Test container context manager interface without Docker."""
        container = IsolatedQdrantContainer()

        # Test context manager methods exist
        assert hasattr(container, '__enter__')
        assert hasattr(container, '__exit__')

        # Test they're callable
        assert callable(container.__enter__)
        assert callable(container.__exit__)

    def test_async_context_manager_import(self):
        """Test async context manager can be imported."""
        from tests.utils.testcontainers_qdrant import isolated_qdrant_instance

        # Should be an async generator function
        import inspect
        assert inspect.isfunction(isolated_qdrant_instance)

    def test_container_configuration_validation(self):
        """Test container configuration validation."""
        # Test valid configurations
        valid_configs = [
            {"image": "qdrant/qdrant:v1.7.4"},
            {"image": "qdrant/qdrant:latest", "startup_timeout": 60},
            {"http_port": 6333, "grpc_port": 6334},
        ]

        for config in valid_configs:
            container = IsolatedQdrantContainer(**config)
            assert container is not None

    def test_container_health_check_configuration(self):
        """Test container health check configuration."""
        container = IsolatedQdrantContainer(
            startup_timeout=45,
            health_check_interval=2.0
        )

        assert container.startup_timeout == 45
        assert container.health_check_interval == 2.0

    def test_container_manager_cleanup_interface(self):
        """Test container manager cleanup interface."""
        manager = QdrantContainerManager()

        # Test cleanup methods exist and are callable
        assert hasattr(manager, 'cleanup_container')
        assert hasattr(manager, 'cleanup_session')
        assert hasattr(manager, 'cleanup_all')

        assert callable(manager.cleanup_container)
        assert callable(manager.cleanup_session)
        assert callable(manager.cleanup_all)

        # Test they can be called without error (no-op when no containers)
        manager.cleanup_container("test_id")
        manager.cleanup_session()
        manager.cleanup_all()

    def test_module_level_constants(self):
        """Test module-level constants and functions."""
        from tests.utils.testcontainers_qdrant import get_container_manager

        assert callable(get_container_manager)

        # Test that we can get the global container manager
        manager = get_container_manager()
        assert isinstance(manager, QdrantContainerManager)