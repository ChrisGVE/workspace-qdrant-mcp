"""
Simple validation test for external dependency mocking infrastructure.

This test validates the core functionality without complex imports.
"""

import asyncio
import pytest


def test_mock_imports():
    """Test that all mock components can be imported."""
    from tests.mocks import (
        create_realistic_qdrant_mock,
        create_basic_embedding_service,
        create_filesystem_mock,
        create_realistic_network_client,
        ErrorModeManager
    )

    assert create_realistic_qdrant_mock is not None
    assert create_basic_embedding_service is not None
    assert create_filesystem_mock is not None
    assert create_realistic_network_client is not None
    assert ErrorModeManager is not None


def test_qdrant_mock_basic_functionality():
    """Test basic Qdrant mock functionality."""
    from tests.mocks import create_realistic_qdrant_mock

    mock = create_realistic_qdrant_mock()
    assert mock is not None
    assert hasattr(mock, 'search')
    assert hasattr(mock, 'upsert')
    assert hasattr(mock, 'error_injector')

    # Test operation history
    history = mock.get_operation_history()
    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_qdrant_mock_async_operations():
    """Test Qdrant mock async operations."""
    from tests.mocks import create_realistic_qdrant_mock

    mock = create_realistic_qdrant_mock()

    # Test search operation
    results = await mock.search("test-collection", [0.1] * 384, limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3

    # Verify operation was recorded
    history = mock.get_operation_history()
    assert len(history) > 0
    assert history[-1]["operation"] == "search"


def test_embedding_mock_functionality():
    """Test embedding service mock functionality."""
    from tests.mocks import create_basic_embedding_service

    mock = create_basic_embedding_service()
    assert mock is not None
    assert hasattr(mock, 'initialize')
    assert hasattr(mock, 'generate_embeddings')
    assert mock.vector_dim == 384


@pytest.mark.asyncio
async def test_embedding_mock_operations():
    """Test embedding mock operations."""
    from tests.mocks import create_basic_embedding_service

    mock = create_basic_embedding_service()

    # Initialize
    await mock.initialize()
    assert mock.initialized

    # Generate embeddings
    result = await mock.generate_embeddings("test text")
    assert "dense" in result
    assert "sparse" in result
    assert isinstance(result["dense"], list)
    assert len(result["dense"]) == 384


def test_filesystem_mock_functionality():
    """Test filesystem mock functionality."""
    from tests.mocks import create_filesystem_mock

    mock = create_filesystem_mock()
    assert mock is not None
    assert hasattr(mock, 'read_text')
    assert hasattr(mock, 'write_text')

    # Test virtual filesystem
    mock.add_file("/test/file.txt", "test content")
    content = mock.read_text("/test/file.txt")
    assert content == "test content"


def test_error_injection_framework():
    """Test error injection framework."""
    from tests.mocks import ErrorModeManager, create_realistic_qdrant_mock

    # Create error manager
    manager = ErrorModeManager()
    assert manager.global_enabled

    # Create mock and register injector
    mock = create_realistic_qdrant_mock()
    manager.register_injector("qdrant", mock.error_injector)

    # Apply scenario
    manager.apply_scenario("connection_issues", ["qdrant"])

    # Verify scenario is active
    stats = manager.get_global_statistics()
    assert stats["scenario_active"]
    assert "qdrant" in stats["components"]


@pytest.mark.asyncio
async def test_network_mock_operations():
    """Test network mock operations."""
    from tests.mocks import create_realistic_network_client

    mock = create_realistic_network_client()
    assert mock is not None

    # Test GET request
    response = await mock.get("https://api.example.com/test")
    assert response.status_code == 200
    assert hasattr(response, 'json')

    # Verify operation was recorded
    history = mock.get_operation_history()
    assert len(history) > 0
    assert history[-1]["operation"] == "GET"


def test_mock_state_management():
    """Test mock state reset and management."""
    from tests.mocks import create_realistic_qdrant_mock

    mock = create_realistic_qdrant_mock()

    # Perform operation to create history
    asyncio.run(mock.search("test", [0.1] * 384))

    # Verify history exists
    history = mock.get_operation_history()
    assert len(history) > 0

    # Reset state
    mock.reset_state()

    # Verify clean state
    new_history = mock.get_operation_history()
    assert len(new_history) == 0


def test_error_injection_configuration():
    """Test error injection configuration."""
    from tests.mocks import create_failing_qdrant_mock

    # Create mock with high error rate
    mock = create_failing_qdrant_mock(error_rate=0.8)
    assert mock.error_injector is not None

    # Verify error injector is configured
    stats = mock.error_injector.get_statistics()
    assert stats["enabled"]


@pytest.mark.asyncio
async def test_realistic_behavior_simulation():
    """Test that mocks provide realistic behavior."""
    from tests.mocks import create_realistic_qdrant_mock
    import time

    mock = create_realistic_qdrant_mock()

    # Test that operations have some delay (realistic timing)
    start_time = time.time()
    await mock.search("test", [0.1] * 384)
    duration = time.time() - start_time

    # Should have minimal delay but some realistic simulation
    assert 0.001 <= duration <= 1.0  # Between 1ms and 1s


def test_comprehensive_mock_coverage():
    """Test that all external dependencies have mocks."""
    from tests.mocks import (
        create_realistic_qdrant_mock,
        create_filesystem_mock,
        create_realistic_daemon_communication,
        create_realistic_network_client,
        create_basic_lsp_server,
        create_basic_embedding_service,
        create_basic_external_service
    )

    # Verify all major external dependencies are covered
    mocks = {
        "qdrant": create_realistic_qdrant_mock(),
        "filesystem": create_filesystem_mock(),
        "grpc": create_realistic_daemon_communication(),
        "network": create_realistic_network_client(),
        "lsp": create_basic_lsp_server(),
        "embedding": create_basic_embedding_service(),
        "external": create_basic_external_service()
    }

    # All mocks should be created successfully
    for name, mock in mocks.items():
        assert mock is not None, f"Failed to create {name} mock"

        # Each mock should have basic expected attributes
        if hasattr(mock, 'reset_state'):
            assert callable(mock.reset_state)

        if hasattr(mock, 'get_operation_history'):
            assert callable(mock.get_operation_history)


if __name__ == "__main__":
    # Run tests manually for validation
    test_mock_imports()
    print("✓ Mock imports test passed")

    test_qdrant_mock_basic_functionality()
    print("✓ Qdrant mock basic functionality test passed")

    test_embedding_mock_functionality()
    print("✓ Embedding mock functionality test passed")

    test_filesystem_mock_functionality()
    print("✓ Filesystem mock functionality test passed")

    test_error_injection_framework()
    print("✓ Error injection framework test passed")

    test_comprehensive_mock_coverage()
    print("✓ Comprehensive mock coverage test passed")

    print("✓ All simple mock validation tests passed!")