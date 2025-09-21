"""
Tests for the external dependency mocking infrastructure.

Validates that all mock components work correctly and provide the expected
behavior for testing different scenarios.
"""

import asyncio
import pytest
from unittest.mock import Mock

from tests.mocks import (
    create_realistic_qdrant_mock,
    create_failing_qdrant_mock,
    create_filesystem_mock,
    create_realistic_daemon_communication,
    create_realistic_network_client,
    create_basic_lsp_server,
    create_realistic_embedding_service,
    create_basic_external_service,
    ErrorModeManager,
    FailureScenarios
)


class TestQdrantMocks:
    """Test Qdrant client mocking functionality."""

    def test_realistic_qdrant_mock_creation(self):
        """Test creating realistic Qdrant mock."""
        mock = create_realistic_qdrant_mock()
        assert mock is not None
        assert hasattr(mock, 'search')
        assert hasattr(mock, 'upsert')
        assert hasattr(mock, 'error_injector')

    @pytest.mark.asyncio
    async def test_qdrant_search_operation(self):
        """Test Qdrant search operation."""
        mock = create_realistic_qdrant_mock()

        results = await mock.search("test-collection", [0.1] * 384, limit=5)
        assert isinstance(results, list)
        assert len(results) <= 5

        # Check operation history
        history = mock.get_operation_history()
        assert len(history) > 0
        assert history[-1]["operation"] == "search"

    @pytest.mark.asyncio
    async def test_qdrant_error_injection(self):
        """Test Qdrant error injection."""
        mock = create_failing_qdrant_mock(error_rate=1.0)  # 100% failure

        with pytest.raises(Exception):
            await mock.search("test-collection", [0.1] * 384)

    def test_qdrant_state_reset(self):
        """Test Qdrant mock state reset."""
        mock = create_realistic_qdrant_mock()

        # Perform some operations
        asyncio.run(mock.search("test", [0.1] * 384))

        # Verify history exists
        assert len(mock.get_operation_history()) > 0

        # Reset and verify clean state
        mock.reset_state()
        assert len(mock.get_operation_history()) == 0


class TestFilesystemMocks:
    """Test filesystem mocking functionality."""

    def test_filesystem_mock_creation(self):
        """Test creating filesystem mock."""
        mock = create_filesystem_mock()
        assert mock is not None
        assert hasattr(mock, 'read_text')
        assert hasattr(mock, 'write_text')

    def test_filesystem_operations(self):
        """Test filesystem operations."""
        mock = create_filesystem_mock()

        # Add file and read it
        mock.add_file("/test/file.txt", "test content")
        content = mock.read_text("/test/file.txt")
        assert content == "test content"

        # Write new file
        mock.write_text("/test/new.txt", "new content")
        new_content = mock.read_text("/test/new.txt")
        assert new_content == "new content"

        # Check operation history
        history = mock.get_operation_history()
        assert any(op["operation"] == "read_text" for op in history)
        assert any(op["operation"] == "write_text" for op in history)

    def test_filesystem_error_injection(self):
        """Test filesystem error injection."""
        mock = create_filesystem_mock(with_error_injection=True, error_probability=1.0)

        with pytest.raises(Exception):
            mock.read_text("/nonexistent/file.txt")


class TestGRPCMocks:
    """Test gRPC communication mocking functionality."""

    @pytest.mark.asyncio
    async def test_grpc_daemon_creation(self):
        """Test creating gRPC daemon mock."""
        mock = create_realistic_daemon_communication()
        assert mock is not None
        assert hasattr(mock, 'initialize_daemon')
        assert hasattr(mock, 'client')

    @pytest.mark.asyncio
    async def test_grpc_daemon_operations(self):
        """Test gRPC daemon operations."""
        mock = create_realistic_daemon_communication()

        # Initialize daemon
        result = await mock.initialize_daemon({"test": "config"})
        assert result["status"] == "initialized"

        # Test document ingestion
        doc_result = await mock.client.ingest_document(
            collection_name="test",
            document_path="/test/doc.txt",
            content="test content"
        )
        assert "document_id" in doc_result
        assert doc_result["status"] == "ingested"

        # Check operation history
        history = mock.get_operation_history()
        assert len(history) > 0


class TestNetworkMocks:
    """Test network operation mocking functionality."""

    @pytest.mark.asyncio
    async def test_network_client_creation(self):
        """Test creating network client mock."""
        mock = create_realistic_network_client()
        assert mock is not None
        assert hasattr(mock, 'get')
        assert hasattr(mock, 'post')

    @pytest.mark.asyncio
    async def test_network_operations(self):
        """Test network operations."""
        mock = create_realistic_network_client()

        # Test GET request
        response = await mock.get("https://api.example.com/data")
        assert response.status_code == 200
        assert hasattr(response, 'json')

        # Test POST request
        post_response = await mock.post(
            "https://api.example.com/items",
            json={"name": "test"}
        )
        assert post_response.status_code in [200, 201]

        # Check operation history
        history = mock.get_operation_history()
        assert any(op["operation"] == "GET" for op in history)
        assert any(op["operation"] == "POST" for op in history)


class TestLSPMocks:
    """Test LSP server mocking functionality."""

    @pytest.mark.asyncio
    async def test_lsp_server_creation(self):
        """Test creating LSP server mock."""
        mock = create_basic_lsp_server("python")
        assert mock is not None
        assert mock.language == "python"
        assert hasattr(mock, 'initialize')

    @pytest.mark.asyncio
    async def test_lsp_operations(self):
        """Test LSP operations."""
        mock = create_basic_lsp_server("python")

        # Initialize server
        result = await mock.initialize(["/workspace"], {})
        assert "capabilities" in result
        assert mock.initialized

        # Open document and get symbols
        await mock.open_document("/test.py", "def hello(): pass", "python")
        symbols = await mock.get_symbols("/test.py")
        assert isinstance(symbols, list)

        # Check operation history
        history = mock.get_operation_history()
        assert any(op["operation"] == "initialize" for op in history)
        assert any(op["operation"] == "open_document" for op in history)


class TestEmbeddingMocks:
    """Test embedding service mocking functionality."""

    @pytest.mark.asyncio
    async def test_embedding_service_creation(self):
        """Test creating embedding service mock."""
        mock = create_realistic_embedding_service()
        assert mock is not None
        assert hasattr(mock, 'initialize')
        assert hasattr(mock, 'generate_embeddings')

    @pytest.mark.asyncio
    async def test_embedding_operations(self):
        """Test embedding operations."""
        mock = create_realistic_embedding_service()

        # Initialize service
        await mock.initialize()
        assert mock.initialized

        # Generate embeddings
        result = await mock.generate_embeddings("test text")
        assert "dense" in result
        assert "sparse" in result
        assert isinstance(result["dense"], list)
        assert len(result["dense"]) == mock.vector_dim

        # Test batch generation
        batch_results = await mock.generate_batch_embeddings(["text1", "text2"])
        assert len(batch_results) == 2
        assert all("dense" in res for res in batch_results)


class TestExternalServiceMocks:
    """Test external service mocking functionality."""

    @pytest.mark.asyncio
    async def test_external_service_creation(self):
        """Test creating external service mock."""
        mock = create_basic_external_service()
        assert mock is not None
        assert hasattr(mock, 'authenticate')
        assert hasattr(mock, 'make_request')

    @pytest.mark.asyncio
    async def test_external_service_operations(self):
        """Test external service operations."""
        mock = create_basic_external_service()

        # Authenticate
        auth_result = await mock.authenticate("test_api_key")
        assert auth_result["authenticated"]
        assert mock.authenticated

        # Make API request
        response = await mock.make_request("GET", "/api/test")
        assert "message" in response
        assert "timestamp" in response

        # Check operation history
        history = mock.get_operation_history()
        assert any(op["operation"] == "authenticate" for op in history)
        assert any(op["operation"] == "make_request" for op in history)


class TestErrorInjection:
    """Test error injection framework."""

    def test_error_manager_creation(self):
        """Test creating error mode manager."""
        manager = ErrorModeManager()
        assert manager is not None
        assert manager.global_enabled

    def test_error_scenario_application(self):
        """Test applying error scenarios."""
        manager = ErrorModeManager()

        # Create mocks and register injectors
        qdrant_mock = create_realistic_qdrant_mock()
        manager.register_injector("qdrant", qdrant_mock.error_injector)

        # Apply scenario
        manager.apply_scenario("connection_issues", ["qdrant"])

        # Verify scenario is active
        stats = manager.get_global_statistics()
        assert stats["scenario_active"]
        assert "qdrant" in stats["components"]

    def test_failure_scenarios(self):
        """Test predefined failure scenarios."""
        connection_scenarios = FailureScenarios.connection_issues()
        assert "connection_timeout" in connection_scenarios
        assert "connection_refused" in connection_scenarios

        production_scenarios = FailureScenarios.realistic_production()
        assert len(production_scenarios) > 0

        # Verify production scenarios have lower error rates
        for scenario in production_scenarios.values():
            assert scenario["probability"] <= 0.1


class TestMockFixtures:
    """Test mock fixture integration."""

    def test_mock_dependency_suite(self, mock_dependency_suite):
        """Test complete mock dependency suite."""
        assert "qdrant" in mock_dependency_suite
        assert "filesystem" in mock_dependency_suite
        assert "grpc" in mock_dependency_suite
        assert "network" in mock_dependency_suite
        assert "lsp" in mock_dependency_suite
        assert "embedding" in mock_dependency_suite
        assert "external" in mock_dependency_suite

    def test_mock_validation_helpers(self, mock_validation_helpers):
        """Test mock validation helpers."""
        # Create a mock with operations
        qdrant_mock = create_realistic_qdrant_mock()
        asyncio.run(qdrant_mock.search("test", [0.1] * 384))

        # Validate operation history
        mock_validation_helpers.assert_operation_history(qdrant_mock, ["search"])

    def test_mock_data_generators(self, mock_data_generators):
        """Test mock data generators."""
        # Generate documents
        docs = mock_data_generators.generate_documents(5)
        assert len(docs) == 5
        assert all("id" in doc for doc in docs)
        assert all("content" in doc for doc in docs)

        # Generate vectors
        vectors = mock_data_generators.generate_vectors(3, 384)
        assert len(vectors) == 3
        assert all(len(vec) == 384 for vec in vectors)

        # Generate queries
        queries = mock_data_generators.generate_search_queries(3)
        assert len(queries) == 3
        assert all(isinstance(q, str) for q in queries)


class TestMockRealism:
    """Test mock realism and behavior accuracy."""

    @pytest.mark.asyncio
    async def test_realistic_delays(self):
        """Test that mocks have realistic delays."""
        import time

        mock = create_realistic_qdrant_mock()

        start_time = time.time()
        await mock.search("test", [0.1] * 384)
        duration = time.time() - start_time

        # Should have some delay but not too much for tests
        assert 0.01 <= duration <= 1.0

    def test_deterministic_behavior(self):
        """Test that mocks produce deterministic results when needed."""
        mock = create_realistic_qdrant_mock()

        # Same input should produce similar results
        vector = [0.1] * 384
        result1 = asyncio.run(mock.search("test", vector))
        result2 = asyncio.run(mock.search("test", vector))

        # Results should be consistent in structure
        assert len(result1) == len(result2)
        if result1 and result2:
            assert result1[0]["id"] == result2[0]["id"]

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test that mocks can recover from error states."""
        mock = create_failing_qdrant_mock(error_rate=0.5)

        # Some operations should succeed even with error injection
        successes = 0
        attempts = 20

        for _ in range(attempts):
            try:
                await mock.search("test", [0.1] * 384)
                successes += 1
            except Exception:
                pass

        # With 50% error rate, should have some successes
        assert 0 < successes < attempts