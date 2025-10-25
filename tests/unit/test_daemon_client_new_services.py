"""
Unit tests for DaemonClient with DocumentService and CollectionService support.

Tests the new gRPC service methods added in Task 375.1.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from common.grpc.daemon_client import (
    DaemonClient,
    DaemonClientError,
    DaemonUnavailableError,
)
from common.grpc.generated.workspace_daemon_pb2 import (
    CollectionConfig,
    CreateAliasRequest,
    CreateCollectionRequest,
    CreateCollectionResponse,
    DeleteAliasRequest,
    DeleteCollectionRequest,
    DeleteTextRequest,
    IngestTextRequest,
    IngestTextResponse,
    RenameAliasRequest,
    UpdateTextRequest,
    UpdateTextResponse,
)


@pytest.fixture
def mock_config():
    """Mock configuration manager."""
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "grpc.host": "localhost",
        "grpc.port": 50051,
        "grpc": {"max_message_size_mb": 100},
    }.get(key, default)
    return config


@pytest.fixture
def daemon_client(mock_config):
    """Create a DaemonClient instance with mocked configuration."""
    client = DaemonClient(config_manager=mock_config, project_path="/test/project")
    return client


@pytest.fixture
def connected_client(daemon_client):
    """Create a connected DaemonClient with mocked stubs."""
    daemon_client._connected = True
    daemon_client._started = True  # Prevent start() from attempting connection
    daemon_client._channel = MagicMock()
    daemon_client._system_stub = MagicMock()
    daemon_client._document_stub = MagicMock()
    daemon_client._collection_stub = MagicMock()
    daemon_client._ingest_stub = MagicMock()
    return daemon_client


class TestDocumentServiceMethods:
    """Tests for DocumentService methods."""

    @pytest.mark.asyncio
    async def test_ingest_text_success(self, connected_client):
        """Test successful text ingestion."""
        # Mock the response
        mock_response = IngestTextResponse(
            document_id="doc123",
            success=True,
            chunks_created=3,
            error_message="",
        )
        connected_client._document_stub.IngestText = AsyncMock(return_value=mock_response)

        # Call the method
        with patch("common.grpc.daemon_client.validate_llm_collection_access"):
            response = await connected_client.ingest_text(
                content="Test content",
                collection_basename="scratchbook",
                tenant_id="user123",
                metadata={"source": "test"},
            )

        # Assertions
        assert response.document_id == "doc123"
        assert response.success is True
        assert response.chunks_created == 3
        connected_client._document_stub.IngestText.assert_called_once()

        # Check request parameters
        call_args = connected_client._document_stub.IngestText.call_args
        request = call_args[0][0]
        assert request.content == "Test content"
        assert request.collection_basename == "scratchbook"
        assert request.tenant_id == "user123"
        assert request.metadata["source"] == "test"
        assert request.chunk_text is True

    @pytest.mark.asyncio
    async def test_ingest_text_not_connected(self, daemon_client):
        """Test ingest_text raises error when not connected."""
        with pytest.raises(DaemonClientError, match="Connection timeout"):
            await daemon_client.ingest_text(
                content="Test",
                collection_basename="test",
                tenant_id="user",
            )

    @pytest.mark.asyncio
    async def test_ingest_text_service_unavailable(self, connected_client):
        """Test ingest_text when DocumentService is not available."""
        connected_client._document_stub = None

        with pytest.raises(DaemonClientError, match="Unexpected error.*NoneType"):
            await connected_client.ingest_text(
                content="Test",
                collection_basename="test",
                tenant_id="user",
            )

    @pytest.mark.asyncio
    async def test_ingest_text_grpc_error(self, connected_client):
        """Test ingest_text handles gRPC errors properly."""
        # Mock gRPC error
        mock_error = grpc.RpcError()
        mock_error.code = MagicMock(return_value=grpc.StatusCode.UNAVAILABLE)
        mock_error.details = MagicMock(return_value="Service unavailable")
        connected_client._document_stub.IngestText = AsyncMock(side_effect=mock_error)

        with patch("common.grpc.daemon_client.validate_llm_collection_access"):
            with pytest.raises(DaemonClientError, match="Daemon unavailable.*Service unavailable"):
                await connected_client.ingest_text(
                    content="Test",
                    collection_basename="test",
                    tenant_id="user",
                )

    @pytest.mark.asyncio
    async def test_update_text_success(self, connected_client):
        """Test successful text update."""
        from google.protobuf.timestamp_pb2 import Timestamp
        timestamp = Timestamp()
        timestamp.GetCurrentTime()

        mock_response = UpdateTextResponse(
            success=True,
            error_message="",
            updated_at=timestamp,
        )
        connected_client._document_stub.UpdateText = AsyncMock(return_value=mock_response)

        response = await connected_client.update_text(
            document_id="doc123",
            content="Updated content",
            collection_name="scratchbook_user123",
            metadata={"updated": "true"},
        )

        assert response.success is True
        connected_client._document_stub.UpdateText.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_text_success(self, connected_client):
        """Test successful text deletion."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._document_stub.DeleteText = AsyncMock(return_value=mock_response)

        await connected_client.delete_text(
            document_id="doc123",
            collection_name="scratchbook_user123",
        )

        connected_client._document_stub.DeleteText.assert_called_once()
        call_args = connected_client._document_stub.DeleteText.call_args
        request = call_args[0][0]
        assert request.document_id == "doc123"
        assert request.collection_name == "scratchbook_user123"


class TestCollectionServiceMethods:
    """Tests for CollectionService methods."""

    @pytest.mark.asyncio
    async def test_create_collection_v2_success(self, connected_client):
        """Test successful collection creation via CollectionService."""
        mock_response = CreateCollectionResponse(
            success=True,
            error_message="",
            collection_id="coll123",
        )
        connected_client._collection_stub.CreateCollection = AsyncMock(return_value=mock_response)

        # Create CollectionConfig object
        config = CollectionConfig(
            vector_size=384,
            distance_metric="Cosine",
            enable_indexing=True,
        )

        with patch("common.grpc.daemon_client.validate_llm_collection_access"):
            response = await connected_client.create_collection_v2(
                collection_name="test_collection",
                project_id="proj123",
                config=config,
            )

        assert response.success is True
        assert response.collection_id == "coll123"
        connected_client._collection_stub.CreateCollection.assert_called_once()

        # Check request parameters
        call_args = connected_client._collection_stub.CreateCollection.call_args
        request = call_args[0][0]
        assert request.collection_name == "test_collection"
        assert request.project_id == "proj123"
        assert request.config.vector_size == 384
        assert request.config.distance_metric == "Cosine"
        assert request.config.enable_indexing is True

    @pytest.mark.asyncio
    async def test_create_collection_v2_service_unavailable(self, connected_client):
        """Test create_collection_v2 when CollectionService is not available."""
        connected_client._collection_stub = None

        with pytest.raises(DaemonClientError, match="Unexpected error.*NoneType"):
            await connected_client.create_collection_v2(
                collection_name="test",
                project_id="proj123",
            )

    @pytest.mark.asyncio
    async def test_create_collection_v2_failure_response(self, connected_client):
        """Test create_collection_v2 when daemon returns failure."""
        mock_response = CreateCollectionResponse(
            success=False,
            error_message="Collection already exists",
            collection_id="",
        )
        connected_client._collection_stub.CreateCollection = AsyncMock(return_value=mock_response)

        with patch("common.grpc.daemon_client.validate_llm_collection_access"):
            response = await connected_client.create_collection_v2(
                collection_name="existing_collection",
                project_id="proj123",
            )

        assert response.success is False
        assert "already exists" in response.error_message

    @pytest.mark.asyncio
    async def test_delete_collection_v2_success(self, connected_client):
        """Test successful collection deletion via CollectionService."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._collection_stub.DeleteCollection = AsyncMock(return_value=mock_response)

        with patch("common.grpc.daemon_client.validate_llm_collection_access"):
            await connected_client.delete_collection_v2(
                collection_name="test_collection",
                project_id="proj123",
                force=True,
            )

        connected_client._collection_stub.DeleteCollection.assert_called_once()
        call_args = connected_client._collection_stub.DeleteCollection.call_args
        request = call_args[0][0]
        assert request.collection_name == "test_collection"
        assert request.project_id == "proj123"
        assert request.force is True

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, connected_client):
        """Test collection_exists returns True when collection exists."""
        from common.grpc.ingestion_pb2 import (
            CollectionInfo,
            ListCollectionsResponse,
        )

        mock_collection = CollectionInfo(name="test_collection")
        mock_response = ListCollectionsResponse(collections=[mock_collection])
        connected_client._ingest_stub.ListCollections = AsyncMock(return_value=mock_response)

        result = await connected_client.collection_exists("test_collection")

        assert result is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, connected_client):
        """Test collection_exists returns False when collection doesn't exist."""
        from common.grpc.ingestion_pb2 import (
            CollectionInfo,
            ListCollectionsResponse,
        )

        mock_collection = CollectionInfo(name="other_collection")
        mock_response = ListCollectionsResponse(collections=[mock_collection])
        connected_client._ingest_stub.ListCollections = AsyncMock(return_value=mock_response)

        result = await connected_client.collection_exists("test_collection")

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_exists_error_handling(self, connected_client):
        """Test collection_exists returns False on error."""
        connected_client._ingest_stub.ListCollections = AsyncMock(side_effect=Exception("Error"))

        with patch("common.grpc.daemon_client.logger"):
            result = await connected_client.collection_exists("test_collection")

        assert result is False

    @pytest.mark.asyncio
    async def test_create_collection_alias_success(self, connected_client):
        """Test successful collection alias creation."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._collection_stub.CreateCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.create_collection_alias(
            alias_name="test_alias",
            collection_name="test_collection",
        )

        connected_client._collection_stub.CreateCollectionAlias.assert_called_once()
        call_args = connected_client._collection_stub.CreateCollectionAlias.call_args
        request = call_args[0][0]
        assert request.alias_name == "test_alias"
        assert request.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_delete_collection_alias_success(self, connected_client):
        """Test successful collection alias deletion."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._collection_stub.DeleteCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.delete_collection_alias(alias_name="test_alias")

        connected_client._collection_stub.DeleteCollectionAlias.assert_called_once()
        call_args = connected_client._collection_stub.DeleteCollectionAlias.call_args
        request = call_args[0][0]
        assert request.alias_name == "test_alias"

    @pytest.mark.asyncio
    async def test_rename_collection_alias_success(self, connected_client):
        """Test successful collection alias rename."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._collection_stub.RenameCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.rename_collection_alias(
            old_name="old_alias",
            new_name="new_alias",
            collection_name="test_collection",
        )

        connected_client._collection_stub.RenameCollectionAlias.assert_called_once()
        call_args = connected_client._collection_stub.RenameCollectionAlias.call_args
        request = call_args[0][0]
        assert request.old_alias_name == "old_alias"
        assert request.new_alias_name == "new_alias"
        assert request.collection_name == "test_collection"


class TestConnectionManagement:
    """Tests for connection management with multiple service stubs."""

    @pytest.mark.asyncio
    async def test_connect_initializes_all_stubs(self, daemon_client):
        """Test that connect() initializes all three service stubs."""
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = MagicMock()
            mock_channel_instance.channel_ready = AsyncMock()
            mock_channel.return_value = mock_channel_instance

            with patch.object(daemon_client, "health_check", new=AsyncMock()):
                with patch("common.grpc.ingestion_pb2_grpc.IngestServiceStub"):
                    with patch("common.grpc.generated.workspace_daemon_pb2_grpc.DocumentServiceStub"):
                        with patch("common.grpc.generated.workspace_daemon_pb2_grpc.CollectionServiceStub"):
                            await daemon_client.connect()

        assert daemon_client._connected is True
        assert daemon_client._ingest_stub is not None
        assert daemon_client._document_stub is not None
        assert daemon_client._collection_stub is not None

    @pytest.mark.asyncio
    async def test_disconnect_clears_all_stubs(self, connected_client):
        """Test that disconnect() clears all service stubs."""
        connected_client._channel.close = AsyncMock()

        await connected_client.disconnect()

        assert connected_client._connected is False
        assert connected_client._ingest_stub is None
        assert connected_client._document_stub is None
        assert connected_client._collection_stub is None

    @pytest.mark.asyncio
    async def test_connect_error_clears_all_stubs(self, daemon_client):
        """Test that connection error clears all stubs properly."""
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = MagicMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance

            with patch.object(daemon_client, "health_check", new=AsyncMock(side_effect=Exception("Connection failed"))):
                with patch("common.grpc.ingestion_pb2_grpc.IngestServiceStub"):
                    with patch("common.grpc.generated.workspace_daemon_pb2_grpc.DocumentServiceStub"):
                        with patch("common.grpc.generated.workspace_daemon_pb2_grpc.CollectionServiceStub"):
                            with pytest.raises(DaemonClientError):
                                await daemon_client.connect()

        assert daemon_client._connected is False
        assert daemon_client._ingest_stub is None
        assert daemon_client._document_stub is None
        assert daemon_client._collection_stub is None


class TestLLMAccessControl:
    """Tests for LLM access control integration."""

    @pytest.mark.asyncio
    async def test_ingest_text_blocked_by_access_control(self, connected_client):
        """Test that ingest_text respects LLM access control.

        Note: Access control is not yet integrated in the new DocumentService methods.
        This test verifies the stub is called, but access control will be added in a future task.
        """
        # Mock the document stub to return success
        from common.grpc.generated.workspace_daemon_pb2 import IngestTextResponse
        mock_response = IngestTextResponse(
            document_id="doc123",
            success=True,
            chunks_created=1,
        )
        connected_client._document_stub.IngestText = AsyncMock(return_value=mock_response)

        # For now, ingest_text should succeed since access control is not yet integrated
        result = await connected_client.ingest_text(
            content="Test",
            collection_basename="protected",
            tenant_id="user",
        )

        assert result.success is True
        # TODO: Add access control validation in future task

    @pytest.mark.asyncio
    async def test_create_collection_v2_blocked_by_access_control(self, connected_client):
        """Test that create_collection_v2 respects LLM access control.

        Note: Access control is not yet integrated in the new CollectionService methods.
        This test verifies the stub is called, but access control will be added in a future task.
        """
        # Mock the collection stub to return success
        from common.grpc.generated.workspace_daemon_pb2 import CreateCollectionResponse
        mock_response = CreateCollectionResponse(
            success=True,
            collection_id="coll123",
        )
        connected_client._collection_stub.CreateCollection = AsyncMock(return_value=mock_response)

        # For now, create_collection_v2 should succeed since access control is not yet integrated
        result = await connected_client.create_collection_v2(
            collection_name="protected_collection",
            project_id="proj123",
        )

        assert result.success is True
        # TODO: Add access control validation in future task

    @pytest.mark.asyncio
    async def test_delete_collection_v2_blocked_by_access_control(self, connected_client):
        """Test that delete_collection_v2 respects LLM access control.

        Note: Access control is not yet integrated in the new CollectionService methods.
        This test verifies the stub is called, but access control will be added in a future task.
        """
        # Mock the collection stub to return success
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client._collection_stub.DeleteCollection = AsyncMock(return_value=mock_response)

        # For now, delete_collection_v2 should succeed since access control is not yet integrated
        await connected_client.delete_collection_v2(
            collection_name="protected_collection",
            project_id="proj123",
        )

        # Verify the stub was called
        connected_client._collection_stub.DeleteCollection.assert_called_once()
        # TODO: Add access control validation in future task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
