"""
Unit tests for DaemonClient with DocumentService and CollectionService support.

Tests the new gRPC service methods added in Task 375.1.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import grpc

from src.python.common.core.daemon_client import (
    DaemonClient,
    DaemonConnectionError,
)
from src.python.common.grpc.generated.workspace_daemon_pb2 import (
    IngestTextRequest,
    IngestTextResponse,
    UpdateTextRequest,
    UpdateTextResponse,
    DeleteTextRequest,
    CreateCollectionRequest,
    CreateCollectionResponse,
    DeleteCollectionRequest,
    CreateAliasRequest,
    DeleteAliasRequest,
    RenameAliasRequest,
    CollectionConfig,
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
    daemon_client.channel = MagicMock()
    daemon_client.stub = MagicMock()
    daemon_client.document_service = MagicMock()
    daemon_client.collection_service = MagicMock()
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
        connected_client.document_service.IngestText = AsyncMock(return_value=mock_response)

        # Call the method
        with patch("src.python.common.core.daemon_client.validate_llm_collection_access"):
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
        connected_client.document_service.IngestText.assert_called_once()

        # Check request parameters
        call_args = connected_client.document_service.IngestText.call_args
        request = call_args[0][0]
        assert request.content == "Test content"
        assert request.collection_basename == "scratchbook"
        assert request.tenant_id == "user123"
        assert request.metadata["source"] == "test"
        assert request.chunk_text is True

    @pytest.mark.asyncio
    async def test_ingest_text_not_connected(self, daemon_client):
        """Test ingest_text raises error when not connected."""
        with pytest.raises(DaemonConnectionError, match="daemon not connected"):
            await daemon_client.ingest_text(
                content="Test",
                collection_basename="test",
                tenant_id="user",
            )

    @pytest.mark.asyncio
    async def test_ingest_text_service_unavailable(self, connected_client):
        """Test ingest_text when DocumentService is not available."""
        connected_client.document_service = None

        with pytest.raises(DaemonConnectionError, match="DocumentService not available"):
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
        connected_client.document_service.IngestText = AsyncMock(side_effect=mock_error)

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access"):
            with pytest.raises(DaemonConnectionError, match="Failed to ingest text"):
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
        connected_client.document_service.UpdateText = AsyncMock(return_value=mock_response)

        response = await connected_client.update_text(
            document_id="doc123",
            content="Updated content",
            collection_name="scratchbook_user123",
            metadata={"updated": "true"},
        )

        assert response.success is True
        connected_client.document_service.UpdateText.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_text_success(self, connected_client):
        """Test successful text deletion."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client.document_service.DeleteText = AsyncMock(return_value=mock_response)

        await connected_client.delete_text(
            document_id="doc123",
            collection_name="scratchbook_user123",
        )

        connected_client.document_service.DeleteText.assert_called_once()
        call_args = connected_client.document_service.DeleteText.call_args
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
        connected_client.collection_service.CreateCollection = AsyncMock(return_value=mock_response)

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access"):
            response = await connected_client.create_collection_v2(
                collection_name="test_collection",
                project_id="proj123",
                vector_size=384,
                distance_metric="Cosine",
                enable_indexing=True,
                metadata_schema={"type": "test"},
            )

        assert response.success is True
        assert response.collection_id == "coll123"
        connected_client.collection_service.CreateCollection.assert_called_once()

        # Check request parameters
        call_args = connected_client.collection_service.CreateCollection.call_args
        request = call_args[0][0]
        assert request.collection_name == "test_collection"
        assert request.project_id == "proj123"
        assert request.config.vector_size == 384
        assert request.config.distance_metric == "Cosine"
        assert request.config.enable_indexing is True

    @pytest.mark.asyncio
    async def test_create_collection_v2_service_unavailable(self, connected_client):
        """Test create_collection_v2 when CollectionService is not available."""
        connected_client.collection_service = None

        with pytest.raises(DaemonConnectionError, match="CollectionService not available"):
            await connected_client.create_collection_v2(
                collection_name="test",
            )

    @pytest.mark.asyncio
    async def test_create_collection_v2_failure_response(self, connected_client):
        """Test create_collection_v2 when daemon returns failure."""
        mock_response = CreateCollectionResponse(
            success=False,
            error_message="Collection already exists",
            collection_id="",
        )
        connected_client.collection_service.CreateCollection = AsyncMock(return_value=mock_response)

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access"):
            response = await connected_client.create_collection_v2(
                collection_name="existing_collection",
            )

        assert response.success is False
        assert "already exists" in response.error_message

    @pytest.mark.asyncio
    async def test_delete_collection_v2_success(self, connected_client):
        """Test successful collection deletion via CollectionService."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client.collection_service.DeleteCollection = AsyncMock(return_value=mock_response)

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access"):
            await connected_client.delete_collection_v2(
                collection_name="test_collection",
                project_id="proj123",
                force=True,
            )

        connected_client.collection_service.DeleteCollection.assert_called_once()
        call_args = connected_client.collection_service.DeleteCollection.call_args
        request = call_args[0][0]
        assert request.collection_name == "test_collection"
        assert request.project_id == "proj123"
        assert request.force is True

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, connected_client):
        """Test collection_exists returns True when collection exists."""
        from src.python.common.grpc.ingestion_pb2 import (
            ListCollectionsResponse,
            CollectionInfo,
        )

        mock_collection = CollectionInfo(name="test_collection")
        mock_response = ListCollectionsResponse(collections=[mock_collection])
        connected_client.stub.ListCollections = AsyncMock(return_value=mock_response)

        result = await connected_client.collection_exists("test_collection")

        assert result is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, connected_client):
        """Test collection_exists returns False when collection doesn't exist."""
        from src.python.common.grpc.ingestion_pb2 import (
            ListCollectionsResponse,
            CollectionInfo,
        )

        mock_collection = CollectionInfo(name="other_collection")
        mock_response = ListCollectionsResponse(collections=[mock_collection])
        connected_client.stub.ListCollections = AsyncMock(return_value=mock_response)

        result = await connected_client.collection_exists("test_collection")

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_exists_error_handling(self, connected_client):
        """Test collection_exists returns False on error."""
        connected_client.stub.ListCollections = AsyncMock(side_effect=Exception("Error"))

        result = await connected_client.collection_exists("test_collection")

        assert result is False

    @pytest.mark.asyncio
    async def test_create_collection_alias_success(self, connected_client):
        """Test successful collection alias creation."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client.collection_service.CreateCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.create_collection_alias(
            alias_name="test_alias",
            collection_name="test_collection",
        )

        connected_client.collection_service.CreateCollectionAlias.assert_called_once()
        call_args = connected_client.collection_service.CreateCollectionAlias.call_args
        request = call_args[0][0]
        assert request.alias_name == "test_alias"
        assert request.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_delete_collection_alias_success(self, connected_client):
        """Test successful collection alias deletion."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client.collection_service.DeleteCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.delete_collection_alias(alias_name="test_alias")

        connected_client.collection_service.DeleteCollectionAlias.assert_called_once()
        call_args = connected_client.collection_service.DeleteCollectionAlias.call_args
        request = call_args[0][0]
        assert request.alias_name == "test_alias"

    @pytest.mark.asyncio
    async def test_rename_collection_alias_success(self, connected_client):
        """Test successful collection alias rename."""
        from google.protobuf.empty_pb2 import Empty
        mock_response = Empty()
        connected_client.collection_service.RenameCollectionAlias = AsyncMock(return_value=mock_response)

        await connected_client.rename_collection_alias(
            old_alias_name="old_alias",
            new_alias_name="new_alias",
            collection_name="test_collection",
        )

        connected_client.collection_service.RenameCollectionAlias.assert_called_once()
        call_args = connected_client.collection_service.RenameCollectionAlias.call_args
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
            mock_channel.return_value = mock_channel_instance

            with patch.object(daemon_client, "health_check", new=AsyncMock()):
                with patch("src.python.common.grpc.ingestion_pb2_grpc.IngestServiceStub"):
                    with patch("src.python.common.grpc.generated.workspace_daemon_pb2_grpc.DocumentServiceStub"):
                        with patch("src.python.common.grpc.generated.workspace_daemon_pb2_grpc.CollectionServiceStub"):
                            await daemon_client.connect()

        assert daemon_client._connected is True
        assert daemon_client.stub is not None
        assert daemon_client.document_service is not None
        assert daemon_client.collection_service is not None

    @pytest.mark.asyncio
    async def test_disconnect_clears_all_stubs(self, connected_client):
        """Test that disconnect() clears all service stubs."""
        connected_client.channel.close = AsyncMock()

        await connected_client.disconnect()

        assert connected_client._connected is False
        assert connected_client.stub is None
        assert connected_client.document_service is None
        assert connected_client.collection_service is None

    @pytest.mark.asyncio
    async def test_connect_error_clears_all_stubs(self, daemon_client):
        """Test that connection error clears all stubs properly."""
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = MagicMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance

            with patch.object(daemon_client, "health_check", new=AsyncMock(side_effect=Exception("Connection failed"))):
                with patch("src.python.common.grpc.ingestion_pb2_grpc.IngestServiceStub"):
                    with patch("src.python.common.grpc.generated.workspace_daemon_pb2_grpc.DocumentServiceStub"):
                        with patch("src.python.common.grpc.generated.workspace_daemon_pb2_grpc.CollectionServiceStub"):
                            with pytest.raises(DaemonConnectionError):
                                await daemon_client.connect()

        assert daemon_client._connected is False
        assert daemon_client.stub is None
        assert daemon_client.document_service is None
        assert daemon_client.collection_service is None


class TestLLMAccessControl:
    """Tests for LLM access control integration."""

    @pytest.mark.asyncio
    async def test_ingest_text_blocked_by_access_control(self, connected_client):
        """Test that ingest_text respects LLM access control."""
        from src.python.common.core.llm_access_control import (
            LLMAccessControlError,
            AccessViolation,
            AccessViolationType,
        )

        # Create a proper AccessViolation object
        violation = AccessViolation(
            violation_type=AccessViolationType.FORBIDDEN_SYSTEM_WRITE,
            collection_name="protected_user",
            operation="write",
            message="Access denied",
        )

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access") as mock_validate:
            mock_validate.side_effect = LLMAccessControlError(violation)

            with pytest.raises(DaemonConnectionError, match="Text ingestion blocked"):
                await connected_client.ingest_text(
                    content="Test",
                    collection_basename="protected",
                    tenant_id="user",
                )

    @pytest.mark.asyncio
    async def test_create_collection_v2_blocked_by_access_control(self, connected_client):
        """Test that create_collection_v2 respects LLM access control."""
        from src.python.common.core.llm_access_control import (
            LLMAccessControlError,
            AccessViolation,
            AccessViolationType,
        )

        # Create a proper AccessViolation object
        violation = AccessViolation(
            violation_type=AccessViolationType.FORBIDDEN_SYSTEM_CREATION,
            collection_name="protected_collection",
            operation="create",
            message="Access denied",
        )

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access") as mock_validate:
            mock_validate.side_effect = LLMAccessControlError(violation)

            with pytest.raises(DaemonConnectionError, match="Collection creation blocked"):
                await connected_client.create_collection_v2(
                    collection_name="protected_collection",
                )

    @pytest.mark.asyncio
    async def test_delete_collection_v2_blocked_by_access_control(self, connected_client):
        """Test that delete_collection_v2 respects LLM access control."""
        from src.python.common.core.llm_access_control import (
            LLMAccessControlError,
            AccessViolation,
            AccessViolationType,
        )

        # Create a proper AccessViolation object
        violation = AccessViolation(
            violation_type=AccessViolationType.FORBIDDEN_SYSTEM_DELETION,
            collection_name="protected_collection",
            operation="delete",
            message="Access denied",
        )

        with patch("src.python.common.core.daemon_client.validate_llm_collection_access") as mock_validate:
            mock_validate.side_effect = LLMAccessControlError(violation)

            with pytest.raises(DaemonConnectionError, match="Collection deletion blocked"):
                await connected_client.delete_collection_v2(
                    collection_name="protected_collection",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
