"""
Unit tests for DaemonClient.

Tests all 15 RPCs with mock gRPC server responses, error handling,
timeout behavior, retry logic, and circuit breaker functionality.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from common.grpc.connection_manager import ConnectionConfig
from common.grpc.daemon_client import (
    DaemonClient,
    DaemonClientError,
    DaemonTimeoutError,
    DaemonUnavailableError,
)
from common.grpc.generated import workspace_daemon_pb2 as pb2
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp


@pytest.fixture
def mock_channel():
    """Create a mock gRPC channel."""
    channel = AsyncMock()
    channel.channel_ready = AsyncMock()
    channel.close = AsyncMock()
    return channel


@pytest.fixture
def mock_system_stub():
    """Create a mock SystemService stub."""
    stub = AsyncMock()

    # Health check response
    health_response = pb2.HealthCheckResponse(
        status=pb2.SERVICE_STATUS_HEALTHY,
        timestamp=Timestamp()
    )
    stub.HealthCheck = AsyncMock(return_value=health_response)

    # Status response
    status_response = pb2.SystemStatusResponse(
        status=pb2.SERVICE_STATUS_HEALTHY,
        active_projects=["project1", "project2"],
        total_documents=100,
        total_collections=5,
    )
    stub.GetStatus = AsyncMock(return_value=status_response)

    # Metrics response
    metrics_response = pb2.MetricsResponse(collected_at=Timestamp())
    stub.GetMetrics = AsyncMock(return_value=metrics_response)

    # Other methods return Empty
    stub.SendRefreshSignal = AsyncMock(return_value=Empty())
    stub.NotifyServerStatus = AsyncMock(return_value=Empty())
    stub.PauseAllWatchers = AsyncMock(return_value=Empty())
    stub.ResumeAllWatchers = AsyncMock(return_value=Empty())

    return stub


@pytest.fixture
def mock_collection_stub():
    """Create a mock CollectionService stub."""
    stub = AsyncMock()

    # Create collection response
    create_response = pb2.CreateCollectionResponse(
        success=True,
        collection_id="collection_123"
    )
    stub.CreateCollection = AsyncMock(return_value=create_response)

    # Other methods return Empty
    stub.DeleteCollection = AsyncMock(return_value=Empty())
    stub.CreateCollectionAlias = AsyncMock(return_value=Empty())
    stub.DeleteCollectionAlias = AsyncMock(return_value=Empty())
    stub.RenameCollectionAlias = AsyncMock(return_value=Empty())

    return stub


@pytest.fixture
def mock_document_stub():
    """Create a mock DocumentService stub."""
    stub = AsyncMock()

    # Ingest text response
    ingest_response = pb2.IngestTextResponse(
        document_id="doc_123",
        success=True,
        chunks_created=3
    )
    stub.IngestText = AsyncMock(return_value=ingest_response)

    # Update text response
    update_response = pb2.UpdateTextResponse(
        success=True,
        updated_at=Timestamp()
    )
    stub.UpdateText = AsyncMock(return_value=update_response)

    # Delete text returns Empty
    stub.DeleteText = AsyncMock(return_value=Empty())

    return stub


@pytest.fixture
async def daemon_client(mock_channel, mock_system_stub, mock_collection_stub, mock_document_stub):
    """Create a DaemonClient with mocked stubs."""
    config = ConnectionConfig(
        host="localhost",
        port=50051,
        max_retries=2,
        initial_retry_delay=0.1,
        max_retry_delay=0.5,
        connection_timeout=1.0,
    )

    client = DaemonClient(connection_config=config)

    # Patch the channel and stubs
    with patch('common.grpc.daemon_client.grpc.aio.insecure_channel', return_value=mock_channel):
        await client.start()
        # After start, replace the stubs with our mocks
        client._system_stub = mock_system_stub
        client._collection_stub = mock_collection_stub
        client._document_stub = mock_document_stub
        yield client
        await client.stop()


class TestDaemonClientConnection:
    """Test connection management."""

    @pytest.mark.asyncio
    async def test_start_creates_channel(self, mock_channel):
        """Test that start() creates and connects channel."""
        config = ConnectionConfig(connection_timeout=1.0)
        client = DaemonClient(connection_config=config)

        with patch('common.grpc.daemon_client.grpc.aio.insecure_channel', return_value=mock_channel):
            await client.start()

            assert client._started
            assert client._channel is not None
            mock_channel.channel_ready.assert_called_once()

            await client.stop()

    @pytest.mark.asyncio
    async def test_start_timeout_raises_error(self):
        """Test that connection timeout raises DaemonUnavailableError."""
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock(side_effect=asyncio.TimeoutError())

        config = ConnectionConfig(connection_timeout=0.1)
        client = DaemonClient(connection_config=config)

        with patch('common.grpc.daemon_client.grpc.aio.insecure_channel', return_value=mock_channel):
            with pytest.raises(DaemonUnavailableError, match="Connection timeout"):
                await client.start()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_channel):
        """Test async context manager starts and stops client."""
        config = ConnectionConfig(connection_timeout=1.0)

        with patch('common.grpc.daemon_client.grpc.aio.insecure_channel', return_value=mock_channel):
            async with DaemonClient(connection_config=config) as client:
                assert client._started

            assert not client._started
            mock_channel.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_closes_channel(self, daemon_client):
        """Test that stop() closes the channel."""
        await daemon_client.stop()

        assert not daemon_client._started
        assert daemon_client._channel is None


class TestSystemServiceRPCs:
    """Test SystemService RPC methods."""

    @pytest.mark.asyncio
    async def test_health_check(self, daemon_client):
        """Test health_check() returns HealthCheckResponse."""
        response = await daemon_client.health_check()

        assert isinstance(response, pb2.HealthCheckResponse)
        assert response.status == pb2.SERVICE_STATUS_HEALTHY
        daemon_client._system_stub.HealthCheck.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status(self, daemon_client):
        """Test get_status() returns SystemStatusResponse."""
        response = await daemon_client.get_status()

        assert isinstance(response, pb2.SystemStatusResponse)
        assert response.status == pb2.SERVICE_STATUS_HEALTHY
        assert len(response.active_projects) == 2
        assert response.total_documents == 100
        daemon_client._system_stub.GetStatus.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_metrics(self, daemon_client):
        """Test get_metrics() returns MetricsResponse."""
        response = await daemon_client.get_metrics()

        assert isinstance(response, pb2.MetricsResponse)
        daemon_client._system_stub.GetMetrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_refresh_signal(self, daemon_client):
        """Test send_refresh_signal() with queue type."""
        await daemon_client.send_refresh_signal(pb2.INGEST_QUEUE)

        daemon_client._system_stub.SendRefreshSignal.assert_called_once()
        call_args = daemon_client._system_stub.SendRefreshSignal.call_args[0][0]
        assert call_args.queue_type == pb2.INGEST_QUEUE

    @pytest.mark.asyncio
    async def test_send_refresh_signal_with_languages(self, daemon_client):
        """Test send_refresh_signal() with language filters."""
        await daemon_client.send_refresh_signal(
            pb2.TOOLS_AVAILABLE,
            lsp_languages=["python", "rust"],
            grammar_languages=["javascript"]
        )

        call_args = daemon_client._system_stub.SendRefreshSignal.call_args[0][0]
        assert call_args.queue_type == pb2.TOOLS_AVAILABLE
        assert list(call_args.lsp_languages) == ["python", "rust"]
        assert list(call_args.grammar_languages) == ["javascript"]

    @pytest.mark.asyncio
    async def test_notify_server_status(self, daemon_client):
        """Test notify_server_status() with project info."""
        await daemon_client.notify_server_status(
            pb2.SERVER_STATE_UP,
            project_name="myapp",
            project_root="/path/to/myapp"
        )

        call_args = daemon_client._system_stub.NotifyServerStatus.call_args[0][0]
        assert call_args.state == pb2.SERVER_STATE_UP
        assert call_args.project_name == "myapp"
        assert call_args.project_root == "/path/to/myapp"

    @pytest.mark.asyncio
    async def test_pause_all_watchers(self, daemon_client):
        """Test pause_all_watchers()."""
        await daemon_client.pause_all_watchers()

        daemon_client._system_stub.PauseAllWatchers.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_all_watchers(self, daemon_client):
        """Test resume_all_watchers()."""
        await daemon_client.resume_all_watchers()

        daemon_client._system_stub.ResumeAllWatchers.assert_called_once()


class TestCollectionServiceRPCs:
    """Test CollectionService RPC methods."""

    @pytest.mark.asyncio
    async def test_create_collection(self, daemon_client):
        """Test create_collection() with config."""
        config = pb2.CollectionConfig(
            vector_size=768,
            distance_metric="Cosine",
            enable_indexing=True
        )

        response = await daemon_client.create_collection(
            collection_name="myapp-code",
            project_id="project_123",
            config=config
        )

        assert isinstance(response, pb2.CreateCollectionResponse)
        assert response.success
        assert response.collection_id == "collection_123"

        call_args = daemon_client._collection_stub.CreateCollection.call_args[0][0]
        assert call_args.collection_name == "myapp-code"
        assert call_args.project_id == "project_123"
        assert call_args.config.vector_size == 768

    @pytest.mark.asyncio
    async def test_create_collection_default_config(self, daemon_client):
        """Test create_collection() with default config."""
        response = await daemon_client.create_collection(
            collection_name="myapp-code",
            project_id="project_123"
        )

        assert response.success
        call_args = daemon_client._collection_stub.CreateCollection.call_args[0][0]
        assert call_args.config.vector_size == 384  # Default

    @pytest.mark.asyncio
    async def test_delete_collection(self, daemon_client):
        """Test delete_collection()."""
        await daemon_client.delete_collection(
            collection_name="myapp-code",
            project_id="project_123",
            force=True
        )

        call_args = daemon_client._collection_stub.DeleteCollection.call_args[0][0]
        assert call_args.collection_name == "myapp-code"
        assert call_args.project_id == "project_123"
        assert call_args.force is True

    @pytest.mark.asyncio
    async def test_create_collection_alias(self, daemon_client):
        """Test create_collection_alias()."""
        await daemon_client.create_collection_alias(
            alias_name="myapp-code-old",
            collection_name="myapp-code"
        )

        call_args = daemon_client._collection_stub.CreateCollectionAlias.call_args[0][0]
        assert call_args.alias_name == "myapp-code-old"
        assert call_args.collection_name == "myapp-code"

    @pytest.mark.asyncio
    async def test_delete_collection_alias(self, daemon_client):
        """Test delete_collection_alias()."""
        await daemon_client.delete_collection_alias(alias_name="myapp-code-old")

        call_args = daemon_client._collection_stub.DeleteCollectionAlias.call_args[0][0]
        assert call_args.alias_name == "myapp-code-old"

    @pytest.mark.asyncio
    async def test_rename_collection_alias(self, daemon_client):
        """Test rename_collection_alias()."""
        await daemon_client.rename_collection_alias(
            old_name="myapp-code-old",
            new_name="myapp-code-backup",
            collection_name="myapp-code"
        )

        call_args = daemon_client._collection_stub.RenameCollectionAlias.call_args[0][0]
        assert call_args.old_alias_name == "myapp-code-old"
        assert call_args.new_alias_name == "myapp-code-backup"
        assert call_args.collection_name == "myapp-code"


class TestDocumentServiceRPCs:
    """Test DocumentService RPC methods."""

    @pytest.mark.asyncio
    async def test_ingest_text(self, daemon_client):
        """Test ingest_text() with content."""
        response = await daemon_client.ingest_text(
            content="Sample document content",
            collection_basename="myapp-notes",
            tenant_id="project_123",
            metadata={"type": "note", "author": "user1"}
        )

        assert isinstance(response, pb2.IngestTextResponse)
        assert response.success
        assert response.document_id == "doc_123"
        assert response.chunks_created == 3

        call_args = daemon_client._document_stub.IngestText.call_args[0][0]
        assert call_args.content == "Sample document content"
        assert call_args.collection_basename == "myapp-notes"
        assert call_args.tenant_id == "project_123"
        assert call_args.metadata["type"] == "note"
        assert call_args.chunk_text is True

    @pytest.mark.asyncio
    async def test_ingest_text_with_document_id(self, daemon_client):
        """Test ingest_text() with explicit document_id."""
        await daemon_client.ingest_text(
            content="Sample content",
            collection_basename="myapp-notes",
            tenant_id="project_123",
            document_id="custom_doc_id"
        )

        call_args = daemon_client._document_stub.IngestText.call_args[0][0]
        assert call_args.document_id == "custom_doc_id"

    @pytest.mark.asyncio
    async def test_update_text(self, daemon_client):
        """Test update_text()."""
        response = await daemon_client.update_text(
            document_id="doc_123",
            content="Updated content",
            metadata={"updated": "true"}
        )

        assert isinstance(response, pb2.UpdateTextResponse)
        assert response.success

        call_args = daemon_client._document_stub.UpdateText.call_args[0][0]
        assert call_args.document_id == "doc_123"
        assert call_args.content == "Updated content"
        assert call_args.metadata["updated"] == "true"

    @pytest.mark.asyncio
    async def test_update_text_with_collection(self, daemon_client):
        """Test update_text() with collection_name."""
        await daemon_client.update_text(
            document_id="doc_123",
            content="Updated content",
            collection_name="myapp-notes"
        )

        call_args = daemon_client._document_stub.UpdateText.call_args[0][0]
        assert call_args.collection_name == "myapp-notes"

    @pytest.mark.asyncio
    async def test_delete_text(self, daemon_client):
        """Test delete_text()."""
        await daemon_client.delete_text(
            document_id="doc_123",
            collection_name="myapp-notes"
        )

        call_args = daemon_client._document_stub.DeleteText.call_args[0][0]
        assert call_args.document_id == "doc_123"
        assert call_args.collection_name == "myapp-notes"


class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_timeout_raises_daemon_timeout_error(self, daemon_client):
        """Test that operation timeout raises DaemonTimeoutError."""
        daemon_client._system_stub.HealthCheck = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with pytest.raises(DaemonTimeoutError, match="timed out"):
            await daemon_client.health_check(timeout=0.1)

    @pytest.mark.asyncio
    async def test_unavailable_error_raises_daemon_unavailable(self, daemon_client):
        """Test that UNAVAILABLE gRPC error raises DaemonUnavailableError."""
        # Create a proper gRPC RpcError by using grpc.aio.AioRpcError
        mock_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=None,
            trailing_metadata=None,
            details="Service unavailable",
            debug_error_string=None
        )

        daemon_client._system_stub.HealthCheck = AsyncMock(side_effect=mock_error)

        with pytest.raises(DaemonUnavailableError, match="unavailable"):
            await daemon_client.health_check()

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self, daemon_client):
        """Test that non-retryable errors don't retry."""
        mock_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.INVALID_ARGUMENT,
            initial_metadata=None,
            trailing_metadata=None,
            details="Invalid argument",
            debug_error_string=None
        )

        daemon_client._system_stub.HealthCheck = AsyncMock(side_effect=mock_error)

        with pytest.raises(DaemonClientError, match="gRPC error"):
            await daemon_client.health_check()

        # Should be called only once (no retries)
        assert daemon_client._system_stub.HealthCheck.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, daemon_client):
        """Test retry logic on transient failures."""
        mock_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=None,
            trailing_metadata=None,
            details="Temporary unavailable",
            debug_error_string=None
        )

        success_response = pb2.HealthCheckResponse(status=pb2.SERVICE_STATUS_HEALTHY)

        # Fail twice, then succeed
        daemon_client._system_stub.HealthCheck = AsyncMock(
            side_effect=[mock_error, mock_error, success_response]
        )

        response = await daemon_client.health_check()

        assert response.status == pb2.SERVICE_STATUS_HEALTHY
        assert daemon_client._system_stub.HealthCheck.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, daemon_client):
        """Test that max retries is respected."""
        mock_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=None,
            trailing_metadata=None,
            details="Service unavailable",
            debug_error_string=None
        )

        daemon_client._system_stub.HealthCheck = AsyncMock(side_effect=mock_error)

        with pytest.raises(DaemonUnavailableError):
            await daemon_client.health_check()

        # max_retries=2, so should try 3 times total (initial + 2 retries)
        assert daemon_client._system_stub.HealthCheck.call_count == 3


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, daemon_client):
        """Test circuit breaker opens after threshold failures."""
        mock_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=None,
            trailing_metadata=None,
            details="Service unavailable",
            debug_error_string=None
        )

        daemon_client._system_stub.HealthCheck = AsyncMock(side_effect=mock_error)

        # Fail enough times to trigger circuit breaker
        for _ in range(5):
            try:
                await daemon_client.health_check()
            except DaemonUnavailableError:
                pass

        # Circuit breaker should now be open
        assert daemon_client._circuit_breaker_state == "open"

        # Next request should fail immediately
        with pytest.raises(DaemonUnavailableError, match="Circuit breaker is open"):
            await daemon_client.health_check()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self, daemon_client):
        """Test circuit breaker transitions to half-open after timeout."""
        import time

        # Manually set circuit breaker to open state
        daemon_client._circuit_breaker_state = "open"
        daemon_client._circuit_breaker_failures = 5
        # Set last failure to be past the circuit breaker timeout (60.0 seconds)
        daemon_client._circuit_breaker_last_failure = time.time() - 61.0

        # Should transition to half-open
        assert daemon_client._can_attempt_request()

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self, daemon_client):
        """Test circuit breaker closes on successful request in half-open state."""
        daemon_client._circuit_breaker_state = "half-open"
        daemon_client._circuit_breaker_failures = 3

        success_response = pb2.HealthCheckResponse(status=pb2.SERVICE_STATUS_HEALTHY)
        daemon_client._system_stub.HealthCheck = AsyncMock(return_value=success_response)

        await daemon_client.health_check()

        assert daemon_client._circuit_breaker_state == "closed"
        assert daemon_client._circuit_breaker_failures == 0


class TestConnectionInfo:
    """Test connection info reporting."""

    @pytest.mark.asyncio
    async def test_get_connection_info(self, daemon_client):
        """Test get_connection_info() returns complete info."""
        info = daemon_client.get_connection_info()

        assert info["address"] == "localhost:50051"
        assert info["connected"] is True
        assert "circuit_breaker" in info
        assert "connection_pooling" in info
        assert "retry_config" in info

        # Check circuit breaker info
        assert info["circuit_breaker"]["enabled"] is True
        assert info["circuit_breaker"]["state"] == "closed"

        # Check retry config
        assert info["retry_config"]["max_retries"] == 2
