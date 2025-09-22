"""
Comprehensive unit tests for python.common.grpc.client module.

Tests cover AsyncIngestClient and all gRPC communication functionality
with 100% coverage including async patterns, streaming, and error handling.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any, AsyncIterator

import grpc
from google.protobuf.empty_pb2 import Empty

# Import modules under test
from src.python.common.grpc.client import AsyncIngestClient
from src.python.common.grpc.connection_manager import ConnectionConfig
from src.python.common.grpc.types import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    HealthCheckResponse,
)


class TestAsyncIngestClient:
    """Test AsyncIngestClient functionality."""

    @pytest.fixture
    def mock_connection_config(self):
        """Create a mock ConnectionConfig."""
        return Mock(spec=ConnectionConfig, host="127.0.0.1", port=50051)

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock GrpcConnectionManager."""
        manager = Mock()
        manager.start = AsyncMock()
        manager.stop = AsyncMock()
        manager.with_retry = AsyncMock()
        manager.with_stream = AsyncMock()
        manager.get_stub = Mock()
        manager.get_connection_info = Mock(return_value={"status": "connected"})
        return manager

    @pytest.fixture
    def client(self, mock_connection_manager):
        """Create AsyncIngestClient instance with mocked connection manager."""
        with patch('src.python.common.grpc.client.GrpcConnectionManager') as mock_mgr_class:
            mock_mgr_class.return_value = mock_connection_manager
            return AsyncIngestClient(host="localhost", port=8080)

    def test_init_with_default_params(self, mock_connection_manager):
        """Test client initialization with default parameters."""
        with patch('src.python.common.grpc.client.GrpcConnectionManager') as mock_mgr_class:
            mock_mgr_class.return_value = mock_connection_manager

            client = AsyncIngestClient()

            assert client.connection_config.host == "127.0.0.1"
            assert client.connection_config.port == 50051
            assert client._started == False

    def test_init_with_custom_params(self, mock_connection_manager):
        """Test client initialization with custom parameters."""
        with patch('src.python.common.grpc.client.GrpcConnectionManager') as mock_mgr_class:
            mock_mgr_class.return_value = mock_connection_manager

            client = AsyncIngestClient(host="custom-host", port=9999)

            assert client.connection_config.host == "custom-host"
            assert client.connection_config.port == 9999

    def test_init_with_connection_config(self, mock_connection_config, mock_connection_manager):
        """Test client initialization with provided connection config."""
        with patch('src.python.common.grpc.client.GrpcConnectionManager') as mock_mgr_class:
            mock_mgr_class.return_value = mock_connection_manager

            client = AsyncIngestClient(connection_config=mock_connection_config)

            assert client.connection_config == mock_connection_config

    async def test_start(self, client, mock_connection_manager):
        """Test starting the client."""
        await client.start()

        assert client._started == True
        mock_connection_manager.start.assert_called_once()

    async def test_start_idempotent(self, client, mock_connection_manager):
        """Test that starting an already started client is idempotent."""
        await client.start()
        await client.start()  # Second call

        # Should only call start once
        mock_connection_manager.start.assert_called_once()

    async def test_stop(self, client, mock_connection_manager):
        """Test stopping the client."""
        await client.start()
        await client.stop()

        assert client._started == False
        mock_connection_manager.stop.assert_called_once()

    async def test_stop_not_started(self, client, mock_connection_manager):
        """Test stopping a client that wasn't started."""
        await client.stop()

        # Should not call stop if not started
        mock_connection_manager.stop.assert_not_called()

    async def test_async_context_manager(self, client, mock_connection_manager):
        """Test using client as async context manager."""
        async with client as ctx_client:
            assert ctx_client == client
            assert client._started == True

        assert client._started == False
        mock_connection_manager.start.assert_called_once()
        mock_connection_manager.stop.assert_called_once()

    async def test_process_document(self, client, mock_connection_manager):
        """Test processing a document."""
        # Mock response
        mock_response = Mock(spec=ProcessDocumentResponse)
        mock_response.success = True
        mock_response.document_id = "doc123"

        # Mock the retry mechanism
        async def mock_with_retry(func):
            # Create a mock stub
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        # Mock ProcessDocumentRequest and Response
        with patch('src.python.common.grpc.client.ProcessDocumentRequest') as mock_req_class:
            with patch('src.python.common.grpc.client.ProcessDocumentResponse') as mock_resp_class:
                mock_request = Mock()
                mock_request.to_pb.return_value = Mock()
                mock_req_class.return_value = mock_request

                mock_resp_class.from_pb.return_value = mock_response

                result = await client.process_document(
                    file_path="/test/file.txt",
                    collection="test-collection",
                    metadata={"key": "value"},
                    document_id="custom-id",
                    chunk_text=False,
                    timeout=10.0
                )

                assert result == mock_response
                mock_req_class.assert_called_once_with(
                    file_path="/test/file.txt",
                    collection="test-collection",
                    metadata={"key": "value"},
                    document_id="custom-id",
                    chunk_text=False
                )

    async def test_process_document_auto_start(self, client, mock_connection_manager):
        """Test that process_document auto-starts the client."""
        mock_response = Mock(spec=ProcessDocumentResponse)

        async def mock_with_retry(func):
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.ProcessDocumentRequest'):
            with patch('src.python.common.grpc.client.ProcessDocumentResponse') as mock_resp_class:
                mock_resp_class.from_pb.return_value = mock_response

                await client.process_document("/test/file.txt", "collection")

                # Should auto-start
                mock_connection_manager.start.assert_called_once()

    async def test_process_document_timeout(self, client, mock_connection_manager):
        """Test process_document with timeout."""
        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.ProcessDocument = AsyncMock(side_effect=asyncio.TimeoutError())

            # The function should raise TimeoutError when asyncio.wait_for times out
            with pytest.raises(asyncio.TimeoutError):
                await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.ProcessDocumentRequest'):
            with pytest.raises(asyncio.TimeoutError):
                await client.process_document("/test/file.txt", "collection", timeout=0.1)

    async def test_execute_query(self, client, mock_connection_manager):
        """Test executing a search query."""
        mock_response = Mock(spec=ExecuteQueryResponse)
        mock_response.results = []
        mock_response.total_hits = 0

        async def mock_with_retry(func):
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.ExecuteQueryRequest') as mock_req_class:
            with patch('src.python.common.grpc.client.ExecuteQueryResponse') as mock_resp_class:
                mock_request = Mock()
                mock_request.to_pb.return_value = Mock()
                mock_req_class.return_value = mock_request

                mock_resp_class.from_pb.return_value = mock_response

                result = await client.execute_query(
                    query="test query",
                    collections=["col1", "col2"],
                    mode="sparse",
                    limit=20,
                    score_threshold=0.8,
                    timeout=30.0
                )

                assert result == mock_response
                mock_req_class.assert_called_once_with(
                    query="test query",
                    collections=["col1", "col2"],
                    mode="sparse",
                    limit=20,
                    score_threshold=0.8
                )

    async def test_execute_query_defaults(self, client, mock_connection_manager):
        """Test execute_query with default parameters."""
        mock_response = Mock(spec=ExecuteQueryResponse)

        async def mock_with_retry(func):
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.ExecuteQueryRequest') as mock_req_class:
            with patch('src.python.common.grpc.client.ExecuteQueryResponse') as mock_resp_class:
                mock_request = Mock()
                mock_request.to_pb.return_value = Mock()
                mock_req_class.return_value = mock_request
                mock_resp_class.from_pb.return_value = mock_response

                await client.execute_query("test query")

                mock_req_class.assert_called_once_with(
                    query="test query",
                    collections=None,
                    mode="hybrid",
                    limit=10,
                    score_threshold=0.7
                )

    async def test_health_check(self, client, mock_connection_manager):
        """Test health check."""
        mock_response = Mock(spec=HealthCheckResponse)
        mock_response.status = "healthy"

        async def mock_with_retry(func):
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.HealthCheckResponse') as mock_resp_class:
            mock_resp_class.from_pb.return_value = mock_response

            result = await client.health_check(timeout=3.0)

            assert result == mock_response

    async def test_health_check_auto_start(self, client, mock_connection_manager):
        """Test that health_check auto-starts the client."""
        mock_response = Mock(spec=HealthCheckResponse)

        async def mock_with_retry(func):
            mock_stub = Mock()
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with patch('src.python.common.grpc.client.HealthCheckResponse') as mock_resp_class:
            mock_resp_class.from_pb.return_value = mock_response

            await client.health_check()

            mock_connection_manager.start.assert_called_once()

    async def test_start_watching(self, client, mock_connection_manager):
        """Test starting file watching."""
        # Mock watch events
        mock_event1 = Mock()
        mock_event1.watch_id = "watch123"
        mock_event1.event_type = "created"
        mock_event1.file_path = "/test/file1.txt"
        mock_event1.timestamp.ToDatetime.return_value = "2023-01-01T00:00:00Z"
        mock_event1.status = "success"
        mock_event1.HasField.return_value = False

        mock_event2 = Mock()
        mock_event2.watch_id = "watch123"
        mock_event2.event_type = "modified"
        mock_event2.file_path = "/test/file2.txt"
        mock_event2.timestamp.ToDatetime.return_value = "2023-01-01T00:01:00Z"
        mock_event2.status = "error"
        mock_event2.HasField.return_value = True
        mock_event2.error_message = "Processing failed"

        # Mock async iterator
        async def mock_start_watching(request):
            yield mock_event1
            yield mock_event2

        # Mock connection manager context
        mock_stub = Mock()
        mock_stub.StartWatching = mock_start_watching

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_stub)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_stub.return_value = mock_context

        events = []
        async for event in client.start_watching(
            path="/test/dir",
            collection="test-collection",
            patterns=["*.txt"],
            ignore_patterns=["*.tmp"],
            auto_ingest=True,
            recursive=True,
            recursive_depth=5,
            debounce_seconds=3,
            update_frequency_ms=500,
            watch_id="custom-watch-id"
        ):
            events.append(event)

        assert len(events) == 2

        # Check first event
        assert events[0]["watch_id"] == "watch123"
        assert events[0]["event_type"] == "created"
        assert events[0]["file_path"] == "/test/file1.txt"
        assert events[0]["status"] == "success"
        assert "error_message" not in events[0]

        # Check second event (with error)
        assert events[1]["watch_id"] == "watch123"
        assert events[1]["event_type"] == "modified"
        assert events[1]["file_path"] == "/test/file2.txt"
        assert events[1]["status"] == "error"
        assert events[1]["error_message"] == "Processing failed"

    async def test_start_watching_auto_start(self, client, mock_connection_manager):
        """Test that start_watching auto-starts the client."""
        async def mock_start_watching(request):
            return
            yield  # Make it an async generator

        mock_stub = Mock()
        mock_stub.StartWatching = mock_start_watching

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_stub)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_stub.return_value = mock_context

        # Consume the async iterator
        async for _ in client.start_watching("/test", "collection"):
            break

        mock_connection_manager.start.assert_called_once()

    async def test_stop_watching(self, client, mock_connection_manager):
        """Test stopping file watching."""
        mock_response = Mock()
        mock_response.success = True
        mock_response.message = "Watch stopped"

        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.StopWatching = AsyncMock(return_value=mock_response)
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        result = await client.stop_watching("watch123", timeout=5.0)

        assert result == {"success": True, "message": "Watch stopped"}

    async def test_stop_watching_timeout(self, client, mock_connection_manager):
        """Test stop_watching with timeout."""
        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.StopWatching = AsyncMock(side_effect=asyncio.TimeoutError())

            with pytest.raises(asyncio.TimeoutError):
                await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with pytest.raises(asyncio.TimeoutError):
            await client.stop_watching("watch123", timeout=0.1)

    async def test_get_stats(self, client, mock_connection_manager):
        """Test getting engine statistics."""
        # Mock protobuf response
        mock_response = Mock()

        # Engine stats
        mock_engine_stats = Mock()
        mock_engine_stats.started_at.ToDatetime.return_value = "2023-01-01T00:00:00Z"
        mock_engine_stats.uptime.total_seconds.return_value = 3600.0
        mock_engine_stats.total_documents_processed = 100
        mock_engine_stats.total_documents_indexed = 95
        mock_engine_stats.active_watches = 3
        mock_engine_stats.version = "1.0.0"
        mock_response.engine_stats = mock_engine_stats

        # Collection stats
        mock_col_stat = Mock()
        mock_col_stat.name = "test-collection"
        mock_col_stat.document_count = 50
        mock_col_stat.total_size_bytes = 1024000
        mock_col_stat.last_updated.ToDatetime.return_value = "2023-01-01T00:30:00Z"
        mock_response.collection_stats = [mock_col_stat]

        # Watch stats
        mock_watch_stat = Mock()
        mock_watch_stat.watch_id = "watch123"
        mock_watch_stat.path = "/test/dir"
        mock_watch_stat.collection = "test-collection"
        mock_watch_stat.status = "active"
        mock_watch_stat.files_processed = 25
        mock_watch_stat.files_failed = 2
        mock_watch_stat.created_at.ToDatetime.return_value = "2023-01-01T00:00:00Z"
        mock_watch_stat.last_activity.ToDatetime.return_value = "2023-01-01T00:30:00Z"
        mock_response.watch_stats = [mock_watch_stat]

        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.GetStats = AsyncMock(return_value=mock_response)
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        result = await client.get_stats(
            include_collection_stats=True,
            include_watch_stats=True,
            timeout=15.0
        )

        # Verify engine stats
        assert result["engine_stats"]["version"] == "1.0.0"
        assert result["engine_stats"]["total_documents_processed"] == 100
        assert result["engine_stats"]["uptime_seconds"] == 3600.0

        # Verify collection stats
        assert len(result["collection_stats"]) == 1
        assert result["collection_stats"][0]["name"] == "test-collection"
        assert result["collection_stats"][0]["document_count"] == 50

        # Verify watch stats
        assert len(result["watch_stats"]) == 1
        assert result["watch_stats"][0]["watch_id"] == "watch123"
        assert result["watch_stats"][0]["files_processed"] == 25

    async def test_get_stats_minimal(self, client, mock_connection_manager):
        """Test getting minimal stats."""
        mock_response = Mock()
        mock_engine_stats = Mock()
        mock_engine_stats.started_at.ToDatetime.return_value = "2023-01-01T00:00:00Z"
        mock_engine_stats.uptime.total_seconds.return_value = 3600.0
        mock_engine_stats.total_documents_processed = 100
        mock_engine_stats.total_documents_indexed = 95
        mock_engine_stats.active_watches = 3
        mock_engine_stats.version = "1.0.0"
        mock_response.engine_stats = mock_engine_stats

        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.GetStats = AsyncMock(return_value=mock_response)
            return await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        result = await client.get_stats(
            include_collection_stats=False,
            include_watch_stats=False
        )

        assert "engine_stats" in result
        assert "collection_stats" not in result
        assert "watch_stats" not in result

    def test_get_connection_info(self, client, mock_connection_manager):
        """Test getting connection information."""
        result = client.get_connection_info()

        assert result == {"status": "connected"}
        mock_connection_manager.get_connection_info.assert_called_once()

    async def test_test_connection_success(self, client):
        """Test successful connection test."""
        client.health_check = AsyncMock(return_value=Mock())

        result = await client.test_connection()

        assert result == True
        client.health_check.assert_called_once_with(timeout=5.0)

    async def test_test_connection_failure(self, client):
        """Test failed connection test."""
        client.health_check = AsyncMock(side_effect=Exception("Connection failed"))

        result = await client.test_connection()

        assert result == False

    async def test_stream_processing_status(self, client, mock_connection_manager):
        """Test streaming processing status."""
        # Mock status updates
        mock_update1 = Mock()
        mock_update1.timestamp.ToJsonString.return_value = "2023-01-01T00:00:00Z"

        # Mock active tasks
        mock_task1 = Mock()
        mock_task1.task_id = "task123"
        mock_task1.task_type = "document_processing"
        mock_task1.file_path = "/test/file.txt"
        mock_task1.collection = "test-collection"
        mock_task1.status = "processing"
        mock_task1.progress_percent = 50.0
        mock_task1.started_at.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_task1.HasField.side_effect = lambda field: field == "error_message"
        mock_task1.error_message = "Processing error"
        mock_update1.active_tasks = [mock_task1]

        # Mock recent completed tasks
        mock_completed = Mock()
        mock_completed.task_id = "task122"
        mock_completed.task_type = "document_processing"
        mock_completed.file_path = "/test/completed.txt"
        mock_completed.collection = "test-collection"
        mock_completed.status = "completed"
        mock_completed.progress_percent = 100.0
        mock_completed.started_at.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_completed.completed_at.ToJsonString.return_value = "2023-01-01T00:01:00Z"
        mock_completed.HasField.return_value = False
        mock_update1.recent_completed = [mock_completed]

        # Mock current stats
        mock_stats = Mock()
        mock_stats.total_files_processed = 100
        mock_stats.total_files_failed = 5
        mock_stats.total_files_skipped = 2
        mock_stats.active_tasks = 3
        mock_stats.queued_tasks = 10
        mock_stats.average_processing_time.ToJsonString.return_value = "00:02:30"
        mock_stats.last_activity.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_update1.current_stats = mock_stats
        mock_update1.HasField.side_effect = lambda field: field in ["current_stats", "queue_status"]

        # Mock queue status
        mock_queue = Mock()
        mock_queue.total_queued = 10
        mock_queue.high_priority = 2
        mock_queue.normal_priority = 6
        mock_queue.low_priority = 2
        mock_queue.urgent_priority = 0
        mock_queue.collections_with_queued = ["col1", "col2"]
        mock_queue.estimated_completion_time.ToJsonString.return_value = "2023-01-01T01:00:00Z"
        mock_update1.queue_status = mock_queue

        async def mock_stream_func(func):
            async def inner():
                yield mock_update1
            return inner()

        mock_connection_manager.with_stream.side_effect = mock_stream_func

        # Mock the imported StreamStatusRequest
        with patch('src.python.common.grpc.client.StreamStatusRequest') as mock_req:
            mock_request = Mock()
            mock_req.return_value = mock_request

            updates = []
            async for update in client.stream_processing_status(
                update_interval_seconds=10,
                include_history=True,
                collection_filter="test-collection"
            ):
                updates.append(update)
                break  # Only get first update

            assert len(updates) == 1
            update = updates[0]

            assert update["timestamp"] == "2023-01-01T00:00:00Z"
            assert len(update["active_tasks"]) == 1
            assert update["active_tasks"][0]["task_id"] == "task123"
            assert update["active_tasks"][0]["progress_percent"] == 50.0

            assert len(update["recent_completed"]) == 1
            assert update["recent_completed"][0]["task_id"] == "task122"

            assert update["current_stats"]["total_files_processed"] == 100
            assert update["queue_status"]["total_queued"] == 10

    async def test_stream_system_metrics(self, client, mock_connection_manager):
        """Test streaming system metrics."""
        mock_update = Mock()
        mock_update.timestamp.ToJsonString.return_value = "2023-01-01T00:00:00Z"

        # Mock resource usage
        mock_resource = Mock()
        mock_resource.cpu_percent = 25.5
        mock_resource.memory_bytes = 1024000000
        mock_resource.memory_peak_bytes = 1500000000
        mock_resource.open_files = 150
        mock_resource.active_connections = 5
        mock_resource.disk_usage_percent = 45.0
        mock_update.resource_usage = mock_resource

        # Mock engine stats
        mock_engine = Mock()
        mock_engine.started_at.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_engine.uptime.ToJsonString.return_value = "PT1H30M"
        mock_engine.total_documents_processed = 1000
        mock_engine.total_documents_indexed = 950
        mock_engine.active_watches = 5
        mock_engine.version = "1.0.0"
        mock_update.engine_stats = mock_engine

        # Mock collection stats
        mock_col_stats = Mock()
        mock_col_stats.name = "test-collection"
        mock_col_stats.document_count = 500
        mock_col_stats.total_size_bytes = 50000000
        mock_col_stats.last_updated.ToJsonString.return_value = "2023-01-01T00:30:00Z"
        mock_update.collection_stats = [mock_col_stats]

        # Mock performance metrics
        mock_perf = Mock()
        mock_perf.processing_rate_files_per_hour = 120.5
        mock_perf.average_processing_time.ToJsonString.return_value = "PT30S"
        mock_perf.success_rate_percent = 95.5
        mock_perf.concurrent_tasks = 8
        mock_perf.throughput_bytes_per_second = 1024000

        # Mock bottlenecks
        mock_bottleneck = Mock()
        mock_bottleneck.component = "disk_io"
        mock_bottleneck.description = "High disk I/O latency"
        mock_bottleneck.severity = "medium"
        mock_bottleneck.suggestion = "Consider SSD upgrade"
        mock_perf.bottlenecks = [mock_bottleneck]
        mock_update.performance_metrics = mock_perf

        mock_update.HasField.side_effect = lambda field: field in [
            "resource_usage", "engine_stats", "performance_metrics"
        ]

        async def mock_stream_func(func):
            async def inner():
                yield mock_update
            return inner()

        mock_connection_manager.with_stream.side_effect = mock_stream_func

        with patch('src.python.common.grpc.client.StreamMetricsRequest') as mock_req:
            mock_request = Mock()
            mock_req.return_value = mock_request

            updates = []
            async for update in client.stream_system_metrics(
                update_interval_seconds=15,
                include_detailed_metrics=True
            ):
                updates.append(update)
                break

            assert len(updates) == 1
            update = updates[0]

            assert update["timestamp"] == "2023-01-01T00:00:00Z"

            # Check resource usage
            assert update["resource_usage"]["cpu_percent"] == 25.5
            assert update["resource_usage"]["memory_bytes"] == 1024000000

            # Check engine stats
            assert update["engine_stats"]["version"] == "1.0.0"
            assert update["engine_stats"]["total_documents_processed"] == 1000

            # Check collection stats
            assert len(update["collection_stats"]) == 1
            assert update["collection_stats"][0]["name"] == "test-collection"

            # Check performance metrics
            assert update["performance_metrics"]["processing_rate_files_per_hour"] == 120.5
            assert len(update["performance_metrics"]["bottlenecks"]) == 1
            assert update["performance_metrics"]["bottlenecks"][0]["component"] == "disk_io"

    async def test_stream_queue_status(self, client, mock_connection_manager):
        """Test streaming queue status."""
        mock_update = Mock()
        mock_update.timestamp.ToJsonString.return_value = "2023-01-01T00:00:00Z"

        # Mock queue status
        mock_queue_status = Mock()
        mock_queue_status.total_queued = 15
        mock_queue_status.high_priority = 3
        mock_queue_status.normal_priority = 10
        mock_queue_status.low_priority = 2
        mock_queue_status.urgent_priority = 0
        mock_queue_status.collections_with_queued = ["col1", "col2", "col3"]
        mock_queue_status.estimated_completion_time.ToJsonString.return_value = "2023-01-01T01:00:00Z"
        mock_update.queue_status = mock_queue_status

        # Mock recent additions
        mock_recent_file = Mock()
        mock_recent_file.file_path = "/test/new_file.txt"
        mock_recent_file.collection = "test-collection"
        mock_recent_file.priority = "normal"
        mock_recent_file.queued_at.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_recent_file.file_size_bytes = 1024
        mock_update.recent_additions = [mock_recent_file]

        # Mock active processing
        mock_active_progress = Mock()
        mock_active_progress.task_id = "task456"
        mock_active_progress.file_path = "/test/processing.txt"
        mock_active_progress.collection = "test-collection"
        mock_active_progress.progress_percent = 75.0
        mock_active_progress.current_stage = "indexing"
        mock_active_progress.started_at.ToJsonString.return_value = "2023-01-01T00:00:00Z"
        mock_active_progress.estimated_remaining.ToJsonString.return_value = "PT2M"
        mock_update.active_processing = [mock_active_progress]

        mock_update.HasField.side_effect = lambda field: field == "queue_status"

        async def mock_stream_func(func):
            async def inner():
                yield mock_update
            return inner()

        mock_connection_manager.with_stream.side_effect = mock_stream_func

        with patch('src.python.common.grpc.client.StreamQueueRequest') as mock_req:
            mock_request = Mock()
            mock_req.return_value = mock_request

            updates = []
            async for update in client.stream_queue_status(
                update_interval_seconds=5,
                collection_filter="test-collection"
            ):
                updates.append(update)
                break

            assert len(updates) == 1
            update = updates[0]

            assert update["timestamp"] == "2023-01-01T00:00:00Z"

            # Check queue status
            assert update["queue_status"]["total_queued"] == 15
            assert update["queue_status"]["high_priority"] == 3
            assert len(update["queue_status"]["collections_with_queued"]) == 3

            # Check recent additions
            assert len(update["recent_additions"]) == 1
            assert update["recent_additions"][0]["file_path"] == "/test/new_file.txt"
            assert update["recent_additions"][0]["priority"] == "normal"

            # Check active processing
            assert len(update["active_processing"]) == 1
            assert update["active_processing"][0]["task_id"] == "task456"
            assert update["active_processing"][0]["progress_percent"] == 75.0

    async def test_stream_auto_start(self, client, mock_connection_manager):
        """Test that streaming methods auto-start the client."""
        async def mock_stream_func(func):
            async def inner():
                return
                yield  # Make it async generator
            return inner()

        mock_connection_manager.with_stream.side_effect = mock_stream_func

        with patch('src.python.common.grpc.client.StreamStatusRequest'):
            async for _ in client.stream_processing_status():
                break

        mock_connection_manager.start.assert_called_once()

    async def test_grpc_error_handling(self, client, mock_connection_manager):
        """Test gRPC error handling."""
        async def mock_with_retry(func):
            mock_stub = Mock()
            mock_stub.HealthCheck = AsyncMock(side_effect=grpc.RpcError("Connection failed"))
            await func(mock_stub)

        mock_connection_manager.with_retry.side_effect = mock_with_retry

        with pytest.raises(grpc.RpcError):
            await client.health_check()

    async def test_connection_manager_error_propagation(self, client, mock_connection_manager):
        """Test that connection manager errors are properly propagated."""
        mock_connection_manager.with_retry.side_effect = Exception("Connection manager error")

        with pytest.raises(Exception, match="Connection manager error"):
            await client.health_check()


if __name__ == "__main__":
    pytest.main([__file__])