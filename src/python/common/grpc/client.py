"""
Async gRPC client for communicating with the Rust ingestion engine.

This module provides a high-level async Python client that wraps the
generated gRPC stubs with connection management, error handling, and
integration with the MCP server's async architecture.
"""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any

from google.protobuf.empty_pb2 import Empty

from .connection_manager import ConnectionConfig, GrpcConnectionManager
from .ingestion_pb2 import (
    GetStatsRequest,
    StartWatchingRequest,
    StopWatchingRequest,
    StreamMetricsRequest,
    StreamQueueRequest,
    StreamStatusRequest,
)
from .ingestion_pb2_grpc import IngestServiceStub
from .types import (
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    HealthCheckResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)

logger = logging.getLogger(__name__)


async def _maybe_wait_for(value: Any, timeout: float | None):
    """Await awaitables with a timeout, or return sync values unchanged."""
    if inspect.isawaitable(value):
        return await asyncio.wait_for(value, timeout=timeout)
    return value


class AsyncIngestClient:
    """
    Async gRPC client for the Rust ingestion engine.

    Provides high-level Python async methods for document processing,
    search queries, file watching, and health monitoring.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        connection_config: ConnectionConfig | None = None,
    ):
        """Initialize the async gRPC client.

        Args:
            host: gRPC server host address
            port: gRPC server port
            connection_config: Optional connection configuration
        """
        if connection_config:
            self.connection_config = connection_config
        else:
            self.connection_config = ConnectionConfig(host=host, port=port)

        self.connection_manager = GrpcConnectionManager(self.connection_config)
        self._started = False

        logger.info("AsyncIngestClient initialized", host=host, port=port)

    async def start(self):
        """Start the client and connection management."""
        if not self._started:
            await self.connection_manager.start()
            self._started = True
            logger.info("AsyncIngestClient started")

    async def stop(self):
        """Stop the client and clean up resources."""
        if self._started:
            await self.connection_manager.stop()
            self._started = False
            logger.info("AsyncIngestClient stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def process_document(
        self,
        file_path: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        document_id: str | None = None,
        chunk_text: bool = True,
        timeout: float = 30.0,
    ) -> ProcessDocumentResponse:
        """
        Process a document for ingestion into Qdrant.

        Args:
            file_path: Path to the document file
            collection: Target collection name
            metadata: Optional metadata dictionary
            document_id: Optional custom document ID
            chunk_text: Whether to chunk large documents
            timeout: Request timeout in seconds

        Returns:
            ProcessDocumentResponse with processing results

        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()

        request = ProcessDocumentRequest(
            file_path=file_path,
            collection=collection,
            metadata=metadata,
            document_id=document_id,
            chunk_text=chunk_text,
        )

        async def _process_doc(stub: IngestServiceStub):
            pb_request = request.to_pb()
            pb_response = await _maybe_wait_for(
                stub.ProcessDocument(pb_request), timeout=timeout
            )
            return ProcessDocumentResponse.from_pb(pb_response)

        response = await self.connection_manager.with_retry(_process_doc)
        if response is None:
            raise asyncio.TimeoutError("ProcessDocument request timed out")
        return response

    async def execute_query(
        self,
        query: str,
        collections: list[str] | None = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
        timeout: float = 15.0,
    ) -> ExecuteQueryResponse:
        """
        Execute a search query against indexed documents.

        Args:
            query: Search query text
            collections: Optional list of collections to search
            mode: Search mode ("hybrid", "dense", "sparse")
            limit: Maximum number of results
            score_threshold: Minimum relevance score
            timeout: Request timeout in seconds

        Returns:
            ExecuteQueryResponse with search results

        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()

        request = ExecuteQueryRequest(
            query=query,
            collections=collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
        )

        async def _execute_query(stub: IngestServiceStub):
            pb_request = request.to_pb()
            pb_response = await _maybe_wait_for(
                stub.ExecuteQuery(pb_request), timeout=timeout
            )
            return ExecuteQueryResponse.from_pb(pb_response)

        return await self.connection_manager.with_retry(_execute_query)

    async def health_check(self, timeout: float = 5.0) -> HealthCheckResponse:
        """
        Perform a health check on the ingestion service.

        Args:
            timeout: Request timeout in seconds

        Returns:
            HealthCheckResponse with service health status

        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()

        async def _health_check(stub: IngestServiceStub):
            pb_response = await _maybe_wait_for(
                stub.HealthCheck(Empty()), timeout=timeout
            )
            return HealthCheckResponse.from_pb(pb_response)

        return await self.connection_manager.with_retry(_health_check)

    async def start_watching(
        self,
        path: str,
        collection: str,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        recursive_depth: int = -1,
        debounce_seconds: int = 5,
        update_frequency_ms: int = 1000,
        watch_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Start watching a directory for file changes.

        This returns an async iterator that yields watch events.

        Args:
            path: Directory path to watch
            collection: Target collection for ingested files
            patterns: File patterns to include
            ignore_patterns: File patterns to ignore
            auto_ingest: Enable automatic ingestion
            recursive: Watch subdirectories
            recursive_depth: Maximum recursion depth (-1 for unlimited)
            debounce_seconds: Debounce delay before processing
            update_frequency_ms: File system check frequency
            watch_id: Optional custom watch identifier

        Yields:
            Dict containing watch event information

        Raises:
            grpc.RpcError: If the request fails
        """
        if not self._started:
            await self.start()

        request = StartWatchingRequest()
        request.path = path
        request.collection = collection
        request.auto_ingest = auto_ingest
        request.recursive = recursive
        request.recursive_depth = recursive_depth
        request.debounce_seconds = debounce_seconds
        request.update_frequency_ms = update_frequency_ms

        if patterns:
            request.patterns.extend(patterns)
        if ignore_patterns:
            request.ignore_patterns.extend(ignore_patterns)
        if watch_id:
            request.watch_id = watch_id

        async with self.connection_manager.get_stub() as stub:
            async for update in stub.StartWatching(request):
                # Convert protobuf response to dict
                event_data = {
                    "watch_id": update.watch_id,
                    "event_type": update.event_type,
                    "file_path": update.file_path,
                    "timestamp": update.timestamp.ToDatetime(),
                    "status": update.status,
                }

                if update.HasField("error_message"):
                    event_data["error_message"] = update.error_message

                yield event_data

    async def stop_watching(
        self, watch_id: str, timeout: float = 10.0
    ) -> dict[str, Any]:
        """
        Stop watching a specific watch configuration.

        Args:
            watch_id: Watch identifier to stop
            timeout: Request timeout in seconds

        Returns:
            Dict with stop operation results

        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()

        async def _stop_watching(stub: IngestServiceStub):
            request = StopWatchingRequest()
            request.watch_id = watch_id

            response = await _maybe_wait_for(
                stub.StopWatching(request), timeout=timeout
            )

            return {"success": response.success, "message": response.message}

        response = await self.connection_manager.with_retry(_stop_watching)
        if response is None:
            raise asyncio.TimeoutError("StopWatching request timed out")
        return response

    async def get_stats(
        self,
        include_collection_stats: bool = True,
        include_watch_stats: bool = True,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """
        Get statistics and health information about the ingestion engine.

        Args:
            include_collection_stats: Include collection statistics
            include_watch_stats: Include watch statistics
            timeout: Request timeout in seconds

        Returns:
            Dict with engine statistics

        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()

        async def _get_stats(stub: IngestServiceStub):
            request = GetStatsRequest()
            request.include_collection_stats = include_collection_stats
            request.include_watch_stats = include_watch_stats

            response = await _maybe_wait_for(
                stub.GetStats(request), timeout=timeout
            )

            # Convert protobuf response to dict
            stats = {
                "engine_stats": {
                    "started_at": response.engine_stats.started_at.ToDatetime(),
                    "uptime_seconds": response.engine_stats.uptime.total_seconds(),
                    "total_documents_processed": response.engine_stats.total_documents_processed,
                    "total_documents_indexed": response.engine_stats.total_documents_indexed,
                    "active_watches": response.engine_stats.active_watches,
                    "version": response.engine_stats.version,
                }
            }

            if include_collection_stats:
                stats["collection_stats"] = []
                for col_stat in response.collection_stats:
                    stats["collection_stats"].append(
                        {
                            "name": col_stat.name,
                            "document_count": col_stat.document_count,
                            "total_size_bytes": col_stat.total_size_bytes,
                            "last_updated": col_stat.last_updated.ToDatetime(),
                        }
                    )

            if include_watch_stats:
                stats["watch_stats"] = []
                for watch_stat in response.watch_stats:
                    stats["watch_stats"].append(
                        {
                            "watch_id": watch_stat.watch_id,
                            "path": watch_stat.path,
                            "collection": watch_stat.collection,
                            "status": watch_stat.status,
                            "files_processed": watch_stat.files_processed,
                            "files_failed": watch_stat.files_failed,
                            "created_at": watch_stat.created_at.ToDatetime(),
                            "last_activity": watch_stat.last_activity.ToDatetime(),
                        }
                    )

            return stats

        return await self.connection_manager.with_retry(_get_stats)

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the current gRPC connection."""
        return self.connection_manager.get_connection_info()

    async def test_connection(self) -> bool:
        """
        Test if we can connect to the gRPC server.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            await self.health_check(timeout=5.0)
            return True
        except Exception as e:
            logger.warning("Connection test failed: %s", e)
            return False

    async def stream_processing_status(
        self,
        update_interval_seconds: int = 5,
        include_history: bool = True,
        collection_filter: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream real-time processing status updates.

        Args:
            update_interval_seconds: How often to send updates
            include_history: Whether to include recent processing history
            collection_filter: Optional collection name to filter by

        Yields:
            Dict containing processing status updates with:
            - timestamp: When the update was generated
            - active_tasks: Currently processing files
            - recent_completed: Recently completed files
            - current_stats: Current processing statistics
            - queue_status: Queue depth and priority breakdown
        """
        if not self._started:
            await self.start()

        request = StreamStatusRequest(
            update_interval_seconds=update_interval_seconds,
            include_history=include_history,
        )

        if collection_filter:
            request.collection_filter = collection_filter

        def _format_status_update(update):
            return {
                "timestamp": update.timestamp.ToJsonString(),
                "active_tasks": [
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "file_path": task.file_path,
                        "collection": task.collection,
                        "status": task.status,
                        "progress_percent": task.progress_percent,
                        "started_at": task.started_at.ToJsonString(),
                        "completed_at": task.completed_at.ToJsonString()
                        if task.HasField("completed_at")
                        else None,
                        "error_message": task.error_message
                        if task.HasField("error_message")
                        else None,
                    }
                    for task in update.active_tasks
                ],
                "recent_completed": [
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "file_path": task.file_path,
                        "collection": task.collection,
                        "status": task.status,
                        "progress_percent": task.progress_percent,
                        "started_at": task.started_at.ToJsonString(),
                        "completed_at": task.completed_at.ToJsonString()
                        if task.HasField("completed_at")
                        else None,
                        "error_message": task.error_message
                        if task.HasField("error_message")
                        else None,
                    }
                    for task in update.recent_completed
                ],
                "current_stats": {
                    "total_files_processed": update.current_stats.total_files_processed,
                    "total_files_failed": update.current_stats.total_files_failed,
                    "total_files_skipped": update.current_stats.total_files_skipped,
                    "active_tasks": update.current_stats.active_tasks,
                    "queued_tasks": update.current_stats.queued_tasks,
                    "average_processing_time": update.current_stats.average_processing_time.ToJsonString(),
                    "last_activity": update.current_stats.last_activity.ToJsonString(),
                }
                if update.HasField("current_stats")
                else None,
                "queue_status": {
                    "total_queued": update.queue_status.total_queued,
                    "high_priority": update.queue_status.high_priority,
                    "normal_priority": update.queue_status.normal_priority,
                    "low_priority": update.queue_status.low_priority,
                    "urgent_priority": update.queue_status.urgent_priority,
                    "collections_with_queued": list(
                        update.queue_status.collections_with_queued
                    ),
                    "estimated_completion_time": update.queue_status.estimated_completion_time.ToJsonString(),
                }
                if update.HasField("queue_status")
                else None,
            }

        async def _stream_status(stub: IngestServiceStub):
            async for update in stub.StreamProcessingStatus(request):
                yield _format_status_update(update)

        stream = self.connection_manager.with_stream(_stream_status)
        if inspect.isawaitable(stream):
            stream = await stream
        async for status_update in stream:
            if isinstance(status_update, dict):
                yield status_update
            else:
                yield _format_status_update(status_update)

    async def stream_system_metrics(
        self, update_interval_seconds: int = 10, include_detailed_metrics: bool = True
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream real-time system metrics updates.

        Args:
            update_interval_seconds: How often to send updates
            include_detailed_metrics: Whether to include detailed performance metrics

        Yields:
            Dict containing system metrics with:
            - timestamp: When the update was generated
            - resource_usage: CPU, memory, disk usage
            - engine_stats: Engine uptime and processing stats
            - collection_stats: Per-collection statistics
            - performance_metrics: Processing rates and bottlenecks
        """
        if not self._started:
            await self.start()

        request = StreamMetricsRequest(
            update_interval_seconds=update_interval_seconds,
            include_detailed_metrics=include_detailed_metrics,
        )

        def _format_metrics_update(update):
            return {
                "timestamp": update.timestamp.ToJsonString(),
                "resource_usage": {
                    "cpu_percent": update.resource_usage.cpu_percent,
                    "memory_bytes": update.resource_usage.memory_bytes,
                    "memory_peak_bytes": update.resource_usage.memory_peak_bytes,
                    "open_files": update.resource_usage.open_files,
                    "active_connections": update.resource_usage.active_connections,
                    "disk_usage_percent": update.resource_usage.disk_usage_percent,
                }
                if update.HasField("resource_usage")
                else None,
                "engine_stats": {
                    "started_at": update.engine_stats.started_at.ToJsonString(),
                    "uptime": update.engine_stats.uptime.ToJsonString(),
                    "total_documents_processed": update.engine_stats.total_documents_processed,
                    "total_documents_indexed": update.engine_stats.total_documents_indexed,
                    "active_watches": update.engine_stats.active_watches,
                    "version": update.engine_stats.version,
                }
                if update.HasField("engine_stats")
                else None,
                "collection_stats": [
                    {
                        "name": stats.name,
                        "document_count": stats.document_count,
                        "total_size_bytes": stats.total_size_bytes,
                        "last_updated": stats.last_updated.ToJsonString(),
                    }
                    for stats in update.collection_stats
                ],
                "performance_metrics": {
                    "processing_rate_files_per_hour": update.performance_metrics.processing_rate_files_per_hour,
                    "average_processing_time": update.performance_metrics.average_processing_time.ToJsonString(),
                    "success_rate_percent": update.performance_metrics.success_rate_percent,
                    "concurrent_tasks": update.performance_metrics.concurrent_tasks,
                    "throughput_bytes_per_second": update.performance_metrics.throughput_bytes_per_second,
                    "bottlenecks": [
                        {
                            "component": bottleneck.component,
                            "description": bottleneck.description,
                            "severity": bottleneck.severity,
                            "suggestion": bottleneck.suggestion,
                        }
                        for bottleneck in update.performance_metrics.bottlenecks
                    ],
                }
                if update.HasField("performance_metrics")
                else None,
            }

        async def _stream_metrics(stub: IngestServiceStub):
            async for update in stub.StreamSystemMetrics(request):
                yield _format_metrics_update(update)

        stream = self.connection_manager.with_stream(_stream_metrics)
        if inspect.isawaitable(stream):
            stream = await stream
        async for metrics_update in stream:
            if isinstance(metrics_update, dict):
                yield metrics_update
            else:
                yield _format_metrics_update(metrics_update)

    async def stream_queue_status(
        self, update_interval_seconds: int = 3, collection_filter: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream real-time queue status updates.

        Args:
            update_interval_seconds: How often to send updates
            collection_filter: Optional collection name to filter by

        Yields:
            Dict containing queue status with:
            - timestamp: When the update was generated
            - queue_status: Current queue depth by priority
            - recent_additions: Files recently added to queue
            - active_processing: Currently processing files with progress
        """
        if not self._started:
            await self.start()

        request = StreamQueueRequest(update_interval_seconds=update_interval_seconds)

        if collection_filter:
            request.collection_filter = collection_filter

        def _format_queue_update(update):
            return {
                "timestamp": update.timestamp.ToJsonString(),
                "queue_status": {
                    "total_queued": update.queue_status.total_queued,
                    "high_priority": update.queue_status.high_priority,
                    "normal_priority": update.queue_status.normal_priority,
                    "low_priority": update.queue_status.low_priority,
                    "urgent_priority": update.queue_status.urgent_priority,
                    "collections_with_queued": list(
                        update.queue_status.collections_with_queued
                    ),
                    "estimated_completion_time": update.queue_status.estimated_completion_time.ToJsonString(),
                }
                if update.HasField("queue_status")
                else None,
                "recent_additions": [
                    {
                        "file_path": file.file_path,
                        "collection": file.collection,
                        "priority": file.priority,
                        "queued_at": file.queued_at.ToJsonString(),
                        "file_size_bytes": file.file_size_bytes,
                    }
                    for file in update.recent_additions
                ],
                "active_processing": [
                    {
                        "task_id": progress.task_id,
                        "file_path": progress.file_path,
                        "collection": progress.collection,
                        "progress_percent": progress.progress_percent,
                        "current_stage": progress.current_stage,
                        "started_at": progress.started_at.ToJsonString(),
                        "estimated_remaining": progress.estimated_remaining.ToJsonString(),
                    }
                    for progress in update.active_processing
                ],
            }

        async def _stream_queue(stub: IngestServiceStub):
            async for update in stub.StreamQueueStatus(request):
                yield _format_queue_update(update)

        stream = self.connection_manager.with_stream(_stream_queue)
        if inspect.isawaitable(stream):
            stream = await stream
        async for queue_update in stream:
            if isinstance(queue_update, dict):
                yield queue_update
            else:
                yield _format_queue_update(queue_update)
