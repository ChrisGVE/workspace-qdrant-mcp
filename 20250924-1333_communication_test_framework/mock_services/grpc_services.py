"""
Mock gRPC Services

Mock implementations of all Rust daemon gRPC services for testing
communication patterns without requiring actual gRPC servers.
"""

import asyncio
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json

from ..dummy_data.grpc_messages import GrpcMessageGenerator


class ServiceState(Enum):
    """Service state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class MockServiceConfig:
    """Configuration for mock service behavior."""
    latency_ms: int = 100
    error_rate: float = 0.05
    timeout_rate: float = 0.02
    max_concurrent_requests: int = 100
    enable_callbacks: bool = True
    callback_delay_ms: int = 50
    state: ServiceState = ServiceState.RUNNING


@dataclass
class RequestMetrics:
    """Metrics tracking for service requests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_request_time: float = 0.0
    concurrent_requests: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)


class MockGrpcService:
    """Base class for mock gRPC services."""

    def __init__(self, service_name: str, config: MockServiceConfig):
        """Initialize mock service."""
        self.service_name = service_name
        self.config = config
        self.metrics = RequestMetrics()
        self.message_generator = GrpcMessageGenerator()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.callbacks: List[Callable[[str, Dict[str, Any]], Awaitable[None]]] = []

    async def simulate_request(
        self,
        method_name: str,
        request_data: Dict[str, Any],
        simulate_errors: bool = True
    ) -> Dict[str, Any]:
        """Simulate a gRPC service request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.concurrent_requests += 1
        self.metrics.last_request_time = start_time

        try:
            # Check service state
            if self.config.state == ServiceState.STOPPED:
                raise ServiceUnavailableError("Service is stopped")
            elif self.config.state == ServiceState.ERROR:
                raise InternalServiceError("Service is in error state")

            # Check concurrent request limit
            if self.metrics.concurrent_requests > self.config.max_concurrent_requests:
                raise ServiceOverloadedError("Too many concurrent requests")

            # Simulate network latency
            await asyncio.sleep(self.config.latency_ms / 1000)

            # Simulate errors if enabled
            if simulate_errors and random.random() < self.config.error_rate:
                error_type = random.choice(["timeout", "internal_error", "invalid_request"])
                if error_type == "timeout":
                    raise TimeoutError("Request timed out")
                elif error_type == "internal_error":
                    raise InternalServiceError("Internal server error")
                else:
                    raise InvalidRequestError("Invalid request parameters")

            # Simulate timeout
            if simulate_errors and random.random() < self.config.timeout_rate:
                await asyncio.sleep(30)  # Simulate long timeout
                raise TimeoutError("Request timed out")

            # Generate response
            response = self.message_generator.generate_service_message(
                self.service_name,
                method_name,
                request_data=request_data,
                is_response=True
            )

            # Add request tracking
            response["request_id"] = request_id
            response["processing_time_ms"] = (time.time() - start_time) * 1000

            # Store active operation if applicable
            if self._is_long_running_operation(method_name):
                self.active_operations[request_id] = {
                    "method": method_name,
                    "request": request_data,
                    "response": response,
                    "start_time": start_time,
                    "status": "in_progress"
                }

            # Schedule callback if enabled
            if self.config.enable_callbacks and self._should_send_callback(method_name):
                asyncio.create_task(self._send_callback(method_name, response, request_data))

            self.metrics.successful_requests += 1
            return response

        except Exception as e:
            self.metrics.failed_requests += 1
            error_type = type(e).__name__
            self.metrics.errors_by_type[error_type] = self.metrics.errors_by_type.get(error_type, 0) + 1
            raise

        finally:
            self.metrics.concurrent_requests -= 1
            # Update average response time
            response_time = (time.time() - start_time) * 1000
            if self.metrics.total_requests > 0:
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (self.metrics.total_requests - 1) + response_time)
                    / self.metrics.total_requests
                )

    def _is_long_running_operation(self, method_name: str) -> bool:
        """Check if operation is long-running and should be tracked."""
        long_running_methods = [
            "ProcessDocument", "BatchProcess", "SearchStream",
            "StartFileWatcher", "BackupCollection"
        ]
        return method_name in long_running_methods

    def _should_send_callback(self, method_name: str) -> bool:
        """Check if method should send callbacks."""
        callback_methods = [
            "ProcessDocument", "BatchProcess", "Search", "HybridSearch"
        ]
        return method_name in callback_methods

    async def _send_callback(self, method_name: str, response: Dict[str, Any], request_data: Dict[str, Any]):
        """Send callback after processing delay."""
        await asyncio.sleep(self.config.callback_delay_ms / 1000)

        callback_type = {
            "ProcessDocument": "document_processed",
            "BatchProcess": "batch_completed",
            "Search": "search_results",
            "HybridSearch": "search_results"
        }.get(method_name, "operation_completed")

        callback_data = self.message_generator.generate_callback_message(callback_type)

        # Notify all registered callbacks
        for callback in self.callbacks:
            try:
                await callback(callback_type, callback_data)
            except Exception as e:
                print(f"Callback error: {e}")

    def register_callback(self, callback: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        """Register a callback function for service events."""
        self.callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "service_name": self.service_name,
            "state": self.config.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / max(1, self.metrics.total_requests)
            ),
            "avg_response_time_ms": self.metrics.avg_response_time_ms,
            "concurrent_requests": self.metrics.concurrent_requests,
            "errors_by_type": self.metrics.errors_by_type,
            "last_request_time": self.metrics.last_request_time,
            "active_operations": len(self.active_operations)
        }

    def get_active_operations(self) -> Dict[str, Any]:
        """Get currently active operations."""
        return self.active_operations.copy()

    def update_config(self, **kwargs):
        """Update service configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def reset_metrics(self):
        """Reset service metrics."""
        self.metrics = RequestMetrics()


# Custom exceptions for service errors
class ServiceError(Exception):
    """Base service error."""
    pass


class ServiceUnavailableError(ServiceError):
    """Service is unavailable."""
    pass


class InternalServiceError(ServiceError):
    """Internal service error."""
    pass


class InvalidRequestError(ServiceError):
    """Invalid request error."""
    pass


class ServiceOverloadedError(ServiceError):
    """Service is overloaded."""
    pass


class MockDocumentProcessor(MockGrpcService):
    """Mock DocumentProcessor service."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize DocumentProcessor mock."""
        super().__init__("DocumentProcessor", config or MockServiceConfig())
        self.processed_documents = {}
        self.processing_queue = {}

    async def process_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock document processing."""
        response = await self.simulate_request("ProcessDocument", request)

        # Store document processing result
        document_id = request.get("document_id", str(uuid.uuid4()))
        self.processed_documents[document_id] = {
            "status": response.get("status", "SUCCESS"),
            "processed_at": time.time(),
            "processing_time_ms": response.get("processing_time_ms", 0),
            "extracted_text_length": len(request.get("content", "")),
            "embeddings_generated": True
        }

        return response

    async def get_document_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get document processing status."""
        document_id = request.get("document_id")
        if document_id in self.processed_documents:
            doc_data = self.processed_documents[document_id]
            return {
                "message_type": "GetDocumentStatusResponse",
                "document_id": document_id,
                "status": doc_data["status"],
                "processed_at": int(doc_data["processed_at"]),
                "processing_time_ms": doc_data["processing_time_ms"]
            }
        else:
            return await self.simulate_request("GetDocumentStatus", request)

    async def batch_process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock batch processing."""
        response = await self.simulate_request("BatchProcess", request)

        # Track batch operation
        batch_id = response.get("batch_id", str(uuid.uuid4()))
        document_ids = request.get("document_ids", [])

        self.processing_queue[batch_id] = {
            "total_documents": len(document_ids),
            "processed_count": 0,
            "failed_count": 0,
            "status": "QUEUED",
            "started_at": time.time()
        }

        # Simulate processing progress
        asyncio.create_task(self._simulate_batch_progress(batch_id, document_ids))

        return response

    async def _simulate_batch_progress(self, batch_id: str, document_ids: List[str]):
        """Simulate batch processing progress."""
        if batch_id not in self.processing_queue:
            return

        batch_data = self.processing_queue[batch_id]
        batch_data["status"] = "PROCESSING"

        total_docs = len(document_ids)
        for i, doc_id in enumerate(document_ids):
            # Simulate processing time per document
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Simulate occasional failures
            if random.random() < 0.1:  # 10% failure rate
                batch_data["failed_count"] += 1
            else:
                batch_data["processed_count"] += 1
                # Add to processed documents
                self.processed_documents[doc_id] = {
                    "status": "SUCCESS",
                    "processed_at": time.time(),
                    "processing_time_ms": random.randint(100, 1000),
                    "batch_id": batch_id
                }

            # Send progress update
            progress_percent = int((i + 1) / total_docs * 100)
            await self._send_progress_callback(batch_id, progress_percent, batch_data)

        # Mark batch as completed
        batch_data["status"] = "COMPLETED"
        batch_data["completed_at"] = time.time()

        # Send completion callback
        await self._send_callback("BatchProcess", {
            "batch_id": batch_id,
            "status": "COMPLETED",
            **batch_data
        }, {"batch_id": batch_id})

    async def _send_progress_callback(self, batch_id: str, progress_percent: int, batch_data: Dict[str, Any]):
        """Send progress update callback."""
        callback_data = {
            "callback_type": "progress_update",
            "batch_id": batch_id,
            "progress_percent": progress_percent,
            "processed_count": batch_data["processed_count"],
            "failed_count": batch_data["failed_count"],
            "status": batch_data["status"]
        }

        for callback in self.callbacks:
            try:
                await callback("progress_update", callback_data)
            except Exception as e:
                print(f"Progress callback error: {e}")


class MockSearchService(MockGrpcService):
    """Mock SearchService."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize SearchService mock."""
        super().__init__("SearchService", config or MockServiceConfig())
        self.search_history = {}
        self.search_index = {}

    async def search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock search operation."""
        response = await self.simulate_request("Search", request)

        # Store search in history
        query_id = response.get("query_id", str(uuid.uuid4()))
        self.search_history[query_id] = {
            "query": request.get("query", ""),
            "collection": request.get("collection_name", ""),
            "timestamp": time.time(),
            "results_count": response.get("total_count", 0),
            "search_time_ms": response.get("search_time_ms", 0)
        }

        return response

    async def hybrid_search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock hybrid search operation."""
        response = await self.simulate_request("HybridSearch", request)

        # Enhanced response for hybrid search
        if "results" in response:
            for result in response["results"]:
                result["hybrid_score"] = {
                    "dense_score": random.uniform(0.0, 1.0),
                    "sparse_score": random.uniform(0.0, 1.0),
                    "fusion_method": "rrf",
                    "final_score": result.get("score", 0.0)
                }

        return response

    async def search_stream(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock streaming search operation."""
        stream_responses = []
        total_chunks = random.randint(3, 8)

        for chunk_index in range(total_chunks):
            chunk_response = await self.simulate_request("SearchStream", request)
            chunk_response["chunk_index"] = chunk_index
            chunk_response["is_final"] = (chunk_index == total_chunks - 1)
            chunk_response["stream_id"] = str(uuid.uuid4())

            stream_responses.append(chunk_response)

            # Simulate streaming delay
            await asyncio.sleep(0.1)

        return stream_responses

    async def save_search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Save search for later retrieval."""
        response = await self.simulate_request("SaveSearch", request)

        search_id = response.get("search_id", str(uuid.uuid4()))
        self.search_index[search_id] = {
            "query": request.get("query", ""),
            "filters": request.get("filters", {}),
            "saved_at": time.time(),
            "name": request.get("search_name", f"search_{search_id[:8]}")
        }

        return response

    def get_search_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent search history."""
        history_items = list(self.search_history.values())
        history_items.sort(key=lambda x: x["timestamp"], reverse=True)
        return history_items[:limit]


class MockMemoryService(MockGrpcService):
    """Mock MemoryService."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize MemoryService mock."""
        super().__init__("MemoryService", config or MockServiceConfig())
        self.collections = {}
        self.collection_stats = {}

    async def create_collection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock collection creation."""
        response = await self.simulate_request("CreateCollection", request)

        collection_name = request.get("collection_name", f"collection_{uuid.uuid4().hex[:8]}")
        collection_id = response.get("collection_id", str(uuid.uuid4()))

        self.collections[collection_id] = {
            "name": collection_name,
            "vector_size": request.get("vector_size", 768),
            "distance_metric": request.get("distance_metric", "Cosine"),
            "created_at": time.time(),
            "status": "active",
            "vector_count": 0,
            "config": request
        }

        self.collection_stats[collection_id] = {
            "total_operations": 0,
            "last_operation": time.time(),
            "size_mb": 0
        }

        return response

    async def delete_collection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock collection deletion."""
        response = await self.simulate_request("DeleteCollection", request)

        collection_id = request.get("collection_id")
        if collection_id in self.collections:
            del self.collections[collection_id]
            if collection_id in self.collection_stats:
                del self.collection_stats[collection_id]

        return response

    async def list_collections(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock collection listing."""
        response = await self.simulate_request("ListCollections", request)

        # Override with actual collection data
        collections = []
        for collection_id, collection_data in self.collections.items():
            collections.append({
                "collection_id": collection_id,
                "name": collection_data["name"],
                "vector_size": collection_data["vector_size"],
                "vector_count": collection_data["vector_count"],
                "status": collection_data["status"],
                "created_at": int(collection_data["created_at"])
            })

        response["collections"] = collections
        response["total_count"] = len(collections)

        return response

    async def get_collection_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed collection information."""
        collection_id = request.get("collection_id")
        if collection_id in self.collections:
            collection_data = self.collections[collection_id]
            stats = self.collection_stats.get(collection_id, {})

            return {
                "message_type": "GetCollectionInfoResponse",
                "collection_id": collection_id,
                "collection_info": {
                    **collection_data,
                    "statistics": stats,
                    "health": "healthy" if collection_data["status"] == "active" else "degraded"
                }
            }
        else:
            raise InvalidRequestError(f"Collection {collection_id} not found")

    async def backup_collection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock collection backup."""
        response = await self.simulate_request("BackupCollection", request)

        collection_id = request.get("collection_id")
        backup_id = str(uuid.uuid4())

        # Simulate backup process
        asyncio.create_task(self._simulate_backup_progress(collection_id, backup_id))

        response["backup_id"] = backup_id
        response["status"] = "STARTED"

        return response

    async def _simulate_backup_progress(self, collection_id: str, backup_id: str):
        """Simulate backup progress."""
        stages = ["preparing", "copying_data", "compressing", "finalizing"]

        for i, stage in enumerate(stages):
            await asyncio.sleep(random.uniform(1.0, 3.0))  # Simulate work

            progress_percent = int((i + 1) / len(stages) * 100)

            # Send progress callback
            callback_data = {
                "callback_type": "backup_progress",
                "backup_id": backup_id,
                "collection_id": collection_id,
                "stage": stage,
                "progress_percent": progress_percent,
                "status": "IN_PROGRESS" if i < len(stages) - 1 else "COMPLETED"
            }

            for callback in self.callbacks:
                try:
                    await callback("backup_progress", callback_data)
                except Exception as e:
                    print(f"Backup progress callback error: {e}")


class MockSystemService(MockGrpcService):
    """Mock SystemService."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize SystemService mock."""
        super().__init__("SystemService", config or MockServiceConfig())
        self.watchers = {}
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_received": 0},
            "uptime": 0
        }

    async def get_system_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock system status retrieval."""
        response = await self.simulate_request("GetSystemStatus", request)

        # Update with current metrics
        self._update_system_metrics()
        response.update(self.system_metrics)

        if request.get("include_metrics", False):
            response["detailed_metrics"] = {
                "service_metrics": {
                    "DocumentProcessor": {"requests_per_sec": random.uniform(0, 100)},
                    "SearchService": {"requests_per_sec": random.uniform(0, 200)},
                    "MemoryService": {"requests_per_sec": random.uniform(0, 50)}
                },
                "resource_usage": {
                    "threads": random.randint(10, 100),
                    "file_descriptors": random.randint(100, 1000),
                    "heap_size_mb": random.randint(100, 2048)
                }
            }

        return response

    async def start_file_watcher(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock file watcher start."""
        response = await self.simulate_request("StartFileWatcher", request)

        watcher_id = response.get("watcher_id", str(uuid.uuid4()))
        watch_paths = request.get("watch_paths", [])

        self.watchers[watcher_id] = {
            "paths": watch_paths,
            "recursive": request.get("recursive", True),
            "patterns": request.get("file_patterns", ["*"]),
            "started_at": time.time(),
            "status": "active",
            "files_detected": 0,
            "events_processed": 0
        }

        # Start simulating file events
        asyncio.create_task(self._simulate_file_events(watcher_id))

        return response

    async def stop_file_watcher(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock file watcher stop."""
        response = await self.simulate_request("StopFileWatcher", request)

        watcher_id = request.get("watcher_id")
        if watcher_id in self.watchers:
            self.watchers[watcher_id]["status"] = "stopped"
            self.watchers[watcher_id]["stopped_at"] = time.time()

        return response

    async def _simulate_file_events(self, watcher_id: str):
        """Simulate file system events."""
        if watcher_id not in self.watchers:
            return

        watcher_data = self.watchers[watcher_id]

        while watcher_data["status"] == "active":
            await asyncio.sleep(random.uniform(5.0, 30.0))  # Random intervals

            if watcher_data["status"] != "active":
                break

            # Generate file event
            event_type = random.choice(["created", "modified", "deleted"])
            file_path = f"{random.choice(watcher_data['paths'])}/file_{random.randint(1, 1000)}.txt"

            watcher_data["events_processed"] += 1

            # Send file event callback
            callback_data = {
                "callback_type": "file_event",
                "watcher_id": watcher_id,
                "event_type": event_type,
                "file_path": file_path,
                "timestamp": time.time()
            }

            for callback in self.callbacks:
                try:
                    await callback("file_event", callback_data)
                except Exception as e:
                    print(f"File event callback error: {e}")

    def _update_system_metrics(self):
        """Update system metrics with random but realistic values."""
        # Simulate gradual changes in system metrics
        self.system_metrics["cpu_usage"] = max(0, min(100,
            self.system_metrics["cpu_usage"] + random.uniform(-5, 5)))
        self.system_metrics["memory_usage"] = max(0, min(100,
            self.system_metrics["memory_usage"] + random.uniform(-2, 2)))
        self.system_metrics["disk_usage"] = max(0, min(100,
            self.system_metrics["disk_usage"] + random.uniform(-0.1, 0.1)))

        # Update network I/O
        self.system_metrics["network_io"]["bytes_sent"] += random.randint(0, 10000)
        self.system_metrics["network_io"]["bytes_received"] += random.randint(0, 20000)

        self.system_metrics["uptime"] = int(time.time() - (time.time() % 86400))  # Simplified uptime


class MockServiceDiscovery(MockGrpcService):
    """Mock ServiceDiscovery."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize ServiceDiscovery mock."""
        super().__init__("ServiceDiscovery", config or MockServiceConfig())
        self.registered_services = {}
        self.service_health = {}

    async def register_service(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock service registration."""
        response = await self.simulate_request("RegisterService", request)

        service_id = response.get("service_id", str(uuid.uuid4()))
        service_name = request.get("service_name", "unknown_service")

        self.registered_services[service_id] = {
            "service_name": service_name,
            "address": request.get("address", "127.0.0.1"),
            "port": request.get("port", 50051),
            "metadata": request.get("metadata", {}),
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "status": "active"
        }

        self.service_health[service_id] = {
            "status": "healthy",
            "last_check": time.time(),
            "checks_performed": 0,
            "consecutive_failures": 0
        }

        # Start health monitoring
        asyncio.create_task(self._monitor_service_health(service_id))

        return response

    async def unregister_service(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock service unregistration."""
        response = await self.simulate_request("UnregisterService", request)

        service_id = request.get("service_id")
        if service_id in self.registered_services:
            del self.registered_services[service_id]
            if service_id in self.service_health:
                del self.service_health[service_id]

        return response

    async def discover_services(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock service discovery."""
        response = await self.simulate_request("DiscoverServices", request)

        service_filter = request.get("service_name_filter")
        health_filter = request.get("health_status_filter")

        services = []
        for service_id, service_data in self.registered_services.items():
            # Apply filters
            if service_filter and service_data["service_name"] != service_filter:
                continue

            health_data = self.service_health.get(service_id, {})
            if health_filter and health_data.get("status") != health_filter:
                continue

            services.append({
                "service_id": service_id,
                **service_data,
                "health": health_data
            })

        response["services"] = services
        response["total_count"] = len(services)

        return response

    async def update_service_health(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Update service health status."""
        response = await self.simulate_request("UpdateServiceHealth", request)

        service_id = request.get("service_id")
        if service_id in self.service_health:
            health_status = request.get("health_status", "healthy")
            self.service_health[service_id].update({
                "status": health_status,
                "last_check": time.time(),
                "checks_performed": self.service_health[service_id]["checks_performed"] + 1
            })

            if health_status != "healthy":
                self.service_health[service_id]["consecutive_failures"] += 1
            else:
                self.service_health[service_id]["consecutive_failures"] = 0

        return response

    async def _monitor_service_health(self, service_id: str):
        """Monitor service health with periodic checks."""
        while service_id in self.registered_services:
            await asyncio.sleep(random.uniform(30.0, 60.0))  # Health check interval

            if service_id not in self.service_health:
                break

            # Simulate health check
            health_data = self.service_health[service_id]

            # Randomly simulate health changes
            if random.random() < 0.05:  # 5% chance of status change
                current_status = health_data["status"]
                if current_status == "healthy":
                    new_status = random.choice(["degraded", "unhealthy"])
                else:
                    new_status = "healthy"

                health_data["status"] = new_status
                health_data["last_check"] = time.time()
                health_data["checks_performed"] += 1

                # Send health status callback
                callback_data = {
                    "callback_type": "health_status_changed",
                    "service_id": service_id,
                    "old_status": current_status,
                    "new_status": new_status,
                    "timestamp": time.time()
                }

                for callback in self.callbacks:
                    try:
                        await callback("health_status_changed", callback_data)
                    except Exception as e:
                        print(f"Health status callback error: {e}")


class MockGrpcServices:
    """Orchestrator for all mock gRPC services."""

    def __init__(self, configs: Optional[Dict[str, MockServiceConfig]] = None):
        """Initialize all mock gRPC services."""
        configs = configs or {}

        self.document_processor = MockDocumentProcessor(configs.get("DocumentProcessor"))
        self.search_service = MockSearchService(configs.get("SearchService"))
        self.memory_service = MockMemoryService(configs.get("MemoryService"))
        self.system_service = MockSystemService(configs.get("SystemService"))
        self.service_discovery = MockServiceDiscovery(configs.get("ServiceDiscovery"))

        self.services = {
            "DocumentProcessor": self.document_processor,
            "SearchService": self.search_service,
            "MemoryService": self.memory_service,
            "SystemService": self.system_service,
            "ServiceDiscovery": self.service_discovery
        }

        self.global_callbacks = []

    def get_service(self, service_name: str) -> MockGrpcService:
        """Get service by name."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        return self.services[service_name]

    def register_global_callback(self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[None]]):
        """Register global callback for all service events."""
        async def service_callback(callback_type: str, data: Dict[str, Any]):
            await callback(self.__class__.__name__, callback_type, data)

        for service in self.services.values():
            service.register_callback(service_callback)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all services."""
        return {
            service_name: service.get_metrics()
            for service_name, service in self.services.items()
        }

    def update_all_configs(self, **kwargs):
        """Update configuration for all services."""
        for service in self.services.values():
            service.update_config(**kwargs)

    def reset_all_metrics(self):
        """Reset metrics for all services."""
        for service in self.services.values():
            service.reset_metrics()

    async def shutdown_all(self):
        """Shutdown all services gracefully."""
        for service in self.services.values():
            service.config.state = ServiceState.STOPPING
            # Give services time to finish current operations
            await asyncio.sleep(0.1)
            service.config.state = ServiceState.STOPPED