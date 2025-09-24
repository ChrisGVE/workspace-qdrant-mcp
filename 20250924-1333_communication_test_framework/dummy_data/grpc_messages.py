"""
gRPC Message Generator

Generates dummy gRPC messages for all services in the Rust daemon:
- DocumentProcessor service messages
- SearchService messages
- MemoryService messages
- SystemService messages
- ServiceDiscovery messages

Supports both request and response messages with realistic data structures.
"""

import random
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class GrpcServiceSpec:
    """Specification for a gRPC service and its methods."""
    name: str
    methods: List[str]
    message_types: Dict[str, Dict[str, Any]]


class GrpcMessageGenerator:
    """Generates realistic gRPC messages for all daemon services."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.seed = seed or int(time.time())
        random.seed(self.seed)

        self._init_service_specs()

    def _init_service_specs(self):
        """Initialize specifications for all gRPC services."""
        self.services = {
            "DocumentProcessor": GrpcServiceSpec(
                name="DocumentProcessor",
                methods=[
                    "ProcessDocument", "GetDocumentStatus", "ListDocuments",
                    "DeleteDocument", "UpdateDocument", "BatchProcess"
                ],
                message_types={
                    "ProcessDocumentRequest": {
                        "document_id": "string",
                        "content": "bytes",
                        "metadata": "map<string, string>",
                        "project_id": "string",
                        "processing_options": "ProcessingOptions"
                    },
                    "ProcessDocumentResponse": {
                        "document_id": "string",
                        "status": "ProcessingStatus",
                        "extracted_text": "string",
                        "embeddings": "repeated float",
                        "processing_time_ms": "int64"
                    }
                }
            ),

            "SearchService": GrpcServiceSpec(
                name="SearchService",
                methods=[
                    "Search", "HybridSearch", "SemanticSearch", "KeywordSearch",
                    "SearchStream", "GetSearchHistory", "SaveSearch"
                ],
                message_types={
                    "SearchRequest": {
                        "query": "string",
                        "collection_id": "string",
                        "search_type": "SearchType",
                        "filters": "map<string, string>",
                        "limit": "int32",
                        "offset": "int32",
                        "include_vectors": "bool"
                    },
                    "SearchResponse": {
                        "results": "repeated SearchResult",
                        "total_count": "int64",
                        "search_time_ms": "int64",
                        "query_id": "string"
                    }
                }
            ),

            "MemoryService": GrpcServiceSpec(
                name="MemoryService",
                methods=[
                    "CreateCollection", "DeleteCollection", "ListCollections",
                    "GetCollectionInfo", "UpdateCollection", "BackupCollection"
                ],
                message_types={
                    "CreateCollectionRequest": {
                        "collection_name": "string",
                        "vector_size": "int32",
                        "distance_metric": "DistanceMetric",
                        "metadata_schema": "map<string, FieldType>",
                        "project_id": "string"
                    },
                    "CreateCollectionResponse": {
                        "collection_id": "string",
                        "status": "OperationStatus",
                        "message": "string",
                        "created_at": "int64"
                    }
                }
            ),

            "SystemService": GrpcServiceSpec(
                name="SystemService",
                methods=[
                    "GetSystemStatus", "StartFileWatcher", "StopFileWatcher",
                    "GetMetrics", "SetConfiguration", "GetConfiguration"
                ],
                message_types={
                    "GetSystemStatusRequest": {
                        "include_metrics": "bool",
                        "service_filter": "repeated string"
                    },
                    "GetSystemStatusResponse": {
                        "status": "SystemStatus",
                        "uptime_seconds": "int64",
                        "memory_usage_mb": "int64",
                        "cpu_usage_percent": "float",
                        "active_connections": "int32"
                    }
                }
            ),

            "ServiceDiscovery": GrpcServiceSpec(
                name="ServiceDiscovery",
                methods=[
                    "RegisterService", "UnregisterService", "DiscoverServices",
                    "GetServiceHealth", "UpdateServiceHealth"
                ],
                message_types={
                    "RegisterServiceRequest": {
                        "service_name": "string",
                        "service_id": "string",
                        "address": "string",
                        "port": "int32",
                        "metadata": "map<string, string>"
                    },
                    "RegisterServiceResponse": {
                        "success": "bool",
                        "service_id": "string",
                        "message": "string",
                        "registration_time": "int64"
                    }
                }
            )
        }

    def generate_corresponding_grpc_message(self, mcp_tool: str, mcp_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gRPC message corresponding to an MCP tool request."""
        tool_to_service = {
            "add_document": ("DocumentProcessor", "ProcessDocument"),
            "get_document": ("DocumentProcessor", "GetDocumentStatus"),
            "search_workspace": ("SearchService", "Search"),
            "hybrid_search_advanced": ("SearchService", "HybridSearch"),
            "list_collections": ("MemoryService", "ListCollections"),
            "setup_folder_watch": ("SystemService", "StartFileWatcher"),
            "workspace_status": ("SystemService", "GetSystemStatus")
        }

        service_name, method_name = tool_to_service.get(mcp_tool, ("SystemService", "GetSystemStatus"))
        return self.generate_service_message(service_name, method_name, request_data=mcp_request)

    def generate_service_message(
        self,
        service_name: str,
        method_name: str,
        request_data: Optional[Dict[str, Any]] = None,
        is_response: bool = False
    ) -> Dict[str, Any]:
        """Generate a message for a specific service method."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        service = self.services[service_name]
        if method_name not in service.methods:
            raise ValueError(f"Unknown method {method_name} for service {service_name}")

        message_type_suffix = "Response" if is_response else "Request"
        message_type = f"{method_name}{message_type_suffix}"

        if service_name == "DocumentProcessor":
            return self._generate_document_processor_message(method_name, is_response, request_data)
        elif service_name == "SearchService":
            return self._generate_search_service_message(method_name, is_response, request_data)
        elif service_name == "MemoryService":
            return self._generate_memory_service_message(method_name, is_response, request_data)
        elif service_name == "SystemService":
            return self._generate_system_service_message(method_name, is_response, request_data)
        elif service_name == "ServiceDiscovery":
            return self._generate_service_discovery_message(method_name, is_response, request_data)
        else:
            return self._generate_generic_message(service_name, method_name, is_response)

    def _generate_document_processor_message(
        self,
        method_name: str,
        is_response: bool,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate DocumentProcessor service messages."""
        if method_name == "ProcessDocument":
            if is_response:
                return {
                    "message_type": "ProcessDocumentResponse",
                    "document_id": str(uuid.uuid4()),
                    "status": random.choice(["SUCCESS", "PROCESSING", "FAILED"]),
                    "extracted_text": self._generate_sample_text(),
                    "embeddings": [random.uniform(-1.0, 1.0) for _ in range(768)],
                    "processing_time_ms": random.randint(100, 5000),
                    "error_message": None if random.random() > 0.1 else "Processing error occurred"
                }
            else:
                return {
                    "message_type": "ProcessDocumentRequest",
                    "document_id": request_data.get("document_id") if request_data else str(uuid.uuid4()),
                    "content": self._generate_sample_content(),
                    "metadata": self._generate_metadata(),
                    "project_id": self._generate_project_id(),
                    "processing_options": {
                        "extract_text": True,
                        "generate_embeddings": True,
                        "detect_language": True,
                        "chunk_size": random.randint(512, 2048)
                    }
                }

        elif method_name == "GetDocumentStatus":
            if is_response:
                return {
                    "message_type": "GetDocumentStatusResponse",
                    "document_id": str(uuid.uuid4()),
                    "status": random.choice(["COMPLETED", "PROCESSING", "FAILED", "NOT_FOUND"]),
                    "progress_percent": random.randint(0, 100),
                    "created_at": int(time.time() - random.randint(0, 86400)),
                    "updated_at": int(time.time() - random.randint(0, 3600))
                }
            else:
                return {
                    "message_type": "GetDocumentStatusRequest",
                    "document_id": str(uuid.uuid4()),
                    "include_content": random.choice([True, False])
                }

        elif method_name == "BatchProcess":
            if is_response:
                return {
                    "message_type": "BatchProcessResponse",
                    "batch_id": str(uuid.uuid4()),
                    "total_documents": random.randint(1, 100),
                    "processed_count": random.randint(0, 50),
                    "failed_count": random.randint(0, 5),
                    "status": random.choice(["QUEUED", "PROCESSING", "COMPLETED", "FAILED"]),
                    "estimated_completion_time": int(time.time() + random.randint(60, 3600))
                }
            else:
                return {
                    "message_type": "BatchProcessRequest",
                    "document_ids": [str(uuid.uuid4()) for _ in range(random.randint(1, 20))],
                    "processing_options": {
                        "parallel_processing": True,
                        "max_concurrency": random.randint(1, 10)
                    }
                }

        return self._generate_generic_message("DocumentProcessor", method_name, is_response)

    def _generate_search_service_message(
        self,
        method_name: str,
        is_response: bool,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate SearchService messages."""
        if method_name in ["Search", "HybridSearch", "SemanticSearch"]:
            if is_response:
                return {
                    "message_type": f"{method_name}Response",
                    "results": [self._generate_search_result() for _ in range(random.randint(0, 10))],
                    "total_count": random.randint(0, 1000),
                    "search_time_ms": random.randint(10, 500),
                    "query_id": str(uuid.uuid4()),
                    "has_more_results": random.choice([True, False])
                }
            else:
                return {
                    "message_type": f"{method_name}Request",
                    "query": request_data.get("query") if request_data else self._generate_search_query(),
                    "collection_id": self._generate_collection_id(),
                    "search_type": random.choice(["SEMANTIC", "KEYWORD", "HYBRID"]),
                    "filters": self._generate_search_filters(),
                    "limit": random.randint(1, 100),
                    "offset": random.randint(0, 100),
                    "include_vectors": random.choice([True, False]),
                    "search_params": {
                        "ef": random.randint(64, 512),
                        "rerank": random.choice([True, False])
                    }
                }

        elif method_name == "SearchStream":
            if is_response:
                return {
                    "message_type": "SearchStreamResponse",
                    "stream_id": str(uuid.uuid4()),
                    "chunk_data": [self._generate_search_result() for _ in range(random.randint(1, 5))],
                    "is_final": random.choice([True, False]),
                    "chunk_index": random.randint(0, 10)
                }
            else:
                return {
                    "message_type": "SearchStreamRequest",
                    "query": self._generate_search_query(),
                    "stream_config": {
                        "chunk_size": random.randint(10, 100),
                        "max_results": random.randint(100, 1000)
                    }
                }

        return self._generate_generic_message("SearchService", method_name, is_response)

    def _generate_memory_service_message(
        self,
        method_name: str,
        is_response: bool,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate MemoryService messages."""
        if method_name == "CreateCollection":
            if is_response:
                return {
                    "message_type": "CreateCollectionResponse",
                    "collection_id": str(uuid.uuid4()),
                    "status": random.choice(["SUCCESS", "FAILED", "EXISTS"]),
                    "message": "Collection created successfully" if random.random() > 0.1 else "Collection already exists",
                    "created_at": int(time.time()),
                    "vector_count": 0
                }
            else:
                return {
                    "message_type": "CreateCollectionRequest",
                    "collection_name": f"test-collection-{random.randint(1000, 9999)}",
                    "vector_size": random.choice([384, 768, 1536]),
                    "distance_metric": random.choice(["COSINE", "EUCLIDEAN", "DOT_PRODUCT"]),
                    "metadata_schema": self._generate_metadata_schema(),
                    "project_id": self._generate_project_id(),
                    "config": {
                        "replication_factor": random.randint(1, 3),
                        "shard_number": random.randint(1, 4)
                    }
                }

        elif method_name == "ListCollections":
            if is_response:
                return {
                    "message_type": "ListCollectionsResponse",
                    "collections": [self._generate_collection_info() for _ in range(random.randint(0, 10))],
                    "total_count": random.randint(0, 50)
                }
            else:
                return {
                    "message_type": "ListCollectionsRequest",
                    "project_filter": self._generate_project_id() if random.random() > 0.5 else None,
                    "include_stats": random.choice([True, False])
                }

        return self._generate_generic_message("MemoryService", method_name, is_response)

    def _generate_system_service_message(
        self,
        method_name: str,
        is_response: bool,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate SystemService messages."""
        if method_name == "GetSystemStatus":
            if is_response:
                return {
                    "message_type": "GetSystemStatusResponse",
                    "status": random.choice(["HEALTHY", "DEGRADED", "UNHEALTHY"]),
                    "uptime_seconds": random.randint(0, 86400 * 30),  # Up to 30 days
                    "memory_usage_mb": random.randint(100, 8192),
                    "cpu_usage_percent": random.uniform(0.0, 100.0),
                    "active_connections": random.randint(0, 1000),
                    "disk_usage_percent": random.uniform(0.0, 100.0),
                    "service_health": {
                        "document_processor": random.choice(["HEALTHY", "DEGRADED"]),
                        "search_service": random.choice(["HEALTHY", "DEGRADED"]),
                        "memory_service": random.choice(["HEALTHY", "DEGRADED"])
                    }
                }
            else:
                return {
                    "message_type": "GetSystemStatusRequest",
                    "include_metrics": random.choice([True, False]),
                    "service_filter": random.sample(
                        ["document_processor", "search_service", "memory_service"],
                        random.randint(1, 3)
                    )
                }

        elif method_name == "StartFileWatcher":
            if is_response:
                return {
                    "message_type": "StartFileWatcherResponse",
                    "watcher_id": str(uuid.uuid4()),
                    "status": random.choice(["STARTED", "FAILED", "ALREADY_RUNNING"]),
                    "watched_paths": [f"/tmp/path_{i}" for i in range(random.randint(1, 5))]
                }
            else:
                return {
                    "message_type": "StartFileWatcherRequest",
                    "watch_paths": [f"/tmp/watch_{i}" for i in range(random.randint(1, 3))],
                    "recursive": random.choice([True, False]),
                    "file_patterns": ["*.txt", "*.md", "*.py"] if random.random() > 0.5 else None
                }

        return self._generate_generic_message("SystemService", method_name, is_response)

    def _generate_service_discovery_message(
        self,
        method_name: str,
        is_response: bool,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate ServiceDiscovery messages."""
        if method_name == "RegisterService":
            if is_response:
                return {
                    "message_type": "RegisterServiceResponse",
                    "success": random.choice([True, False]),
                    "service_id": str(uuid.uuid4()),
                    "message": "Service registered successfully" if random.random() > 0.1 else "Registration failed",
                    "registration_time": int(time.time())
                }
            else:
                return {
                    "message_type": "RegisterServiceRequest",
                    "service_name": random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
                    "service_id": str(uuid.uuid4()),
                    "address": f"127.0.0.{random.randint(1, 254)}",
                    "port": random.randint(50000, 60000),
                    "metadata": {
                        "version": f"1.{random.randint(0, 10)}.{random.randint(0, 10)}",
                        "environment": random.choice(["development", "staging", "production"])
                    }
                }

        elif method_name == "DiscoverServices":
            if is_response:
                return {
                    "message_type": "DiscoverServicesResponse",
                    "services": [self._generate_service_instance() for _ in range(random.randint(1, 5))],
                    "total_count": random.randint(1, 10)
                }
            else:
                return {
                    "message_type": "DiscoverServicesRequest",
                    "service_name_filter": random.choice(["DocumentProcessor", "SearchService", None]),
                    "health_status_filter": random.choice(["HEALTHY", "DEGRADED", None])
                }

        return self._generate_generic_message("ServiceDiscovery", method_name, is_response)

    def _generate_generic_message(self, service_name: str, method_name: str, is_response: bool) -> Dict[str, Any]:
        """Generate a generic message for unknown service/method combinations."""
        message_type = f"{method_name}{'Response' if is_response else 'Request'}"
        return {
            "message_type": message_type,
            "service": service_name,
            "method": method_name,
            "timestamp": int(time.time()),
            "request_id": str(uuid.uuid4()),
            "generic_data": {
                "field_1": random.choice(["value_a", "value_b", "value_c"]),
                "field_2": random.randint(1, 1000),
                "field_3": random.uniform(0.0, 1.0)
            }
        }

    def generate_callback_message(self, callback_type: str) -> Dict[str, Any]:
        """Generate daemon-to-MCP callback messages."""
        callback_generators = {
            "document_processed": self._generate_document_processed_callback,
            "search_results": self._generate_search_results_callback,
            "error_notification": self._generate_error_notification_callback,
            "progress_update": self._generate_progress_update_callback,
            "health_status": self._generate_health_status_callback
        }

        generator = callback_generators.get(callback_type, self._generate_generic_callback)
        return generator(callback_type)

    def _generate_document_processed_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate document processing completion callback."""
        return {
            "callback_type": callback_type,
            "document_id": str(uuid.uuid4()),
            "status": random.choice(["SUCCESS", "FAILED"]),
            "processing_time_ms": random.randint(100, 10000),
            "extracted_text_length": random.randint(100, 50000),
            "embeddings_generated": random.choice([True, False]),
            "error_details": None if random.random() > 0.2 else "Processing failed due to invalid format"
        }

    def _generate_search_results_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate search results callback."""
        return {
            "callback_type": callback_type,
            "query_id": str(uuid.uuid4()),
            "results_count": random.randint(0, 100),
            "search_time_ms": random.randint(10, 1000),
            "results": [self._generate_search_result() for _ in range(random.randint(0, 5))]
        }

    def _generate_error_notification_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate error notification callback."""
        return {
            "callback_type": callback_type,
            "error_code": random.choice(["INTERNAL_ERROR", "NETWORK_ERROR", "TIMEOUT", "INVALID_REQUEST"]),
            "error_message": "A simulated error occurred during processing",
            "service_name": random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
            "retry_possible": random.choice([True, False])
        }

    def _generate_progress_update_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate progress update callback."""
        return {
            "callback_type": callback_type,
            "operation_id": str(uuid.uuid4()),
            "progress_percent": random.randint(0, 100),
            "current_step": random.choice(["initializing", "processing", "finalizing"]),
            "estimated_remaining_time_ms": random.randint(1000, 60000)
        }

    def _generate_health_status_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate health status callback."""
        return {
            "callback_type": callback_type,
            "service_name": random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
            "status": random.choice(["HEALTHY", "DEGRADED", "UNHEALTHY"]),
            "metrics": {
                "response_time_ms": random.randint(1, 1000),
                "error_rate": random.uniform(0.0, 0.1),
                "active_requests": random.randint(0, 100)
            }
        }

    def _generate_generic_callback(self, callback_type: str) -> Dict[str, Any]:
        """Generate generic callback message."""
        return {
            "callback_type": callback_type,
            "timestamp": int(time.time()),
            "data": {
                "generic_field": "generic_value",
                "numeric_value": random.randint(1, 1000)
            }
        }

    # Helper methods for generating realistic data

    def _generate_sample_text(self) -> str:
        """Generate sample text content."""
        texts = [
            "This is a sample document for testing purposes.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "The quick brown fox jumps over the lazy dog.",
            "Python is a versatile programming language.",
            "Machine learning models require large datasets."
        ]
        return random.choice(texts) + f" Generated at {time.time()}"

    def _generate_sample_content(self) -> bytes:
        """Generate sample binary content."""
        text = self._generate_sample_text()
        return text.encode('utf-8')

    def _generate_metadata(self) -> Dict[str, str]:
        """Generate sample metadata."""
        return {
            "author": random.choice(["Alice", "Bob", "Charlie"]),
            "type": random.choice(["document", "note", "report"]),
            "language": random.choice(["en", "es", "fr", "de"]),
            "created_by": "test_system",
            "version": str(random.randint(1, 10))
        }

    def _generate_project_id(self) -> str:
        """Generate project ID."""
        projects = ["workspace-qdrant-mcp", "test-project", "demo-app", "ml-pipeline"]
        return random.choice(projects)

    def _generate_search_query(self) -> str:
        """Generate realistic search queries."""
        queries = [
            "machine learning algorithms",
            "python async programming",
            "gRPC communication patterns",
            "vector database operations",
            "document processing techniques"
        ]
        return random.choice(queries)

    def _generate_search_result(self) -> Dict[str, Any]:
        """Generate a search result item."""
        return {
            "document_id": str(uuid.uuid4()),
            "score": random.uniform(0.0, 1.0),
            "title": f"Document {random.randint(1, 1000)}",
            "content_preview": self._generate_sample_text()[:100],
            "metadata": self._generate_metadata(),
            "highlight_spans": [
                {"start": random.randint(0, 50), "end": random.randint(51, 100)}
                for _ in range(random.randint(0, 3))
            ]
        }

    def _generate_search_filters(self) -> Dict[str, str]:
        """Generate search filters."""
        filters = {}
        if random.random() > 0.5:
            filters["project_id"] = self._generate_project_id()
        if random.random() > 0.5:
            filters["document_type"] = random.choice(["text", "pdf", "markdown"])
        if random.random() > 0.5:
            filters["author"] = random.choice(["Alice", "Bob", "Charlie"])
        return filters

    def _generate_collection_id(self) -> str:
        """Generate collection ID."""
        return f"collection_{random.randint(1000, 9999)}"

    def _generate_metadata_schema(self) -> Dict[str, str]:
        """Generate metadata schema definition."""
        return {
            "title": "string",
            "author": "string",
            "created_at": "timestamp",
            "tags": "array<string>",
            "score": "float"
        }

    def _generate_collection_info(self) -> Dict[str, Any]:
        """Generate collection information."""
        return {
            "collection_id": self._generate_collection_id(),
            "name": f"Collection {random.randint(1, 100)}",
            "vector_size": random.choice([384, 768, 1536]),
            "vectors_count": random.randint(0, 10000),
            "status": random.choice(["ACTIVE", "CREATING", "ERROR"]),
            "created_at": int(time.time() - random.randint(0, 86400 * 30))
        }

    def _generate_service_instance(self) -> Dict[str, Any]:
        """Generate service instance information."""
        return {
            "service_id": str(uuid.uuid4()),
            "service_name": random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
            "address": f"127.0.0.{random.randint(1, 254)}",
            "port": random.randint(50000, 60000),
            "health_status": random.choice(["HEALTHY", "DEGRADED", "UNHEALTHY"]),
            "last_seen": int(time.time() - random.randint(0, 300))
        }

    # Additional specialized generators for streaming and batch operations

    def generate_document_upload(self) -> Dict[str, Any]:
        """Generate document upload request."""
        return self._generate_document_processor_message("ProcessDocument", False)

    def generate_streaming_search(self) -> Dict[str, Any]:
        """Generate streaming search request."""
        return self._generate_search_service_message("SearchStream", False)

    def generate_batch_request(self) -> Dict[str, Any]:
        """Generate batch processing request."""
        return self._generate_document_processor_message("BatchProcess", False)

    def generate_progress_update(self) -> Dict[str, Any]:
        """Generate progress update message."""
        return self._generate_progress_update_callback("progress_update")

    def generate_upload_response(self) -> Dict[str, Any]:
        """Generate upload response."""
        return self._generate_document_processor_message("ProcessDocument", True)

    def generate_completion_response(self) -> Dict[str, Any]:
        """Generate completion response."""
        return {
            "message_type": "OperationComplete",
            "operation_id": str(uuid.uuid4()),
            "status": "SUCCESS",
            "completion_time": int(time.time()),
            "result_summary": "Operation completed successfully"
        }

    def generate_search_response(self) -> Dict[str, Any]:
        """Generate search response."""
        return self._generate_search_service_message("Search", True)

    def generate_search_chunk(self) -> Dict[str, Any]:
        """Generate search result chunk."""
        return {
            "chunk_id": random.randint(1, 100),
            "results": [self._generate_search_result() for _ in range(random.randint(1, 5))],
            "has_more": random.choice([True, False])
        }

    def generate_search_completion(self) -> Dict[str, Any]:
        """Generate search completion message."""
        return {
            "message_type": "SearchComplete",
            "total_results": random.randint(0, 1000),
            "total_time_ms": random.randint(100, 5000),
            "query_id": str(uuid.uuid4())
        }

    def generate_batch_response(self) -> Dict[str, Any]:
        """Generate batch response."""
        return self._generate_document_processor_message("BatchProcess", True)

    def generate_batch_status(self) -> Dict[str, Any]:
        """Generate batch status update."""
        return {
            "batch_id": str(uuid.uuid4()),
            "status": random.choice(["QUEUED", "PROCESSING", "COMPLETED", "FAILED"]),
            "processed_count": random.randint(0, 100),
            "failed_count": random.randint(0, 10),
            "progress_percent": random.randint(0, 100)
        }

    def generate_batch_completion(self) -> Dict[str, Any]:
        """Generate batch completion message."""
        return {
            "message_type": "BatchComplete",
            "batch_id": str(uuid.uuid4()),
            "total_processed": random.randint(1, 100),
            "successful_count": random.randint(1, 95),
            "failed_count": random.randint(0, 5),
            "completion_time": int(time.time())
        }

    def generate_health_check(self) -> Dict[str, Any]:
        """Generate health check message."""
        return self._generate_system_service_message("GetSystemStatus", False)

    def generate_metrics(self) -> Dict[str, Any]:
        """Generate metrics collection message."""
        return {
            "message_type": "MetricsCollection",
            "timestamp": int(time.time()),
            "cpu_usage": random.uniform(0.0, 100.0),
            "memory_usage": random.randint(100, 8192),
            "disk_usage": random.uniform(0.0, 100.0),
            "network_io": {
                "bytes_sent": random.randint(1000, 1000000),
                "bytes_received": random.randint(1000, 1000000)
            }
        }

    def generate_alert(self) -> Dict[str, Any]:
        """Generate alert notification."""
        return {
            "message_type": "AlertNotification",
            "alert_id": str(uuid.uuid4()),
            "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "message": "System alert generated for testing",
            "service_affected": random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
            "timestamp": int(time.time())
        }

    def generate_health_status(self) -> Dict[str, Any]:
        """Generate health status message."""
        return self._generate_system_service_message("GetSystemStatus", True)

    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate metrics report."""
        return {
            "message_type": "MetricsReport",
            "report_id": str(uuid.uuid4()),
            "time_range": {
                "start": int(time.time() - 3600),
                "end": int(time.time())
            },
            "service_metrics": {
                service: {
                    "avg_response_time": random.randint(10, 1000),
                    "request_count": random.randint(100, 10000),
                    "error_count": random.randint(0, 100)
                }
                for service in ["DocumentProcessor", "SearchService", "MemoryService"]
            }
        }

    def generate_alert_resolution(self) -> Dict[str, Any]:
        """Generate alert resolution message."""
        return {
            "message_type": "AlertResolution",
            "alert_id": str(uuid.uuid4()),
            "resolved_at": int(time.time()),
            "resolution_method": random.choice(["AUTOMATIC", "MANUAL", "TIMEOUT"]),
            "resolution_notes": "Alert resolved automatically"
        }

    def generate_cli_grpc_message(self, cli_command: str, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gRPC message for CLI command."""
        if "service" in cli_command:
            return self.generate_service_message("SystemService", "GetSystemStatus")
        elif "admin" in cli_command:
            return self.generate_service_message("MemoryService", "ListCollections")
        elif "health" in cli_command:
            return self.generate_service_message("SystemService", "GetSystemStatus")
        elif "document" in cli_command:
            return self.generate_service_message("DocumentProcessor", "ProcessDocument")
        elif "search" in cli_command:
            return self.generate_service_message("SearchService", "Search", request_data=cli_data)
        else:
            return self.generate_service_message("SystemService", "GetSystemStatus")