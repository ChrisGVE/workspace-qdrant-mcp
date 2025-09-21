"""
External service mocking for testing third-party API interactions.

Provides comprehensive mocking for external services including API endpoints,
authentication, rate limiting, and various service-specific error scenarios.
"""

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlparse

from .error_injection import ErrorInjector, FailureScenarios


class ExternalServiceErrorInjector(ErrorInjector):
    """Specialized error injector for external service operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "service_unavailable": {"probability": 0.0, "status_code": 503},
            "api_rate_limited": {"probability": 0.0, "status_code": 429, "retry_after": 60},
            "authentication_failed": {"probability": 0.0, "status_code": 401},
            "authorization_denied": {"probability": 0.0, "status_code": 403},
            "api_key_invalid": {"probability": 0.0, "status_code": 401},
            "quota_exceeded": {"probability": 0.0, "status_code": 402},
            "request_timeout": {"probability": 0.0, "timeout_seconds": 30.0},
            "invalid_request": {"probability": 0.0, "status_code": 400},
            "resource_not_found": {"probability": 0.0, "status_code": 404},
            "server_error": {"probability": 0.0, "status_code": 500},
            "bad_gateway": {"probability": 0.0, "status_code": 502},
            "maintenance_mode": {"probability": 0.0, "status_code": 503},
            "api_version_deprecated": {"probability": 0.0, "status_code": 410},
            "payload_too_large": {"probability": 0.0, "status_code": 413},
            "unsupported_media_type": {"probability": 0.0, "status_code": 415},
        }

    def configure_availability_issues(self, probability: float = 0.1):
        """Configure service availability failures."""
        self.failure_modes["service_unavailable"]["probability"] = probability
        self.failure_modes["maintenance_mode"]["probability"] = probability / 3
        self.failure_modes["bad_gateway"]["probability"] = probability / 2

    def configure_auth_issues(self, probability: float = 0.1):
        """Configure authentication-related failures."""
        self.failure_modes["authentication_failed"]["probability"] = probability
        self.failure_modes["authorization_denied"]["probability"] = probability / 2
        self.failure_modes["api_key_invalid"]["probability"] = probability / 3

    def configure_rate_limiting(self, probability: float = 0.1):
        """Configure rate limiting failures."""
        self.failure_modes["api_rate_limited"]["probability"] = probability
        self.failure_modes["quota_exceeded"]["probability"] = probability / 2

    def configure_client_errors(self, probability: float = 0.1):
        """Configure client-side error failures."""
        self.failure_modes["invalid_request"]["probability"] = probability
        self.failure_modes["resource_not_found"]["probability"] = probability / 2
        self.failure_modes["payload_too_large"]["probability"] = probability / 3


class ExternalServiceMock:
    """Generic mock for external service API interactions."""

    def __init__(self,
                 service_name: str = "mock-service",
                 base_url: str = "https://api.mock-service.com",
                 error_injector: Optional[ExternalServiceErrorInjector] = None):
        self.service_name = service_name
        self.base_url = base_url
        self.error_injector = error_injector or ExternalServiceErrorInjector()
        self.operation_history: List[Dict[str, Any]] = []
        self.authenticated = False
        self.api_key: Optional[str] = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset_time = time.time() + 3600

        # Setup method mocks
        self._setup_service_methods()

    def _setup_service_methods(self):
        """Setup external service method mocks."""
        self.authenticate = AsyncMock(side_effect=self._mock_authenticate)
        self.make_request = AsyncMock(side_effect=self._mock_make_request)
        self.get_resource = AsyncMock(side_effect=self._mock_get_resource)
        self.create_resource = AsyncMock(side_effect=self._mock_create_resource)
        self.update_resource = AsyncMock(side_effect=self._mock_update_resource)
        self.delete_resource = AsyncMock(side_effect=self._mock_delete_resource)
        self.health_check = AsyncMock(side_effect=self._mock_health_check)

    async def _inject_service_error(self, operation: str, endpoint: str) -> None:
        """Inject service errors based on configuration."""
        # Simulate rate limiting
        if self.rate_limit_remaining <= 0:
            if time.time() < self.rate_limit_reset_time:
                await self._raise_service_error("api_rate_limited")

        # Decrease rate limit
        self.rate_limit_remaining = max(0, self.rate_limit_remaining - 1)

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_service_error(error_type)

    async def _raise_service_error(self, error_type: str) -> None:
        """Raise appropriate service error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})

        if error_type == "service_unavailable":
            raise ConnectionError("Service unavailable")
        elif error_type == "api_rate_limited":
            retry_after = error_config.get("retry_after", 60)
            raise Exception(f"Rate limit exceeded. Retry after {retry_after} seconds")
        elif error_type == "authentication_failed":
            raise PermissionError("Authentication failed")
        elif error_type == "authorization_denied":
            raise PermissionError("Authorization denied")
        elif error_type == "api_key_invalid":
            raise PermissionError("Invalid API key")
        elif error_type == "quota_exceeded":
            raise Exception("API quota exceeded")
        elif error_type == "request_timeout":
            timeout = error_config.get("timeout_seconds", 30.0)
            await asyncio.sleep(timeout)
            raise TimeoutError("Request timeout")
        elif error_type == "invalid_request":
            raise ValueError("Invalid request format")
        elif error_type == "resource_not_found":
            raise FileNotFoundError("Resource not found")
        elif error_type == "server_error":
            raise Exception("Internal server error")
        elif error_type == "bad_gateway":
            raise ConnectionError("Bad gateway")
        elif error_type == "maintenance_mode":
            raise ConnectionError("Service in maintenance mode")
        elif error_type == "api_version_deprecated":
            raise Exception("API version deprecated")
        elif error_type == "payload_too_large":
            raise ValueError("Payload too large")
        elif error_type == "unsupported_media_type":
            raise ValueError("Unsupported media type")

    async def _mock_authenticate(self, api_key: str, **kwargs) -> Dict[str, Any]:
        """Mock service authentication."""
        await self._inject_service_error("authenticate", "/auth")

        if not api_key or api_key == "invalid_key":
            raise PermissionError("Invalid API key")

        self.authenticated = True
        self.api_key = api_key

        self.operation_history.append({
            "operation": "authenticate",
            "api_key": api_key[:8] + "..." if len(api_key) > 8 else api_key,
            "success": True
        })

        return {
            "authenticated": True,
            "user_id": "user_12345",
            "expires_at": "2024-12-31T23:59:59Z",
            "rate_limit": 1000,
            "permissions": ["read", "write"]
        }

    async def _mock_make_request(self,
                                method: str,
                                endpoint: str,
                                data: Optional[Dict[str, Any]] = None,
                                headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Mock generic API request."""
        await self._inject_service_error("make_request", endpoint)

        if not self.authenticated:
            raise PermissionError("Not authenticated")

        self.operation_history.append({
            "operation": "make_request",
            "method": method,
            "endpoint": endpoint,
            "data_keys": list(data.keys()) if data else [],
            "has_headers": bool(headers)
        })

        # Generate response based on endpoint
        return self._generate_endpoint_response(method, endpoint, data)

    async def _mock_get_resource(self, resource_id: str, resource_type: str = "item") -> Dict[str, Any]:
        """Mock getting a resource by ID."""
        endpoint = f"/{resource_type}s/{resource_id}"
        await self._inject_service_error("get_resource", endpoint)

        if not self.authenticated:
            raise PermissionError("Not authenticated")

        self.operation_history.append({
            "operation": "get_resource",
            "resource_id": resource_id,
            "resource_type": resource_type
        })

        return {
            "id": resource_id,
            "type": resource_type,
            "name": f"Mock {resource_type} {resource_id}",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
            "status": "active",
            "metadata": {
                "service": self.service_name,
                "version": "1.0"
            }
        }

    async def _mock_create_resource(self,
                                   resource_type: str,
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock creating a new resource."""
        endpoint = f"/{resource_type}s"
        await self._inject_service_error("create_resource", endpoint)

        if not self.authenticated:
            raise PermissionError("Not authenticated")

        resource_id = f"{resource_type}_{random.randint(1000, 9999)}"

        self.operation_history.append({
            "operation": "create_resource",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "data_keys": list(data.keys())
        })

        return {
            "id": resource_id,
            "type": resource_type,
            "status": "created",
            "created_at": "2024-01-01T12:00:00Z",
            "data": data
        }

    async def _mock_update_resource(self,
                                   resource_id: str,
                                   resource_type: str,
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock updating an existing resource."""
        endpoint = f"/{resource_type}s/{resource_id}"
        await self._inject_service_error("update_resource", endpoint)

        if not self.authenticated:
            raise PermissionError("Not authenticated")

        self.operation_history.append({
            "operation": "update_resource",
            "resource_id": resource_id,
            "resource_type": resource_type,
            "data_keys": list(data.keys())
        })

        return {
            "id": resource_id,
            "type": resource_type,
            "status": "updated",
            "updated_at": "2024-01-01T12:00:00Z",
            "changes": data
        }

    async def _mock_delete_resource(self, resource_id: str, resource_type: str) -> Dict[str, Any]:
        """Mock deleting a resource."""
        endpoint = f"/{resource_type}s/{resource_id}"
        await self._inject_service_error("delete_resource", endpoint)

        if not self.authenticated:
            raise PermissionError("Not authenticated")

        self.operation_history.append({
            "operation": "delete_resource",
            "resource_id": resource_id,
            "resource_type": resource_type
        })

        return {
            "id": resource_id,
            "type": resource_type,
            "status": "deleted",
            "deleted_at": "2024-01-01T12:00:00Z"
        }

    async def _mock_health_check(self) -> Dict[str, Any]:
        """Mock service health check."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type in ["service_unavailable", "maintenance_mode"]:
                await self._raise_service_error(error_type)

        self.operation_history.append({
            "operation": "health_check"
        })

        return {
            "status": "healthy",
            "service": self.service_name,
            "version": "1.0.0",
            "timestamp": "2024-01-01T12:00:00Z",
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset_time
        }

    def _generate_endpoint_response(self,
                                   method: str,
                                   endpoint: str,
                                   data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate realistic response based on endpoint and method."""
        if endpoint.startswith("/search"):
            return self._generate_search_response(data)
        elif endpoint.startswith("/analytics"):
            return self._generate_analytics_response()
        elif endpoint.startswith("/upload"):
            return self._generate_upload_response(data)
        elif endpoint.startswith("/webhook"):
            return self._generate_webhook_response(method, data)
        else:
            return self._generate_generic_response(method, endpoint, data)

    def _generate_search_response(self, query_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate search API response."""
        query = query_data.get("query", "test") if query_data else "test"
        limit = query_data.get("limit", 10) if query_data else 10

        results = []
        for i in range(min(limit, random.randint(1, 8))):
            results.append({
                "id": f"result_{i}",
                "title": f"Search Result {i} for '{query}'",
                "score": 0.95 - (i * 0.1),
                "snippet": f"This is a mock search result snippet for {query}",
                "url": f"https://example.com/result/{i}"
            })

        return {
            "query": query,
            "results": results,
            "total_count": len(results),
            "page": 1,
            "has_more": len(results) == limit
        }

    def _generate_analytics_response(self) -> Dict[str, Any]:
        """Generate analytics API response."""
        return {
            "metrics": {
                "page_views": random.randint(1000, 10000),
                "unique_visitors": random.randint(500, 5000),
                "bounce_rate": round(random.uniform(0.2, 0.8), 2),
                "avg_session_duration": random.randint(120, 600)
            },
            "period": "7d",
            "generated_at": "2024-01-01T12:00:00Z"
        }

    def _generate_upload_response(self, upload_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate file upload API response."""
        file_name = upload_data.get("file_name", "unknown.txt") if upload_data else "unknown.txt"
        file_size = upload_data.get("file_size", random.randint(1024, 1048576)) if upload_data else random.randint(1024, 1048576)

        return {
            "upload_id": f"upload_{random.randint(1000, 9999)}",
            "file_name": file_name,
            "file_size": file_size,
            "url": f"https://cdn.{self.service_name}.com/{file_name}",
            "status": "uploaded",
            "uploaded_at": "2024-01-01T12:00:00Z"
        }

    def _generate_webhook_response(self, method: str, webhook_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate webhook API response."""
        if method == "POST":
            return {
                "webhook_id": f"webhook_{random.randint(1000, 9999)}",
                "url": webhook_data.get("url", "https://example.com/webhook") if webhook_data else "https://example.com/webhook",
                "events": webhook_data.get("events", ["all"]) if webhook_data else ["all"],
                "status": "active",
                "created_at": "2024-01-01T12:00:00Z"
            }
        else:
            return {
                "message": f"Webhook {method} processed",
                "timestamp": "2024-01-01T12:00:00Z"
            }

    def _generate_generic_response(self,
                                  method: str,
                                  endpoint: str,
                                  data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate generic API response."""
        return {
            "message": f"{method} request to {endpoint} processed successfully",
            "request_id": f"req_{random.randint(1000, 9999)}",
            "timestamp": "2024-01-01T12:00:00Z",
            "data": data,
            "service": self.service_name
        }

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of service operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset service state."""
        self.operation_history.clear()
        self.authenticated = False
        self.api_key = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset_time = time.time() + 3600
        self.error_injector.reset()


class ThirdPartyAPIMock:
    """Mock for specific third-party API services."""

    def __init__(self, api_type: str = "generic", error_injector: Optional[ExternalServiceErrorInjector] = None):
        self.api_type = api_type
        self.error_injector = error_injector or ExternalServiceErrorInjector()
        self.operation_history: List[Dict[str, Any]] = []

        # Setup API-specific methods
        self._setup_api_methods()

    def _setup_api_methods(self):
        """Setup API-specific method mocks."""
        if self.api_type == "openai":
            self._setup_openai_methods()
        elif self.api_type == "anthropic":
            self._setup_anthropic_methods()
        elif self.api_type == "pinecone":
            self._setup_pinecone_methods()
        elif self.api_type == "elasticsearch":
            self._setup_elasticsearch_methods()
        else:
            self._setup_generic_methods()

    def _setup_openai_methods(self):
        """Setup OpenAI API mock methods."""
        self.create_completion = AsyncMock(side_effect=self._mock_openai_completion)
        self.create_embedding = AsyncMock(side_effect=self._mock_openai_embedding)
        self.list_models = AsyncMock(side_effect=self._mock_openai_list_models)

    def _setup_anthropic_methods(self):
        """Setup Anthropic API mock methods."""
        self.create_message = AsyncMock(side_effect=self._mock_anthropic_message)
        self.stream_message = AsyncMock(side_effect=self._mock_anthropic_stream)

    def _setup_pinecone_methods(self):
        """Setup Pinecone API mock methods."""
        self.upsert_vectors = AsyncMock(side_effect=self._mock_pinecone_upsert)
        self.query_vectors = AsyncMock(side_effect=self._mock_pinecone_query)
        self.delete_vectors = AsyncMock(side_effect=self._mock_pinecone_delete)

    def _setup_elasticsearch_methods(self):
        """Setup Elasticsearch API mock methods."""
        self.index_document = AsyncMock(side_effect=self._mock_elasticsearch_index)
        self.search_documents = AsyncMock(side_effect=self._mock_elasticsearch_search)
        self.delete_document = AsyncMock(side_effect=self._mock_elasticsearch_delete)

    def _setup_generic_methods(self):
        """Setup generic API mock methods."""
        self.api_call = AsyncMock(side_effect=self._mock_generic_api_call)

    async def _mock_openai_completion(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        """Mock OpenAI completion API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("openai_completion")

        self.operation_history.append({
            "operation": "openai_completion",
            "model": model,
            "prompt_length": len(prompt)
        })

        return {
            "id": f"cmpl-{random.randint(1000000, 9999999)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": f"Mock completion response for: {prompt[:50]}...",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 50,
                "total_tokens": len(prompt) // 4 + 50
            }
        }

    async def _mock_openai_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """Mock OpenAI embedding API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("openai_embedding")

        self.operation_history.append({
            "operation": "openai_embedding",
            "model": model,
            "text_length": len(text)
        })

        # Generate mock embedding
        embedding = [random.uniform(-1, 1) for _ in range(1536)]  # OpenAI embedding dimension

        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embedding,
                "index": 0
            }],
            "model": model,
            "usage": {
                "prompt_tokens": len(text) // 4,
                "total_tokens": len(text) // 4
            }
        }

    async def _mock_openai_list_models(self) -> Dict[str, Any]:
        """Mock OpenAI list models API."""
        return {
            "object": "list",
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
                {"id": "gpt-4", "object": "model", "owned_by": "openai"},
                {"id": "text-embedding-ada-002", "object": "model", "owned_by": "openai"}
            ]
        }

    async def _mock_anthropic_message(self, messages: List[Dict[str, str]], model: str = "claude-3-sonnet-20240229") -> Dict[str, Any]:
        """Mock Anthropic messages API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("anthropic_message")

        self.operation_history.append({
            "operation": "anthropic_message",
            "model": model,
            "message_count": len(messages)
        })

        return {
            "id": f"msg_{random.randint(1000000, 9999999)}",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": f"Mock response from {model}"
            }],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": sum(len(msg.get("content", "")) for msg in messages) // 4,
                "output_tokens": 50
            }
        }

    async def _mock_anthropic_stream(self, messages: List[Dict[str, str]], model: str = "claude-3-sonnet-20240229"):
        """Mock Anthropic streaming API."""
        for i in range(5):  # Mock streaming chunks
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": f"Chunk {i} "}
            }
            await asyncio.sleep(0.1)

    async def _mock_pinecone_upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> Dict[str, Any]:
        """Mock Pinecone upsert API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("pinecone_upsert")

        self.operation_history.append({
            "operation": "pinecone_upsert",
            "vector_count": len(vectors),
            "namespace": namespace
        })

        return {
            "upserted_count": len(vectors)
        }

    async def _mock_pinecone_query(self, vector: List[float], top_k: int = 10, namespace: str = "") -> Dict[str, Any]:
        """Mock Pinecone query API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("pinecone_query")

        self.operation_history.append({
            "operation": "pinecone_query",
            "vector_dim": len(vector),
            "top_k": top_k,
            "namespace": namespace
        })

        matches = []
        for i in range(min(top_k, random.randint(1, 5))):
            matches.append({
                "id": f"vec_{i}",
                "score": 0.95 - (i * 0.1),
                "values": [random.uniform(-1, 1) for _ in range(len(vector))],
                "metadata": {"source": f"document_{i}"}
            })

        return {
            "matches": matches,
            "namespace": namespace
        }

    async def _mock_pinecone_delete(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """Mock Pinecone delete API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("pinecone_delete")

        self.operation_history.append({
            "operation": "pinecone_delete",
            "id_count": len(ids),
            "namespace": namespace
        })

        return {"deleted_count": len(ids)}

    async def _mock_elasticsearch_index(self, index: str, doc_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Elasticsearch index API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("elasticsearch_index")

        self.operation_history.append({
            "operation": "elasticsearch_index",
            "index": index,
            "doc_id": doc_id
        })

        return {
            "_index": index,
            "_id": doc_id,
            "_version": 1,
            "result": "created"
        }

    async def _mock_elasticsearch_search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Elasticsearch search API."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("elasticsearch_search")

        self.operation_history.append({
            "operation": "elasticsearch_search",
            "index": index,
            "query": query
        })

        hits = []
        for i in range(random.randint(1, 5)):
            hits.append({
                "_index": index,
                "_id": f"doc_{i}",
                "_score": 0.95 - (i * 0.1),
                "_source": {
                    "title": f"Document {i}",
                    "content": f"Mock search result content {i}"
                }
            })

        return {
            "took": random.randint(5, 50),
            "timed_out": False,
            "hits": {
                "total": {"value": len(hits)},
                "hits": hits
            }
        }

    async def _mock_elasticsearch_delete(self, index: str, doc_id: str) -> Dict[str, Any]:
        """Mock Elasticsearch delete API."""
        return {
            "_index": index,
            "_id": doc_id,
            "_version": 2,
            "result": "deleted"
        }

    async def _mock_generic_api_call(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock generic API call."""
        if self.error_injector.should_inject_error():
            await self._raise_api_error("generic_api_call")

        self.operation_history.append({
            "operation": "generic_api_call",
            "endpoint": endpoint,
            "method": method
        })

        return {
            "status": "success",
            "endpoint": endpoint,
            "method": method,
            "data": data,
            "timestamp": "2024-01-01T12:00:00Z"
        }

    async def _raise_api_error(self, operation: str) -> None:
        """Raise API-specific errors."""
        error_type = self.error_injector.get_random_error()
        if error_type == "api_rate_limited":
            raise Exception("API rate limit exceeded")
        elif error_type == "authentication_failed":
            raise PermissionError("API authentication failed")
        elif error_type == "quota_exceeded":
            raise Exception("API quota exceeded")
        else:
            raise Exception(f"API error in {operation}")

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of API operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset API state."""
        self.operation_history.clear()
        self.error_injector.reset()


class ServiceUnavailableMock:
    """Mock for simulating complete service unavailability."""

    def __init__(self, service_name: str = "unavailable-service"):
        self.service_name = service_name
        self.operation_history: List[Dict[str, Any]] = []

    def __getattr__(self, name: str):
        """Mock any method call to raise service unavailable error."""
        async def unavailable_method(*args, **kwargs):
            self.operation_history.append({
                "operation": name,
                "args": len(args),
                "kwargs": list(kwargs.keys()),
                "error": "service_unavailable"
            })
            raise ConnectionError(f"{self.service_name} is unavailable")

        return unavailable_method

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of failed operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset unavailable service state."""
        self.operation_history.clear()


def create_external_service_mock(
    service_type: str = "generic",
    api_type: str = "generic",
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> Union[ExternalServiceMock, ThirdPartyAPIMock, ServiceUnavailableMock]:
    """
    Create an external service mock with optional error injection.

    Args:
        service_type: Type of service mock ("generic", "api", "unavailable")
        api_type: Specific API type for ThirdPartyAPIMock
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured external service mock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = ExternalServiceErrorInjector()
        error_injector.configure_availability_issues(error_probability)
        error_injector.configure_auth_issues(error_probability / 2)
        error_injector.configure_rate_limiting(error_probability / 3)

    if service_type == "generic":
        return ExternalServiceMock(error_injector=error_injector)
    elif service_type == "api":
        return ThirdPartyAPIMock(api_type, error_injector)
    elif service_type == "unavailable":
        return ServiceUnavailableMock()
    else:
        raise ValueError(f"Unknown service type: {service_type}")


# Convenience functions for common scenarios
def create_basic_external_service() -> ExternalServiceMock:
    """Create basic external service mock without error injection."""
    return create_external_service_mock("generic")


def create_failing_external_service(error_rate: float = 0.4) -> ExternalServiceMock:
    """Create external service mock with high failure rate."""
    return create_external_service_mock("generic", with_error_injection=True, error_probability=error_rate)


def create_openai_api_mock() -> ThirdPartyAPIMock:
    """Create OpenAI API mock with realistic behavior."""
    return create_external_service_mock("api", "openai", with_error_injection=True, error_probability=0.05)


def create_unavailable_service() -> ServiceUnavailableMock:
    """Create completely unavailable service mock."""
    return create_external_service_mock("unavailable")