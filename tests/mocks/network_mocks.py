"""
Network operation mocking for testing HTTP requests and connectivity.

Provides comprehensive mocking for network operations including HTTP requests,
connection failures, timeouts, and various network-related error scenarios.
"""

import asyncio
import json
import random
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlparse

from .error_injection import ErrorInjector, FailureScenarios


class NetworkErrorInjector(ErrorInjector):
    """Specialized error injector for network operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "connection_timeout": {"probability": 0.0, "timeout_seconds": 30.0},
            "connection_refused": {"probability": 0.0, "errno": 61},
            "connection_reset": {"probability": 0.0, "errno": 54},
            "dns_resolution_failed": {"probability": 0.0, "errno": -2},
            "network_unreachable": {"probability": 0.0, "errno": 51},
            "ssl_error": {"probability": 0.0, "error_type": "certificate"},
            "http_500": {"probability": 0.0, "status_code": 500},
            "http_502": {"probability": 0.0, "status_code": 502},
            "http_503": {"probability": 0.0, "status_code": 503},
            "http_429": {"probability": 0.0, "status_code": 429},
            "http_404": {"probability": 0.0, "status_code": 404},
            "http_401": {"probability": 0.0, "status_code": 401},
            "http_403": {"probability": 0.0, "status_code": 403},
            "partial_content": {"probability": 0.0, "truncate_at": 0.5},
            "malformed_response": {"probability": 0.0, "corruption_type": "json"},
        }

    def configure_connection_issues(self, probability: float = 0.1):
        """Configure connection-related failures."""
        self.failure_modes["connection_timeout"]["probability"] = probability
        self.failure_modes["connection_refused"]["probability"] = probability / 2
        self.failure_modes["connection_reset"]["probability"] = probability / 3
        self.failure_modes["dns_resolution_failed"]["probability"] = probability / 4

    def configure_server_issues(self, probability: float = 0.1):
        """Configure server-side failures."""
        self.failure_modes["http_500"]["probability"] = probability
        self.failure_modes["http_502"]["probability"] = probability / 2
        self.failure_modes["http_503"]["probability"] = probability / 2

    def configure_auth_issues(self, probability: float = 0.1):
        """Configure authentication-related failures."""
        self.failure_modes["http_401"]["probability"] = probability
        self.failure_modes["http_403"]["probability"] = probability / 2

    def configure_rate_limiting(self, probability: float = 0.1):
        """Configure rate limiting failures."""
        self.failure_modes["http_429"]["probability"] = probability

    def configure_data_corruption(self, probability: float = 0.05):
        """Configure data corruption scenarios."""
        self.failure_modes["partial_content"]["probability"] = probability
        self.failure_modes["malformed_response"]["probability"] = probability / 2


class MockHTTPResponse:
    """Mock HTTP response object."""

    def __init__(self, status_code: int = 200, content: Union[str, bytes] = "",
                 headers: Optional[Dict[str, str]] = None, json_data: Optional[Any] = None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}
        self._json_data = json_data

    @property
    def content(self) -> bytes:
        if isinstance(self._content, str):
            return self._content.encode('utf-8')
        return self._content

    @property
    def text(self) -> str:
        if isinstance(self._content, bytes):
            return self._content.decode('utf-8')
        return self._content

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        if self._content:
            try:
                return json.loads(self.text)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON content")
        return {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} Error")


class NetworkClientMock:
    """Mock network client for HTTP operations."""

    def __init__(self, error_injector: Optional[NetworkErrorInjector] = None):
        self.error_injector = error_injector or NetworkErrorInjector()
        self.operation_history: List[Dict[str, Any]] = []
        self.session_active = False
        self.performance_delays = {
            "get": 0.1,
            "post": 0.15,
            "put": 0.15,
            "delete": 0.1,
            "head": 0.05,
        }

        # Setup method mocks
        self._setup_http_methods()

    def _setup_http_methods(self):
        """Setup HTTP method mocks."""
        self.get = AsyncMock(side_effect=self._mock_get)
        self.post = AsyncMock(side_effect=self._mock_post)
        self.put = AsyncMock(side_effect=self._mock_put)
        self.delete = AsyncMock(side_effect=self._mock_delete)
        self.head = AsyncMock(side_effect=self._mock_head)
        self.request = AsyncMock(side_effect=self._mock_request)

    async def _inject_network_error(self, method: str, url: str) -> None:
        """Inject network errors based on configuration."""
        # Add realistic network delay
        if method.lower() in self.performance_delays:
            await asyncio.sleep(self.performance_delays[method.lower()])

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_network_error(error_type, url)

    async def _raise_network_error(self, error_type: str, url: str) -> None:
        """Raise appropriate network error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})

        if error_type == "connection_timeout":
            timeout = error_config.get("timeout_seconds", 30.0)
            await asyncio.sleep(timeout)
            raise ConnectionError(f"Connection timeout to {url}")

        elif error_type == "connection_refused":
            raise ConnectionError(f"Connection refused to {url}")

        elif error_type == "connection_reset":
            raise ConnectionError(f"Connection reset by peer to {url}")

        elif error_type == "dns_resolution_failed":
            parsed = urlparse(url)
            raise ConnectionError(f"DNS resolution failed for {parsed.hostname}")

        elif error_type == "network_unreachable":
            raise ConnectionError(f"Network unreachable to {url}")

        elif error_type == "ssl_error":
            raise ConnectionError(f"SSL certificate verification failed for {url}")

        elif error_type.startswith("http_"):
            status_code = error_config.get("status_code", 500)
            response = MockHTTPResponse(
                status_code=status_code,
                content=f"HTTP {status_code} Error"
            )
            response.raise_for_status()

    async def _mock_get(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock HTTP GET request."""
        await self._inject_network_error("GET", url)

        self.operation_history.append({
            "operation": "GET",
            "url": url,
            "kwargs": kwargs
        })

        return self._generate_mock_response(url, "GET", kwargs)

    async def _mock_post(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock HTTP POST request."""
        await self._inject_network_error("POST", url)

        self.operation_history.append({
            "operation": "POST",
            "url": url,
            "data": kwargs.get("data"),
            "json": kwargs.get("json"),
            "kwargs": {k: v for k, v in kwargs.items() if k not in ["data", "json"]}
        })

        return self._generate_mock_response(url, "POST", kwargs)

    async def _mock_put(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock HTTP PUT request."""
        await self._inject_network_error("PUT", url)

        self.operation_history.append({
            "operation": "PUT",
            "url": url,
            "data": kwargs.get("data"),
            "json": kwargs.get("json"),
            "kwargs": {k: v for k, v in kwargs.items() if k not in ["data", "json"]}
        })

        return self._generate_mock_response(url, "PUT", kwargs)

    async def _mock_delete(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock HTTP DELETE request."""
        await self._inject_network_error("DELETE", url)

        self.operation_history.append({
            "operation": "DELETE",
            "url": url,
            "kwargs": kwargs
        })

        return self._generate_mock_response(url, "DELETE", kwargs)

    async def _mock_head(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock HTTP HEAD request."""
        await self._inject_network_error("HEAD", url)

        self.operation_history.append({
            "operation": "HEAD",
            "url": url,
            "kwargs": kwargs
        })

        return MockHTTPResponse(
            status_code=200,
            headers={
                "content-type": "application/json",
                "content-length": "1024",
                "server": "mock-server/1.0"
            }
        )

    async def _mock_request(self, method: str, url: str, **kwargs) -> MockHTTPResponse:
        """Mock generic HTTP request."""
        await self._inject_network_error(method, url)

        self.operation_history.append({
            "operation": method.upper(),
            "url": url,
            "kwargs": kwargs
        })

        return self._generate_mock_response(url, method, kwargs)

    def _generate_mock_response(self, url: str, method: str, kwargs: Dict[str, Any]) -> MockHTTPResponse:
        """Generate realistic mock response based on URL and method."""
        parsed_url = urlparse(url)

        # Generate response based on URL patterns
        if "api" in parsed_url.path.lower():
            return self._generate_api_response(url, method, kwargs)
        elif "health" in parsed_url.path.lower():
            return self._generate_health_response()
        elif "auth" in parsed_url.path.lower():
            return self._generate_auth_response(method, kwargs)
        else:
            return self._generate_generic_response(url, method)

    def _generate_api_response(self, url: str, method: str, kwargs: Dict[str, Any]) -> MockHTTPResponse:
        """Generate API-specific mock response."""
        if method == "GET":
            return MockHTTPResponse(
                status_code=200,
                json_data={
                    "data": [
                        {"id": 1, "name": "Item 1", "value": "value1"},
                        {"id": 2, "name": "Item 2", "value": "value2"}
                    ],
                    "total": 2,
                    "page": 1
                },
                headers={"content-type": "application/json"}
            )
        elif method in ["POST", "PUT"]:
            request_data = kwargs.get("json", kwargs.get("data", {}))
            return MockHTTPResponse(
                status_code=201 if method == "POST" else 200,
                json_data={
                    "id": random.randint(1000, 9999),
                    "created": method == "POST",
                    "data": request_data
                },
                headers={"content-type": "application/json"}
            )
        elif method == "DELETE":
            return MockHTTPResponse(
                status_code=204,
                headers={"content-type": "application/json"}
            )

    def _generate_health_response(self) -> MockHTTPResponse:
        """Generate health check response."""
        return MockHTTPResponse(
            status_code=200,
            json_data={
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "checks": {
                    "database": "healthy",
                    "cache": "healthy",
                    "external_api": "healthy"
                }
            },
            headers={"content-type": "application/json"}
        )

    def _generate_auth_response(self, method: str, kwargs: Dict[str, Any]) -> MockHTTPResponse:
        """Generate authentication response."""
        if method == "POST":
            credentials = kwargs.get("json", {})
            if credentials.get("username") == "test" and credentials.get("password") == "test":
                return MockHTTPResponse(
                    status_code=200,
                    json_data={
                        "access_token": "mock_access_token_12345",
                        "token_type": "bearer",
                        "expires_in": 3600,
                        "user_id": "user_123"
                    },
                    headers={"content-type": "application/json"}
                )
            else:
                return MockHTTPResponse(
                    status_code=401,
                    json_data={"error": "Invalid credentials"},
                    headers={"content-type": "application/json"}
                )

    def _generate_generic_response(self, url: str, method: str) -> MockHTTPResponse:
        """Generate generic mock response."""
        return MockHTTPResponse(
            status_code=200,
            json_data={
                "message": f"Mock response for {method} {url}",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": f"req_{random.randint(1000, 9999)}"
            },
            headers={"content-type": "application/json"}
        )

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of network operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset mock state."""
        self.operation_history.clear()
        self.session_active = False
        self.error_injector.reset()


class HTTPRequestMock:
    """Mock for requests-style HTTP library."""

    def __init__(self, error_injector: Optional[NetworkErrorInjector] = None):
        self.error_injector = error_injector or NetworkErrorInjector()
        self.operation_history: List[Dict[str, Any]] = []

        # Setup method mocks
        self.get = Mock(side_effect=self._mock_sync_get)
        self.post = Mock(side_effect=self._mock_sync_post)
        self.put = Mock(side_effect=self._mock_sync_put)
        self.delete = Mock(side_effect=self._mock_sync_delete)
        self.head = Mock(side_effect=self._mock_sync_head)
        self.request = Mock(side_effect=self._mock_sync_request)

    def _inject_sync_network_error(self, method: str, url: str) -> None:
        """Inject network errors for synchronous operations."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            self._raise_sync_network_error(error_type, url)

    def _raise_sync_network_error(self, error_type: str, url: str) -> None:
        """Raise appropriate network error for sync operations."""
        if error_type == "connection_timeout":
            raise ConnectionError(f"Connection timeout to {url}")
        elif error_type == "connection_refused":
            raise ConnectionError(f"Connection refused to {url}")
        elif error_type.startswith("http_"):
            error_config = self.error_injector.failure_modes.get(error_type, {})
            status_code = error_config.get("status_code", 500)
            response = MockHTTPResponse(status_code=status_code)
            response.raise_for_status()

    def _mock_sync_get(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous GET request."""
        self._inject_sync_network_error("GET", url)

        self.operation_history.append({
            "operation": "GET",
            "url": url,
            "kwargs": kwargs
        })

        return MockHTTPResponse(
            status_code=200,
            json_data={"message": f"GET response for {url}"},
            headers={"content-type": "application/json"}
        )

    def _mock_sync_post(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous POST request."""
        self._inject_sync_network_error("POST", url)

        self.operation_history.append({
            "operation": "POST",
            "url": url,
            "data": kwargs.get("data"),
            "json": kwargs.get("json")
        })

        return MockHTTPResponse(
            status_code=201,
            json_data={"message": f"POST response for {url}", "data": kwargs.get("json")},
            headers={"content-type": "application/json"}
        )

    def _mock_sync_put(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous PUT request."""
        self._inject_sync_network_error("PUT", url)

        self.operation_history.append({
            "operation": "PUT",
            "url": url,
            "data": kwargs.get("data"),
            "json": kwargs.get("json")
        })

        return MockHTTPResponse(
            status_code=200,
            json_data={"message": f"PUT response for {url}", "data": kwargs.get("json")},
            headers={"content-type": "application/json"}
        )

    def _mock_sync_delete(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous DELETE request."""
        self._inject_sync_network_error("DELETE", url)

        self.operation_history.append({
            "operation": "DELETE",
            "url": url,
            "kwargs": kwargs
        })

        return MockHTTPResponse(status_code=204)

    def _mock_sync_head(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous HEAD request."""
        self._inject_sync_network_error("HEAD", url)

        self.operation_history.append({
            "operation": "HEAD",
            "url": url,
            "kwargs": kwargs
        })

        return MockHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json", "content-length": "1024"}
        )

    def _mock_sync_request(self, method: str, url: str, **kwargs) -> MockHTTPResponse:
        """Mock synchronous generic request."""
        self._inject_sync_network_error(method, url)

        self.operation_history.append({
            "operation": method.upper(),
            "url": url,
            "kwargs": kwargs
        })

        return MockHTTPResponse(
            status_code=200,
            json_data={"message": f"{method} response for {url}"},
            headers={"content-type": "application/json"}
        )

    def reset_state(self) -> None:
        """Reset mock state."""
        self.operation_history.clear()
        self.error_injector.reset()


class ConnectionFailureMock:
    """Specialized mock for connection failure scenarios."""

    def __init__(self):
        self.failure_scenarios = {
            "total_network_failure": {"all_requests_fail": True},
            "intermittent_failures": {"failure_rate": 0.3},
            "dns_issues": {"dns_fails": True},
            "ssl_problems": {"ssl_fails": True},
            "server_overload": {"server_errors": True},
        }
        self.active_scenario = None

    def activate_scenario(self, scenario_name: str) -> None:
        """Activate a specific failure scenario."""
        if scenario_name in self.failure_scenarios:
            self.active_scenario = scenario_name

    def deactivate_scenario(self) -> None:
        """Deactivate current failure scenario."""
        self.active_scenario = None

    def should_fail_request(self, url: str, method: str) -> bool:
        """Determine if a request should fail based on active scenario."""
        if not self.active_scenario:
            return False

        scenario = self.failure_scenarios[self.active_scenario]

        if scenario.get("all_requests_fail"):
            return True
        elif "failure_rate" in scenario:
            return random.random() < scenario["failure_rate"]
        elif scenario.get("dns_fails") and "://" in url:
            return True
        elif scenario.get("ssl_fails") and url.startswith("https://"):
            return True
        elif scenario.get("server_errors"):
            return random.random() < 0.5

        return False


def create_network_mock(
    mock_type: str = "async",
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> Union[NetworkClientMock, HTTPRequestMock]:
    """
    Create a network mock with optional error injection.

    Args:
        mock_type: Type of mock ("async" or "sync")
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured network mock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = NetworkErrorInjector()
        error_injector.configure_connection_issues(error_probability)
        error_injector.configure_server_issues(error_probability)
        error_injector.configure_auth_issues(error_probability / 2)

    if mock_type == "async":
        return NetworkClientMock(error_injector)
    elif mock_type == "sync":
        return HTTPRequestMock(error_injector)
    else:
        raise ValueError(f"Unknown mock type: {mock_type}")


# Convenience functions for common scenarios
def create_basic_network_client() -> NetworkClientMock:
    """Create basic async network client mock."""
    return create_network_mock("async")


def create_failing_network_client(error_rate: float = 0.5) -> NetworkClientMock:
    """Create network client mock with high failure rate."""
    return create_network_mock("async", with_error_injection=True, error_probability=error_rate)


def create_realistic_network_client() -> NetworkClientMock:
    """Create network client mock with realistic error rates."""
    return create_network_mock("async", with_error_injection=True, error_probability=0.02)