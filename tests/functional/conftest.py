"""
Functional Testing Configuration

This module provides shared fixtures and configuration for functional tests,
including test containers, mock services, and performance monitoring.
"""

import asyncio
import os
import shutil
import tempfile

# Performance and monitoring
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest
import pytest_asyncio
import respx

# Playwright for web UI testing
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# Test containers and service mocking
from testcontainers import compose
from testcontainers.qdrant import QdrantContainer


@dataclass
class TestMetrics:
    """Container for test performance metrics."""
    test_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_requests: int = 0
    assertions_count: int = 0

    @property
    def duration_ms(self) -> float:
        """Calculate test duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    def finalize(self):
        """Finalize metrics collection."""
        self.end_time = time.time()
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = process.cpu_percent()


class TestEnvironment:
    """Manages the test environment setup and teardown."""

    def __init__(self):
        self.temp_dir = None
        self.qdrant_container = None
        self.test_metrics = {}
        self.cleanup_tasks = []

    async def setup(self):
        """Set up the test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="wqm_functional_test_")

        # Set environment variables for testing
        os.environ["WQM_TEST_MODE"] = "true"
        os.environ["WQM_DATA_DIR"] = self.temp_dir
        os.environ["QDRANT_URL"] = "http://localhost:6333"

    async def teardown(self):
        """Clean up the test environment."""
        # Run cleanup tasks
        for task in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
            except Exception as e:
                print(f"Warning: Cleanup task failed: {e}")

        # Stop containers
        if self.qdrant_container:
            self.qdrant_container.stop()

        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clean up environment
        for key in ["WQM_TEST_MODE", "WQM_DATA_DIR", "QDRANT_URL"]:
            os.environ.pop(key, None)

    def add_cleanup_task(self, task):
        """Add a cleanup task to be executed during teardown."""
        self.cleanup_tasks.append(task)


# Global test environment instance
_test_environment = TestEnvironment()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def test_environment():
    """Session-wide test environment setup."""
    await _test_environment.setup()
    yield _test_environment
    await _test_environment.teardown()


@pytest.fixture
def temp_directory(test_environment):
    """Provide a temporary directory for test files."""
    test_dir = os.path.join(test_environment.temp_dir, f"test_{os.getpid()}")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup handled by test_environment teardown


@pytest_asyncio.fixture
async def qdrant_container():
    """Provide a Qdrant container for integration testing."""
    if _test_environment.qdrant_container is None:
        container = QdrantContainer("qdrant/qdrant:latest")
        container.start()

        # Wait for Qdrant to be ready
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        qdrant_url = f"http://{host}:{port}"

        # Health check with retry
        async with httpx.AsyncClient() as client:
            for _ in range(30):  # 30 second timeout
                try:
                    response = await client.get(f"{qdrant_url}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Qdrant container failed to start")

        # Update environment variables
        os.environ["QDRANT_URL"] = qdrant_url
        _test_environment.qdrant_container = container

    yield _test_environment.qdrant_container


@pytest_asyncio.fixture
async def mock_http_services():
    """Provide mock HTTP services for testing."""
    with respx.mock(base_url="http://localhost:8000") as respx_mock:
        # Mock health endpoint
        respx_mock.get("/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        # Mock MCP endpoints
        respx_mock.post("/mcp/tools/list").mock(
            return_value=httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"tools": []}
            })
        )

        respx_mock.post("/mcp/tools/call").mock(
            return_value=httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": [{"type": "text", "text": "Success"}]}
            })
        )

        # Mock document endpoints
        respx_mock.post("/api/documents").mock(
            return_value=httpx.Response(201, json={"id": "doc_123", "status": "created"})
        )

        respx_mock.get("/api/documents/search").mock(
            return_value=httpx.Response(200, json={
                "results": [],
                "total": 0,
                "query_time_ms": 10
            })
        )

        yield respx_mock


@pytest_asyncio.fixture
async def playwright_browser():
    """Provide a Playwright browser instance."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions"
        ]
    )
    yield browser
    await browser.close()
    await playwright.stop()


@pytest_asyncio.fixture
async def browser_context(playwright_browser: Browser):
    """Provide a browser context with testing configuration."""
    context = await playwright_browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent="WQM-FunctionalTest/1.0",
        ignore_https_errors=True,
        # Record video for debugging failures
        record_video_dir="test-results/videos/" if os.getenv("RECORD_VIDEO") else None
    )
    yield context
    await context.close()


@pytest_asyncio.fixture
async def page(browser_context: BrowserContext):
    """Provide a page with error handling and monitoring."""
    page = await browser_context.new_page()

    # Set up error monitoring
    errors = []

    def handle_page_error(error):
        errors.append(str(error))

    def handle_console_message(msg):
        if msg.type == "error":
            errors.append(f"Console error: {msg.text}")

    page.on("pageerror", handle_page_error)
    page.on("console", handle_console_message)

    yield page

    # Report any errors that occurred
    if errors:
        print(f"Page errors during test: {errors}")

    await page.close()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring for tests."""
    metrics = {}

    @asynccontextmanager
    async def monitor_test(test_name: str):
        test_metrics = TestMetrics(test_name)
        metrics[test_name] = test_metrics

        try:
            yield test_metrics
        finally:
            test_metrics.finalize()

    return monitor_test


@pytest_asyncio.fixture
async def http_client():
    """Provide an HTTP client for API testing."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "content": "This is a sample document about artificial intelligence and machine learning.",
            "metadata": {
                "title": "AI Introduction",
                "author": "Test Author",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["ai", "ml", "technology"]
            }
        },
        {
            "id": "doc_2",
            "content": "Vector databases provide efficient similarity search capabilities for embeddings.",
            "metadata": {
                "title": "Vector Databases",
                "author": "Test Author",
                "created_at": "2024-01-02T00:00:00Z",
                "tags": ["database", "vectors", "search"]
            }
        },
        {
            "id": "doc_3",
            "content": "Hybrid search combines dense and sparse retrieval methods for better results.",
            "metadata": {
                "title": "Hybrid Search",
                "author": "Test Author",
                "created_at": "2024-01-03T00:00:00Z",
                "tags": ["search", "hybrid", "retrieval"]
            }
        }
    ]


@pytest_asyncio.fixture
async def loaded_documents(qdrant_container, sample_documents):
    """Provide pre-loaded documents in Qdrant for testing."""
    # This would be implemented to actually load documents into Qdrant
    # For now, it's a placeholder that returns the sample documents
    yield sample_documents


@pytest.fixture(autouse=True)
def test_metrics_collection(request, performance_monitor):
    """Automatically collect metrics for each test."""
    test_name = f"{request.module.__name__}::{request.function.__name__}"

    async def run_with_metrics():
        async with performance_monitor(test_name) as metrics:
            yield metrics

    # This fixture runs automatically but doesn't interfere with test execution
    yield


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest for functional testing."""
    # Add custom markers
    config.addinivalue_line("markers", "functional: Functional test suite")
    config.addinivalue_line("markers", "integration: Integration test suite")
    config.addinivalue_line("markers", "performance: Performance test suite")
    config.addinivalue_line("markers", "web_ui: Web UI test suite")
    config.addinivalue_line("markers", "mcp_protocol: MCP protocol test suite")
    config.addinivalue_line("markers", "slow: Slow running tests")

    # Create test results directory
    os.makedirs("test-results", exist_ok=True)
    os.makedirs("test-results/videos", exist_ok=True)
    os.makedirs("test-results/screenshots", exist_ok=True)


def pytest_runtest_makereport(item, call):
    """Generate test reports with additional information."""
    if call.when == "call":
        # Collect additional test information
        test_name = f"{item.module.__name__}::{item.function.__name__}"

        # Add performance metrics to test report if available
        if hasattr(_test_environment, 'test_metrics') and test_name in _test_environment.test_metrics:
            metrics = _test_environment.test_metrics[test_name]
            item.user_properties.append(("duration_ms", metrics.duration_ms))
            item.user_properties.append(("memory_usage_mb", metrics.memory_usage_mb))
            item.user_properties.append(("cpu_usage_percent", metrics.cpu_usage_percent))


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    # Generate performance report
    if hasattr(_test_environment, 'test_metrics') and _test_environment.test_metrics:
        report_path = "test-results/performance_report.json"
        with open(report_path, 'w') as f:
            import json
            metrics_data = {
                name: {
                    "duration_ms": metrics.duration_ms,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "cpu_usage_percent": metrics.cpu_usage_percent
                }
                for name, metrics in _test_environment.test_metrics.items()
            }
            json.dump(metrics_data, f, indent=2)

        print(f"Performance report generated: {report_path}")
