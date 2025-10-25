"""
Enhanced Qdrant client mocking with comprehensive error scenarios.

Provides sophisticated mocking capabilities for Qdrant operations including
search, insertion, deletion, collection management, and failure simulation.
"""

import asyncio
import random
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .error_injection import ErrorInjector, FailureScenarios


class QdrantErrorInjector(ErrorInjector):
    """Specialized error injector for Qdrant operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "connection_timeout": {"probability": 0.0, "delay": 30.0},
            "service_unavailable": {"probability": 0.0, "status_code": 503},
            "rate_limit": {"probability": 0.0, "status_code": 429},
            "invalid_collection": {"probability": 0.0, "error": "Collection not found"},
            "vector_dimension_mismatch": {"probability": 0.0, "error": "Vector dimension mismatch"},
            "quota_exceeded": {"probability": 0.0, "error": "Storage quota exceeded"},
            "authentication_failure": {"probability": 0.0, "status_code": 401},
            "network_partition": {"probability": 0.0, "error": "Network unreachable"},
        }

    def configure_connection_issues(self, probability: float = 0.1):
        """Configure connection-related failures."""
        self.failure_modes["connection_timeout"]["probability"] = probability
        self.failure_modes["network_partition"]["probability"] = probability / 2

    def configure_service_issues(self, probability: float = 0.1):
        """Configure service-level failures."""
        self.failure_modes["service_unavailable"]["probability"] = probability
        self.failure_modes["rate_limit"]["probability"] = probability / 2

    def configure_data_issues(self, probability: float = 0.1):
        """Configure data-related failures."""
        self.failure_modes["invalid_collection"]["probability"] = probability
        self.failure_modes["vector_dimension_mismatch"]["probability"] = probability / 2
        self.failure_modes["quota_exceeded"]["probability"] = probability / 3


class EnhancedQdrantClientMock:
    """Enhanced mock Qdrant client with realistic behavior and error injection."""

    def __init__(self, error_injector: QdrantErrorInjector | None = None):
        self.error_injector = error_injector or QdrantErrorInjector()
        self.collections = {}
        self.points = {}
        self.operation_history = []
        self.performance_delays = {
            "search": 0.1,
            "upsert": 0.05,
            "delete": 0.03,
            "collection_ops": 0.02,
        }

        # Initialize mocked methods
        self._setup_search_methods()
        self._setup_crud_methods()
        self._setup_collection_methods()
        self._setup_admin_methods()

    def _setup_search_methods(self):
        """Setup search-related methods."""
        self.search = AsyncMock(side_effect=self._mock_search)
        self.search_batch = AsyncMock(side_effect=self._mock_search_batch)
        self.recommend = AsyncMock(side_effect=self._mock_recommend)
        self.discover = AsyncMock(side_effect=self._mock_discover)

    def _setup_crud_methods(self):
        """Setup CRUD operation methods."""
        self.upsert = AsyncMock(side_effect=self._mock_upsert)
        self.delete = AsyncMock(side_effect=self._mock_delete)
        self.retrieve = AsyncMock(side_effect=self._mock_retrieve)
        self.scroll = AsyncMock(side_effect=self._mock_scroll)
        self.count = AsyncMock(side_effect=self._mock_count)

    def _setup_collection_methods(self):
        """Setup collection management methods."""
        self.create_collection = AsyncMock(side_effect=self._mock_create_collection)
        self.delete_collection = AsyncMock(side_effect=self._mock_delete_collection)
        self.get_collection = AsyncMock(side_effect=self._mock_get_collection)
        self.get_collections = AsyncMock(side_effect=self._mock_get_collections)
        self.collection_exists = AsyncMock(side_effect=self._mock_collection_exists)
        self.update_collection = AsyncMock(side_effect=self._mock_update_collection)

    def _setup_admin_methods(self):
        """Setup administrative methods."""
        self.info = AsyncMock(side_effect=self._mock_info)
        self.health = AsyncMock(side_effect=self._mock_health)
        self.close = Mock(side_effect=self._mock_close)

    async def _inject_errors_and_delay(self, operation: str) -> None:
        """Inject configured errors and realistic delays."""
        # Add realistic operation delay
        if operation in self.performance_delays:
            await asyncio.sleep(self.performance_delays[operation])

        # Check for error injection
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_injected_error(error_type)

    async def _raise_injected_error(self, error_type: str) -> None:
        """Raise the appropriate error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})

        if error_type == "connection_timeout":
            await asyncio.sleep(error_config.get("delay", 30.0))
            raise ConnectionError("Connection timeout")

        elif error_type == "service_unavailable":
            raise ResponseHandlingException("Service unavailable", status_code=503)

        elif error_type == "rate_limit":
            raise ResponseHandlingException("Rate limit exceeded", status_code=429)

        elif error_type == "invalid_collection":
            raise ValueError("Collection not found")

        elif error_type == "vector_dimension_mismatch":
            raise ValueError("Vector dimension mismatch")

        elif error_type == "quota_exceeded":
            raise ResponseHandlingException("Storage quota exceeded", status_code=507)

        elif error_type == "authentication_failure":
            raise ResponseHandlingException("Authentication failed", status_code=401)

        elif error_type == "network_partition":
            raise ConnectionError("Network unreachable")

    async def _mock_search(self, collection_name: str, query_vector: list[float] | models.NamedVector, **kwargs) -> list[models.ScoredPoint]:
        """Mock search operation with realistic behavior."""
        await self._inject_errors_and_delay("search")

        self.operation_history.append({
            "operation": "search",
            "collection": collection_name,
            "vector_size": len(query_vector) if isinstance(query_vector, list) else None,
            "limit": kwargs.get("limit", 10)
        })

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        # Generate realistic search results
        limit = kwargs.get("limit", 10)
        results = []

        for i in range(min(limit, 5)):  # Simulate finding some results
            results.append(models.ScoredPoint(
                id=f"doc_{i}",
                score=0.9 - (i * 0.1) + random.uniform(-0.05, 0.05),
                payload={
                    "content": f"Mock document {i} content",
                    "source": f"test_source_{i}",
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                vector=query_vector if isinstance(query_vector, list) else [0.1] * 384
            ))

        return results

    async def _mock_search_batch(self, collection_name: str, requests: list[models.SearchRequest]) -> list[list[models.ScoredPoint]]:
        """Mock batch search operation."""
        await self._inject_errors_and_delay("search")

        results = []
        for request in requests:
            batch_results = await self._mock_search(collection_name, request.vector, limit=request.limit)
            results.append(batch_results)

        return results

    async def _mock_recommend(self, collection_name: str, **kwargs) -> list[models.ScoredPoint]:
        """Mock recommendation operation."""
        await self._inject_errors_and_delay("search")

        # Simulate recommendation results
        return [
            models.ScoredPoint(
                id="rec_1",
                score=0.85,
                payload={"content": "Recommended content", "type": "recommendation"},
                vector=[0.1] * 384
            )
        ]

    async def _mock_discover(self, collection_name: str, **kwargs) -> list[models.ScoredPoint]:
        """Mock discovery operation."""
        await self._inject_errors_and_delay("search")

        # Simulate discovery results
        return [
            models.ScoredPoint(
                id="disc_1",
                score=0.80,
                payload={"content": "Discovered content", "type": "discovery"},
                vector=[0.1] * 384
            )
        ]

    async def _mock_upsert(self, collection_name: str, points: list[models.PointStruct] | models.Batch, **kwargs) -> models.UpdateResult:
        """Mock upsert operation."""
        await self._inject_errors_and_delay("upsert")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        if isinstance(points, list):
            point_count = len(points)
        else:
            point_count = len(points.ids) if hasattr(points, 'ids') else 1

        self.operation_history.append({
            "operation": "upsert",
            "collection": collection_name,
            "point_count": point_count
        })

        return models.UpdateResult(
            operation_id=random.randint(1000, 9999),
            status=models.UpdateStatus.COMPLETED
        )

    async def _mock_delete(self, collection_name: str, points_selector: models.PointsSelector, **kwargs) -> models.UpdateResult:
        """Mock delete operation."""
        await self._inject_errors_and_delay("delete")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        self.operation_history.append({
            "operation": "delete",
            "collection": collection_name,
            "selector": str(points_selector)
        })

        return models.UpdateResult(
            operation_id=random.randint(1000, 9999),
            status=models.UpdateStatus.COMPLETED
        )

    async def _mock_retrieve(self, collection_name: str, ids: list[str | int], **kwargs) -> list[models.Record]:
        """Mock retrieve operation."""
        await self._inject_errors_and_delay("search")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        records = []
        for point_id in ids[:5]:  # Simulate finding some records
            records.append(models.Record(
                id=point_id,
                payload={
                    "content": f"Retrieved content for {point_id}",
                    "id": point_id
                },
                vector=[0.1] * 384
            ))

        return records

    async def _mock_scroll(self, collection_name: str, **kwargs) -> tuple[list[models.Record], str | None]:
        """Mock scroll operation."""
        await self._inject_errors_and_delay("search")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        limit = kwargs.get("limit", 10)
        records = []

        for i in range(min(limit, 5)):
            records.append(models.Record(
                id=f"scroll_{i}",
                payload={
                    "content": f"Scrolled content {i}",
                    "index": i
                },
                vector=[0.1] * 384
            ))

        next_page_offset = f"offset_{random.randint(100, 999)}" if records else None
        return records, next_page_offset

    async def _mock_count(self, collection_name: str, **kwargs) -> models.CountResult:
        """Mock count operation."""
        await self._inject_errors_and_delay("search")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        # Simulate realistic count
        count = random.randint(100, 1000)
        return models.CountResult(count=count)

    async def _mock_create_collection(self, collection_name: str, vectors_config: models.VectorParams | dict[str, models.VectorParams], **kwargs) -> bool:
        """Mock collection creation."""
        await self._inject_errors_and_delay("collection_ops")

        if collection_name in self.collections:
            raise ValueError(f"Collection '{collection_name}' already exists")

        self.collections[collection_name] = {
            "name": collection_name,
            "vectors_config": vectors_config,
            "created_at": "2024-01-01T12:00:00Z",
            "status": "green",
            "points_count": 0
        }

        self.operation_history.append({
            "operation": "create_collection",
            "collection": collection_name
        })

        return True

    async def _mock_delete_collection(self, collection_name: str, **kwargs) -> bool:
        """Mock collection deletion."""
        await self._inject_errors_and_delay("collection_ops")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        del self.collections[collection_name]

        self.operation_history.append({
            "operation": "delete_collection",
            "collection": collection_name
        })

        return True

    async def _mock_get_collection(self, collection_name: str) -> models.CollectionInfo:
        """Mock get collection info."""
        await self._inject_errors_and_delay("collection_ops")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        collection_data = self.collections[collection_name]

        return models.CollectionInfo(
            status=models.CollectionStatus.GREEN,
            optimizer_status=models.OptimizersStatusOneOf.ok,
            vectors_count=collection_data.get("points_count", 0),
            indexed_vectors_count=collection_data.get("points_count", 0),
            points_count=collection_data.get("points_count", 0),
            segments_count=1,
            config=models.CollectionConfig(
                params=models.CollectionParams(
                    vectors=collection_data["vectors_config"]
                )
            ),
            payload_schema={}
        )

    async def _mock_get_collections(self) -> models.CollectionsResponse:
        """Mock get collections list."""
        await self._inject_errors_and_delay("collection_ops")

        collections = []
        for name, _data in self.collections.items():
            collections.append(models.CollectionDescription(
                name=name
            ))

        return models.CollectionsResponse(collections=collections)

    async def _mock_collection_exists(self, collection_name: str) -> bool:
        """Mock collection existence check."""
        await self._inject_errors_and_delay("collection_ops")
        return collection_name in self.collections

    async def _mock_update_collection(self, collection_name: str, **kwargs) -> bool:
        """Mock collection update."""
        await self._inject_errors_and_delay("collection_ops")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        # Update collection data
        self.collections[collection_name].update(kwargs)

        self.operation_history.append({
            "operation": "update_collection",
            "collection": collection_name,
            "updates": list(kwargs.keys())
        })

        return True

    async def _mock_info(self) -> dict[str, Any]:
        """Mock server info."""
        return {
            "title": "qdrant",
            "version": "1.7.4",
            "description": "Mock Qdrant instance for testing"
        }

    async def _mock_health(self) -> dict[str, Any]:
        """Mock health check."""
        if self.error_injector.should_inject_error():
            raise ConnectionError("Health check failed")

        return {
            "title": "qdrant",
            "version": "1.7.4",
            "status": "ok"
        }

    def _mock_close(self) -> None:
        """Mock client close."""
        self.operation_history.append({"operation": "close"})

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Get history of operations performed."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset mock state."""
        self.collections.clear()
        self.points.clear()
        self.operation_history.clear()
        self.error_injector.reset()

    def configure_performance(self, **delays) -> None:
        """Configure performance delays for operations."""
        self.performance_delays.update(delays)


def create_enhanced_qdrant_client(
    with_error_injection: bool = False,
    error_probability: float = 0.1,
    collections: list[str] | None = None
) -> EnhancedQdrantClientMock:
    """
    Create an enhanced Qdrant client mock with optional error injection.

    Args:
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)
        collections: Pre-create collections

    Returns:
        Configured EnhancedQdrantClientMock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = QdrantErrorInjector()
        error_injector.configure_connection_issues(error_probability)
        error_injector.configure_service_issues(error_probability)
        error_injector.configure_data_issues(error_probability)

    client = EnhancedQdrantClientMock(error_injector)

    # Pre-create collections if specified
    if collections:
        for collection_name in collections:
            client.collections[collection_name] = {
                "name": collection_name,
                "vectors_config": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "created_at": "2024-01-01T12:00:00Z",
                "status": "green",
                "points_count": random.randint(10, 100)
            }

    return client


# Commonly used fixture configurations
def create_basic_qdrant_mock() -> EnhancedQdrantClientMock:
    """Create basic Qdrant mock without error injection."""
    return create_enhanced_qdrant_client()


def create_failing_qdrant_mock(error_rate: float = 0.5) -> EnhancedQdrantClientMock:
    """Create Qdrant mock with high failure rate for testing error handling."""
    return create_enhanced_qdrant_client(
        with_error_injection=True,
        error_probability=error_rate
    )


def create_realistic_qdrant_mock() -> EnhancedQdrantClientMock:
    """Create Qdrant mock with realistic behavior and occasional errors."""
    return create_enhanced_qdrant_client(
        with_error_injection=True,
        error_probability=0.05,  # 5% error rate
        collections=["test-project_docs", "test-project_scratchbook", "memory"]
    )
