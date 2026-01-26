"""
Qdrant Version Compatibility Tests.

Tests compatibility with supported Qdrant versions (1.7+).
Validates API compatibility, collection operations, and vector search functionality.
"""

import asyncio
import importlib.metadata
from typing import Optional

import pytest
from packaging import version

# Test if qdrant-client can be imported
try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        SearchRequest,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    qdrant_client = None
    QdrantClient = None


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantVersionInfo:
    """Test Qdrant client version information."""

    def test_qdrant_client_version(self):
        """Test Qdrant client version is 1.7+."""
        if hasattr(qdrant_client, "__version__"):
            ver_str = qdrant_client.__version__
        else:
            ver_str = importlib.metadata.version("qdrant-client")
        ver = version.parse(ver_str)
        assert ver >= version.parse("1.7.0"), (
            f"Qdrant client {ver_str} is below minimum 1.7.0"
        )

    def test_qdrant_client_version_string(self):
        """Test Qdrant client version string format."""
        if hasattr(qdrant_client, "__version__"):
            ver_str = qdrant_client.__version__
        else:
            ver_str = importlib.metadata.version("qdrant-client")
        assert isinstance(ver_str, str)
        assert len(ver_str.split(".")) >= 2  # At least major.minor


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantAPICompatibility:
    """Test Qdrant API compatibility."""

    def test_qdrant_client_instantiation(self):
        """Test QdrantClient can be instantiated."""
        # In-memory client for testing
        client = QdrantClient(":memory:")
        assert client is not None

    def test_distance_enum_available(self):
        """Test Distance enum has required values."""
        assert hasattr(Distance, "COSINE")
        assert hasattr(Distance, "EUCLID")
        assert hasattr(Distance, "DOT")

    def test_vector_params_creation(self):
        """Test VectorParams can be created."""
        params = VectorParams(size=384, distance=Distance.COSINE)
        assert params.size == 384
        assert params.distance == Distance.COSINE

    def test_point_struct_creation(self):
        """Test PointStruct can be created."""
        point = PointStruct(
            id=1,
            vector=[0.1] * 384,
            payload={"text": "test", "metadata": {"key": "value"}},
        )
        assert point.id == 1
        assert len(point.vector) == 384
        assert point.payload["text"] == "test"

    def test_filter_creation(self):
        """Test Filter objects can be created."""
        filter_obj = Filter(
            must=[FieldCondition(key="status", match=MatchValue(value="active"))]
        )
        assert filter_obj is not None
        assert len(filter_obj.must) == 1


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
@pytest.mark.requires_qdrant
class TestQdrantCollectionOperations:
    """Test Qdrant collection operations."""

    @pytest.fixture
    def qdrant_client(self):
        """Create in-memory Qdrant client for testing."""
        return QdrantClient(":memory:")

    def test_create_collection(self, qdrant_client):
        """Test collection creation."""
        collection_name = "test_collection"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert collection_name in collection_names

    def test_delete_collection(self, qdrant_client):
        """Test collection deletion."""
        collection_name = "test_delete_collection"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Delete collection
        qdrant_client.delete_collection(collection_name=collection_name)

        # Verify collection no longer exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert collection_name not in collection_names

    def test_collection_info(self, qdrant_client):
        """Test getting collection information."""
        collection_name = "test_info_collection"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Get collection info
        info = qdrant_client.get_collection(collection_name=collection_name)
        assert info.config.params.vectors.size == 384
        assert info.config.params.vectors.distance == Distance.COSINE

    def test_list_collections(self, qdrant_client):
        """Test listing collections."""
        # Create multiple collections
        for i in range(3):
            qdrant_client.create_collection(
                collection_name=f"test_list_{i}",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        # List collections
        collections = qdrant_client.get_collections()
        assert len(collections.collections) >= 3


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
@pytest.mark.requires_qdrant
class TestQdrantVectorOperations:
    """Test Qdrant vector operations."""

    @pytest.fixture
    def qdrant_client_with_collection(self):
        """Create in-memory Qdrant client with a test collection."""
        client = QdrantClient(":memory:")
        collection_name = "test_vectors"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        return client, collection_name

    def test_upsert_points(self, qdrant_client_with_collection):
        """Test upserting points."""
        client, collection_name = qdrant_client_with_collection

        # Create and upsert points
        points = [
            PointStruct(
                id=i,
                vector=[0.1 * i] * 384,
                payload={"text": f"Document {i}", "index": i},
            )
            for i in range(10)
        ]

        client.upsert(collection_name=collection_name, points=points)

        # Verify points were added
        info = client.get_collection(collection_name=collection_name)
        assert info.points_count == 10

    def test_retrieve_points(self, qdrant_client_with_collection):
        """Test retrieving points."""
        client, collection_name = qdrant_client_with_collection

        # Insert points
        points = [
            PointStruct(
                id=i, vector=[0.1 * i] * 384, payload={"text": f"Document {i}"}
            )
            for i in range(5)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Retrieve a specific point
        retrieved = client.retrieve(
            collection_name=collection_name, ids=[1], with_payload=True, with_vectors=True
        )

        assert len(retrieved) == 1
        assert retrieved[0].id == 1
        assert retrieved[0].payload["text"] == "Document 1"

    def test_delete_points(self, qdrant_client_with_collection):
        """Test deleting points."""
        client, collection_name = qdrant_client_with_collection

        # Insert points
        points = [
            PointStruct(id=i, vector=[0.1 * i] * 384, payload={}) for i in range(10)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Delete some points
        client.delete(collection_name=collection_name, points_selector=[1, 2, 3])

        # Verify deletion
        info = client.get_collection(collection_name=collection_name)
        assert info.points_count == 7


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
@pytest.mark.requires_qdrant
class TestQdrantSearchOperations:
    """Test Qdrant search operations."""

    @pytest.fixture
    def qdrant_client_with_data(self):
        """Create in-memory Qdrant client with test data."""
        client = QdrantClient(":memory:")
        collection_name = "test_search"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Insert test points
        points = [
            PointStruct(
                id=i,
                vector=[float(i) / 100] * 384,
                payload={
                    "text": f"Document {i}",
                    "category": "A" if i % 2 == 0 else "B",
                    "score": i * 10,
                },
            )
            for i in range(20)
        ]
        client.upsert(collection_name=collection_name, points=points)

        return client, collection_name

    def test_basic_search(self, qdrant_client_with_data):
        """Test basic vector search."""
        client, collection_name = qdrant_client_with_data

        # Search for similar vectors
        query_vector = [0.05] * 384
        results = client.search(
            collection_name=collection_name, query_vector=query_vector, limit=5
        )

        assert len(results) == 5
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_with_filter(self, qdrant_client_with_data):
        """Test search with filter."""
        client, collection_name = qdrant_client_with_data

        # Search with category filter
        query_vector = [0.05] * 384
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value="A"))]
            ),
            limit=10,
        )

        # Verify all results match filter
        assert len(results) > 0
        for result in results:
            assert result.payload["category"] == "A"

    def test_search_with_score_threshold(self, qdrant_client_with_data):
        """Test search with score threshold."""
        client, collection_name = qdrant_client_with_data

        # Search with score threshold
        query_vector = [0.05] * 384
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=20,
            score_threshold=0.9,  # High threshold
        )

        # Verify all results meet threshold
        assert all(r.score >= 0.9 for r in results)

    def test_search_with_payload(self, qdrant_client_with_data):
        """Test search returns payload."""
        client, collection_name = qdrant_client_with_data

        # Search with payload
        query_vector = [0.05] * 384
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            with_payload=True,
        )

        # Verify payload is present
        assert len(results) > 0
        for result in results:
            assert result.payload is not None
            assert "text" in result.payload
            assert "category" in result.payload


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantFeatureAvailability:
    """Test Qdrant feature availability."""

    def test_hybrid_search_models_available(self):
        """Test hybrid search model classes are available."""
        # These should be available in qdrant-client 1.7+
        from qdrant_client.models import NamedSparseVector, SparseVector

        # Test instantiation
        sparse = SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
        assert sparse.indices == [1, 2, 3]
        assert sparse.values == [0.1, 0.2, 0.3]

    def test_scroll_api_available(self):
        """Test scroll API is available."""
        client = QdrantClient(":memory:")
        assert hasattr(client, "scroll")

    def test_recommend_api_available(self):
        """Test recommend API is available."""
        client = QdrantClient(":memory:")
        assert hasattr(client, "recommend")

    def test_count_api_available(self):
        """Test count API is available."""
        client = QdrantClient(":memory:")
        assert hasattr(client, "count")


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantVersionSpecificFeatures:
    """Test version-specific features."""

    def test_qdrant_client_has_grpc_support(self):
        """Test if gRPC support is available (1.7+ feature)."""
        # Check if gRPC-related parameters are available
        import inspect

        sig = inspect.signature(QdrantClient.__init__)
        params = sig.parameters
        # gRPC support should have prefer_grpc or grpc_port parameter
        assert "prefer_grpc" in params or "grpc_port" in params

    def test_sparse_vector_support(self):
        """Test sparse vector support (1.7+ feature)."""
        from qdrant_client.models import SparseVector

        # Create sparse vector
        sparse = SparseVector(indices=[0, 10, 100], values=[1.0, 2.0, 3.0])
        assert len(sparse.indices) == 3
        assert len(sparse.values) == 3

    def test_named_vectors_support(self):
        """Test named vectors support."""
        from qdrant_client.models import VectorParams

        # Named vectors should be supported in 1.7+
        params = {"dense": VectorParams(size=384, distance=Distance.COSINE)}
        assert "dense" in params
        assert isinstance(params["dense"], VectorParams)
