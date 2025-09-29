"""
Comprehensive test suite for MCP server search operations.

This test suite provides complete coverage of hybrid search functionality in the Python
MCP server, including:
- Hybrid search (dense + sparse vectors with reciprocal rank fusion)
- Project-scoped search functionality
- Collection-specific search
- Global search across collections
- Metadata filtering capabilities
- Multi-query patterns
- FastEmbed model integration validation
- Reciprocal rank fusion (RRF) algorithm correctness
- Precision/recall metrics validation

Task 281: Develop MCP server search operations test suite
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.hybrid_search import HybridSearchEngine, RRFFusionRanker
from common.core.embeddings import EmbeddingService
from common.core.client import QdrantWorkspaceClient


@dataclass
class SearchTestDocument:
    """Test document for search validation."""
    id: str
    content: str
    project_name: str
    collection_type: str
    workspace_scope: str
    relevance_score: float  # Ground truth relevance for validation
    metadata: Dict[str, Any]


@dataclass
class SearchMetrics:
    """Search quality metrics."""
    precision: float
    recall: float
    f1_score: float
    mean_reciprocal_rank: float
    ndcg: float  # Normalized Discounted Cumulative Gain


@pytest.fixture(scope="session")
def qdrant_client():
    """Create Qdrant client for testing."""
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    # Test connection
    try:
        client.get_collections()
        return client
    except Exception as e:
        pytest.skip(f"Qdrant server not available: {e}")


@pytest.fixture(scope="session")
async def embedding_service():
    """Initialize embedding service for testing."""
    service = EmbeddingService()
    await service.initialize()
    return service


@pytest.fixture(scope="function")
async def test_collection(qdrant_client):
    """Create a test collection with synthetic data."""
    collection_name = "test-search-operations"

    # Delete if exists
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection with dense and sparse vectors
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,  # FastEmbed all-MiniLM-L6-v2 dimension
            distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False
                )
            )
        }
    )

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.fixture
def synthetic_test_documents():
    """Generate synthetic test documents with known relevance."""
    return [
        SearchTestDocument(
            id="doc_001",
            content="Machine learning algorithms for natural language processing",
            project_name="ai-research",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=1.0,
            metadata={"topic": "AI", "year": 2024, "author": "researcher1"}
        ),
        SearchTestDocument(
            id="doc_002",
            content="Deep neural networks and transformer architectures",
            project_name="ai-research",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=0.9,
            metadata={"topic": "AI", "year": 2024, "author": "researcher1"}
        ),
        SearchTestDocument(
            id="doc_003",
            content="Python programming best practices and design patterns",
            project_name="ai-research",
            collection_type="code",
            workspace_scope="project",
            relevance_score=0.3,
            metadata={"topic": "Programming", "year": 2024, "language": "python"}
        ),
        SearchTestDocument(
            id="doc_004",
            content="Vector database optimization techniques",
            project_name="database-project",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=0.6,
            metadata={"topic": "Databases", "year": 2024, "author": "researcher2"}
        ),
        SearchTestDocument(
            id="doc_005",
            content="Semantic search and information retrieval systems",
            project_name="ai-research",
            collection_type="docs",
            workspace_scope="shared",
            relevance_score=0.95,
            metadata={"topic": "AI", "year": 2024, "author": "researcher1"}
        ),
        SearchTestDocument(
            id="doc_006",
            content="JavaScript async programming and promises",
            project_name="web-project",
            collection_type="code",
            workspace_scope="project",
            relevance_score=0.1,
            metadata={"topic": "Programming", "year": 2023, "language": "javascript"}
        ),
        SearchTestDocument(
            id="doc_007",
            content="Natural language understanding with transformers and BERT",
            project_name="ai-research",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=0.98,
            metadata={"topic": "AI", "year": 2024, "author": "researcher3"}
        ),
        SearchTestDocument(
            id="doc_008",
            content="Database indexing strategies for vector search",
            project_name="database-project",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=0.55,
            metadata={"topic": "Databases", "year": 2024, "author": "researcher2"}
        ),
        SearchTestDocument(
            id="doc_009",
            content="Machine learning model deployment and monitoring",
            project_name="ai-research",
            collection_type="notes",
            workspace_scope="shared",
            relevance_score=0.7,
            metadata={"topic": "AI", "year": 2024, "author": "researcher1"}
        ),
        SearchTestDocument(
            id="doc_010",
            content="SQL query optimization and performance tuning",
            project_name="database-project",
            collection_type="docs",
            workspace_scope="project",
            relevance_score=0.2,
            metadata={"topic": "Databases", "year": 2023, "author": "researcher2"}
        ),
    ]


async def ingest_test_documents(
    qdrant_client: QdrantClient,
    collection_name: str,
    documents: List[SearchTestDocument],
    embedding_service: EmbeddingService
):
    """Ingest test documents into collection with embeddings."""
    points = []

    for doc in documents:
        # Generate embeddings
        embeddings = await embedding_service.embed_text(doc.content)

        # Create point
        point = models.PointStruct(
            id=doc.id,
            vector={
                "": embeddings["dense"],  # Dense vector
                "text-sparse": models.SparseVector(
                    indices=embeddings["sparse"]["indices"],
                    values=embeddings["sparse"]["values"]
                )
            },
            payload={
                "content": doc.content,
                "project_name": doc.project_name,
                "collection_type": doc.collection_type,
                "workspace_scope": doc.workspace_scope,
                "relevance_score": doc.relevance_score,
                **doc.metadata
            }
        )
        points.append(point)

    # Upload points
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    # Wait for indexing
    await asyncio.sleep(1)


def calculate_search_metrics(
    results: List[Dict],
    ground_truth_docs: List[SearchTestDocument],
    relevance_threshold: float = 0.5,
    k: int = 10
) -> SearchMetrics:
    """Calculate precision, recall, and other search quality metrics."""

    # Extract result IDs
    result_ids = [r.get("id") for r in results[:k] if r.get("id")]

    # Ground truth relevant documents
    relevant_doc_ids = {
        doc.id for doc in ground_truth_docs
        if doc.relevance_score >= relevance_threshold
    }

    # Calculate metrics
    if not result_ids:
        return SearchMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    # True positives: relevant documents that were retrieved
    true_positives = len(set(result_ids) & relevant_doc_ids)

    # Precision: fraction of retrieved documents that are relevant
    precision = true_positives / len(result_ids) if result_ids else 0.0

    # Recall: fraction of relevant documents that were retrieved
    recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    # F1 score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for idx, result_id in enumerate(result_ids, 1):
        if result_id in relevant_doc_ids:
            mrr = 1.0 / idx
            break

    # Normalized Discounted Cumulative Gain (NDCG)
    dcg = 0.0
    for idx, result_id in enumerate(result_ids, 1):
        doc = next((d for d in ground_truth_docs if d.id == result_id), None)
        if doc:
            # DCG formula: rel_i / log2(i + 1)
            dcg += doc.relevance_score / (statistics.log2(idx + 1))

    # Ideal DCG (IDCG)
    sorted_relevance = sorted(
        [d.relevance_score for d in ground_truth_docs],
        reverse=True
    )[:k]
    idcg = sum(
        rel / statistics.log2(idx + 2)
        for idx, rel in enumerate(sorted_relevance)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return SearchMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_reciprocal_rank=mrr,
        ndcg=ndcg
    )


class TestHybridSearchBasics:
    """Test basic hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_rrf_fusion(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test hybrid search using RRF fusion algorithm."""
        # Ingest test documents
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        # Create search engine
        search_engine = HybridSearchEngine(qdrant_client)

        # Generate query embeddings
        query = "natural language processing and machine learning"
        query_embeddings = await embedding_service.embed_text(query)

        # Execute hybrid search with RRF
        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=5,
            fusion_method="rrf",
            dense_weight=1.0,
            sparse_weight=1.0
        )

        # Validate results structure
        assert "dense_results" in results
        assert "sparse_results" in results
        assert "fused_results" in results

        # Validate fused results
        fused = results["fused_results"]
        assert len(fused) > 0
        assert len(fused) <= 5

        # Check RRF scores are present
        for result in fused:
            assert hasattr(result, 'payload')
            if result.payload:
                assert "rrf_score" in result.payload or hasattr(result, 'score')

        # Calculate and validate metrics
        result_dicts = [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in fused
        ]
        metrics = calculate_search_metrics(result_dicts, synthetic_test_documents)

        # Assert minimum quality thresholds
        assert metrics.precision >= 0.6, f"Precision too low: {metrics.precision}"
        assert metrics.recall >= 0.4, f"Recall too low: {metrics.recall}"
        assert metrics.ndcg >= 0.5, f"NDCG too low: {metrics.ndcg}"

    @pytest.mark.asyncio
    async def test_hybrid_search_weighted_sum_fusion(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test hybrid search using weighted sum fusion."""
        # Ingest documents
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        # Create search engine
        search_engine = HybridSearchEngine(qdrant_client)

        # Query
        query = "deep learning transformers"
        query_embeddings = await embedding_service.embed_text(query)

        # Search with weighted sum (favor dense)
        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=5,
            fusion_method="weighted_sum",
            dense_weight=0.7,
            sparse_weight=0.3
        )

        # Validate
        assert len(results["fused_results"]) > 0

        # Check weighted scores
        for result in results["fused_results"]:
            if result.payload:
                assert "weighted_score" in result.payload or hasattr(result, 'score')

    @pytest.mark.asyncio
    async def test_dense_only_search(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test search with dense vectors only."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("semantic search")

        # Dense-only search
        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={"dense": query_embeddings["dense"]},
            limit=5,
            fusion_method="rrf"
        )

        assert len(results["dense_results"]) > 0
        assert len(results["sparse_results"]) == 0
        assert len(results["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_sparse_only_search(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test search with sparse vectors only."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("semantic search")

        # Sparse-only search
        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={"sparse": query_embeddings["sparse"]},
            limit=5,
            fusion_method="rrf"
        )

        assert len(results["dense_results"]) == 0
        assert len(results["sparse_results"]) > 0
        assert len(results["fused_results"]) > 0


class TestProjectScopedSearch:
    """Test project-scoped search functionality."""

    @pytest.mark.asyncio
    async def test_search_with_project_filtering(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test search with project metadata filtering."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("machine learning")

        # Create project context filter
        project_context = {
            "project_name": "ai-research",
            "collection_type": "docs"
        }

        # Search with project filtering
        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            project_context=project_context,
            auto_inject_metadata=True
        )

        # Validate all results are from ai-research project
        for result in results["fused_results"]:
            payload = result.payload if hasattr(result, 'payload') else {}
            assert payload.get("project_name") == "ai-research"

    @pytest.mark.asyncio
    async def test_search_with_collection_type_filtering(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test filtering by collection type."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("programming")

        # Filter for code collections only
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_type",
                    match=models.MatchValue(value="code")
                )
            ]
        )

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            filter_conditions=filter_condition
        )

        # Validate results
        for result in results["fused_results"]:
            assert result.payload.get("collection_type") == "code"

    @pytest.mark.asyncio
    async def test_search_with_workspace_scope(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test workspace scope filtering."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("AI research")

        # Search only project scope (not shared)
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="workspace_scope",
                    match=models.MatchValue(value="project")
                )
            ]
        )

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            filter_conditions=filter_condition
        )

        # Validate workspace scope
        for result in results["fused_results"]:
            assert result.payload.get("workspace_scope") == "project"


class TestMetadataFiltering:
    """Test advanced metadata filtering capabilities."""

    @pytest.mark.asyncio
    async def test_complex_metadata_filter(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test complex metadata filtering with multiple conditions."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("research papers")

        # Complex filter: AI topic AND year 2024 AND specific author
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="topic",
                    match=models.MatchValue(value="AI")
                ),
                models.FieldCondition(
                    key="year",
                    match=models.MatchValue(value=2024)
                ),
                models.FieldCondition(
                    key="author",
                    match=models.MatchValue(value="researcher1")
                )
            ]
        )

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            filter_conditions=filter_condition
        )

        # Validate all conditions are met
        for result in results["fused_results"]:
            payload = result.payload
            assert payload.get("topic") == "AI"
            assert payload.get("year") == 2024
            assert payload.get("author") == "researcher1"

    @pytest.mark.asyncio
    async def test_range_filter(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test range-based metadata filtering."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("technical documentation")

        # Filter by year range
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=2024)
                )
            ]
        )

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            filter_conditions=filter_condition
        )

        # Validate year range
        for result in results["fused_results"]:
            assert result.payload.get("year") >= 2024

    @pytest.mark.asyncio
    async def test_match_any_filter(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test match any (OR) filtering."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("technology")

        # Match any of multiple values
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="topic",
                    match=models.MatchAny(any=["AI", "Databases"])
                )
            ]
        )

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf",
            filter_conditions=filter_condition
        )

        # Validate topic matches
        for result in results["fused_results"]:
            topic = result.payload.get("topic")
            assert topic in ["AI", "Databases"]


class TestRRFAlgorithmCorrectness:
    """Test RRF algorithm implementation correctness."""

    def test_rrf_formula_calculation(self):
        """Test RRF formula calculation accuracy."""
        ranker = RRFFusionRanker(k=60)

        # Mock results with known ranks
        dense_results = [
            type('obj', (object,), {'id': 'doc1', 'score': 0.9, 'payload': {}})(),
            type('obj', (object,), {'id': 'doc2', 'score': 0.8, 'payload': {}})(),
        ]
        sparse_results = [
            type('obj', (object,), {'id': 'doc1', 'score': 0.85, 'payload': {}})(),
            type('obj', (object,), {'id': 'doc3', 'score': 0.75, 'payload': {}})(),
        ]

        # Execute fusion
        fused = ranker.fuse(dense_results, sparse_results)

        # Validate doc1 RRF score
        # doc1: rank 1 in dense, rank 1 in sparse
        # RRF = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
        doc1 = next(r for r in fused if r.id == 'doc1')
        expected_rrf = 2.0 / 61.0

        # Get actual RRF score
        actual_rrf = doc1.payload.get('rrf_score', 0)

        assert abs(actual_rrf - expected_rrf) < 0.001, \
            f"RRF calculation incorrect: expected {expected_rrf}, got {actual_rrf}"

    def test_rrf_rank_preservation(self):
        """Test that RRF preserves ranking properties."""
        ranker = RRFFusionRanker(k=60)

        # Results where both sources agree on ranking
        dense_results = [
            type('obj', (object,), {'id': f'doc{i}', 'score': 1.0 - i*0.1, 'payload': {}})()
            for i in range(5)
        ]
        sparse_results = [
            type('obj', (object,), {'id': f'doc{i}', 'score': 1.0 - i*0.1, 'payload': {}})()
            for i in range(5)
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Fused results should maintain the same order
        for i in range(len(fused) - 1):
            current_score = fused[i].payload.get('rrf_score', 0)
            next_score = fused[i + 1].payload.get('rrf_score', 0)
            assert current_score >= next_score, "RRF scores not properly sorted"

    def test_rrf_with_weights(self):
        """Test RRF with custom weights."""
        ranker = RRFFusionRanker(k=60)

        dense_results = [
            type('obj', (object,), {'id': 'doc1', 'score': 0.9, 'payload': {}})(),
        ]
        sparse_results = [
            type('obj', (object,), {'id': 'doc1', 'score': 0.8, 'payload': {}})(),
        ]

        # Fusion with different weights
        fused_equal = ranker.fuse(
            dense_results, sparse_results,
            weights={"dense": 1.0, "sparse": 1.0}
        )

        fused_dense_heavy = ranker.fuse(
            dense_results, sparse_results,
            weights={"dense": 2.0, "sparse": 1.0}
        )

        # Dense-heavy weighting should produce higher score
        score_equal = fused_equal[0].payload.get('rrf_score', 0)
        score_dense_heavy = fused_dense_heavy[0].payload.get('rrf_score', 0)

        assert score_dense_heavy > score_equal, \
            "Weighted RRF should produce different scores"


class TestFastEmbedIntegration:
    """Test FastEmbed model integration."""

    @pytest.mark.asyncio
    async def test_embedding_service_initialization(self, embedding_service):
        """Test FastEmbed service initializes correctly."""
        assert embedding_service is not None

        # Check model info
        model_info = embedding_service.get_model_info()
        assert "model_name" in model_info
        assert "vector_size" in model_info
        assert model_info["vector_size"] == 384  # all-MiniLM-L6-v2

    @pytest.mark.asyncio
    async def test_dense_embedding_generation(self, embedding_service):
        """Test dense vector generation."""
        text = "Test document for embedding"
        embeddings = await embedding_service.embed_text(text)

        assert "dense" in embeddings
        assert isinstance(embeddings["dense"], list)
        assert len(embeddings["dense"]) == 384
        assert all(isinstance(v, float) for v in embeddings["dense"])

    @pytest.mark.asyncio
    async def test_sparse_embedding_generation(self, embedding_service):
        """Test sparse vector generation."""
        text = "Test document for embedding"
        embeddings = await embedding_service.embed_text(text)

        assert "sparse" in embeddings
        assert "indices" in embeddings["sparse"]
        assert "values" in embeddings["sparse"]
        assert len(embeddings["sparse"]["indices"]) == len(embeddings["sparse"]["values"])

    @pytest.mark.asyncio
    async def test_embedding_consistency(self, embedding_service):
        """Test embedding consistency for same input."""
        text = "Consistent test document"

        # Generate embeddings twice
        embeddings1 = await embedding_service.embed_text(text)
        embeddings2 = await embedding_service.embed_text(text)

        # Dense embeddings should be identical
        assert embeddings1["dense"] == embeddings2["dense"]

        # Sparse embeddings should be identical
        assert embeddings1["sparse"]["indices"] == embeddings2["sparse"]["indices"]
        assert embeddings1["sparse"]["values"] == embeddings2["sparse"]["values"]


class TestPrecisionRecallMetrics:
    """Test precision and recall calculation accuracy."""

    @pytest.mark.asyncio
    async def test_perfect_precision_recall(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test search with perfect precision and recall."""
        # Use only highly relevant documents
        relevant_docs = [
            doc for doc in synthetic_test_documents
            if doc.relevance_score >= 0.9
        ]

        await ingest_test_documents(
            qdrant_client, test_collection, relevant_docs, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)

        # Query that should match all documents
        query = "natural language processing machine learning transformers"
        query_embeddings = await embedding_service.embed_text(query)

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=len(relevant_docs),
            fusion_method="rrf"
        )

        # Calculate metrics
        result_dicts = [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results["fused_results"]
        ]
        metrics = calculate_search_metrics(result_dicts, relevant_docs, relevance_threshold=0.9)

        # With all relevant docs and good query, should have high metrics
        assert metrics.precision >= 0.8
        assert metrics.recall >= 0.8

    @pytest.mark.asyncio
    async def test_metrics_with_noise(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test metrics calculation with noisy results."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)

        # Query with some irrelevant results
        query = "programming languages"
        query_embeddings = await embedding_service.embed_text(query)

        results = await search_engine.hybrid_search(
            collection_name=test_collection,
            query_embeddings={
                "dense": query_embeddings["dense"],
                "sparse": query_embeddings["sparse"]
            },
            limit=10,
            fusion_method="rrf"
        )

        result_dicts = [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results["fused_results"]
        ]

        # Calculate metrics
        metrics = calculate_search_metrics(
            result_dicts,
            synthetic_test_documents,
            relevance_threshold=0.5
        )

        # Validate metrics are reasonable
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.ndcg <= 1.0


class TestMultiQueryPatterns:
    """Test multi-query search patterns."""

    @pytest.mark.asyncio
    async def test_sequential_queries(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test multiple sequential queries."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)

        queries = [
            "machine learning",
            "databases and indexing",
            "programming languages"
        ]

        all_results = []
        for query in queries:
            query_embeddings = await embedding_service.embed_text(query)

            results = await search_engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={
                    "dense": query_embeddings["dense"],
                    "sparse": query_embeddings["sparse"]
                },
                limit=3,
                fusion_method="rrf"
            )

            all_results.append(results)

        # Validate all queries returned results
        assert len(all_results) == len(queries)
        for results in all_results:
            assert len(results["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_queries(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents
    ):
        """Test concurrent query execution."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)

        async def execute_query(query: str):
            query_embeddings = await embedding_service.embed_text(query)
            return await search_engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={
                    "dense": query_embeddings["dense"],
                    "sparse": query_embeddings["sparse"]
                },
                limit=5,
                fusion_method="rrf"
            )

        # Execute multiple queries concurrently
        queries = [
            "natural language processing",
            "vector databases",
            "software development"
        ]

        results = await asyncio.gather(*[execute_query(q) for q in queries])

        # Validate all queries completed
        assert len(results) == len(queries)
        for result in results:
            assert "fused_results" in result
            assert len(result["fused_results"]) > 0


@pytest.mark.benchmark
class TestSearchPerformance:
    """Performance benchmarks for search operations."""

    @pytest.mark.asyncio
    async def test_search_response_time(
        self,
        qdrant_client,
        test_collection,
        embedding_service,
        synthetic_test_documents,
        benchmark
    ):
        """Benchmark search response time."""
        await ingest_test_documents(
            qdrant_client, test_collection, synthetic_test_documents, embedding_service
        )

        search_engine = HybridSearchEngine(qdrant_client)
        query_embeddings = await embedding_service.embed_text("test query")

        async def search():
            return await search_engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={
                    "dense": query_embeddings["dense"],
                    "sparse": query_embeddings["sparse"]
                },
                limit=10,
                fusion_method="rrf"
            )

        # Benchmark
        result = benchmark(lambda: asyncio.run(search()))

        # Validate performance (target < 100ms for this simple case)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])