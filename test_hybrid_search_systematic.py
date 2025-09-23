"""Systematic test for hybrid_search.py coverage - Target: 20%+ coverage in 2-3 minutes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import core classes from hybrid_search module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

from common.core.hybrid_search import (
    TenantAwareResult,
    TenantAwareResultDeduplicator
)

class TestTenantAwareResult:
    """Test TenantAwareResult dataclass and its methods."""

    def test_basic_initialization(self):
        """Test basic TenantAwareResult creation."""
        result = TenantAwareResult(
            id="test_id",
            score=0.85,
            payload={"content": "test content"},
            collection="test_collection",
            search_type="dense"
        )

        assert result.id == "test_id"
        assert result.score == 0.85
        assert result.payload == {"content": "test content"}
        assert result.collection == "test_collection"
        assert result.search_type == "dense"
        assert result.tenant_metadata == {}
        assert result.project_context == {}
        assert result.deduplication_key == "test_id"

    def test_initialization_with_metadata(self):
        """Test TenantAwareResult with tenant metadata."""
        tenant_metadata = {"tenant_id": "abc123", "permissions": ["read"]}
        project_context = {"project": "test_project", "branch": "main"}

        result = TenantAwareResult(
            id="test_id",
            score=0.75,
            payload={"content": "test"},
            collection="test_collection",
            search_type="sparse",
            tenant_metadata=tenant_metadata,
            project_context=project_context,
            deduplication_key="custom_key"
        )

        assert result.tenant_metadata == tenant_metadata
        assert result.project_context == project_context
        assert result.deduplication_key == "custom_key"

    def test_deduplication_key_generation_content_hash(self):
        """Test deduplication key generation using content hash."""
        payload = {"content_hash": "abc123", "file_path": "/test/file.py"}

        result = TenantAwareResult(
            id="test_id",
            score=0.5,
            payload=payload,
            collection="test",
            search_type="dense"
        )

        assert result.deduplication_key == "abc123"

    def test_deduplication_key_generation_file_path(self):
        """Test deduplication key generation using file path fallback."""
        payload = {"file_path": "/test/file.py", "document_id": "doc123"}

        result = TenantAwareResult(
            id="test_id",
            score=0.5,
            payload=payload,
            collection="test",
            search_type="dense"
        )

        assert result.deduplication_key == "/test/file.py"

    def test_deduplication_key_generation_document_id(self):
        """Test deduplication key generation using document_id fallback."""
        payload = {"document_id": "doc123", "other": "data"}

        result = TenantAwareResult(
            id="test_id",
            score=0.5,
            payload=payload,
            collection="test",
            search_type="dense"
        )

        assert result.deduplication_key == "doc123"

    def test_deduplication_key_generation_id_fallback(self):
        """Test deduplication key generation using ID as final fallback."""
        payload = {"other": "data"}

        result = TenantAwareResult(
            id="test_id",
            score=0.5,
            payload=payload,
            collection="test",
            search_type="dense"
        )

        assert result.deduplication_key == "test_id"


class TestTenantAwareResultDeduplicator:
    """Test TenantAwareResultDeduplicator functionality."""

    def test_deduplicator_initialization_default(self):
        """Test deduplicator initialization with defaults."""
        dedup = TenantAwareResultDeduplicator()
        assert dedup.preserve_tenant_isolation == True

    def test_deduplicator_initialization_custom(self):
        """Test deduplicator initialization with custom settings."""
        dedup = TenantAwareResultDeduplicator(preserve_tenant_isolation=False)
        assert dedup.preserve_tenant_isolation == False

    def test_deduplicator_with_duplicate_results(self):
        """Test deduplication with duplicate content."""
        dedup = TenantAwareResultDeduplicator()

        # Create duplicate results with same deduplication key
        result1 = TenantAwareResult(
            id="id1",
            score=0.8,
            payload={"content_hash": "hash123", "content": "test"},
            collection="collection1",
            search_type="dense"
        )

        result2 = TenantAwareResult(
            id="id2",
            score=0.9,
            payload={"content_hash": "hash123", "content": "test"},
            collection="collection2",
            search_type="sparse"
        )

        # Both should have same deduplication key
        assert result1.deduplication_key == result2.deduplication_key == "hash123"

    def test_deduplicator_with_unique_results(self):
        """Test deduplicator with unique results."""
        dedup = TenantAwareResultDeduplicator()

        result1 = TenantAwareResult(
            id="id1",
            score=0.8,
            payload={"content_hash": "hash1", "content": "test1"},
            collection="collection1",
            search_type="dense"
        )

        result2 = TenantAwareResult(
            id="id2",
            score=0.9,
            payload={"content_hash": "hash2", "content": "test2"},
            collection="collection2",
            search_type="sparse"
        )

        # Should have different deduplication keys
        assert result1.deduplication_key != result2.deduplication_key


# Mock classes for testing more complex functionality
@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for testing."""
    client = Mock()
    client.search = AsyncMock()
    return client

@pytest.fixture
def sample_dense_results():
    """Sample dense search results."""
    return [
        Mock(id="doc1", score=0.9, payload={"content": "dense result 1"}),
        Mock(id="doc2", score=0.8, payload={"content": "dense result 2"}),
        Mock(id="doc3", score=0.7, payload={"content": "dense result 3"})
    ]

@pytest.fixture
def sample_sparse_results():
    """Sample sparse search results."""
    return [
        Mock(id="doc2", score=0.85, payload={"content": "sparse result 2"}),
        Mock(id="doc4", score=0.75, payload={"content": "sparse result 4"}),
        Mock(id="doc1", score=0.65, payload={"content": "sparse result 1"})
    ]


class TestHybridSearchIntegration:
    """Integration tests for hybrid search functionality."""

    def test_tenant_aware_result_creation_from_search_results(self, sample_dense_results):
        """Test converting search results to TenantAwareResult."""
        search_result = sample_dense_results[0]

        tenant_result = TenantAwareResult(
            id=search_result.id,
            score=search_result.score,
            payload=search_result.payload,
            collection="test_collection",
            search_type="dense"
        )

        assert tenant_result.id == "doc1"
        assert tenant_result.score == 0.9
        assert tenant_result.payload == {"content": "dense result 1"}
        assert tenant_result.search_type == "dense"

    def test_multiple_tenant_results_creation(self, sample_dense_results, sample_sparse_results):
        """Test creating multiple tenant-aware results."""
        tenant_results = []

        # Convert dense results
        for result in sample_dense_results:
            tenant_result = TenantAwareResult(
                id=result.id,
                score=result.score,
                payload=result.payload,
                collection="dense_collection",
                search_type="dense"
            )
            tenant_results.append(tenant_result)

        # Convert sparse results
        for result in sample_sparse_results:
            tenant_result = TenantAwareResult(
                id=result.id,
                score=result.score,
                payload=result.payload,
                collection="sparse_collection",
                search_type="sparse"
            )
            tenant_results.append(tenant_result)

        assert len(tenant_results) == 6
        assert sum(1 for r in tenant_results if r.search_type == "dense") == 3
        assert sum(1 for r in tenant_results if r.search_type == "sparse") == 3

    def test_tenant_metadata_preservation(self):
        """Test that tenant metadata is properly preserved."""
        tenant_metadata = {
            "tenant_id": "tenant123",
            "user_id": "user456",
            "permissions": ["read", "write"]
        }

        result = TenantAwareResult(
            id="test_id",
            score=0.8,
            payload={"content": "test"},
            collection="test_collection",
            search_type="dense",
            tenant_metadata=tenant_metadata
        )

        assert result.tenant_metadata["tenant_id"] == "tenant123"
        assert result.tenant_metadata["user_id"] == "user456"
        assert result.tenant_metadata["permissions"] == ["read", "write"]

    def test_project_context_handling(self):
        """Test project context handling in tenant-aware results."""
        project_context = {
            "project_name": "test_project",
            "branch": "feature/testing",
            "commit_hash": "abc123def"
        }

        result = TenantAwareResult(
            id="test_id",
            score=0.8,
            payload={"content": "test"},
            collection="test_collection",
            search_type="dense",
            project_context=project_context
        )

        assert result.project_context["project_name"] == "test_project"
        assert result.project_context["branch"] == "feature/testing"
        assert result.project_context["commit_hash"] == "abc123def"


if __name__ == "__main__":
    # Quick test execution for development
    pytest.main([__file__, "-v", "--tb=short"])