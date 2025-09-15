#!/usr/bin/env python3
"""
Test script for multi-tenant result aggregation functionality.

Task 233.5: Test implementation of multi-tenant search result aggregation
including deduplication, tenant-aware ranking, and API consistency.

Usage:
    python 20250915-1030_test_multitenant_aggregation.py
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

try:
    from common.core.hybrid_search import (
        TenantAwareResult,
        TenantAwareResultDeduplicator,
        MultiTenantResultAggregator,
        HybridSearchEngine
    )
    print("✅ Successfully imported multi-tenant aggregation components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)


def create_mock_search_result(result_id: str, score: float, collection: str,
                             content_hash: str = None, project_name: str = None,
                             tenant_namespace: str = None):
    """Create mock search result for testing."""
    result = Mock()
    result.id = result_id
    result.score = score
    result.payload = {
        "content": f"Mock content for {result_id}",
        "content_hash": content_hash or f"hash_{result_id}",
        "project_name": project_name,
        "tenant_namespace": tenant_namespace,
        "collection_type": "docs",
        "workspace_scope": "project"
    }
    return result


def test_tenant_aware_result_creation():
    """Test TenantAwareResult creation and deduplication key generation."""
    print("\n🧪 Testing TenantAwareResult creation...")

    # Test with content hash
    result1 = TenantAwareResult(
        id="doc1",
        score=0.9,
        payload={"content_hash": "abc123", "content": "Test content"},
        collection="docs",
        search_type="hybrid"
    )

    assert result1.deduplication_key == "abc123", "Should use content hash as deduplication key"
    print("✅ Content hash deduplication key works")

    # Test fallback to file path
    result2 = TenantAwareResult(
        id="doc2",
        score=0.8,
        payload={"file_path": "/path/to/file.txt", "content": "Test content"},
        collection="docs",
        search_type="hybrid"
    )

    assert result2.deduplication_key == "/path/to/file.txt", "Should use file path as fallback"
    print("✅ File path fallback deduplication key works")

    # Test fallback to document ID
    result3 = TenantAwareResult(
        id="doc3",
        score=0.7,
        payload={"document_id": "unique_doc_id", "content": "Test content"},
        collection="docs",
        search_type="hybrid"
    )

    assert result3.deduplication_key == "unique_doc_id", "Should use document ID as fallback"
    print("✅ Document ID fallback deduplication key works")


def test_result_deduplication():
    """Test TenantAwareResultDeduplicator functionality."""
    print("\n🧪 Testing result deduplication...")

    deduplicator = TenantAwareResultDeduplicator(preserve_tenant_isolation=True)

    # Create duplicate results with same content hash but different tenant contexts
    results = [
        TenantAwareResult(
            id="doc1", score=0.9, payload={"content_hash": "same_hash"},
            collection="docs", search_type="hybrid",
            project_context={"project_name": "project_a"},
            tenant_metadata={"tenant_namespace": "tenant_a"}
        ),
        TenantAwareResult(
            id="doc1", score=0.8, payload={"content_hash": "same_hash"},
            collection="docs", search_type="hybrid",
            project_context={"project_name": "project_b"},
            tenant_metadata={"tenant_namespace": "tenant_b"}
        ),
        TenantAwareResult(
            id="doc2", score=0.7, payload={"content_hash": "different_hash"},
            collection="notes", search_type="hybrid",
            project_context={"project_name": "project_a"},
            tenant_metadata={"tenant_namespace": "tenant_a"}
        )
    ]

    deduplicated = deduplicator.deduplicate_results(results, "max_score")

    # With tenant isolation, should keep separate results for different tenants
    assert len(deduplicated) == 3, f"Expected 3 results with tenant isolation, got {len(deduplicated)}"
    print("✅ Tenant isolation preserves separate results")

    # Test without tenant isolation
    deduplicator_global = TenantAwareResultDeduplicator(preserve_tenant_isolation=False)
    deduplicated_global = deduplicator_global.deduplicate_results(results, "max_score")

    # Without tenant isolation, should deduplicate across tenants
    assert len(deduplicated_global) == 2, f"Expected 2 results without tenant isolation, got {len(deduplicated_global)}"
    print("✅ Global deduplication works across tenants")

    # Verify score aggregation
    best_result = next(r for r in deduplicated_global if r.deduplication_key == "same_hash")
    assert best_result.score == 0.9, f"Expected max score 0.9, got {best_result.score}"
    print("✅ Max score aggregation works correctly")


def test_multi_tenant_aggregator():
    """Test MultiTenantResultAggregator functionality."""
    print("\n🧪 Testing multi-tenant result aggregator...")

    aggregator = MultiTenantResultAggregator(
        preserve_tenant_isolation=True,
        enable_score_normalization=True,
        default_aggregation_method="max_score"
    )

    # Create mock collection results
    collection_results = {
        "docs": [
            create_mock_search_result("doc1", 0.9, "docs", "hash1", "project_a"),
            create_mock_search_result("doc2", 0.7, "docs", "hash2", "project_a"),
        ],
        "notes": [
            create_mock_search_result("note1", 0.8, "notes", "hash3", "project_a"),
            create_mock_search_result("doc1", 0.6, "notes", "hash1", "project_a"),  # Duplicate
        ]
    }

    project_contexts = {
        "docs": {"project_name": "project_a"},
        "notes": {"project_name": "project_a"}
    }

    aggregated = aggregator.aggregate_multi_collection_results(
        collection_results=collection_results,
        project_contexts=project_contexts,
        limit=10,
        score_threshold=0.0,
        aggregation_method="max_score"
    )

    assert "total_results" in aggregated, "Response should include total_results"
    assert "results" in aggregated, "Response should include results"
    assert "aggregation_metadata" in aggregated, "Response should include aggregation_metadata"

    metadata = aggregated["aggregation_metadata"]
    assert metadata["collection_count"] == 2, "Should track collection count"
    assert metadata["raw_result_count"] == 4, "Should track raw result count"
    assert metadata["score_normalization_enabled"] == True, "Should track normalization setting"

    print("✅ Multi-tenant aggregator produces expected response structure")
    print(f"   - Raw results: {metadata['raw_result_count']}")
    print(f"   - Final results: {metadata['final_count']}")
    print(f"   - Collections: {metadata['collection_count']}")


async def test_hybrid_search_engine_integration():
    """Test HybridSearchEngine integration with multi-tenant aggregation."""
    print("\n🧪 Testing HybridSearchEngine multi-collection search...")

    # Mock Qdrant client
    mock_client = Mock()
    mock_client.search.return_value = [
        create_mock_search_result("doc1", 0.9, "docs", "hash1", "project_a")
    ]

    # Create hybrid search engine with multi-tenant aggregation (disable optimizations to avoid validation issues)
    engine = HybridSearchEngine(
        client=mock_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=True
    )

    assert engine.multi_tenant_aggregation_enabled == True, "Multi-tenant aggregation should be enabled"
    assert engine.result_aggregator is not None, "Result aggregator should be initialized"

    print("✅ HybridSearchEngine initializes with multi-tenant aggregation")

    # Test configuration
    config_result = engine.configure_result_aggregation(
        preserve_tenant_isolation=False,
        enable_score_normalization=False,
        default_aggregation_method="avg_score"
    )

    assert config_result["preserve_tenant_isolation"] == False, "Configuration should update"
    assert config_result["enable_score_normalization"] == False, "Configuration should update"
    assert config_result["default_aggregation_method"] == "avg_score", "Configuration should update"

    print("✅ Result aggregation configuration works")

    # Test stats
    stats = engine.get_result_aggregation_stats()
    assert stats["multi_tenant_aggregation_enabled"] == True, "Stats should show enabled state"

    print("✅ Result aggregation statistics work")


def test_api_consistency():
    """Test that new aggregation maintains API consistency."""
    print("\n🧪 Testing API consistency...")

    # Test that TenantAwareResult can be converted to standard format
    tenant_result = TenantAwareResult(
        id="doc1",
        score=0.9,
        payload={"content": "test"},
        collection="docs",
        search_type="hybrid",
        tenant_metadata={"project_name": "test_project"},
        project_context={"project_name": "test_project"}
    )

    aggregator = MultiTenantResultAggregator()
    api_results = aggregator._convert_to_api_format([tenant_result])

    assert len(api_results) == 1, "Should convert single result"
    api_result = api_results[0]

    # Check required API fields
    required_fields = ["id", "score", "payload", "collection", "search_type"]
    for field in required_fields:
        assert field in api_result, f"API result should include {field}"

    # Check optional tenant fields
    assert "tenant_metadata" in api_result, "Should include tenant metadata"
    assert "project_context" in api_result, "Should include project context"

    print("✅ API format conversion maintains required fields")
    print("✅ API format includes enhanced tenant metadata")


def main():
    """Run all tests."""
    print("🚀 Starting multi-tenant result aggregation tests...\n")

    try:
        # Run synchronous tests
        test_tenant_aware_result_creation()
        test_result_deduplication()
        test_multi_tenant_aggregator()
        test_api_consistency()

        # Run async tests
        asyncio.run(test_hybrid_search_engine_integration())

        print("\n🎉 All tests passed! Multi-tenant result aggregation is working correctly.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)