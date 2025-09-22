"""
Unit tests for metadata filtering system.

Tests the comprehensive metadata filtering functionality including project isolation,
performance optimization, caching, and edge case handling.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from qdrant_client.http import models

# Import the modules under test
from workspace_qdrant_mcp.core.metadata_filtering import (
    MetadataFilterManager,
    FilterCriteria,
    FilterResult,
    FilterStrategy,
    FilterPerformanceLevel
)
from workspace_qdrant_mcp.core.metadata_schema import (
    MultiTenantMetadataSchema,
    CollectionCategory,
    WorkspaceScope,
    AccessLevel
)


class TestFilterCriteria:
    """Test the FilterCriteria class."""

    def test_criteria_initialization(self):
        """Test basic FilterCriteria initialization."""
        criteria = FilterCriteria(
            project_name="test-project",
            collection_types=["docs", "notes"],
            access_levels=[AccessLevel.PRIVATE]
        )

        # Should auto-generate project_id
        assert criteria.project_id is not None
        assert len(criteria.project_id) == 12
        assert criteria.project_name == "test-project"
        assert criteria.collection_types == ["docs", "notes"]
        assert criteria.access_levels == [AccessLevel.PRIVATE]

    def test_criteria_single_value_to_list_conversion(self):
        """Test conversion of single values to lists."""
        criteria = FilterCriteria(
            collection_types="docs",  # Single string
            tags="important",         # Single tag
            categories="documentation" # Single category
        )

        assert criteria.collection_types == ["docs"]
        assert criteria.tags == ["important"]
        assert criteria.categories == ["documentation"]

    def test_criteria_tenant_namespace_generation(self):
        """Test automatic tenant namespace generation."""
        criteria = FilterCriteria(
            project_name="my-project",
            collection_types=["docs"]
        )

        assert criteria.tenant_namespace == "my-project.docs"

    def test_criteria_cache_key_generation(self):
        """Test cache key generation for criteria."""
        criteria1 = FilterCriteria(
            project_name="project1",
            collection_types=["docs"],
            include_global=True
        )

        criteria2 = FilterCriteria(
            project_name="project1",
            collection_types=["docs"],
            include_global=True
        )

        criteria3 = FilterCriteria(
            project_name="project2",
            collection_types=["docs"],
            include_global=True
        )

        # Same criteria should generate same cache key
        assert criteria1.to_cache_key() == criteria2.to_cache_key()
        # Different criteria should generate different cache keys
        assert criteria1.to_cache_key() != criteria3.to_cache_key()

    def test_criteria_project_id_generation(self):
        """Test project ID generation from project name."""
        criteria = FilterCriteria(project_name="workspace-qdrant-mcp")

        # Should generate consistent 12-character hex project ID
        assert len(criteria.project_id) == 12
        assert all(c in '0123456789abcdef' for c in criteria.project_id)

        # Same project name should generate same ID
        criteria2 = FilterCriteria(project_name="workspace-qdrant-mcp")
        assert criteria.project_id == criteria2.project_id


class TestMetadataFilterManager:
    """Test the MetadataFilterManager class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = Mock()
        client.get_collection.return_value = Mock()
        return client

    @pytest.fixture
    def filter_manager(self, mock_qdrant_client):
        """Create a MetadataFilterManager instance."""
        return MetadataFilterManager(
            qdrant_client=mock_qdrant_client,
            enable_caching=True,
            enable_performance_monitoring=True
        )

    def test_manager_initialization(self, mock_qdrant_client):
        """Test MetadataFilterManager initialization."""
        manager = MetadataFilterManager(
            qdrant_client=mock_qdrant_client,
            enable_caching=False,
            enable_performance_monitoring=False
        )

        assert manager.qdrant_client == mock_qdrant_client
        assert manager.enable_caching is False
        assert manager.enable_performance_monitoring is False
        assert len(manager.indexed_fields) > 0

    def test_project_isolation_filter_with_project_name(self, filter_manager):
        """Test creating project isolation filter with project name."""
        result = filter_manager.create_project_isolation_filter("test-project")

        assert isinstance(result, FilterResult)
        assert result.filter is not None
        assert result.criteria.project_name == "test-project"
        assert result.criteria.project_id is not None
        assert len(result.criteria.project_id) == 12

        # Should have performance metrics
        assert "construction_time_ms" in result.performance_metrics
        assert "condition_count" in result.performance_metrics
        assert result.performance_metrics["condition_count"] > 0

    def test_project_isolation_filter_with_project_id(self, filter_manager):
        """Test creating project isolation filter with project ID."""
        project_id = "a1b2c3d4e5f6"
        result = filter_manager.create_project_isolation_filter(project_id)

        assert isinstance(result, FilterResult)
        assert result.criteria.project_id == project_id
        assert result.criteria.project_name is None

    def test_project_isolation_filter_with_metadata_schema(self, filter_manager):
        """Test creating project isolation filter with metadata schema."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test-project",
            collection_type="docs"
        )

        result = filter_manager.create_project_isolation_filter(metadata)

        assert isinstance(result, FilterResult)
        assert result.criteria.project_id == metadata.project_id
        assert result.criteria.project_name == metadata.project_name

    def test_project_isolation_filter_strategies(self, filter_manager):
        """Test different filtering strategies for project isolation."""
        project_name = "test-project"

        # Test strict strategy
        strict_result = filter_manager.create_project_isolation_filter(
            project_name, strategy=FilterStrategy.STRICT
        )
        assert strict_result.criteria.strategy == FilterStrategy.STRICT

        # Test lenient strategy
        lenient_result = filter_manager.create_project_isolation_filter(
            project_name, strategy=FilterStrategy.LENIENT
        )
        assert lenient_result.criteria.strategy == FilterStrategy.LENIENT

        # Different strategies should produce different filters
        # (This would need more detailed comparison of filter structures)

    def test_composite_filter_creation(self, filter_manager):
        """Test creating composite filters with multiple criteria."""
        criteria = FilterCriteria(
            project_name="test-project",
            collection_types=["docs", "notes"],
            access_levels=[AccessLevel.PRIVATE, AccessLevel.SHARED],
            workspace_scopes=[WorkspaceScope.PROJECT],
            tags=["important", "documentation"],
            priority_range=(3, 5)
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter is not None
        assert result.criteria == criteria
        assert result.performance_metrics["condition_count"] > 1
        assert len(result.optimizations_applied) > 0

    def test_collection_type_filter(self, filter_manager):
        """Test creating collection type filters."""
        # Single collection type
        single_result = filter_manager.create_collection_type_filter("docs")
        assert single_result.criteria.collection_types == ["docs"]

        # Multiple collection types
        multi_result = filter_manager.create_collection_type_filter(["docs", "notes"])
        assert set(multi_result.criteria.collection_types) == {"docs", "notes"}

    def test_access_control_filter(self, filter_manager):
        """Test creating access control filters."""
        result = filter_manager.create_access_control_filter(
            access_levels=AccessLevel.PRIVATE,
            created_by=["user", "admin"],
            mcp_readonly=True
        )

        assert result.criteria.access_levels == [AccessLevel.PRIVATE]
        assert result.criteria.created_by == ["user", "admin"]
        assert result.criteria.mcp_readonly is True

    def test_filter_caching(self, filter_manager):
        """Test filter caching functionality."""
        criteria = FilterCriteria(project_name="test-project")

        # First call should create and cache
        result1 = filter_manager.create_composite_filter(criteria)
        assert result1.cache_hit is False

        # Second call should hit cache
        result2 = filter_manager.create_composite_filter(criteria)
        assert result2.cache_hit is True

        # Results should be equivalent
        assert result1.criteria.to_cache_key() == result2.criteria.to_cache_key()

    def test_filter_caching_disabled(self, mock_qdrant_client):
        """Test behavior when caching is disabled."""
        manager = MetadataFilterManager(
            qdrant_client=mock_qdrant_client,
            enable_caching=False
        )

        criteria = FilterCriteria(project_name="test-project")

        result1 = manager.create_composite_filter(criteria)
        result2 = manager.create_composite_filter(criteria)

        # Should never hit cache when disabled
        assert result1.cache_hit is False
        assert result2.cache_hit is False

    def test_cache_expiration(self, filter_manager):
        """Test cache expiration functionality."""
        # Set very short TTL for testing
        filter_manager.cache_ttl_seconds = 0.1

        criteria = FilterCriteria(project_name="test-project")

        # Create initial filter
        result1 = filter_manager.create_composite_filter(criteria)
        assert result1.cache_hit is False

        # Wait for cache to expire
        time.sleep(0.2)

        # Should not hit expired cache
        result2 = filter_manager.create_composite_filter(criteria)
        assert result2.cache_hit is False

    def test_performance_stats_collection(self, filter_manager):
        """Test performance statistics collection."""
        # Create several filters to generate stats
        for i in range(5):
            criteria = FilterCriteria(project_name=f"project-{i}")
            filter_manager.create_composite_filter(criteria)

        stats = filter_manager.get_filter_performance_stats()

        assert "composite" in stats
        assert stats["composite"]["total_operations"] == 5
        assert "avg_construction_time_ms" in stats["composite"]
        assert "avg_condition_count" in stats["composite"]
        assert "cache" in stats

    def test_performance_stats_disabled(self, mock_qdrant_client):
        """Test behavior when performance monitoring is disabled."""
        manager = MetadataFilterManager(
            qdrant_client=mock_qdrant_client,
            enable_performance_monitoring=False
        )

        criteria = FilterCriteria(project_name="test-project")
        manager.create_composite_filter(criteria)

        stats = manager.get_filter_performance_stats()
        assert stats == {"monitoring_disabled": True}

    def test_cache_clearing(self, filter_manager):
        """Test cache clearing functionality."""
        criteria = FilterCriteria(project_name="test-project")

        # Create cached filter
        result1 = filter_manager.create_composite_filter(criteria)
        assert result1.cache_hit is False

        # Should hit cache
        result2 = filter_manager.create_composite_filter(criteria)
        assert result2.cache_hit is True

        # Clear cache
        filter_manager.clear_cache()

        # Should not hit cache after clearing
        result3 = filter_manager.create_composite_filter(criteria)
        assert result3.cache_hit is False

    def test_invalid_project_identifier(self, filter_manager):
        """Test handling of invalid project identifiers."""
        with pytest.raises(ValueError):
            filter_manager.create_project_isolation_filter(123)  # Invalid type

    def test_empty_filter_criteria(self, filter_manager):
        """Test handling of empty filter criteria."""
        criteria = FilterCriteria()
        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        assert len(result.warnings) > 0  # Should warn about empty criteria

    def test_filter_condition_counting(self, filter_manager):
        """Test filter condition counting methods."""
        # Create a filter with known number of conditions
        criteria = FilterCriteria(
            project_name="test-project",
            collection_types=["docs"],
            access_levels=[AccessLevel.PRIVATE]
        )

        result = filter_manager.create_composite_filter(criteria)

        # Should have counted conditions correctly
        assert result.performance_metrics["condition_count"] >= 3
        assert result.performance_metrics["indexed_conditions"] >= 0

    def test_complexity_score_calculation(self, filter_manager):
        """Test filter complexity score calculation."""
        # Simple filter
        simple_criteria = FilterCriteria(project_name="test-project")
        simple_result = filter_manager.create_composite_filter(simple_criteria)

        # Complex filter
        complex_criteria = FilterCriteria(
            project_name="test-project",
            collection_types=["docs", "notes", "scratchbook"],
            access_levels=[AccessLevel.PRIVATE, AccessLevel.SHARED],
            workspace_scopes=[WorkspaceScope.PROJECT, WorkspaceScope.SHARED],
            tags=["tag1", "tag2", "tag3"],
            priority_range=(1, 5),
            created_after="2024-01-01T00:00:00Z",
            updated_before="2024-12-31T23:59:59Z"
        )
        complex_result = filter_manager.create_composite_filter(complex_criteria)

        # Complex filter should have higher complexity score
        simple_score = simple_result.performance_metrics["complexity_score"]
        complex_score = complex_result.performance_metrics["complexity_score"]

        assert complex_score > simple_score
        assert 0 <= simple_score <= 10
        assert 0 <= complex_score <= 10

    @patch('src.python.common.core.metadata_filtering.logger')
    def test_logging_behavior(self, mock_logger, filter_manager):
        """Test that appropriate logging occurs."""
        criteria = FilterCriteria(project_name="test-project")
        filter_manager.create_composite_filter(criteria)

        # Should have logged debug messages
        assert mock_logger.debug.called

    def test_temporal_filtering(self, filter_manager):
        """Test temporal filtering functionality."""
        criteria = FilterCriteria(
            project_name="test-project",
            created_after="2024-01-01T00:00:00Z",
            created_before="2024-12-31T23:59:59Z",
            updated_after="2024-06-01T00:00:00Z"
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        assert result.performance_metrics["condition_count"] >= 2  # Created and updated ranges
        assert "temporal_filtering" in result.optimizations_applied

    def test_organizational_filtering(self, filter_manager):
        """Test organizational filtering functionality."""
        criteria = FilterCriteria(
            project_name="test-project",
            tags=["documentation", "important"],
            categories=["reference"],
            priority_range=(4, 5)
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        assert "organizational_filtering" in result.optimizations_applied

    def test_special_conditions_handling(self, filter_manager):
        """Test handling of special conditions like global/shared inclusion."""
        criteria = FilterCriteria(
            project_name="test-project",
            include_global=True,
            include_shared=True,
            include_legacy=False,
            require_metadata=True
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        assert "special_conditions" in result.optimizations_applied


class TestFilterIntegration:
    """Test integration aspects of the filtering system."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client with collection info."""
        client = Mock()
        collection_info = Mock()
        collection_info.config = Mock()
        collection_info.config.params = Mock()
        client.get_collection.return_value = collection_info
        return client

    def test_filter_validation_with_existing_collection(self, mock_qdrant_client):
        """Test filter validation against existing collections."""
        manager = MetadataFilterManager(qdrant_client=mock_qdrant_client)

        criteria = FilterCriteria(
            project_name="test-project",
            performance_level=FilterPerformanceLevel.FAST
        )

        validation_result = manager.validate_filter_compatibility("test-collection", criteria)

        assert validation_result.is_valid
        # Should call get_collection to validate
        mock_qdrant_client.get_collection.assert_called_with("test-collection")

    def test_filter_validation_with_nonexistent_collection(self, mock_qdrant_client):
        """Test filter validation with nonexistent collection."""
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")

        manager = MetadataFilterManager(qdrant_client=mock_qdrant_client)

        criteria = FilterCriteria(project_name="test-project")

        validation_result = manager.validate_filter_compatibility("nonexistent", criteria)

        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0

    def test_metadata_schema_integration(self):
        """Test integration with MultiTenantMetadataSchema."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="integration-test",
            collection_type="docs",
            created_by="test-user"
        )

        # Should be able to create criteria from metadata
        criteria = FilterCriteria(
            project_id=metadata.project_id,
            project_name=metadata.project_name,
            collection_types=[metadata.collection_type],
            access_levels=[metadata.access_level],
            workspace_scopes=[metadata.workspace_scope]
        )

        assert criteria.project_id == metadata.project_id
        assert criteria.project_name == metadata.project_name


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def filter_manager(self):
        """Create a filter manager for edge case testing."""
        mock_client = Mock()
        return MetadataFilterManager(qdrant_client=mock_client)

    def test_extremely_large_filter_criteria(self, filter_manager):
        """Test handling of extremely large filter criteria."""
        # Create criteria with many conditions
        large_tags = [f"tag-{i}" for i in range(100)]
        large_categories = [f"category-{i}" for i in range(50)]

        criteria = FilterCriteria(
            project_name="test-project",
            tags=large_tags,
            categories=large_categories
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        # Should handle large number of conditions
        assert result.performance_metrics["condition_count"] > 100

    def test_unicode_project_names(self, filter_manager):
        """Test handling of unicode project names."""
        unicode_names = [
            "测试项目",  # Chinese
            "プロジェクト",  # Japanese
            "проект",  # Russian
            "proyecto-español",  # Spanish with special chars
            "🚀-rocket-project"  # Emoji
        ]

        for name in unicode_names:
            criteria = FilterCriteria(project_name=name)
            result = filter_manager.create_composite_filter(criteria)

            assert isinstance(result, FilterResult)
            assert result.criteria.project_id is not None
            assert len(result.criteria.project_id) == 12

    def test_null_and_empty_values(self, filter_manager):
        """Test handling of null and empty values."""
        criteria = FilterCriteria(
            project_name="",  # Empty string
            collection_types=[],  # Empty list
            tags=None,  # None value
        )

        result = filter_manager.create_composite_filter(criteria)

        assert isinstance(result, FilterResult)
        # Should handle gracefully

    def test_memory_usage_with_large_cache(self, filter_manager):
        """Test memory usage with large number of cached filters."""
        # Create many different filter criteria to fill cache
        for i in range(1500):  # More than cleanup threshold
            criteria = FilterCriteria(project_name=f"project-{i}")
            filter_manager.create_composite_filter(criteria)

        # Cache should be cleaned up automatically
        assert len(filter_manager._filter_cache) <= 1000

    def test_concurrent_access_simulation(self, filter_manager):
        """Test simulation of concurrent access patterns."""
        import threading
        import time

        results = []
        errors = []

        def create_filters():
            try:
                for i in range(10):
                    criteria = FilterCriteria(project_name=f"concurrent-{i}")
                    result = filter_manager.create_composite_filter(criteria)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_filters)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 filters each

    def test_performance_under_load(self, filter_manager):
        """Test performance characteristics under load."""
        start_time = time.time()

        # Create many filters rapidly
        for i in range(100):
            criteria = FilterCriteria(
                project_name=f"load-test-{i}",
                collection_types=["docs", "notes"],
                access_levels=[AccessLevel.PRIVATE]
            )
            result = filter_manager.create_composite_filter(criteria)

        total_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert total_time < 5.0  # 5 seconds for 100 filters

        # Check performance stats
        stats = filter_manager.get_filter_performance_stats()
        assert stats["composite"]["total_operations"] >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])