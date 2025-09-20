"""
Integration tests for metadata filtering system with hybrid search.

Tests the integration between the new metadata filtering system and the existing
hybrid search infrastructure to ensure project isolation works correctly.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the integrated components
from src.python.common.core.metadata_filtering import (
    MetadataFilterManager,
    FilterCriteria,
    FilterStrategy,
    FilterPerformanceLevel
)
from src.python.common.core.metadata_schema import (
    MultiTenantMetadataSchema,
    CollectionCategory,
    WorkspaceScope,
    AccessLevel
)


class TestMetadataFilteringIntegration:
    """Test integration between metadata filtering and hybrid search systems."""

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

    def test_project_isolation_filter_creation(self, filter_manager):
        """Test creating project isolation filters."""
        # Test with project name
        result = filter_manager.create_project_isolation_filter("test-project")
        assert result.filter is not None
        assert result.criteria.project_name == "test-project"
        assert result.criteria.project_id is not None

        # Test with project ID
        project_id = "a1b2c3d4e5f6"
        result = filter_manager.create_project_isolation_filter(project_id)
        assert result.filter is not None
        assert result.criteria.project_id == project_id

    def test_project_isolation_filter_with_metadata_schema(self, filter_manager):
        """Test creating filters with MultiTenantMetadataSchema."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="integration-test",
            collection_type="docs",
            created_by="test"
        )

        result = filter_manager.create_project_isolation_filter(metadata)
        assert result.filter is not None
        assert result.criteria.project_id == metadata.project_id
        assert result.criteria.project_name == metadata.project_name

    def test_composite_filter_with_multiple_criteria(self, filter_manager):
        """Test creating composite filters with complex criteria."""
        criteria = FilterCriteria(
            project_name="test-project",
            collection_types=["docs", "notes"],
            workspace_scopes=[WorkspaceScope.PROJECT],
            access_levels=[AccessLevel.PRIVATE],
            tags=["important"],
            strategy=FilterStrategy.STRICT,
            performance_level=FilterPerformanceLevel.FAST
        )

        result = filter_manager.create_composite_filter(criteria)
        assert result.filter is not None
        assert result.criteria == criteria
        assert result.performance_metrics["condition_count"] > 0

    def test_filter_performance_monitoring(self, filter_manager):
        """Test that performance monitoring works correctly."""
        # Create several filters to generate performance data
        for i in range(3):
            criteria = FilterCriteria(project_name=f"project-{i}")
            filter_manager.create_composite_filter(criteria)

        stats = filter_manager.get_filter_performance_stats()
        assert "composite" in stats
        assert stats["composite"]["total_operations"] >= 3
        assert "cache" in stats

    def test_filter_caching_functionality(self, filter_manager):
        """Test that filter caching works correctly."""
        criteria = FilterCriteria(project_name="cached-project")

        # First call should create and cache
        result1 = filter_manager.create_composite_filter(criteria)
        assert result1.cache_hit is False

        # Second call should hit cache
        result2 = filter_manager.create_composite_filter(criteria)
        assert result2.cache_hit is True

    def test_filter_strategies(self, filter_manager):
        """Test different filtering strategies."""
        project_name = "strategy-test"

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

    def test_collection_type_filtering(self, filter_manager):
        """Test collection type specific filtering."""
        # Single collection type
        result1 = filter_manager.create_collection_type_filter("docs")
        assert result1.criteria.collection_types == ["docs"]

        # Multiple collection types
        result2 = filter_manager.create_collection_type_filter(["docs", "notes"])
        assert set(result2.criteria.collection_types) == {"docs", "notes"}

    def test_access_control_filtering(self, filter_manager):
        """Test access control filtering functionality."""
        result = filter_manager.create_access_control_filter(
            access_levels=[AccessLevel.PRIVATE, AccessLevel.SHARED],
            created_by=["user", "admin"],
            mcp_readonly=False
        )

        assert AccessLevel.PRIVATE in result.criteria.access_levels
        assert AccessLevel.SHARED in result.criteria.access_levels
        assert result.criteria.created_by == ["user", "admin"]
        assert result.criteria.mcp_readonly is False

    def test_filter_validation(self, filter_manager, mock_qdrant_client):
        """Test filter validation against collections."""
        criteria = FilterCriteria(
            project_name="validation-test",
            performance_level=FilterPerformanceLevel.FAST
        )

        # Mock successful collection retrieval
        mock_qdrant_client.get_collection.return_value = Mock()

        validation_result = filter_manager.validate_filter_compatibility(
            "test-collection", criteria
        )
        assert validation_result.is_valid

    def test_error_handling(self, mock_qdrant_client):
        """Test error handling in filter creation."""
        # Create manager with problematic client
        mock_qdrant_client.get_collection.side_effect = Exception("Connection error")

        manager = MetadataFilterManager(
            qdrant_client=mock_qdrant_client,
            enable_caching=False,
            enable_performance_monitoring=False
        )

        # Should still create filters despite client issues
        result = manager.create_project_isolation_filter("error-test")
        assert result.filter is not None

    def test_metadata_schema_integration(self):
        """Test integration with the metadata schema system."""
        # Test different metadata schema factory methods
        project_metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="schema-test",
            collection_type="docs"
        )

        system_metadata = MultiTenantMetadataSchema.create_for_system(
            collection_name="__test_system",
            collection_type="memory_collection"
        )

        library_metadata = MultiTenantMetadataSchema.create_for_library(
            collection_name="_test_library",
            collection_type="code_collection"
        )

        # Should be able to create criteria from all metadata types
        mock_client = Mock()
        manager = MetadataFilterManager(mock_client)

        for metadata in [project_metadata, system_metadata, library_metadata]:
            result = manager.create_project_isolation_filter(metadata)
            assert result.filter is not None

    def test_performance_optimization(self, filter_manager):
        """Test performance optimization features."""
        # Test with different performance levels
        for perf_level in [
            FilterPerformanceLevel.FAST,
            FilterPerformanceLevel.BALANCED,
            FilterPerformanceLevel.COMPREHENSIVE
        ]:
            criteria = FilterCriteria(
                project_name="perf-test",
                performance_level=perf_level
            )

            result = filter_manager.create_composite_filter(criteria)
            assert result.filter is not None
            assert result.performance_metrics["construction_time_ms"] >= 0

    def test_edge_cases(self, filter_manager):
        """Test edge cases and boundary conditions."""
        # Empty criteria
        empty_criteria = FilterCriteria()
        result = filter_manager.create_composite_filter(empty_criteria)
        assert result.filter is not None
        assert len(result.warnings) > 0

        # Very long project name
        long_name = "a" * 200
        try:
            criteria = FilterCriteria(project_name=long_name)
            result = filter_manager.create_composite_filter(criteria)
            # Should handle gracefully
            assert result.filter is not None
        except ValueError:
            # Or raise appropriate validation error
            pass

        # Unicode project names
        unicode_names = ["测试项目", "プロジェクト", "🚀-project"]
        for name in unicode_names:
            criteria = FilterCriteria(project_name=name)
            result = filter_manager.create_composite_filter(criteria)
            assert result.filter is not None
            assert result.criteria.project_id is not None

    @patch('src.python.common.core.metadata_filtering.logger')
    def test_logging_integration(self, mock_logger, filter_manager):
        """Test that logging works correctly throughout the integration."""
        criteria = FilterCriteria(project_name="logging-test")
        filter_manager.create_composite_filter(criteria)

        # Should have logged debug messages
        assert mock_logger.debug.called

    def test_concurrent_filter_creation(self, filter_manager):
        """Test concurrent filter creation for thread safety."""
        import threading
        import time

        results = []
        errors = []

        def create_filters():
            try:
                for i in range(5):
                    criteria = FilterCriteria(project_name=f"concurrent-{i}")
                    result = filter_manager.create_composite_filter(criteria)
                    results.append(result)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_filters)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have errors and should have created filters
        assert len(errors) == 0
        assert len(results) == 15  # 3 threads * 5 filters each

    def test_memory_usage(self, filter_manager):
        """Test memory usage with many filter operations."""
        # Create many different filters
        for i in range(100):
            criteria = FilterCriteria(project_name=f"memory-test-{i}")
            filter_manager.create_composite_filter(criteria)

        # Cache should be managed automatically
        stats = filter_manager.get_filter_performance_stats()
        assert stats["cache"]["total_entries"] <= 1000  # Should be cleaned up

    def test_integration_with_real_metadata(self):
        """Test integration with realistic metadata scenarios."""
        mock_client = Mock()
        manager = MetadataFilterManager(mock_client)

        # Simulate real project scenarios
        scenarios = [
            {
                "project_name": "workspace-qdrant-mcp",
                "collection_types": ["docs", "notes"],
                "access_levels": [AccessLevel.PRIVATE],
                "tags": ["development", "mcp"]
            },
            {
                "project_name": "ai-assistant",
                "collection_types": ["knowledge", "context"],
                "access_levels": [AccessLevel.SHARED],
                "workspace_scopes": [WorkspaceScope.PROJECT]
            },
            {
                "project_name": "data-pipeline",
                "collection_types": ["docs"],
                "access_levels": [AccessLevel.PUBLIC],
                "include_global": True,
                "include_shared": True
            }
        ]

        for scenario in scenarios:
            criteria = FilterCriteria(**scenario)
            result = manager.create_composite_filter(criteria)
            assert result.filter is not None
            assert result.performance_metrics["condition_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])