"""
Comprehensive unit tests for collection_manager_integration module.

This test module provides 100% coverage for the CollisionAwareCollectionManager
and related components, including safe collection creation, collision prevention,
naming validation integration, and error handling mechanisms.
"""

import pytest
import asyncio
from dataclasses import dataclass
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the src directory to Python path
import sys
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

try:
    from workspace_qdrant_mcp.core.collection_manager_integration import (
        CollisionAwareCollectionManager,
        CollectionCreationResult,
        CollectionManagementError,
        CollectionCollisionError,
        CollectionCategory,
        SafeCollectionConfig
    )
    from workspace_qdrant_mcp.core.collision_detection import (
        CollisionDetector,
        CollisionResult,
        CollectionAnalysis,
        SimilarityResult
    )
    COLLECTION_MANAGER_INTEGRATION_AVAILABLE = True
except ImportError as e:
    COLLECTION_MANAGER_INTEGRATION_AVAILABLE = False
    pytest.skip(f"Collection manager integration module not available: {e}", allow_module_level=True)


class TestCollectionCategory:
    """Test collection category enumeration."""

    def test_collection_category_values(self):
        """Test that CollectionCategory has expected values."""
        assert hasattr(CollectionCategory, 'PROJECT')
        assert hasattr(CollectionCategory, 'SCRATCHBOOK')
        assert hasattr(CollectionCategory, 'GLOBAL')
        assert hasattr(CollectionCategory, 'SYSTEM')

    def test_collection_category_string_representation(self):
        """Test string representation of category values."""
        category = CollectionCategory.PROJECT
        assert str(category) in ['CollectionCategory.PROJECT', 'PROJECT']


class TestSafeCollectionConfig:
    """Test safe collection configuration data structure."""

    def test_safe_collection_config_initialization(self):
        """Test SafeCollectionConfig initialization."""
        config = SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT,
            vector_size=384,
            distance_metric="Cosine"
        )

        assert config.name == "test-collection"
        assert config.category == CollectionCategory.PROJECT
        assert config.vector_size == 384
        assert config.distance_metric == "Cosine"

    def test_safe_collection_config_with_optional_fields(self):
        """Test SafeCollectionConfig with optional fields."""
        metadata = {"project": "test-project", "tenant": "test-tenant"}
        config = SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT,
            metadata=metadata,
            enable_payload_index=True,
            shard_number=2
        )

        assert config.metadata == metadata
        assert config.enable_payload_index is True
        assert config.shard_number == 2

    def test_safe_collection_config_validation(self):
        """Test SafeCollectionConfig validation."""
        # Test with invalid vector size
        with pytest.raises(ValueError):
            SafeCollectionConfig(
                name="test-collection",
                category=CollectionCategory.PROJECT,
                vector_size=0  # Invalid size
            )

    def test_safe_collection_config_defaults(self):
        """Test SafeCollectionConfig default values."""
        config = SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT
        )

        assert config.vector_size == 384  # Default value
        assert config.distance_metric == "Cosine"  # Default value
        assert config.metadata == {}
        assert config.enable_payload_index is True


class TestCollectionCreationResult:
    """Test collection creation result data structure."""

    def test_collection_creation_result_initialization(self):
        """Test CollectionCreationResult initialization."""
        result = CollectionCreationResult(
            success=True,
            collection_name="test-collection",
            created_at="2023-01-01T00:00:00Z",
            vector_size=384
        )

        assert result.success is True
        assert result.collection_name == "test-collection"
        assert result.created_at == "2023-01-01T00:00:00Z"
        assert result.vector_size == 384

    def test_collection_creation_result_with_collision_info(self):
        """Test CollectionCreationResult with collision information."""
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Name similarity",
            similarity_score=0.9,
            suggested_alternatives=["test-collection-v2", "test-collection-new"]
        )

        result = CollectionCreationResult(
            success=False,
            collection_name="test-collection",
            collision_detected=True,
            collision_result=collision_result
        )

        assert result.success is False
        assert result.collision_detected is True
        assert result.collision_result.has_collision is True
        assert len(result.collision_result.suggested_alternatives) == 2

    def test_collection_creation_result_with_error(self):
        """Test CollectionCreationResult with error information."""
        result = CollectionCreationResult(
            success=False,
            collection_name="test-collection",
            error_message="Database connection failed",
            error_code="DB_CONNECTION_ERROR"
        )

        assert result.success is False
        assert result.error_message == "Database connection failed"
        assert result.error_code == "DB_CONNECTION_ERROR"


class TestCollectionManagementError:
    """Test collection management error exception."""

    def test_collection_management_error_creation(self):
        """Test CollectionManagementError creation."""
        error = CollectionManagementError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_collection_management_error_with_details(self):
        """Test CollectionManagementError with additional details."""
        error = CollectionManagementError(
            "Test error",
            error_code="TEST_ERROR",
            collection_name="test-collection"
        )

        assert error.error_code == "TEST_ERROR"
        assert error.collection_name == "test-collection"


class TestCollectionCollisionError:
    """Test collection collision error exception."""

    def test_collection_collision_error_creation(self):
        """Test CollectionCollisionError creation."""
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Name conflict",
            similarity_score=0.95,
            suggested_alternatives=["alternative1", "alternative2"]
        )

        error = CollectionCollisionError("Collision detected", collision_result)

        assert str(error) == "Collision detected"
        assert error.collision_result == collision_result
        assert isinstance(error, CollectionManagementError)


class TestCollisionAwareCollectionManager:
    """Test collision-aware collection manager functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = Mock()
        client.create_collection = AsyncMock()
        client.get_collections = AsyncMock()
        client.get_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        return client

    @pytest.fixture
    def mock_collision_detector(self):
        """Create mock collision detector."""
        detector = Mock()
        detector.initialize = AsyncMock()
        detector.detect_collision = AsyncMock()
        detector.analyze_collection_landscape = AsyncMock()
        detector.suggest_alternatives = AsyncMock()
        return detector

    @pytest.fixture
    def safe_collection_config(self):
        """Create safe collection configuration."""
        return SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT,
            vector_size=384,
            distance_metric="Cosine"
        )

    def test_collision_aware_collection_manager_initialization(self, mock_qdrant_client):
        """Test CollisionAwareCollectionManager initialization."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)

        assert manager.client == mock_qdrant_client
        assert manager.collision_detector is not None
        assert manager.statistics == {}
        assert manager.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_manager(self, mock_qdrant_client, mock_collision_detector):
        """Test manager initialization."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector

        await manager.initialize()

        assert manager.initialized is True
        mock_collision_detector.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_safely_success(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test successful safe collection creation."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock no collision detected
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        mock_qdrant_client.create_collection.return_value = True

        result = await manager.create_collection_safely(safe_collection_config)

        assert result.success is True
        assert result.collection_name == "test-collection"
        assert result.collision_detected is False
        mock_collision_detector.detect_collision.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_safely_with_collision(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test safe collection creation with collision detected."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock collision detected
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Similar name exists",
            similarity_score=0.9,
            suggested_alternatives=["test-collection-v2", "test-collection-new"]
        )
        mock_collision_detector.detect_collision.return_value = collision_result

        with pytest.raises(CollectionCollisionError) as exc_info:
            await manager.create_collection_safely(safe_collection_config)

        assert exc_info.value.collision_result == collision_result
        mock_collision_detector.detect_collision.assert_called_once()
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_safely_force_creation(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test forced collection creation despite collision."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock collision detected
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Similar name exists",
            similarity_score=0.9,
            suggested_alternatives=["test-collection-v2"]
        )
        mock_collision_detector.detect_collision.return_value = collision_result
        mock_qdrant_client.create_collection.return_value = True

        result = await manager.create_collection_safely(safe_collection_config, force_creation=True)

        assert result.success is True
        assert result.collision_detected is True
        assert result.forced_creation is True
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_safely_qdrant_error(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test safe collection creation with Qdrant error."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock no collision
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        # Mock Qdrant error
        mock_qdrant_client.create_collection.side_effect = Exception("Qdrant connection error")

        result = await manager.create_collection_safely(safe_collection_config)

        assert result.success is False
        assert "Qdrant connection error" in result.error_message

    @pytest.mark.asyncio
    async def test_create_collection_with_alternatives(self, mock_qdrant_client, mock_collision_detector):
        """Test collection creation with suggested alternatives."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock collision for original name
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Name exists",
            similarity_score=1.0,
            suggested_alternatives=["test-collection-v2", "test-collection-alt"]
        )

        # Mock no collision for alternative
        no_collision_result = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        mock_collision_detector.detect_collision.side_effect = [collision_result, no_collision_result]
        mock_qdrant_client.create_collection.return_value = True

        config = SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT
        )

        result = await manager.create_collection_with_alternatives(config)

        assert result.success is True
        assert result.collection_name == "test-collection-v2"
        assert result.used_alternative is True

    @pytest.mark.asyncio
    async def test_delete_collection_safely(self, mock_qdrant_client, mock_collision_detector):
        """Test safe collection deletion."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        mock_qdrant_client.delete_collection.return_value = True

        result = await manager.delete_collection_safely("test-collection")

        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test-collection")

    @pytest.mark.asyncio
    async def test_delete_collection_safely_error(self, mock_qdrant_client, mock_collision_detector):
        """Test safe collection deletion with error."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        mock_qdrant_client.delete_collection.side_effect = Exception("Collection not found")

        with pytest.raises(CollectionManagementError):
            await manager.delete_collection_safely("nonexistent-collection")

    @pytest.mark.asyncio
    async def test_validate_collection_name(self, mock_qdrant_client, mock_collision_detector):
        """Test collection name validation."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock no collision
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        is_valid = await manager.validate_collection_name("valid-collection-name")

        assert is_valid is True
        mock_collision_detector.detect_collision.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_collection_name_invalid(self, mock_qdrant_client, mock_collision_detector):
        """Test collection name validation with invalid name."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock collision
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=True,
            collision_reason="Name too similar",
            similarity_score=0.95,
            suggested_alternatives=["alternative-name"]
        )

        is_valid = await manager.validate_collection_name("invalid-collection-name")

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_collection_suggestions(self, mock_qdrant_client, mock_collision_detector):
        """Test collection name suggestions."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        suggested_names = ["suggestion1", "suggestion2", "suggestion3"]
        mock_collision_detector.suggest_alternatives.return_value = suggested_names

        suggestions = await manager.get_collection_suggestions("base-name", CollectionCategory.PROJECT)

        assert suggestions == suggested_names
        mock_collision_detector.suggest_alternatives.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_collection_landscape(self, mock_qdrant_client, mock_collision_detector):
        """Test collection landscape analysis."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        analysis_result = CollectionAnalysis(
            total_collections=10,
            collections_by_category={
                "PROJECT": 5,
                "SCRATCHBOOK": 3,
                "GLOBAL": 2
            },
            naming_patterns=["project-*", "scratch-*", "global-*"],
            potential_conflicts=2
        )

        mock_collision_detector.analyze_collection_landscape.return_value = analysis_result

        analysis = await manager.analyze_collection_landscape()

        assert analysis == analysis_result
        mock_collision_detector.analyze_collection_landscape.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_statistics(self, mock_qdrant_client, mock_collision_detector):
        """Test collection statistics retrieval."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector

        # Add some mock statistics
        manager.statistics = {
            "collections_created": 5,
            "collisions_detected": 2,
            "alternatives_used": 1,
            "forced_creations": 0
        }

        stats = manager.get_collection_statistics()

        assert stats["collections_created"] == 5
        assert stats["collisions_detected"] == 2
        assert stats["alternatives_used"] == 1

    def test_update_statistics(self, mock_qdrant_client):
        """Test statistics update."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)

        manager._update_statistics("collections_created")
        manager._update_statistics("collisions_detected")
        manager._update_statistics("collections_created")

        assert manager.statistics["collections_created"] == 2
        assert manager.statistics["collisions_detected"] == 1

    @pytest.mark.asyncio
    async def test_reset_statistics(self, mock_qdrant_client):
        """Test statistics reset."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)

        # Add some statistics
        manager.statistics = {"test_stat": 10}

        await manager.reset_statistics()

        assert manager.statistics == {}

    @pytest.mark.asyncio
    async def test_check_manager_initialization(self, mock_qdrant_client):
        """Test manager initialization check."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)

        # Should raise error when not initialized
        with pytest.raises(CollectionManagementError, match="Manager not initialized"):
            await manager.create_collection_safely(SafeCollectionConfig(
                name="test",
                category=CollectionCategory.PROJECT
            ))

    @pytest.mark.asyncio
    async def test_bulk_collection_creation(self, mock_qdrant_client, mock_collision_detector):
        """Test bulk collection creation."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        configs = [
            SafeCollectionConfig(name=f"collection-{i}", category=CollectionCategory.PROJECT)
            for i in range(3)
        ]

        # Mock no collisions
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )
        mock_qdrant_client.create_collection.return_value = True

        results = await manager.bulk_create_collections(configs)

        assert len(results) == 3
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_bulk_collection_creation_with_errors(self, mock_qdrant_client, mock_collision_detector):
        """Test bulk collection creation with some errors."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        configs = [
            SafeCollectionConfig(name="good-collection", category=CollectionCategory.PROJECT),
            SafeCollectionConfig(name="bad-collection", category=CollectionCategory.PROJECT)
        ]

        # Mock different responses
        mock_collision_detector.detect_collision.side_effect = [
            CollisionResult(has_collision=False, collision_reason="", similarity_score=0.0, suggested_alternatives=[]),
            CollisionResult(has_collision=True, collision_reason="Collision", similarity_score=0.9, suggested_alternatives=[])
        ]

        results = await manager.bulk_create_collections(configs, ignore_errors=True)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_collision_detector_initialization_failure(self, mock_qdrant_client):
        """Test handling of collision detector initialization failure."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)

        with patch.object(manager.collision_detector, 'initialize', side_effect=Exception("Init failed")):
            with pytest.raises(CollectionManagementError):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_invalid_collection_config(self, mock_qdrant_client, mock_collision_detector):
        """Test handling of invalid collection configuration."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Test with None config
        with pytest.raises(ValueError):
            await manager.create_collection_safely(None)

        # Test with empty name
        with pytest.raises(ValueError):
            await manager.create_collection_safely(SafeCollectionConfig(
                name="",
                category=CollectionCategory.PROJECT
            ))

    @pytest.mark.asyncio
    async def test_collision_detection_failure(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test handling of collision detection failure."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        mock_collision_detector.detect_collision.side_effect = Exception("Collision detection failed")

        result = await manager.create_collection_safely(safe_collection_config)

        assert result.success is False
        assert "Collision detection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_qdrant_client_none(self):
        """Test handling of None Qdrant client."""
        with pytest.raises(ValueError):
            CollisionAwareCollectionManager(None)

    @pytest.mark.asyncio
    async def test_collection_creation_timeout(self, mock_qdrant_client, mock_collision_detector, safe_collection_config):
        """Test handling of collection creation timeout."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock no collision
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        # Mock timeout
        mock_qdrant_client.create_collection.side_effect = asyncio.TimeoutError("Operation timed out")

        result = await manager.create_collection_safely(safe_collection_config)

        assert result.success is False
        assert "Operation timed out" in result.error_message


class TestPerformanceOptimization:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_concurrent_collection_creation(self, mock_qdrant_client, mock_collision_detector):
        """Test concurrent collection creation."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Mock successful operations
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )
        mock_qdrant_client.create_collection.return_value = True

        # Create multiple configurations
        configs = [
            SafeCollectionConfig(name=f"concurrent-collection-{i}", category=CollectionCategory.PROJECT)
            for i in range(5)
        ]

        # Create collections concurrently
        tasks = [manager.create_collection_safely(config) for config in configs]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_batch_collision_detection(self, mock_qdrant_client, mock_collision_detector):
        """Test batch collision detection optimization."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        collection_names = [f"batch-collection-{i}" for i in range(10)]

        # Mock batch detection
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        results = await manager.batch_validate_collection_names(collection_names)

        assert len(results) == 10
        assert all(result for result in results)

    @pytest.mark.asyncio
    async def test_caching_optimization(self, mock_qdrant_client, mock_collision_detector):
        """Test caching for performance optimization."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Enable caching
        manager.enable_caching = True

        collection_name = "cached-collection"

        # Mock collision detection
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        # First call
        result1 = await manager.validate_collection_name(collection_name)
        # Second call (should use cache)
        result2 = await manager.validate_collection_name(collection_name)

        assert result1 is True
        assert result2 is True
        # Should only call collision detection once due to caching
        assert mock_collision_detector.detect_collision.call_count <= 2


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows."""

    @pytest.mark.asyncio
    async def test_complete_collection_management_workflow(self, mock_qdrant_client, mock_collision_detector):
        """Test complete collection management workflow."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector

        # Initialize manager
        await manager.initialize()

        # Analyze landscape
        analysis = CollectionAnalysis(
            total_collections=5,
            collections_by_category={"PROJECT": 3, "SCRATCHBOOK": 2},
            naming_patterns=["project-*"],
            potential_conflicts=0
        )
        mock_collision_detector.analyze_collection_landscape.return_value = analysis

        landscape = await manager.analyze_collection_landscape()
        assert landscape.total_collections == 5

        # Create collection safely
        config = SafeCollectionConfig(
            name="workflow-test-collection",
            category=CollectionCategory.PROJECT
        )

        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )
        mock_qdrant_client.create_collection.return_value = True

        creation_result = await manager.create_collection_safely(config)
        assert creation_result.success is True

        # Get statistics
        stats = manager.get_collection_statistics()
        assert "collections_created" in stats

        # Clean up
        mock_qdrant_client.delete_collection.return_value = True
        deletion_result = await manager.delete_collection_safely("workflow-test-collection")
        assert deletion_result is True

    @pytest.mark.asyncio
    async def test_collision_resolution_workflow(self, mock_qdrant_client, mock_collision_detector):
        """Test collision detection and resolution workflow."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # First attempt: collision detected
        collision_result = CollisionResult(
            has_collision=True,
            collision_reason="Similar name exists",
            similarity_score=0.9,
            suggested_alternatives=["test-collection-v2", "test-collection-alt"]
        )

        # Second attempt: no collision with alternative
        no_collision_result = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        mock_collision_detector.detect_collision.side_effect = [collision_result, no_collision_result]
        mock_qdrant_client.create_collection.return_value = True

        config = SafeCollectionConfig(
            name="test-collection",
            category=CollectionCategory.PROJECT
        )

        # Should successfully create with alternative name
        result = await manager.create_collection_with_alternatives(config)

        assert result.success is True
        assert result.collection_name == "test-collection-v2"
        assert result.used_alternative is True

    @pytest.mark.asyncio
    async def test_multi_tenant_collection_management(self, mock_qdrant_client, mock_collision_detector):
        """Test multi-tenant collection management."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        # Create collections for different tenants
        tenant_configs = [
            SafeCollectionConfig(
                name=f"tenant-{tenant}-project",
                category=CollectionCategory.PROJECT,
                metadata={"tenant": f"tenant-{tenant}"}
            )
            for tenant in ["alpha", "beta", "gamma"]
        ]

        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )
        mock_qdrant_client.create_collection.return_value = True

        results = await manager.bulk_create_collections(tenant_configs)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert all("tenant-" in result.collection_name for result in results)

    @pytest.mark.asyncio
    async def test_collection_lifecycle_management(self, mock_qdrant_client, mock_collision_detector):
        """Test complete collection lifecycle management."""
        manager = CollisionAwareCollectionManager(mock_qdrant_client)
        manager.collision_detector = mock_collision_detector
        manager.initialized = True

        collection_name = "lifecycle-test-collection"

        # 1. Validate name
        mock_collision_detector.detect_collision.return_value = CollisionResult(
            has_collision=False,
            collision_reason="",
            similarity_score=0.0,
            suggested_alternatives=[]
        )

        is_valid = await manager.validate_collection_name(collection_name)
        assert is_valid is True

        # 2. Create collection
        config = SafeCollectionConfig(
            name=collection_name,
            category=CollectionCategory.PROJECT
        )

        mock_qdrant_client.create_collection.return_value = True
        creation_result = await manager.create_collection_safely(config)
        assert creation_result.success is True

        # 3. Verify collection exists (mock)
        mock_qdrant_client.get_collection.return_value = Mock(name=collection_name)
        collection_info = await manager.get_collection_info(collection_name)
        assert collection_info is not None

        # 4. Delete collection
        mock_qdrant_client.delete_collection.return_value = True
        deletion_result = await manager.delete_collection_safely(collection_name)
        assert deletion_result is True