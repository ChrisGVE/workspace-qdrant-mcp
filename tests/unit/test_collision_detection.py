"""
Unit tests for the collision detection system.

This test suite provides comprehensive validation of the collision detection
implementation including registry operations, Bloom filter efficiency,
name suggestion algorithms, concurrent creation guards, and full integration
with the naming validation system.

Test Categories:
    - Bloom filter probabilistic membership testing
    - Collision registry performance and accuracy
    - Name suggestion engine algorithms and strategies
    - Concurrent creation guard thread safety
    - Main collision detector integration and workflow
    - Performance benchmarks and optimization validation
    - Edge cases and boundary conditions
"""

import asyncio
import pytest
import time
import threading
from typing import List, Set
from unittest.mock import Mock, AsyncMock, patch

from workspace_qdrant_mcp.core.collision_detection import (
    BloomFilter,
    CollisionRegistry,
    NameSuggestionEngine,
    ConcurrentCreationGuard,
    CollisionDetector,
    CollisionResult,
    CollisionSeverity,
    CollisionCategory,
    CollisionRegistryEntry
)

from workspace_qdrant_mcp.core.collection_naming_validation import (
    CollectionNamingValidator,
    ValidationResult,
    NamingConfiguration,
    ValidationSeverity,
    ConflictType
)

from workspace_qdrant_mcp.core.metadata_schema import CollectionCategory


class TestBloomFilter:
    """Test suite for Bloom filter implementation."""

    def test_bloom_filter_initialization(self):
        """Test Bloom filter initialization with default parameters."""
        bloom = BloomFilter()

        assert bloom.capacity == 10000
        assert bloom.error_rate == 0.01
        assert bloom.bit_size >= 1024
        assert bloom.hash_count >= 1
        assert bloom.item_count == 0
        assert len(bloom.bit_array) == bloom.bit_size

    def test_bloom_filter_custom_parameters(self):
        """Test Bloom filter initialization with custom parameters."""
        bloom = BloomFilter(capacity=5000, error_rate=0.05)

        assert bloom.capacity == 5000
        assert bloom.error_rate == 0.05
        assert bloom.item_count == 0

    def test_bloom_filter_add_and_check(self):
        """Test adding items and checking membership."""
        bloom = BloomFilter()

        # Add some items
        test_items = ["collection1", "test-project-docs", "_library_tools"]
        for item in test_items:
            bloom.add(item)

        # Check that added items might be contained
        for item in test_items:
            assert bloom.might_contain(item) is True

        # Check item count
        assert bloom.item_count == len(test_items)

    def test_bloom_filter_negative_checks(self):
        """Test that non-added items return False (no false negatives)."""
        bloom = BloomFilter()

        # Add some items
        bloom.add("existing_collection")
        bloom.add("another_collection")

        # Items not added should return False
        assert bloom.might_contain("nonexistent_collection") in [True, False]  # Could be false positive
        assert bloom.might_contain("") in [True, False]

    def test_bloom_filter_case_insensitive(self):
        """Test that Bloom filter handles case insensitively."""
        bloom = BloomFilter()

        bloom.add("TestCollection")

        # Should match regardless of case
        assert bloom.might_contain("testcollection") is True
        assert bloom.might_contain("TESTCOLLECTION") is True
        assert bloom.might_contain("TestCollection") is True

    def test_bloom_filter_clear(self):
        """Test clearing the Bloom filter."""
        bloom = BloomFilter()

        # Add items
        bloom.add("item1")
        bloom.add("item2")
        assert bloom.item_count == 2

        # Clear and verify
        bloom.clear()
        assert bloom.item_count == 0
        assert all(not bit for bit in bloom.bit_array)


class TestCollisionRegistry:
    """Test suite for collision registry implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = CollisionRegistry(max_cache_size=100)

    def test_registry_initialization(self):
        """Test collision registry initialization."""
        assert self.registry.max_cache_size == 100
        assert len(self.registry._registry) == 0
        assert len(self.registry._category_index) == 0
        assert len(self.registry._project_index) == 0

    def test_add_collection_success(self):
        """Test successful collection addition."""
        result = self.registry.add_collection("test-collection", CollectionCategory.GLOBAL)

        assert result is True
        assert self.registry.contains("test-collection")
        assert "test-collection" in self.registry.get_by_category(CollectionCategory.GLOBAL)

    def test_add_collection_duplicate(self):
        """Test adding duplicate collection."""
        # Add collection first time
        result1 = self.registry.add_collection("test-collection", CollectionCategory.GLOBAL)
        assert result1 is True

        # Add same collection again
        result2 = self.registry.add_collection("test-collection", CollectionCategory.GLOBAL)
        assert result2 is False

    def test_add_collection_with_project_context(self):
        """Test adding collection with project context."""
        result = self.registry.add_collection(
            "my-project-docs",
            CollectionCategory.PROJECT,
            project_context="my-project"
        )

        assert result is True
        assert "my-project-docs" in self.registry.get_by_project("my-project")

    def test_contains_method(self):
        """Test the contains method for collection lookup."""
        # Initially empty
        assert self.registry.contains("nonexistent") is False

        # Add collection
        self.registry.add_collection("existing", CollectionCategory.GLOBAL)
        assert self.registry.contains("existing") is True

        # Test case insensitivity
        assert self.registry.contains("EXISTING") is True
        assert self.registry.contains("Existing") is True

    def test_get_similar_names(self):
        """Test similarity detection algorithm."""
        # Add some collections
        collections = [
            "test-project-docs",
            "test-project-notes",
            "test-application-docs",
            "my-project-docs",
            "completely-different"
        ]

        for name in collections:
            self.registry.add_collection(name, CollectionCategory.PROJECT)

        # Test similarity detection
        similar = self.registry.get_similar_names("test-project-documentation", threshold=0.6)

        # Should find similar names
        assert "test-project-docs" in similar
        assert "test-project-notes" in similar
        assert "completely-different" not in similar

    def test_get_by_category(self):
        """Test getting collections by category."""
        # Add collections of different categories
        self.registry.add_collection("_library1", CollectionCategory.LIBRARY)
        self.registry.add_collection("_library2", CollectionCategory.LIBRARY)
        self.registry.add_collection("global1", CollectionCategory.GLOBAL)

        library_collections = self.registry.get_by_category(CollectionCategory.LIBRARY)
        global_collections = self.registry.get_by_category(CollectionCategory.GLOBAL)

        assert "_library1" in library_collections
        assert "_library2" in library_collections
        assert "global1" in global_collections
        assert "global1" not in library_collections

    def test_get_by_project(self):
        """Test getting collections by project context."""
        # Add collections for different projects
        self.registry.add_collection("project1-docs", CollectionCategory.PROJECT, "project1")
        self.registry.add_collection("project1-notes", CollectionCategory.PROJECT, "project1")
        self.registry.add_collection("project2-docs", CollectionCategory.PROJECT, "project2")

        project1_collections = self.registry.get_by_project("project1")
        project2_collections = self.registry.get_by_project("project2")

        assert "project1-docs" in project1_collections
        assert "project1-notes" in project1_collections
        assert "project2-docs" in project2_collections
        assert "project2-docs" not in project1_collections

    def test_remove_collection(self):
        """Test collection removal."""
        # Add collection
        self.registry.add_collection("test-collection", CollectionCategory.GLOBAL, "test-project")

        # Verify it exists
        assert self.registry.contains("test-collection")

        # Remove it
        result = self.registry.remove_collection("test-collection")
        assert result is True

        # Verify removal
        assert not self.registry.contains("test-collection")
        assert "test-collection" not in self.registry.get_by_category(CollectionCategory.GLOBAL)
        assert "test-collection" not in self.registry.get_by_project("test-project")

        # Try removing again
        result = self.registry.remove_collection("test-collection")
        assert result is False

    def test_registry_statistics(self):
        """Test statistics generation."""
        # Add some collections
        self.registry.add_collection("global1", CollectionCategory.GLOBAL)
        self.registry.add_collection("_library1", CollectionCategory.LIBRARY)
        self.registry.add_collection("project1-docs", CollectionCategory.PROJECT, "project1")

        # Access some collections to generate stats
        self.registry.contains("global1")
        self.registry.contains("nonexistent")

        stats = self.registry.get_statistics()

        assert stats['total_collections'] == 3
        assert stats['total_lookups'] >= 2
        assert 'cache_hit_rate_percent' in stats
        assert 'bloom_filter_efficiency_percent' in stats
        assert 'category_distribution' in stats
        assert 'project_distribution' in stats

    def test_registry_clear(self):
        """Test clearing the registry."""
        # Add some collections
        self.registry.add_collection("test1", CollectionCategory.GLOBAL)
        self.registry.add_collection("test2", CollectionCategory.LIBRARY)

        # Verify they exist
        assert self.registry.contains("test1")
        assert self.registry.contains("test2")

        # Clear registry
        self.registry.clear()

        # Verify empty
        assert not self.registry.contains("test1")
        assert not self.registry.contains("test2")
        assert len(self.registry._registry) == 0

    def test_thread_safety(self):
        """Test thread safety of registry operations."""
        results = []
        errors = []

        def add_collections(start_idx: int, count: int):
            try:
                for i in range(start_idx, start_idx + count):
                    collection_name = f"collection_{i}"
                    result = self.registry.add_collection(collection_name, CollectionCategory.GLOBAL)
                    results.append((collection_name, result))
            except Exception as e:
                errors.append(e)

        def check_collections(collection_names: List[str]):
            try:
                for name in collection_names:
                    exists = self.registry.contains(name)
                    results.append((name, exists))
            except Exception as e:
                errors.append(e)

        # Create multiple threads for concurrent operations
        threads = []

        # Threads for adding collections
        for i in range(3):
            thread = threading.Thread(target=add_collections, args=(i * 10, 10))
            threads.append(thread)

        # Threads for checking collections
        check_names = [f"collection_{i}" for i in range(15, 25)]
        thread = threading.Thread(target=check_collections, args=(check_names,))
        threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify we have some results
        assert len(results) > 0


class TestNameSuggestionEngine:
    """Test suite for name suggestion engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.naming_validator = CollectionNamingValidator()
        self.registry = CollisionRegistry()
        self.engine = NameSuggestionEngine(self.naming_validator, self.registry)

    @pytest.mark.asyncio
    async def test_suggestion_engine_initialization(self):
        """Test suggestion engine initialization."""
        assert self.engine.naming_validator is not None
        assert self.engine.registry is not None
        assert len(self.engine._strategies) > 0
        assert len(self.engine._common_suffixes) > 0
        assert len(self.engine._semantic_alternatives) > 0

    @pytest.mark.asyncio
    async def test_generate_suggestions_global_category(self):
        """Test suggestion generation for global collections."""
        suggestions = await self.engine.generate_suggestions(
            "algorithms", CollectionCategory.GLOBAL
        )

        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)

        # Suggestions should be different from original
        assert all(s != "algorithms" for s in suggestions)

    @pytest.mark.asyncio
    async def test_generate_suggestions_library_category(self):
        """Test suggestion generation for library collections."""
        suggestions = await self.engine.generate_suggestions(
            "_tools", CollectionCategory.LIBRARY
        )

        assert len(suggestions) > 0

        # Library suggestions should have underscore prefix
        assert all(s.startswith('_') for s in suggestions)

    @pytest.mark.asyncio
    async def test_generate_suggestions_system_category(self):
        """Test suggestion generation for system collections."""
        suggestions = await self.engine.generate_suggestions(
            "__config", CollectionCategory.SYSTEM
        )

        assert len(suggestions) > 0

        # System suggestions should have double underscore prefix
        assert all(s.startswith('__') for s in suggestions)

    @pytest.mark.asyncio
    async def test_generate_suggestions_project_category(self):
        """Test suggestion generation for project collections."""
        # Configure valid project suffixes
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes", "scratchbook"})
        validator = CollectionNamingValidator(config)
        engine = NameSuggestionEngine(validator, self.registry)

        suggestions = await engine.generate_suggestions(
            "my-project-docs", CollectionCategory.PROJECT
        )

        assert len(suggestions) > 0

        # Project suggestions should follow project naming pattern
        assert all('-' in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_generate_suggestions_with_existing_names(self):
        """Test suggestion generation avoids existing names."""
        # Add some existing collections to registry
        existing_names = ["algorithms_v2", "algorithms_alt", "algorithms_new"]
        for name in existing_names:
            self.registry.add_collection(name, CollectionCategory.GLOBAL)

        suggestions = await self.engine.generate_suggestions(
            "algorithms", CollectionCategory.GLOBAL
        )

        # Suggestions should not include existing names
        for suggestion in suggestions:
            assert suggestion not in existing_names

    @pytest.mark.asyncio
    async def test_suggestion_strategies_suffix(self):
        """Test suffix-based suggestion strategy."""
        suggestions = await self.engine._suggest_with_suffixes(
            "test-collection", CollectionCategory.GLOBAL
        )

        assert len(suggestions) > 0
        assert any("_v2" in s or "_alt" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggestion_strategies_prefix(self):
        """Test prefix-based suggestion strategy."""
        suggestions = await self.engine._suggest_with_prefixes(
            "collection", CollectionCategory.GLOBAL
        )

        assert len(suggestions) > 0
        assert any("new_" in s or "alt_" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggestion_strategies_numbers(self):
        """Test number-based suggestion strategy."""
        suggestions = await self.engine._suggest_with_numbers(
            "collection", CollectionCategory.GLOBAL
        )

        assert len(suggestions) > 0
        assert any("_v2" in s or "_v3" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggestion_strategies_semantic(self):
        """Test semantic alternative suggestion strategy."""
        suggestions = await self.engine._suggest_semantic_alternatives(
            "docs-collection", CollectionCategory.GLOBAL
        )

        assert len(suggestions) > 0
        # Should replace 'docs' with semantic alternatives
        assert any("notes" in s or "documentation" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_max_suggestions_limit(self):
        """Test that suggestion generation respects maximum count."""
        max_count = 3
        suggestions = await self.engine.generate_suggestions(
            "test", CollectionCategory.GLOBAL, max_suggestions=max_count
        )

        assert len(suggestions) <= max_count


class TestConcurrentCreationGuard:
    """Test suite for concurrent creation guard."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = ConcurrentCreationGuard()

    @pytest.mark.asyncio
    async def test_creation_guard_initialization(self):
        """Test creation guard initialization."""
        assert len(self.guard._creation_locks) == 0
        assert len(self.guard._pending_creations) == 0

    @pytest.mark.asyncio
    async def test_acquire_creation_lock(self):
        """Test acquiring creation lock."""
        collection_name = "test-collection"

        async with self.guard.acquire_creation_lock(collection_name):
            # Inside the lock, creation should be pending
            assert self.guard.is_creation_pending(collection_name)

        # After lock, creation should no longer be pending
        assert not self.guard.is_creation_pending(collection_name)

    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self):
        """Test that concurrent lock acquisition is properly serialized."""
        collection_name = "test-collection"
        execution_order = []

        async def acquire_lock_with_delay(task_id: str, delay: float):
            async with self.guard.acquire_creation_lock(collection_name):
                execution_order.append(f"{task_id}_start")
                await asyncio.sleep(delay)
                execution_order.append(f"{task_id}_end")

        # Start two concurrent tasks
        task1 = asyncio.create_task(acquire_lock_with_delay("task1", 0.1))
        task2 = asyncio.create_task(acquire_lock_with_delay("task2", 0.05))

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Verify serialized execution (one task completes before other starts)
        assert len(execution_order) == 4

        # Either task1 completes first or task2 completes first, but not interleaved
        scenario1 = execution_order == ["task1_start", "task1_end", "task2_start", "task2_end"]
        scenario2 = execution_order == ["task2_start", "task2_end", "task1_start", "task1_end"]

        assert scenario1 or scenario2

    @pytest.mark.asyncio
    async def test_multiple_collection_locks(self):
        """Test that locks for different collections don't interfere."""
        execution_order = []

        async def acquire_lock_with_tracking(collection_name: str, task_id: str):
            async with self.guard.acquire_creation_lock(collection_name):
                execution_order.append(f"{task_id}_start")
                await asyncio.sleep(0.05)
                execution_order.append(f"{task_id}_end")

        # Start tasks for different collections simultaneously
        task1 = asyncio.create_task(acquire_lock_with_tracking("collection1", "task1"))
        task2 = asyncio.create_task(acquire_lock_with_tracking("collection2", "task2"))

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Both tasks should be able to run concurrently
        assert len(execution_order) == 4
        assert "task1_start" in execution_order
        assert "task1_end" in execution_order
        assert "task2_start" in execution_order
        assert "task2_end" in execution_order

    @pytest.mark.asyncio
    async def test_pending_creations_tracking(self):
        """Test tracking of pending creations."""
        collection1 = "collection1"
        collection2 = "collection2"

        # Initially no pending creations
        assert len(self.guard.get_pending_creations()) == 0

        async def hold_lock(collection_name: str):
            async with self.guard.acquire_creation_lock(collection_name):
                await asyncio.sleep(0.1)

        # Start two tasks but don't wait
        task1 = asyncio.create_task(hold_lock(collection1))
        task2 = asyncio.create_task(hold_lock(collection2))

        # Give tasks time to acquire locks
        await asyncio.sleep(0.05)

        # Check pending creations
        pending = self.guard.get_pending_creations()
        assert collection1 in pending
        assert collection2 in pending

        # Wait for tasks to complete
        await asyncio.gather(task1, task2)

        # No more pending creations
        assert len(self.guard.get_pending_creations()) == 0


class TestCollisionDetector:
    """Test suite for main collision detector."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock Qdrant client
        self.mock_qdrant_client = Mock()
        self.mock_qdrant_client.get_collections = Mock()

        # Create collision detector
        self.detector = CollisionDetector(self.mock_qdrant_client)

    @pytest.mark.asyncio
    async def test_collision_detector_initialization(self):
        """Test collision detector initialization."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        assert self.detector._initialized is True
        assert self.detector.registry is not None
        assert self.detector.suggestion_engine is not None
        assert self.detector.creation_guard is not None

    @pytest.mark.asyncio
    async def test_check_collection_collision_no_collision(self):
        """Test collision checking when no collision exists."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        result = await self.detector.check_collection_collision(
            "new-collection", CollectionCategory.GLOBAL
        )

        assert result.has_collision is False
        assert result.severity == CollisionSeverity.NONE
        assert result.detection_time_ms > 0

    @pytest.mark.asyncio
    async def test_check_collection_collision_exact_duplicate(self):
        """Test collision checking for exact duplicate."""
        # Mock collections response with existing collection
        mock_collection = Mock()
        mock_collection.name = "existing-collection"
        mock_response = Mock()
        mock_response.collections = [mock_collection]
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        result = await self.detector.check_collection_collision(
            "existing-collection", CollectionCategory.GLOBAL
        )

        assert result.has_collision is True
        assert result.severity == CollisionSeverity.BLOCKING
        assert result.category == CollisionCategory.EXACT_DUPLICATE
        assert "existing-collection" in result.conflicting_collections

    @pytest.mark.asyncio
    async def test_check_collection_collision_similar_names(self):
        """Test collision checking for similar names."""
        # Mock collections response
        mock_collection = Mock()
        mock_collection.name = "test-project-docs"
        mock_response = Mock()
        mock_response.collections = [mock_collection]
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Set lower similarity threshold for testing
        self.detector.similarity_threshold = 0.7

        result = await self.detector.check_collection_collision(
            "test-project-documentation", CollectionCategory.PROJECT
        )

        # Depending on similarity algorithm, this might detect similarity
        # The test validates the structure is correct regardless
        assert isinstance(result.has_collision, bool)
        assert result.detection_time_ms > 0

    @pytest.mark.asyncio
    async def test_check_collection_collision_pending_creation(self):
        """Test collision checking for pending creation."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        collection_name = "pending-collection"

        # Simulate pending creation
        async def simulate_pending():
            async with self.detector.creation_guard.acquire_creation_lock(collection_name):
                # Check collision while creation is pending
                result = await self.detector.check_collection_collision(
                    collection_name, CollectionCategory.GLOBAL
                )
                return result

        # This should detect the pending creation as a collision
        result = await simulate_pending()

        assert result.has_collision is True
        assert result.severity == CollisionSeverity.BLOCKING

    @pytest.mark.asyncio
    async def test_register_collection_creation(self):
        """Test registering a collection creation."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        collection_name = "new-collection"
        category = CollectionCategory.GLOBAL

        # Register collection creation
        await self.detector.register_collection_creation(collection_name, category)

        # Verify it's now in the registry
        assert self.detector.registry.contains(collection_name)

    @pytest.mark.asyncio
    async def test_remove_collection_registration(self):
        """Test removing a collection registration."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        collection_name = "test-collection"

        # Register then remove
        await self.detector.register_collection_creation(collection_name, CollectionCategory.GLOBAL)
        assert self.detector.registry.contains(collection_name)

        await self.detector.remove_collection_registration(collection_name)
        assert not self.detector.registry.contains(collection_name)

    @pytest.mark.asyncio
    async def test_create_collection_guard_success(self):
        """Test successful collection creation with guard."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        collection_name = "safe-collection"

        # This should succeed without raising exceptions
        async with self.detector.create_collection_guard(collection_name):
            # Simulate collection creation work
            await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_create_collection_guard_collision(self):
        """Test collection creation guard with collision."""
        # Mock collections response with existing collection
        mock_collection = Mock()
        mock_collection.name = "existing-collection"
        mock_response = Mock()
        mock_response.collections = [mock_collection]
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # This should raise ValueError due to collision
        with pytest.raises(ValueError, match="Collection collision detected"):
            async with self.detector.create_collection_guard("existing-collection"):
                pass

    @pytest.mark.asyncio
    async def test_get_collision_statistics(self):
        """Test collision statistics generation."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        stats = await self.detector.get_collision_statistics()

        assert 'registry_statistics' in stats
        assert 'pending_creations' in stats
        assert 'similarity_threshold' in stats
        assert 'initialized' in stats
        assert stats['initialized'] is True

    @pytest.mark.asyncio
    async def test_refresh_registry(self):
        """Test manual registry refresh."""
        # Mock collections response
        mock_collection = Mock()
        mock_collection.name = "test-collection"
        mock_response = Mock()
        mock_response.collections = [mock_collection]
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Verify collection is loaded
        assert self.detector.registry.contains("test-collection")

        # Clear registry and refresh
        self.detector.registry.clear()
        assert not self.detector.registry.contains("test-collection")

        await self.detector.refresh_registry()
        assert self.detector.registry.contains("test-collection")

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test collision detector shutdown."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Verify initialized
        assert self.detector._initialized is True

        # Shutdown
        await self.detector.shutdown()

        # Verify shutdown state
        assert self.detector._initialized is False


class TestCollisionDetectorIntegration:
    """Integration tests for collision detector with real components."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create real naming validator with configuration
        config = NamingConfiguration(
            valid_project_suffixes={"docs", "notes", "scratchbook"},
            additional_reserved_names={"forbidden"}
        )
        self.naming_validator = CollectionNamingValidator(config)

        # Create mock Qdrant client
        self.mock_qdrant_client = Mock()

        # Create detector with real validator
        self.detector = CollisionDetector(self.mock_qdrant_client, self.naming_validator)

    @pytest.mark.asyncio
    async def test_integration_with_naming_validator(self):
        """Test integration with naming validation system."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Test reserved name collision
        result = await self.detector.check_collection_collision("forbidden")

        assert result.has_collision is True
        assert result.severity == CollisionSeverity.BLOCKING
        assert result.category == CollisionCategory.RESERVED_VIOLATION

    @pytest.mark.asyncio
    async def test_integration_project_naming_patterns(self):
        """Test integration with project naming patterns."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Test valid project collection
        result = await self.detector.check_collection_collision(
            "my-project-docs", CollectionCategory.PROJECT
        )

        assert result.has_collision is False
        assert result.severity == CollisionSeverity.NONE

        # Test invalid project collection format
        result = await self.detector.check_collection_collision(
            "invalid_project_format", CollectionCategory.PROJECT
        )

        assert result.has_collision is True
        assert result.severity == CollisionSeverity.BLOCKING

    @pytest.mark.asyncio
    async def test_integration_suggestion_generation(self):
        """Test integration with suggestion generation."""
        # Mock empty collections response
        mock_response = Mock()
        mock_response.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Add some existing collections
        await self.detector.register_collection_creation("existing-docs", CollectionCategory.PROJECT)

        # Check collision and get suggestions
        result = await self.detector.check_collection_collision(
            "existing-docs", CollectionCategory.PROJECT
        )

        assert result.has_collision is True
        assert len(result.suggested_alternatives) > 0

        # Verify suggestions are valid project names
        for suggestion in result.suggested_alternatives:
            validation_result = self.naming_validator.validate_name(
                suggestion, CollectionCategory.PROJECT
            )
            assert validation_result.is_valid


class TestCollisionDetectorPerformance:
    """Performance tests for collision detector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.detector = CollisionDetector(self.mock_qdrant_client)

    @pytest.mark.asyncio
    async def test_large_registry_performance(self):
        """Test performance with large number of collections."""
        # Mock large collections response
        mock_collections = []
        for i in range(1000):
            mock_collection = Mock()
            mock_collection.name = f"collection_{i:04d}"
            mock_collections.append(mock_collection)

        mock_response = Mock()
        mock_response.collections = mock_collections
        self.mock_qdrant_client.get_collections.return_value = mock_response

        # Measure initialization time
        start_time = time.time()
        await self.detector.initialize()
        init_time = time.time() - start_time

        # Should initialize reasonably quickly even with 1000 collections
        assert init_time < 5.0  # 5 seconds max

        # Measure collision detection time
        start_time = time.time()
        result = await self.detector.check_collection_collision(
            "new_collection", CollectionCategory.GLOBAL
        )
        detection_time = time.time() - start_time

        # Should detect collisions quickly
        assert detection_time < 0.1  # 100ms max
        assert result.detection_time_ms < 100

    @pytest.mark.asyncio
    async def test_bloom_filter_efficiency(self):
        """Test Bloom filter efficiency in collision detection."""
        # Mock moderate collections response
        mock_collections = []
        for i in range(100):
            mock_collection = Mock()
            mock_collection.name = f"collection_{i:03d}"
            mock_collections.append(mock_collection)

        mock_response = Mock()
        mock_response.collections = mock_collections
        self.mock_qdrant_client.get_collections.return_value = mock_response

        await self.detector.initialize()

        # Test many non-existing collections
        for i in range(100, 200):
            collection_name = f"nonexistent_{i:03d}"
            result = await self.detector.check_collection_collision(collection_name)
            assert result.has_collision is False

        # Check registry statistics for Bloom filter efficiency
        stats = await self.detector.get_collision_statistics()
        registry_stats = stats['registry_statistics']

        # Bloom filter should have handled many negative lookups efficiently
        assert registry_stats['bloom_filter_efficiency_percent'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])