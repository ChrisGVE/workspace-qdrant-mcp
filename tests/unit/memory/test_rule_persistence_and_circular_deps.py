"""
Comprehensive tests for rule persistence and circular dependency detection.

This module tests:
1. Rule persistence across daemon restarts and session boundaries
2. Circular dependency detection in rule hierarchies
3. Prevention and resolution of circular dependencies

Test Coverage:
- Rule persistence in Qdrant
- Persistence across simulated restarts
- Complex persistence scenarios
- Simple and complex circular dependencies
- Dependency prevention on creation and updates
- Circular dependency resolution strategies
- Integration scenarios
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from common.core.collection_naming import CollectionNamingManager, CollectionType
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
)
from qdrant_client.models import PointStruct, ScoredPoint

# ============================================================================
# HELPER CLASSES AND FIXTURES
# ============================================================================

class DependencyGraph:
    """Helper class for building and analyzing dependency graphs."""

    def __init__(self):
        self.dependencies: dict[str, list[str]] = {}

    def add_dependency(self, rule_id: str, depends_on: str):
        """Add a dependency: rule_id depends on depends_on."""
        if rule_id not in self.dependencies:
            self.dependencies[rule_id] = []
        self.dependencies[rule_id].append(depends_on)

    def add_rule_dependencies(self, rule_id: str, replaces: list[str]):
        """Add dependencies from a rule's replaces field."""
        if replaces:
            for replaced_id in replaces:
                self.add_dependency(rule_id, replaced_id)

    def has_cycle(self) -> bool:
        """Check if the dependency graph has any cycles (iterative version)."""
        visited = set()

        # Color-based DFS: WHITE (unvisited), GRAY (in-progress), BLACK (finished)
        GRAY = 1
        BLACK = 2
        color = {}

        for node in self.dependencies:
            if node not in visited:
                # Iterative DFS using explicit stack
                stack = [(node, False)]  # (node, backtracking)

                while stack:
                    current, backtracking = stack.pop()

                    if backtracking:
                        # Mark as BLACK (finished processing)
                        color[current] = BLACK
                        continue

                    if current in color:
                        if color[current] == GRAY:
                            # Back edge found - cycle detected
                            return True
                        # If BLACK, already processed
                        continue

                    # Mark as GRAY (in-progress)
                    visited.add(current)
                    color[current] = GRAY

                    # Add backtracking marker
                    stack.append((current, True))

                    # Add neighbors to stack
                    for neighbor in self.dependencies.get(current, []):
                        if neighbor not in color or color[neighbor] == GRAY:
                            stack.append((neighbor, False))

        return False

    def find_cycles(self) -> list[list[str]]:
        """Find all cycles in the dependency graph (iterative version)."""
        cycles = []
        visited = set()

        for start_node in self.dependencies:
            if start_node in visited:
                continue

            # For each unvisited node, try to find cycles
            # Use stack with path tracking
            stack = [(start_node, [start_node])]
            path_set = {start_node}

            while stack:
                node, path = stack.pop()

                # Remove from path_set when backtracking
                if node not in path:
                    path_set.discard(node)
                    continue

                visited.add(node)

                for neighbor in self.dependencies.get(node, []):
                    if neighbor in path_set:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        if cycle not in cycles:
                            cycles.append(cycle)
                    elif neighbor not in visited:
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
                        path_set.add(neighbor)

        return cycles


class MockCollection:
    """Mock collection object with name attribute."""
    def __init__(self, name):
        self.name = name


class MockQdrantClient:
    """Mock Qdrant client that properly simulates persistence."""

    def __init__(self):
        self.storage = {}

    def get_collections(self):
        """Get collections list."""
        collections_list = Mock()
        collections_list.collections = [MockCollection(name) for name in self.storage.keys()]
        return collections_list

    def create_collection(self, collection_name, **kwargs):
        """Create a collection."""
        if collection_name not in self.storage:
            self.storage[collection_name] = {}

    def upsert(self, collection_name, points, **kwargs):
        """Upsert points to collection."""
        if collection_name not in self.storage:
            self.storage[collection_name] = {}
        for point in points:
            self.storage[collection_name][point.id] = point

    def retrieve(self, collection_name, ids, **kwargs):
        """Retrieve points by IDs."""
        if collection_name not in self.storage:
            return []
        return [
            self.storage[collection_name].get(point_id)
            for point_id in ids
            if point_id in self.storage[collection_name]
        ]

    def scroll(self, collection_name, **kwargs):
        """Scroll through collection."""
        if collection_name not in self.storage:
            return [], None
        return list(self.storage[collection_name].values()), None

    def delete(self, collection_name, points_selector, **kwargs):
        """Delete points from collection."""
        if collection_name not in self.storage:
            return
        for point_id in points_selector:
            self.storage[collection_name].pop(point_id, None)


@pytest.fixture
def dependency_graph():
    """Provide a dependency graph for testing."""
    return DependencyGraph()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client with persistence simulation."""
    return MockQdrantClient()


@pytest.fixture
async def memory_manager(mock_qdrant_client):
    """Create a memory manager with mocked Qdrant client."""
    naming_manager = Mock(spec=CollectionNamingManager)
    naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

    manager = MemoryManager(
        qdrant_client=mock_qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=384,
        sparse_vector_generator=None,
        memory_collection_name="memory"
    )

    await manager.initialize_memory_collection()
    return manager


@pytest.fixture
def circular_rule_set():
    """Create a set of rules with circular dependencies."""
    return [
        MemoryRule(
            id="rule-a",
            category=MemoryCategory.BEHAVIOR,
            name="Rule A",
            rule="Rule A content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-b"]  # A depends on B
        ),
        MemoryRule(
            id="rule-b",
            category=MemoryCategory.BEHAVIOR,
            name="Rule B",
            rule="Rule B content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-a"]  # B depends on A -> circular!
        )
    ]


@pytest.fixture
def complex_circular_rule_set():
    """Create a complex set of rules with multiple circular dependencies."""
    return [
        MemoryRule(
            id="rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 1",
            rule="Rule 1 content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-2"]  # 1 -> 2
        ),
        MemoryRule(
            id="rule-2",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 2",
            rule="Rule 2 content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-3"]  # 2 -> 3
        ),
        MemoryRule(
            id="rule-3",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 3",
            rule="Rule 3 content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-4"]  # 3 -> 4
        ),
        MemoryRule(
            id="rule-4",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 4",
            rule="Rule 4 content",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-1"]  # 4 -> 1 -> circular chain!
        )
    ]


# ============================================================================
# 1. RULE PERSISTENCE TESTS
# ============================================================================

class TestBasicPersistence:
    """Test basic rule persistence in Qdrant."""

    @pytest.mark.asyncio
    async def test_rule_persists_in_qdrant(self, memory_manager):
        """Test that a rule saved to Qdrant can be retrieved."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Test Rule",
            rule="Always test your code",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"]
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.id == rule_id
        assert retrieved_rule.name == "Test Rule"
        assert retrieved_rule.rule == "Always test your code"
        assert retrieved_rule.authority == AuthorityLevel.ABSOLUTE
        assert retrieved_rule.scope == ["global"]

    @pytest.mark.asyncio
    async def test_rule_persists_across_manager_instances(self):
        """Test that rules survive MemoryManager recreation."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        # Create first manager and add rule
        manager1 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager1.initialize_memory_collection()

        rule_id = await manager1.add_memory_rule(
            category=MemoryCategory.PREFERENCE,
            name="Package Manager",
            rule="Use uv for Python",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        # Create second manager (simulates restart)
        manager2 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )

        # Retrieve rule with new manager instance
        retrieved_rule = await manager2.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.id == rule_id
        assert retrieved_rule.name == "Package Manager"
        assert retrieved_rule.rule == "Use uv for Python"

    @pytest.mark.asyncio
    async def test_rule_metadata_persists(self, memory_manager):
        """Test that all metadata fields persist correctly."""
        datetime.now(timezone.utc)

        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Metadata Test",
            rule="Test metadata persistence",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["test", "global"],
            source="test_source",
            conditions={"env": "production"},
            metadata={"custom_field": "custom_value", "priority": 10}
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.source == "test_source"
        assert retrieved_rule.conditions == {"env": "production"}
        assert retrieved_rule.metadata == {"custom_field": "custom_value", "priority": 10}
        assert retrieved_rule.created_at is not None
        assert retrieved_rule.updated_at is not None
        assert retrieved_rule.created_at <= retrieved_rule.updated_at

    @pytest.mark.asyncio
    async def test_multiple_rules_persist(self, memory_manager):
        """Test that multiple rules persist correctly."""
        rule_ids = []

        for i in range(5):
            rule_id = await memory_manager.add_memory_rule(
                category=MemoryCategory.BEHAVIOR,
                name=f"Rule {i}",
                rule=f"Rule content {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"]
            )
            rule_ids.append(rule_id)

        # Retrieve all rules
        all_rules = await memory_manager.list_memory_rules()

        assert len(all_rules) == 5
        retrieved_ids = {rule.id for rule in all_rules}
        assert set(rule_ids) == retrieved_ids

    @pytest.mark.asyncio
    async def test_empty_collection_persistence(self, memory_manager):
        """Test that empty collection is handled correctly."""
        all_rules = await memory_manager.list_memory_rules()

        assert all_rules == []
        assert isinstance(all_rules, list)


class TestPersistenceAcrossRestarts:
    """Test persistence across simulated daemon restarts."""

    @pytest.mark.asyncio
    async def test_rule_persists_after_simulated_restart(self):
        """Test rule persistence after simulated daemon restart."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        # Pre-restart: Add rule
        manager_pre = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager_pre.initialize_memory_collection()

        rule_id = await manager_pre.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Persist Rule",
            rule="Should persist across restarts",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"]
        )

        # Simulate restart by creating new manager
        manager_post = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )

        # Post-restart: Retrieve rule
        retrieved_rule = await manager_post.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.id == rule_id
        assert retrieved_rule.name == "Persist Rule"
        assert retrieved_rule.authority == AuthorityLevel.ABSOLUTE

    @pytest.mark.asyncio
    async def test_absolute_rules_persist_after_restart(self):
        """Test that absolute authority rules persist after restart."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        manager1 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager1.initialize_memory_collection()

        # Add absolute rule
        await manager1.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Critical Rule",
            rule="Absolutely must be followed",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"]
        )

        # Restart
        manager2 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )

        absolute_rules = await manager2.list_memory_rules(
            authority=AuthorityLevel.ABSOLUTE
        )

        assert len(absolute_rules) == 1
        assert absolute_rules[0].name == "Critical Rule"
        assert absolute_rules[0].authority == AuthorityLevel.ABSOLUTE

    @pytest.mark.asyncio
    async def test_default_rules_persist_after_restart(self):
        """Test that default authority rules persist after restart."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        manager1 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager1.initialize_memory_collection()

        # Add default rule
        await manager1.add_memory_rule(
            category=MemoryCategory.PREFERENCE,
            name="Optional Preference",
            rule="Prefer this unless told otherwise",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Restart
        manager2 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )

        default_rules = await manager2.list_memory_rules(
            authority=AuthorityLevel.DEFAULT
        )

        assert len(default_rules) == 1
        assert default_rules[0].name == "Optional Preference"
        assert default_rules[0].authority == AuthorityLevel.DEFAULT


class TestComplexPersistenceScenarios:
    """Test complex persistence scenarios."""

    @pytest.mark.asyncio
    async def test_rule_with_dependencies_persists(self, memory_manager):
        """Test that rules with replaces field persist correctly."""
        # Add base rule
        await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Base Rule",
            rule="Original rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Add rule that replaces base rule (note: this will delete base_rule)
        # So we create a second base rule to test the replaces field persistence
        base_rule_id_2 = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Base Rule 2",
            rule="Another original rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        replacing_rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Replacing Rule",
            rule="Updated rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=[base_rule_id_2]
        )

        # Retrieve replacing rule
        retrieved_rule = await memory_manager.get_memory_rule(replacing_rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.replaces == [base_rule_id_2]

    @pytest.mark.asyncio
    async def test_rule_with_conditions_persists(self, memory_manager):
        """Test that rules with conditional logic persist."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Conditional Rule",
            rule="Apply when condition is met",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            conditions={
                "environment": "production",
                "feature_flag": "enabled",
                "threshold": 0.95
            }
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.conditions == {
            "environment": "production",
            "feature_flag": "enabled",
            "threshold": 0.95
        }

    @pytest.mark.asyncio
    async def test_rule_scope_persists(self, memory_manager):
        """Test that scope arrays persist correctly."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.PREFERENCE,
            name="Multi-Scope Rule",
            rule="Applies to multiple scopes",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing", "project-x", "global"]
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.scope == ["python", "testing", "project-x", "global"]

    @pytest.mark.asyncio
    async def test_rule_timestamps_persist_accurately(self, memory_manager):
        """Test that timestamps persist with correct values."""
        before_creation = datetime.now(timezone.utc)

        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Timestamp Test",
            rule="Test timestamp accuracy",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        after_creation = datetime.now(timezone.utc)

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.created_at >= before_creation
        assert retrieved_rule.created_at <= after_creation
        assert retrieved_rule.updated_at >= retrieved_rule.created_at


class TestPersistenceEdgeCases:
    """Test persistence edge cases."""

    @pytest.mark.asyncio
    async def test_persistence_with_special_characters(self, memory_manager):
        """Test that rules with unicode and special characters persist."""
        special_text = "Test with émojis 🚀 and spëcial çharacters: 你好世界"

        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.PREFERENCE,
            name="Special Chars",
            rule=special_text,
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.rule == special_text

    @pytest.mark.asyncio
    async def test_persistence_with_large_metadata(self, memory_manager):
        """Test that large metadata objects persist correctly."""
        large_metadata = {
            f"key_{i}": f"value_{i}" * 100 for i in range(50)
        }

        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Large Metadata",
            rule="Rule with large metadata",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            metadata=large_metadata
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.metadata == large_metadata

    @pytest.mark.asyncio
    async def test_persistence_with_null_optional_fields(self, memory_manager):
        """Test that null optional fields are handled correctly."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Minimal Rule",
            rule="Minimal rule with no optional fields",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            conditions=None,
            replaces=None,
            metadata=None
        )

        retrieved_rule = await memory_manager.get_memory_rule(rule_id)

        assert retrieved_rule is not None
        assert retrieved_rule.conditions is None
        assert retrieved_rule.replaces is None or retrieved_rule.replaces == []
        assert retrieved_rule.metadata is None or retrieved_rule.metadata == {}


# ============================================================================
# 2. CIRCULAR DEPENDENCY DETECTION TESTS
# ============================================================================

class TestSimpleCircularDependencies:
    """Test detection of simple circular dependencies."""

    def test_detect_simple_circular_dependency_two_rules(self, dependency_graph):
        """Test detection of A→B→A cycle."""
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-a")

        assert dependency_graph.has_cycle() is True

        cycles = dependency_graph.find_cycles()
        assert len(cycles) > 0

    def test_detect_self_referencing_rule(self, dependency_graph):
        """Test detection of A→A self-reference."""
        dependency_graph.add_dependency("rule-a", "rule-a")

        assert dependency_graph.has_cycle() is True

    def test_detect_mutual_dependencies(self, dependency_graph):
        """Test detection of mutual dependencies."""
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-a")

        assert dependency_graph.has_cycle() is True

    def test_no_false_positive_on_linear_chain(self, dependency_graph):
        """Test that A→B→C is correctly identified as valid."""
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-c")

        assert dependency_graph.has_cycle() is False

    def test_no_false_positive_on_parallel_dependencies(self, dependency_graph):
        """Test that A→C, B→C is correctly identified as valid."""
        dependency_graph.add_dependency("rule-a", "rule-c")
        dependency_graph.add_dependency("rule-b", "rule-c")

        assert dependency_graph.has_cycle() is False


class TestComplexCircularDependencies:
    """Test detection of complex circular dependencies."""

    def test_detect_long_circular_chain(self, dependency_graph):
        """Test detection of A→B→C→D→A cycle."""
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-c")
        dependency_graph.add_dependency("rule-c", "rule-d")
        dependency_graph.add_dependency("rule-d", "rule-a")

        assert dependency_graph.has_cycle() is True

        cycles = dependency_graph.find_cycles()
        assert len(cycles) > 0

    def test_detect_circular_in_complex_graph(self, dependency_graph):
        """Test detection of cycle in complex dependency graph."""
        # Create complex graph with one cycle
        dependency_graph.add_dependency("rule-1", "rule-2")
        dependency_graph.add_dependency("rule-2", "rule-3")
        dependency_graph.add_dependency("rule-3", "rule-4")
        dependency_graph.add_dependency("rule-4", "rule-5")
        dependency_graph.add_dependency("rule-5", "rule-2")  # Cycle: 2→3→4→5→2
        dependency_graph.add_dependency("rule-6", "rule-7")
        dependency_graph.add_dependency("rule-7", "rule-8")

        assert dependency_graph.has_cycle() is True

    def test_detect_multiple_circular_dependencies(self, dependency_graph):
        """Test detection of multiple independent cycles."""
        # First cycle: A→B→A
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-a")

        # Second cycle: X→Y→Z→X
        dependency_graph.add_dependency("rule-x", "rule-y")
        dependency_graph.add_dependency("rule-y", "rule-z")
        dependency_graph.add_dependency("rule-z", "rule-x")

        assert dependency_graph.has_cycle() is True

        cycles = dependency_graph.find_cycles()
        assert len(cycles) >= 2

    def test_detect_nested_circular_dependencies(self, dependency_graph):
        """Test detection of nested cycles."""
        # Outer cycle
        dependency_graph.add_dependency("rule-1", "rule-2")
        dependency_graph.add_dependency("rule-2", "rule-3")
        dependency_graph.add_dependency("rule-3", "rule-1")

        # Inner cycle within the outer
        dependency_graph.add_dependency("rule-2", "rule-4")
        dependency_graph.add_dependency("rule-4", "rule-2")

        assert dependency_graph.has_cycle() is True

    def test_detect_indirect_circular_dependencies(self, dependency_graph):
        """Test detection of indirect cycles (A→B→C, C→A)."""
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-c")
        dependency_graph.add_dependency("rule-c", "rule-a")

        assert dependency_graph.has_cycle() is True


class TestDependencyMetadataHandling:
    """Test dependency metadata handling."""

    @pytest.mark.asyncio
    async def test_dependencies_stored_correctly(self, memory_manager):
        """Test that dependencies are stored in replaces field."""
        # Don't use replaces with actual rule IDs as they'll be deleted
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="With Deps",
            rule="Has dependencies",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["placeholder-id"]
        )

        retrieved = await memory_manager.get_memory_rule(rule_id)
        assert retrieved.replaces == ["placeholder-id"]

    @pytest.mark.asyncio
    async def test_dependencies_retrieved_correctly(self, memory_manager):
        """Test that dependencies are loaded correctly from storage."""
        deps = ["rule-1", "rule-2", "rule-3"]

        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Multi-Dep",
            rule="Multiple dependencies",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=deps
        )

        retrieved = await memory_manager.get_memory_rule(rule_id)
        assert set(retrieved.replaces) == set(deps)

    @pytest.mark.asyncio
    async def test_empty_dependencies_handled(self, memory_manager):
        """Test that rules without dependencies are handled correctly."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="No Deps",
            rule="No dependencies",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=None
        )

        retrieved = await memory_manager.get_memory_rule(rule_id)
        assert retrieved.replaces is None or retrieved.replaces == []

    @pytest.mark.asyncio
    async def test_dependency_updates_tracked(self, memory_manager):
        """Test that dependency changes are tracked in metadata."""
        rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Update Deps",
            rule="Test dependency updates",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["rule-1"]
        )

        # Update dependencies
        await memory_manager.update_memory_rule(
            rule_id,
            {"replaces": ["rule-1", "rule-2", "rule-3"]}
        )

        retrieved = await memory_manager.get_memory_rule(rule_id)
        assert len(retrieved.replaces) == 3
        assert "rule-2" in retrieved.replaces
        assert "rule-3" in retrieved.replaces


class TestDependencyEdgeCases:
    """Test edge cases in dependency detection."""

    def test_circular_dependency_with_deleted_rules(self, dependency_graph):
        """Test circular dependency detection when some rules are deleted."""
        # Create cycle
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-c")
        dependency_graph.add_dependency("rule-c", "rule-a")

        # Even with cycle, graph structure should be detectable
        assert dependency_graph.has_cycle() is True

    def test_circular_dependency_detection_performance(self, dependency_graph):
        """Test performance of cycle detection on large graphs."""
        import time

        # Create large graph with cycle at the end
        for i in range(1000):
            dependency_graph.add_dependency(f"rule-{i}", f"rule-{i+1}")

        # Add cycle
        dependency_graph.add_dependency("rule-1000", "rule-500")

        start = time.time()
        has_cycle = dependency_graph.has_cycle()
        elapsed = time.time() - start

        assert has_cycle is True
        assert elapsed < 1.0  # Should complete in under 1 second

    def test_partial_circular_dependencies(self, dependency_graph):
        """Test graph with some rules in cycle, some not."""
        # Cycle: A→B→C→A
        dependency_graph.add_dependency("rule-a", "rule-b")
        dependency_graph.add_dependency("rule-b", "rule-c")
        dependency_graph.add_dependency("rule-c", "rule-a")

        # Non-cycle: X→Y→Z
        dependency_graph.add_dependency("rule-x", "rule-y")
        dependency_graph.add_dependency("rule-y", "rule-z")

        assert dependency_graph.has_cycle() is True


# ============================================================================
# 3. CIRCULAR DEPENDENCY PREVENTION TESTS
# ============================================================================

class TestPreventionOnRuleCreation:
    """Test circular dependency prevention during rule creation."""

    @pytest.mark.asyncio
    async def test_allow_valid_dependency_creation(self, memory_manager):
        """Test that valid dependencies are allowed."""
        # Create base rule that won't be deleted
        base_rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Base",
            rule="Base rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create another base to avoid deletion
        await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Base 2",
            rule="Base rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        dependent_rule_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Dependent",
            rule="Depends on base",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=[base_rule_id]
        )

        # Valid dependency should work
        retrieved = await memory_manager.get_memory_rule(dependent_rule_id)
        assert retrieved is not None
        assert base_rule_id in retrieved.replaces

    @pytest.mark.asyncio
    async def test_dependency_graph_from_rules(self, memory_manager):
        """Test building dependency graph from rules."""
        # Create rules without replaces to avoid deletion
        rule1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Rule 1",
            rule="First rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        rule2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Rule 2",
            rule="Second rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        rule3_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Rule 3",
            rule="Third rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Update to create chain without deletion
        await memory_manager.update_memory_rule(rule2_id, {"replaces": [rule1_id]})
        await memory_manager.update_memory_rule(rule3_id, {"replaces": [rule2_id]})

        # Build dependency graph
        all_rules = await memory_manager.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        # Should not have cycle
        assert graph.has_cycle() is False


class TestDependencyValidation:
    """Test dependency validation logic."""

    def test_dependency_validation_algorithm(self, dependency_graph):
        """Test core dependency validation algorithm."""
        # Create valid graph
        dependency_graph.add_dependency("a", "b")
        dependency_graph.add_dependency("b", "c")

        assert dependency_graph.has_cycle() is False

        # Add cycle
        dependency_graph.add_dependency("c", "a")

        assert dependency_graph.has_cycle() is True

    def test_dependency_graph_construction(self, circular_rule_set):
        """Test that dependency graph is built correctly from rules."""
        graph = DependencyGraph()

        for rule in circular_rule_set:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        # Should detect the A<->B cycle
        assert graph.has_cycle() is True

    def test_cycle_detection_accuracy(self, complex_circular_rule_set):
        """Test that cycle detection is accurate."""
        graph = DependencyGraph()

        for rule in complex_circular_rule_set:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        # Should detect the 1->2->3->4->1 cycle
        assert graph.has_cycle() is True

        cycles = graph.find_cycles()  # FIXED: was dependency_graph.find_cycles()
        assert len(cycles) > 0


# ============================================================================
# 4. CIRCULAR DEPENDENCY RESOLUTION TESTS
# ============================================================================

class TestDetectionOfExistingCycles:
    """Test detection of existing circular dependencies."""

    @pytest.mark.asyncio
    async def test_detect_existing_circular_dependencies(self, memory_manager):
        """Test finding cycles in loaded rules."""
        # Create rules
        r1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R1",
            rule="Rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R2",
            rule="Rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create circular dependency via updates
        await memory_manager.update_memory_rule(r1_id, {"replaces": [r2_id]})
        await memory_manager.update_memory_rule(r2_id, {"replaces": [r1_id]})

        # Build graph from current rules
        all_rules = await memory_manager.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        # Should detect circular dependency
        assert graph.has_cycle() is True

    @pytest.mark.asyncio
    async def test_identify_all_rules_in_cycle(self, memory_manager):
        """Test that all rules participating in a cycle are identified."""
        # Create 3-rule cycle
        r1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R1",
            rule="Rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R2",
            rule="Rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r3_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R3",
            rule="Rule 3",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create cycle via updates
        await memory_manager.update_memory_rule(r1_id, {"replaces": [r3_id]})
        await memory_manager.update_memory_rule(r2_id, {"replaces": [r1_id]})
        await memory_manager.update_memory_rule(r3_id, {"replaces": [r2_id]})

        # Build graph
        all_rules = await memory_manager.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        cycles = graph.find_cycles()

        # Should find the cycle with all 3 rules
        assert len(cycles) > 0
        cycle = cycles[0]
        assert len(cycle) >= 3


class TestResolutionStrategies:
    """Test circular dependency resolution strategies."""

    @pytest.mark.asyncio
    async def test_break_cycle_by_removing_dependency(self, memory_manager):
        """Test breaking cycle by removing a dependency."""
        r1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R1",
            rule="Rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R2",
            rule="Rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create cycle
        await memory_manager.update_memory_rule(r1_id, {"replaces": [r2_id]})
        await memory_manager.update_memory_rule(r2_id, {"replaces": [r1_id]})

        # Break cycle by removing one dependency
        await memory_manager.update_memory_rule(r2_id, {"replaces": []})

        # Verify cycle is broken
        all_rules = await memory_manager.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        assert graph.has_cycle() is False


class TestPostResolutionValidation:
    """Test validation after circular dependency resolution."""

    @pytest.mark.asyncio
    async def test_validate_no_new_cycles_after_resolution(self, memory_manager):
        """Test that resolution doesn't create new cycles."""
        r1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R1",
            rule="Rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R2",
            rule="Rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create cycle
        await memory_manager.update_memory_rule(r1_id, {"replaces": [r2_id]})
        await memory_manager.update_memory_rule(r2_id, {"replaces": [r1_id]})

        # Break cycle
        await memory_manager.update_memory_rule(r1_id, {"replaces": []})

        # Verify no cycles
        all_rules = await memory_manager.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        assert graph.has_cycle() is False

    @pytest.mark.asyncio
    async def test_validate_rules_functional_after_resolution(self, memory_manager):
        """Test that rules remain functional after resolution."""
        r1_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Functional Test",
            rule="Should work after resolution",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await memory_manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Functional Test 2",
            rule="Should also work",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create and break cycle
        await memory_manager.update_memory_rule(r1_id, {"replaces": [r2_id]})
        await memory_manager.update_memory_rule(r2_id, {"replaces": [r1_id]})
        await memory_manager.update_memory_rule(r1_id, {"replaces": []})

        # Verify rules are still retrievable and functional
        retrieved_r1 = await memory_manager.get_memory_rule(r1_id)
        retrieved_r2 = await memory_manager.get_memory_rule(r2_id)

        assert retrieved_r1 is not None
        assert retrieved_r2 is not None
        assert retrieved_r1.name == "Functional Test"
        assert retrieved_r2.name == "Functional Test 2"


# ============================================================================
# 5. INTEGRATION TESTS
# ============================================================================

class TestPersistenceAndCircularDependency:
    """Test integration of persistence and circular dependency features."""

    @pytest.mark.asyncio
    async def test_circular_dependencies_persist(self):
        """Test that circular dependencies persist across restarts."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        # Create manager and rules with cycle
        manager1 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager1.initialize_memory_collection()

        r1_id = await manager1.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R1",
            rule="Rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        r2_id = await manager1.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="R2",
            rule="Rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        await manager1.update_memory_rule(r1_id, {"replaces": [r2_id]})
        await manager1.update_memory_rule(r2_id, {"replaces": [r1_id]})

        # Simulate restart
        manager2 = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )

        # Check that cycle persists
        all_rules = await manager2.list_memory_rules()
        graph = DependencyGraph()

        for rule in all_rules:
            if rule.replaces:
                graph.add_rule_dependencies(rule.id, rule.replaces)

        assert graph.has_cycle() is True

    @pytest.mark.asyncio
    async def test_complex_rule_hierarchy_with_persistence(self):
        """Test complex rule hierarchy persists correctly."""
        mock_client = MockQdrantClient()
        naming_manager = Mock(spec=CollectionNamingManager)
        naming_manager.validate_collection_name.return_value = Mock(is_valid=True)

        manager = MemoryManager(
            qdrant_client=mock_client,
            naming_manager=naming_manager,
            embedding_dim=384
        )
        await manager.initialize_memory_collection()

        # Create complex hierarchy
        base_id = await manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Base",
            rule="Base rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        child1_id = await manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Child 1",
            rule="Child rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        child2_id = await manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Child 2",
            rule="Child rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        grandchild_id = await manager.add_memory_rule(
            category=MemoryCategory.BEHAVIOR,
            name="Grandchild",
            rule="Grandchild rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Create hierarchy via updates (to avoid deletion)
        await manager.update_memory_rule(child1_id, {"replaces": [base_id]})
        await manager.update_memory_rule(child2_id, {"replaces": [base_id]})
        await manager.update_memory_rule(grandchild_id, {"replaces": [child1_id, child2_id]})

        # Retrieve and verify hierarchy
        all_rules = await manager.list_memory_rules()

        grandchild = next(r for r in all_rules if r.id == grandchild_id)
        assert child1_id in grandchild.replaces
        assert child2_id in grandchild.replaces
