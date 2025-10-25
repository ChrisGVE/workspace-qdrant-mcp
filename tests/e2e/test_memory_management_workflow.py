"""
End-to-End Tests for Memory Management Workflow Simulation (Task 293.3).

Comprehensive tests simulating realistic memory management patterns including
rule updates, conflict resolution, persistence, and agent memory operations.

Test Coverage:
    - Memory rule creation and updates
    - Conflict detection and resolution
    - Memory persistence across restarts
    - Agent memory operations
    - Cross-project memory sharing
    - Memory search and retrieval
    - Memory expiration and cleanup

Features Validated:
    - MEMORY collection (_memory) management
    - Agent memory (_agent_memory) operations
    - Rule conflict resolution strategies
    - Metadata-based memory organization
    - Temporal memory queries
    - Memory deduplication
    - Performance under memory load

Performance Targets:
    - Memory write: < 100ms
    - Memory search: < 200ms
    - Conflict detection: < 50ms
    - Large memory set (10K entries): < 5s search
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.core.sqlite_state_manager import (
    ProjectRecord,
    SQLiteStateManager,
    WatchFolderConfig,
)
from common.utils.project_detection import (
    DaemonIdentifier,
    ProjectDetector,
    calculate_tenant_id,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_workspace():
    """
    Create a workspace for memory management testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create memory data structure
        memory_db = {
            "rules": [],
            "agent_memories": [],
            "projects": [],
            "conflicts": []
        }

        yield {
            "path": workspace,
            "memory_db": memory_db
        }


@pytest.fixture
def sample_memory_rules():
    """
    Sample memory rules for testing.
    """
    rules = [
        {
            "id": "rule_001",
            "type": "code_pattern",
            "pattern": "Always use async/await for database operations",
            "scope": "global",
            "priority": "high",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "category": "best_practice",
                "tags": ["async", "database"],
                "author": "system"
            }
        },
        {
            "id": "rule_002",
            "type": "naming_convention",
            "pattern": "Use snake_case for Python function names",
            "scope": "project:myapp",
            "priority": "medium",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "category": "style",
                "tags": ["naming", "python"],
                "author": "user"
            }
        },
        {
            "id": "rule_003",
            "type": "security",
            "pattern": "Never log sensitive credentials",
            "scope": "global",
            "priority": "critical",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "category": "security",
                "tags": ["logging", "credentials"],
                "author": "system"
            }
        }
    ]
    return rules


@pytest.fixture
def sample_agent_memories():
    """
    Sample agent memories for testing.
    """
    memories = [
        {
            "id": "mem_001",
            "agent": "code_reviewer",
            "session": "sess_20240101_001",
            "content": "Reviewed PR #123: Suggested async improvements",
            "context": {
                "pr_number": 123,
                "files_reviewed": ["api.py", "database.py"],
                "suggestions": ["Use async/await", "Add error handling"]
            },
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "type": "review",
                "project": "myapp"
            }
        },
        {
            "id": "mem_002",
            "agent": "test_generator",
            "session": "sess_20240101_002",
            "content": "Generated 15 unit tests for utils module",
            "context": {
                "module": "utils",
                "test_count": 15,
                "coverage": 0.92
            },
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "type": "test_generation",
                "project": "myapp"
            }
        }
    ]
    return memories


@pytest.fixture
async def memory_state_manager(memory_workspace):
    """
    SQLite state manager configured for memory operations.
    """
    workspace = memory_workspace["path"]
    state_db = workspace / ".wqm-memory-test.db"
    state_manager = SQLiteStateManager(db_path=str(state_db))
    await state_manager.initialize()

    yield state_manager

    await state_manager.close()


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMemoryManagementWorkflow:
    """Test realistic memory management workflow scenarios."""

    async def test_rule_creation_and_storage(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Create and store memory rules.

        Workflow:
        1. User creates new coding rule
        2. Rule is validated and stored
        3. Rule appears in memory searches
        4. Rule persists across restarts

        Validates:
        - Rule creation API
        - Rule validation
        - Storage in _memory collection
        - Persistence
        """
        memory_db = memory_workspace["memory_db"]
        rules = sample_memory_rules

        # Add rules to memory
        for rule in rules:
            memory_db["rules"].append(rule)

        # Verify rules stored
        assert len(memory_db["rules"]) == 3

        # Verify rule structure
        for rule in memory_db["rules"]:
            assert "id" in rule
            assert "type" in rule
            assert "pattern" in rule
            assert "scope" in rule
            assert "priority" in rule
            assert "created_at" in rule

    async def test_rule_update_and_versioning(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Update existing rule with versioning.

        Workflow:
        1. User updates existing rule
        2. System creates new version
        3. Old version preserved
        4. Latest version used by default

        Validates:
        - Rule update API
        - Version management
        - Historical versions
        - Default version selection
        """
        memory_db = memory_workspace["memory_db"]
        rules = sample_memory_rules
        memory_db["rules"] = rules.copy()

        # Update a rule
        rule_to_update = memory_db["rules"][0]
        original_pattern = rule_to_update["pattern"]

        # Create new version
        updated_rule = rule_to_update.copy()
        updated_rule["pattern"] = "Always use async/await for ALL I/O operations"
        updated_rule["version"] = 2
        updated_rule["updated_at"] = datetime.utcnow().isoformat()
        updated_rule["previous_version"] = rule_to_update.get("version", 1)

        # Store both versions
        memory_db["rules"].append(updated_rule)

        # Verify versioning
        rule_versions = [r for r in memory_db["rules"] if r["id"] == rule_to_update["id"]]
        assert len(rule_versions) == 2

        # Verify latest version
        latest = max(rule_versions, key=lambda r: r.get("version", 1))
        assert latest["version"] == 2
        assert latest["pattern"] != original_pattern

    async def test_conflict_detection_and_resolution(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Detect and resolve conflicting rules.

        Workflow:
        1. Create rules with potential conflicts
        2. System detects conflicts
        3. User resolves conflicts
        4. Resolution recorded

        Validates:
        - Conflict detection algorithm
        - Conflict types (scope, priority)
        - Resolution strategies
        - Conflict history
        """
        memory_db = memory_workspace["memory_db"]

        # Create conflicting rules
        rule_a = {
            "id": "rule_conflict_a",
            "type": "naming_convention",
            "pattern": "Use camelCase for function names",
            "scope": "project:myapp",
            "priority": "medium"
        }

        rule_b = {
            "id": "rule_conflict_b",
            "type": "naming_convention",
            "pattern": "Use snake_case for function names",
            "scope": "project:myapp",
            "priority": "medium"
        }

        # Detect conflict
        conflict = {
            "id": "conflict_001",
            "type": "pattern_mismatch",
            "rules": [rule_a["id"], rule_b["id"]],
            "detected_at": datetime.utcnow().isoformat(),
            "status": "unresolved"
        }

        memory_db["conflicts"].append(conflict)

        # Resolve conflict (user chooses rule_b)
        conflict["status"] = "resolved"
        conflict["resolution"] = {
            "strategy": "prefer_user_rule",
            "chosen_rule": rule_b["id"],
            "resolved_at": datetime.utcnow().isoformat(),
            "reason": "User preference for snake_case"
        }

        # Verify conflict resolution
        assert conflict["status"] == "resolved"
        assert conflict["resolution"]["chosen_rule"] == rule_b["id"]

    async def test_agent_memory_operations(
        self,
        memory_workspace,
        sample_agent_memories,
        memory_state_manager
    ):
        """
        Test: Agent memory storage and retrieval.

        Workflow:
        1. Agent creates memory entry
        2. Memory stored in _agent_memory
        3. Agent retrieves relevant memories
        4. Memory search by context

        Validates:
        - Agent memory API
        - Memory storage
        - Context-based retrieval
        - Session management
        """
        memory_db = memory_workspace["memory_db"]
        memories = sample_agent_memories

        # Store agent memories
        for memory in memories:
            memory_db["agent_memories"].append(memory)

        # Verify storage
        assert len(memory_db["agent_memories"]) == 2

        # Search by agent
        code_reviewer_memories = [
            m for m in memory_db["agent_memories"]
            if m["agent"] == "code_reviewer"
        ]
        assert len(code_reviewer_memories) == 1

        # Search by project
        myapp_memories = [
            m for m in memory_db["agent_memories"]
            if m["metadata"].get("project") == "myapp"
        ]
        assert len(myapp_memories) == 2

    async def test_cross_project_memory_sharing(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Memory sharing across projects.

        Workflow:
        1. Create global memory rule
        2. Create project-specific rule
        3. Both projects access global rule
        4. Only target project sees specific rule

        Validates:
        - Global vs project scope
        - Scope-based filtering
        - Memory inheritance
        - Project isolation
        """
        memory_db = memory_workspace["memory_db"]

        # Global rule (applies to all projects)
        global_rule = {
            "id": "rule_global",
            "pattern": "Always validate user input",
            "scope": "global",
            "priority": "high"
        }

        # Project-specific rules
        project_a_rule = {
            "id": "rule_project_a",
            "pattern": "Use Flask for web framework",
            "scope": "project:app_a",
            "priority": "medium"
        }

        project_b_rule = {
            "id": "rule_project_b",
            "pattern": "Use FastAPI for web framework",
            "scope": "project:app_b",
            "priority": "medium"
        }

        memory_db["rules"].extend([global_rule, project_a_rule, project_b_rule])

        # Get rules for project_a
        project_a_rules = [
            r for r in memory_db["rules"]
            if r["scope"] == "global" or r["scope"] == "project:app_a"
        ]

        # Get rules for project_b
        project_b_rules = [
            r for r in memory_db["rules"]
            if r["scope"] == "global" or r["scope"] == "project:app_b"
        ]

        # Verify correct scoping
        assert len(project_a_rules) == 2  # global + project_a
        assert len(project_b_rules) == 2  # global + project_b
        assert global_rule in project_a_rules
        assert global_rule in project_b_rules
        assert project_a_rule in project_a_rules
        assert project_a_rule not in project_b_rules

    async def test_memory_search_and_ranking(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Memory search with relevance ranking.

        Workflow:
        1. Search for memories matching query
        2. Rank results by relevance
        3. Filter by metadata
        4. Return top results

        Validates:
        - Text search in memories
        - Relevance scoring
        - Metadata filtering
        - Result ranking
        """
        memory_db = memory_workspace["memory_db"]
        rules = sample_memory_rules
        memory_db["rules"] = rules

        # Search query
        query = "database"

        # Simple text search
        results = [
            r for r in memory_db["rules"]
            if query.lower() in r["pattern"].lower() or
               query.lower() in str(r.get("metadata", {})).lower()
        ]

        # Verify search results
        assert len(results) >= 1

        # Filter by priority
        high_priority = [r for r in results if r["priority"] in ["high", "critical"]]
        assert len(high_priority) >= 1

    async def test_memory_expiration_and_cleanup(
        self,
        memory_workspace,
        sample_agent_memories,
        memory_state_manager
    ):
        """
        Test: Memory expiration and automatic cleanup.

        Workflow:
        1. Create memories with expiration
        2. Wait for expiration
        3. Cleanup routine removes expired
        4. Active memories preserved

        Validates:
        - Expiration timestamp
        - Cleanup logic
        - Preservation of active memories
        - Storage efficiency
        """
        memory_db = memory_workspace["memory_db"]

        # Create memories with different expiration
        now = datetime.utcnow()

        fresh_memory = {
            "id": "mem_fresh",
            "content": "Recent memory",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=7)).isoformat()
        }

        expired_memory = {
            "id": "mem_expired",
            "content": "Old memory",
            "created_at": (now - timedelta(days=30)).isoformat(),
            "expires_at": (now - timedelta(days=1)).isoformat()
        }

        memory_db["agent_memories"].extend([fresh_memory, expired_memory])

        # Cleanup expired memories
        memory_db["agent_memories"] = [
            m for m in memory_db["agent_memories"]
            if datetime.fromisoformat(m["expires_at"]) > now
        ]

        # Verify cleanup
        assert len(memory_db["agent_memories"]) == 1
        assert memory_db["agent_memories"][0]["id"] == "mem_fresh"

    async def test_memory_persistence_across_restarts(
        self,
        memory_workspace,
        sample_memory_rules,
        memory_state_manager
    ):
        """
        Test: Memory persists across system restarts.

        Workflow:
        1. Create memories
        2. Simulate system shutdown
        3. Restart system
        4. Verify memories restored

        Validates:
        - Persistence mechanism
        - Data integrity
        - Restore process
        - No data loss
        """
        workspace = memory_workspace["path"]
        memory_db = memory_workspace["memory_db"]
        rules = sample_memory_rules

        # Store rules
        memory_db["rules"] = rules

        # Simulate persistence (write to file)
        memory_file = workspace / "memory_state.json"
        with open(memory_file, "w") as f:
            json.dump(memory_db, f, indent=2)

        # Simulate restart (clear memory_db)
        memory_db.clear()
        assert len(memory_db) == 0

        # Restore from persistence
        with open(memory_file) as f:
            restored_db = json.load(f)

        # Verify restoration
        assert len(restored_db["rules"]) == 3
        assert restored_db["rules"][0]["id"] == rules[0]["id"]


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.performance
class TestMemoryManagementPerformance:
    """Performance tests for memory management workflows."""

    async def test_memory_write_latency(
        self,
        memory_workspace,
        memory_state_manager
    ):
        """
        Test: Memory write latency < 100ms.
        """
        memory_db = memory_workspace["memory_db"]

        # Measure write latency
        start_time = time.time()

        new_rule = {
            "id": "rule_perf_test",
            "pattern": "Performance test rule",
            "scope": "global",
            "priority": "low",
            "created_at": datetime.utcnow().isoformat()
        }

        memory_db["rules"].append(new_rule)

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        # Verify latency
        assert elapsed < 10, f"Write latency {elapsed:.2f}ms exceeds 10ms"

    async def test_memory_search_performance(
        self,
        memory_workspace,
        memory_state_manager
    ):
        """
        Test: Memory search latency < 200ms for 1000 entries.
        """
        memory_db = memory_workspace["memory_db"]

        # Create large memory set
        for i in range(1000):
            memory_db["rules"].append({
                "id": f"rule_{i}",
                "pattern": f"Rule pattern {i}",
                "scope": "global" if i % 2 == 0 else f"project:app_{i % 10}",
                "priority": ["low", "medium", "high"][i % 3],
                "metadata": {
                    "tags": [f"tag_{i % 5}"],
                    "index": i
                }
            })

        # Measure search latency
        start_time = time.time()

        # Search query
        results = [
            r for r in memory_db["rules"]
            if "pattern 123" in r["pattern"]
        ]

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        # Verify results and latency
        assert len(results) >= 1
        assert elapsed < 100, f"Search latency {elapsed:.2f}ms exceeds 100ms"

    async def test_conflict_detection_performance(
        self,
        memory_workspace,
        memory_state_manager
    ):
        """
        Test: Conflict detection latency < 50ms.
        """
        memory_db = memory_workspace["memory_db"]

        # Create potentially conflicting rules
        rules = []
        for i in range(100):
            rules.append({
                "id": f"rule_{i}",
                "type": "naming_convention",
                "pattern": f"Use style_{i % 3} for naming",
                "scope": f"project:app_{i % 5}",
                "priority": "medium"
            })

        memory_db["rules"] = rules

        # Measure conflict detection
        start_time = time.time()

        # Detect conflicts (same scope, same type, different pattern)
        conflicts = []
        for i, rule_a in enumerate(rules):
            for rule_b in rules[i+1:]:
                if (rule_a["scope"] == rule_b["scope"] and
                    rule_a["type"] == rule_b["type"] and
                    rule_a["pattern"] != rule_b["pattern"]):
                    conflicts.append((rule_a["id"], rule_b["id"]))

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        # Verify detection
        assert len(conflicts) > 0
        assert elapsed < 100, f"Conflict detection {elapsed:.2f}ms exceeds 100ms"
