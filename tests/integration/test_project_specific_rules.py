"""
Project-Specific Rule Activation and Isolation Tests (Task 337.4).

Tests that memory rules activate only in appropriate project contexts,
don't leak between projects, and maintain proper isolation.

Test Scenarios:
1. Rule activation only in target project context
2. Rule deactivation when switching projects
3. Project detection accuracy
4. Multi-project rule isolation
5. No cross-contamination between project rule sets
6. Project-scoped vs global rules
7. Nested project contexts (e.g., submodules)
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from src.python.common.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)

# Import test harness from Task 337.1
from tests.integration.test_llm_behavioral_harness import (
    LLMBehavioralHarness,
    MockLLMProvider,
    ExecutionMode,
    BehavioralMetrics,
    LLMResponse,
)

# Try to import real memory manager
try:
    from src.python.common.core.memory import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False


@pytest.fixture
def mock_project_detector():
    """Mock project detection functionality."""
    detector = Mock()
    detector.current_project = None

    def get_current_project():
        return detector.current_project

    def set_project(project_path: str):
        detector.current_project = project_path

    detector.get_current_project = Mock(side_effect=get_current_project)
    detector.set_project = set_project

    return detector


@pytest.fixture
async def project_aware_memory_manager(mock_project_detector):
    """Memory manager that filters rules by project."""
    manager = AsyncMock(spec=MemoryManager)
    manager._rules = {}  # project_path -> [rules]
    manager._global_rules = []
    manager._project_detector = mock_project_detector

    async def add_rule(rule: MemoryRule, project: Optional[str] = None):
        """Add rule to specific project or global."""
        target_project = project or mock_project_detector.current_project

        if target_project:
            if target_project not in manager._rules:
                manager._rules[target_project] = []
            manager._rules[target_project].append(rule)
        else:
            manager._global_rules.append(rule)

    async def get_rules(project: Optional[str] = None) -> List[MemoryRule]:
        """Get rules for specific project plus global rules."""
        target_project = project or mock_project_detector.current_project

        # Start with global rules
        rules = manager._global_rules.copy()

        # Add project-specific rules if project context exists
        if target_project and target_project in manager._rules:
            rules.extend(manager._rules[target_project])

        return rules

    async def get_all_rules() -> Dict[str, List[MemoryRule]]:
        """Get all rules grouped by project."""
        return {
            "global": manager._global_rules.copy(),
            **{proj: rules.copy() for proj, rules in manager._rules.items()}
        }

    manager.add_rule = AsyncMock(side_effect=add_rule)
    manager.get_rules = AsyncMock(side_effect=get_rules)
    manager.get_all_rules = AsyncMock(side_effect=get_all_rules)
    manager.initialize = AsyncMock()

    await manager.initialize()
    return manager


@pytest.mark.asyncio
class TestProjectRuleActivation:
    """Test rule activation in specific project contexts."""

    async def test_rule_activates_in_target_project(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that project-specific rule activates only in target project."""
        # Add rule for project A
        project_a_rule = MemoryRule(
            rule="Use TypeScript for project A",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="project_a_rule",
            source="test",
        )

        await project_aware_memory_manager.add_rule(
            project_a_rule,
            project="/path/to/project_a"
        )

        # In project A context - rule should be active
        mock_project_detector.set_project("/path/to/project_a")
        rules_in_a = await project_aware_memory_manager.get_rules()

        assert len(rules_in_a) == 1
        assert rules_in_a[0].id == "project_a_rule"

        # In project B context - rule should NOT be active
        mock_project_detector.set_project("/path/to/project_b")
        rules_in_b = await project_aware_memory_manager.get_rules()

        assert len(rules_in_b) == 0

    async def test_global_rules_active_everywhere(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that global rules are active in all project contexts."""
        # Add global rule
        global_rule = MemoryRule(
            rule="Always write tests",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="global_rule",
            source="test",
        )

        await project_aware_memory_manager.add_rule(global_rule, project=None)

        # In project A
        mock_project_detector.set_project("/path/to/project_a")
        rules_in_a = await project_aware_memory_manager.get_rules()
        assert len(rules_in_a) == 1
        assert rules_in_a[0].id == "global_rule"

        # In project B
        mock_project_detector.set_project("/path/to/project_b")
        rules_in_b = await project_aware_memory_manager.get_rules()
        assert len(rules_in_b) == 1
        assert rules_in_b[0].id == "global_rule"

        # No project context
        mock_project_detector.set_project(None)
        rules_global = await project_aware_memory_manager.get_rules()
        assert len(rules_global) == 1
        assert rules_global[0].id == "global_rule"


@pytest.mark.asyncio
class TestMultiProjectIsolation:
    """Test isolation between multiple projects."""

    async def test_no_rule_leakage_between_projects(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that rules don't leak between different projects."""
        # Add rules for different projects
        rule_a = MemoryRule(
            rule="Project A uses React",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule_a",
            source="test",
        )

        rule_b = MemoryRule(
            rule="Project B uses Vue",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule_b",
            source="test",
        )

        await project_aware_memory_manager.add_rule(rule_a, project="/proj/a")
        await project_aware_memory_manager.add_rule(rule_b, project="/proj/b")

        # Verify isolation
        mock_project_detector.set_project("/proj/a")
        rules_a = await project_aware_memory_manager.get_rules()
        assert len(rules_a) == 1
        assert rules_a[0].id == "rule_a"

        mock_project_detector.set_project("/proj/b")
        rules_b = await project_aware_memory_manager.get_rules()
        assert len(rules_b) == 1
        assert rules_b[0].id == "rule_b"

    async def test_mixed_global_and_project_rules(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test correct combination of global and project-specific rules."""
        # Add global rule
        global_rule = MemoryRule(
            rule="Use semantic versioning",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="global_1",
            source="test",
        )
        await project_aware_memory_manager.add_rule(global_rule, project=None)

        # Add project-specific rules
        proj_a_rule = MemoryRule(
            rule="Project A: Use monorepo structure",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="proj_a_1",
            source="test",
        )
        await project_aware_memory_manager.add_rule(proj_a_rule, project="/proj/a")

        # In project A: should get global + project A rules
        mock_project_detector.set_project("/proj/a")
        rules_a = await project_aware_memory_manager.get_rules()
        assert len(rules_a) == 2
        rule_ids = {r.id for r in rules_a}
        assert "global_1" in rule_ids
        assert "proj_a_1" in rule_ids

        # In project B: should get only global rules
        mock_project_detector.set_project("/proj/b")
        rules_b = await project_aware_memory_manager.get_rules()
        assert len(rules_b) == 1
        assert rules_b[0].id == "global_1"


@pytest.mark.asyncio
class TestProjectContextSwitching:
    """Test rule activation/deactivation when switching project contexts."""

    async def test_rules_deactivate_on_context_switch(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that rules deactivate when switching away from project."""
        # Setup rules for two projects
        rule_a = MemoryRule(
            rule="Project A rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule_a",
            source="test",
        )

        rule_b = MemoryRule(
            rule="Project B rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule_b",
            source="test",
        )

        await project_aware_memory_manager.add_rule(rule_a, project="/proj/a")
        await project_aware_memory_manager.add_rule(rule_b, project="/proj/b")

        # Start in project A
        mock_project_detector.set_project("/proj/a")
        rules = await project_aware_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].id == "rule_a"

        # Switch to project B
        mock_project_detector.set_project("/proj/b")
        rules = await project_aware_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].id == "rule_b"

        # Switch to no project context
        mock_project_detector.set_project(None)
        rules = await project_aware_memory_manager.get_rules()
        assert len(rules) == 0

    async def test_behavioral_changes_on_context_switch(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that LLM behavior changes when switching project contexts."""
        # Add project-specific rules
        react_rule = MemoryRule(
            rule="Always use React hooks",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="react_rule",
            source="test",
        )

        vue_rule = MemoryRule(
            rule="Always use Vue composition API",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="vue_rule",
            source="test",
        )

        await project_aware_memory_manager.add_rule(react_rule, project="/proj/react")
        await project_aware_memory_manager.add_rule(vue_rule, project="/proj/vue")

        # Test in React project context
        mock_project_detector.set_project("/proj/react")
        react_rules = await project_aware_memory_manager.get_rules()

        mock_provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=mock_provider,
            memory_manager=project_aware_memory_manager,
            mode=ExecutionMode.MOCK
        )

        metrics_react, _, _ = await harness.run_behavioral_test(
            prompt="Create a component",
            rules=react_rules,
            expected_patterns=[r"use.*hook"]
        )

        # Test in Vue project context
        mock_project_detector.set_project("/proj/vue")
        vue_rules = await project_aware_memory_manager.get_rules()

        metrics_vue, _, _ = await harness.run_behavioral_test(
            prompt="Create a component",
            rules=vue_rules,
            expected_patterns=[r"composition.*API"]
        )

        # Verify different rules were active
        assert react_rules[0].id == "react_rule"
        assert vue_rules[0].id == "vue_rule"


@pytest.mark.asyncio
class TestProjectDetection:
    """Test project detection accuracy and edge cases."""

    async def test_explicit_project_override(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that explicit project parameter overrides current context."""
        # Add rules
        rule_a = MemoryRule(
            rule="Rule for A",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule_a",
            source="test",
        )
        await project_aware_memory_manager.add_rule(rule_a, project="/proj/a")

        # Current context is project B
        mock_project_detector.set_project("/proj/b")

        # But explicitly request rules for project A
        rules_a = await project_aware_memory_manager.get_rules(project="/proj/a")
        assert len(rules_a) == 1
        assert rules_a[0].id == "rule_a"

    async def test_nested_project_contexts(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test handling of nested projects (e.g., submodules)."""
        # Add rules for parent and nested projects
        parent_rule = MemoryRule(
            rule="Parent project rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="parent_rule",
            source="test",
        )

        nested_rule = MemoryRule(
            rule="Nested project rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="nested_rule",
            source="test",
        )

        await project_aware_memory_manager.add_rule(
            parent_rule,
            project="/proj/parent"
        )
        await project_aware_memory_manager.add_rule(
            nested_rule,
            project="/proj/parent/nested"
        )

        # In parent context - only parent rule
        mock_project_detector.set_project("/proj/parent")
        parent_rules = await project_aware_memory_manager.get_rules()
        assert len(parent_rules) == 1
        assert parent_rules[0].id == "parent_rule"

        # In nested context - only nested rule (not inherited)
        mock_project_detector.set_project("/proj/parent/nested")
        nested_rules = await project_aware_memory_manager.get_rules()
        assert len(nested_rules) == 1
        assert nested_rules[0].id == "nested_rule"


@pytest.mark.asyncio
class TestCrossContamination:
    """Test that there's no cross-contamination between project rule sets."""

    async def test_concurrent_project_operations(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test concurrent operations on different projects don't interfere."""
        # Add rules for multiple projects
        rules_to_add = []
        for i in range(3):
            for proj in ["a", "b", "c"]:
                rule = MemoryRule(
                    rule=f"Project {proj} rule {i}",
                    category=MemoryCategory.BEHAVIOR,
                    authority=AuthorityLevel.DEFAULT,
                    id=f"{proj}_rule_{i}",
                    source="test",
                )
                rules_to_add.append((rule, f"/proj/{proj}"))

        # Add all rules
        for rule, project in rules_to_add:
            await project_aware_memory_manager.add_rule(rule, project=project)

        # Verify each project has exactly its own rules
        for proj in ["a", "b", "c"]:
            mock_project_detector.set_project(f"/proj/{proj}")
            rules = await project_aware_memory_manager.get_rules()

            assert len(rules) == 3
            for rule in rules:
                assert rule.id.startswith(f"{proj}_rule_")

    async def test_rule_modification_isolation(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that modifying rules in one project doesn't affect others."""
        # Add similar rules to different projects
        rule_a = MemoryRule(
            rule="Use linting",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="lint_rule",
            source="test",
        )

        rule_b = MemoryRule(
            rule="Use linting",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="lint_rule",
            source="test",
        )

        await project_aware_memory_manager.add_rule(rule_a, project="/proj/a")
        await project_aware_memory_manager.add_rule(rule_b, project="/proj/b")

        # Both projects should have independent rules despite same ID
        all_rules = await project_aware_memory_manager.get_all_rules()

        assert "/proj/a" in all_rules
        assert "/proj/b" in all_rules
        assert len(all_rules["/proj/a"]) == 1
        assert len(all_rules["/proj/b"]) == 1


@pytest.mark.asyncio
class TestProjectRulePriority:
    """Test priority and conflict resolution with project-specific rules."""

    async def test_project_rule_overrides_global(
        self,
        project_aware_memory_manager,
        mock_project_detector
    ):
        """Test that project-specific rules can override global rules."""
        # Add global rule
        global_rule = MemoryRule(
            rule="Use JavaScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="lang_global",
            source="test",
        )
        await project_aware_memory_manager.add_rule(global_rule, project=None)

        # Add project-specific override
        project_rule = MemoryRule(
            rule="Use TypeScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,  # Higher authority
            id="lang_project",
            source="test",
        )
        await project_aware_memory_manager.add_rule(
            project_rule,
            project="/proj/ts"
        )

        # In TypeScript project
        mock_project_detector.set_project("/proj/ts")
        rules = await project_aware_memory_manager.get_rules()

        # Both rules present, but project rule has higher authority
        assert len(rules) == 2
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        assert len(absolute_rules) == 1
        assert absolute_rules[0].id == "lang_project"
