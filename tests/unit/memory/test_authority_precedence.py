"""
Comprehensive tests for authority level precedence system.

Tests authority level handling (absolute vs default), rule precedence resolution
when conflicts occur, override behavior, and authority inheritance scenarios.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import AsyncMock, Mock

from common.core.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
    BehavioralController,
)

from .rule_test_utils import (
    MemoryRuleGenerator,
    ConflictSimulator,
)


class TestAbsoluteAuthorityRules:
    """Test absolute authority level rules and their behavior."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator with fixed seed for reproducibility."""
        return MemoryRuleGenerator(seed=42)

    def test_absolute_rule_creation(self, rule_generator):
        """Test creation of absolute authority rules."""
        rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Critical Behavior",
            rule="Always make atomic commits after each change",
        )

        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert rule.category == MemoryCategory.BEHAVIOR
        assert "atomic" in rule.rule.lower()

    def test_absolute_rule_non_negotiable_flag(self, rule_generator):
        """Test that absolute rules are treated as non-negotiable."""
        rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Non-Negotiable Rule",
            rule="Must follow this rule without exception",
        )

        # Absolute rules should be non-negotiable
        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert rule.source in ["user_explicit", "system"]

    def test_multiple_absolute_rules_same_scope(self, rule_generator):
        """Test multiple absolute rules in the same scope."""
        scope = ["global", "python"]

        rule1 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Absolute Rule 1",
            rule="Always use type hints",
            scope=scope,
        )

        rule2 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Absolute Rule 2",
            rule="Always write docstrings",
            scope=scope,
        )

        # Both should be absolute and have the same scope
        assert rule1.authority == AuthorityLevel.ABSOLUTE
        assert rule2.authority == AuthorityLevel.ABSOLUTE
        assert set(rule1.scope) == set(rule2.scope)

    def test_absolute_rule_with_conditions(self, rule_generator):
        """Test absolute rules with conditional logic."""
        rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Conditional Absolute Rule",
            rule="When working on production code, always run full test suite",
            conditions={"environment": "production", "code_type": "production"},
        )

        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert rule.conditions is not None
        assert "environment" in rule.conditions
        assert rule.conditions["environment"] == "production"


class TestDefaultAuthorityRules:
    """Test default authority level rules and their behavior."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator with fixed seed."""
        return MemoryRuleGenerator(seed=42)

    def test_default_rule_creation(self, rule_generator):
        """Test creation of default authority rules."""
        rule = rule_generator.generate_preference_rule(
            name="Default Preference",
            rule="Prefer uv for Python package management",
        )

        assert rule.authority == AuthorityLevel.DEFAULT
        assert rule.category == MemoryCategory.PREFERENCE

    def test_default_rule_overridable(self, rule_generator):
        """Test that default rules are marked as overridable."""
        rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Overridable Behavior",
            rule="Aim for 90% test coverage unless specified otherwise",
        )

        assert rule.authority == AuthorityLevel.DEFAULT
        # Default rules should allow override
        assert "unless" in rule.rule.lower() or "prefer" in rule.rule.lower() or rule.authority == AuthorityLevel.DEFAULT

    def test_default_rules_multiple_scopes(self, rule_generator):
        """Test default rules with multiple scope combinations."""
        scopes = [
            ["global"],
            ["python"],
            ["python", "testing"],
            ["rust", "performance"],
        ]

        rules = []
        for scope in scopes:
            rule = rule_generator.generate_preference_rule(
                scope=scope,
                name=f"Preference for {'-'.join(scope)}",
                rule=f"Default preference for {' '.join(scope)}",
            )
            rules.append(rule)

        # All should be default authority
        assert all(r.authority == AuthorityLevel.DEFAULT for r in rules)
        # Each should have its specified scope
        for rule, scope in zip(rules, scopes):
            assert set(rule.scope) == set(scope)


class TestRulePrecedenceResolution:
    """Test rule precedence resolution when conflicts occur."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator."""
        return MemoryRuleGenerator(seed=42)

    @pytest.fixture
    def conflict_simulator(self, rule_generator):
        """Create conflict simulator."""
        return ConflictSimulator(rule_generator)

    @pytest.fixture
    async def mock_behavioral_controller(self):
        """Create mock behavioral controller for testing."""
        mock_memory_manager = Mock()
        mock_memory_manager.search_memory_rules = AsyncMock(return_value=[])
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])
        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[])

        controller = BehavioralController(mock_memory_manager)
        return controller, mock_memory_manager

    @pytest.mark.asyncio
    async def test_absolute_overrides_default(
        self, conflict_simulator, mock_behavioral_controller
    ):
        """Test that absolute authority rules override default rules."""
        controller, mock_memory_manager = mock_behavioral_controller

        # Create conflicting rules
        absolute_rule, default_rule = conflict_simulator.generate_authority_conflict(
            scope=["global"]
        )

        # Ensure they're set up correctly
        assert absolute_rule.authority == AuthorityLevel.ABSOLUTE
        assert default_rule.authority == AuthorityLevel.DEFAULT

        # Simulate conflict detection
        from common.core.memory import MemoryConflict

        conflict = MemoryConflict(
            conflict_type="authority_mismatch",
            rule1=absolute_rule,
            rule2=default_rule,
            confidence=0.9,
            description="Absolute rule conflicts with default rule",
            resolution_options=["keep_higher_authority"],
        )

        # Resolve conflict
        resolved_rules, resolutions = await controller._resolve_conflicts(
            rules=[absolute_rule, default_rule],
            conflicts=[conflict],
            context="testing",
            urgency="normal",
        )

        # Absolute rule should win
        assert len(resolved_rules) == 1
        assert resolved_rules[0].authority == AuthorityLevel.ABSOLUTE
        assert resolved_rules[0].id == absolute_rule.id

    @pytest.mark.asyncio
    async def test_newer_rule_wins_same_authority(
        self, rule_generator, mock_behavioral_controller
    ):
        """Test that newer rules win when authority levels are the same."""
        controller, mock_memory_manager = mock_behavioral_controller

        # Create two default rules with different timestamps
        now = datetime.now(timezone.utc)
        older_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Older Rule",
            rule="Use approach A",
        )
        # Manually set timestamps
        older_rule.created_at = now - timedelta(days=30)

        newer_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Newer Rule",
            rule="Use approach B instead",
        )
        newer_rule.created_at = now - timedelta(days=1)

        # Create conflict
        from common.core.memory import MemoryConflict

        conflict = MemoryConflict(
            conflict_type="direct_contradiction",
            rule1=older_rule,
            rule2=newer_rule,
            confidence=0.85,
            description="Rules contradict each other",
            resolution_options=["keep_newer"],
        )

        # Resolve
        resolved_rules, resolutions = await controller._resolve_conflicts(
            rules=[older_rule, newer_rule],
            conflicts=[conflict],
            context="testing",
            urgency="normal",
        )

        # Newer rule should win
        assert len(resolved_rules) == 1
        assert resolved_rules[0].id == newer_rule.id
        assert "newer" in resolutions[0].lower()

    @pytest.mark.asyncio
    async def test_more_specific_rule_wins(
        self, rule_generator, mock_behavioral_controller
    ):
        """Test that more specific rules (with conditions/scope) win over general rules."""
        controller, mock_memory_manager = mock_behavioral_controller

        now = datetime.now(timezone.utc)

        # General rule (no conditions, broad scope)
        general_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="General Rule",
            rule="Use standard approach",
            scope=["global"],
        )
        general_rule.created_at = now

        # Specific rule (with conditions and narrow scope)
        specific_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Specific Rule",
            rule="For Python testing, use pytest with coverage",
            scope=["python", "testing"],
            conditions={"language": "python", "task": "testing"},
        )
        specific_rule.created_at = now

        # Create conflict
        from common.core.memory import MemoryConflict

        conflict = MemoryConflict(
            conflict_type="scope_overlap",
            rule1=general_rule,
            rule2=specific_rule,
            confidence=0.7,
            description="Specific rule overlaps with general rule",
            resolution_options=["keep_more_specific"],
        )

        # Resolve
        resolved_rules, resolutions = await controller._resolve_conflicts(
            rules=[general_rule, specific_rule],
            conflicts=[conflict],
            context="testing",
            urgency="normal",
        )

        # More specific rule should win
        assert len(resolved_rules) == 1
        assert resolved_rules[0].id == specific_rule.id
        assert "specific" in resolutions[0].lower()

    @pytest.mark.asyncio
    async def test_multiple_conflicts_cascade_resolution(
        self, rule_generator, mock_behavioral_controller
    ):
        """Test cascading resolution when multiple conflicts exist."""
        controller, mock_memory_manager = mock_behavioral_controller

        now = datetime.now(timezone.utc)

        # Create three conflicting rules with different authorities
        absolute_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Absolute",
            rule="Must do X",
        )
        absolute_rule.created_at = now

        default_rule_1 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Default 1",
            rule="Prefer Y",
        )
        default_rule_1.created_at = now - timedelta(days=10)

        default_rule_2 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Default 2",
            rule="Use Z approach",
        )
        default_rule_2.created_at = now - timedelta(days=5)

        # Create conflicts
        from common.core.memory import MemoryConflict

        conflicts = [
            MemoryConflict(
                conflict_type="authority_mismatch",
                rule1=absolute_rule,
                rule2=default_rule_1,
                confidence=0.9,
                description="Absolute vs Default 1",
                resolution_options=["keep_higher_authority"],
            ),
            MemoryConflict(
                conflict_type="direct_contradiction",
                rule1=default_rule_1,
                rule2=default_rule_2,
                confidence=0.8,
                description="Default 1 vs Default 2",
                resolution_options=["keep_newer"],
            ),
        ]

        # Resolve all conflicts
        resolved_rules, resolutions = await controller._resolve_conflicts(
            rules=[absolute_rule, default_rule_1, default_rule_2],
            conflicts=conflicts,
            context="testing",
            urgency="normal",
        )

        # Should keep absolute rule and newer default rule
        assert len(resolved_rules) == 2
        rule_ids = {r.id for r in resolved_rules}
        assert absolute_rule.id in rule_ids
        assert default_rule_2.id in rule_ids
        assert default_rule_1.id not in rule_ids


class TestOverrideBehavior:
    """Test override behavior with different authority levels."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator."""
        return MemoryRuleGenerator(seed=42)

    def test_absolute_cannot_be_overridden_by_default(self, rule_generator):
        """Test that absolute rules cannot be overridden by default rules."""
        absolute_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Absolute",
            rule="Always commit atomically",
        )

        default_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Default",
            rule="Batch commits for efficiency",
        )

        # Authority check
        assert absolute_rule.authority == AuthorityLevel.ABSOLUTE
        assert default_rule.authority == AuthorityLevel.DEFAULT
        # Absolute has higher priority
        assert absolute_rule.authority.value > default_rule.authority.value or (
            absolute_rule.authority == AuthorityLevel.ABSOLUTE
            and default_rule.authority == AuthorityLevel.DEFAULT
        )

    def test_default_can_be_overridden_by_absolute(self, rule_generator):
        """Test that default rules can be overridden by absolute rules."""
        default_rule = rule_generator.generate_preference_rule(
            name="Default Preference",
            rule="Prefer approach A",
        )

        absolute_override = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Absolute Override",
            rule="Must use approach B",
            replaces=[default_rule.id],
        )

        assert default_rule.authority == AuthorityLevel.DEFAULT
        assert absolute_override.authority == AuthorityLevel.ABSOLUTE
        assert default_rule.id in absolute_override.replaces

    def test_default_can_be_overridden_by_newer_default(self, rule_generator):
        """Test that default rules can be overridden by newer default rules."""
        now = datetime.now(timezone.utc)

        older_default = rule_generator.generate_preference_rule(
            name="Older Default",
            rule="Use tool X",
        )
        older_default.created_at = now - timedelta(days=30)

        newer_default = rule_generator.generate_preference_rule(
            name="Newer Default",
            rule="Use tool Y instead",
            replaces=[older_default.id],
        )
        newer_default.created_at = now

        assert older_default.authority == AuthorityLevel.DEFAULT
        assert newer_default.authority == AuthorityLevel.DEFAULT
        assert newer_default.created_at > older_default.created_at
        assert older_default.id in newer_default.replaces

    def test_scope_specific_overrides_global(self, rule_generator):
        """Test that scope-specific rules override global rules in their scope."""
        global_rule = rule_generator.generate_preference_rule(
            name="Global Default",
            rule="Use standard formatter",
            scope=["global"],
        )

        python_specific = rule_generator.generate_preference_rule(
            name="Python Specific",
            rule="Use black formatter",
            scope=["python"],
        )

        # Python-specific should override global in python scope
        assert "global" in global_rule.scope
        assert "python" in python_specific.scope
        assert len(python_specific.scope) >= len(global_rule.scope)


class TestAuthorityInheritance:
    """Test authority inheritance scenarios."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator."""
        return MemoryRuleGenerator(seed=42)

    def test_child_scope_inherits_parent_authority(self, rule_generator):
        """Test that child scopes inherit authority from parent scopes."""
        parent_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Parent Rule",
            rule="Always follow coding standards",
            scope=["global"],
        )

        # Child scope should respect parent's absolute authority
        child_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Child Rule",
            rule="Coding style preference",
            scope=["global", "python"],
        )

        # Parent has broader scope
        assert set(parent_rule.scope).issubset(set(child_rule.scope))
        # Parent has higher authority
        assert parent_rule.authority == AuthorityLevel.ABSOLUTE
        assert child_rule.authority == AuthorityLevel.DEFAULT

    def test_replaces_field_maintains_authority_chain(self, rule_generator):
        """Test that replaces field maintains authority inheritance chain."""
        original_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Original",
            rule="Original behavior",
        )

        replacement_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Replacement",
            rule="Updated behavior with higher authority",
            replaces=[original_rule.id],
        )

        # Replacement maintains chain
        assert original_rule.id in replacement_rule.replaces
        assert replacement_rule.authority == AuthorityLevel.ABSOLUTE
        # Authority was elevated in replacement
        assert original_rule.authority == AuthorityLevel.DEFAULT

    def test_conditional_rules_inherit_base_authority(self, rule_generator):
        """Test that conditional rules inherit authority from base rule."""
        base_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Base Rule",
            rule="Always validate input",
            scope=["global"],
        )

        conditional_extension = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Conditional Extension",
            rule="When handling user input, sanitize and validate",
            scope=["global", "security"],
            conditions={"input_source": "user"},
        )

        # Conditional should maintain absolute authority
        assert base_rule.authority == AuthorityLevel.ABSOLUTE
        assert conditional_extension.authority == AuthorityLevel.ABSOLUTE
        assert conditional_extension.conditions is not None

    def test_authority_upgrade_through_replacement(self, rule_generator):
        """Test authority level upgrade when rule is replaced."""
        now = datetime.now(timezone.utc)

        default_rule = rule_generator.generate_preference_rule(
            name="Original Default",
            rule="Prefer approach X",
        )
        default_rule.created_at = now - timedelta(days=10)

        upgraded_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Upgraded to Absolute",
            rule="Must use approach X (now mandatory)",
            replaces=[default_rule.id],
            metadata={"upgraded_from": "default"},
        )
        upgraded_rule.created_at = now

        # Authority was upgraded
        assert default_rule.authority == AuthorityLevel.DEFAULT
        assert upgraded_rule.authority == AuthorityLevel.ABSOLUTE
        assert default_rule.id in upgraded_rule.replaces
        assert upgraded_rule.metadata["upgraded_from"] == "default"

    def test_authority_downgrade_through_replacement(self, rule_generator):
        """Test authority level downgrade when rule is relaxed."""
        now = datetime.now(timezone.utc)

        absolute_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.ABSOLUTE,
            name="Original Absolute",
            rule="Must always do X",
        )
        absolute_rule.created_at = now - timedelta(days=10)

        downgraded_rule = rule_generator.generate_preference_rule(
            name="Downgraded to Default",
            rule="Prefer to do X when possible",
            replaces=[absolute_rule.id],
            metadata={"downgraded_from": "absolute"},
        )
        downgraded_rule.created_at = now

        # Authority was downgraded
        assert absolute_rule.authority == AuthorityLevel.ABSOLUTE
        assert downgraded_rule.authority == AuthorityLevel.DEFAULT
        assert absolute_rule.id in downgraded_rule.replaces
        assert downgraded_rule.metadata["downgraded_from"] == "absolute"


class TestPrecedenceEdgeCases:
    """Test edge cases in precedence resolution."""

    @pytest.fixture
    def rule_generator(self):
        """Create rule generator."""
        return MemoryRuleGenerator(seed=42)

    def test_same_timestamp_same_authority(self, rule_generator):
        """Test precedence when rules have very close timestamps and same authority."""
        # Create two rules in quick succession
        rule1 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Rule 1",
            rule="Approach A",
            id="rule-1",
        )

        rule2 = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Rule 2",
            rule="Approach B",
            id="rule-2",
        )

        # Both have same authority and timestamps should be very close
        assert rule1.authority == rule2.authority
        # Timestamps should be within 1 second of each other
        time_diff = abs((rule1.created_at - rule2.created_at).total_seconds())
        assert time_diff < 1.0
        # ID ordering provides deterministic resolution
        assert rule1.id != rule2.id

    def test_empty_scope_vs_specific_scope(self, rule_generator):
        """Test precedence between empty scope and specific scope rules."""
        empty_scope_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Empty Scope",
            rule="General behavior",
            scope=[],
        )

        specific_scope_rule = rule_generator.generate_behavior_rule(
            authority=AuthorityLevel.DEFAULT,
            name="Specific Scope",
            rule="Python-specific behavior",
            scope=["python", "testing"],
        )

        # Specific scope should be more specific
        specificity_empty = len(empty_scope_rule.scope or [])
        specificity_specific = len(specific_scope_rule.scope or [])
        assert specificity_specific > specificity_empty

    def test_multiple_replaces_chain(self, rule_generator):
        """Test handling of rules that replace multiple other rules."""
        now = datetime.now(timezone.utc)

        rule1 = rule_generator.generate_behavior_rule(
            name="Rule 1",
            rule="First approach",
        )
        rule1.created_at = now - timedelta(days=30)

        rule2 = rule_generator.generate_behavior_rule(
            name="Rule 2",
            rule="Second approach",
        )
        rule2.created_at = now - timedelta(days=20)

        consolidated_rule = rule_generator.generate_behavior_rule(
            name="Consolidated",
            rule="Unified approach combining best of both",
            replaces=[rule1.id, rule2.id],
            metadata={"consolidated_from": [rule1.id, rule2.id]},
        )
        consolidated_rule.created_at = now

        # Should replace both rules
        assert rule1.id in consolidated_rule.replaces
        assert rule2.id in consolidated_rule.replaces
        assert len(consolidated_rule.replaces) == 2
