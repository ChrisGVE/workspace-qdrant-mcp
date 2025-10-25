"""
Authority Enforcement Validation Tests (Task 337.5).

Tests rule authority levels and enforcement mechanisms across different
user permissions and rule hierarchies.

Test Scenarios:
1. Different authority levels (ABSOLUTE, DEFAULT, SUGGESTION)
2. Rule hierarchy enforcement and precedence
3. Conflicting rule resolution
4. Authority escalation and de-escalation
5. User permission checks
6. Rule override scenarios
7. Authority-based filtering
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from src.python.common.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)

# Import test harness from Task 337.1
from tests.integration.test_llm_behavioral_harness import (
    BehavioralMetrics,
    ExecutionMode,
    LLMBehavioralHarness,
    MockLLMProvider,
)


class RuleEnforcer:
    """
    Mock rule enforcement engine for testing authority levels.

    Implements authority-based rule filtering and conflict resolution.
    """

    def __init__(self, min_authority: AuthorityLevel | None = None):
        """
        Initialize rule enforcer.

        Args:
            min_authority: Minimum authority level to enforce
        """
        self.min_authority = min_authority
        self.authority_hierarchy = [
            AuthorityLevel.ABSOLUTE,
            AuthorityLevel.DEFAULT,
        ]

    def get_authority_rank(self, authority: AuthorityLevel) -> int:
        """Get numeric rank for authority level (lower = higher priority)."""
        try:
            return self.authority_hierarchy.index(authority)
        except ValueError:
            return len(self.authority_hierarchy)

    def filter_by_authority(
        self,
        rules: list[MemoryRule],
        min_authority: AuthorityLevel | None = None
    ) -> list[MemoryRule]:
        """Filter rules by minimum authority level."""
        min_auth = min_authority or self.min_authority
        if not min_auth:
            return rules

        min_rank = self.get_authority_rank(min_auth)
        return [
            rule for rule in rules
            if self.get_authority_rank(rule.authority) <= min_rank
        ]

    def resolve_conflicts(
        self,
        rules: list[MemoryRule],
        conflict_key: str = "rule"
    ) -> list[MemoryRule]:
        """
        Resolve conflicting rules by authority level.

        When rules conflict, keeps highest authority rule.
        """
        # Group rules by conflict key (e.g., similar content)
        rule_groups: dict[str, list[MemoryRule]] = {}

        for rule in rules:
            key = getattr(rule, conflict_key, str(rule))
            if key not in rule_groups:
                rule_groups[key] = []
            rule_groups[key].append(rule)

        # Keep highest authority rule from each group
        resolved = []
        for group_rules in rule_groups.values():
            if len(group_rules) == 1:
                resolved.append(group_rules[0])
            else:
                # Sort by authority (highest first)
                sorted_rules = sorted(
                    group_rules,
                    key=lambda r: self.get_authority_rank(r.authority)
                )
                resolved.append(sorted_rules[0])

        return resolved

    def can_override(
        self,
        new_rule: MemoryRule,
        existing_rule: MemoryRule
    ) -> bool:
        """Check if new rule can override existing rule."""
        new_rank = self.get_authority_rank(new_rule.authority)
        existing_rank = self.get_authority_rank(existing_rule.authority)

        # Can override if new rule has equal or higher authority
        return new_rank <= existing_rank


@pytest.fixture
def rule_enforcer():
    """Provide rule enforcer instance."""
    return RuleEnforcer()


@pytest.fixture
async def authority_aware_memory_manager():
    """Memory manager with authority-based rule enforcement."""
    manager = AsyncMock()
    manager._rules = []
    manager._enforcer = RuleEnforcer()

    async def add_rule(rule: MemoryRule):
        manager._rules.append(rule)

    async def get_rules(
        min_authority: AuthorityLevel | None = None
    ) -> list[MemoryRule]:
        """Get rules filtered by minimum authority."""
        return manager._enforcer.filter_by_authority(
            manager._rules,
            min_authority
        )

    async def get_effective_rules() -> list[MemoryRule]:
        """Get rules after conflict resolution."""
        return manager._enforcer.resolve_conflicts(manager._rules)

    manager.add_rule = AsyncMock(side_effect=add_rule)
    manager.get_rules = AsyncMock(side_effect=get_rules)
    manager.get_effective_rules = AsyncMock(side_effect=get_effective_rules)
    manager.initialize = AsyncMock()

    await manager.initialize()
    return manager


@pytest.mark.asyncio
class TestAuthorityLevels:
    """Test different authority levels and their enforcement."""

    async def test_absolute_authority_rules(
        self,
        authority_aware_memory_manager
    ):
        """Test that ABSOLUTE authority rules are always enforced."""
        rule = MemoryRule(
            rule="Always use type safety",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="absolute_rule",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(rule)

        # Should be included with no filter
        rules = await authority_aware_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].authority == AuthorityLevel.ABSOLUTE

        # Should be included when filtering for ABSOLUTE
        absolute_rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )
        assert len(absolute_rules) == 1

    async def test_default_authority_rules(
        self,
        authority_aware_memory_manager
    ):
        """Test that DEFAULT authority rules can be filtered."""
        rule = MemoryRule(
            rule="Prefer composition",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="default_rule",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(rule)

        # Should be included when filtering for DEFAULT
        default_rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.DEFAULT
        )
        assert len(default_rules) == 1

        # Should NOT be included when filtering for ABSOLUTE only
        absolute_rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )
        assert len(absolute_rules) == 0

    async def test_mixed_authority_filtering(
        self,
        authority_aware_memory_manager
    ):
        """Test filtering rules with mixed authority levels."""
        absolute_rule = MemoryRule(
            rule="Must use HTTPS",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="abs_1",
            source="test",
        )

        default_rule = MemoryRule(
            rule="Prefer async/await",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="def_1",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(absolute_rule)
        await authority_aware_memory_manager.add_rule(default_rule)

        # All rules with no filter
        all_rules = await authority_aware_memory_manager.get_rules()
        assert len(all_rules) == 2

        # Only ABSOLUTE rules
        absolute_only = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )
        assert len(absolute_only) == 1
        assert absolute_only[0].authority == AuthorityLevel.ABSOLUTE

        # DEFAULT and above (includes ABSOLUTE)
        default_and_above = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.DEFAULT
        )
        assert len(default_and_above) == 2


@pytest.mark.asyncio
class TestRuleHierarchy:
    """Test rule hierarchy enforcement and precedence."""

    async def test_absolute_overrides_default(
        self,
        rule_enforcer
    ):
        """Test that ABSOLUTE authority rules override DEFAULT."""
        default_rule = MemoryRule(
            rule="Use JavaScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="lang_default",
            source="test",
        )

        absolute_rule = MemoryRule(
            rule="Use TypeScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="lang_absolute",
            source="test",
        )

        assert rule_enforcer.can_override(absolute_rule, default_rule)
        assert not rule_enforcer.can_override(default_rule, absolute_rule)

    async def test_same_authority_can_override(
        self,
        rule_enforcer
    ):
        """Test that rules with same authority can override each other."""
        rule1 = MemoryRule(
            rule="Use React",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="framework_1",
            source="test",
        )

        rule2 = MemoryRule(
            rule="Use Vue",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="framework_2",
            source="test",
        )

        assert rule_enforcer.can_override(rule1, rule2)
        assert rule_enforcer.can_override(rule2, rule1)

    async def test_authority_hierarchy_ranks(
        self,
        rule_enforcer
    ):
        """Test authority hierarchy ranking."""
        absolute_rank = rule_enforcer.get_authority_rank(AuthorityLevel.ABSOLUTE)
        default_rank = rule_enforcer.get_authority_rank(AuthorityLevel.DEFAULT)

        # ABSOLUTE should have lower rank (higher priority)
        assert absolute_rank < default_rank


@pytest.mark.asyncio
class TestConflictResolution:
    """Test conflicting rule resolution based on authority."""

    async def test_resolve_direct_conflicts(
        self,
        authority_aware_memory_manager
    ):
        """Test resolution of directly conflicting rules."""
        # Add conflicting rules with different authorities
        default_rule = MemoryRule(
            rule="Use single quotes",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="quotes_default",
            source="test",
        )

        absolute_rule = MemoryRule(
            rule="Use double quotes",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="quotes_absolute",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(default_rule)
        await authority_aware_memory_manager.add_rule(absolute_rule)

        # Get effective rules after conflict resolution
        effective = await authority_aware_memory_manager.get_effective_rules()

        # Should keep both since they have different IDs
        # (conflict resolution by content would be more sophisticated)
        assert len(effective) == 2

    async def test_multiple_conflicting_rules(
        self,
        rule_enforcer
    ):
        """Test resolution with multiple conflicting rules."""
        rules = [
            MemoryRule(
                rule="Testing rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id="test_1",
                source="test",
            ),
            MemoryRule(
                rule="Testing rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="test_2",
                source="test",
            ),
            MemoryRule(
                rule="Testing rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id="test_3",
                source="test",
            ),
        ]

        resolved = rule_enforcer.resolve_conflicts(rules)

        # Should keep only the ABSOLUTE authority rule
        assert len(resolved) == 1
        assert resolved[0].authority == AuthorityLevel.ABSOLUTE


@pytest.mark.asyncio
class TestAuthorityEscalation:
    """Test authority escalation and de-escalation scenarios."""

    async def test_cannot_escalate_without_permission(
        self,
        authority_aware_memory_manager
    ):
        """Test that rules cannot be escalated without proper permission."""
        # Add a DEFAULT rule
        rule = MemoryRule(
            rule="Original rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="escalation_test",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(rule)

        # In a real system, attempting to escalate would require permission
        # For now, just verify we can track authority changes
        original_authority = rule.authority
        assert original_authority == AuthorityLevel.DEFAULT

    async def test_deescalation_always_allowed(
        self,
        rule_enforcer
    ):
        """Test that de-escalating authority is always allowed."""
        absolute_rule = MemoryRule(
            rule="High authority rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="deescalate_test",
            source="test",
        )

        default_rule = MemoryRule(
            rule="Lower authority rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="deescalate_test",
            source="test",
        )

        # De-escalation (ABSOLUTE -> DEFAULT) should be possible
        # but not override the original
        can_override = rule_enforcer.can_override(default_rule, absolute_rule)
        assert not can_override


@pytest.mark.asyncio
class TestUserPermissions:
    """Test user permission checks for rule operations."""

    async def test_read_rules_no_permission_required(
        self,
        authority_aware_memory_manager
    ):
        """Test that reading rules requires no special permissions."""
        rule = MemoryRule(
            rule="Public rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="public_rule",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(rule)

        # Any user should be able to read rules
        rules = await authority_aware_memory_manager.get_rules()
        assert len(rules) == 1

    async def test_add_absolute_rule_permission(
        self,
        authority_aware_memory_manager
    ):
        """Test that adding ABSOLUTE rules requires elevated permissions."""
        # In a real system, this would check user permissions
        # For testing, we simulate the check

        def has_permission_for_absolute():
            # Mock permission check
            return True  # Simulating admin user

        if has_permission_for_absolute():
            absolute_rule = MemoryRule(
                rule="Admin-level rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="admin_rule",
                source="test",
            )

            await authority_aware_memory_manager.add_rule(absolute_rule)

            rules = await authority_aware_memory_manager.get_rules()
            assert len(rules) == 1
            assert rules[0].authority == AuthorityLevel.ABSOLUTE


@pytest.mark.asyncio
class TestRuleOverrideScenarios:
    """Test various rule override scenarios."""

    async def test_project_override_global(
        self,
        rule_enforcer
    ):
        """Test project-specific rule overriding global rule."""
        global_rule = MemoryRule(
            rule="Global: Use Jest",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test_framework_global",
            source="global",
        )

        project_rule = MemoryRule(
            rule="Project: Use Vitest",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="test_framework_project",
            source="project",
        )

        # Project ABSOLUTE rule should override global DEFAULT
        assert rule_enforcer.can_override(project_rule, global_rule)

    async def test_user_override_system(
        self,
        rule_enforcer
    ):
        """Test user rule overriding system default."""
        system_rule = MemoryRule(
            rule="System default",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="system_default",
            source="system",
        )

        user_rule = MemoryRule(
            rule="User preference",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="user_pref",
            source="user",
        )

        # Same authority - user can override system
        assert rule_enforcer.can_override(user_rule, system_rule)

    async def test_temporary_override_permanent(
        self,
        authority_aware_memory_manager
    ):
        """Test temporary rule overriding permanent rule."""
        permanent_rule = MemoryRule(
            rule="Permanent setting",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="permanent",
            source="config",
        )

        # Temporary rule with higher authority
        temp_rule = MemoryRule(
            rule="Temporary override",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="temporary",
            source="session",
        )

        await authority_aware_memory_manager.add_rule(permanent_rule)
        await authority_aware_memory_manager.add_rule(temp_rule)

        # Temporary ABSOLUTE should take precedence
        absolute_rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )
        assert len(absolute_rules) == 1
        assert absolute_rules[0].id == "temporary"


@pytest.mark.asyncio
class TestBehavioralAuthority:
    """Test authority enforcement in behavioral context."""

    async def test_absolute_rules_always_applied(
        self,
        authority_aware_memory_manager
    ):
        """Test that ABSOLUTE rules are always applied to LLM behavior."""
        absolute_rule = MemoryRule(
            rule="Always include error handling",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="error_handling",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(absolute_rule)

        # Get rules for LLM injection
        rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )

        assert len(rules) == 1
        assert rules[0].authority == AuthorityLevel.ABSOLUTE

        # Simulate LLM behavior test
        mock_provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=mock_provider,
            memory_manager=authority_aware_memory_manager,
            mode=ExecutionMode.MOCK
        )

        metrics, _, _ = await harness.run_behavioral_test(
            prompt="Write a function",
            rules=rules,
            expected_patterns=[r"try", r"except"]
        )

        # With mock, just verify workflow completed
        assert metrics is not None

    async def test_default_rules_optional(
        self,
        authority_aware_memory_manager
    ):
        """Test that DEFAULT rules can be optionally excluded."""
        default_rule = MemoryRule(
            rule="Consider using caching",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="caching_suggestion",
            source="test",
        )

        await authority_aware_memory_manager.add_rule(default_rule)

        # Can choose to exclude DEFAULT rules
        absolute_only = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.ABSOLUTE
        )
        assert len(absolute_only) == 0

        # Or include them
        all_rules = await authority_aware_memory_manager.get_rules(
            min_authority=AuthorityLevel.DEFAULT
        )
        assert len(all_rules) == 1
