"""
Unit tests for authority level filtering system.
"""

from datetime import datetime, timezone, timedelta

import pytest

from src.python.common.core.context_injection.authority_filter import (
    AuthorityFilter,
    AuthorityPrecedence,
    FilteredRules,
    RuleHierarchy,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


class TestAuthorityFilter:
    """Test AuthorityFilter class."""

    @pytest.fixture
    def authority_filter(self):
        """Create AuthorityFilter instance."""
        return AuthorityFilter(enable_precedence_resolution=True)

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules with different authorities and scopes."""
        now = datetime.now(timezone.utc)

        return [
            # Global absolute rule
            MemoryRule(
                id="global_abs",
                category=MemoryCategory.BEHAVIOR,
                name="Global Absolute",
                rule="Always use type hints",
                authority=AuthorityLevel.ABSOLUTE,
                scope=[],
                created_at=now,
                updated_at=now,
                metadata={"priority": 90},
            ),
            # Global default rule
            MemoryRule(
                id="global_def",
                category=MemoryCategory.PREFERENCE,
                name="Global Default",
                rule="Prefer pytest",
                authority=AuthorityLevel.DEFAULT,
                scope=[],
                created_at=now,
                updated_at=now,
                metadata={"priority": 70},
            ),
            # Project-specific absolute rule (no scope - project-wide)
            MemoryRule(
                id="proj_abs",
                category=MemoryCategory.BEHAVIOR,
                name="Project Absolute",
                rule="Use specific format",
                authority=AuthorityLevel.ABSOLUTE,
                scope=[],  # No scope - applies to whole project
                created_at=now,
                updated_at=now,
                metadata={"priority": 85, "project_id": "proj123"},
            ),
            # Project-specific default rule (no scope - project-wide)
            MemoryRule(
                id="proj_def",
                category=MemoryCategory.PREFERENCE,
                name="Project Default",
                rule="Use uv for dependencies",
                authority=AuthorityLevel.DEFAULT,
                scope=[],  # No scope - applies to whole project
                created_at=now,
                updated_at=now,
                metadata={"priority": 75, "project_id": "proj123"},
            ),
            # Local scope absolute rule
            MemoryRule(
                id="local_abs",
                category=MemoryCategory.BEHAVIOR,
                name="Local Absolute",
                rule="Use docstrings",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["testing", "python"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 95},
            ),
        ]

    def test_filter_by_authority_basic(self, authority_filter, sample_rules):
        """Test basic authority filtering."""
        result = authority_filter.filter_by_authority(sample_rules)

        assert isinstance(result, FilteredRules)
        assert len(result.absolute_rules) == 3  # global_abs, proj_abs, local_abs
        assert len(result.default_rules) == 2  # global_def, proj_def

    def test_filter_by_authority_with_project(self, authority_filter, sample_rules):
        """Test authority filtering with project context."""
        result = authority_filter.filter_by_authority(
            sample_rules, project_id="proj123"
        )

        # Verify hierarchy grouping
        assert RuleHierarchy.GLOBAL in result.rules_by_hierarchy
        assert RuleHierarchy.PROJECT in result.rules_by_hierarchy
        assert RuleHierarchy.LOCAL in result.rules_by_hierarchy

        # Check project-specific rules are in project hierarchy
        project_rules = result.rules_by_hierarchy[RuleHierarchy.PROJECT]
        assert any(r.id == "proj_abs" for r in project_rules)
        assert any(r.id == "proj_def" for r in project_rules)

    def test_filter_by_authority_with_scope(self, authority_filter, sample_rules):
        """Test authority filtering with scope context."""
        result = authority_filter.filter_by_authority(
            sample_rules, scope=["python", "testing"]
        )

        # Local rules should be identified
        local_rules = result.rules_by_hierarchy[RuleHierarchy.LOCAL]
        assert any(r.id == "local_abs" for r in local_rules)

    def test_apply_authority_precedence(self, authority_filter, sample_rules):
        """Test authority precedence application."""
        resolved = authority_filter.apply_authority_precedence(sample_rules)

        # All rules should be included (no conflicts in this set)
        assert len(resolved) == 5

        # Verify sorting order (absolute first, then by hierarchy)
        assert resolved[0].authority == AuthorityLevel.ABSOLUTE
        assert resolved[1].authority == AuthorityLevel.ABSOLUTE
        assert resolved[2].authority == AuthorityLevel.ABSOLUTE

    def test_apply_authority_precedence_with_conflicts(self, authority_filter):
        """Test precedence with conflicting rules."""
        now = datetime.now(timezone.utc)

        # Create conflicting rules (same category, same scope)
        rules = [
            MemoryRule(
                id="abs_rule",
                category=MemoryCategory.BEHAVIOR,
                name="Absolute Rule",
                rule="Use absolute approach",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 90},
            ),
            MemoryRule(
                id="def_rule",
                category=MemoryCategory.BEHAVIOR,
                name="Default Rule",
                rule="Use default approach",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 80},
            ),
        ]

        resolved = authority_filter.apply_authority_precedence(rules)

        # Should only keep absolute rule (higher precedence)
        assert len(resolved) == 1
        assert resolved[0].id == "abs_rule"

    def test_validate_authority_level(self, authority_filter):
        """Test authority level validation."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
        )

        # Should be valid
        assert authority_filter.validate_authority_level(rule) is True

        # Should be invalid when absolute not allowed
        assert (
            authority_filter.validate_authority_level(rule, allow_absolute=False)
            is False
        )

    def test_get_effective_rules(self, authority_filter, sample_rules):
        """Test getting effective rules with full filtering."""
        effective = authority_filter.get_effective_rules(
            sample_rules, project_id="proj123", scope=["python"]
        )

        # All non-conflicting rules should be included
        assert len(effective) > 0
        # Absolute rules should be first
        assert effective[0].authority == AuthorityLevel.ABSOLUTE

    def test_determine_hierarchy_global(self, authority_filter):
        """Test hierarchy determination for global rules."""
        rule = MemoryRule(
            id="global",
            category=MemoryCategory.BEHAVIOR,
            name="Global",
            rule="Global rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        hierarchy = authority_filter._determine_hierarchy(rule, "proj123", ["python"])
        assert hierarchy == RuleHierarchy.GLOBAL

    def test_determine_hierarchy_project(self, authority_filter):
        """Test hierarchy determination for project rules."""
        rule = MemoryRule(
            id="project",
            category=MemoryCategory.BEHAVIOR,
            name="Project",
            rule="Project rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],  # No scope - project-wide
            metadata={"project_id": "proj123"},
        )

        hierarchy = authority_filter._determine_hierarchy(rule, "proj123", ["python"])
        assert hierarchy == RuleHierarchy.PROJECT

    def test_determine_hierarchy_local(self, authority_filter):
        """Test hierarchy determination for local rules."""
        rule = MemoryRule(
            id="local",
            category=MemoryCategory.BEHAVIOR,
            name="Local",
            rule="Local rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["testing", "python"],
        )

        hierarchy = authority_filter._determine_hierarchy(
            rule, "proj123", ["testing", "python"]
        )
        assert hierarchy == RuleHierarchy.LOCAL

    def test_calculate_precedence(self, authority_filter):
        """Test precedence calculation."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            metadata={"priority": 95},
        )

        precedence = authority_filter._calculate_precedence(rule, "proj123")

        assert isinstance(precedence, AuthorityPrecedence)
        assert precedence.authority == AuthorityLevel.ABSOLUTE
        assert precedence.hierarchy == RuleHierarchy.GLOBAL
        assert precedence.priority == 95

    def test_precedence_sort_key(self, authority_filter):
        """Test precedence sort key generation."""
        now = datetime.now(timezone.utc)

        # Absolute rule
        abs_rule = MemoryRule(
            id="abs",
            category=MemoryCategory.BEHAVIOR,
            name="Abs",
            rule="Absolute",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            created_at=now,
            updated_at=now,
            metadata={"priority": 90},
        )

        # Default rule
        def_rule = MemoryRule(
            id="def",
            category=MemoryCategory.BEHAVIOR,
            name="Def",
            rule="Default",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
            created_at=now,
            updated_at=now,
            metadata={"priority": 90},
        )

        abs_prec = authority_filter._calculate_precedence(abs_rule, None)
        def_prec = authority_filter._calculate_precedence(def_rule, None)

        abs_key = authority_filter._precedence_sort_key(abs_prec, abs_rule)
        def_key = authority_filter._precedence_sort_key(def_prec, def_rule)

        # Absolute should have higher precedence (first element)
        assert abs_key[0] > def_key[0]

    def test_get_context_key(self, authority_filter):
        """Test context key generation for conflict detection."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing"],
            metadata={"project_id": "proj123"},
        )

        context_key = authority_filter._get_context_key(rule, "proj123")

        # Should include category, sorted scope, and project
        assert "behavior" in context_key
        assert "python:testing" in context_key or "testing:python" in context_key

    def test_remove_conflicts_no_conflicts(self, authority_filter):
        """Test conflict removal when no conflicts exist."""
        now = datetime.now(timezone.utc)

        rules = [
            MemoryRule(
                id="rule1",
                category=MemoryCategory.BEHAVIOR,
                name="Rule 1",
                rule="Rule 1",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                created_at=now,
                updated_at=now,
            ),
            MemoryRule(
                id="rule2",
                category=MemoryCategory.PREFERENCE,
                name="Rule 2",
                rule="Rule 2",
                authority=AuthorityLevel.DEFAULT,
                scope=["testing"],
                created_at=now,
                updated_at=now,
            ),
        ]

        resolved = authority_filter._remove_conflicts(rules, "proj123")

        # No conflicts, all rules should remain
        assert len(resolved) == 2

    def test_remove_conflicts_with_conflicts(self, authority_filter):
        """Test conflict removal when conflicts exist."""
        now = datetime.now(timezone.utc)

        # Create rules with same context (will conflict)
        rules = [
            MemoryRule(
                id="high_prec",
                category=MemoryCategory.BEHAVIOR,
                name="High Precedence",
                rule="High precedence rule",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 90},
            ),
            MemoryRule(
                id="low_prec",
                category=MemoryCategory.BEHAVIOR,
                name="Low Precedence",
                rule="Low precedence rule",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                created_at=now - timedelta(days=1),
                updated_at=now,
                metadata={"priority": 80},
            ),
        ]

        # Sort by precedence first (absolute before default)
        sorted_rules = sorted(
            rules,
            key=lambda r: authority_filter._precedence_sort_key(
                authority_filter._calculate_precedence(r, "proj123"), r
            ),
            reverse=True,
        )

        resolved = authority_filter._remove_conflicts(sorted_rules, "proj123")

        # Should only keep high precedence rule
        assert len(resolved) == 1
        assert resolved[0].id == "high_prec"

    def test_resolve_authority_conflicts(self, authority_filter):
        """Test authority conflict resolution."""
        now = datetime.now(timezone.utc)

        # Absolute rule
        absolute_rules = [
            MemoryRule(
                id="abs",
                category=MemoryCategory.BEHAVIOR,
                name="Absolute",
                rule="Absolute rule",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                created_at=now,
                updated_at=now,
            ),
        ]

        # Default rule in same context
        default_rules = [
            MemoryRule(
                id="def",
                category=MemoryCategory.BEHAVIOR,
                name="Default",
                rule="Default rule",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                created_at=now,
                updated_at=now,
            ),
        ]

        conflicts = authority_filter._resolve_authority_conflicts(
            absolute_rules, default_rules, "proj123", ["python"]
        )

        # Should detect conflict
        assert conflicts == 1

    def test_filter_disabled_precedence_resolution(self):
        """Test filter with precedence resolution disabled."""
        filter_no_prec = AuthorityFilter(enable_precedence_resolution=False)

        now = datetime.now(timezone.utc)
        rules = [
            MemoryRule(
                id="rule1",
                category=MemoryCategory.BEHAVIOR,
                name="Rule 1",
                rule="Rule 1",
                authority=AuthorityLevel.ABSOLUTE,
                scope=[],
                created_at=now,
                updated_at=now,
            ),
        ]

        result = filter_no_prec.filter_by_authority(rules)

        # Should still separate by authority but not resolve conflicts
        assert result.conflicts_resolved == 0

    def test_hierarchy_precedence_local_over_project(self, authority_filter):
        """Test that local scope takes precedence over project scope."""
        now = datetime.now(timezone.utc)

        # Project rule (no scope, just project_id)
        project_rule = MemoryRule(
            id="project",
            category=MemoryCategory.BEHAVIOR,
            name="Project Rule",
            rule="Project approach",
            authority=AuthorityLevel.DEFAULT,
            scope=[],  # No scope - project-wide
            created_at=now,
            updated_at=now,
            metadata={"priority": 80, "project_id": "proj123"},
        )

        # Local rule (has specific scope)
        local_rule = MemoryRule(
            id="local",
            category=MemoryCategory.BEHAVIOR,
            name="Local Rule",
            rule="Local approach",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing"],
            created_at=now,
            updated_at=now,
            metadata={"priority": 80},
        )

        project_prec = authority_filter._calculate_precedence(project_rule, "proj123")
        local_prec = authority_filter._calculate_precedence(local_rule, "proj123")

        project_key = authority_filter._precedence_sort_key(project_prec, project_rule)
        local_key = authority_filter._precedence_sort_key(local_prec, local_rule)

        # Local should have higher hierarchy level (element [1] in key)
        assert local_key[1] > project_key[1]  # Local (3) > Project (2)
