"""
Unit tests for code comment injection strategy.

Tests the CommentInject or's ability to format rules as language-specific
comments for GitHub Copilot context injection.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.python.common.core.context_injection.comment_injector import (
    CommentInjector,
    PlacementStrategy,
    ConflictResolution,
    ConflictType,
    RuleConflict,
    InjectedComment,
)
from src.python.common.core.context_injection.copilot_detector import CopilotDetector
from src.python.common.memory import MemoryRule, AuthorityLevel, MemoryCategory


@pytest.fixture
def sample_rules():
    """Create sample rules for testing."""
    return [
        MemoryRule(
            name="atomic_commits",
            rule="Always make atomic commits with clear, descriptive messages",
            authority=AuthorityLevel.ABSOLUTE,
            category=MemoryCategory.BEHAVIOR,
            scope=["git", "version-control"],
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            metadata={"priority": 1},
        ),
        MemoryRule(
            name="type_hints",
            rule="Use type hints for all function parameters and return values",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.BEHAVIOR,
            scope=["python", "typing"],
            created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            metadata={"priority": 2},
        ),
        MemoryRule(
            name="error_handling",
            rule="Always handle errors gracefully with try-except blocks",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.BEHAVIOR,
            scope=["python", "error-handling"],
            created_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            metadata={"priority": 2},
        ),
    ]


@pytest.fixture
def conflicting_rules():
    """Create rules with conflicts for testing."""
    return [
        MemoryRule(
            name="use_tabs",
            rule="Always use tabs for indentation",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.FORMAT,
            scope=["python", "formatting"],
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            metadata={"priority": 2},
        ),
        MemoryRule(
            name="use_spaces",
            rule="Never use tabs, always use spaces for indentation",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.FORMAT,
            scope=["python", "formatting"],
            created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            metadata={"priority": 2},
        ),
    ]


@pytest.fixture
def injector():
    """Create CommentInjector instance for testing."""
    return CommentInjector()


class TestCommentInjectorBasics:
    """Test basic CommentInjector functionality."""

    def test_initialization(self):
        """Test CommentInjector initialization."""
        injector = CommentInjector()
        assert injector.detector is not None
        assert isinstance(injector.detector, CopilotDetector)

    def test_initialization_with_detector(self):
        """Test initialization with custom detector."""
        detector = CopilotDetector()
        injector = CommentInjector(detector=detector)
        assert injector.detector is detector


class TestFileHeaderFormatting:
    """Test file header comment formatting."""

    def test_python_file_header(self, injector, sample_rules):
        """Test Python file header formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert isinstance(result, InjectedComment)
        assert result.language == "python"
        assert result.placement == PlacementStrategy.FILE_HEADER
        assert result.rules_included == 3

        # Check content structure
        assert "COPILOT CODING RULES" in result.content
        assert "atomic_commits" in result.content
        assert "type_hints" in result.content
        assert "[P1]" in result.content  # Priority 1 rule
        assert "[P2]" in result.content  # Priority 2 rules
        assert "CRITICAL" in result.content  # Absolute authority marker

        # Check comment prefix
        assert result.content.startswith("#")

    def test_javascript_file_header(self, injector, sample_rules):
        """Test JavaScript file header formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="javascript",
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert result.language == "javascript"
        assert "//" in result.content  # JavaScript comment prefix
        assert "COPILOT CODING RULES" in result.content

    def test_rust_file_header(self, injector, sample_rules):
        """Test Rust file header formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="rust",
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert result.language == "rust"
        assert "//" in result.content  # Rust comment prefix

    def test_file_header_with_priority_filter(self, injector, sample_rules):
        """Test file header with priority filter."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.FILE_HEADER,
            priority_filter="P1",
        )

        # Should only include P1 rule
        assert result.rules_included == 1
        assert "atomic_commits" in result.content
        assert "type_hints" not in result.content


class TestDocstringFormatting:
    """Test docstring comment formatting."""

    def test_python_docstring(self, injector, sample_rules):
        """Test Python docstring formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.DOCSTRING,
        )

        assert result.placement == PlacementStrategy.DOCSTRING
        assert "Coding Rules:" in result.content
        assert "[P1]" in result.content
        assert "atomic_commits" in result.content

    def test_javascript_jsdoc(self, injector, sample_rules):
        """Test JSDoc formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="javascript",
            placement=PlacementStrategy.DOCSTRING,
        )

        assert "@copilot-rule" in result.content
        assert "[P1]" in result.content

    def test_typescript_jsdoc(self, injector, sample_rules):
        """Test TypeScript JSDoc formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="typescript",
            placement=PlacementStrategy.DOCSTRING,
        )

        assert "@copilot-rule" in result.content

    def test_rust_doc_comments(self, injector, sample_rules):
        """Test Rust doc comment formatting."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="rust",
            placement=PlacementStrategy.DOCSTRING,
        )

        assert "///" in result.content
        assert "CODING RULE" in result.content


class TestInlineFormatting:
    """Test inline comment directive formatting."""

    def test_python_inline(self, injector, sample_rules):
        """Test Python inline directives."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.INLINE,
        )

        assert result.placement == PlacementStrategy.INLINE
        assert "# RULE" in result.content
        assert "[P1]" in result.content

    def test_javascript_inline(self, injector, sample_rules):
        """Test JavaScript inline directives."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="javascript",
            placement=PlacementStrategy.INLINE,
        )

        assert "// RULE" in result.content

    def test_inline_multiple_rules(self, injector, sample_rules):
        """Test inline formatting with multiple rules."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.INLINE,
        )

        # Should have one directive per rule
        assert result.content.count("# RULE") == 3


class TestConflictDetection:
    """Test conflict detection between rules."""

    def test_detect_contradictory_rules(self, injector, conflicting_rules):
        """Test detection of contradictory rules."""
        conflicts = injector.detect_conflicts(conflicting_rules)

        assert len(conflicts) > 0
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.CONTRADICTORY
        assert "use_tabs" in conflict.description or "use_spaces" in conflict.description

    def test_detect_redundant_rules(self, injector):
        """Test detection of redundant (duplicate) rules."""
        redundant_rules = [
            MemoryRule(
                name="rule1",
                rule="Always use type hints",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
            ),
            MemoryRule(
                name="rule2",
                rule="always use type hints",  # Same content, different case
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
            ),
        ]

        conflicts = injector.detect_conflicts(redundant_rules)

        assert len(conflicts) > 0
        assert any(c.conflict_type == ConflictType.REDUNDANT for c in conflicts)

    def test_detect_overlapping_scope(self, injector):
        """Test detection of overlapping scope with different priorities."""
        overlapping_rules = [
            MemoryRule(
                name="general_rule",
                rule="Follow PEP 8 style guide",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.FORMAT,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 2},
            ),
            MemoryRule(
                name="specific_rule",
                rule="Use 2-space indentation",
                authority=AuthorityLevel.ABSOLUTE,
                category=MemoryCategory.FORMAT,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 1},
            ),
        ]

        conflicts = injector.detect_conflicts(overlapping_rules)

        # Should detect overlapping scope with different priorities
        assert any(c.conflict_type == ConflictType.OVERLAPPING for c in conflicts)

    def test_no_conflicts_with_different_scopes(self, injector, sample_rules):
        """Test that rules with different scopes don't conflict."""
        conflicts = injector.detect_conflicts(sample_rules)

        # Sample rules have different scopes, shouldn't have major conflicts
        contradictory_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.CONTRADICTORY
        ]
        assert len(contradictory_conflicts) == 0


class TestConflictResolution:
    """Test conflict resolution strategies."""

    def test_resolve_highest_priority(self, injector):
        """Test resolution using highest priority strategy."""
        conflicting_rules = [
            MemoryRule(
                name="low_priority",
                rule="Use tabs",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.FORMAT,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 3},
            ),
            MemoryRule(
                name="high_priority",
                rule="Never use tabs",
                authority=AuthorityLevel.ABSOLUTE,
                category=MemoryCategory.FORMAT,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 1},
            ),
        ]

        resolved = injector.resolve_conflicts(
            conflicting_rules, strategy=ConflictResolution.HIGHEST_PRIORITY
        )

        # Should keep only the high priority rule
        assert len(resolved) == 1
        assert resolved[0].name == "high_priority"

    def test_resolve_most_specific(self, injector):
        """Test resolution using most specific scope strategy."""
        rules = [
            MemoryRule(
                name="general",
                rule="Follow best practices",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 2},
            ),
            MemoryRule(
                name="specific",
                rule="Follow best practices for async code",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python", "async", "concurrency"],
                created_at=datetime.now(timezone.utc),
                metadata={"priority": 2},
            ),
        ]

        resolved = injector.resolve_conflicts(
            rules, strategy=ConflictResolution.MOST_SPECIFIC
        )

        # More specific rule should be kept
        assert any(r.name == "specific" for r in resolved)

    def test_resolve_last_wins(self, injector):
        """Test resolution using last-wins strategy."""
        now = datetime.now(timezone.utc)
        rules = [
            MemoryRule(
                name="old_rule",
                rule="Old instruction",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python"],
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                metadata={"priority": 2},
            ),
            MemoryRule(
                name="new_rule",
                rule="Never follow old instruction",
                authority=AuthorityLevel.DEFAULT,
                category=MemoryCategory.BEHAVIOR,
                scope=["python"],
                created_at=now,
                metadata={"priority": 2},
            ),
        ]

        resolved = injector.resolve_conflicts(
            rules, strategy=ConflictResolution.LAST_WINS
        )

        # Newer rule should be kept
        assert any(r.name == "new_rule" for r in resolved)

    def test_resolve_merge_all(self, injector, conflicting_rules):
        """Test merge all strategy keeps all rules."""
        resolved = injector.resolve_conflicts(
            conflicting_rules, strategy=ConflictResolution.MERGE_ALL
        )

        # All rules should be kept with merge strategy
        assert len(resolved) == len(conflicting_rules)


class TestPriorityHandling:
    """Test priority level handling and sorting."""

    def test_get_rule_priority_from_metadata(self, injector):
        """Test getting priority from rule metadata."""
        rule = MemoryRule(
            name="test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.BEHAVIOR,
            scope=["python"],
            created_at=datetime.now(timezone.utc),
            metadata={"priority": 2},
        )

        priority = injector._get_rule_priority(rule)
        assert priority == 2

    def test_get_rule_priority_from_authority(self, injector):
        """Test priority inference from authority level."""
        absolute_rule = MemoryRule(
            name="test",
            rule="Test rule",
            authority=AuthorityLevel.ABSOLUTE,
            category=MemoryCategory.BEHAVIOR,
            scope=["python"],
            created_at=datetime.now(timezone.utc),
        )

        default_rule = MemoryRule(
            name="test2",
            rule="Test rule 2",
            authority=AuthorityLevel.DEFAULT,
            category=MemoryCategory.BEHAVIOR,
            scope=["python"],
            created_at=datetime.now(timezone.utc),
        )

        assert injector._get_rule_priority(absolute_rule) == 1
        assert injector._get_rule_priority(default_rule) == 2

    def test_sort_by_priority(self, injector, sample_rules):
        """Test sorting rules by priority."""
        sorted_rules = injector._sort_by_priority(sample_rules)

        # P1 rule should come first
        assert sorted_rules[0].name == "atomic_commits"
        # P2 rules should follow
        assert sorted_rules[1].metadata["priority"] == 2
        assert sorted_rules[2].metadata["priority"] == 2

    def test_filter_by_priority(self, injector, sample_rules):
        """Test filtering rules by priority."""
        p1_rules = injector._filter_by_priority(sample_rules, "P1")
        assert len(p1_rules) == 1
        assert p1_rules[0].name == "atomic_commits"

        p2_rules = injector._filter_by_priority(sample_rules, "P2")
        assert len(p2_rules) == 2

    def test_filter_by_priority_invalid_filter(self, injector, sample_rules):
        """Test filtering with invalid priority filter."""
        # Should return all rules on invalid filter
        result = injector._filter_by_priority(sample_rules, "invalid")
        assert len(result) == len(sample_rules)


class TestLanguageSupport:
    """Test support for multiple programming languages."""

    @pytest.mark.parametrize(
        "language,expected_prefix",
        [
            ("python", "#"),
            ("javascript", "//"),
            ("typescript", "//"),
            ("rust", "//"),
            ("go", "//"),
            ("java", "//"),
            ("csharp", "//"),
            ("ruby", "#"),
            ("shell", "#"),
            ("sql", "--"),
        ],
    )
    def test_language_comment_prefixes(
        self, injector, sample_rules, language, expected_prefix
    ):
        """Test correct comment prefix for each language."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language=language,
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert expected_prefix in result.content


class TestMetadata:
    """Test metadata in injected comments."""

    def test_metadata_includes_timestamp(self, injector, sample_rules):
        """Test that metadata includes generation timestamp."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert "generated_at" in result.metadata
        # Should be valid ISO format timestamp
        datetime.fromisoformat(result.metadata["generated_at"])

    def test_metadata_includes_conflict_detection_flag(self, injector, sample_rules):
        """Test that metadata includes conflict detection flag."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.FILE_HEADER,
            detect_conflicts=True,
        )

        assert result.metadata["conflict_detection"] is True

    def test_content_includes_timestamp(self, injector, sample_rules):
        """Test that generated content includes timestamp."""
        result = injector.format_as_comment(
            rules=sample_rules,
            language="python",
            placement=PlacementStrategy.FILE_HEADER,
        )

        assert "Last Updated:" in result.content
