"""
Comprehensive tests for conflict detection algorithms.

Tests semantic conflict detection, contradictory rules, overlapping scopes,
resolution strategies, and conflict scoring/ranking mechanisms.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
    MemoryRuleConflict,
)
from common.memory.conflict_detector import ConflictDetector

from .rule_test_utils import (
    MemoryRuleGenerator,
    ConflictSimulator,
    MemoryRuleValidator,
)


class TestSemanticConflictDetection:
    """Test AI-powered semantic conflict detection."""

    @pytest.fixture
    def detector_with_mock_ai(self):
        """Create detector with mocked AI client."""
        with patch("common.memory.conflict_detector.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            detector = ConflictDetector(
                anthropic_api_key="test-key",
                enable_ai_analysis=True
            )
            return detector, mock_client

    @pytest.mark.asyncio
    async def test_implicit_contradiction_detection(self, detector_with_mock_ai):
        """Test detection of implicit contradictions like 'prefer X' vs 'avoid X'."""
        detector, mock_client = detector_with_mock_ai

        rule1 = MemoryRule(
            rule="Prefer functional programming style with immutable data structures",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "coding_style"]
        )

        rule2 = MemoryRule(
            rule="Avoid functional programming, use object-oriented design patterns",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "coding_style"]
        )

        # Mock AI response detecting implicit conflict
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "conflicts": [
                {
                    "rule1": "NEW RULE",
                    "rule2": "EXISTING RULE 1",
                    "conflict_type": "semantic",
                    "confidence": 0.85,
                    "description": "Contradictory preferences on programming paradigms",
                    "severity": "medium",
                    "resolution_suggestion": "Choose one paradigm or define contexts for each"
                }
            ]
        })

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect the implicit contradiction
        assert len(conflicts) > 0
        semantic_conflicts = [c for c in conflicts if c.conflict_type == "semantic"]
        assert len(semantic_conflicts) > 0
        assert semantic_conflicts[0].confidence >= 0.8
        assert "paradigm" in semantic_conflicts[0].description.lower() or "programming" in semantic_conflicts[0].description.lower()

    @pytest.mark.asyncio
    async def test_conflicting_approaches_detection(self, detector_with_mock_ai):
        """Test detection of conflicting methodologies."""
        detector, mock_client = detector_with_mock_ai

        rule1 = MemoryRule(
            rule="Use test-driven development: write tests before implementation",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing", "development"]
        )

        rule2 = MemoryRule(
            rule="Write code first, then add tests for coverage",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing", "development"]
        )

        # Mock AI response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "conflicts": [
                {
                    "rule1": "NEW RULE",
                    "rule2": "EXISTING RULE 1",
                    "conflict_type": "semantic",
                    "confidence": 0.9,
                    "description": "Conflicting testing methodologies: TDD vs code-first approach",
                    "severity": "high",
                    "resolution_suggestion": "Choose one approach as default"
                }
            ]
        })

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect methodology conflict
        assert len(conflicts) > 0
        assert any(c.conflict_type == "semantic" for c in conflicts)
        assert any(c.severity in ["high", "medium"] for c in conflicts)

    @pytest.mark.asyncio
    async def test_context_aware_conflict_detection(self, detector_with_mock_ai):
        """Test conflict detection in same scope but different contexts."""
        detector, mock_client = detector_with_mock_ai

        rule1 = MemoryRule(
            rule="Use async/await for all database operations",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["database", "async"]
        )

        rule2 = MemoryRule(
            rule="Use synchronous database calls for admin scripts",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["database", "admin"]
        )

        # Mock AI response - should detect context makes these compatible
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "conflicts": []  # No conflict due to different contexts
        })

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should not find critical conflicts due to context differentiation
        critical_conflicts = [c for c in conflicts if c.severity == "critical"]
        assert len(critical_conflicts) == 0

    @pytest.mark.asyncio
    async def test_batch_semantic_analysis(self, detector_with_mock_ai):
        """Test batch processing of semantic analysis."""
        detector, mock_client = detector_with_mock_ai

        new_rule = MemoryRule(
            rule="Use TypeScript for all frontend code",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend"]
        )

        # Create batch of existing rules
        existing_rules = [
            MemoryRule(
                rule="JavaScript is fine for simple components",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["frontend"]
            ),
            MemoryRule(
                rule="Use vanilla JS to avoid build complexity",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["frontend"]
            ),
            MemoryRule(
                rule="Python for backend, any JS for frontend",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["backend", "frontend"]
            ),
        ]

        # Mock AI response with multiple conflicts
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "conflicts": [
                {
                    "rule1": "NEW RULE",
                    "rule2": "EXISTING RULE 1",
                    "conflict_type": "semantic",
                    "confidence": 0.75,
                    "description": "TypeScript requirement conflicts with JS preference",
                    "severity": "medium"
                },
                {
                    "rule1": "NEW RULE",
                    "rule2": "EXISTING RULE 2",
                    "conflict_type": "semantic",
                    "confidence": 0.8,
                    "description": "TypeScript requirement conflicts with vanilla JS preference",
                    "severity": "medium"
                }
            ]
        })

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        conflicts = await detector.detect_conflicts(new_rule, existing_rules)

        # Should process batch and find multiple conflicts
        assert len(conflicts) >= 2
        assert mock_client.messages.create.called


class TestContradictoryRulesDetection:
    """Test detection of direct contradictions using keyword analysis."""

    @pytest.fixture
    def detector(self):
        """Create detector without AI for fast rule-based testing."""
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_always_never_contradiction(self, detector):
        """Test detection of 'always X' vs 'never X' patterns."""
        rule1 = MemoryRule(
            rule="Always use type hints in Python functions",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "typing"]
        )

        rule2 = MemoryRule(
            rule="Never use type hints, they add unnecessary complexity",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "typing"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect direct contradiction
        assert len(conflicts) > 0
        # Should have high severity due to both being absolute
        assert any(c.severity in ["high", "critical"] for c in conflicts)

    @pytest.mark.asyncio
    async def test_use_dont_use_contradiction(self, detector):
        """Test detection of 'use X' vs 'don't use X' patterns."""
        rule1 = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing"]
        )

        rule2 = MemoryRule(
            rule="Don't use pytest, use unittest instead",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect contradiction
        assert len(conflicts) > 0
        direct_conflicts = [c for c in conflicts if c.conflict_type == "direct"]
        assert len(direct_conflicts) > 0
        assert "pytest" in direct_conflicts[0].description.lower()

    @pytest.mark.asyncio
    async def test_prefer_avoid_contradiction(self, detector):
        """Test detection of 'prefer X' vs 'avoid X' patterns."""
        rule1 = MemoryRule(
            rule="Prefer async functions for I/O operations",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "async"]
        )

        rule2 = MemoryRule(
            rule="Avoid async functions when synchronous code works",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "async"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect preference conflict
        assert len(conflicts) > 0
        assert any("async" in c.description.lower() for c in conflicts)

    @pytest.mark.asyncio
    async def test_enable_disable_contradiction(self, detector):
        """Test detection of 'enable X' vs 'disable X' patterns."""
        rule1 = MemoryRule(
            rule="Enable strict type checking in TypeScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["typescript", "config"]
        )

        rule2 = MemoryRule(
            rule="Disable strict type checking for faster development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["typescript", "config"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect enable/disable conflict
        assert len(conflicts) > 0
        assert any(c.conflict_type in ["direct", "authority"] for c in conflicts)

    @pytest.mark.asyncio
    async def test_multiple_contradiction_patterns(self, detector):
        """Test detection of multiple contradiction patterns in complex rules."""
        rule1 = MemoryRule(
            rule="Always use linters and formatters. Enable auto-formatting on save. Prefer black for Python.",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "formatting"]
        )

        rule2 = MemoryRule(
            rule="Don't use auto-formatters. Never enable format-on-save. Avoid black, use autopep8 instead.",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "formatting"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect at least one contradiction
        assert len(conflicts) > 0
        # Description should mention the specific conflict
        assert any(len(c.description) > 10 for c in conflicts)


class TestOverlappingScopesIdentification:
    """Test detection and analysis of overlapping scopes."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_exact_scope_overlap(self, detector):
        """Test detection of exact scope matches."""
        rule1 = MemoryRule(
            rule="Use black for formatting",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "formatting"]
        )

        rule2 = MemoryRule(
            rule="Use autopep8 for formatting",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "formatting"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect scope overlap
        assert len(conflicts) > 0
        # Should be authority conflict due to both being absolute with same scope
        assert any(c.conflict_type == "authority" for c in conflicts)
        assert any("scope" in c.description.lower() for c in conflicts)

    @pytest.mark.asyncio
    async def test_partial_scope_overlap(self, detector):
        """Test detection of partial scope overlaps."""
        rule1 = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "testing", "unit_tests"]
        )

        rule2 = MemoryRule(
            rule="Use unittest for all tests",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "testing"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect partial overlap in 'python' and 'testing'
        assert len(conflicts) > 0
        authority_conflicts = [c for c in conflicts if c.conflict_type == "authority"]
        assert len(authority_conflicts) > 0

    @pytest.mark.asyncio
    async def test_global_scope_conflicts(self, detector):
        """Test conflicts with global scope (empty list)."""
        rule1 = MemoryRule(
            rule="Always make atomic commits",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[]  # Global scope
        )

        rule2 = MemoryRule(
            rule="Make larger commits for related changes",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[]  # Global scope
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect global scope conflict
        assert len(conflicts) > 0
        assert any("global" in c.description.lower() for c in conflicts)

    @pytest.mark.asyncio
    async def test_no_scope_overlap(self, detector):
        """Test that non-overlapping scopes don't generate conflicts."""
        rule1 = MemoryRule(
            rule="Use React for frontend",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend", "react"]
        )

        rule2 = MemoryRule(
            rule="Use FastAPI for backend",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["backend", "api"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should not detect conflicts (different domains)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_nested_scope_hierarchies(self, detector):
        """Test handling of nested/hierarchical scopes."""
        rule1 = MemoryRule(
            rule="Use strict linting",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["project-x", "backend", "api"]
        )

        rule2 = MemoryRule(
            rule="Disable strict linting for prototypes",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["project-x", "backend"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should handle nested scopes gracefully (may or may not conflict)
        assert isinstance(conflicts, list)


class TestResolutionStrategies:
    """Test conflict resolution strategy suggestions."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_authority_based_resolution(self, detector):
        """Test that absolute authority takes precedence over default."""
        rule1 = MemoryRule(
            rule="Must use Python 3.11",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        rule2 = MemoryRule(
            rule="Prefer Python 3.12",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should provide resolution suggestion
        if conflicts:
            assert any(c.resolution_suggestion is not None for c in conflicts)
            # Suggestion should mention authority
            suggestions = [c.resolution_suggestion for c in conflicts if c.resolution_suggestion]
            assert any("authority" in s.lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_resolution_suggestions_present(self, detector):
        """Test that all conflicts have resolution suggestions."""
        rule1 = MemoryRule(
            rule="Use tabs for indentation",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        rule2 = MemoryRule(
            rule="Use spaces for indentation",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # All conflicts should have suggestions
        assert len(conflicts) > 0
        for conflict in conflicts:
            assert conflict.resolution_suggestion is not None
            assert len(conflict.resolution_suggestion) > 0

    @pytest.mark.asyncio
    async def test_scope_refinement_suggestion(self, detector):
        """Test suggestion to refine scopes for overlapping rules."""
        rule1 = MemoryRule(
            rule="Use async for all operations",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        rule2 = MemoryRule(
            rule="Use sync for database migrations",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should suggest refining scopes
        assert len(conflicts) > 0
        suggestions = [c.resolution_suggestion for c in conflicts if c.resolution_suggestion]
        assert any("scope" in s.lower() or "refin" in s.lower() for s in suggestions)


class TestConflictScoringAndRanking:
    """Test conflict severity classification, confidence scoring, and ranking."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_severity_classification(self, detector):
        """Test that conflicts are classified by severity."""
        # High severity: Two absolute rules in same scope
        rule1 = MemoryRule(
            rule="Never use eval()",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["security"]
        )

        rule2 = MemoryRule(
            rule="Use eval() for dynamic code execution",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["security"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        assert len(conflicts) > 0
        # Should have high or critical severity
        assert any(c.severity in ["high", "critical"] for c in conflicts)

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, detector):
        """Test that conflicts have confidence scores."""
        rule1 = MemoryRule(
            rule="Use TypeScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend"]
        )

        rule2 = MemoryRule(
            rule="Don't use TypeScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            # Confidence should be between 0.0 and 1.0
            assert 0.0 <= conflict.confidence <= 1.0
            # Direct contradictions should have high confidence
            if conflict.conflict_type == "direct":
                assert conflict.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_conflict_prioritization_by_severity(self, detector):
        """Test that conflicts are sorted by severity."""
        # Create rules with different severity levels
        absolute_rule = MemoryRule(
            rule="Must use secure connections",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["security"]
        )

        conflicting_rules = [
            MemoryRule(
                rule="Allow insecure connections for testing",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,  # High severity
                scope=["security", "testing"]
            ),
            MemoryRule(
                rule="Prefer secure connections but allow fallback",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,  # Lower severity
                scope=["security"]
            ),
        ]

        conflicts = await detector.detect_conflicts(absolute_rule, conflicting_rules)

        # Conflicts should be sorted by severity
        if len(conflicts) > 1:
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            for i in range(len(conflicts) - 1):
                assert severity_order[conflicts[i].severity] >= severity_order[conflicts[i + 1].severity]

    @pytest.mark.asyncio
    async def test_conflict_deduplication(self, detector):
        """Test that identical DEFAULT authority rules don't conflict."""
        rule1 = MemoryRule(
            rule="Use React for frontend",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["frontend"]
        )

        # Same rule text, DEFAULT authority - should not conflict
        rule2 = MemoryRule(
            rule="Use React for frontend",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["frontend"]
        )

        existing_rules = [rule2]

        conflicts = await detector.detect_conflicts(rule1, existing_rules)

        # DEFAULT authority identical rules should not conflict
        assert len(conflicts) == 0
    def test_conflict_summary_generation(self, detector):
        """Test generation of conflict summary statistics."""
        # Create sample conflicts
        conflicts = [
            MemoryRuleConflict(
                rule1=MemoryRule(
                    rule="Test 1",
                    category=MemoryCategory.BEHAVIOR,
                    authority=AuthorityLevel.ABSOLUTE
                ),
                rule2=MemoryRule(
                    rule="Test 2",
                    category=MemoryCategory.BEHAVIOR,
                    authority=AuthorityLevel.ABSOLUTE
                ),
                conflict_type="direct",
                confidence=0.9,
                description="Direct conflict",
                severity="critical"
            ),
            MemoryRuleConflict(
                rule1=MemoryRule(
                    rule="Test 3",
                    category=MemoryCategory.PREFERENCE,
                    authority=AuthorityLevel.DEFAULT
                ),
                rule2=MemoryRule(
                    rule="Test 4",
                    category=MemoryCategory.PREFERENCE,
                    authority=AuthorityLevel.DEFAULT
                ),
                conflict_type="semantic",
                confidence=0.7,
                description="Semantic conflict",
                severity="medium"
            ),
        ]

        summary = detector.get_conflict_summary(conflicts)

        # Verify summary structure
        assert summary["total"] == 2
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_type"]["direct"] == 1
        assert summary["by_type"]["semantic"] == 1
        assert summary["highest_confidence"] == 0.9
        assert len(summary["recommendations"]) > 0


class TestConflictDetectionEdgeCases:
    """Test edge cases and error handling in conflict detection."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_empty_existing_rules(self, detector):
        """Test detection with empty existing rules list."""
        new_rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        conflicts = await detector.detect_conflicts(new_rule, [])

        # Should return empty list, not error
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_identical_rules(self, detector):
        """Test detection between identical rules."""
        rule1 = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        rule2 = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Identical rules should not conflict
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_rules_with_empty_scopes(self, detector):
        """Test conflict detection with empty scopes (global rules)."""
        rule1 = MemoryRule(
            rule="Global rule 1",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[]
        )

        rule2 = MemoryRule(
            rule="Specific rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Global rule should potentially conflict with specific rule
        # (depends on implementation - testing it handles gracefully)
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_large_rule_set_performance(self, detector):
        """Test performance with large number of rules."""
        import time

        generator = MemoryRuleGenerator(seed=42)
        new_rule = generator.generate_rule()

        # Create 50 existing rules
        existing_rules = generator.generate_rules(count=50)

        start_time = time.time()
        conflicts = await detector.detect_conflicts(new_rule, existing_rules)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 0.5 seconds without AI)
        assert elapsed_time < 0.5
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_analyze_all_conflicts(self, detector):
        """Test analyzing all conflicts among a set of rules."""
        generator = MemoryRuleGenerator(seed=42)
        simulator = ConflictSimulator(generator)

        # Create some rules with known conflicts
        rule1, rule2 = simulator.generate_contradictory_pair(scope=["python"])
        rule3, rule4 = simulator.generate_authority_conflict(scope=["frontend"])

        all_rules = [rule1, rule2, rule3, rule4]

        conflicts = await detector.analyze_all_conflicts(all_rules)

        # Should detect at least one intentional conflict
        assert len(conflicts) >= 1
        # Each conflict should involve different rule pairs
        conflict_pairs = set()
        for conflict in conflicts:
            pair = tuple(sorted([conflict.rule1.id, conflict.rule2.id]))
            conflict_pairs.add(pair)

        # Should have unique conflict pairs (at least one detected)
        assert len(conflict_pairs) >= 1
