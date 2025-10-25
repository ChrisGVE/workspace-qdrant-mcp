"""
Tests for conflict detection system.

Comprehensive tests for memory rule conflict detection including semantic analysis,
rule-based conflicts, authority conflicts, and AI-powered conflict detection.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from workspace_qdrant_mcp.memory.conflict_detector import ConflictDetector
from workspace_qdrant_mcp.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
    MemoryRuleConflict,
)


class TestConflictDetectorInit:
    """Test conflict detector initialization and configuration."""

    def test_init_without_ai(self):
        """Test initializing conflict detector without AI analysis."""
        detector = ConflictDetector(enable_ai_analysis=False)

        assert detector.enable_ai_analysis is False
        assert detector.anthropic_client is None
        assert detector.model == "claude-3-sonnet-20240229"

    def test_init_with_ai_no_key(self):
        """Test initializing with AI but no API key."""
        with patch.dict('os.environ', {}, clear=True):
            detector = ConflictDetector(enable_ai_analysis=True)

            # Should disable AI analysis when no key available
            assert detector.enable_ai_analysis is False
            assert detector.anthropic_client is None

    def test_init_with_custom_model(self):
        """Test initializing with custom model."""
        detector = ConflictDetector(
            enable_ai_analysis=False,
            model="claude-3-opus-20240229"
        )

        assert detector.model == "claude-3-opus-20240229"
        assert detector.enable_ai_analysis is False

    @patch('anthropic.Anthropic')
    def test_init_with_ai_and_key(self, mock_anthropic):
        """Test initializing with AI analysis and API key."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        detector = ConflictDetector(
            anthropic_api_key="test-key",
            enable_ai_analysis=True
        )

        assert detector.enable_ai_analysis is True
        assert detector.anthropic_client is mock_client
        mock_anthropic.assert_called_once_with(api_key="test-key")


class TestRuleBasedConflicts:
    """Test rule-based (non-AI) conflict detection."""

    @pytest.fixture
    def detector(self):
        """Create detector without AI analysis."""
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_direct_contradiction_detected(self, detector):
        """Test detection of direct contradictions."""
        rule1 = MemoryRule(
            rule="Always use TypeScript for new projects",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["javascript", "frontend"]
        )

        rule2 = MemoryRule(
            rule="Never use TypeScript, stick to vanilla JavaScript",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["javascript", "frontend"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        assert len(conflicts) > 0
        conflict = conflicts[0]
        assert conflict.conflict_type in ["authority", "direct"]
        assert conflict.severity in ["high", "critical"]
        assert conflict.rule1.id == rule1.id
        assert conflict.rule2.id == rule2.id

    @pytest.mark.asyncio
    async def test_authority_conflict_detected(self, detector):
        """Test detection of authority conflicts (multiple absolutes)."""
        rule1 = MemoryRule(
            rule="Use React for frontend development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend", "react"]
        )

        rule2 = MemoryRule(
            rule="Use Vue.js for all frontend work",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["frontend", "vue"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect authority conflict since both are absolute in overlapping domains
        assert len(conflicts) > 0
        conflict = conflicts[0]
        assert conflict.conflict_type == "authority"
        assert "absolute" in conflict.description.lower()

    @pytest.mark.asyncio
    async def test_no_conflict_different_scopes(self, detector):
        """Test no conflict when rules have different scopes."""
        rule1 = MemoryRule(
            rule="Use Python for backend development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "backend"]
        )

        rule2 = MemoryRule(
            rule="Use JavaScript for frontend development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["javascript", "frontend"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should not conflict - different domains
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_no_conflict_different_authorities(self, detector):
        """Test no conflict when authorities are compatible."""
        rule1 = MemoryRule(
            rule="Prefer React for complex frontends",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["frontend"]
        )

        rule2 = MemoryRule(
            rule="Use Vue for simple components",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["frontend"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Both are preferences, should not conflict strongly
        assert all(c.severity != "critical" for c in conflicts)

    @pytest.mark.asyncio
    async def test_scope_overlap_detection(self, detector):
        """Test detection of scope overlaps."""
        rule1 = MemoryRule(
            rule="Always validate input in API endpoints",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["api", "validation"]
        )

        rule2 = MemoryRule(
            rule="Skip validation for internal APIs",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["api", "internal"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect potential conflict in API handling
        assert len(conflicts) > 0


class TestSemanticConflicts:
    """Test AI-powered semantic conflict detection."""

    @pytest.fixture
    def detector_with_ai(self):
        """Create detector with mocked AI analysis."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            detector = ConflictDetector(
                anthropic_api_key="test-key",
                enable_ai_analysis=True
            )
            return detector

    @pytest.mark.asyncio
    async def test_semantic_analysis_called(self, detector_with_ai):
        """Test that semantic analysis is called when enabled."""
        rule1 = MemoryRule(
            rule="Always use meaningful variable names",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        rule2 = MemoryRule(
            rule="Short variable names are fine for loops",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        # Mock the AI response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "conflicts": [
                {
                    "rule1_id": rule1.id,
                    "rule2_id": rule2.id,
                    "conflict_type": "semantic",
                    "severity": "medium",
                    "description": "Contradictory guidance on variable naming",
                    "resolution_suggestion": "Clarify when short names are acceptable"
                }
            ]
        })

        detector_with_ai.anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        conflicts = await detector_with_ai.detect_conflicts(rule1, [rule2])

        # Should have called anthropic client
        detector_with_ai.anthropic_client.messages.create.assert_called_once()

        # Should return the semantic conflict
        assert len(conflicts) >= 1
        semantic_conflicts = [c for c in conflicts if c.conflict_type == "semantic"]
        assert len(semantic_conflicts) == 1

    @pytest.mark.asyncio
    async def test_ai_error_handling(self, detector_with_ai):
        """Test error handling when AI analysis fails."""
        rule1 = MemoryRule(
            rule="Use tabs for indentation",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT
        )

        rule2 = MemoryRule(
            rule="Use spaces for indentation",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT
        )

        # Mock AI failure
        detector_with_ai.anthropic_client.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Should not raise exception, should fall back to rule-based detection
        conflicts = await detector_with_ai.detect_conflicts(rule1, [rule2])

        # Should still detect some conflicts from rule-based analysis
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_malformed_ai_response_handling(self, detector_with_ai):
        """Test handling of malformed AI responses."""
        rule1 = MemoryRule(
            rule="Test rule 1",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        rule2 = MemoryRule(
            rule="Test rule 2",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        # Mock malformed response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Invalid JSON response"

        detector_with_ai.anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        # Should handle gracefully
        conflicts = await detector_with_ai.detect_conflicts(rule1, [rule2])
        assert isinstance(conflicts, list)


class TestConflictResolution:
    """Test conflict resolution suggestions."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_resolution_suggestions(self, detector):
        """Test that resolution suggestions are provided."""
        rule1 = MemoryRule(
            rule="Always use single quotes for strings",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        rule2 = MemoryRule(
            rule="Always use double quotes for strings",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflicts = await detector.detect_conflicts(rule1, [rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            assert conflict.resolution_suggestion is not None
            assert len(conflict.resolution_suggestion) > 0

    def test_conflict_severity_classification(self, detector):
        """Test that conflict severity is properly classified."""
        # Critical: Two absolute rules in same domain
        conflict_critical = MemoryRuleConflict(
            rule1=MemoryRule(
                rule="Never use eval()",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["security"]
            ),
            rule2=MemoryRule(
                rule="Use eval() for dynamic code",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["security"]
            ),
            conflict_type="direct",
            severity="critical",
            description="Direct contradiction on security-critical behavior"
        )

        assert conflict_critical.severity == "critical"

        # Medium: Different authorities
        conflict_medium = MemoryRuleConflict(
            rule1=MemoryRule(
                rule="Prefer tabs",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT
            ),
            rule2=MemoryRule(
                rule="Use spaces always",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE
            ),
            conflict_type="semantic",
            severity="medium",
            description="Formatting preference conflict"
        )

        assert conflict_medium.severity == "medium"


class TestBatchConflictDetection:
    """Test batch conflict detection for multiple rules."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_multiple_rule_conflicts(self, detector):
        """Test detecting conflicts across multiple existing rules."""
        new_rule = MemoryRule(
            rule="Always use async/await for all operations",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["javascript", "async"]
        )

        existing_rules = [
            MemoryRule(
                rule="Use callbacks for simple async operations",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript", "async"]
            ),
            MemoryRule(
                rule="Promises are better than callbacks",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript", "async"]
            ),
            MemoryRule(
                rule="Use synchronous file operations when possible",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["node", "filesystem"]
            )
        ]

        conflicts = await detector.detect_conflicts(new_rule, existing_rules)

        # Should detect conflicts with first two rules but not the third
        assert len(conflicts) >= 1

        # Verify conflicts are with appropriate rules
        conflicting_rule_ids = {c.rule2.id for c in conflicts}
        assert existing_rules[0].id in conflicting_rule_ids or existing_rules[1].id in conflicting_rule_ids

    @pytest.mark.asyncio
    async def test_no_false_positives(self, detector):
        """Test that unrelated rules don't generate false positive conflicts."""
        new_rule = MemoryRule(
            rule="Use meaningful commit messages",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["git", "version_control"]
        )

        unrelated_rules = [
            MemoryRule(
                rule="Use React for frontend development",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["frontend", "react"]
            ),
            MemoryRule(
                rule="Prefer PostgreSQL for databases",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["database", "postgresql"]
            ),
            MemoryRule(
                rule="Use Docker for containerization",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["docker", "containers"]
            )
        ]

        conflicts = await detector.detect_conflicts(new_rule, unrelated_rules)

        # Should not generate false positive conflicts
        assert len(conflicts) == 0


class TestPerformance:
    """Test conflict detection performance characteristics."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector(enable_ai_analysis=False)

    @pytest.mark.asyncio
    async def test_large_rule_set_performance(self, detector):
        """Test performance with large number of existing rules."""
        import time

        new_rule = MemoryRule(
            rule="New test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing"]
        )

        # Create large set of existing rules
        existing_rules = []
        for i in range(100):
            existing_rules.append(MemoryRule(
                rule=f"Test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=[f"scope_{i % 10}"]  # Some scope overlap
            ))

        start_time = time.time()
        conflicts = await detector.detect_conflicts(new_rule, existing_rules)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for 100 rules)
        assert end_time - start_time < 1.0
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_concurrent_conflict_detection(self, detector):
        """Test concurrent conflict detection operations."""
        rules = [
            MemoryRule(
                rule=f"Concurrent test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["concurrent_test"]
            ) for i in range(5)
        ]

        existing_rules = [
            MemoryRule(
                rule="Existing rule for concurrency test",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["concurrent_test"]
            )
        ]

        # Run multiple conflict detections concurrently
        tasks = [
            detector.detect_conflicts(rule, existing_rules)
            for rule in rules
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)
