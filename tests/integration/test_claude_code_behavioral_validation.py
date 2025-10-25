"""
Integration tests for Claude Code behavioral validation.

This module provides comprehensive behavioral validation testing to verify that
injected memory rules actually change Claude Code's behavior in measurable ways.

Test Approach:
1. Create controlled test scenarios with and without rules
2. Measure behavioral differences in responses
3. Validate rule compliance and quality metrics
4. Track token usage and performance overhead

Key Components:
- BehavioralValidator: Orchestrates behavioral testing
- ScenarioDefinition: Defines test scenarios with expected outcomes
- ResponseComparator: Compares responses with/without rules
- RuleComplianceChecker: Validates rule adherence in responses
- QualityMetrics: Measures response quality and correctness
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.python.common.core.context_injection import (
    AllocationStrategy,
    ClaudeBudgetManager,
    ClaudeCodeDetector,
    ClaudeCodeSession,
    ClaudeMdInjector,
    FormatManager,
    LiveRefreshManager,
    ProjectContext,
    ProjectContextDetector,
    RefreshMode,
    RuleFilter,
    RuleRetrieval,
    SessionTrigger,
    SystemPromptConfig,
    SystemPromptInjector,
    TokenBudgetManager,
    TriggerContext,
    TriggerManager,
    TriggerPhase,
    prepare_claude_code_session,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)


class BehaviorType(Enum):
    """Types of behavioral changes to validate."""

    CODE_STYLE = "code_style"  # Formatting, naming conventions
    API_USAGE = "api_usage"  # Preferred libraries, methods
    ERROR_HANDLING = "error_handling"  # Exception handling approaches
    DOCUMENTATION = "documentation"  # Comment styles, docstrings
    TESTING = "testing"  # Test coverage, assertion styles
    SECURITY = "security"  # Input validation, sanitization
    PERFORMANCE = "performance"  # Optimization preferences


@dataclass
class ComplianceMetrics:
    """
    Metrics for rule compliance in responses.

    Attributes:
        rules_applied: Number of rules that were applied
        rules_followed: Number of rules followed in response
        compliance_rate: Percentage of rules followed (0-100)
        violations: List of rules that were violated
        partial_compliance: Rules with partial compliance
    """

    rules_applied: int
    rules_followed: int
    compliance_rate: float
    violations: list[str] = field(default_factory=list)
    partial_compliance: list[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """
    Quality metrics for response evaluation.

    Attributes:
        correctness_score: Code correctness (0-100)
        completeness_score: Response completeness (0-100)
        style_score: Code style adherence (0-100)
        documentation_score: Documentation quality (0-100)
        overall_score: Weighted average quality score
    """

    correctness_score: float
    completeness_score: float
    style_score: float
    documentation_score: float
    overall_score: float


@dataclass
class TokenMetrics:
    """
    Token usage metrics.

    Attributes:
        context_tokens: Tokens used for context injection
        response_tokens: Tokens in generated response
        total_tokens: Total tokens used
        overhead_percentage: Context overhead as % of total
    """

    context_tokens: int
    response_tokens: int
    total_tokens: int
    overhead_percentage: float


@dataclass
class ScenarioDefinition:
    """
    Definition of a behavioral test scenario.

    Attributes:
        name: Scenario name
        behavior_type: Type of behavior being tested
        prompt: Test prompt to send
        rules: Memory rules to inject
        expected_patterns: Regex patterns that should appear in response
        forbidden_patterns: Regex patterns that should NOT appear
        quality_criteria: Specific quality criteria to check
        metadata: Additional scenario metadata
    """

    name: str
    behavior_type: BehaviorType
    prompt: str
    rules: list[MemoryRule]
    expected_patterns: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    quality_criteria: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralTestResult:
    """
    Result of a behavioral validation test.

    Attributes:
        scenario_name: Name of test scenario
        with_rules_response: Response with rules injected
        without_rules_response: Baseline response without rules
        compliance_metrics: Rule compliance metrics
        quality_metrics: Response quality metrics
        token_metrics: Token usage metrics
        behavior_changed: Whether measurable behavior change occurred
        test_passed: Whether test passed validation
        failure_reason: Reason if test failed
        execution_time_ms: Test execution time
        metadata: Additional result metadata
    """

    scenario_name: str
    with_rules_response: str
    without_rules_response: str
    compliance_metrics: ComplianceMetrics
    quality_metrics: QualityMetrics
    token_metrics: TokenMetrics
    behavior_changed: bool
    test_passed: bool
    failure_reason: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RuleComplianceChecker:
    """
    Validates that responses comply with injected rules.

    This checker analyzes generated responses and verifies that they follow
    the rules that were injected into the context.
    """

    def __init__(self):
        """Initialize the compliance checker."""
        pass

    def check_compliance(
        self, response: str, rules: list[MemoryRule]
    ) -> ComplianceMetrics:
        """
        Check if response complies with rules.

        Args:
            response: Generated response to check
            rules: Rules that should be followed

        Returns:
            ComplianceMetrics with compliance analysis
        """
        rules_followed = 0
        violations = []
        partial_compliance = []

        for rule in rules:
            compliance_status = self._check_rule_compliance(response, rule)

            if compliance_status == "full":
                rules_followed += 1
            elif compliance_status == "partial":
                partial_compliance.append(rule.name)
            else:
                violations.append(rule.name)

        compliance_rate = (
            (rules_followed / len(rules) * 100) if rules else 0.0
        )

        return ComplianceMetrics(
            rules_applied=len(rules),
            rules_followed=rules_followed,
            compliance_rate=compliance_rate,
            violations=violations,
            partial_compliance=partial_compliance,
        )

    def _check_rule_compliance(
        self, response: str, rule: MemoryRule
    ) -> str:
        """
        Check compliance with a specific rule.

        Args:
            response: Response text to check
            rule: Rule to validate against

        Returns:
            "full", "partial", or "none" compliance status
        """
        # Extract validation patterns from rule
        validation_patterns = self._extract_validation_patterns(rule)

        if not validation_patterns:
            # No specific patterns, do keyword matching
            return self._check_keyword_compliance(response, rule)

        # Check pattern matching (use DOTALL for multi-line patterns like docstrings)
        full_matches = 0
        for pattern in validation_patterns:
            if re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                full_matches += 1

        match_rate = full_matches / len(validation_patterns)

        if match_rate >= 0.8:
            return "full"
        elif match_rate >= 0.4:
            return "partial"
        else:
            return "none"

    def _extract_validation_patterns(
        self, rule: MemoryRule
    ) -> list[str]:
        """
        Extract validation patterns from rule metadata.

        Args:
            rule: Memory rule to extract patterns from

        Returns:
            List of regex patterns for validation
        """
        if not rule.metadata:
            return []

        patterns = rule.metadata.get("validation_patterns", [])
        if isinstance(patterns, list):
            return patterns

        return []

    def _check_keyword_compliance(
        self, response: str, rule: MemoryRule
    ) -> str:
        """
        Check compliance using keyword matching.

        Args:
            response: Response text
            rule: Rule to check

        Returns:
            Compliance status
        """
        # Extract keywords from rule text
        keywords = self._extract_keywords(rule.rule)

        if not keywords:
            return "partial"  # Can't determine without keywords

        # Check keyword presence
        found_keywords = sum(
            1 for kw in keywords if kw.lower() in response.lower()
        )

        match_rate = found_keywords / len(keywords)

        if match_rate >= 0.6:
            return "full"
        elif match_rate >= 0.3:
            return "partial"
        else:
            return "none"

    def _extract_keywords(self, rule_text: str) -> list[str]:
        """
        Extract important keywords from rule text.

        Args:
            rule_text: Rule text

        Returns:
            List of keywords
        """
        # Simple keyword extraction - split on common words
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

        words = re.findall(r"\w+", rule_text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return keywords[:5]  # Top 5 keywords


class ResponseComparator:
    """
    Compares responses with and without rules to detect behavioral changes.

    This comparator analyzes two responses to identify measurable differences
    that indicate rule injection had an effect.
    """

    def __init__(self):
        """Initialize the comparator."""
        pass

    def compare_responses(
        self,
        with_rules: str,
        without_rules: str,
        scenario: ScenarioDefinition,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Compare two responses to detect behavioral changes.

        Args:
            with_rules: Response with rules injected
            without_rules: Baseline response
            scenario: Test scenario definition

        Returns:
            Tuple of (behavior_changed, differences_dict)
        """
        differences = {
            "length_diff": len(with_rules) - len(without_rules),
            "expected_patterns_found": 0,
            "forbidden_patterns_avoided": 0,
            "structural_changes": [],
            "style_changes": [],
        }

        # Check expected patterns
        for pattern in scenario.expected_patterns:
            if re.search(pattern, with_rules, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                differences["expected_patterns_found"] += 1

        # Check forbidden patterns
        for pattern in scenario.forbidden_patterns:
            in_without = bool(
                re.search(pattern, without_rules, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            )
            in_with = bool(
                re.search(pattern, with_rules, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            )

            if in_without and not in_with:
                differences["forbidden_patterns_avoided"] += 1

        # Detect structural changes
        structural_changes = self._detect_structural_changes(
            with_rules, without_rules
        )
        differences["structural_changes"] = structural_changes

        # Detect style changes
        style_changes = self._detect_style_changes(with_rules, without_rules)
        differences["style_changes"] = style_changes

        # Determine if behavior changed
        behavior_changed = (
            differences["expected_patterns_found"] > 0
            or differences["forbidden_patterns_avoided"] > 0
            or len(structural_changes) > 0
            or len(style_changes) > 0
        )

        return behavior_changed, differences

    def _detect_structural_changes(
        self, with_rules: str, without_rules: str
    ) -> list[str]:
        """
        Detect structural changes in code/content.

        Args:
            with_rules: Response with rules
            without_rules: Response without rules

        Returns:
            List of detected structural changes
        """
        changes = []

        # Check for function definitions
        with_funcs = set(re.findall(r"def\s+(\w+)", with_rules))
        without_funcs = set(re.findall(r"def\s+(\w+)", without_rules))

        new_funcs = with_funcs - without_funcs
        if new_funcs:
            changes.append(f"Added functions: {', '.join(new_funcs)}")

        # Check for class definitions
        with_classes = set(re.findall(r"class\s+(\w+)", with_rules))
        without_classes = set(re.findall(r"class\s+(\w+)", without_rules))

        new_classes = with_classes - without_classes
        if new_classes:
            changes.append(f"Added classes: {', '.join(new_classes)}")

        # Check for imports
        with_imports = set(
            re.findall(r"(?:from|import)\s+([\w.]+)", with_rules)
        )
        without_imports = set(
            re.findall(r"(?:from|import)\s+([\w.]+)", without_rules)
        )

        new_imports = with_imports - without_imports
        if new_imports:
            changes.append(f"Added imports: {', '.join(new_imports)}")

        return changes

    def _detect_style_changes(
        self, with_rules: str, without_rules: str
    ) -> list[str]:
        """
        Detect code style changes.

        Args:
            with_rules: Response with rules
            without_rules: Response without rules

        Returns:
            List of detected style changes
        """
        changes = []

        # Check for docstrings
        with_docstrings = len(re.findall(r'""".*?"""', with_rules, re.DOTALL))
        without_docstrings = len(
            re.findall(r'""".*?"""', without_rules, re.DOTALL)
        )

        if with_docstrings > without_docstrings:
            changes.append(
                f"Added {with_docstrings - without_docstrings} docstrings"
            )

        # Check for type hints
        with_hints = len(re.findall(r":\s*\w+(?:\[.*?\])?(?:\s*=|\s*\))", with_rules))
        without_hints = len(
            re.findall(r":\s*\w+(?:\[.*?\])?(?:\s*=|\s*\))", without_rules)
        )

        if with_hints > without_hints:
            changes.append(
                f"Added {with_hints - without_hints} type hints"
            )

        # Check for comments
        with_comments = len(re.findall(r"#.*$", with_rules, re.MULTILINE))
        without_comments = len(
            re.findall(r"#.*$", without_rules, re.MULTILINE)
        )

        if with_comments > without_comments:
            changes.append(
                f"Added {with_comments - without_comments} comments"
            )

        return changes


class QualityAnalyzer:
    """
    Analyzes response quality against multiple criteria.

    Provides scoring for correctness, completeness, style, and documentation.
    """

    def __init__(self):
        """Initialize the quality analyzer."""
        pass

    def analyze_quality(
        self, response: str, scenario: ScenarioDefinition
    ) -> QualityMetrics:
        """
        Analyze response quality.

        Args:
            response: Response to analyze
            scenario: Test scenario with quality criteria

        Returns:
            QualityMetrics with scores
        """
        # Calculate individual scores
        correctness = self._score_correctness(response, scenario)
        completeness = self._score_completeness(response, scenario)
        style = self._score_style(response, scenario)
        documentation = self._score_documentation(response, scenario)

        # Calculate weighted overall score
        overall = (
            correctness * 0.35
            + completeness * 0.25
            + style * 0.20
            + documentation * 0.20
        )

        return QualityMetrics(
            correctness_score=correctness,
            completeness_score=completeness,
            style_score=style,
            documentation_score=documentation,
            overall_score=overall,
        )

    def _score_correctness(
        self, response: str, scenario: ScenarioDefinition
    ) -> float:
        """Score code correctness."""
        score = 100.0

        # Check for syntax errors (basic heuristics)
        if "SyntaxError" in response or "IndentationError" in response:
            score -= 50.0

        # Check for obvious errors
        error_patterns = [
            r"undefined",
            r"not defined",
            r"AttributeError",
            r"TypeError",
        ]

        for pattern in error_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 10.0

        return max(0.0, score)

    def _score_completeness(
        self, response: str, scenario: ScenarioDefinition
    ) -> float:
        """Score response completeness."""
        criteria = scenario.quality_criteria.get("completeness", {})

        if not criteria:
            return 80.0  # Default score without criteria

        score = 0.0
        required_items = criteria.get("required_items", [])

        if required_items:
            found = sum(1 for item in required_items if item in response)
            score = (found / len(required_items)) * 100

        return score

    def _score_style(
        self, response: str, scenario: ScenarioDefinition
    ) -> float:
        """Score code style."""
        score = 100.0

        # Check for basic style issues
        lines = response.split("\n")

        # Line length
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines > len(lines) * 0.2:  # More than 20%
            score -= 15.0

        # Naming conventions
        snake_case_funcs = len(re.findall(r"def\s+[a-z_]+\s*\(", response))
        camel_case_funcs = len(re.findall(r"def\s+[a-z][a-zA-Z]+\s*\(", response))

        if camel_case_funcs > snake_case_funcs:
            score -= 10.0  # Python prefers snake_case

        return max(0.0, score)

    def _score_documentation(
        self, response: str, scenario: ScenarioDefinition
    ) -> float:
        """Score documentation quality."""
        score = 0.0

        # Count documentation elements
        docstrings = len(re.findall(r'""".*?"""', response, re.DOTALL))
        functions = len(re.findall(r"def\s+\w+", response))
        classes = len(re.findall(r"class\s+\w+", response))

        total_entities = functions + classes

        if total_entities > 0:
            doc_ratio = docstrings / total_entities
            score = min(doc_ratio * 100, 100.0)
        else:
            score = 50.0  # No entities to document

        return score


class BehavioralValidator:
    """
    Main orchestrator for behavioral validation testing.

    Coordinates scenario execution, response comparison, compliance checking,
    and result aggregation.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        project_root: Path,
        automated_mode: bool = True,
    ):
        """
        Initialize the behavioral validator.

        Args:
            memory_manager: MemoryManager for rule storage/retrieval
            project_root: Project root directory
            automated_mode: Use mocked responses vs real Claude API
        """
        self.memory_manager = memory_manager
        self.project_root = project_root
        self.automated_mode = automated_mode

        self.compliance_checker = RuleComplianceChecker()
        self.comparator = ResponseComparator()
        self.quality_analyzer = QualityAnalyzer()

    async def run_scenario(
        self, scenario: ScenarioDefinition
    ) -> BehavioralTestResult:
        """
        Run a single behavioral test scenario.

        Args:
            scenario: Test scenario to execute

        Returns:
            BehavioralTestResult with validation results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Get baseline response (without rules)
            without_rules_response = await self._get_response(
                scenario.prompt, inject_rules=False
            )

            # Store rules in memory
            for rule in scenario.rules:
                await self.memory_manager.add_rule(rule)

            # Get response with rules injected
            with_rules_response = await self._get_response(
                scenario.prompt, inject_rules=True, rules=scenario.rules
            )

            # Compare responses
            behavior_changed, differences = self.comparator.compare_responses(
                with_rules_response, without_rules_response, scenario
            )

            # Check rule compliance
            compliance = self.compliance_checker.check_compliance(
                with_rules_response, scenario.rules
            )

            # Analyze quality
            quality = self.quality_analyzer.analyze_quality(
                with_rules_response, scenario
            )

            # Calculate token metrics
            token_metrics = self._calculate_token_metrics(
                with_rules_response, scenario.rules
            )

            # Determine test pass/fail
            test_passed = (
                behavior_changed
                and compliance.compliance_rate >= 70.0
                and quality.overall_score >= 60.0
            )

            failure_reason = None
            if not test_passed:
                if not behavior_changed:
                    failure_reason = "No measurable behavior change detected"
                elif compliance.compliance_rate < 70.0:
                    failure_reason = (
                        f"Low compliance rate: {compliance.compliance_rate:.1f}%"
                    )
                else:
                    failure_reason = (
                        f"Low quality score: {quality.overall_score:.1f}"
                    )

            execution_time = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

            return BehavioralTestResult(
                scenario_name=scenario.name,
                with_rules_response=with_rules_response,
                without_rules_response=without_rules_response,
                compliance_metrics=compliance,
                quality_metrics=quality,
                token_metrics=token_metrics,
                behavior_changed=behavior_changed,
                test_passed=test_passed,
                failure_reason=failure_reason,
                execution_time_ms=execution_time,
                metadata=differences,
            )

        finally:
            # Cleanup - remove test rules
            for rule in scenario.rules:
                try:
                    await self.memory_manager.delete_rule(rule.id)
                except Exception:
                    pass  # Best effort cleanup

    async def _get_response(
        self,
        prompt: str,
        inject_rules: bool,
        rules: list[MemoryRule] | None = None,
    ) -> str:
        """
        Get response from Claude (mocked or real).

        Args:
            prompt: Prompt to send
            inject_rules: Whether to inject rules
            rules: Rules to inject if applicable

        Returns:
            Generated response text
        """
        if self.automated_mode:
            # Return mock response based on rules
            return self._generate_mock_response(prompt, inject_rules, rules)
        else:
            # Would make real API call here
            raise NotImplementedError(
                "Live Claude API integration not implemented"
            )

    def _generate_mock_response(
        self,
        prompt: str,
        with_rules: bool,
        rules: list[MemoryRule] | None = None,
    ) -> str:
        """
        Generate mock response for automated testing.

        Args:
            prompt: Input prompt
            with_rules: Whether rules are injected
            rules: Injected rules

        Returns:
            Mock response text
        """
        # Base response without rules
        base_response = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

        if not with_rules or not rules:
            return base_response

        # Modify response based on rules
        enhanced_response = base_response

        for rule in rules:
            if "docstring" in rule.rule.lower() or "document" in rule.rule.lower():
                # Add docstrings
                enhanced_response = '''
def process_data(data):
    """
    Process data by filtering and doubling positive values.

    Args:
        data: List of numeric values

    Returns:
        List of doubled positive values
    """
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''

            if "type hint" in rule.rule.lower() or "typing" in rule.rule.lower():
                # Add type hints
                enhanced_response = '''

def process_data(data: List[float]) -> List[float]:
    """
    Process data by filtering and doubling positive values.

    Args:
        data: List of numeric values

    Returns:
        List of doubled positive values
    """
    result: List[float] = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''

            if "error handling" in rule.rule.lower() or "exception" in rule.rule.lower():
                # Add error handling
                enhanced_response = '''

def process_data(data: List[float]) -> List[float]:
    """
    Process data by filtering and doubling positive values.

    Args:
        data: List of numeric values

    Returns:
        List of doubled positive values

    Raises:
        ValueError: If data is None or contains non-numeric values
    """
    if data is None:
        raise ValueError("Data cannot be None")

    result: List[float] = []
    for item in data:
        try:
            if item > 0:
                result.append(item * 2)
        except TypeError:
            raise ValueError(f"Non-numeric value encountered: {item}")

    return result
'''

        return enhanced_response

    def _calculate_token_metrics(
        self, response: str, rules: list[MemoryRule]
    ) -> TokenMetrics:
        """
        Calculate token usage metrics.

        Args:
            response: Generated response
            rules: Injected rules

        Returns:
            TokenMetrics with usage data
        """
        # Estimate tokens (rough approximation)
        context_tokens = sum(len(r.rule.split()) * 1.3 for r in rules)
        context_tokens = int(context_tokens)

        response_tokens = int(len(response.split()) * 1.3)
        total_tokens = context_tokens + response_tokens

        overhead_percentage = (
            (context_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
        )

        return TokenMetrics(
            context_tokens=context_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            overhead_percentage=overhead_percentage,
        )


# Test Fixtures


@pytest.fixture
async def memory_manager(tmp_path):
    """Create a memory manager for testing."""
    # Use mock memory manager for automated testing
    manager = AsyncMock(spec=MemoryManager)
    manager.add_rule = AsyncMock()
    manager.delete_rule = AsyncMock()
    manager.get_rules = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def project_root(tmp_path):
    """Create a temporary project root."""
    return tmp_path


@pytest.fixture
def behavioral_validator(memory_manager, project_root):
    """Create behavioral validator."""
    return BehavioralValidator(
        memory_manager=memory_manager,
        project_root=project_root,
        automated_mode=True,
    )


@pytest.fixture
def sample_rules():
    """Create sample rules for testing."""
    now = datetime.now(timezone.utc)

    return [
        MemoryRule(
            id="rule_doc",
            name="docstring_requirement",
            rule="Always add comprehensive docstrings to all functions",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={
                "priority": 100,
                "validation_patterns": [
                    r'""".*?"""',
                    r"Args:",
                    r"Returns:",
                ],
            },
        ),
        MemoryRule(
            id="rule_types",
            name="type_hints",
            rule="Use type hints for all function parameters and return values",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={
                "priority": 95,
                "validation_patterns": [
                    r":\s*List\[",
                    r"->\s*List\[",
                ],
            },
        ),
    ]


# Integration Tests


@pytest.mark.asyncio
class TestBehavioralValidator:
    """Test behavioral validation framework."""

    async def test_code_style_scenario(
        self, behavioral_validator, sample_rules
    ):
        """Test code style behavioral change detection."""
        scenario = ScenarioDefinition(
            name="code_style_validation",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="Write a function to process a list of numbers",
            rules=[sample_rules[0]],  # Docstring rule
            expected_patterns=[r'""".*?"""', r"Args:", r"Returns:"],
            forbidden_patterns=[],
            quality_criteria={
                "completeness": {
                    "required_items": ["def", "return", "Args:", "Returns:"]
                }
            },
        )

        result = await behavioral_validator.run_scenario(scenario)

        assert result.test_passed
        assert result.behavior_changed
        assert result.compliance_metrics.compliance_rate >= 70.0
        assert result.quality_metrics.documentation_score > 50.0

    async def test_type_hints_scenario(
        self, behavioral_validator, sample_rules
    ):
        """Test type hints behavioral change."""
        scenario = ScenarioDefinition(
            name="type_hints_validation",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="Create a data processing function",
            rules=[sample_rules[1]],  # Type hints rule
            expected_patterns=[r":\s*List\[", r"->\s*List\["],
            forbidden_patterns=[],
        )

        result = await behavioral_validator.run_scenario(scenario)

        assert result.test_passed
        assert result.behavior_changed
        assert "typ" in result.with_rules_response.lower()  # Contains "typing" or "type"

    async def test_combined_rules_scenario(
        self, behavioral_validator, sample_rules
    ):
        """Test multiple rules applied together."""
        scenario = ScenarioDefinition(
            name="combined_rules",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="Write a robust data processing function",
            rules=sample_rules,  # Both docstring and type hints
            expected_patterns=[
                r'""".*?"""',
                r":\s*List\[",
                r"Args:",
                r"Returns:",
            ],
        )

        result = await behavioral_validator.run_scenario(scenario)

        assert result.test_passed
        assert result.behavior_changed
        assert result.compliance_metrics.rules_applied == 2
        assert len(result.metadata["structural_changes"]) > 0

    async def test_token_overhead_measurement(
        self, behavioral_validator, sample_rules
    ):
        """Test token usage overhead calculation."""
        scenario = ScenarioDefinition(
            name="token_overhead",
            behavior_type=BehaviorType.PERFORMANCE,
            prompt="Simple function",
            rules=sample_rules,
        )

        result = await behavioral_validator.run_scenario(scenario)

        assert result.token_metrics.context_tokens > 0
        assert result.token_metrics.response_tokens > 0
        assert result.token_metrics.total_tokens > 0
        assert 0 <= result.token_metrics.overhead_percentage <= 100

    async def test_quality_metrics_calculation(
        self, behavioral_validator, sample_rules
    ):
        """Test quality metrics scoring."""
        scenario = ScenarioDefinition(
            name="quality_test",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="Write quality code",
            rules=sample_rules,
        )

        result = await behavioral_validator.run_scenario(scenario)

        # Verify all quality scores are calculated
        assert 0 <= result.quality_metrics.correctness_score <= 100
        assert 0 <= result.quality_metrics.completeness_score <= 100
        assert 0 <= result.quality_metrics.style_score <= 100
        assert 0 <= result.quality_metrics.documentation_score <= 100
        assert 0 <= result.quality_metrics.overall_score <= 100


class TestRuleComplianceChecker:
    """Test rule compliance checking."""

    def test_full_compliance_detection(self):
        """Test detection of full rule compliance."""
        checker = RuleComplianceChecker()

        now = datetime.now(timezone.utc)
        rule = MemoryRule(
            id="test_rule",
            name="test",
            rule="Add docstrings to functions",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={
                "validation_patterns": [r'""".*?"""', r"Args:", r"Returns:"]
            },
        )

        response = '''
def test():
    """
    Test function.

    Args:
        None

    Returns:
        None
    """
    pass
'''

        metrics = checker.check_compliance(response, [rule])

        assert metrics.rules_applied == 1
        assert metrics.rules_followed == 1
        assert metrics.compliance_rate == 100.0
        assert len(metrics.violations) == 0

    def test_partial_compliance_detection(self):
        """Test detection of partial compliance."""
        checker = RuleComplianceChecker()

        now = datetime.now(timezone.utc)
        rule = MemoryRule(
            id="test_rule",
            name="test",
            rule="Add comprehensive docstrings",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={
                "validation_patterns": [r'""".*?"""', r"Args:", r"Returns:"]
            },
        )

        # Response with docstring but missing Args/Returns
        response = '''
def test():
    """Test function."""
    pass
'''

        metrics = checker.check_compliance(response, [rule])

        assert metrics.compliance_rate < 100.0
        assert len(metrics.partial_compliance) > 0 or metrics.rules_followed < 1


class TestResponseComparator:
    """Test response comparison functionality."""

    def test_behavioral_change_detection(self):
        """Test detection of behavioral changes."""
        comparator = ResponseComparator()

        scenario = ScenarioDefinition(
            name="test",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="test",
            rules=[],
            expected_patterns=[r'""".*?"""'],
        )

        with_rules = 'def test():\n    """Docstring"""\n    pass'
        without_rules = "def test():\n    pass"

        changed, diffs = comparator.compare_responses(
            with_rules, without_rules, scenario
        )

        assert changed
        assert diffs["expected_patterns_found"] > 0

    def test_structural_change_detection(self):
        """Test detection of structural changes."""
        comparator = ResponseComparator()

        with_rules = "from typing import List\n\ndef test(): pass"
        without_rules = "def test(): pass"

        changes = comparator._detect_structural_changes(
            with_rules, without_rules
        )

        assert len(changes) > 0
        assert any("import" in c.lower() for c in changes)

    def test_style_change_detection(self):
        """Test detection of style changes."""
        comparator = ResponseComparator()

        with_rules = 'def test():\n    """Docstring"""\n    pass'
        without_rules = "def test():\n    pass"

        changes = comparator._detect_style_changes(with_rules, without_rules)

        assert len(changes) > 0
        assert any("docstring" in c.lower() for c in changes)


class TestQualityAnalyzer:
    """Test quality analysis."""

    def test_documentation_scoring(self):
        """Test documentation quality scoring."""
        analyzer = QualityAnalyzer()

        scenario = ScenarioDefinition(
            name="test",
            behavior_type=BehaviorType.DOCUMENTATION,
            prompt="test",
            rules=[],
        )

        # Well-documented code
        response = '''
def test1():
    """Docstring 1"""
    pass

def test2():
    """Docstring 2"""
    pass
'''

        metrics = analyzer.analyze_quality(response, scenario)

        assert metrics.documentation_score >= 80.0

    def test_style_scoring(self):
        """Test style quality scoring."""
        analyzer = QualityAnalyzer()

        scenario = ScenarioDefinition(
            name="test",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="test",
            rules=[],
        )

        # Good style code
        response = """
def process_data():
    pass

def calculate_result():
    pass
"""

        metrics = analyzer.analyze_quality(response, scenario)

        assert metrics.style_score >= 70.0

    def test_correctness_scoring(self):
        """Test correctness scoring."""
        analyzer = QualityAnalyzer()

        scenario = ScenarioDefinition(
            name="test",
            behavior_type=BehaviorType.CODE_STYLE,
            prompt="test",
            rules=[],
        )

        # Correct code
        response = "def test():\n    return 42"

        metrics = analyzer.analyze_quality(response, scenario)

        assert metrics.correctness_score >= 90.0

        # Code with errors
        error_response = "def test():\n    undefined_variable"

        error_metrics = analyzer.analyze_quality(error_response, scenario)

        assert error_metrics.correctness_score < metrics.correctness_score
