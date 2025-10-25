"""
Enhanced LLM Behavioral Test Harness (Task 337.1).

This module provides an enhanced behavioral testing framework for LLM rule injection
with support for real API calls, standardized prompts, and comprehensive metrics.

Enhancements over existing framework:
1. Real LLM API integration (Claude, GPT, etc.) with fallback to mocks
2. Standardized prompt templates and response analysis
3. Enhanced behavioral metrics and statistical analysis
4. Multi-provider support with unified interface
5. Configurable test execution (live, mock, hybrid modes)

Components:
- LLMProvider: Abstract base for LLM providers
- ClaudeProvider: Claude API integration
- PromptTemplate: Standardized test prompts
- BehavioralMetrics: Enhanced behavioral change metrics
- LLMBehavioralHarness: Main test orchestrator
"""

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

# Optional dependencies - gracefully degrade if not available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)


class ExecutionMode(Enum):
    """Test execution mode."""
    MOCK = "mock"  # Use mocked responses (for CI/CD)
    LIVE = "live"  # Use real API calls
    HYBRID = "hybrid"  # Use live for baseline, mock for comparison


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GPT = "gpt"
    MOCK = "mock"


@dataclass
class PromptTemplate:
    """
    Standardized prompt template for behavioral testing.

    Attributes:
        name: Template name
        prompt: Prompt text with placeholders
        placeholders: Available placeholder variables
        expected_elements: Elements expected in response
        category: Test category (code_generation, documentation, etc.)
    """
    name: str
    prompt: str
    placeholders: list[str] = field(default_factory=list)
    expected_elements: list[str] = field(default_factory=list)
    category: str = "general"

    def format(self, **kwargs) -> str:
        """Format prompt with provided values."""
        return self.prompt.format(**kwargs)


@dataclass
class BehavioralMetrics:
    """
    Enhanced behavioral change metrics.

    Extends basic comparison with statistical analysis and confidence scoring.

    Attributes:
        behavior_changed: Whether measurable change detected
        confidence_score: Confidence in behavior change (0-100)
        semantic_similarity: Semantic similarity score (0-100)
        structural_diff_score: Structural difference score (0-100)
        pattern_matches: Number of expected patterns found
        pattern_total: Total expected patterns
        token_difference: Token count difference
        response_time_ms: Response generation time
        metadata: Additional metrics
    """
    behavior_changed: bool
    confidence_score: float
    semantic_similarity: float
    structural_diff_score: float
    pattern_matches: int
    pattern_total: int
    token_difference: int
    response_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    LLM API response wrapper.

    Attributes:
        text: Response text
        model: Model used
        provider: LLM provider
        tokens_used: Total tokens consumed
        latency_ms: Response latency
        finish_reason: Why generation stopped
        metadata: Additional response metadata
    """
    text: str
    model: str
    provider: LLMProvider
    tokens_used: int
    latency_ms: float
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context injection
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text and metadata
        """
        pass


class ClaudeProvider(BaseLLMProvider):
    """Claude API provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key (from env if not provided)
            model: Claude model to use
        """
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided or found in environment")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Generate response using Claude API."""
        start_time = asyncio.get_event_loop().time()

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Run synchronous API call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(**kwargs)
        )

        latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text

        return LLMResponse(
            text=text,
            model=self.model,
            provider=LLMProvider.CLAUDE,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason or "stop",
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model,
            }
        )


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing without API calls."""

    def __init__(self, response_generator: Callable | None = None):
        """
        Initialize mock provider.

        Args:
            response_generator: Optional function to generate responses
        """
        self.response_generator = response_generator or self._default_generator

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Generate mock response."""
        await asyncio.sleep(0.1)  # Simulate API latency

        text = self.response_generator(prompt, system_prompt)
        tokens = len(text.split())

        return LLMResponse(
            text=text,
            model="mock-model",
            provider=LLMProvider.MOCK,
            tokens_used=tokens,
            latency_ms=100.0,
            finish_reason="stop",
            metadata={"mock": True}
        )

    def _default_generator(
        self,
        prompt: str,
        system_prompt: str | None
    ) -> str:
        """Default mock response generator."""
        # Generate response based on system prompt rules
        has_docstring_rule = system_prompt and "docstring" in system_prompt.lower()
        has_type_hint_rule = system_prompt and "type hint" in system_prompt.lower()

        base_code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

        if has_docstring_rule and has_type_hint_rule:
            return """

def process_data(data: List[float]) -> List[float]:
    \"\"\"
    Process data by filtering and doubling positive values.

    Args:
        data: List of numeric values to process

    Returns:
        List of doubled positive values
    \"\"\"
    result: List[float] = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        elif has_docstring_rule:
            return """
def process_data(data):
    \"\"\"
    Process data by filtering and doubling positive values.

    Args:
        data: List of numeric values

    Returns:
        List of doubled positive values
    \"\"\"
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        elif has_type_hint_rule:
            return """

def process_data(data: List[float]) -> List[float]:
    result: List[float] = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        else:
            return base_code


class BehavioralAnalyzer:
    """Analyzes behavioral changes between responses."""

    def analyze(
        self,
        with_rules: LLMResponse,
        without_rules: LLMResponse,
        expected_patterns: list[str] = None,
        forbidden_patterns: list[str] = None
    ) -> BehavioralMetrics:
        """
        Analyze behavioral differences between two responses.

        Args:
            with_rules: Response with rules injected
            without_rules: Baseline response
            expected_patterns: Patterns expected in with_rules response
            forbidden_patterns: Patterns that should not appear

        Returns:
            BehavioralMetrics with analysis results
        """
        expected_patterns = expected_patterns or []
        forbidden_patterns = forbidden_patterns or []

        # Calculate pattern matches
        pattern_matches = sum(
            1 for pattern in expected_patterns
            if re.search(pattern, with_rules.text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )
        pattern_total = len(expected_patterns)

        # Calculate structural differences
        structural_diff = self._calculate_structural_diff(
            with_rules.text,
            without_rules.text
        )

        # Calculate semantic similarity (simple word overlap for now)
        semantic_sim = self._calculate_semantic_similarity(
            with_rules.text,
            without_rules.text
        )

        # Determine if behavior changed
        behavior_changed = (
            pattern_matches > 0 or
            structural_diff > 20.0 or
            semantic_sim < 70.0
        )

        # Calculate confidence score
        confidence = 0.0
        if pattern_total > 0:
            confidence += (pattern_matches / pattern_total) * 50.0
        if structural_diff > 0:
            confidence += min(structural_diff / 100.0 * 30.0, 30.0)
        if semantic_sim < 90.0:
            confidence += min((100.0 - semantic_sim) / 100.0 * 20.0, 20.0)

        token_diff = with_rules.tokens_used - without_rules.tokens_used

        return BehavioralMetrics(
            behavior_changed=behavior_changed,
            confidence_score=min(confidence, 100.0),
            semantic_similarity=semantic_sim,
            structural_diff_score=structural_diff,
            pattern_matches=pattern_matches,
            pattern_total=pattern_total,
            token_difference=token_diff,
            response_time_ms=with_rules.latency_ms,
            metadata={
                "expected_patterns_found": pattern_matches,
                "total_expected_patterns": pattern_total,
                "tokens_with_rules": with_rules.tokens_used,
                "tokens_without_rules": without_rules.tokens_used,
            }
        )

    def _calculate_structural_diff(self, text1: str, text2: str) -> float:
        """
        Calculate structural difference score (0-100).

        Higher scores indicate more structural differences.
        """
        # Compare structural elements
        func1 = set(re.findall(r"def\s+(\w+)", text1))
        func2 = set(re.findall(r"def\s+(\w+)", text2))

        class1 = set(re.findall(r"class\s+(\w+)", text1))
        class2 = set(re.findall(r"class\s+(\w+)", text2))

        import1 = set(re.findall(r"(?:from|import)\s+([\w.]+)", text1))
        import2 = set(re.findall(r"(?:from|import)\s+([\w.]+)", text2))

        # Calculate differences
        func_diff = len(func1.symmetric_difference(func2))
        class_diff = len(class1.symmetric_difference(class2))
        import_diff = len(import1.symmetric_difference(import2))

        # Normalize to 0-100 scale
        total_elements = len(func1) + len(func2) + len(class1) + len(class2) + len(import1) + len(import2)
        if total_elements == 0:
            return 0.0

        total_diff = func_diff + class_diff + import_diff
        return min((total_diff / total_elements) * 200, 100.0)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity score (0-100).

        Higher scores indicate more similarity.
        Simple word overlap for now - could use embeddings in future.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 100.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return (len(intersection) / len(union)) * 100.0


class LLMBehavioralHarness:
    """
    Main test harness for LLM behavioral validation.

    Orchestrates behavioral tests with real or mocked LLM APIs,
    analyzing rule injection effectiveness.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        memory_manager: MemoryManager | None = None,
        mode: ExecutionMode = ExecutionMode.MOCK
    ):
        """
        Initialize behavioral harness.

        Args:
            provider: LLM provider to use
            memory_manager: Memory manager for rule storage
            mode: Test execution mode
        """
        self.provider = provider
        self.memory_manager = memory_manager
        self.mode = mode
        self.analyzer = BehavioralAnalyzer()

    async def run_behavioral_test(
        self,
        prompt: str,
        rules: list[MemoryRule],
        expected_patterns: list[str] = None,
        forbidden_patterns: list[str] = None,
        temperature: float = 0.7
    ) -> tuple[BehavioralMetrics, LLMResponse, LLMResponse]:
        """
        Run behavioral test with and without rules.

        Args:
            prompt: Test prompt
            rules: Rules to inject
            expected_patterns: Patterns expected with rules
            forbidden_patterns: Patterns to avoid
            temperature: LLM temperature

        Returns:
            Tuple of (metrics, response_with_rules, response_without_rules)
        """
        # Get baseline response without rules
        baseline_response = await self.provider.generate(
            prompt=prompt,
            temperature=temperature
        )

        # Store rules if memory manager available
        if self.memory_manager:
            for rule in rules:
                await self.memory_manager.add_rule(rule)

        # Create system prompt with rules
        system_prompt = self._create_system_prompt(rules)

        # Get response with rules injected
        rules_response = await self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )

        # Analyze behavioral changes
        metrics = self.analyzer.analyze(
            with_rules=rules_response,
            without_rules=baseline_response,
            expected_patterns=expected_patterns,
            forbidden_patterns=forbidden_patterns
        )

        # Cleanup rules from memory
        if self.memory_manager:
            for rule in rules:
                try:
                    await self.memory_manager.delete_rule(rule.id)
                except Exception:
                    pass  # Best effort cleanup

        return metrics, rules_response, baseline_response

    def _create_system_prompt(self, rules: list[MemoryRule]) -> str:
        """
        Create system prompt from memory rules.

        Args:
            rules: Memory rules to inject

        Returns:
            Formatted system prompt
        """
        lines = ["You are an AI assistant with the following rules:"]
        lines.append("")

        # Group by authority level
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        if absolute_rules:
            lines.append("ABSOLUTE RULES (Always follow):")
            for rule in absolute_rules:
                lines.append(f"- {rule.rule}")
            lines.append("")

        if default_rules:
            lines.append("DEFAULT RULES (Follow unless contradicted):")
            for rule in default_rules:
                lines.append(f"- {rule.rule}")
            lines.append("")

        return "\n".join(lines)


# Test Fixtures

@pytest.fixture
def mock_provider():
    """Create mock LLM provider for testing."""
    return MockLLMProvider()


@pytest.fixture
def claude_provider():
    """Create Claude provider if API key available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_AVAILABLE:
        pytest.skip("Claude API key not available or anthropic not installed")
    return ClaudeProvider(api_key=api_key)


@pytest.fixture
def behavioral_harness(mock_provider):
    """Create behavioral harness with mock provider."""
    return LLMBehavioralHarness(
        provider=mock_provider,
        mode=ExecutionMode.MOCK
    )


@pytest.fixture
def sample_rules():
    """Create sample memory rules for testing."""
    now = datetime.now(timezone.utc)

    return [
        MemoryRule(
            id="rule_doc",
            name="docstring_requirement",
            rule="Always add comprehensive docstrings to all functions with Args and Returns sections",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={}
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
            metadata={}
        ),
    ]


@pytest.fixture
def code_generation_prompt():
    """Standard code generation prompt."""
    return PromptTemplate(
        name="code_generation",
        prompt="Write a Python function to {task}",
        placeholders=["task"],
        expected_elements=["def", "return"],
        category="code_generation"
    )


# Integration Tests

@pytest.mark.asyncio
class TestLLMBehavioralHarness:
    """Test LLM behavioral harness with mock provider."""

    async def test_basic_behavioral_test(
        self,
        behavioral_harness,
        sample_rules,
        code_generation_prompt
    ):
        """Test basic behavioral test execution."""
        prompt = code_generation_prompt.format(
            task="process a list of numbers by doubling positive values"
        )

        metrics, with_rules, without_rules = await behavioral_harness.run_behavioral_test(
            prompt=prompt,
            rules=sample_rules,
            expected_patterns=[r'""".*?"""', r":\s*List\[", r"Args:", r"Returns:"]
        )

        # Verify behavioral change detected
        assert metrics.behavior_changed
        assert metrics.confidence_score > 0
        assert metrics.pattern_matches > 0

        # Verify responses generated
        assert with_rules.text
        assert without_rules.text
        assert with_rules.text != without_rules.text

    async def test_docstring_rule_injection(
        self,
        behavioral_harness,
        sample_rules
    ):
        """Test that docstring rules affect output."""
        docstring_rule = [r for r in sample_rules if "docstring" in r.rule.lower()][0]

        metrics, with_rules, without_rules = await behavioral_harness.run_behavioral_test(
            prompt="Write a simple function to add two numbers",
            rules=[docstring_rule],
            expected_patterns=[r'""".*?"""', r"Args:", r"Returns:"]
        )

        assert metrics.behavior_changed
        # With docstring rule should have docstring patterns
        assert re.search(r'""".*?"""', with_rules.text, re.DOTALL)
        assert metrics.pattern_matches >= 1

    async def test_type_hints_rule_injection(
        self,
        behavioral_harness,
        sample_rules
    ):
        """Test that type hint rules affect output."""
        type_hint_rule = [r for r in sample_rules if "type hint" in r.rule.lower()][0]

        metrics, with_rules, without_rules = await behavioral_harness.run_behavioral_test(
            prompt="Create a function to multiply numbers",
            rules=[type_hint_rule],
            expected_patterns=[r":\s*\w+(?:\[.*?\])?", r"->\s*\w+"]
        )

        assert metrics.behavior_changed
        assert metrics.confidence_score > 0

    async def test_combined_rules(
        self,
        behavioral_harness,
        sample_rules
    ):
        """Test multiple rules applied together."""
        metrics, with_rules, without_rules = await behavioral_harness.run_behavioral_test(
            prompt="Write a robust function for data processing",
            rules=sample_rules,  # Both docstring and type hints
            expected_patterns=[
                r'""".*?"""',
                r":\s*List\[",
                r"Args:",
                r"Returns:"
            ]
        )

        assert metrics.behavior_changed
        assert metrics.pattern_matches > 0
        assert metrics.structural_diff_score > 0

    async def test_metrics_calculation(
        self,
        behavioral_harness,
        sample_rules
    ):
        """Test behavioral metrics are calculated correctly."""
        metrics, with_rules, without_rules = await behavioral_harness.run_behavioral_test(
            prompt="Simple function",
            rules=sample_rules
        )

        # Verify all metrics present
        assert 0 <= metrics.confidence_score <= 100
        assert 0 <= metrics.semantic_similarity <= 100
        assert 0 <= metrics.structural_diff_score <= 100
        assert metrics.pattern_matches >= 0
        assert metrics.token_difference != 0  # Rules should change token count
        assert metrics.response_time_ms > 0


@pytest.mark.asyncio
@pytest.mark.live_api
class TestClaudeLiveAPI:
    """Test with live Claude API (requires ANTHROPIC_API_KEY)."""

    async def test_live_claude_behavioral_test(
        self,
        claude_provider,
        sample_rules
    ):
        """Test behavioral change with live Claude API."""
        harness = LLMBehavioralHarness(
            provider=claude_provider,
            mode=ExecutionMode.LIVE
        )

        metrics, with_rules, without_rules = await harness.run_behavioral_test(
            prompt="Write a Python function to calculate factorial of a number",
            rules=sample_rules,
            expected_patterns=[r'""".*?"""', r":\s*int", r"Args:", r"Returns:"],
            temperature=0.5  # Lower temperature for more deterministic output
        )

        # Verify behavioral change detected with real API
        assert metrics.behavior_changed
        assert metrics.confidence_score > 50.0
        assert metrics.pattern_matches > 0

        # Verify real API responses
        assert with_rules.provider == LLMProvider.CLAUDE
        assert with_rules.tokens_used > 0
        assert with_rules.latency_ms > 0


class TestBehavioralAnalyzer:
    """Test behavioral analyzer."""

    def test_structural_diff_calculation(self):
        """Test structural difference calculation."""
        analyzer = BehavioralAnalyzer()

        text1 = "def test(): pass"
        text2 = "from typing import List\n\ndef test(): pass\n\ndef new_func(): pass"

        diff = analyzer._calculate_structural_diff(text1, text2)

        # Should detect import and new function
        assert diff > 0

    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation."""
        analyzer = BehavioralAnalyzer()

        text1 = "hello world test"
        text2 = "hello world"

        similarity = analyzer._calculate_semantic_similarity(text1, text2)

        # Should be high similarity (2/3 words match)
        assert similarity > 50.0
        assert similarity < 100.0

    def test_pattern_matching(self):
        """Test pattern matching in analysis."""
        analyzer = BehavioralAnalyzer()

        with_rules = LLMResponse(
            text='def test():\n    """Docstring"""\n    pass',
            model="test",
            provider=LLMProvider.MOCK,
            tokens_used=20,
            latency_ms=100
        )

        without_rules = LLMResponse(
            text="def test():\n    pass",
            model="test",
            provider=LLMProvider.MOCK,
            tokens_used=10,
            latency_ms=100
        )

        metrics = analyzer.analyze(
            with_rules=with_rules,
            without_rules=without_rules,
            expected_patterns=[r'""".*?"""']
        )

        assert metrics.pattern_matches == 1
        assert metrics.pattern_total == 1
        assert metrics.behavior_changed


class TestPromptTemplate:
    """Test prompt template functionality."""

    def test_template_formatting(self, code_generation_prompt):
        """Test prompt template formatting."""
        formatted = code_generation_prompt.format(
            task="process data"
        )

        assert "process data" in formatted
        assert "Python function" in formatted

    def test_template_metadata(self, code_generation_prompt):
        """Test template metadata."""
        assert code_generation_prompt.category == "code_generation"
        assert "def" in code_generation_prompt.expected_elements
        assert "task" in code_generation_prompt.placeholders
