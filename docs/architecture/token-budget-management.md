# Token Budget Management Architecture

**Version:** 0.2.1dev1
**Date:** 2025-10-04
**Task:** 295.4 - Design token budget management architecture

## Overview

Token budget management ensures memory rules fit within LLM context windows through intelligent prioritization, allocation, and compression strategies.

## Architecture

### Token Budget Manager

**File:** `src/python/common/core/context_injection/token_budget.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
from ...memory import MemoryRule, AuthorityLevel


class AllocationStrategy(Enum):
    """Token allocation strategies."""
    PRIORITY_BASED = "priority"  # Prioritize by authority + priority
    ROUND_ROBIN = "round_robin"  # Distribute equally across categories
    EQUAL = "equal"  # Fixed allocation per rule


class CompressionStrategy(Enum):
    """Rule compression strategies."""
    NONE = "none"  # No compression
    SIMPLE = "simple"  # Remove optional fields
    INTELLIGENT = "intelligent"  # Smart summarization


@dataclass
class BudgetAllocation:
    """Token budget allocation result."""
    total_budget: int
    absolute_tokens: int
    default_tokens: int
    overhead_tokens: int
    rules_included: List[MemoryRule]
    rules_skipped: List[MemoryRule]
    compression_applied: bool


class TokenBudgetManager:
    """Manages token budget allocation and optimization."""

    def __init__(
        self,
        allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED,
        compression_strategy: CompressionStrategy = CompressionStrategy.INTELLIGENT,
        absolute_rules_protected: bool = True,
    ):
        self.allocation_strategy = allocation_strategy
        self.compression_strategy = compression_strategy
        self.absolute_rules_protected = absolute_rules_protected

    def allocate_budget(
        self,
        rules: List[MemoryRule],
        total_budget: int,
        tool_name: str,
    ) -> BudgetAllocation:
        """
        Allocate token budget across rules.

        Algorithm:
        1. Separate absolute vs default rules
        2. Allocate tokens to absolute rules (protected)
        3. Allocate remaining budget to default rules
        4. Apply compression if over budget
        5. Return allocation result
        """
        # Separate by authority
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Count absolute rule tokens
        absolute_tokens = sum(self._estimate_tokens(r, tool_name) for r in absolute_rules)

        # Reserve budget for overhead (headers, footers)
        overhead_tokens = int(total_budget * 0.05)  # 5% overhead
        available_budget = total_budget - overhead_tokens

        # Check if absolute rules fit
        if absolute_tokens > available_budget:
            if self.absolute_rules_protected:
                # Include all absolute rules even if over budget
                return BudgetAllocation(
                    total_budget=total_budget,
                    absolute_tokens=absolute_tokens,
                    default_tokens=0,
                    overhead_tokens=overhead_tokens,
                    rules_included=absolute_rules,
                    rules_skipped=default_rules,
                    compression_applied=False,
                )
            else:
                # Truncate absolute rules to fit budget
                absolute_rules = self._truncate_rules(
                    absolute_rules, available_budget, tool_name
                )

        # Allocate remaining budget to default rules
        remaining_budget = available_budget - absolute_tokens
        selected_default_rules = self._allocate_to_default_rules(
            default_rules, remaining_budget, tool_name
        )

        all_included = absolute_rules + selected_default_rules
        all_skipped = [r for r in default_rules if r not in selected_default_rules]

        return BudgetAllocation(
            total_budget=total_budget,
            absolute_tokens=absolute_tokens,
            default_tokens=sum(self._estimate_tokens(r, tool_name) for r in selected_default_rules),
            overhead_tokens=overhead_tokens,
            rules_included=all_included,
            rules_skipped=all_skipped,
            compression_applied=False,
        )

    def _allocate_to_default_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """Allocate budget to default rules based on strategy."""
        if self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            return self._priority_based_allocation(rules, budget, tool_name)
        elif self.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            return self._round_robin_allocation(rules, budget, tool_name)
        else:  # EQUAL
            return self._equal_allocation(rules, budget, tool_name)

    def _priority_based_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """Allocate by priority (highest priority first)."""
        sorted_rules = sorted(
            rules,
            key=lambda r: (
                getattr(r.metadata, 'priority', 50) if r.metadata else 50,
                -r.created_at.timestamp()
            ),
            reverse=True
        )

        selected = []
        used_tokens = 0

        for rule in sorted_rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if used_tokens + rule_tokens <= budget:
                selected.append(rule)
                used_tokens += rule_tokens
            else:
                break

        return selected

    def _round_robin_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """Distribute tokens equally across categories."""
        from collections import defaultdict
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        budget_per_category = budget // len(by_category) if by_category else 0
        selected = []

        for category_rules in by_category.values():
            category_selected = self._priority_based_allocation(
                category_rules, budget_per_category, tool_name
            )
            selected.extend(category_selected)

        return selected

    def _equal_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """Fixed token allocation per rule."""
        tokens_per_rule = budget // len(rules) if rules else 0
        selected = []

        for rule in rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if rule_tokens <= tokens_per_rule:
                selected.append(rule)

        return selected

    def _truncate_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """Truncate rules to fit within budget."""
        return self._priority_based_allocation(rules, budget, tool_name)

    def _estimate_tokens(self, rule: MemoryRule, tool_name: str) -> int:
        """Estimate token count for a rule (tool-specific)."""
        # Simple estimation: rule text + metadata
        text = rule.rule
        if rule.scope:
            text += " " + " ".join(rule.scope)

        words = len(text.split())
        punctuation = len([c for c in text if c in '.,!?;:()[]{}"\'-'])

        # Tool-specific multipliers
        multipliers = {
            "claude": 1.3,
            "codex": 1.2,
            "gemini": 1.3,
        }
        multiplier = multipliers.get(tool_name, 1.3)

        return int(words * multiplier + punctuation * 0.5)
```

## Token Counting Algorithms

### Tool-Specific Token Counters

```python
class TokenCounter:
    """Tool-specific token counting."""

    @staticmethod
    def count_claude_tokens(text: str) -> int:
        """Estimate tokens for Claude (approximation)."""
        # Claude uses roughly 1 token per 4 characters
        return len(text) // 4

    @staticmethod
    def count_codex_tokens(text: str) -> int:
        """Estimate tokens for Codex (GPT-based)."""
        # GPT tokenizer approximation
        words = len(text.split())
        return int(words * 1.3)

    @staticmethod
    def count_gemini_tokens(text: str) -> int:
        """Estimate tokens for Gemini."""
        # Similar to Claude
        return len(text) // 4
```

## Budget Allocation Strategies

### Priority-Based Allocation

1. Sort rules by priority (authority > priority > recency)
2. Include rules sequentially until budget exhausted
3. Protect absolute rules (never skip)

### Round-Robin Allocation

1. Distribute budget equally across categories
2. Within each category, use priority-based allocation
3. Ensures representation from all categories

### Equal Allocation

1. Fixed token budget per rule
2. Include rules that fit within per-rule budget
3. Simpler but less flexible

## Compression Strategies

### Simple Compression

- Remove optional fields (source, conditions, metadata)
- Keep only essential rule text
- 10-20% token reduction

### Intelligent Compression

- Summarize verbose rules
- Remove redundant information
- Combine similar rules
- 30-50% token reduction

## Implementation Plan

### Phase 1: Core Budget Manager

**Files:**
- `src/python/common/core/context_injection/token_budget.py`

**Tasks:**
- [ ] Implement TokenBudgetManager class
- [ ] Implement allocation strategies
- [ ] Implement token estimation

### Phase 2: Tool-Specific Counters

**Tasks:**
- [ ] Add Claude token counter
- [ ] Add Codex token counter
- [ ] Add Gemini token counter

### Phase 3: Compression

**Tasks:**
- [ ] Implement simple compression
- [ ] Implement intelligent compression

### Phase 4: Testing

**Files:**
- `tests/unit/test_token_budget.py`

**Tasks:**
- [ ] Test allocation strategies
- [ ] Test budget overflow handling
- [ ] Test compression accuracy

## References

- **llm-formatting-strategies.md**: Formatting design (dependency)
- **rule-fetching-mechanism.md**: Rule retrieval
