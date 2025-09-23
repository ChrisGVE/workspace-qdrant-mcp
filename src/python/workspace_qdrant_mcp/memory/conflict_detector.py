"""
Conflict detection for memory rules using semantic analysis.

This module uses Claude/Sonnet models to detect semantic conflicts between memory rules
and provide intelligent conflict resolution suggestions.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .types import AuthorityLevel, MemoryRule, MemoryRuleConflict

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detects conflicts between memory rules using semantic analysis.

    Uses Claude/Sonnet models to identify:
    - Direct contradictions ("Use X" vs "Don't use X")
    - Semantic conflicts ("Prefer Y" vs "Avoid Y")
    - Authority conflicts (multiple absolute rules for same domain)
    - Scope overlaps that might cause issues
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        model: str = "claude-3-sonnet-20240229",
        enable_ai_analysis: bool = True,
    ):
        """
        Initialize conflict detector.

        Args:
            anthropic_api_key: API key for Claude models (optional - will use env var)
            model: Claude model to use for analysis
            enable_ai_analysis: Whether to enable AI-powered semantic analysis
        """
        self.model = model
        self.enable_ai_analysis = enable_ai_analysis
        self.anthropic_client = None

        if enable_ai_analysis:
            try:
                import anthropic

                api_key = anthropic_api_key or self._get_api_key_from_env()
                if api_key:
                    self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Initialized Claude conflict detection")
                else:
                    logger.warning(
                        "No Anthropic API key found - disabling AI conflict detection"
                    )
                    self.enable_ai_analysis = False
            except ImportError:
                logger.warning(
                    "Anthropic library not installed - disabling AI conflict detection"
                )
                self.enable_ai_analysis = False

    async def detect_conflicts(
        self, new_rule: MemoryRule, existing_rules: list[MemoryRule]
    ) -> list[MemoryRuleConflict]:
        """
        Detect conflicts between a new rule and existing rules.

        Args:
            new_rule: The rule being added
            existing_rules: List of existing rules to check against

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # First run rule-based detection (fast)
        for existing_rule in existing_rules:
            rule_conflicts = await self._detect_rule_based_conflicts(
                new_rule, existing_rule
            )
            conflicts.extend(rule_conflicts)

        # Then run AI-based semantic analysis if enabled (slower but more accurate)
        if self.enable_ai_analysis and self.anthropic_client:
            semantic_conflicts = await self._detect_semantic_conflicts(
                new_rule, existing_rules
            )

            # Merge conflicts, avoiding duplicates
            for semantic_conflict in semantic_conflicts:
                # Check if we already found this conflict pair
                existing_pair = any(
                    (
                        c.rule1.id == semantic_conflict.rule1.id
                        and c.rule2.id == semantic_conflict.rule2.id
                    )
                    or (
                        c.rule1.id == semantic_conflict.rule2.id
                        and c.rule2.id == semantic_conflict.rule1.id
                    )
                    for c in conflicts
                )

                if not existing_pair:
                    conflicts.append(semantic_conflict)
                else:
                    # Enhance existing conflict with semantic information
                    for existing_conflict in conflicts:
                        if (
                            existing_conflict.rule1.id == semantic_conflict.rule1.id
                            and existing_conflict.rule2.id == semantic_conflict.rule2.id
                        ) or (
                            existing_conflict.rule1.id == semantic_conflict.rule2.id
                            and existing_conflict.rule2.id == semantic_conflict.rule1.id
                        ):
                            # Enhance with semantic analysis if confidence is higher
                            if (
                                semantic_conflict.confidence
                                > existing_conflict.confidence
                            ):
                                existing_conflict.confidence = (
                                    semantic_conflict.confidence
                                )
                                existing_conflict.description = (
                                    semantic_conflict.description
                                )
                                if semantic_conflict.resolution_suggestion:
                                    existing_conflict.resolution_suggestion = (
                                        semantic_conflict.resolution_suggestion
                                    )
                            break

        # Sort by severity and confidence
        conflicts.sort(
            key=lambda c: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[c.severity],
                c.confidence,
            ),
            reverse=True,
        )

        return conflicts

    async def analyze_all_conflicts(
        self, rules: list[MemoryRule]
    ) -> list[MemoryRuleConflict]:
        """
        Analyze all possible conflicts among a set of rules.

        Args:
            rules: List of all rules to analyze

        Returns:
            List of all detected conflicts
        """
        all_conflicts = []

        for i, rule1 in enumerate(rules):
            for rule2 in rules[i + 1 :]:  # Avoid duplicate comparisons
                conflicts = await self.detect_conflicts(rule1, [rule2])
                all_conflicts.extend(conflicts)

        return all_conflicts

    async def _detect_rule_based_conflicts(
        self, rule1: MemoryRule, rule2: MemoryRule
    ) -> list[MemoryRuleConflict]:
        """
        Detect conflicts using rule-based analysis (fast, deterministic).

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Authority conflicts - multiple absolute rules in same scope
        if (
            rule1.authority == AuthorityLevel.ABSOLUTE
            and rule2.authority == AuthorityLevel.ABSOLUTE
        ):
            # Check for scope overlap
            scope_overlap = self._check_scope_overlap(rule1.scope, rule2.scope)
            if scope_overlap:
                conflicts.append(
                    MemoryRuleConflict(
                        rule1=rule1,
                        rule2=rule2,
                        conflict_type="authority",
                        confidence=0.9,
                        description=f"Both rules have absolute authority and overlapping scope: {scope_overlap}",
                        severity="high",
                        resolution_suggestion="Consider making one rule default authority or refining scopes",
                    )
                )

        # Direct text contradictions (simple keyword detection)
        text_conflict = self._detect_text_contradictions(rule1.rule, rule2.rule)
        if text_conflict:
            conflicts.append(
                MemoryRuleConflict(
                    rule1=rule1,
                    rule2=rule2,
                    conflict_type="direct",
                    confidence=text_conflict["confidence"],
                    description=text_conflict["description"],
                    severity="medium",
                    resolution_suggestion="Review rules for contradictory instructions",
                )
            )

        return conflicts

    async def _detect_semantic_conflicts(
        self, new_rule: MemoryRule, existing_rules: list[MemoryRule]
    ) -> list[MemoryRuleConflict]:
        """
        Detect conflicts using Claude semantic analysis.

        Args:
            new_rule: The new rule to check
            existing_rules: Existing rules to check against

        Returns:
            List of semantic conflicts
        """
        if not self.anthropic_client:
            return []

        conflicts = []

        # Analyze in batches to manage API costs and rate limits
        batch_size = 5
        for i in range(0, len(existing_rules), batch_size):
            batch = existing_rules[i : i + batch_size]
            batch_conflicts = await self._analyze_rule_batch(new_rule, batch)
            conflicts.extend(batch_conflicts)

            # Add small delay to avoid rate limits
            if len(existing_rules) > batch_size:
                await asyncio.sleep(0.1)

        return conflicts

    async def _analyze_rule_batch(
        self, new_rule: MemoryRule, rule_batch: list[MemoryRule]
    ) -> list[MemoryRuleConflict]:
        """
        Analyze conflicts for a batch of rules using Claude.

        Args:
            new_rule: New rule to check
            rule_batch: Batch of existing rules

        Returns:
            List of conflicts found in this batch
        """
        try:
            # Prepare rules for analysis
            rules_text = []
            rules_text.append(
                f"NEW RULE: {new_rule.rule} (Authority: {new_rule.authority.value}, Category: {new_rule.category.value})"
            )

            for i, rule in enumerate(rule_batch):
                rules_text.append(
                    f"EXISTING RULE {i + 1}: {rule.rule} (Authority: {rule.authority.value}, Category: {rule.category.value})"
                )

            rules_context = "\n".join(rules_text)

            # Claude analysis prompt
            prompt = f"""
You are analyzing memory rules for an AI assistant to detect conflicts. Please analyze the following rules for conflicts:

{rules_context}

Look for:
1. Direct contradictions (opposite instructions)
2. Semantic conflicts (conflicting preferences or approaches)
3. Authority conflicts (conflicting absolute rules)
4. Scope conflicts (overlapping domains with different guidance)

For each conflict you find, provide:
- Which rules conflict (use "NEW RULE" and "EXISTING RULE X")
- Conflict type (direct, semantic, authority, scope)
- Confidence level (0.0 to 1.0)
- Brief description of the conflict
- Severity (low, medium, high, critical)
- Resolution suggestion (optional)

Respond in JSON format:
{{
    "conflicts": [
        {{
            "rule1": "NEW RULE",
            "rule2": "EXISTING RULE 1",
            "conflict_type": "semantic",
            "confidence": 0.8,
            "description": "Brief conflict description",
            "severity": "medium",
            "resolution_suggestion": "Optional suggestion"
        }}
    ]
}}

If no conflicts are found, return {{"conflicts": []}}.
"""

            # Make API call to Claude
            response = await self._call_claude_api(prompt)

            if not response:
                return []

            # Parse response
            try:
                result = json.loads(response)
                conflicts = []

                for conflict_data in result.get("conflicts", []):
                    # Map rule references back to actual rules
                    rule1 = new_rule if conflict_data["rule1"] == "NEW RULE" else None
                    rule2 = None

                    if conflict_data["rule2"].startswith("EXISTING RULE"):
                        try:
                            rule_index = int(conflict_data["rule2"].split()[-1]) - 1
                            if 0 <= rule_index < len(rule_batch):
                                rule2 = rule_batch[rule_index]
                        except (ValueError, IndexError):
                            continue

                    if rule1 and rule2:
                        conflicts.append(
                            MemoryRuleConflict(
                                rule1=rule1,
                                rule2=rule2,
                                conflict_type=conflict_data.get(
                                    "conflict_type", "semantic"
                                ),
                                confidence=float(conflict_data.get("confidence", 0.5)),
                                description=conflict_data.get(
                                    "description", "Semantic conflict detected"
                                ),
                                severity=conflict_data.get("severity", "medium"),
                                resolution_suggestion=conflict_data.get(
                                    "resolution_suggestion"
                                ),
                            )
                        )

                return conflicts

            except json.JSONDecodeError:
                logger.error("Failed to parse Claude response as JSON")
                return []

        except Exception as e:
            logger.error(f"Error in semantic conflict analysis: {e}")
            return []

    async def _call_claude_api(self, prompt: str) -> str | None:
        """
        Make API call to Claude.

        Args:
            prompt: The analysis prompt

        Returns:
            Claude's response text or None on error
        """
        try:
            response = await self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent analysis
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None

    def _check_scope_overlap(self, scope1: list[str], scope2: list[str]) -> list[str]:
        """
        Check for overlapping scopes between two rules.

        Args:
            scope1: Scope list for first rule
            scope2: Scope list for second rule

        Returns:
            List of overlapping scope items
        """
        if not scope1 or not scope2:  # Empty scope means global
            return ["global"]

        return list(set(scope1) & set(scope2))

    def _detect_text_contradictions(
        self, text1: str, text2: str
    ) -> dict[str, Any] | None:
        """
        Detect direct text contradictions using keyword analysis.

        Args:
            text1: First rule text
            text2: Second rule text

        Returns:
            Conflict information dict or None
        """
        # Normalize text
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Look for direct contradictions
        contradiction_patterns = [
            (r"use\s+(\w+)", r"don\'?t\s+use\s+\1|avoid\s+\1|never\s+use\s+\1"),
            (r"prefer\s+(\w+)", r"don\'?t\s+prefer\s+\1|avoid\s+\1"),
            (r"always\s+(\w+)", r"never\s+\1|don\'?t\s+\1"),
            (r"enable\s+(\w+)", r"disable\s+\1|turn\s+off\s+\1"),
        ]

        for positive_pattern, negative_pattern in contradiction_patterns:
            positive_matches = re.findall(positive_pattern, text1_lower)
            for match in positive_matches:
                if re.search(negative_pattern.replace(r"\1", match), text2_lower):
                    return {
                        "confidence": 0.8,
                        "description": f"Direct contradiction: '{match}' usage conflict",
                    }

            # Check reverse direction
            positive_matches = re.findall(positive_pattern, text2_lower)
            for match in positive_matches:
                if re.search(negative_pattern.replace(r"\1", match), text1_lower):
                    return {
                        "confidence": 0.8,
                        "description": f"Direct contradiction: '{match}' usage conflict",
                    }

        return None

    def _get_api_key_from_env(self) -> str | None:
        """Get Anthropic API key from environment."""
        import os

        return os.getenv("ANTHROPIC_API_KEY")

    def get_conflict_summary(
        self, conflicts: list[MemoryRuleConflict]
    ) -> dict[str, Any]:
        """
        Generate a summary of conflicts.

        Args:
            conflicts: List of conflicts

        Returns:
            Summary statistics
        """
        if not conflicts:
            return {
                "total": 0,
                "by_severity": {},
                "by_type": {},
                "highest_confidence": 0.0,
                "recommendations": [],
            }

        by_severity = {}
        by_type = {}

        for conflict in conflicts:
            # Count by severity
            by_severity[conflict.severity] = by_severity.get(conflict.severity, 0) + 1

            # Count by type
            by_type[conflict.conflict_type] = by_type.get(conflict.conflict_type, 0) + 1

        # Generate recommendations
        recommendations = []
        critical_count = by_severity.get("critical", 0)
        high_count = by_severity.get("high", 0)

        if critical_count > 0:
            recommendations.append(
                f"Resolve {critical_count} critical conflicts before proceeding"
            )
        if high_count > 0:
            recommendations.append(f"Review {high_count} high-severity conflicts")
        if by_type.get("authority", 0) > 0:
            recommendations.append("Consider refining rule authority levels and scopes")

        return {
            "total": len(conflicts),
            "by_severity": by_severity,
            "by_type": by_type,
            "highest_confidence": max(c.confidence for c in conflicts),
            "recommendations": recommendations,
        }
