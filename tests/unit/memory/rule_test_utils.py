"""
Utility functions and classes for memory rules testing.

Provides generators, validators, and simulators for comprehensive
memory rules testing.
"""

import random
import string
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from common.core.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
    AgentDefinition,
    MemoryConflict,
)


class MemoryRuleGenerator:
    """
    Generator for creating test memory rules with various configurations.

    Provides methods to generate rules with specific characteristics
    for comprehensive testing.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the rule generator.

        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)

        self._rule_counter = 0
        self._agent_counter = 0

    def generate_rule(
        self,
        category: Optional[MemoryCategory] = None,
        authority: Optional[AuthorityLevel] = None,
        scope: Optional[List[str]] = None,
        **kwargs: Any
    ) -> MemoryRule:
        """
        Generate a single memory rule.

        Args:
            category: Memory category (random if None)
            authority: Authority level (random if None)
            scope: Scope list (default ["global"] if None)
            **kwargs: Additional MemoryRule fields

        Returns:
            Generated MemoryRule
        """
        self._rule_counter += 1

        if category is None:
            category = random.choice(list(MemoryCategory))

        if authority is None:
            authority = random.choice(list(AuthorityLevel))

        if scope is None:
            scope = ["global"]

        rule_id = kwargs.get("id", f"test-rule-{self._rule_counter}")
        name = kwargs.get("name", f"Test Rule {self._rule_counter}")
        rule_text = kwargs.get("rule", f"Test rule text {self._rule_counter}")

        return MemoryRule(
            id=rule_id,
            category=category,
            name=name,
            rule=rule_text,
            authority=authority,
            scope=scope,
            source=kwargs.get("source", "user_explicit"),
            conditions=kwargs.get("conditions"),
            replaces=kwargs.get("replaces"),
            metadata=kwargs.get("metadata"),
        )

    def generate_rules(
        self,
        count: int,
        category: Optional[MemoryCategory] = None,
        authority: Optional[AuthorityLevel] = None,
    ) -> List[MemoryRule]:
        """
        Generate multiple memory rules.

        Args:
            count: Number of rules to generate
            category: Memory category for all rules (random if None)
            authority: Authority level for all rules (random if None)

        Returns:
            List of generated MemoryRule instances
        """
        return [self.generate_rule(category=category, authority=authority) for _ in range(count)]

    def generate_behavior_rule(
        self,
        authority: AuthorityLevel = AuthorityLevel.DEFAULT,
        **kwargs: Any
    ) -> MemoryRule:
        """
        Generate a behavioral rule.

        Args:
            authority: Authority level
            **kwargs: Additional MemoryRule fields

        Returns:
            Generated behavioral MemoryRule
        """
        return self.generate_rule(
            category=MemoryCategory.BEHAVIOR,
            authority=authority,
            **kwargs
        )

    def generate_preference_rule(
        self,
        scope: Optional[List[str]] = None,
        **kwargs: Any
    ) -> MemoryRule:
        """
        Generate a user preference rule.

        Args:
            scope: Scope list
            **kwargs: Additional MemoryRule fields

        Returns:
            Generated preference MemoryRule
        """
        return self.generate_rule(
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=scope,
            **kwargs
        )

    def generate_agent_rule(self, **kwargs: Any) -> MemoryRule:
        """
        Generate an agent-related rule.

        Args:
            **kwargs: Additional MemoryRule fields

        Returns:
            Generated agent MemoryRule
        """
        return self.generate_rule(
            category=MemoryCategory.AGENT,
            **kwargs
        )

    def generate_agent_definition(
        self,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentDefinition:
        """
        Generate an agent definition.

        Args:
            name: Agent name (generated if None)
            capabilities: Agent capabilities (generated if None)
            **kwargs: Additional AgentDefinition fields

        Returns:
            Generated AgentDefinition
        """
        self._agent_counter += 1

        if name is None:
            name = f"test-agent-{self._agent_counter}"

        if capabilities is None:
            capabilities = [
                "capability_1",
                "capability_2",
                "capability_3",
            ]

        agent_id = kwargs.get("id", f"agent-{name}")
        description = kwargs.get("description", f"Test agent {name}")

        return AgentDefinition(
            id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
            deploy_cost=kwargs.get("deploy_cost", "medium"),
            last_used=kwargs.get("last_used"),
            metadata=kwargs.get("metadata"),
        )

    def generate_scoped_rules(
        self,
        scopes: List[str],
        category: Optional[MemoryCategory] = None,
    ) -> List[MemoryRule]:
        """
        Generate rules with specific scopes.

        Args:
            scopes: List of scopes to create rules for
            category: Memory category (random if None)

        Returns:
            List of rules, one per scope
        """
        return [
            self.generate_rule(category=category, scope=[scope])
            for scope in scopes
        ]

    def generate_temporal_rules(
        self,
        count: int,
        time_span_days: int = 30,
    ) -> List[MemoryRule]:
        """
        Generate rules with timestamps spread over a time period.

        Args:
            count: Number of rules to generate
            time_span_days: Days to spread timestamps over

        Returns:
            List of rules with varied timestamps
        """
        rules = []
        now = datetime.now(timezone.utc)

        for i in range(count):
            # Distribute timestamps evenly across the time span
            days_ago = (i * time_span_days) // count
            created_at = now - timedelta(days=days_ago)
            updated_at = created_at + timedelta(days=random.randint(0, days_ago))

            rule = self.generate_rule(
                created_at=created_at,
                updated_at=updated_at,
            )
            rules.append(rule)

        return rules


class MemoryRuleValidator:
    """
    Validator for memory rules.

    Provides validation methods for rule structure, semantics, and relationships.
    """

    @staticmethod
    def validate_structure(rule: MemoryRule) -> Tuple[bool, List[str]]:
        """
        Validate rule structure and required fields.

        Args:
            rule: MemoryRule to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not rule.id:
            errors.append("Rule must have an ID")

        if not rule.category:
            errors.append("Rule must have a category")

        if not isinstance(rule.category, MemoryCategory):
            errors.append("Category must be MemoryCategory enum")

        if not rule.name:
            errors.append("Rule must have a name")

        if not rule.rule:
            errors.append("Rule must have rule text")

        if not rule.authority:
            errors.append("Rule must have authority level")

        if not isinstance(rule.authority, AuthorityLevel):
            errors.append("Authority must be AuthorityLevel enum")

        if not rule.scope:
            errors.append("Rule must have scope")

        if not isinstance(rule.scope, list):
            errors.append("Scope must be a list")

        if rule.scope and not all(isinstance(s, str) for s in rule.scope):
            errors.append("All scope items must be strings")

        if not rule.created_at:
            errors.append("Rule must have created_at timestamp")

        if not rule.updated_at:
            errors.append("Rule must have updated_at timestamp")

        if rule.created_at and rule.updated_at and rule.updated_at < rule.created_at:
            errors.append("updated_at must be >= created_at")

        return len(errors) == 0, errors

    @staticmethod
    def validate_semantics(rule: MemoryRule) -> Tuple[bool, List[str]]:
        """
        Validate rule semantic correctness.

        Args:
            rule: MemoryRule to validate

        Returns:
            Tuple of (is_valid, warning_messages)
        """
        warnings = []

        # Check for empty rule text
        if rule.rule and len(rule.rule.strip()) < 10:
            warnings.append("Rule text is very short (< 10 characters)")

        # Check for very long rule text
        if rule.rule and len(rule.rule) > 1000:
            warnings.append("Rule text is very long (> 1000 characters)")

        # Check for global scope with specific conditions
        if "global" in rule.scope and rule.conditions:
            warnings.append("Global scope with conditions may be contradictory")

        # Check for absolute authority with conversational source
        if rule.authority == AuthorityLevel.ABSOLUTE and rule.source == "conversational":
            warnings.append("Absolute authority from conversational source is unusual")

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_relationships(rules: List[MemoryRule]) -> Tuple[bool, List[str]]:
        """
        Validate relationships between rules.

        Args:
            rules: List of MemoryRule instances

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for duplicate IDs
        rule_ids = [rule.id for rule in rules]
        duplicate_ids = [rid for rid in rule_ids if rule_ids.count(rid) > 1]
        if duplicate_ids:
            errors.append(f"Duplicate rule IDs: {set(duplicate_ids)}")

        # Check for orphaned replaces references
        all_ids = set(rule_ids)
        for rule in rules:
            if rule.replaces:
                missing = [rid for rid in rule.replaces if rid not in all_ids]
                if missing:
                    errors.append(f"Rule {rule.id} replaces non-existent rules: {missing}")

        return len(errors) == 0, errors


class ConflictSimulator:
    """
    Simulator for generating conflicting memory rules.

    Useful for testing conflict detection and resolution.
    """

    def __init__(self, generator: Optional[MemoryRuleGenerator] = None):
        """
        Initialize the conflict simulator.

        Args:
            generator: Rule generator to use (creates new one if None)
        """
        self.generator = generator or MemoryRuleGenerator()

    def generate_contradictory_pair(
        self,
        scope: Optional[List[str]] = None,
    ) -> Tuple[MemoryRule, MemoryRule]:
        """
        Generate a pair of directly contradictory rules.

        Args:
            scope: Scope for both rules (default ["global"])

        Returns:
            Tuple of conflicting MemoryRule instances
        """
        if scope is None:
            scope = ["global"]

        rule1 = self.generator.generate_rule(
            name="Always Use Feature X",
            rule="Always use feature X for all implementations",
            authority=AuthorityLevel.DEFAULT,
            scope=scope,
        )

        rule2 = self.generator.generate_rule(
            name="Never Use Feature X",
            rule="Never use feature X, always use feature Y instead",
            authority=AuthorityLevel.DEFAULT,
            scope=scope,
        )

        return rule1, rule2

    def generate_authority_conflict(
        self,
        scope: Optional[List[str]] = None,
    ) -> Tuple[MemoryRule, MemoryRule]:
        """
        Generate rules with conflicting authority levels.

        Args:
            scope: Scope for both rules (default ["global"])

        Returns:
            Tuple of rules with authority conflict
        """
        if scope is None:
            scope = ["global"]

        rule1 = self.generator.generate_rule(
            name="Absolute Rule",
            rule="Must always do X",
            authority=AuthorityLevel.ABSOLUTE,
            scope=scope,
        )

        rule2 = self.generator.generate_rule(
            name="Contradicting Default Rule",
            rule="Prefer to do Y instead of X",
            authority=AuthorityLevel.DEFAULT,
            scope=scope,
        )

        return rule1, rule2

    def generate_scope_overlap(
        self,
        scope1: List[str],
        scope2: List[str],
    ) -> Tuple[MemoryRule, MemoryRule]:
        """
        Generate rules with overlapping but different scopes.

        Args:
            scope1: Scope for first rule
            scope2: Scope for second rule

        Returns:
            Tuple of rules with overlapping scopes
        """
        rule1 = self.generator.generate_rule(
            name="Rule for Scope 1",
            rule="Apply rule in scope 1",
            scope=scope1,
        )

        rule2 = self.generator.generate_rule(
            name="Rule for Scope 2",
            rule="Apply different rule in scope 2",
            scope=scope2,
        )

        return rule1, rule2


# Assertion helpers
def assert_rule_has_field(rule: MemoryRule, field: str) -> None:
    """
    Assert that a rule has a specific field.

    Args:
        rule: MemoryRule to check
        field: Field name

    Raises:
        AssertionError: If field is missing
    """
    assert hasattr(rule, field), f"Rule must have field: {field}"
    assert getattr(rule, field) is not None, f"Rule field {field} must not be None"


def assert_rules_have_same_scope(rule1: MemoryRule, rule2: MemoryRule) -> None:
    """
    Assert that two rules have the same scope.

    Args:
        rule1: First rule
        rule2: Second rule

    Raises:
        AssertionError: If scopes differ
    """
    assert set(rule1.scope) == set(rule2.scope), f"Rules must have same scope: {rule1.scope} != {rule2.scope}"


def assert_rule_replaces(rule: MemoryRule, replaced_id: str) -> None:
    """
    Assert that a rule replaces a specific rule ID.

    Args:
        rule: MemoryRule to check
        replaced_id: Expected replaced rule ID

    Raises:
        AssertionError: If rule doesn't replace the specified ID
    """
    assert rule.replaces is not None, "Rule must have replaces field"
    assert replaced_id in rule.replaces, f"Rule must replace {replaced_id}"
