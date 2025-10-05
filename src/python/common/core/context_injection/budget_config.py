"""
Per-tool token budget configuration management.

This module provides flexible budget configuration with hierarchical inheritance,
runtime adjustment, and persistent storage integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Any

import yaml
from loguru import logger

from .token_usage_tracker import OperationType


class BudgetScope(Enum):
    """Budget configuration scope levels."""

    GLOBAL = "global"  # System-wide default budget
    TOOL = "tool"  # Tool-specific budget (claude, codex, gemini)
    OPERATION = "operation"  # Operation-specific budget (context_injection, user_query, etc.)


@dataclass
class BudgetConfig:
    """
    Token budget configuration for a specific scope.

    Attributes:
        scope: Configuration scope level
        name: Scope identifier (tool name, operation type, or "global")
        default_budget: Default token budget for this scope
        max_budget: Maximum allowed budget (hard limit)
        min_budget: Minimum allowed budget (safety floor)
        enabled: Whether budget enforcement is enabled
        operation_budgets: Operation-specific budgets (TOOL scope only)
        metadata: Additional configuration metadata
    """

    scope: BudgetScope
    name: str
    default_budget: int
    max_budget: Optional[int] = None
    min_budget: int = 1000
    enabled: bool = True
    operation_budgets: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate budget configuration."""
        # Check max_budget >= min_budget first (if max_budget is set)
        if self.max_budget is not None and self.max_budget < self.min_budget:
            raise ValueError(
                f"max_budget ({self.max_budget}) must be >= "
                f"min_budget ({self.min_budget})"
            )

        # Then check default_budget >= min_budget
        if self.default_budget < self.min_budget:
            raise ValueError(
                f"default_budget ({self.default_budget}) must be >= "
                f"min_budget ({self.min_budget})"
            )

        # Finally check default_budget <= max_budget (if max_budget is set)
        if self.max_budget is not None and self.default_budget > self.max_budget:
            raise ValueError(
                f"default_budget ({self.default_budget}) must be <= "
                f"max_budget ({self.max_budget})"
            )


class BudgetConfigManager:
    """
    Hierarchical token budget configuration manager.

    Provides flexible budget configuration with:
    - Hierarchical inheritance: GLOBAL → TOOL → OPERATION
    - Runtime budget adjustment
    - Persistent storage integration
    - Validation and sensible defaults

    Configuration hierarchy:
    1. GLOBAL: System-wide default budget
    2. TOOL: Tool-specific budgets (override global)
    3. OPERATION: Operation-specific budgets (override tool)

    Usage:
        manager = BudgetConfigManager()
        manager.load_from_file("config.yaml")

        # Get budget with hierarchical resolution
        budget = manager.get_budget("claude", OperationType.CONTEXT_INJECTION)

        # Runtime adjustment
        manager.set_budget("claude", OperationType.USER_QUERY, 60000)
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize budget configuration manager.

        Args:
            config_file: Optional YAML configuration file path
        """
        self._lock = Lock()
        self._global_config: Optional[BudgetConfig] = None
        self._tool_configs: Dict[str, BudgetConfig] = {}
        self._operation_configs: Dict[str, BudgetConfig] = {}

        # Initialize with sensible defaults
        self._initialize_defaults()

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

        logger.debug("Initialized BudgetConfigManager")

    def _initialize_defaults(self) -> None:
        """Initialize with sensible default configurations."""
        # Global default budget
        self._global_config = BudgetConfig(
            scope=BudgetScope.GLOBAL,
            name="global",
            default_budget=50000,
            max_budget=200000,
            min_budget=1000,
            enabled=True,
            metadata={"description": "Global default token budget"},
        )

        # Tool-specific defaults
        tool_defaults = {
            "claude": {
                "default_budget": 100000,
                "max_budget": 200000,
                "operations": {
                    OperationType.CONTEXT_INJECTION.value: 50000,
                    OperationType.USER_QUERY.value: 50000,
                },
                "description": "Claude Code tool budget",
            },
            "codex": {
                "default_budget": 4000,
                "max_budget": 8000,
                "operations": {
                    OperationType.CONTEXT_INJECTION.value: 2000,
                    OperationType.USER_QUERY.value: 2000,
                },
                "description": "GitHub Codex tool budget",
            },
            "gemini": {
                "default_budget": 32000,
                "max_budget": 64000,
                "operations": {
                    OperationType.CONTEXT_INJECTION.value: 16000,
                    OperationType.USER_QUERY.value: 16000,
                },
                "description": "Google Gemini tool budget",
            },
        }

        for tool_name, config in tool_defaults.items():
            self._tool_configs[tool_name] = BudgetConfig(
                scope=BudgetScope.TOOL,
                name=tool_name,
                default_budget=config["default_budget"],
                max_budget=config.get("max_budget"),
                min_budget=1000,
                enabled=True,
                operation_budgets=config.get("operations", {}),
                metadata={"description": config.get("description", "")},
            )

        logger.debug(
            f"Initialized default budgets for {len(self._tool_configs)} tools"
        )

    def get_budget(
        self,
        tool_name: Optional[str] = None,
        operation_type: Optional[OperationType] = None,
    ) -> int:
        """
        Get token budget with hierarchical resolution.

        Resolution order:
        1. Operation-specific budget (if operation_type provided)
        2. Tool-specific budget (if tool_name provided)
        3. Global default budget

        Args:
            tool_name: Tool identifier (e.g., "claude", "codex")
            operation_type: Operation type for operation-specific budget

        Returns:
            Resolved token budget
        """
        with self._lock:
            # Try operation-specific budget first
            if tool_name and operation_type:
                tool_config = self._tool_configs.get(tool_name)
                if tool_config and operation_type.value in tool_config.operation_budgets:
                    budget = tool_config.operation_budgets[operation_type.value]
                    logger.debug(
                        f"Using operation budget for {tool_name}.{operation_type.value}: {budget}"
                    )
                    return budget

            # Try tool-specific budget
            if tool_name:
                tool_config = self._tool_configs.get(tool_name)
                if tool_config and tool_config.enabled:
                    logger.debug(
                        f"Using tool budget for {tool_name}: {tool_config.default_budget}"
                    )
                    return tool_config.default_budget

            # Fall back to global budget
            budget = (
                self._global_config.default_budget if self._global_config else 50000
            )
            logger.debug(f"Using global budget: {budget}")
            return budget

    def set_budget(
        self,
        tool_name: Optional[str] = None,
        operation_type: Optional[OperationType] = None,
        budget: int = 50000,
    ) -> None:
        """
        Set token budget at runtime.

        Validates budget against min/max limits before setting.

        Args:
            tool_name: Tool identifier (None for global budget)
            operation_type: Operation type (None for tool-level budget)
            budget: Token budget to set

        Raises:
            ValueError: If budget is invalid or out of range
        """
        with self._lock:
            # Validate budget is positive
            if budget < 1000:
                raise ValueError(f"Budget must be >= 1000, got {budget}")

            # Global budget update
            if tool_name is None and operation_type is None:
                if self._global_config:
                    if (
                        self._global_config.max_budget
                        and budget > self._global_config.max_budget
                    ):
                        raise ValueError(
                            f"Budget {budget} exceeds global max {self._global_config.max_budget}"
                        )
                    self._global_config.default_budget = budget
                    logger.info(f"Updated global budget to {budget}")
                return

            # Tool-specific budget update
            if tool_name and operation_type is None:
                if tool_name not in self._tool_configs:
                    # Create new tool config
                    self._tool_configs[tool_name] = BudgetConfig(
                        scope=BudgetScope.TOOL,
                        name=tool_name,
                        default_budget=budget,
                        min_budget=1000,
                        enabled=True,
                    )
                    logger.info(f"Created new tool budget for {tool_name}: {budget}")
                else:
                    tool_config = self._tool_configs[tool_name]
                    if tool_config.max_budget and budget > tool_config.max_budget:
                        raise ValueError(
                            f"Budget {budget} exceeds tool max {tool_config.max_budget}"
                        )
                    tool_config.default_budget = budget
                    logger.info(f"Updated tool budget for {tool_name}: {budget}")
                return

            # Operation-specific budget update
            if tool_name and operation_type:
                if tool_name not in self._tool_configs:
                    # Create tool config with operation budget
                    self._tool_configs[tool_name] = BudgetConfig(
                        scope=BudgetScope.TOOL,
                        name=tool_name,
                        default_budget=budget,
                        min_budget=1000,
                        enabled=True,
                        operation_budgets={operation_type.value: budget},
                    )
                    logger.info(
                        f"Created tool {tool_name} with operation budget "
                        f"{operation_type.value}: {budget}"
                    )
                else:
                    tool_config = self._tool_configs[tool_name]
                    tool_config.operation_budgets[operation_type.value] = budget
                    logger.info(
                        f"Updated operation budget for {tool_name}.{operation_type.value}: {budget}"
                    )

    def get_all_budgets(self) -> Dict[str, Any]:
        """
        Get complete budget configuration hierarchy.

        Returns:
            Dictionary with global, tool, and operation budgets
        """
        with self._lock:
            result: Dict[str, Any] = {}

            # Global budget
            if self._global_config:
                result["global"] = {
                    "default": self._global_config.default_budget,
                    "max": self._global_config.max_budget,
                    "min": self._global_config.min_budget,
                    "enabled": self._global_config.enabled,
                }

            # Tool budgets
            result["tools"] = {}
            for tool_name, tool_config in self._tool_configs.items():
                result["tools"][tool_name] = {
                    "default": tool_config.default_budget,
                    "max": tool_config.max_budget,
                    "min": tool_config.min_budget,
                    "enabled": tool_config.enabled,
                    "operations": dict(tool_config.operation_budgets),
                }

            return result

    def load_from_file(self, config_file: Path) -> None:
        """
        Load budget configuration from YAML file.

        Expected format:
        ```yaml
        token_budgets:
          global:
            default: 50000
            max: 200000

          tools:
            claude:
              default: 100000
              operations:
                context_injection: 50000
                user_query: 50000
        ```

        Args:
            config_file: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data or "token_budgets" not in config_data:
                logger.warning("No token_budgets section in config file")
                return

            budgets = config_data["token_budgets"]

            with self._lock:
                # Load global budget
                if "global" in budgets:
                    global_data = budgets["global"]
                    self._global_config = BudgetConfig(
                        scope=BudgetScope.GLOBAL,
                        name="global",
                        default_budget=global_data.get("default", 50000),
                        max_budget=global_data.get("max"),
                        min_budget=global_data.get("min", 1000),
                        enabled=global_data.get("enabled", True),
                    )

                # Load tool budgets
                if "tools" in budgets:
                    for tool_name, tool_data in budgets["tools"].items():
                        operations = tool_data.get("operations", {})
                        self._tool_configs[tool_name] = BudgetConfig(
                            scope=BudgetScope.TOOL,
                            name=tool_name,
                            default_budget=tool_data.get("default", 50000),
                            max_budget=tool_data.get("max"),
                            min_budget=tool_data.get("min", 1000),
                            enabled=tool_data.get("enabled", True),
                            operation_budgets=operations,
                        )

            logger.info(f"Loaded budget configuration from {config_file}")

        except Exception as e:
            logger.error(f"Error loading budget config from {config_file}: {e}")
            raise

    def save_to_file(self, config_file: Path) -> None:
        """
        Save current budget configuration to YAML file.

        Args:
            config_file: Path to YAML configuration file

        Raises:
            IOError: If file cannot be written
        """
        with self._lock:
            config_data: Dict[str, Any] = {"token_budgets": {}}

            # Save global budget
            if self._global_config:
                config_data["token_budgets"]["global"] = {
                    "default": self._global_config.default_budget,
                    "max": self._global_config.max_budget,
                    "min": self._global_config.min_budget,
                    "enabled": self._global_config.enabled,
                }

            # Save tool budgets
            config_data["token_budgets"]["tools"] = {}
            for tool_name, tool_config in self._tool_configs.items():
                config_data["token_budgets"]["tools"][tool_name] = {
                    "default": tool_config.default_budget,
                    "max": tool_config.max_budget,
                    "min": tool_config.min_budget,
                    "enabled": tool_config.enabled,
                    "operations": dict(tool_config.operation_budgets),
                }

        # Write to file
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Saved budget configuration to {config_file}")
        except Exception as e:
            logger.error(f"Error saving budget config to {config_file}: {e}")
            raise

    def validate_budget(self, budget: int, tool_name: Optional[str] = None) -> bool:
        """
        Validate budget against configured limits.

        Args:
            budget: Budget to validate
            tool_name: Tool name for tool-specific validation (optional)

        Returns:
            True if budget is valid, False otherwise
        """
        with self._lock:
            # Check minimum
            min_budget = 1000
            if self._global_config:
                min_budget = self._global_config.min_budget

            if budget < min_budget:
                logger.warning(
                    f"Budget {budget} below minimum {min_budget}"
                )
                return False

            # Check maximum
            if tool_name:
                tool_config = self._tool_configs.get(tool_name)
                if tool_config and tool_config.max_budget:
                    if budget > tool_config.max_budget:
                        logger.warning(
                            f"Budget {budget} exceeds tool max {tool_config.max_budget}"
                        )
                        return False
            elif self._global_config and self._global_config.max_budget:
                if budget > self._global_config.max_budget:
                    logger.warning(
                        f"Budget {budget} exceeds global max {self._global_config.max_budget}"
                    )
                    return False

            return True

    def get_config(self, tool_name: Optional[str] = None) -> Optional[BudgetConfig]:
        """
        Get budget configuration object.

        Args:
            tool_name: Tool name (None for global config)

        Returns:
            BudgetConfig object or None if not found
        """
        with self._lock:
            if tool_name is None:
                return self._global_config
            return self._tool_configs.get(tool_name)

    def reset_to_defaults(self) -> None:
        """Reset all budget configurations to default values."""
        with self._lock:
            self._global_config = None
            self._tool_configs.clear()
            self._operation_configs.clear()
            self._initialize_defaults()
            logger.info("Reset budget configuration to defaults")
