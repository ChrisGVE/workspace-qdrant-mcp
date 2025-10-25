"""
Manual override configuration for LLM tool detection.

This module provides a configuration system that allows users to manually override
the automatic LLM tool detection. This is useful when:
1. Auto-detection fails or is unreliable
2. User wants to test a specific formatter
3. User prefers a specific tool even when another is detected

The override system supports two storage mechanisms:
1. Environment variable: LLM_TOOL_OVERRIDE
2. Configuration file: ~/.wqm/llm_override.json

Priority: environment variable > config file > no override
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from .llm_tool_detector import LLMToolType


@dataclass
class LLMOverrideConfig:
    """
    Configuration for manual LLM tool override.

    Attributes:
        enabled: Whether the override is currently active
        tool_type: Which LLM tool to force (None = disabled)
        reason: Optional reason why the override was set
        set_at: When the override was configured
        set_by: Who/what set the override (e.g., "user", "cli", "api")
    """

    enabled: bool = False
    tool_type: LLMToolType | None = None
    reason: str | None = None
    set_at: str | None = None  # ISO format datetime string
    set_by: str | None = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with tool_type as string
        """
        data = asdict(self)
        if self.tool_type is not None:
            data["tool_type"] = self.tool_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LLMOverrideConfig":
        """
        Create from dictionary (from JSON).

        Args:
            data: Dictionary with configuration data

        Returns:
            LLMOverrideConfig instance

        Raises:
            ValueError: If tool_type is invalid
        """
        # Convert tool_type string to enum
        tool_type_str = data.get("tool_type")
        tool_type = None
        if tool_type_str:
            try:
                tool_type = LLMToolType(tool_type_str)
            except ValueError:
                raise ValueError(
                    f"Invalid tool_type: {tool_type_str}. "
                    f"Valid options: {', '.join(t.value for t in LLMToolType)}"
                )

        return cls(
            enabled=data.get("enabled", False),
            tool_type=tool_type,
            reason=data.get("reason"),
            set_at=data.get("set_at"),
            set_by=data.get("set_by"),
        )


class LLMOverrideManager:
    """
    Manager for LLM tool override configuration.

    Handles reading/writing override configuration from multiple sources
    with defined priority: environment variable > config file > no override.

    The manager supports:
    - Environment variable: LLM_TOOL_OVERRIDE (tool type name)
    - Config file: ~/.wqm/llm_override.json (full configuration)
    """

    # Environment variable name for override
    ENV_VAR_NAME = "LLM_TOOL_OVERRIDE"

    # Default config file location
    DEFAULT_CONFIG_DIR = Path.home() / ".wqm"
    CONFIG_FILE_NAME = "llm_override.json"

    @classmethod
    def _get_config_path(cls) -> Path:
        """
        Get the path to the configuration file.

        Returns:
            Path to llm_override.json
        """
        return cls.DEFAULT_CONFIG_DIR / cls.CONFIG_FILE_NAME

    @classmethod
    def _ensure_config_dir(cls) -> None:
        """
        Ensure the configuration directory exists.

        Creates ~/.wqm if it doesn't exist.
        """
        config_dir = cls.DEFAULT_CONFIG_DIR
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created config directory: {config_dir}")

    @classmethod
    def _read_from_env(cls) -> LLMOverrideConfig | None:
        """
        Read override configuration from environment variable.

        Environment variable format: just the tool type name (e.g., "claude_code")

        Returns:
            LLMOverrideConfig if env var is set and valid, None otherwise
        """
        env_value = os.environ.get(cls.ENV_VAR_NAME)
        if not env_value:
            return None

        # Try to parse as LLMToolType
        try:
            tool_type = LLMToolType(env_value.lower())
            logger.debug(f"Read override from environment: {tool_type.value}")
            return LLMOverrideConfig(
                enabled=True,
                tool_type=tool_type,
                reason="Set via environment variable",
                set_at=datetime.now().isoformat(),
                set_by="environment",
            )
        except ValueError:
            logger.warning(
                f"Invalid LLM_TOOL_OVERRIDE value: {env_value}. "
                f"Valid options: {', '.join(t.value for t in LLMToolType)}"
            )
            return None

    @classmethod
    def _read_from_file(cls) -> LLMOverrideConfig | None:
        """
        Read override configuration from JSON file.

        Returns:
            LLMOverrideConfig if file exists and is valid, None otherwise
        """
        config_path = cls._get_config_path()
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                data = json.load(f)
            config = LLMOverrideConfig.from_dict(data)
            logger.debug(f"Read override from file: {config_path}")
            return config
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to read override config from {config_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading override config: {e}")
            return None

    @classmethod
    def _write_to_file(cls, config: LLMOverrideConfig) -> None:
        """
        Write override configuration to JSON file.

        Args:
            config: Configuration to write

        Raises:
            OSError: If file cannot be written
        """
        cls._ensure_config_dir()
        config_path = cls._get_config_path()

        try:
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"Wrote override config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to write override config to {config_path}: {e}")
            raise

    @classmethod
    def get_override(cls) -> LLMOverrideConfig | None:
        """
        Get the current override configuration.

        Checks sources in priority order:
        1. Environment variable (LLM_TOOL_OVERRIDE)
        2. Configuration file (~/.wqm/llm_override.json)

        Returns:
            LLMOverrideConfig if an override is set and enabled, None otherwise
        """
        # Priority 1: Environment variable
        env_config = cls._read_from_env()
        if env_config and env_config.enabled:
            return env_config

        # Priority 2: Config file
        file_config = cls._read_from_file()
        if file_config and file_config.enabled:
            return file_config

        return None

    @classmethod
    def is_override_active(cls) -> bool:
        """
        Check if an override is currently active.

        Returns:
            True if an enabled override is configured, False otherwise
        """
        config = cls.get_override()
        return config is not None and config.enabled and config.tool_type is not None

    @classmethod
    def set_override(
        cls,
        tool_type: LLMToolType,
        reason: str = "",
        set_by: str = "user",
    ) -> None:
        """
        Set a manual override for LLM tool detection.

        This writes the override to the configuration file. To set via
        environment variable, use: export LLM_TOOL_OVERRIDE=tool_name

        Args:
            tool_type: Which LLM tool to force
            reason: Optional reason for the override
            set_by: Who/what is setting the override (default: "user")

        Raises:
            ValueError: If tool_type is not a valid LLMToolType
            OSError: If config file cannot be written

        Example:
            >>> from context_injection import LLMOverrideManager, LLMToolType
            >>> LLMOverrideManager.set_override(
            ...     LLMToolType.CLAUDE_CODE,
            ...     reason="Testing Claude Code formatter",
            ...     set_by="cli"
            ... )
        """
        if not isinstance(tool_type, LLMToolType):
            raise ValueError(
                f"tool_type must be LLMToolType, got {type(tool_type).__name__}"
            )

        config = LLMOverrideConfig(
            enabled=True,
            tool_type=tool_type,
            reason=reason or f"Manual override to {tool_type.value}",
            set_at=datetime.now().isoformat(),
            set_by=set_by,
        )

        cls._write_to_file(config)
        logger.info(
            f"Set LLM tool override: {tool_type.value} "
            f"(reason: {config.reason}, set_by: {set_by})"
        )

    @classmethod
    def clear_override(cls) -> None:
        """
        Clear the manual override configuration.

        This disables the override in the config file. Environment variable
        overrides must be cleared by unsetting the environment variable.

        Note: If LLM_TOOL_OVERRIDE environment variable is set, it will
        still take effect even after clearing the config file.
        """
        config_path = cls._get_config_path()

        # Check if there's an environment variable override
        if os.environ.get(cls.ENV_VAR_NAME):
            logger.warning(
                f"Environment variable {cls.ENV_VAR_NAME} is set. "
                "Clear it with: unset LLM_TOOL_OVERRIDE"
            )

        # Write disabled config to file
        config = LLMOverrideConfig(enabled=False)
        try:
            cls._write_to_file(config)
            logger.info("Cleared LLM tool override in config file")
        except Exception as e:
            logger.error(f"Failed to clear override config: {e}")
            # Try to delete the file as fallback
            try:
                if config_path.exists():
                    config_path.unlink()
                    logger.info(f"Deleted override config file: {config_path}")
            except Exception as delete_error:
                logger.error(f"Failed to delete config file: {delete_error}")
                raise


# CLI Helper Functions


def set_llm_override_cli(tool_name: str, reason: str = "") -> None:
    """
    CLI helper to set LLM tool override.

    Args:
        tool_name: Name of the tool (e.g., "claude_code", "github_copilot")
        reason: Optional reason for the override

    Raises:
        ValueError: If tool_name is not valid
        OSError: If config cannot be written

    Example:
        >>> from context_injection import set_llm_override_cli
        >>> set_llm_override_cli("claude_code", "Testing formatter")
    """
    try:
        tool_type = LLMToolType(tool_name.lower())
    except ValueError:
        valid_tools = ", ".join(t.value for t in LLMToolType)
        raise ValueError(
            f"Invalid tool name: {tool_name}. Valid options: {valid_tools}"
        )

    LLMOverrideManager.set_override(tool_type, reason=reason, set_by="cli")
    print(f"✓ Set LLM tool override to: {tool_type.value}")
    if reason:
        print(f"  Reason: {reason}")


def clear_llm_override_cli() -> None:
    """
    CLI helper to clear LLM tool override.

    Example:
        >>> from context_injection import clear_llm_override_cli
        >>> clear_llm_override_cli()
    """
    LLMOverrideManager.clear_override()
    print("✓ Cleared LLM tool override")

    # Check for environment variable
    if os.environ.get(LLMOverrideManager.ENV_VAR_NAME):
        print(
            f"⚠ Warning: {LLMOverrideManager.ENV_VAR_NAME} environment variable is still set"
        )
        print("  Clear it with: unset LLM_TOOL_OVERRIDE")


def show_llm_override_cli() -> None:
    """
    CLI helper to show current LLM tool override status.

    Example:
        >>> from context_injection import show_llm_override_cli
        >>> show_llm_override_cli()
    """
    # Check environment variable first
    env_value = os.environ.get(LLMOverrideManager.ENV_VAR_NAME)
    if env_value:
        print(f"Environment Variable Override: {env_value}")
        print(f"  (Set via ${LLMOverrideManager.ENV_VAR_NAME})")
        print()

    # Check config file
    config = LLMOverrideManager.get_override()
    if config and config.enabled:
        print("Active Override Configuration:")
        print(f"  Tool: {config.tool_type.value if config.tool_type else 'None'}")
        if config.reason:
            print(f"  Reason: {config.reason}")
        if config.set_at:
            print(f"  Set at: {config.set_at}")
        if config.set_by:
            print(f"  Set by: {config.set_by}")
    else:
        if not env_value:
            print("No active override configuration")
            print()
            print("Available tools:")
            for tool in LLMToolType:
                print(f"  - {tool.value}")
