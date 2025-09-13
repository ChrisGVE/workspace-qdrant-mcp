"""
Configuration models and MCP detection for the unified logging system.

This module handles the detection of MCP stdio mode and provides Pydantic
configuration models for structured logging setup.
"""

import os
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Pydantic configuration model for logging setup."""

    level: Union[str, int] = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    json_format: bool = Field(
        default=True,
        description="Use JSON formatting for structured logs"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Path to log file (optional, logs to console if None)"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size before rotation"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    console_output: bool = Field(
        default=True,
        description="Whether to output logs to console"
    )
    stdio_mode: bool = Field(
        default=False,
        description="Whether running in MCP stdio mode (disables stdout logging)"
    )
    force_stderr: bool = Field(
        default=False,
        description="Force console output to stderr instead of stdout"
    )

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v):
        """Convert string level to uppercase for consistency."""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    @classmethod
    def from_environment(cls) -> "LoggingConfig":
        """Create configuration from environment variables.

        Environment Variables:
            LOG_LEVEL: Logging level (default: INFO)
            LOG_FORMAT: json or text (default: json)
            LOG_FILE: Path to log file (optional)
            LOG_CONSOLE: Enable console output (default: true)
            LOG_MAX_FILE_SIZE: Max file size in bytes (default: 10MB)
            LOG_BACKUP_COUNT: Number of backup files (default: 5)
            WQM_STDIO_MODE: Enable MCP stdio mode (default: false)
            MCP_QUIET_MODE: Disable console output in MCP stdio mode
            DISABLE_MCP_CONSOLE_LOGS: Alternative env var to disable console output
        """
        # Get basic configuration from environment
        level = os.getenv("LOG_LEVEL", "INFO")
        json_format = os.getenv("LOG_FORMAT", "json").lower() == "json"
        log_file_path = os.getenv("LOG_FILE")
        console_output = os.getenv("LOG_CONSOLE", "true").lower() == "true"
        max_file_size = int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

        # Detect MCP stdio mode
        stdio_mode = detect_mcp_mode()

        # Convert log file path if provided
        log_file = Path(log_file_path) if log_file_path else None

        return cls(
            level=level,
            json_format=json_format,
            log_file=log_file,
            max_file_size=max_file_size,
            backup_count=backup_count,
            console_output=console_output,
            stdio_mode=stdio_mode,
            force_stderr=stdio_mode,  # Force stderr in MCP mode
        )


def detect_mcp_mode() -> bool:
    """Detect if we're running in MCP stdio mode.

    Checks multiple environment variables to determine if console output
    should be suppressed to avoid interfering with the MCP protocol.

    Returns:
        True if MCP stdio mode is detected and console should be suppressed
    """
    # Check explicit stdio mode flag
    if os.getenv("WQM_STDIO_MODE", "false").lower() == "true":
        return True

    # Check MCP quiet mode flag
    if os.getenv("MCP_QUIET_MODE", "false").lower() == "true":
        return True

    # Check explicit console disable flag
    if os.getenv("DISABLE_MCP_CONSOLE_LOGS", "false").lower() == "true":
        return True

    # Check if we're running via MCP (Claude Desktop sets these)
    if os.getenv("MCP_TRANSPORT") == "stdio":
        return True

    # Check if stdout appears to be connected to a pipe (MCP scenario)
    # This is a heuristic - when Claude Desktop runs MCP servers,
    # stdout is typically piped for JSON-RPC communication
    import sys
    if not sys.stdout.isatty() and os.getenv("TERM") is None:
        # Not a TTY and no TERM set - likely running under MCP
        return True

    return False


def should_suppress_console(config: LoggingConfig) -> bool:
    """Determine if console output should be suppressed.

    Args:
        config: Logging configuration

    Returns:
        True if console output should be completely suppressed
    """
    # If console output is explicitly disabled, suppress it
    if not config.console_output:
        return True

    # If we're in stdio mode (MCP protocol), suppress console output by default
    # to prevent interference with the JSON-RPC protocol
    if config.stdio_mode:
        # Check if console output is explicitly enabled despite stdio mode
        force_console = os.getenv("FORCE_CONSOLE_IN_STDIO", "false").lower() == "true"
        if not force_console:
            return True

    # Check explicit console disable flags
    if os.getenv("DISABLE_MCP_CONSOLE_LOGS", "false").lower() == "true":
        return True

    # MCP_QUIET_MODE defaults to true for stdio mode
    if os.getenv("MCP_QUIET_MODE", "false").lower() == "true":
        return True

    return False