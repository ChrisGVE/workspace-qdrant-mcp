"""
Loguru-based logging configuration with MCP stdio mode awareness.

This module provides a simplified logging configuration using loguru, designed to
replace the existing standard library logging system while maintaining compatibility
with MCP stdio mode requirements.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Any, Dict

from loguru import logger
from datetime import datetime, timezone
import platform
import time


def detect_mcp_mode() -> bool:
    """Detect if we're running in MCP stdio mode.

    Checks multiple indicators to determine if console output should be suppressed
    to avoid interfering with the MCP JSON-RPC protocol.

    Returns:
        True if MCP stdio mode is detected and console should be suppressed
    """
    # Check explicit stdio mode flags
    if os.getenv("WQM_STDIO_MODE", "false").lower() == "true":
        return True

    if os.getenv("MCP_QUIET_MODE", "false").lower() == "true":
        return True

    if os.getenv("DISABLE_MCP_CONSOLE_LOGS", "false").lower() == "true":
        return True

    # Check if we're running via MCP transport
    if os.getenv("MCP_TRANSPORT") == "stdio":
        return True

    # Heuristic: check if stdout appears to be piped (typical MCP scenario)
    # When Claude Desktop runs MCP servers, stdout is piped for JSON-RPC
    if not sys.stdout.isatty() and not sys.stdin.isatty() and os.getenv("TERM") is None:
        return True

    return False


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "gz",
    json_format: bool = True,
    console_output: bool = True,
    force_stderr: bool = False,
    **kwargs: Any
) -> None:
    """Configure loguru logging with MCP stdio mode awareness.

    This function configures loguru to provide structured logging with automatic
    MCP stdio mode detection and appropriate output handling.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None disables file logging)
        rotation: File rotation trigger (e.g., "10 MB", "1 day", "12:00")
        retention: Log retention period (e.g., "7 days", "1 week")
        compression: Compression format for rotated files ("gz", "bz2", "xz")
        json_format: Use JSON formatting for structured logs
        console_output: Enable console output (overridden by MCP mode)
        force_stderr: Force console output to stderr instead of stdout
        **kwargs: Additional loguru configuration options

    Note:
        This function is idempotent - safe to call multiple times.
        In MCP stdio mode, console output is automatically suppressed.
    """
    # Remove all existing handlers to ensure clean configuration
    logger.remove()

    # Detect MCP stdio mode
    mcp_mode = detect_mcp_mode()

    # Determine effective console output setting
    # MCP mode overrides console_output setting
    effective_console_output = console_output and not mcp_mode

    # Configure console handler if needed
    if effective_console_output:
        # Choose output stream
        console_stream = sys.stderr if (force_stderr or mcp_mode) else sys.stdout

        # Configure console handler
        if json_format:
            # For JSON, use loguru's serialize feature
            logger.add(
                console_stream,
                level=level,
                serialize=True,  # JSON serialization
                enqueue=True,
                **_filter_loguru_kwargs(kwargs)
            )
        else:
            # For text format, use colored human-readable format
            console_format = _get_text_format()
            logger.add(
                console_stream,
                level=level,
                format=console_format,
                colorize=console_stream.isatty(),
                enqueue=True,
                **_filter_loguru_kwargs(kwargs)
            )

    # Configure file handler if specified
    if log_file:
        log_path = Path(log_file)

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add file handler with JSON serialization and rotation
        logger.add(
            log_path,
            level=level,
            serialize=True,  # Always use JSON for files
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            **_filter_loguru_kwargs(kwargs)
        )

    # Configure third-party library loggers
    _configure_third_party_loggers(mcp_mode)


def _get_text_format() -> str:
    """Get human-readable text format string."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )


def _filter_loguru_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include valid loguru handler parameters."""
    valid_params = {
        'serialize', 'backtrace', 'diagnose', 'catch', 'loop',
        'buffering', 'delay', 'watch', 'mode', 'buffering',
        'encoding', 'errors', 'newline', 'compression'
    }

    return {k: v for k, v in kwargs.items() if k in valid_params}


def _configure_third_party_loggers(mcp_mode: bool) -> None:
    """Configure third-party library loggers to work with loguru.

    Args:
        mcp_mode: Whether MCP stdio mode is active
    """
    # List of third-party loggers to configure
    third_party_loggers = [
        'httpx', 'httpcore', 'qdrant_client', 'urllib3', 'asyncio',
        'fastembed', 'grpc', 'structlog', 'typer', 'pydantic', 'rich',
        'uvicorn', 'fastapi', 'watchdog', 'watchfiles', 'aiohttp',
        'aiofiles', 'requests', 'chardet', 'pypdf', 'python_docx',
        'python_pptx', 'ebooklib', 'beautifulsoup4', 'lxml',
        'markdown', 'pygments', 'psutil', 'tqdm', 'multipart',
        'tokenizers', 'transformers', 'sentence_transformers',
        'bitsandbytes', 'safetensors', 'huggingface_hub'
    ]

    # Configure standard library logging to intercept third-party logs
    import logging

    class InterceptHandler(logging.Handler):
        """Intercept standard logging calls and redirect to loguru."""

        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Install the intercept handler for standard logging
    if not mcp_mode:  # Only in non-MCP mode
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Configure specific third-party loggers
    for logger_name in third_party_loggers:
        third_party_logger = logging.getLogger(logger_name)

        if mcp_mode:
            # In MCP mode, completely silence third-party loggers
            third_party_logger.handlers.clear()
            third_party_logger.addHandler(logging.NullHandler())
            third_party_logger.setLevel(logging.CRITICAL + 1)
            third_party_logger.disabled = True
            third_party_logger.propagate = False
        else:
            # In normal mode, reduce verbosity but allow some logging
            if logger_name in ['httpx', 'httpcore', 'urllib3', 'requests']:
                third_party_logger.setLevel(logging.WARNING)
            elif logger_name in ['qdrant_client', 'fastembed']:
                third_party_logger.setLevel(logging.INFO)
            elif logger_name in ['tokenizers', 'transformers', 'sentence_transformers']:
                third_party_logger.setLevel(logging.ERROR)
            else:
                third_party_logger.setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """Get a loguru logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured loguru logger instance
    """
    return logger.bind(logger_name=name)


def display_development_banner(
    project_name: str = "workspace-qdrant-mcp",
    version: str = "0.2.1dev1",
    server_mode: str = "MCP",
    transport: str = "stdio",
    config_summary: Optional[str] = None,
    startup_time: Optional[float] = None,
    force_display: bool = False
) -> None:
    """Display an elegant development banner using loguru with rich formatting.

    This function creates a FastMCP-style elegant development banner that provides
    useful information about the server startup while respecting MCP stdio mode
    silence requirements.

    Args:
        project_name: Name of the project/server
        version: Version string
        server_mode: Operating mode (MCP, CLI, HTTP, etc.)
        transport: Transport protocol (stdio, http, sse)
        config_summary: Brief configuration description
        startup_time: Time taken to start up in seconds
        force_display: Force display even in stdio mode (for testing)

    Note:
        Banner is automatically suppressed in MCP stdio mode unless force_display=True.
        Output goes to stderr to avoid interfering with MCP JSON-RPC protocol.
    """
    # Detect MCP mode and respect silence requirements
    mcp_mode = detect_mcp_mode()

    if mcp_mode and not force_display:
        # In MCP stdio mode, respect protocol silence
        return

    # Get startup timestamp
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Format startup time if provided
    startup_display = f"{startup_time:.2f}s" if startup_time else "--"

    # Detect platform info
    platform_info = f"{platform.system()} {platform.machine()}"

    # Create banner components with rich formatting
    banner_width = 60
    border_char = "─"

    # Banner header
    header = f"╭{border_char * (banner_width - 2)}╮"
    footer = f"╰{border_char * (banner_width - 2)}╯"

    # Format project info line
    project_line = f"│ 🚀 {project_name:<{banner_width - 6}} │"

    # Format version and mode
    version_mode = f"v{version} • {server_mode} mode"
    version_line = f"│ 📦 {version_mode:<{banner_width - 6}} │"

    # Format transport
    transport_display = f"Transport: {transport.upper()}"
    transport_line = f"│ 🔗 {transport_display:<{banner_width - 6}} │"

    # Format platform
    platform_display = f"Platform: {platform_info}"
    platform_line = f"│ 💻 {platform_display:<{banner_width - 6}} │"

    # Format startup time
    startup_display_full = f"Started in: {startup_display}"
    startup_line = f"│ ⚡ {startup_display_full:<{banner_width - 6}} │"

    # Format timestamp
    time_display = f"Time: {timestamp}"
    time_line = f"│ 🕒 {time_display:<{banner_width - 6}} │"

    # Optional config line
    config_line = None
    if config_summary:
        config_display = config_summary
        if len(config_display) > banner_width - 10:
            config_display = f"{config_display[:banner_width - 13]}..."
        config_line = f"│ ⚙️  {config_display:<{banner_width - 6}} │"

    # Status line
    status_display = "Ready for connections"
    status_line = f"│ ✅ {status_display:<{banner_width - 6}} │"

    # Build complete banner
    banner_lines = [
        header,
        project_line,
        version_line,
        f"│{' ' * (banner_width - 2)}│",  # Spacer
        transport_line,
        platform_line,
        startup_line,
        time_line,
    ]

    if config_line:
        banner_lines.extend([
            f"│{' ' * (banner_width - 2)}│",  # Spacer
            config_line,
        ])

    banner_lines.extend([
        f"│{' ' * (banner_width - 2)}│",  # Spacer
        status_line,
        footer
    ])

    # Use loguru to display with rich formatting
    # Create a temporary logger that outputs to stderr with colors
    temp_logger = logger.bind(component="banner")

    # Display banner with loguru's rich formatting
    for line in banner_lines:
        # Use loguru's info level with rich markup
        temp_logger.info(f"<green>{line}</green>")

    # Add separator line
    temp_logger.info("")


def log_startup_event(
    event: str,
    component: str,
    details: Optional[dict] = None,
    level: str = "INFO"
) -> None:
    """Log startup events with structured formatting.

    Args:
        event: Event description (e.g., "Server starting", "Database connected")
        component: Component name (e.g., "server", "database", "config")
        details: Additional structured details
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    startup_logger = get_logger(f"startup.{component}")

    log_data = {
        "event": event,
        "component": component,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if details:
        log_data.update(details)

    # Log with appropriate level
    log_method = getattr(startup_logger, level.lower(), startup_logger.info)
    log_method(f"🔧 {event}", **log_data)


# Configuration from environment variables
def configure_from_environment() -> None:
    """Configure loguru from environment variables.

    Environment Variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_FORMAT: json or text (default: json)
        LOG_FILE: Path to log file (optional)
        LOG_CONSOLE: Enable console output (default: true)
        LOG_ROTATION: File rotation trigger (default: 10 MB)
        LOG_RETENTION: Log retention period (default: 7 days)
        LOG_COMPRESSION: Compression format (default: gz)
        WQM_STDIO_MODE: Enable MCP stdio mode (default: false)
        MCP_QUIET_MODE: Disable console output in MCP stdio mode
    """
    # Get configuration from environment
    level = os.getenv("LOG_LEVEL", "INFO")
    json_format = os.getenv("LOG_FORMAT", "json").lower() == "json"
    log_file = os.getenv("LOG_FILE")
    console_output = os.getenv("LOG_CONSOLE", "true").lower() == "true"
    rotation = os.getenv("LOG_ROTATION", "10 MB")
    retention = os.getenv("LOG_RETENTION", "7 days")
    compression = os.getenv("LOG_COMPRESSION", "gz")

    # Configure with detected settings
    configure_logging(
        level=level,
        log_file=log_file,
        rotation=rotation,
        retention=retention,
        compression=compression,
        json_format=json_format,
        console_output=console_output,
    )