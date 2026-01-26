"""
CLAUDE.md injection mechanism for Claude Code sessions.

This module provides functionality to read, parse, and inject CLAUDE.md content
into Claude Code sessions, with support for file watching and precedence rules.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..memory import MemoryManager
from .formatters.claude_code import ClaudeCodeAdapter
from .rule_retrieval import RuleFilter, RuleRetrieval


@dataclass
class ClaudeMdLocation:
    """
    Represents a CLAUDE.md file location.

    Attributes:
        path: Absolute path to CLAUDE.md file
        is_global: Whether this is the global CLAUDE.md
        precedence: Priority level (higher = more important)
    """

    path: Path
    is_global: bool
    precedence: int


class ClaudeMdFileHandler(FileSystemEventHandler):
    """
    Watchdog file system event handler for CLAUDE.md changes.

    Monitors CLAUDE.md files for changes and triggers callbacks.
    """

    def __init__(self, callback: Callable[[Path], None]):
        """
        Initialize the file handler.

        Args:
            callback: Function to call when CLAUDE.md changes
        """
        super().__init__()
        self.callback = callback
        self._last_modified: dict[str, float] = {}
        self._debounce_seconds = 1.0

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.name == "CLAUDE.md":
            self._handle_change(file_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.name == "CLAUDE.md":
            self._handle_change(file_path)

    def _handle_change(self, file_path: Path) -> None:
        """
        Handle CLAUDE.md change with debouncing.

        Args:
            file_path: Path to changed CLAUDE.md file
        """
        now = time.time()
        file_key = str(file_path)

        # Debounce rapid changes
        if file_key in self._last_modified:
            if now - self._last_modified[file_key] < self._debounce_seconds:
                logger.debug(f"Debouncing change to {file_path}")
                return

        self._last_modified[file_key] = now
        logger.info(f"Detected change to {file_path}")
        self.callback(file_path)


class _NoopObserver:
    """No-op observer used to avoid watchdog threads in test mode."""

    def schedule(self, *_args, **_kwargs) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, *_args, **_kwargs) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class ClaudeMdInjector:
    """
    CLAUDE.md injection mechanism for Claude Code sessions.

    This class handles:
    - Discovery of CLAUDE.md files (project-level and global)
    - Content parsing and formatting
    - File watching for live updates
    - Precedence rules (project overrides global)
    - Integration with memory rule system
    """

    # Default global CLAUDE.md location
    GLOBAL_CLAUDE_MD = Path.home() / ".claude" / "CLAUDE.md"

    def __init__(
        self,
        memory_manager: MemoryManager,
        rule_retrieval: RuleRetrieval | None = None,
        adapter: ClaudeCodeAdapter | None = None,
        enable_watching: bool = True,
    ):
        """
        Initialize the CLAUDE.md injector.

        Args:
            memory_manager: MemoryManager instance for rule storage
            rule_retrieval: RuleRetrieval instance (created if not provided)
            adapter: ClaudeCodeAdapter instance (created if not provided)
            enable_watching: Enable file watching for live updates
        """
        self.memory_manager = memory_manager
        self.rule_retrieval = rule_retrieval or RuleRetrieval(memory_manager)
        self.adapter = adapter or ClaudeCodeAdapter()
        self.enable_watching = enable_watching

        # File watching infrastructure
        self._observer: Observer | None = None
        self._watched_paths: list[Path] = []
        self._change_callbacks: list[Callable[[Path], None]] = []

    def discover_claude_md_files(
        self, project_root: Path | None = None
    ) -> list[ClaudeMdLocation]:
        """
        Discover CLAUDE.md files with precedence ordering.

        Precedence rules:
        1. Project-level CLAUDE.md (highest precedence)
        2. Global CLAUDE.md (~/.claude/CLAUDE.md)

        Args:
            project_root: Project root directory (default: current working directory)

        Returns:
            List of ClaudeMdLocation objects in precedence order (highest first)
        """
        locations = []

        # Determine project root
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root).resolve()

        # Check project-level CLAUDE.md
        project_claude_md = project_root / "CLAUDE.md"
        if project_claude_md.exists() and project_claude_md.is_file():
            locations.append(
                ClaudeMdLocation(
                    path=project_claude_md, is_global=False, precedence=100
                )
            )
            logger.debug(f"Found project CLAUDE.md: {project_claude_md}")

        # Check global CLAUDE.md
        if self.GLOBAL_CLAUDE_MD.exists() and self.GLOBAL_CLAUDE_MD.is_file():
            locations.append(
                ClaudeMdLocation(
                    path=self.GLOBAL_CLAUDE_MD, is_global=True, precedence=50
                )
            )
            logger.debug(f"Found global CLAUDE.md: {self.GLOBAL_CLAUDE_MD}")

        # Sort by precedence (highest first)
        locations.sort(key=lambda loc: loc.precedence, reverse=True)

        logger.info(f"Discovered {len(locations)} CLAUDE.md file(s)")
        return locations

    async def inject_from_files(
        self,
        project_root: Path | None = None,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
    ) -> str:
        """
        Read CLAUDE.md files and inject content with memory rules.

        This method:
        1. Discovers CLAUDE.md files (project and global)
        2. Reads content with precedence (project overrides global)
        3. Retrieves applicable memory rules
        4. Formats everything using ClaudeCodeAdapter

        Args:
            project_root: Project root directory
            token_budget: Token budget for formatted output
            filter: Optional filter for memory rules

        Returns:
            Formatted markdown content ready for injection
        """
        # Discover CLAUDE.md files
        locations = self.discover_claude_md_files(project_root)

        if not locations:
            logger.warning("No CLAUDE.md files found")
            return ""

        # Read content from highest precedence file
        # (In future, we could merge content, but for now use highest precedence only)
        primary_location = locations[0]
        content = self._read_claude_md(primary_location.path)

        if not content:
            logger.warning(f"Empty content from {primary_location.path}")
            return content

        logger.info(
            f"Read {len(content)} characters from {primary_location.path} "
            f"({'global' if primary_location.is_global else 'project'})"
        )

        # Retrieve and format memory rules
        memory_content = await self._get_memory_rules_content(
            token_budget=token_budget, filter=filter
        )

        if memory_content:
            # Append memory rules section to CLAUDE.md content
            content += f"\n\n---\n\n{memory_content}"
            logger.info("Appended memory rules to CLAUDE.md content")

        return content

    async def inject_to_file(
        self,
        output_path: Path,
        project_root: Path | None = None,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
    ) -> bool:
        """
        Generate injected content and write to file.

        Args:
            output_path: Path to write formatted content
            project_root: Project root directory
            token_budget: Token budget for formatted output
            filter: Optional filter for memory rules

        Returns:
            True if successful, False otherwise
        """
        try:
            content = await self.inject_from_files(
                project_root=project_root, token_budget=token_budget, filter=filter
            )

            if not content:
                logger.warning("No content to inject")
                return False

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Wrote {len(content)} characters to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to inject content to {output_path}: {e}")
            return False

    def start_watching(
        self,
        project_root: Path | None = None,
        callback: Callable[[Path], None] | None = None,
    ) -> bool:
        """
        Start watching CLAUDE.md files for changes.

        Args:
            project_root: Project root directory
            callback: Optional callback for change notifications

        Returns:
            True if watching started successfully, False otherwise
        """
        if not self.enable_watching:
            logger.warning("File watching is disabled")
            return False

        if self._observer and self._observer.is_alive():
            logger.warning("Already watching CLAUDE.md files")
            return False

        # Discover files to watch
        locations = self.discover_claude_md_files(project_root)
        if not locations:
            logger.warning("No CLAUDE.md files to watch")
            return False

        # Create observer (use no-op in test mode to avoid watchdog crashes)
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("WQM_TEST_MODE"):
            self._observer = _NoopObserver()
        else:
            self._observer = Observer()

        # Add callback if provided
        if callback:
            self._change_callbacks.append(callback)

        # Create event handler
        handler = ClaudeMdFileHandler(self._handle_file_change)

        # Watch each location's parent directory
        for location in locations:
            watch_dir = location.path.parent
            if watch_dir not in self._watched_paths:
                self._observer.schedule(handler, str(watch_dir), recursive=False)
                self._watched_paths.append(watch_dir)
                logger.debug(f"Watching directory: {watch_dir}")

        # Start observer
        self._observer.start()
        logger.info(f"Started watching {len(self._watched_paths)} directories")

        return True

    def stop_watching(self) -> None:
        """Stop watching CLAUDE.md files."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5)
            logger.info("Stopped watching CLAUDE.md files")

        self._observer = None
        self._watched_paths = []

    def add_change_callback(self, callback: Callable[[Path], None]) -> None:
        """
        Add a callback to be notified of CLAUDE.md changes.

        Args:
            callback: Function that takes Path as argument
        """
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)

    def _handle_file_change(self, file_path: Path) -> None:
        """
        Handle CLAUDE.md file change event.

        Args:
            file_path: Path to changed file
        """
        logger.info(f"CLAUDE.md changed: {file_path}")

        # Notify all callbacks
        for callback in self._change_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    def _read_claude_md(self, path: Path) -> str:
        """
        Read CLAUDE.md file content.

        Args:
            path: Path to CLAUDE.md file

        Returns:
            File content as string
        """
        try:
            content = path.read_text(encoding="utf-8")
            return content
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return ""

    async def _get_memory_rules_content(
        self, token_budget: int, filter: RuleFilter | None = None
    ) -> str:
        """
        Retrieve and format memory rules.

        Args:
            token_budget: Token budget for formatted output
            filter: Optional filter for memory rules

        Returns:
            Formatted memory rules content
        """
        try:
            # Get rules from retrieval system
            if filter is None:
                filter = RuleFilter(limit=100)

            result = await self.rule_retrieval.get_rules(filter)

            if not result.rules:
                logger.debug("No memory rules found")
                return ""

            # Format using ClaudeCodeAdapter
            formatted = self.adapter.format_rules(
                rules=result.rules, token_budget=token_budget
            )

            if not formatted.content:
                logger.warning("Failed to format memory rules")
                return ""

            logger.info(
                f"Formatted {formatted.rules_included} memory rules "
                f"({formatted.token_count} tokens)"
            )

            return formatted.content

        except Exception as e:
            logger.error(f"Failed to get memory rules content: {e}")
            return ""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop watching."""
        self.stop_watching()


# Convenience function for quick injection
async def inject_claude_md_content(
    memory_manager: MemoryManager,
    project_root: Path | None = None,
    token_budget: int = 50000,
) -> str:
    """
    Convenience function to inject CLAUDE.md content.

    Args:
        memory_manager: MemoryManager instance
        project_root: Project root directory
        token_budget: Token budget for formatted output

    Returns:
        Formatted content ready for injection

    Example:
        >>> from context_injection import inject_claude_md_content
        >>> content = await inject_claude_md_content(memory_manager)
    """
    injector = ClaudeMdInjector(memory_manager)
    return await injector.inject_from_files(
        project_root=project_root, token_budget=token_budget
    )
