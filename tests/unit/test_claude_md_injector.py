"""
Unit tests for CLAUDE.md injection mechanism.

Tests file discovery, content injection, file watching, and precedence rules.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from common.core.context_injection import (
    ClaudeMdInjector,
    ClaudeMdLocation,
    inject_claude_md_content,
)
from common.core.context_injection.formatters import FormattedContext
from common.core.context_injection.rule_retrieval import (
    RuleFilter,
    RuleRetrievalResult,
)
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    manager = AsyncMock(spec=MemoryManager)
    return manager


@pytest.fixture
def sample_memory_rules():
    """Create sample memory rules for testing."""
    from datetime import datetime, timezone

    return [
        MemoryRule(
            id="rule1",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            name="Python Style",
            rule="Use black formatting",
            scope=["python"],
            source="user_explicit",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={"priority": 100},
        ),
        MemoryRule(
            id="rule2",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            name="Test Coverage",
            rule="Maintain 80% test coverage",
            scope=["testing"],
            source="project_convention",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={"priority": 80},
        ),
    ]


@pytest.fixture(autouse=True)
def patch_global_claude_md(tmp_path, monkeypatch):
    """Auto-patch global CLAUDE.md location for all tests to avoid interference."""
    global_dir = tmp_path / ".claude_test_global"
    global_dir.mkdir()
    
    # Create a non-existent path by default (tests can create file if needed)
    global_claude_md = global_dir / "CLAUDE.md"
    
    monkeypatch.setattr(
        "common.core.context_injection.claude_md_injector.ClaudeMdInjector.GLOBAL_CLAUDE_MD",
        global_claude_md,
    )
    
    return global_claude_md


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory with CLAUDE.md."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create project CLAUDE.md
    claude_md = project_dir / "CLAUDE.md"
    claude_md.write_text("# Project Instructions\n\nThis is the project CLAUDE.md\n")

    return project_dir


@pytest.fixture
def temp_global_claude_md(patch_global_claude_md):
    """Create a temporary global CLAUDE.md file (uses patched location)."""
    # The path is already patched by the autouse fixture
    global_claude_md = patch_global_claude_md
    global_claude_md.write_text("# Global Instructions\n\nThis is the global CLAUDE.md\n")
    return global_claude_md


class TestClaudeMdLocation:
    """Test ClaudeMdLocation dataclass."""

    def test_location_creation(self):
        """Test creating ClaudeMdLocation."""
        location = ClaudeMdLocation(
            path=Path("/test/CLAUDE.md"), is_global=False, precedence=100
        )

        assert location.path == Path("/test/CLAUDE.md")
        assert location.is_global is False
        assert location.precedence == 100


class TestClaudeMdInjector:
    """Test ClaudeMdInjector class."""

    def test_initialization(self, mock_memory_manager):
        """Test ClaudeMdInjector initialization."""
        injector = ClaudeMdInjector(mock_memory_manager)

        assert injector.memory_manager == mock_memory_manager
        assert injector.rule_retrieval is not None
        assert injector.adapter is not None
        assert injector.enable_watching is True

    def test_initialization_with_watching_disabled(self, mock_memory_manager):
        """Test initialization with watching disabled."""
        injector = ClaudeMdInjector(mock_memory_manager, enable_watching=False)

        assert injector.enable_watching is False

    def test_discover_project_claude_md(self, temp_project_dir):
        """Test discovering project-level CLAUDE.md."""
        injector = ClaudeMdInjector(AsyncMock())

        locations = injector.discover_claude_md_files(temp_project_dir)

        assert len(locations) == 1
        assert locations[0].path == temp_project_dir / "CLAUDE.md"
        assert locations[0].is_global is False
        assert locations[0].precedence == 100

    def test_discover_global_claude_md(self, temp_global_claude_md, tmp_path):
        """Test discovering global CLAUDE.md."""
        injector = ClaudeMdInjector(AsyncMock())

        # Use directory without project CLAUDE.md
        no_project_dir = tmp_path / "no_project"
        no_project_dir.mkdir()

        locations = injector.discover_claude_md_files(no_project_dir)

        assert len(locations) == 1
        assert locations[0].path == temp_global_claude_md
        assert locations[0].is_global is True
        assert locations[0].precedence == 50

    def test_discover_both_claude_md(self, temp_project_dir, temp_global_claude_md):
        """Test discovering both project and global CLAUDE.md."""
        injector = ClaudeMdInjector(AsyncMock())

        locations = injector.discover_claude_md_files(temp_project_dir)

        assert len(locations) == 2
        # Project should be first (higher precedence)
        assert locations[0].path == temp_project_dir / "CLAUDE.md"
        assert locations[0].precedence == 100
        # Global should be second
        assert locations[1].path == temp_global_claude_md
        assert locations[1].precedence == 50

    def test_discover_no_claude_md(self, tmp_path):
        """Test when no CLAUDE.md files exist."""
        injector = ClaudeMdInjector(AsyncMock())

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        locations = injector.discover_claude_md_files(empty_dir)

        assert len(locations) == 0

    def test_read_claude_md(self, temp_project_dir):
        """Test reading CLAUDE.md content."""
        injector = ClaudeMdInjector(AsyncMock())

        claude_md_path = temp_project_dir / "CLAUDE.md"
        content = injector._read_claude_md(claude_md_path)

        assert "Project Instructions" in content
        assert "This is the project CLAUDE.md" in content

    def test_read_claude_md_missing_file(self, tmp_path):
        """Test reading non-existent CLAUDE.md."""
        injector = ClaudeMdInjector(AsyncMock())

        missing_path = tmp_path / "missing.md"
        content = injector._read_claude_md(missing_path)

        assert content == ""

    @pytest.mark.asyncio
    async def test_inject_from_files_project(
        self, temp_project_dir, mock_memory_manager, sample_memory_rules
    ):
        """Test injecting content from project CLAUDE.md."""
        # Setup mock retrieval
        mock_result = RuleRetrievalResult(
            rules=sample_memory_rules,
            total_count=2,
            filtered_count=2,
            cache_hit=False,
        )

        injector = ClaudeMdInjector(mock_memory_manager)

        # Mock rule retrieval
        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            # Mock adapter formatting
            mock_formatted = FormattedContext(
                tool_name="claude",
                format_type="markdown",
                content="# Memory Rules\n\n- Rule 1\n- Rule 2\n",
                token_count=50,
                rules_included=2,
                rules_skipped=0,
                metadata={},
            )

            with patch.object(
                injector.adapter, "format_rules", return_value=mock_formatted
            ):
                content = await injector.inject_from_files(
                    project_root=temp_project_dir
                )

        assert "Project Instructions" in content
        assert "Memory Rules" in content
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_inject_from_files_no_files(self, tmp_path, mock_memory_manager):
        """Test injection when no CLAUDE.md files exist."""
        injector = ClaudeMdInjector(mock_memory_manager)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        content = await injector.inject_from_files(project_root=empty_dir)

        assert content == ""

    @pytest.mark.asyncio
    async def test_inject_to_file(
        self, temp_project_dir, mock_memory_manager, sample_memory_rules, tmp_path
    ):
        """Test injecting content to output file."""
        output_path = tmp_path / "output" / "injected.md"

        mock_result = RuleRetrievalResult(
            rules=sample_memory_rules,
            total_count=2,
            filtered_count=2,
            cache_hit=False,
        )

        injector = ClaudeMdInjector(mock_memory_manager)

        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            mock_formatted = FormattedContext(
                tool_name="claude",
                format_type="markdown",
                content="# Memory Rules\n\n- Rule 1\n",
                token_count=30,
                rules_included=2,
                rules_skipped=0,
                metadata={},
            )

            with patch.object(
                injector.adapter, "format_rules", return_value=mock_formatted
            ):
                success = await injector.inject_to_file(
                    output_path=output_path, project_root=temp_project_dir
                )

        assert success is True
        assert output_path.exists()

        written_content = output_path.read_text()
        assert "Project Instructions" in written_content
        assert "Memory Rules" in written_content

    @pytest.mark.asyncio
    async def test_inject_to_file_no_content(self, tmp_path, mock_memory_manager):
        """Test injection to file when no content available."""
        output_path = tmp_path / "output.md"

        injector = ClaudeMdInjector(mock_memory_manager)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        success = await injector.inject_to_file(
            output_path=output_path, project_root=empty_dir
        )

        assert success is False
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_get_memory_rules_content(
        self, mock_memory_manager, sample_memory_rules
    ):
        """Test retrieving and formatting memory rules."""
        mock_result = RuleRetrievalResult(
            rules=sample_memory_rules,
            total_count=2,
            filtered_count=2,
            cache_hit=False,
        )

        injector = ClaudeMdInjector(mock_memory_manager)

        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            # Use actual adapter to test formatting
            content = await injector._get_memory_rules_content(token_budget=10000)

        assert len(content) > 0
        assert "CRITICAL RULES" in content or "DEFAULT GUIDELINES" in content

    @pytest.mark.asyncio
    async def test_get_memory_rules_content_no_rules(self, mock_memory_manager):
        """Test memory rules content when no rules found."""
        mock_result = RuleRetrievalResult(
            rules=[], total_count=0, filtered_count=0, cache_hit=False
        )

        injector = ClaudeMdInjector(mock_memory_manager)

        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            content = await injector._get_memory_rules_content(token_budget=10000)

        assert content == ""

    def test_start_watching(self, temp_project_dir, mock_memory_manager):
        """Test starting file watching."""
        injector = ClaudeMdInjector(mock_memory_manager, enable_watching=True)

        success = injector.start_watching(project_root=temp_project_dir)

        assert success is True
        assert injector._observer is not None
        assert len(injector._watched_paths) > 0

        # Cleanup
        injector.stop_watching()

    def test_start_watching_disabled(self, temp_project_dir, mock_memory_manager):
        """Test starting watching when disabled."""
        injector = ClaudeMdInjector(mock_memory_manager, enable_watching=False)

        success = injector.start_watching(project_root=temp_project_dir)

        assert success is False

    def test_start_watching_no_files(self, tmp_path, mock_memory_manager):
        """Test starting watching when no CLAUDE.md files exist."""
        injector = ClaudeMdInjector(mock_memory_manager, enable_watching=True)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        success = injector.start_watching(project_root=empty_dir)

        assert success is False

    def test_stop_watching(self, temp_project_dir, mock_memory_manager):
        """Test stopping file watching."""
        injector = ClaudeMdInjector(mock_memory_manager, enable_watching=True)

        injector.start_watching(project_root=temp_project_dir)
        assert injector._observer is not None

        injector.stop_watching()
        assert injector._observer is None
        assert len(injector._watched_paths) == 0

    def test_add_change_callback(self, mock_memory_manager):
        """Test adding change callback."""
        injector = ClaudeMdInjector(mock_memory_manager)

        callback = Mock()
        injector.add_change_callback(callback)

        assert callback in injector._change_callbacks

    def test_handle_file_change(self, mock_memory_manager):
        """Test handling file change event."""
        injector = ClaudeMdInjector(mock_memory_manager)

        callback1 = Mock()
        callback2 = Mock()

        injector.add_change_callback(callback1)
        injector.add_change_callback(callback2)

        test_path = Path("/test/CLAUDE.md")
        injector._handle_file_change(test_path)

        callback1.assert_called_once_with(test_path)
        callback2.assert_called_once_with(test_path)

    def test_context_manager(self, temp_project_dir, mock_memory_manager):
        """Test using ClaudeMdInjector as context manager."""
        with ClaudeMdInjector(mock_memory_manager, enable_watching=True) as injector:
            injector.start_watching(project_root=temp_project_dir)
            assert injector._observer is not None

        # Should auto-cleanup on exit
        assert injector._observer is None


class TestClaudeMdFileHandler:
    """Test ClaudeMdFileHandler file system event handler."""

    def test_on_modified_claude_md(self):
        """Test handling CLAUDE.md modification."""
        from common.core.context_injection.claude_md_injector import (
            ClaudeMdFileHandler,
        )

        callback = Mock()
        handler = ClaudeMdFileHandler(callback)

        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = "/test/CLAUDE.md"

        handler.on_modified(event)

        # Should call callback after debounce
        time.sleep(0.1)  # Short sleep to allow processing
        assert callback.called

    def test_on_modified_other_file(self):
        """Test ignoring non-CLAUDE.md modifications."""
        from common.core.context_injection.claude_md_injector import (
            ClaudeMdFileHandler,
        )

        callback = Mock()
        handler = ClaudeMdFileHandler(callback)

        event = Mock()
        event.is_directory = False
        event.src_path = "/test/other.md"

        handler.on_modified(event)

        callback.assert_not_called()

    def test_on_created_claude_md(self):
        """Test handling CLAUDE.md creation."""
        from common.core.context_injection.claude_md_injector import (
            ClaudeMdFileHandler,
        )

        callback = Mock()
        handler = ClaudeMdFileHandler(callback)

        event = Mock()
        event.is_directory = False
        event.src_path = "/test/CLAUDE.md"

        handler.on_created(event)

        time.sleep(0.1)
        assert callback.called

    def test_debouncing(self):
        """Test debouncing rapid file changes."""
        from common.core.context_injection.claude_md_injector import (
            ClaudeMdFileHandler,
        )

        callback = Mock()
        handler = ClaudeMdFileHandler(callback)

        event = Mock()
        event.is_directory = False
        event.src_path = "/test/CLAUDE.md"

        # Rapid changes
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)

        # Should only call once due to debouncing
        time.sleep(0.1)
        assert callback.call_count == 1


class TestConvenienceFunction:
    """Test convenience function for injection."""

    @pytest.mark.asyncio
    async def test_inject_claude_md_content(
        self, temp_project_dir, mock_memory_manager, sample_memory_rules
    ):
        """Test convenience function."""
        mock_result = RuleRetrievalResult(
            rules=sample_memory_rules,
            total_count=2,
            filtered_count=2,
            cache_hit=False,
        )

        with patch(
            "common.core.context_injection.claude_md_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = AsyncMock()
            mock_retrieval.get_rules = AsyncMock(return_value=mock_result)
            mock_retrieval_class.return_value = mock_retrieval

            content = await inject_claude_md_content(
                memory_manager=mock_memory_manager, project_root=temp_project_dir
            )

        assert len(content) > 0
        assert "Project Instructions" in content


class TestPrecedenceRules:
    """Test precedence rules for multiple CLAUDE.md files."""

    def test_project_overrides_global(self, temp_project_dir, temp_global_claude_md):
        """Test that project CLAUDE.md takes precedence over global."""
        injector = ClaudeMdInjector(AsyncMock())

        locations = injector.discover_claude_md_files(temp_project_dir)

        # Project should be first
        assert locations[0].path == temp_project_dir / "CLAUDE.md"
        assert locations[0].precedence > locations[1].precedence

    @pytest.mark.asyncio
    async def test_inject_uses_highest_precedence(
        self, temp_project_dir, temp_global_claude_md, mock_memory_manager
    ):
        """Test that injection uses highest precedence file."""
        injector = ClaudeMdInjector(mock_memory_manager)

        mock_result = RuleRetrievalResult(
            rules=[], total_count=0, filtered_count=0, cache_hit=False
        )

        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            content = await injector.inject_from_files(project_root=temp_project_dir)

        # Should use project CLAUDE.md (not global)
        assert "Project Instructions" in content
        assert "This is the project CLAUDE.md" in content
        assert "Global Instructions" not in content


class TestIntegrationWithClaudeCodeDetector:
    """Test integration with ClaudeCodeDetector."""

    @pytest.mark.asyncio
    async def test_inject_in_claude_code_session(
        self, temp_project_dir, mock_memory_manager, sample_memory_rules
    ):
        """Test injection within detected Claude Code session."""
        from common.core.context_injection import ClaudeCodeDetector

        mock_result = RuleRetrievalResult(
            rules=sample_memory_rules,
            total_count=2,
            filtered_count=2,
            cache_hit=False,
        )

        injector = ClaudeMdInjector(mock_memory_manager)

        with patch.object(
            injector.rule_retrieval, "get_rules", return_value=mock_result
        ):
            # Simulate Claude Code session
            with patch.object(ClaudeCodeDetector, "is_active", return_value=True):
                content = await injector.inject_from_files(
                    project_root=temp_project_dir
                )

        assert len(content) > 0
        assert "Project Instructions" in content
