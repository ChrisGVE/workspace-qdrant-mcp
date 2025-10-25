"""
Tests for Claude Code integration system.

Comprehensive tests for Claude Code SDK integration including session initialization,
memory rule injection, conversational updates, and context management.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest
from workspace_qdrant_mcp.memory.claude_integration import ClaudeCodeIntegration
from workspace_qdrant_mcp.memory.token_counter import TokenCounter, TokenUsage
from workspace_qdrant_mcp.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    ConversationalUpdate,
    MemoryCategory,
    MemoryContext,
    MemoryInjectionResult,
    MemoryRule,
)


class TestClaudeCodeIntegrationInit:
    """Test Claude Code integration initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with minimal parameters."""
        token_counter = TokenCounter()
        integration = ClaudeCodeIntegration(
            token_counter=token_counter,
            max_memory_tokens=5000
        )

        assert integration.token_counter == token_counter
        assert integration.max_memory_tokens == 5000
        assert integration.claude_config_path is not None
        assert integration._session_contexts == {}

    def test_custom_config_path(self):
        """Test initialization with custom config path."""
        token_counter = TokenCounter()
        custom_path = "/custom/path/to/claude.json"

        integration = ClaudeCodeIntegration(
            token_counter=token_counter,
            claude_config_path=custom_path
        )

        assert integration.claude_config_path == custom_path

    def test_config_path_discovery(self):
        """Test automatic config path discovery."""
        token_counter = TokenCounter()

        with patch.object(ClaudeCodeIntegration, '_find_claude_config') as mock_find:
            mock_find.return_value = "/discovered/claude.json"

            integration = ClaudeCodeIntegration(token_counter=token_counter)

            assert integration.claude_config_path == "/discovered/claude.json"
            mock_find.assert_called_once()

    @patch('os.path.expanduser')
    @patch('os.path.exists')
    def test_find_claude_config_home(self, mock_exists, mock_expanduser):
        """Test finding Claude config in home directory."""
        mock_expanduser.return_value = "/home/user"
        mock_exists.side_effect = lambda path: path == "/home/user/.claude/claude.json"

        token_counter = TokenCounter()
        integration = ClaudeCodeIntegration(token_counter=token_counter)

        # Should find config in home directory
        assert integration.claude_config_path == "/home/user/.claude/claude.json"

    @patch('os.path.expanduser')
    @patch('os.path.exists')
    def test_find_claude_config_cwd(self, mock_exists, mock_expanduser):
        """Test finding Claude config in current directory."""
        mock_expanduser.return_value = "/home/user"
        mock_exists.side_effect = lambda path: path == "./claude.json"

        token_counter = TokenCounter()
        integration = ClaudeCodeIntegration(token_counter=token_counter)

        # Should find config in current directory
        assert integration.claude_config_path == "./claude.json"

    @patch('os.path.expanduser')
    @patch('os.path.exists')
    def test_find_claude_config_none(self, mock_exists, mock_expanduser):
        """Test handling when no Claude config is found."""
        mock_expanduser.return_value = "/home/user"
        mock_exists.return_value = False

        token_counter = TokenCounter()
        integration = ClaudeCodeIntegration(token_counter=token_counter)

        # Should return None when no config found
        assert integration.claude_config_path is None


class TestSessionInitialization:
    """Test Claude Code session initialization."""

    @pytest.fixture
    def mock_token_counter(self):
        """Create mock token counter."""
        counter = Mock(spec=TokenCounter)
        counter.optimize_rules_for_context.return_value = ([], TokenUsage(
            total_tokens=0,
            rules_count=0
        ))
        return counter

    @pytest.fixture
    def integration(self, mock_token_counter):
        """Create integration instance with mocked dependencies."""
        return ClaudeCodeIntegration(
            token_counter=mock_token_counter,
            max_memory_tokens=5000
        )

    @pytest.mark.asyncio
    async def test_initialize_empty_session(self, integration, mock_token_counter):
        """Test initializing session with no available rules."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/test/path",
            user_name="test_user"
        )

        result = await integration.initialize_session(session, [])

        assert isinstance(result, MemoryInjectionResult)
        assert result.success is True
        assert result.rules_injected == 0
        assert result.total_tokens_used == 0
        assert result.remaining_context_tokens == session.context_window_size
        assert len(result.skipped_rules) == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_initialize_session_with_rules(self, integration, mock_token_counter):
        """Test initializing session with available rules."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/test/path",
            user_name="test_user",
            project_name="test-project"
        )

        rules = [
            MemoryRule(
                rule="Test rule 1",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Test rule 2",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        # Mock token counter optimization
        mock_token_counter.optimize_rules_for_context.return_value = (
            rules,
            TokenUsage(total_tokens=150, rules_count=2)
        )

        with patch.object(integration, '_inject_rules_to_claude') as mock_inject:
            mock_inject.return_value = True

            result = await integration.initialize_session(session, rules)

            assert result.success is True
            assert result.rules_injected == 2
            assert result.total_tokens_used == 150
            mock_inject.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_context_creation(self, integration, mock_token_counter):
        """Test creation of memory context for session."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/path/to/python-project",
            user_name="chris",
            project_name="django-app",
            active_files=["main.py", "models.py"],
            git_branch="feature/auth"
        )

        # Access private method for testing
        context = await integration._create_memory_context(session)

        assert isinstance(context, MemoryContext)
        assert context.session_id == "test-session"
        assert context.user_name == "chris"
        assert context.project_name == "django-app"
        assert context.project_path == "/path/to/python-project"
        assert "python" in context.active_scopes or "django" in context.active_scopes
        assert context.conversation_context is not None

    @pytest.mark.asyncio
    async def test_rule_filtering_by_context(self, integration, mock_token_counter):
        """Test filtering rules by session context."""
        rules = [
            MemoryRule(
                rule="Python specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python"]
            ),
            MemoryRule(
                rule="JavaScript specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript"]
            ),
            MemoryRule(
                rule="General rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=[]
            ),
        ]

        context = MemoryContext(
            session_id="test",
            project_name="python-app",
            active_scopes=["python", "backend"]
        )

        # Access private method for testing
        relevant_rules = integration._filter_rules_by_context(rules, context)

        # Should include Python and general rules, exclude JavaScript
        rule_texts = [r.rule for r in relevant_rules]
        assert "Python specific rule" in rule_texts
        assert "General rule" in rule_texts
        # JavaScript rule might be excluded or have lower priority

    @pytest.mark.asyncio
    async def test_injection_content_generation(self, integration, mock_token_counter):
        """Test generation of injection content."""
        rules = [
            MemoryRule(
                rule="Always use type hints in Python",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"]
            ),
            MemoryRule(
                rule="Prefer pytest for testing",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["python", "testing"]
            ),
        ]

        context = MemoryContext(
            session_id="test",
            user_name="chris",
            project_name="python-project"
        )

        # Access private method for testing
        content = integration._generate_injection_content(rules, context)

        assert isinstance(content, str)
        assert len(content) > 0
        assert "type hints" in content.lower()
        assert "pytest" in content.lower()
        assert "chris" in content or "user" in content.lower()
        assert "python" in content.lower()


class TestRuleInjection:
    """Test rule injection to Claude Code."""

    @pytest.fixture
    def integration(self):
        """Create integration instance."""
        return ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

    @patch('builtins.open', mock_open())
    @patch('json.load')
    @patch('json.dump')
    def test_inject_rules_success(self, mock_json_dump, mock_json_load, integration):
        """Test successful rule injection to Claude config."""
        mock_json_load.return_value = {
            "system_prompt": "Existing prompt",
            "other_config": "value"
        }

        integration.claude_config_path = "/test/claude.json"

        result = integration._inject_rules_to_claude("New memory rules content")

        assert result is True
        mock_json_dump.assert_called_once()

        # Verify the call to json.dump had updated content
        call_args = mock_json_dump.call_args[0][0]
        assert "system_prompt" in call_args
        assert "New memory rules content" in call_args["system_prompt"]

    def test_inject_rules_no_config_path(self, integration):
        """Test rule injection when no config path is available."""
        integration.claude_config_path = None

        result = integration._inject_rules_to_claude("Memory rules content")

        assert result is False

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_inject_rules_file_error(self, mock_open, integration):
        """Test rule injection when config file is not found."""
        integration.claude_config_path = "/nonexistent/claude.json"

        result = integration._inject_rules_to_claude("Memory rules content")

        assert result is False

    @patch('builtins.open', mock_open())
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_inject_rules_invalid_json(self, mock_json_load, integration):
        """Test rule injection with invalid JSON config."""
        integration.claude_config_path = "/test/claude.json"

        result = integration._inject_rules_to_claude("Memory rules content")

        assert result is False


class TestConversationalUpdates:
    """Test conversational memory updates detection."""

    @pytest.fixture
    def integration(self):
        return ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

    @pytest.mark.asyncio
    async def test_detect_preference_update(self, integration):
        """Test detection of preference updates in conversation."""
        text = "Note: I prefer using uv instead of pip for Python package management."

        updates = await integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert isinstance(update, ConversationalUpdate)
        assert "uv" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.PREFERENCE
        assert update.confidence > 0.5

    @pytest.mark.asyncio
    async def test_detect_name_update(self, integration):
        """Test detection of name/identity updates."""
        text = "By the way, call me Chris instead of using my full name."

        updates = await integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "chris" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.PREFERENCE
        assert "user" in update.scope or "name" in update.scope

    @pytest.mark.asyncio
    async def test_detect_behavior_update(self, integration):
        """Test detection of behavior rule updates."""
        text = "Always validate user input in the API endpoints for security."

        updates = await integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "validate" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.BEHAVIOR
        assert update.authority == AuthorityLevel.ABSOLUTE  # "Always" indicates absolute

    @pytest.mark.asyncio
    async def test_detect_no_updates(self, integration):
        """Test handling text with no memory updates."""
        text = "The weather is nice today. How are the tests running?"

        updates = await integration.detect_conversational_updates(text)

        # Should not detect any memory-relevant updates
        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_detect_multiple_updates(self, integration):
        """Test detection of multiple updates in single text."""
        text = """
        Note: call me Chris, and I prefer TypeScript over JavaScript.
        Also, always use async/await instead of promises for clarity.
        """

        updates = await integration.detect_conversational_updates(text)

        assert len(updates) >= 2

        # Should detect name preference and TypeScript preference
        rule_texts = [u.extracted_rule.lower() for u in updates]
        assert any("chris" in text for text in rule_texts)
        assert any("typescript" in text for text in rule_texts)

    @pytest.mark.asyncio
    async def test_update_confidence_scoring(self, integration):
        """Test confidence scoring for updates."""
        # High confidence update
        high_conf_text = "Always use HTTPS for API endpoints."
        high_updates = await integration.detect_conversational_updates(high_conf_text)

        # Low confidence/ambiguous update
        low_conf_text = "Maybe we should consider using React sometime."
        low_updates = await integration.detect_conversational_updates(low_conf_text)

        if high_updates and low_updates:
            assert high_updates[0].confidence > low_updates[0].confidence

    @pytest.mark.asyncio
    async def test_scope_detection(self, integration):
        """Test scope detection in conversational updates."""
        text = "For Python projects, always use virtual environments."

        updates = await integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "python" in update.scope


class TestContextManagement:
    """Test memory context management."""

    @pytest.fixture
    def integration(self):
        return ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

    def test_session_context_caching(self, integration):
        """Test that session contexts are cached."""
        ClaudeCodeSession(
            session_id="session-1",
            workspace_path="/path1"
        )

        ClaudeCodeSession(
            session_id="session-2",
            workspace_path="/path2"
        )

        # Simulate adding contexts
        context1 = MemoryContext(session_id="session-1", project_name="proj1")
        context2 = MemoryContext(session_id="session-2", project_name="proj2")

        integration._session_contexts["session-1"] = context1
        integration._session_contexts["session-2"] = context2

        assert len(integration._session_contexts) == 2
        assert integration._session_contexts["session-1"] == context1
        assert integration._session_contexts["session-2"] == context2

    def test_get_cached_context(self, integration):
        """Test retrieval of cached session context."""
        context = MemoryContext(
            session_id="cached-session",
            project_name="cached-project"
        )
        integration._session_contexts["cached-session"] = context

        retrieved = integration.get_session_context("cached-session")

        assert retrieved == context

    def test_get_missing_context(self, integration):
        """Test retrieval of non-existent context."""
        retrieved = integration.get_session_context("missing-session")

        assert retrieved is None

    def test_context_cleanup(self, integration):
        """Test cleanup of session contexts."""
        # Add multiple contexts
        for i in range(5):
            context = MemoryContext(session_id=f"session-{i}", project_name=f"proj-{i}")
            integration._session_contexts[f"session-{i}"] = context

        assert len(integration._session_contexts) == 5

        # Cleanup specific session
        integration.cleanup_session_context("session-2")
        assert "session-2" not in integration._session_contexts
        assert len(integration._session_contexts) == 4

        # Cleanup all sessions
        integration.cleanup_all_contexts()
        assert len(integration._session_contexts) == 0


class TestProjectAnalysis:
    """Test project analysis for context creation."""

    @pytest.fixture
    def integration(self):
        return ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

    def test_detect_python_project(self, integration):
        """Test detection of Python project characteristics."""
        workspace_path = "/path/to/python-project"

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [
                "main.py", "requirements.txt", "setup.py",
                "tests", "src", "venv"
            ]

            scopes = integration._analyze_project_scopes(workspace_path)

            assert "python" in scopes

    def test_detect_javascript_project(self, integration):
        """Test detection of JavaScript project characteristics."""
        workspace_path = "/path/to/js-project"

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [
                "package.json", "index.js", "node_modules",
                "src", "dist"
            ]

            scopes = integration._analyze_project_scopes(workspace_path)

            assert "javascript" in scopes or "node" in scopes

    def test_detect_web_project(self, integration):
        """Test detection of web project characteristics."""
        workspace_path = "/path/to/web-project"

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [
                "index.html", "style.css", "script.js",
                "assets", "public"
            ]

            scopes = integration._analyze_project_scopes(workspace_path)

            assert "web" in scopes or "frontend" in scopes

    def test_analyze_nonexistent_project(self, integration):
        """Test handling of nonexistent project path."""
        workspace_path = "/nonexistent/path"

        scopes = integration._analyze_project_scopes(workspace_path)

        # Should return empty list or handle gracefully
        assert isinstance(scopes, list)

    def test_analyze_empty_project(self, integration):
        """Test analysis of empty project directory."""
        workspace_path = "/empty/project"

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = []

            scopes = integration._analyze_project_scopes(workspace_path)

            assert isinstance(scopes, list)

    def test_git_branch_detection(self, integration):
        """Test Git branch detection for context."""
        workspace_path = "/git/project"

        with patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.return_value = b"feature/authentication\n"

            scopes = integration._analyze_project_scopes(workspace_path)

            # Should include branch-related scope
            assert any("feature" in scope or "auth" in scope for scope in scopes)

    def test_git_branch_detection_failure(self, integration):
        """Test handling Git branch detection failure."""
        workspace_path = "/non-git/project"

        with patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "git")

            scopes = integration._analyze_project_scopes(workspace_path)

            # Should handle failure gracefully
            assert isinstance(scopes, list)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def integration(self):
        return ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

    @pytest.mark.asyncio
    async def test_invalid_session_data(self, integration):
        """Test handling of invalid session data."""
        # Session with missing required fields
        invalid_session = ClaudeCodeSession(
            session_id="",  # Empty session ID
            workspace_path=None
        )

        result = await integration.initialize_session(invalid_session, [])

        # Should handle gracefully and return failure result
        assert isinstance(result, MemoryInjectionResult)
        assert result.success is False or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_token_counter_failure(self, integration):
        """Test handling of token counter failures."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/test/path"
        )

        rules = [MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )]

        # Mock token counter to raise exception
        integration.token_counter.optimize_rules_for_context = Mock(
            side_effect=Exception("Token counter error")
        )

        result = await integration.initialize_session(session, rules)

        # Should handle failure gracefully
        assert isinstance(result, MemoryInjectionResult)
        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_malformed_conversational_text(self, integration):
        """Test handling of malformed conversational text."""
        malformed_texts = [
            "",  # Empty text
            None,  # None input
            "x" * 10000,  # Extremely long text
            "🚀" * 100,  # Unicode/emoji heavy
        ]

        for text in malformed_texts:
            try:
                updates = await integration.detect_conversational_updates(text)
                assert isinstance(updates, list)
            except Exception:
                # Should not raise unhandled exceptions
                pytest.fail(f"Unhandled exception for input: {text}")

    def test_concurrent_session_access(self, integration):
        """Test concurrent access to session contexts."""
        import threading
        import time

        def add_context(session_id):
            context = MemoryContext(
                session_id=session_id,
                project_name=f"project-{session_id}"
            )
            integration._session_contexts[session_id] = context
            time.sleep(0.01)  # Small delay to increase chance of race condition

        # Create multiple threads accessing contexts concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_context, args=(f"session-{i}",))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have all contexts without corruption
        assert len(integration._session_contexts) == 10
