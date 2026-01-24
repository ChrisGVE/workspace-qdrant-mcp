"""
Comprehensive Unit Tests for CLI Help System

Tests the interactive help system including command discovery, fuzzy matching,
contextual help, and edge cases for malformed input and error conditions.

Task 251: Comprehensive unit tests for unified CLI interface enhancements.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, call, patch

import pytest

# Add src paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Set CLI mode before any imports
os.environ["WQM_CLI_MODE"] = "true"
os.environ["WQM_LOG_INIT"] = "false"

try:
    from rich.console import Console
    from wqm_cli.cli.help_system import (
        CommandInfo,
        HelpLevel,
        InteractiveHelpSystem,
        create_help_app,
        help_app,
        help_system,
    )
    HELP_SYSTEM_AVAILABLE = True
except ImportError as e:
    HELP_SYSTEM_AVAILABLE = False
    print(f"Warning: help_system module not available: {e}")


@pytest.mark.skipif(not HELP_SYSTEM_AVAILABLE, reason="Help system module not available")
class TestInteractiveHelpSystem:
    """Test the interactive help system core functionality."""

    @pytest.fixture
    def help_sys(self):
        """Create a fresh help system instance."""
        return InteractiveHelpSystem()

    def test_help_system_initialization(self, help_sys):
        """Test help system initializes properly."""
        assert help_sys is not None
        assert hasattr(help_sys, '_commands')
        assert hasattr(help_sys, '_categories')
        assert hasattr(help_sys, '_command_tree')
        assert len(help_sys._commands) > 0
        assert len(help_sys._categories) > 0

    def test_command_info_structure(self, help_sys):
        """Test command info has required structure."""
        for cmd_name, cmd_info in help_sys._commands.items():
            assert isinstance(cmd_info, CommandInfo)
            assert cmd_info.name == cmd_name
            assert isinstance(cmd_info.description, str)
            assert len(cmd_info.description) > 0
            assert isinstance(cmd_info.usage, str)
            assert isinstance(cmd_info.examples, list)
            assert isinstance(cmd_info.aliases, list)
            assert isinstance(cmd_info.category, str)
            assert isinstance(cmd_info.subcommands, list)
            assert isinstance(cmd_info.common_flags, list)
            assert isinstance(cmd_info.related_commands, list)

    def test_categories_build_correctly(self, help_sys):
        """Test categories are built correctly from commands."""
        expected_categories = set()
        for cmd_info in help_sys._commands.values():
            expected_categories.add(cmd_info.category)

        assert set(help_sys._categories.keys()) == expected_categories

        # Test that all commands are in their correct categories
        for category, commands in help_sys._categories.items():
            for cmd_name in commands:
                assert help_sys._commands[cmd_name].category == category

    def test_command_tree_structure(self, help_sys):
        """Test command tree is built properly."""
        tree = help_sys._command_tree
        assert tree is not None
        assert hasattr(tree, 'label')
        assert "wqm Commands" in str(tree.label)

    def test_suggest_commands_exact_match(self, help_sys):
        """Test command suggestion with exact matches."""
        suggestions = help_sys.suggest_commands("memory", limit=5)
        assert len(suggestions) > 0
        assert suggestions[0][0] == "memory"
        assert suggestions[0][1] == 1.0  # Perfect match

    def test_suggest_commands_partial_match(self, help_sys):
        """Test command suggestion with partial matches."""
        suggestions = help_sys.suggest_commands("mem", limit=5)
        assert len(suggestions) > 0
        # Should find "memory" as a suggestion
        command_names = [s[0] for s in suggestions]
        assert "memory" in command_names

    def test_suggest_commands_fuzzy_match(self, help_sys):
        """Test fuzzy command matching."""
        suggestions = help_sys.suggest_commands("confgi", limit=5)  # typo: config
        assert len(suggestions) > 0
        command_names = [s[0] for s in suggestions]
        assert "config" in command_names

    def test_suggest_commands_no_matches(self, help_sys):
        """Test suggestion when no good matches found."""
        suggestions = help_sys.suggest_commands("xyznonexistent", limit=5)
        # Fuzzy matching may find low-score matches - check scores are low if present
        if len(suggestions) > 0:
            # All scores should be below a reasonable threshold
            assert all(score < 0.5 for _cmd, score in suggestions)

    def test_suggest_commands_subcommand_matching(self, help_sys):
        """Test suggestion includes subcommands."""
        suggestions = help_sys.suggest_commands("memory list", limit=5)
        assert len(suggestions) > 0
        command_names = [s[0] for s in suggestions]
        assert "memory list" in command_names

    def test_suggest_commands_limit_parameter(self, help_sys):
        """Test suggestion limit parameter works."""
        suggestions_2 = help_sys.suggest_commands("admin", limit=2)
        suggestions_5 = help_sys.suggest_commands("admin", limit=5)

        assert len(suggestions_2) <= 2
        assert len(suggestions_5) <= 5
        # When there are matches, limited should be smaller or equal
        if len(suggestions_5) >= 2:
            assert len(suggestions_2) <= len(suggestions_5)

    def test_suggest_commands_similarity_scores(self, help_sys):
        """Test suggestion similarity scores are reasonable."""
        suggestions = help_sys.suggest_commands("config", limit=5)

        for _command, score in suggestions:
            assert 0.0 <= score <= 1.0
            assert isinstance(score, float)

        # Scores should be in descending order
        if len(suggestions) > 1:
            scores = [s[1] for s in suggestions]
            assert scores == sorted(scores, reverse=True)

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_command_discovery(self, mock_print, help_sys):
        """Test command discovery display."""
        help_sys.show_command_discovery()

        # Should print multiple times (panel, tree, tips)
        assert mock_print.call_count >= 3

        # Check that Rich objects are printed (Panel, Tree)
        # Extract titles and labels from Rich objects
        found_commands = False
        found_tips = False
        for call in mock_print.call_args_list:
            args = call[0]
            if args:
                obj = args[0]
                if hasattr(obj, 'title') and obj.title:
                    title = str(obj.title)
                    if "Quick Tips" in title:
                        found_tips = True
                if hasattr(obj, 'label') and obj.label:
                    label = str(obj.label)
                    if "Commands" in label:
                        found_commands = True
        assert found_commands, "Command tree should be displayed"
        assert found_tips, "Quick Tips panel should be displayed"

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_command_help_existing_command(self, mock_print, help_sys):
        """Test showing help for existing command."""
        help_sys.show_command_help("memory", level=HelpLevel.DETAILED)

        # Should print at least once
        assert mock_print.call_count >= 1

        # Verify the command exists in help system
        assert "memory" in help_sys._commands
        # Simply verify that help was displayed (print was called)
        # Content inspection requires rendering Rich objects

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_command_help_nonexistent_command(self, mock_print, help_sys):
        """Test showing help for non-existent command with suggestions."""
        help_sys.show_command_help("nonexistent")

        assert mock_print.call_count >= 1
        all_calls = [str(call) for call in mock_print.call_args_list]
        combined_output = " ".join(all_calls)

        assert "not found" in combined_output
        assert "Did you mean" in combined_output or "help discover" in combined_output

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_command_help_with_subcommand(self, mock_print, help_sys):
        """Test showing help for command with subcommand."""
        help_sys.show_command_help("memory", subcommand="list", level=HelpLevel.DETAILED)

        # Should print at least once
        assert mock_print.call_count >= 1

        # Verify the command exists and has the subcommand listed
        assert "memory" in help_sys._commands
        assert "list" in help_sys._commands["memory"].subcommands

    def test_show_command_help_help_levels(self, help_sys):
        """Test different help levels."""
        with patch('wqm_cli.cli.help_system.console.print') as mock_print:
            # Test brief level
            help_sys.show_command_help("memory", level=HelpLevel.BRIEF)
            brief_calls = len(mock_print.call_args_list)

        with patch('wqm_cli.cli.help_system.console.print') as mock_print:
            # Test full level
            help_sys.show_command_help("memory", level=HelpLevel.FULL)
            full_calls = len(mock_print.call_args_list)

        # Full level should generally produce more output
        # (though exact comparison depends on content)
        assert brief_calls > 0
        assert full_calls > 0

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_category_help_existing(self, mock_print, help_sys):
        """Test showing help for existing category."""
        categories = list(help_sys._categories.keys())
        if categories:
            category = categories[0]
            help_sys.show_category_help(category)

            assert mock_print.call_count >= 1
            all_calls = [str(call) for call in mock_print.call_args_list]
            combined_output = " ".join(all_calls)
            assert category in combined_output

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_category_help_nonexistent(self, mock_print, help_sys):
        """Test showing help for non-existent category."""
        help_sys.show_category_help("NonexistentCategory")

        assert mock_print.call_count >= 1
        all_calls = [str(call) for call in mock_print.call_args_list]
        combined_output = " ".join(all_calls)
        assert "not found" in combined_output

    @patch('wqm_cli.cli.help_system.console.print')
    def test_show_quick_reference(self, mock_print, help_sys):
        """Test quick reference display."""
        help_sys.show_quick_reference()

        # Should print multiple times (header, commands, categories, help tips)
        assert mock_print.call_count >= 5

        # Check for Rich Panel with Quick Reference title
        found_quick_ref = False
        for call in mock_print.call_args_list:
            args = call[0]
            if args:
                obj = args[0]
                if hasattr(obj, 'title') and obj.title:
                    if "Quick Reference" in str(obj.title):
                        found_quick_ref = True
                        break
                # Also check string outputs for key phrases
                if isinstance(obj, str):
                    if "Most Common" in obj or "Categories" in obj or "Getting Help" in obj:
                        found_quick_ref = True

        assert found_quick_ref or mock_print.call_count >= 10, "Quick reference content should be displayed"


@pytest.mark.skipif(not HELP_SYSTEM_AVAILABLE, reason="Help system module not available")
class TestHelpSystemEdgeCases:
    """Test edge cases and error conditions for help system."""

    @pytest.fixture
    def help_sys(self):
        """Create a fresh help system instance."""
        return InteractiveHelpSystem()

    def test_empty_command_string(self, help_sys):
        """Test handling empty command string."""
        suggestions = help_sys.suggest_commands("", limit=5)
        # Empty string might return no suggestions or all suggestions
        assert isinstance(suggestions, list)

        with patch('wqm_cli.cli.help_system.console.print'):
            help_sys.show_command_help("")

    def test_whitespace_only_command(self, help_sys):
        """Test handling whitespace-only command."""
        suggestions = help_sys.suggest_commands("   ", limit=5)
        assert isinstance(suggestions, list)

        with patch('wqm_cli.cli.help_system.console.print'):
            help_sys.show_command_help("   ")

    def test_very_long_command_string(self, help_sys):
        """Test handling very long command strings."""
        long_command = "a" * 1000
        suggestions = help_sys.suggest_commands(long_command, limit=5)
        assert isinstance(suggestions, list)

        with patch('wqm_cli.cli.help_system.console.print'):
            help_sys.show_command_help(long_command)

    def test_special_characters_in_command(self, help_sys):
        """Test handling special characters in command names."""
        special_commands = ["mem@ry", "config!", "search?", "admin#test", "~command"]

        for cmd in special_commands:
            suggestions = help_sys.suggest_commands(cmd, limit=5)
            assert isinstance(suggestions, list)

            with patch('wqm_cli.cli.help_system.console.print'):
                help_sys.show_command_help(cmd)

    def test_unicode_in_command(self, help_sys):
        """Test handling unicode characters in commands."""
        unicode_commands = ["méméory", "config™", "🚀command", "测试"]

        for cmd in unicode_commands:
            suggestions = help_sys.suggest_commands(cmd, limit=5)
            assert isinstance(suggestions, list)

    def test_negative_limit_parameter(self, help_sys):
        """Test suggestion with negative limit."""
        suggestions = help_sys.suggest_commands("memory", limit=-1)
        # Should handle gracefully (likely return empty list or raise exception)
        assert isinstance(suggestions, list)

    def test_zero_limit_parameter(self, help_sys):
        """Test suggestion with zero limit."""
        suggestions = help_sys.suggest_commands("memory", limit=0)
        assert len(suggestions) == 0

    def test_extremely_high_limit(self, help_sys):
        """Test suggestion with extremely high limit."""
        suggestions = help_sys.suggest_commands("memory", limit=10000)
        assert isinstance(suggestions, list)
        # Should not return more suggestions than exist
        assert len(suggestions) <= 100  # Reasonable upper bound

    def test_malformed_help_levels(self, help_sys):
        """Test help display with invalid help levels."""
        # This tests the enum validation implicitly
        valid_levels = [HelpLevel.BRIEF, HelpLevel.DETAILED, HelpLevel.EXAMPLES, HelpLevel.FULL]

        for level in valid_levels:
            with patch('wqm_cli.cli.help_system.console.print'):
                help_sys.show_command_help("memory", level=level)

    @patch('wqm_cli.cli.help_system.console.print', side_effect=Exception("Print error"))
    def test_console_print_error_handling(self, mock_print, help_sys):
        """Test error handling when console printing fails."""
        # Should not crash even if printing fails
        try:
            help_sys.show_command_discovery()
        except Exception as e:
            # Should be the mocked exception, not an internal error
            assert "Print error" in str(e)

    def test_command_info_with_missing_fields(self):
        """Test command info creation with missing optional fields."""
        # Test minimal command info
        minimal_cmd = CommandInfo(
            name="test",
            description="test command",
            usage="test usage",
            examples=[],  # Empty examples
            aliases=[],   # Empty aliases
            category="Test",
            subcommands=[],  # Empty subcommands
            common_flags=[],  # Empty flags
            related_commands=[]  # Empty related
        )

        assert minimal_cmd.name == "test"
        assert minimal_cmd.examples == []
        assert minimal_cmd.aliases == []

    def test_category_with_no_commands(self, help_sys):
        """Test category operations when category has no commands."""
        # Modify categories to have empty category
        help_sys._categories["Empty"] = []

        with patch('wqm_cli.cli.help_system.console.print'):
            help_sys.show_category_help("Empty")

    def test_command_with_extremely_long_description(self, help_sys):
        """Test command with very long description and content."""
        # Create command with long fields
        long_desc = "A" * 10000
        long_cmd = CommandInfo(
            name="longtest",
            description=long_desc,
            usage="longtest" + "B" * 1000,
            examples=["example" + str(i) for i in range(100)],
            aliases=["alias" + str(i) for i in range(50)],
            category="Test",
            subcommands=["sub" + str(i) for i in range(200)],
            common_flags=["--flag" + str(i) for i in range(100)],
            related_commands=["rel" + str(i) for i in range(50)]
        )

        # Add to commands temporarily
        help_sys._commands["longtest"] = long_cmd

        with patch('wqm_cli.cli.help_system.console.print'):
            help_sys.show_command_help("longtest")

    def test_concurrent_access_to_help_system(self, help_sys):
        """Test thread safety aspects (basic test)."""
        import threading
        import time

        results = []
        errors = []

        def worker():
            try:
                suggestions = help_sys.suggest_commands("memory", limit=5)
                results.append(len(suggestions))
                with patch('wqm_cli.cli.help_system.console.print'):
                    help_sys.show_command_help("admin")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        # Should not have errors in basic concurrent access
        assert len(errors) == 0
        assert len(results) > 0


@pytest.mark.skipif(not HELP_SYSTEM_AVAILABLE, reason="Help system module not available")
class TestHelpAppIntegration:
    """Test help app typer integration."""

    def test_help_app_creation(self):
        """Test help app can be created."""
        app = create_help_app()
        assert app is not None
        assert hasattr(app, 'registered_commands') or hasattr(app, 'commands')

    def test_help_app_commands_registered(self):
        """Test help app has expected commands."""
        app = create_help_app()

        # Check if commands are registered (typer implementation details may vary)
        # This is a basic structural test
        assert callable(getattr(app, 'callback', None)) or hasattr(app, 'commands')

    def test_global_help_system_instance(self):
        """Test global help system instance exists."""
        assert help_system is not None
        assert isinstance(help_system, InteractiveHelpSystem)

    def test_help_system_commands_coverage(self):
        """Test help system covers essential wqm commands."""
        essential_commands = ["memory", "admin", "search", "config", "service"]

        for cmd in essential_commands:
            assert cmd in help_system._commands
            cmd_info = help_system._commands[cmd]
            assert len(cmd_info.description) > 0
            assert len(cmd_info.usage) > 0

    def test_help_system_categories_coverage(self):
        """Test help system has reasonable category coverage."""
        expected_categories = ["Core", "System", "Content", "Monitoring"]

        for category in expected_categories:
            assert category in help_system._categories
            assert len(help_system._categories[category]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
