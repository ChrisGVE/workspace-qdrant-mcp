"""Tests for the init command (shell completion)."""

import pytest
from typer.testing import CliRunner
from wqm_cli.cli.commands.init import init_app


class TestInitCommand:
    """Test the shell completion init command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_init_help(self):
        """Test that init command shows help."""
        result = self.runner.invoke(init_app, ["--help"])
        assert result.exit_code == 0
        # Help text describes the command's purpose
        assert "shell completion" in result.stdout.lower()
        assert "bash" in result.stdout
        assert "zsh" in result.stdout
        assert "fish" in result.stdout

    def test_bash_completion_generation(self):
        """Test bash completion script generation."""
        result = self.runner.invoke(init_app, ["bash"])
        assert result.exit_code == 0

        # Check that it generates a bash completion script
        output = result.stdout
        assert "_wqm_completion()" in output
        assert "COMP_WORDS" in output
        assert "COMPREPLY" in output

        # Ensure it's properly formatted (no trailing newlines etc.)
        assert output.endswith("complete -o default -F _wqm_completion wqm")

    def test_zsh_completion_generation(self):
        """Test zsh completion script generation."""
        result = self.runner.invoke(init_app, ["zsh"])
        assert result.exit_code == 0

        # Check that it generates a zsh completion script
        output = result.stdout
        assert "#compdef wqm" in output
        assert "_wqm_completion()" in output
        assert "compdef _wqm_completion wqm" in output

    def test_fish_completion_generation(self):
        """Test fish completion script generation."""
        result = self.runner.invoke(init_app, ["fish"])
        assert result.exit_code == 0

        # Check that it generates a fish completion script
        output = result.stdout
        assert "complete --command wqm" in output
        assert "_WQM_COMPLETE=complete_fish" in output

    def test_completion_with_custom_prog_name(self):
        """Test completion generation with custom program name."""
        result = self.runner.invoke(init_app, ["bash", "--prog-name", "my-wqm"])
        assert result.exit_code == 0

        output = result.stdout
        assert "_MY-WQM_COMPLETE" in output  # Note: Typer uses the exact prog_name format
        assert "complete -o default -F _my_wqm_completion my-wqm" in output

    def test_detailed_help_command(self):
        """Test the detailed help command."""
        result = self.runner.invoke(init_app, ["help"])
        assert result.exit_code == 0

        output = result.stdout
        assert "Shell Completion Setup for wqm" in output
        assert 'eval "$(wqm init bash)"' in output
        assert "Permanent installation:" in output
        assert "TROUBLESHOOTING:" in output

    def test_no_args_shows_help(self):
        """Test that running init with no arguments shows help."""
        result = self.runner.invoke(init_app, [])
        # Shows usage help (may exit 0 with helpful message or non-zero)
        # Either way, should show available shells
        help_output = result.stdout + (result.stderr or "")
        assert "bash" in help_output

    def test_all_shells_supported(self):
        """Test that all expected shells are supported."""
        shells = ["bash", "zsh", "fish"]

        for shell in shells:
            result = self.runner.invoke(init_app, [shell])
            assert result.exit_code == 0, f"Failed to generate completion for {shell}"
            assert len(result.stdout) > 0, f"Empty output for {shell}"

    def test_completion_scripts_are_valid_format(self):
        """Test that generated completion scripts have valid format."""
        # Test bash
        result = self.runner.invoke(init_app, ["bash"])
        bash_script = result.stdout
        assert bash_script.count("_wqm_completion()") == 1
        assert "complete" in bash_script

        # Test zsh
        result = self.runner.invoke(init_app, ["zsh"])
        zsh_script = result.stdout
        assert zsh_script.startswith("#compdef wqm")
        assert zsh_script.count("_wqm_completion()") == 1

        # Test fish
        result = self.runner.invoke(init_app, ["fish"])
        fish_script = result.stdout
        assert fish_script.startswith("complete --command wqm")

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_scripts_contain_wqm_references(self, shell):
        """Test that completion scripts contain proper wqm references."""
        result = self.runner.invoke(init_app, [shell])
        output = result.stdout

        # All scripts should reference wqm and the completion environment variable
        assert "wqm" in output
        assert "_WQM_COMPLETE" in output
