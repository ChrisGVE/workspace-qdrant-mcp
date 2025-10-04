"""Unit tests for grammar compiler error handling."""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from common.core.grammar_compiler import GrammarCompiler, CompilerInfo, CompilationResult
from common.core.grammar_dependencies import DependencyAnalysis


@pytest.fixture
def temp_grammar(tmp_path):
    """Create a temporary grammar directory structure."""
    grammar_dir = tmp_path / "tree-sitter-test"
    src_dir = grammar_dir / "src"
    src_dir.mkdir(parents=True)

    # Create parser.c
    parser_c = src_dir / "parser.c"
    parser_c.write_text("// parser code")

    return grammar_dir


@pytest.fixture
def mock_compiler():
    """Create a mock C compiler."""
    return CompilerInfo(
        name="gcc",
        path="/usr/bin/gcc",
        version="GCC 11.0.0",
        is_cpp=False
    )


@pytest.fixture
def compiler_with_mocks(mock_compiler):
    """Create a GrammarCompiler with mocked compilers."""
    return GrammarCompiler(c_compiler=mock_compiler, cpp_compiler=None)


class TestCompilationErrorHandling:
    """Tests for compilation error handling."""

    def test_compilation_timeout_error(self, compiler_with_mocks, temp_grammar):
        """Test handling of compilation timeout."""
        with patch("subprocess.run") as mock_run:
            # Simulate timeout
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["gcc"], timeout=120
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "timed out" in result.error.lower()
            assert "120 seconds" in result.error
            assert "system resource" in result.error.lower()

    def test_permission_error(self, compiler_with_mocks, temp_grammar):
        """Test handling of permission errors."""
        with patch("subprocess.run") as mock_run:
            # Simulate permission error
            mock_run.side_effect = PermissionError("Permission denied")

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "permission denied" in result.error.lower()
            assert "write permissions" in result.error.lower()

    def test_linker_error_detection(self, compiler_with_mocks, temp_grammar):
        """Test detection and handling of linker errors."""
        with patch("subprocess.run") as mock_run:
            # Simulate linker error
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr="undefined reference to `tree_sitter_test_external_scanner_create'"
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "linker error" in result.error.lower()
            assert "scanner implementation" in result.error.lower()

    def test_missing_include_error(self, compiler_with_mocks, temp_grammar):
        """Test detection of missing include files."""
        with patch("subprocess.run") as mock_run:
            # Simulate missing include
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr="fatal error: tree_sitter/parser.h: No such file or directory"
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "missing include" in result.error.lower()
            assert "tree-sitter generate" in result.error.lower()

    def test_syntax_error_in_generated_code(self, compiler_with_mocks, temp_grammar):
        """Test detection of syntax errors in generated code."""
        with patch("subprocess.run") as mock_run:
            # Simulate syntax error
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr="parser.c:100:5: error: expected ';' before 'return'"
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "syntax error" in result.error.lower()
            assert "regenerating grammar" in result.error.lower()

    def test_cpp_compiler_mismatch_detection(self, temp_grammar):
        """Test detection of C++ scanner without C++ compiler."""
        # Create C++ scanner
        scanner_cc = temp_grammar / "src" / "scanner.cc"
        scanner_cc.write_text("// C++ scanner")

        # Mock compiler detection to ensure no C++ compiler
        with patch("common.core.grammar_compiler.CompilerDetector") as MockDetector:
            mock_detector = MockDetector.return_value
            mock_detector.detect_c_compiler.return_value = CompilerInfo(
                name="gcc", path="/usr/bin/gcc", version="11.0.0"
            )
            mock_detector.detect_cpp_compiler.return_value = None

            compiler = GrammarCompiler()
            result = compiler.compile(temp_grammar)

            # Should fail validation before compilation
            assert not result.success
            assert "c++ scanner" in result.error.lower()
            assert "no c++ compiler" in result.error.lower()

    def test_error_message_truncation(self, compiler_with_mocks, temp_grammar):
        """Test that very long error messages are truncated."""
        with patch("subprocess.run") as mock_run:
            # Create very long error message
            long_error = "error: " * 500  # Over 1000 characters

            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr=long_error
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "(truncated)" in result.error

    def test_warning_extraction(self, compiler_with_mocks, temp_grammar):
        """Test extraction of compiler warnings from stderr."""
        with patch("subprocess.run") as mock_run:
            # Simulate compilation with warnings
            stderr_with_warnings = """
parser.c:50:10: warning: unused variable 'foo' [-Wunused-variable]
parser.c:100:5: warning: implicit declaration of function 'bar' [-Wimplicit-function-declaration]
error: compilation failed
"""
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr=stderr_with_warnings
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert len(result.warnings) == 2
            assert "unused variable" in result.warnings[0]
            assert "implicit declaration" in result.warnings[1]

    def test_generic_error_with_suggestions(self, compiler_with_mocks, temp_grammar):
        """Test that generic errors still provide helpful suggestions."""
        with patch("subprocess.run") as mock_run:
            # Generic compilation failure
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr="Something went wrong"
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "suggestions" in result.error.lower()
            assert "tree-sitter generate" in result.error.lower()

    def test_unexpected_exception_handling(self, compiler_with_mocks, temp_grammar):
        """Test handling of unexpected exceptions."""
        with patch("subprocess.run") as mock_run:
            # Unexpected exception
            mock_run.side_effect = ValueError("Unexpected error")

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            assert "unexpected" in result.error.lower()

    def test_error_logging(self, compiler_with_mocks, temp_grammar, caplog):
        """Test that errors are properly logged."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcc"],
                stderr="compilation error"
            )

            result = compiler_with_mocks.compile(temp_grammar)

            assert not result.success
            # Check that error was logged
            assert any("compilation failed" in record.message.lower() for record in caplog.records)


class TestWarningExtraction:
    """Tests for compiler warning extraction."""

    def test_extract_warnings_from_gcc_output(self, compiler_with_mocks):
        """Test warning extraction from GCC-style output."""
        stderr = """
parser.c: In function 'ts_parser_parse':
parser.c:123:5: warning: unused variable 'x' [-Wunused-variable]
parser.c:456:10: warning: comparison between signed and unsigned [-Wsign-compare]
"""
        warnings = compiler_with_mocks._extract_warnings(stderr)

        assert len(warnings) == 2
        assert "unused variable" in warnings[0].lower()
        assert "comparison between signed" in warnings[1].lower()

    def test_extract_warnings_from_clang_output(self, compiler_with_mocks):
        """Test warning extraction from Clang-style output."""
        stderr = """
parser.c:100:10: warning: implicit conversion from 'int' to 'char' [-Wimplicit-conversion]
scanner.c:50:5: warning: expression result unused [-Wunused-value]
"""
        warnings = compiler_with_mocks._extract_warnings(stderr)

        assert len(warnings) == 2

    def test_extract_warnings_no_warnings(self, compiler_with_mocks):
        """Test warning extraction when there are no warnings."""
        stderr = "parser.c:100:5: error: expected ';'"
        warnings = compiler_with_mocks._extract_warnings(stderr)

        assert len(warnings) == 0

    def test_extract_warnings_empty_stderr(self, compiler_with_mocks):
        """Test warning extraction with empty stderr."""
        warnings = compiler_with_mocks._extract_warnings(None)
        assert len(warnings) == 0

        warnings = compiler_with_mocks._extract_warnings("")
        assert len(warnings) == 0


class TestErrorMessageFormatting:
    """Tests for error message formatting."""

    def test_format_error_with_stderr(self, compiler_with_mocks, temp_grammar):
        """Test error formatting includes stderr output."""
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["gcc", "-o", "test.so"],
            stderr="error: something went wrong"
        )

        # Need to create a mock analysis
        from common.core.grammar_dependencies import DependencyResolver
        resolver = DependencyResolver()
        analysis = resolver.analyze_grammar(temp_grammar)

        formatted = compiler_with_mocks._format_compilation_error(error, analysis)

        assert "compiler output" in formatted.lower()
        assert "something went wrong" in formatted

    def test_format_error_pattern_matching(self, compiler_with_mocks, temp_grammar):
        """Test that error patterns are correctly identified."""
        from common.core.grammar_dependencies import DependencyResolver
        resolver = DependencyResolver()
        analysis = resolver.analyze_grammar(temp_grammar)

        # Test undefined reference
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["gcc"],
            stderr="undefined reference to `symbol'"
        )
        formatted = compiler_with_mocks._format_compilation_error(error, analysis)
        assert "linker error" in formatted.lower()

        # Test permission denied
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["gcc"],
            stderr="permission denied"
        )
        formatted = compiler_with_mocks._format_compilation_error(error, analysis)
        assert "permission error" in formatted.lower()

    def test_format_error_always_includes_suggestions(self, compiler_with_mocks, temp_grammar):
        """Test that error messages always include helpful suggestions."""
        from common.core.grammar_dependencies import DependencyResolver
        resolver = DependencyResolver()
        analysis = resolver.analyze_grammar(temp_grammar)

        error = subprocess.CalledProcessError(
            returncode=1, cmd=["gcc"], stderr="unknown error"
        )
        formatted = compiler_with_mocks._format_compilation_error(error, analysis)

        # Should include generic suggestions
        assert "suggestions" in formatted.lower() or "possible solutions" in formatted.lower()
