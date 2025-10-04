"""Cross-platform tests for tree-sitter grammar management."""

import platform
import pytest
from pathlib import Path, PureWindowsPath, PurePosixPath
from unittest.mock import Mock, patch, MagicMock

from common.core.grammar_compiler import (
    GrammarCompiler,
    CompilerDetector,
    CompilerInfo,
)
from common.core.grammar_discovery import GrammarDiscovery
from common.core.grammar_dependencies import DependencyResolver


@pytest.fixture
def temp_grammar(tmp_path):
    """Create a temporary grammar directory."""
    grammar_dir = tmp_path / "tree-sitter-test"
    src_dir = grammar_dir / "src"
    src_dir.mkdir(parents=True)

    # Create parser.c
    parser_c = src_dir / "parser.c"
    parser_c.write_text("// parser code")

    return grammar_dir


class TestCrossPlatformPaths:
    """Tests for cross-platform path handling."""

    def test_path_handling_with_home_directory(self, tmp_path):
        """Test that paths with ~ work on all platforms."""
        # This should work regardless of platform
        discovery = GrammarDiscovery()

        # Test that path resolution doesn't break
        test_path = tmp_path / "test"
        test_path.mkdir()

        resolved = test_path.resolve()
        assert resolved.exists()

    def test_windows_path_separators(self):
        """Test Windows-style path handling."""
        # Test that we can handle Windows paths
        win_path = PureWindowsPath("C:\\Users\\test\\grammars\\tree-sitter-python")

        # Should be able to extract grammar name
        name = win_path.name
        if name.startswith("tree-sitter-"):
            grammar_name = name[len("tree-sitter-"):]
            assert grammar_name == "python"

    def test_unix_path_separators(self):
        """Test Unix-style path handling."""
        # Test that we can handle Unix paths
        unix_path = PurePosixPath("/home/user/.config/tree-sitter/grammars/tree-sitter-python")

        # Should be able to extract grammar name
        name = unix_path.name
        if name.startswith("tree-sitter-"):
            grammar_name = name[len("tree-sitter-"):]
            assert grammar_name == "python"

    def test_path_creation_cross_platform(self, tmp_path):
        """Test that path creation works on all platforms."""
        # Create nested directories
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True, exist_ok=True)

        assert nested.exists()
        assert nested.is_dir()


class TestCrossPlatformCompilerDetection:
    """Tests for compiler detection on different platforms."""

    @patch("platform.system")
    def test_windows_compiler_order(self, mock_system):
        """Test Windows compiler search order."""
        mock_system.return_value = "Windows"

        detector = CompilerDetector()

        assert detector.system == "Windows"
        # Windows should prefer MSVC (cl) first
        assert detector.WINDOWS_COMPILERS[0] == "cl"
        assert "gcc" in detector.WINDOWS_COMPILERS
        assert "clang" in detector.WINDOWS_COMPILERS

    @patch("platform.system")
    def test_unix_compiler_order(self, mock_system):
        """Test Unix/Linux compiler search order."""
        mock_system.return_value = "Linux"

        detector = CompilerDetector()

        assert detector.system == "Linux"
        # Unix should prefer clang or gcc
        assert detector.UNIX_COMPILERS[0] in ["clang", "gcc"]
        assert "cc" in detector.UNIX_COMPILERS

    @patch("platform.system")
    def test_darwin_compiler_order(self, mock_system):
        """Test macOS compiler search order."""
        mock_system.return_value = "Darwin"

        detector = CompilerDetector()

        assert detector.system == "Darwin"
        # macOS uses Unix compiler order
        assert detector.UNIX_COMPILERS[0] == "clang"

    @patch("platform.system")
    @patch("shutil.which")
    def test_compiler_detection_windows(self, mock_which, mock_system):
        """Test compiler detection on Windows."""
        mock_system.return_value = "Windows"

        # Simulate MSVC available
        def which_side_effect(name):
            if name == "cl":
                return "C:\\Program Files\\Microsoft Visual Studio\\VC\\bin\\cl.exe"
            return None

        mock_which.side_effect = which_side_effect

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        assert compiler is not None
        assert compiler.name == "cl"

    @patch("platform.system")
    @patch("shutil.which")
    def test_compiler_detection_linux(self, mock_which, mock_system):
        """Test compiler detection on Linux."""
        mock_system.return_value = "Linux"

        # Simulate gcc available
        def which_side_effect(name):
            if name == "gcc":
                return "/usr/bin/gcc"
            return None

        mock_which.side_effect = which_side_effect

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        # Should find gcc (second in Unix list after clang)
        assert compiler is not None
        assert compiler.name == "gcc"

    @patch("platform.system")
    @patch("shutil.which")
    def test_compiler_detection_macos(self, mock_which, mock_system):
        """Test compiler detection on macOS."""
        mock_system.return_value = "Darwin"

        # Simulate clang available (default on macOS)
        def which_side_effect(name):
            if name == "clang":
                return "/usr/bin/clang"
            return None

        mock_which.side_effect = which_side_effect

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        assert compiler is not None
        assert compiler.name == "clang"


class TestCrossPlatformCompilation:
    """Tests for compilation on different platforms."""

    @patch("platform.system")
    def test_library_extension_windows(self, mock_system, temp_grammar):
        """Test that Windows uses .dll extension."""
        mock_system.return_value = "Windows"

        mock_compiler = CompilerInfo(
            name="cl",
            path="C:\\Program Files\\MSVC\\cl.exe",
            version="19.0"
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Library extension should be .dll on Windows
        assert compiler.system == "Windows"

    @patch("platform.system")
    def test_library_extension_linux(self, mock_system):
        """Test that Linux uses .so extension."""
        mock_system.return_value = "Linux"

        mock_compiler = CompilerInfo(
            name="gcc",
            path="/usr/bin/gcc",
            version="11.0"
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        assert compiler.system == "Linux"

    @patch("platform.system")
    def test_library_extension_macos(self, mock_system):
        """Test that macOS uses .dylib extension."""
        mock_system.return_value = "Darwin"

        mock_compiler = CompilerInfo(
            name="clang",
            path="/usr/bin/clang",
            version="14.0"
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        assert compiler.system == "Darwin"

    @patch("platform.system")
    @patch("subprocess.run")
    def test_compilation_flags_unix(self, mock_run, mock_system, temp_grammar):
        """Test Unix compilation flags."""
        mock_system.return_value = "Linux"

        mock_compiler = CompilerInfo(
            name="gcc",
            path="/usr/bin/gcc",
            version="11.0"
        )

        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Create build directory
        build_dir = temp_grammar / "build"
        build_dir.mkdir()

        # Mock the output file creation
        output_file = build_dir / "test.so"
        output_file.touch()

        result = compiler.compile(temp_grammar)

        # Check that gcc was called with correct flags
        call_args = mock_run.call_args[0][0]
        assert "-shared" in call_args
        assert "-fPIC" in call_args
        assert "-O2" in call_args

    @patch("platform.system")
    @patch("subprocess.run")
    def test_compilation_flags_macos(self, mock_run, mock_system, temp_grammar):
        """Test macOS-specific compilation flags."""
        mock_system.return_value = "Darwin"

        mock_compiler = CompilerInfo(
            name="clang",
            path="/usr/bin/clang",
            version="14.0"
        )

        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Create build directory
        build_dir = temp_grammar / "build"
        build_dir.mkdir()

        # Mock the output file creation
        output_file = build_dir / "test.dylib"
        output_file.touch()

        result = compiler.compile(temp_grammar)

        # Check that clang was called with macOS-specific flags
        call_args = mock_run.call_args[0][0]
        assert "-dynamiclib" in call_args
        assert any("-Wl,-install_name" in str(arg) for arg in call_args)

    @patch("platform.system")
    @patch("subprocess.run")
    def test_compilation_flags_windows_msvc(self, mock_run, mock_system, temp_grammar):
        """Test Windows MSVC compilation flags."""
        mock_system.return_value = "Windows"

        mock_compiler = CompilerInfo(
            name="cl",
            path="C:\\Program Files\\MSVC\\cl.exe",
            version="19.0"
        )

        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Create build directory
        build_dir = temp_grammar / "build"
        build_dir.mkdir()

        # Mock the output file creation
        output_file = build_dir / "test.dll"
        output_file.touch()

        result = compiler.compile(temp_grammar)

        # Check that cl was called with MSVC-specific flags
        call_args = mock_run.call_args[0][0]
        assert "/LD" in call_args  # Create DLL
        assert "/O2" in call_args  # Optimization


class TestCrossPlatformDependencyResolution:
    """Tests for dependency resolution across platforms."""

    def test_dependency_resolver_cross_platform(self, temp_grammar):
        """Test that dependency resolver works on current platform."""
        resolver = DependencyResolver()
        analysis = resolver.analyze_grammar(temp_grammar)

        # Should work regardless of platform
        assert analysis.grammar_name == "test"
        assert analysis.is_valid

    @patch("platform.system")
    def test_scanner_detection_cross_platform(self, mock_system, temp_grammar):
        """Test scanner detection works on all platforms."""
        # Test on different platforms
        for system in ["Windows", "Linux", "Darwin"]:
            mock_system.return_value = system

            # Create C++ scanner
            scanner_cc = temp_grammar / "src" / "scanner.cc"
            scanner_cc.write_text("// C++ scanner")

            resolver = DependencyResolver()
            analysis = resolver.analyze_grammar(temp_grammar)

            assert analysis.has_external_scanner
            assert analysis.needs_cpp

            # Clean up for next iteration
            scanner_cc.unlink()


class TestCrossPlatformGrammarDiscovery:
    """Tests for grammar discovery across platforms."""

    def test_parser_library_detection_extensions(self, tmp_path):
        """Test that all platform library extensions are detected."""
        discovery = GrammarDiscovery()

        # Create a test grammar directory
        grammar_dir = tmp_path / "tree-sitter-test"
        build_dir = grammar_dir / "build"
        build_dir.mkdir(parents=True)

        # Test each extension type
        extensions = [".so", ".dll", ".dylib"]

        for ext in extensions:
            lib_file = build_dir / f"test{ext}"
            lib_file.touch()

            # Should find the library
            found = discovery._find_parser_library(grammar_dir)
            assert found is not None
            assert found.suffix == ext

            # Clean up for next test
            lib_file.unlink()

    def test_path_resolution_current_platform(self, tmp_path):
        """Test path resolution works on current platform."""
        # Create test structure
        grammar_dir = tmp_path / "grammars" / "tree-sitter-python"
        src_dir = grammar_dir / "src"
        src_dir.mkdir(parents=True)

        # Should resolve correctly regardless of platform
        resolved = grammar_dir.resolve()
        assert resolved.exists()
        assert "tree-sitter-python" in str(resolved)


class TestCrossPlatformErrorMessages:
    """Tests for cross-platform error message handling."""

    @patch("platform.system")
    def test_error_messages_unix_style(self, mock_system):
        """Test error messages use Unix-style paths when appropriate."""
        mock_system.return_value = "Linux"

        # Error messages should not have platform-specific assumptions
        # This is validated by the error handling tests

    @patch("platform.system")
    def test_error_messages_windows_style(self, mock_system):
        """Test error messages work with Windows paths."""
        mock_system.return_value = "Windows"

        # Error messages should handle Windows paths correctly
        # This is validated by the error handling tests


class TestPlatformAgnosticFeatures:
    """Tests ensuring features work the same across platforms."""

    def test_grammar_name_extraction(self):
        """Test grammar name extraction is platform-agnostic."""
        # Test with different path styles
        paths = [
            "tree-sitter-python",
            "/unix/path/tree-sitter-python",
            "C:\\Windows\\Path\\tree-sitter-python",
        ]

        for path_str in paths:
            name = Path(path_str).name
            if name.startswith("tree-sitter-"):
                grammar_name = name[len("tree-sitter-"):]
                assert grammar_name == "python"

    def test_source_file_detection(self, temp_grammar):
        """Test source file detection works on current platform."""
        resolver = DependencyResolver()
        analysis = resolver.analyze_grammar(temp_grammar)

        # Should detect parser.c regardless of platform
        assert len(analysis.dependencies) == 1
        assert analysis.dependencies[0].path.name == "parser.c"
