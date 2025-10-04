"""
Unit tests for Tree-sitter Grammar Compiler.

Tests compiler detection, grammar compilation, external scanner support,
and cross-platform compatibility.
"""

import pytest
import tempfile
import shutil
import platform
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from src.python.common.core.grammar_compiler import (
    GrammarCompiler,
    CompilerDetector,
    CompilerInfo,
    CompilationResult
)


class TestCompilerInfo:
    """Test suite for CompilerInfo dataclass."""

    def test_compiler_info_creation(self):
        """Test creating CompilerInfo."""
        info = CompilerInfo(
            name="gcc",
            path="/usr/bin/gcc",
            version="gcc (GCC) 11.2.0",
            is_cpp=False
        )

        assert info.name == "gcc"
        assert info.path == "/usr/bin/gcc"
        assert info.version == "gcc (GCC) 11.2.0"
        assert info.is_cpp is False

    def test_compiler_info_to_dict(self):
        """Test CompilerInfo serialization."""
        info = CompilerInfo(
            name="clang",
            path="/usr/bin/clang",
            version="clang version 13.0.0"
        )

        data = info.to_dict()
        assert data["name"] == "clang"
        assert data["path"] == "/usr/bin/clang"
        assert data["version"] == "clang version 13.0.0"
        assert data["is_cpp"] is False


class TestCompilationResult:
    """Test suite for CompilationResult dataclass."""

    def test_compilation_result_success(self):
        """Test successful compilation result."""
        result = CompilationResult(
            success=True,
            grammar_name="python",
            output_path=Path("/path/to/python.so"),
            message="Compilation successful"
        )

        assert result.success is True
        assert result.grammar_name == "python"
        assert result.output_path == Path("/path/to/python.so")
        assert result.error is None
        assert result.warnings == []

    def test_compilation_result_failure(self):
        """Test failed compilation result."""
        result = CompilationResult(
            success=False,
            grammar_name="python",
            error="Compiler not found"
        )

        assert result.success is False
        assert result.error == "Compiler not found"
        assert result.output_path is None

    def test_compilation_result_with_warnings(self):
        """Test compilation result with warnings."""
        result = CompilationResult(
            success=True,
            grammar_name="python",
            output_path=Path("/path/to/python.so"),
            warnings=["warning: unused variable"]
        )

        assert result.success is True
        assert len(result.warnings) == 1
        assert "unused variable" in result.warnings[0]


class TestCompilerDetector:
    """Test suite for CompilerDetector class."""

    def test_detector_initialization(self):
        """Test detector initializes with correct system."""
        detector = CompilerDetector()
        assert detector.system == platform.system()
        assert isinstance(detector._cache, dict)

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_detect_c_compiler_gcc(self, mock_run, mock_which):
        """Test C compiler detection finds gcc."""
        mock_which.side_effect = lambda x: "/usr/bin/gcc" if x == "gcc" else None
        mock_run.return_value = Mock(
            returncode=0,
            stdout="gcc (GCC) 11.2.0\n"
        )

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        assert compiler is not None
        assert compiler.name == "gcc"
        assert compiler.path == "/usr/bin/gcc"
        assert "GCC" in compiler.version
        assert compiler.is_cpp is False

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_detect_c_compiler_clang(self, mock_run, mock_which):
        """Test C compiler detection finds clang."""
        mock_which.side_effect = lambda x: "/usr/bin/clang" if x == "clang" else None
        mock_run.return_value = Mock(
            returncode=0,
            stdout="clang version 13.0.0\n"
        )

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        assert compiler is not None
        assert compiler.name == "clang"
        assert "clang" in compiler.version

    @patch('shutil.which')
    def test_detect_c_compiler_not_found(self, mock_which):
        """Test C compiler detection when no compiler found."""
        mock_which.return_value = None

        detector = CompilerDetector()
        compiler = detector.detect_c_compiler()

        assert compiler is None

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_detect_cpp_compiler(self, mock_run, mock_which):
        """Test C++ compiler detection."""
        mock_which.side_effect = lambda x: "/usr/bin/g++" if x == "g++" else None
        mock_run.return_value = Mock(
            returncode=0,
            stdout="g++ (GCC) 11.2.0\n"
        )

        detector = CompilerDetector()
        compiler = detector.detect_cpp_compiler()

        assert compiler is not None
        assert compiler.name == "g++"
        assert compiler.is_cpp is True

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_compiler_detection_caching(self, mock_run, mock_which):
        """Test compiler detection results are cached."""
        mock_which.side_effect = lambda x: "/usr/bin/gcc" if x in ["gcc", "clang", "cc"] else None
        mock_run.return_value = Mock(
            returncode=0,
            stdout="gcc (GCC) 11.2.0\n"
        )

        detector = CompilerDetector()
        compiler1 = detector.detect_c_compiler()
        compiler2 = detector.detect_c_compiler()

        assert compiler1 is compiler2
        # which may be called multiple times during detection but results are cached
        # Second call to detect_c_compiler should not trigger any more which calls
        initial_count = mock_which.call_count
        detector.detect_c_compiler()
        assert mock_which.call_count == initial_count  # No additional calls

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_get_compiler_version_timeout(self, mock_run, mock_which):
        """Test version detection handles timeout."""
        mock_which.return_value = "/usr/bin/gcc"
        mock_run.side_effect = subprocess.TimeoutExpired("gcc", 5)

        detector = CompilerDetector()
        version = detector._get_compiler_version("gcc", "/usr/bin/gcc")

        assert version is None


class TestGrammarCompiler:
    """Test suite for GrammarCompiler class."""

    @pytest.fixture
    def temp_grammar_dir(self):
        """Create temporary grammar directory structure."""
        temp_dir = tempfile.mkdtemp()
        grammar_dir = Path(temp_dir) / "tree-sitter-python"
        grammar_dir.mkdir()

        src_dir = grammar_dir / "src"
        src_dir.mkdir()

        # Create parser.c
        (src_dir / "parser.c").write_text("// parser code")

        yield grammar_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_compiler(self):
        """Create mock compiler info."""
        return CompilerInfo(
            name="gcc",
            path="/usr/bin/gcc",
            version="gcc (GCC) 11.2.0",
            is_cpp=False
        )

    @pytest.fixture
    def mock_cpp_compiler(self):
        """Create mock C++ compiler info."""
        return CompilerInfo(
            name="g++",
            path="/usr/bin/g++",
            version="g++ (GCC) 11.2.0",
            is_cpp=True
        )

    def test_compiler_initialization_with_compilers(self, mock_compiler, mock_cpp_compiler):
        """Test compiler initialization with provided compilers."""
        compiler = GrammarCompiler(
            c_compiler=mock_compiler,
            cpp_compiler=mock_cpp_compiler
        )

        assert compiler.c_compiler == mock_compiler
        assert compiler.cpp_compiler == mock_cpp_compiler

    @patch.object(CompilerDetector, 'detect_c_compiler')
    @patch.object(CompilerDetector, 'detect_cpp_compiler')
    def test_compiler_initialization_auto_detect(self, mock_detect_cpp, mock_detect_c):
        """Test compiler initialization auto-detects compilers."""
        mock_c = Mock()
        mock_cpp = Mock()
        mock_detect_c.return_value = mock_c
        mock_detect_cpp.return_value = mock_cpp

        compiler = GrammarCompiler()

        assert compiler.c_compiler == mock_c
        assert compiler.cpp_compiler == mock_cpp

    def test_detect_scanner_c_scanner(self, temp_grammar_dir):
        """Test detecting C scanner file."""
        src_dir = temp_grammar_dir / "src"
        (src_dir / "scanner.c").write_text("// C scanner")

        compiler = GrammarCompiler(c_compiler=Mock())
        has_scanner, scanner_path, needs_cpp = compiler._detect_scanner(src_dir)

        assert has_scanner is True
        assert scanner_path == src_dir / "scanner.c"
        assert needs_cpp is False

    def test_detect_scanner_cpp_scanner(self, temp_grammar_dir):
        """Test detecting C++ scanner file."""
        src_dir = temp_grammar_dir / "src"
        (src_dir / "scanner.cc").write_text("// C++ scanner")

        compiler = GrammarCompiler(c_compiler=Mock())
        has_scanner, scanner_path, needs_cpp = compiler._detect_scanner(src_dir)

        assert has_scanner is True
        assert scanner_path == src_dir / "scanner.cc"
        assert needs_cpp is True

    def test_detect_scanner_no_scanner(self, temp_grammar_dir):
        """Test scanner detection when no scanner exists."""
        src_dir = temp_grammar_dir / "src"

        compiler = GrammarCompiler(c_compiler=Mock())
        has_scanner, scanner_path, needs_cpp = compiler._detect_scanner(src_dir)

        assert has_scanner is False
        assert scanner_path is None
        assert needs_cpp is False

    def test_compile_no_src_directory(self, mock_compiler):
        """Test compilation fails when src directory missing."""
        compiler = GrammarCompiler(c_compiler=mock_compiler)
        grammar_dir = Path(tempfile.mkdtemp())

        result = compiler.compile(grammar_dir)

        assert result.success is False
        assert "Source directory not found" in result.error
        shutil.rmtree(grammar_dir)

    def test_compile_no_parser_c(self, temp_grammar_dir, mock_compiler):
        """Test compilation fails when parser.c missing."""
        # Remove parser.c
        (temp_grammar_dir / "src" / "parser.c").unlink()

        compiler = GrammarCompiler(c_compiler=mock_compiler)
        result = compiler.compile(temp_grammar_dir)

        assert result.success is False
        assert "parser.c not found" in result.error

    @patch.object(CompilerDetector, 'detect_c_compiler', return_value=None)
    @patch.object(CompilerDetector, 'detect_cpp_compiler', return_value=None)
    def test_compile_no_compiler(self, mock_cpp, mock_c, temp_grammar_dir):
        """Test compilation fails when no compiler available."""
        compiler = GrammarCompiler()
        result = compiler.compile(temp_grammar_dir)

        assert result.success is False
        assert "No C compiler found" in result.error

    @patch.object(CompilerDetector, 'detect_cpp_compiler', return_value=None)
    def test_compile_cpp_scanner_no_cpp_compiler(self, mock_detect, temp_grammar_dir, mock_compiler):
        """Test compilation fails when C++ scanner but no C++ compiler."""
        (temp_grammar_dir / "src" / "scanner.cc").write_text("// C++ scanner")

        compiler = GrammarCompiler(c_compiler=mock_compiler, cpp_compiler=None)
        result = compiler.compile(temp_grammar_dir)

        assert result.success is False
        assert "C++ scanner" in result.error
        assert "no C++ compiler" in result.error

    @patch('subprocess.run')
    def test_compile_success(self, mock_run, temp_grammar_dir, mock_compiler):
        """Test successful grammar compilation."""
        # Mock successful compilation
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Patch exists to return True for output file
        with patch.object(Path, 'exists', return_value=True):
            result = compiler.compile(temp_grammar_dir)

        assert result.success is True
        assert result.grammar_name == "python"
        assert result.output_path is not None

    @patch('subprocess.run')
    def test_compile_with_scanner(self, mock_run, temp_grammar_dir, mock_compiler):
        """Test compilation with external scanner."""
        (temp_grammar_dir / "src" / "scanner.c").write_text("// scanner")

        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        with patch.object(Path, 'exists', return_value=True):
            result = compiler.compile(temp_grammar_dir)

        assert result.success is True
        # Check that scanner.c was included in compilation
        call_args = mock_run.call_args[0][0]
        assert any("scanner.c" in str(arg) for arg in call_args)

    @patch('subprocess.run')
    def test_compile_failure(self, mock_run, temp_grammar_dir, mock_compiler):
        """Test compilation failure handling."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1,
            ["gcc"],
            stderr="compilation error"
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)
        result = compiler.compile(temp_grammar_dir)

        assert result.success is False
        assert "Compilation failed" in result.error

    @patch('subprocess.run')
    def test_compile_output_file_not_created(self, mock_run, temp_grammar_dir, mock_compiler):
        """Test compilation fails if output file not created."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        # Create a more selective exists patch that only affects the output file check
        original_exists = Path.exists

        def selective_exists(self):
            # Return False for .so/.dylib/.dll files, True for everything else
            if str(self).endswith(('.so', '.dylib', '.dll')):
                return False
            return original_exists(self)

        with patch.object(Path, 'exists', selective_exists):
            result = compiler.compile(temp_grammar_dir)

        assert result.success is False
        assert "output file not found" in result.error

    @patch('subprocess.run')
    def test_compile_custom_output_dir(self, mock_run, temp_grammar_dir, mock_compiler):
        """Test compilation with custom output directory."""
        custom_output = temp_grammar_dir / "custom_build"

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        compiler = GrammarCompiler(c_compiler=mock_compiler)

        with patch.object(Path, 'exists', return_value=True):
            result = compiler.compile(temp_grammar_dir, output_dir=custom_output)

        assert result.success is True
        # Verify custom output directory was used
        assert custom_output.exists()

    @patch('subprocess.run')
    @patch('platform.system')
    def test_compile_darwin_flags(self, mock_system, mock_run, temp_grammar_dir, mock_compiler):
        """Test Darwin-specific compilation flags."""
        mock_system.return_value = "Darwin"
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        compiler = GrammarCompiler(c_compiler=mock_compiler)
        compiler.system = "Darwin"

        with patch.object(Path, 'exists', return_value=True):
            compiler.compile(temp_grammar_dir)

        # Check for Darwin-specific flags
        call_args = mock_run.call_args[0][0]
        assert "-dynamiclib" in call_args

    @patch('subprocess.run')
    @patch('platform.system')
    def test_compile_windows_msvc(self, mock_system, mock_run, temp_grammar_dir):
        """Test Windows MSVC compilation."""
        mock_system.return_value = "Windows"
        msvc_compiler = CompilerInfo(name="cl", path="cl.exe", version="MSVC 19.0")

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        compiler = GrammarCompiler(c_compiler=msvc_compiler)
        compiler.system = "Windows"

        with patch.object(Path, 'exists', return_value=True):
            compiler.compile(temp_grammar_dir)

        # Check for MSVC-specific flags
        call_args = mock_run.call_args[0][0]
        assert "/LD" in call_args  # Create DLL

    def test_compile_extracts_grammar_name(self, temp_grammar_dir, mock_compiler):
        """Test grammar name extraction from directory."""
        compiler = GrammarCompiler(c_compiler=None)

        # Test with tree-sitter prefix
        result = compiler.compile(temp_grammar_dir)
        assert result.grammar_name == "python"

        # Test without tree-sitter prefix
        no_prefix_dir = temp_grammar_dir.parent / "mylang"
        temp_grammar_dir.rename(no_prefix_dir)
        (no_prefix_dir / "src").mkdir(exist_ok=True)
        (no_prefix_dir / "src" / "parser.c").touch()

        result = compiler.compile(no_prefix_dir)
        assert result.grammar_name == "mylang"
