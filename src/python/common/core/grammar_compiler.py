"""
Grammar Compilation System for Tree-sitter.

This module provides functionality to compile tree-sitter grammars with
automatic C compiler detection and external scanner support.

Key features:
- Automatic C/C++ compiler detection (gcc, clang, msvc, cc)
- Support for external scanners (scanner.c/scanner.cc)
- Cross-platform compilation (Windows, macOS, Linux)
- Build artifact management
- Comprehensive error reporting
- Dependency resolution for build prerequisites
"""

import logging
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import os

from .grammar_dependencies import DependencyResolver, CompilerRequirement

logger = logging.getLogger(__name__)


@dataclass
class CompilerInfo:
    """Information about a detected C/C++ compiler."""

    name: str
    """Compiler name (e.g., 'gcc', 'clang', 'msvc')"""

    path: str
    """Full path to compiler executable"""

    version: Optional[str] = None
    """Compiler version string"""

    is_cpp: bool = False
    """Whether this is a C++ compiler"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "is_cpp": self.is_cpp
        }


@dataclass
class CompilationResult:
    """Result of a grammar compilation operation."""

    success: bool
    """Whether compilation succeeded"""

    grammar_name: str
    """Name of the grammar"""

    output_path: Optional[Path] = None
    """Path to compiled library file (.so/.dll/.dylib)"""

    message: str = ""
    """Human-readable status message"""

    error: Optional[str] = None
    """Error message if compilation failed"""

    warnings: List[str] = None
    """Compilation warnings"""

    def __post_init__(self):
        """Initialize warnings list if None."""
        if self.warnings is None:
            self.warnings = []


class CompilerDetector:
    """
    Detects available C/C++ compilers on the system.

    Searches for common compilers in order of preference:
    - Unix: clang, gcc, cc
    - Windows: cl (MSVC), gcc (MinGW), clang
    """

    # Compiler search order by platform
    UNIX_COMPILERS = ["clang", "gcc", "cc"]
    UNIX_CPP_COMPILERS = ["clang++", "g++", "c++"]
    WINDOWS_COMPILERS = ["cl", "gcc", "clang"]
    WINDOWS_CPP_COMPILERS = ["cl", "g++", "clang++"]

    def __init__(self):
        """Initialize compiler detector."""
        self.system = platform.system()
        self._cache: Dict[str, Optional[CompilerInfo]] = {}

    def detect_c_compiler(self) -> Optional[CompilerInfo]:
        """
        Detect available C compiler.

        Returns:
            CompilerInfo if found, None otherwise
        """
        if "c" in self._cache:
            return self._cache["c"]

        compilers = (
            self.WINDOWS_COMPILERS if self.system == "Windows"
            else self.UNIX_COMPILERS
        )

        for compiler_name in compilers:
            compiler = self._try_compiler(compiler_name, is_cpp=False)
            if compiler:
                self._cache["c"] = compiler
                return compiler

        self._cache["c"] = None
        return None

    def detect_cpp_compiler(self) -> Optional[CompilerInfo]:
        """
        Detect available C++ compiler.

        Returns:
            CompilerInfo if found, None otherwise
        """
        if "cpp" in self._cache:
            return self._cache["cpp"]

        compilers = (
            self.WINDOWS_CPP_COMPILERS if self.system == "Windows"
            else self.UNIX_CPP_COMPILERS
        )

        for compiler_name in compilers:
            compiler = self._try_compiler(compiler_name, is_cpp=True)
            if compiler:
                self._cache["cpp"] = compiler
                return compiler

        self._cache["cpp"] = None
        return None

    def _try_compiler(self, name: str, is_cpp: bool = False) -> Optional[CompilerInfo]:
        """
        Try to detect a specific compiler.

        Args:
            name: Compiler executable name
            is_cpp: Whether this is a C++ compiler

        Returns:
            CompilerInfo if compiler found and working, None otherwise
        """
        # Find compiler in PATH
        compiler_path = shutil.which(name)
        if not compiler_path:
            return None

        # Get version
        version = self._get_compiler_version(name, compiler_path)

        logger.info(f"Found {'C++' if is_cpp else 'C'} compiler: {name} ({version or 'unknown version'})")

        return CompilerInfo(
            name=name,
            path=compiler_path,
            version=version,
            is_cpp=is_cpp
        )

    def _get_compiler_version(self, name: str, path: str) -> Optional[str]:
        """
        Get compiler version string.

        Args:
            name: Compiler name
            path: Path to compiler executable

        Returns:
            Version string if available
        """
        try:
            # Try --version flag (works for gcc, clang, g++, clang++)
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Extract first line which usually contains version
                first_line = result.stdout.strip().split("\n")[0]
                return first_line

            # MSVC uses different flag
            if name == "cl":
                result = subprocess.run(
                    [path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 or "Microsoft" in result.stdout:
                    lines = result.stdout.strip().split("\n")
                    for line in lines:
                        if "Version" in line:
                            return line.strip()

        except Exception as e:
            logger.debug(f"Failed to get version for {name}: {e}")

        return None


class GrammarCompiler:
    """
    Compiles tree-sitter grammars to shared libraries.

    Handles compilation of parser.c and optional external scanners
    (scanner.c or scanner.cc) into platform-specific shared libraries.
    """

    def __init__(
        self,
        c_compiler: Optional[CompilerInfo] = None,
        cpp_compiler: Optional[CompilerInfo] = None
    ):
        """
        Initialize grammar compiler.

        Args:
            c_compiler: C compiler to use (auto-detected if None)
            cpp_compiler: C++ compiler to use (auto-detected if None)
        """
        self.detector = CompilerDetector()
        self.dependency_resolver = DependencyResolver()

        # Use provided or detect compilers
        self.c_compiler = c_compiler or self.detector.detect_c_compiler()
        self.cpp_compiler = cpp_compiler or self.detector.detect_cpp_compiler()

        if not self.c_compiler:
            logger.warning("No C compiler detected - compilation will fail")

        self.system = platform.system()

    def compile(self, grammar_path: Path, output_dir: Optional[Path] = None) -> CompilationResult:
        """
        Compile a tree-sitter grammar.

        Uses dependency resolution to analyze build requirements and validate
        all prerequisites before compilation.

        Args:
            grammar_path: Path to grammar directory
            output_dir: Output directory for compiled library (defaults to grammar_path/build)

        Returns:
            CompilationResult with status and details

        Example:
            >>> compiler = GrammarCompiler()
            >>> result = compiler.compile(Path("/path/to/tree-sitter-python"))
            >>> if result.success:
            ...     print(f"Compiled to {result.output_path}")
        """
        # Analyze dependencies
        analysis = self.dependency_resolver.analyze_grammar(grammar_path)

        logger.info(f"Compiling grammar '{analysis.grammar_name}' at {grammar_path}")
        logger.debug(f"Dependency analysis:\n{self.dependency_resolver.get_compilation_summary(analysis)}")

        # Validate grammar structure
        if not analysis.is_valid:
            error_msg = "; ".join(analysis.validation_errors)
            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg
            )

        # Validate dependencies with available compilers
        valid, errors = self.dependency_resolver.validate_dependencies(
            analysis,
            available_c_compiler=self.c_compiler is not None,
            available_cpp_compiler=self.cpp_compiler is not None
        )

        if not valid:
            error_msg = "; ".join(errors)
            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg
            )

        # Set up output directory
        if output_dir is None:
            output_dir = grammar_path / "build"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compile with resolved dependencies
        try:
            source_files = analysis.get_source_files()

            output_path = self._compile_grammar(
                grammar_name=analysis.grammar_name,
                source_files=source_files,
                output_dir=output_dir,
                needs_cpp=analysis.needs_cpp
            )

            return CompilationResult(
                success=True,
                grammar_name=analysis.grammar_name,
                output_path=output_path,
                message=f"Successfully compiled grammar '{analysis.grammar_name}'"
            )

        except subprocess.CalledProcessError as e:
            # Enhanced error message with specific guidance
            error_msg = self._format_compilation_error(e, analysis)
            logger.error(error_msg)

            # Extract compiler warnings from stderr
            warnings = self._extract_warnings(e.stderr) if e.stderr else []

            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg,
                warnings=warnings
            )
        except subprocess.TimeoutExpired as e:
            error_msg = (
                f"Compilation timed out after {e.timeout} seconds. "
                f"This may indicate a very large grammar or system resource constraints. "
                f"Consider: (1) checking system resources, (2) closing other applications, "
                f"(3) trying again with fewer parallel builds."
            )
            logger.error(error_msg)
            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg
            )
        except PermissionError as e:
            error_msg = (
                f"Permission denied during compilation: {e}. "
                f"Ensure you have write permissions for: (1) output directory, "
                f"(2) source files. Try running with appropriate permissions."
            )
            logger.error(error_msg)
            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected compilation error: {str(e)}"
            logger.error(error_msg)
            logger.exception("Full error traceback:")
            return CompilationResult(
                success=False,
                grammar_name=analysis.grammar_name,
                error=error_msg
            )

    def _format_compilation_error(self, error: subprocess.CalledProcessError, analysis) -> str:
        """
        Format compilation error with helpful guidance.

        Args:
            error: Compilation process error
            analysis: Dependency analysis for context

        Returns:
            Formatted error message with suggestions
        """
        from .grammar_dependencies import DependencyAnalysis

        base_msg = f"Compilation failed for grammar '{analysis.grammar_name}'"

        # Extract stderr for analysis
        stderr = error.stderr or ""

        # Common error patterns and suggestions
        suggestions = []

        if "undefined reference" in stderr.lower() or "unresolved external" in stderr.lower():
            suggestions.append(
                "Linker error detected. This may indicate: "
                "(1) missing scanner implementation, (2) incompatible compiler flags, "
                "(3) missing system libraries"
            )

        if "no such file or directory" in stderr.lower():
            suggestions.append(
                "Missing include file. Ensure: (1) grammar was generated properly "
                "('tree-sitter generate'), (2) all source files are present"
            )

        if "permission denied" in stderr.lower():
            suggestions.append(
                "Permission error. Check: (1) file permissions, (2) directory permissions, "
                "(3) antivirus software blocking compilation"
            )

        if "syntax error" in stderr.lower() or "parse error" in stderr.lower() or "error: expected" in stderr.lower():
            suggestions.append(
                "Syntax error in generated code. Try: (1) regenerating grammar "
                "('tree-sitter generate'), (2) checking grammar.js for issues"
            )

        if analysis.needs_cpp and "g++" not in str(error.cmd) and "clang++" not in str(error.cmd):
            suggestions.append(
                "Grammar requires C++ compiler but may have been compiled with C compiler. "
                "Ensure C++ compiler is available and properly detected"
            )

        # Build final message
        parts = [base_msg]

        if stderr.strip():
            # Truncate very long error messages
            if len(stderr) > 1000:
                stderr_display = stderr[:1000] + "\n... (truncated)"
            else:
                stderr_display = stderr
            parts.append(f"\nCompiler output:\n{stderr_display}")

        if suggestions:
            parts.append(f"\nPossible solutions:\n" + "\n".join(f"  • {s}" for s in suggestions))
        else:
            parts.append(
                f"\nSuggestions:\n"
                f"  • Regenerate grammar: tree-sitter generate\n"
                f"  • Check compiler installation and version\n"
                f"  • Verify all source files are present"
            )

        return "\n".join(parts)

    def _extract_warnings(self, stderr: Optional[str]) -> List[str]:
        """
        Extract compiler warnings from stderr.

        Args:
            stderr: Compiler stderr output

        Returns:
            List of warning messages
        """
        if not stderr:
            return []

        warnings = []
        for line in stderr.split("\n"):
            line_lower = line.lower()
            if "warning" in line_lower and line.strip():
                # Clean up the warning message
                warnings.append(line.strip())

        return warnings

    def _detect_scanner(self, src_dir: Path) -> Tuple[bool, Optional[Path], bool]:
        """
        Detect external scanner file.

        Args:
            src_dir: Source directory to search

        Returns:
            Tuple of (has_scanner, scanner_path, needs_cpp)
        """
        scanner_files = {
            "scanner.c": False,     # C scanner
            "scanner.cc": True,     # C++ scanner
            "scanner.cpp": True,    # C++ scanner
        }

        for filename, needs_cpp in scanner_files.items():
            scanner_path = src_dir / filename
            if scanner_path.exists():
                logger.info(f"Found external scanner: {filename}")
                return True, scanner_path, needs_cpp

        return False, None, False

    def _compile_grammar(
        self,
        grammar_name: str,
        source_files: List[Path],
        output_dir: Path,
        needs_cpp: bool
    ) -> Path:
        """
        Compile grammar to shared library.

        Args:
            grammar_name: Name of grammar
            source_files: List of source files to compile (in order)
            output_dir: Output directory
            needs_cpp: Whether C++ compiler is needed

        Returns:
            Path to compiled library

        Raises:
            subprocess.CalledProcessError: If compilation fails
        """
        # Select compiler
        compiler = self.cpp_compiler if needs_cpp else self.c_compiler

        # Determine library extension
        if self.system == "Windows":
            lib_ext = ".dll"
        elif self.system == "Darwin":
            lib_ext = ".dylib"
        else:
            lib_ext = ".so"

        output_path = output_dir / f"{grammar_name}{lib_ext}"

        # Build compilation command from source files
        sources = [str(source) for source in source_files]

        # Get include directory from first source file
        include_dir = source_files[0].parent

        if compiler.name in ["gcc", "g++", "clang", "clang++", "cc", "c++"]:
            # GCC/Clang style
            cmd = [
                compiler.path,
                "-shared",
                "-fPIC",
                "-O2",
                "-o", str(output_path),
                *sources,
                "-I", str(include_dir)
            ]

            # Add platform-specific flags
            if self.system == "Darwin":
                cmd.extend(["-dynamiclib", "-Wl,-install_name," + str(output_path)])

        elif compiler.name == "cl":
            # MSVC style
            cmd = [
                compiler.path,
                "/LD",  # Create DLL
                "/O2",  # Optimization
                f"/Fe{output_path}",  # Output file
                *sources,
                f"/I{include_dir}"  # Include directory
            ]
        else:
            raise ValueError(f"Unsupported compiler: {compiler.name}")

        logger.info(f"Running: {' '.join(cmd)}")

        # Run compilation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=output_dir,
            timeout=120  # 2 minute timeout
        )

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr
            )

        # Log warnings if any
        if result.stderr:
            for line in result.stderr.split("\n"):
                if line.strip() and "warning" in line.lower():
                    logger.warning(f"Compiler warning: {line.strip()}")

        if not output_path.exists():
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        logger.info(f"Compiled grammar to: {output_path}")
        return output_path


# Export main classes
__all__ = [
    "GrammarCompiler",
    "CompilerDetector",
    "CompilerInfo",
    "CompilationResult"
]
