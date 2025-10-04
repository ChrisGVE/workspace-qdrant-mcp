"""
Dependency Resolution for Tree-sitter Grammar Compilation.

This module provides dependency analysis and resolution for tree-sitter
grammar compilation, ensuring all build prerequisites are met and
determining correct compilation order for source files.

Key features:
- Source file discovery (parser.c, scanner.c/cc/cpp)
- Compiler requirement detection
- Build dependency validation
- Compilation order determination
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Type of source file in grammar."""

    PARSER = "parser"
    """Main parser source (parser.c)"""

    SCANNER_C = "scanner_c"
    """C external scanner (scanner.c)"""

    SCANNER_CPP = "scanner_cpp"
    """C++ external scanner (scanner.cc/scanner.cpp)"""


class CompilerRequirement(Enum):
    """Compiler requirements for grammar compilation."""

    C_COMPILER = "c"
    """C compiler required (gcc, clang, cc, or MSVC)"""

    CPP_COMPILER = "cpp"
    """C++ compiler required (g++, clang++, c++, or MSVC)"""


@dataclass
class BuildDependency:
    """Represents a source file and its build requirements."""

    source_type: SourceType
    """Type of source file"""

    path: Path
    """Path to source file"""

    compiler_requirement: CompilerRequirement
    """Compiler needed to build this file"""

    compilation_order: int = 0
    """Order in which this should be compiled (0 = first)"""

    def __str__(self) -> str:
        """String representation."""
        return f"{self.source_type.value}: {self.path.name}"


@dataclass
class DependencyAnalysis:
    """Result of grammar dependency analysis."""

    grammar_name: str
    """Name of the grammar"""

    grammar_path: Path
    """Path to grammar directory"""

    dependencies: List[BuildDependency] = field(default_factory=list)
    """List of build dependencies"""

    required_compilers: Set[CompilerRequirement] = field(default_factory=set)
    """Set of required compilers"""

    is_valid: bool = True
    """Whether grammar has valid structure"""

    validation_errors: List[str] = field(default_factory=list)
    """List of validation errors if invalid"""

    has_external_scanner: bool = False
    """Whether grammar includes external scanner"""

    needs_cpp: bool = False
    """Whether C++ compiler is needed"""

    def get_source_files(self) -> List[Path]:
        """
        Get list of source files in compilation order.

        Returns:
            List of source file paths
        """
        sorted_deps = sorted(self.dependencies, key=lambda d: d.compilation_order)
        return [dep.path for dep in sorted_deps]

    def get_dependencies_by_type(self, source_type: SourceType) -> List[BuildDependency]:
        """
        Get dependencies of a specific type.

        Args:
            source_type: Type to filter by

        Returns:
            List of matching dependencies
        """
        return [dep for dep in self.dependencies if dep.source_type == source_type]


class DependencyResolver:
    """
    Analyzes and resolves build dependencies for tree-sitter grammars.

    Scans grammar directory to identify all source files, determines
    compiler requirements, and validates build prerequisites.
    """

    # Known scanner file patterns
    SCANNER_PATTERNS = {
        "scanner.c": (SourceType.SCANNER_C, CompilerRequirement.C_COMPILER),
        "scanner.cc": (SourceType.SCANNER_CPP, CompilerRequirement.CPP_COMPILER),
        "scanner.cpp": (SourceType.SCANNER_CPP, CompilerRequirement.CPP_COMPILER),
    }

    def __init__(self):
        """Initialize dependency resolver."""
        pass

    def analyze_grammar(self, grammar_path: Path, grammar_name: Optional[str] = None) -> DependencyAnalysis:
        """
        Analyze grammar directory for build dependencies.

        Args:
            grammar_path: Path to grammar directory
            grammar_name: Optional grammar name (derived from path if None)

        Returns:
            DependencyAnalysis with complete dependency information

        Example:
            >>> resolver = DependencyResolver()
            >>> analysis = resolver.analyze_grammar(Path("/path/to/tree-sitter-python"))
            >>> if analysis.is_valid:
            ...     print(f"Sources: {analysis.get_source_files()}")
            ...     print(f"Needs C++: {analysis.needs_cpp}")
        """
        # Determine grammar name
        if grammar_name is None:
            grammar_name = grammar_path.name
            if grammar_name.startswith("tree-sitter-"):
                grammar_name = grammar_name[len("tree-sitter-"):]

        analysis = DependencyAnalysis(
            grammar_name=grammar_name,
            grammar_path=grammar_path
        )

        # Check for src directory
        src_dir = grammar_path / "src"
        if not src_dir.exists():
            analysis.is_valid = False
            analysis.validation_errors.append(f"Source directory not found: {src_dir}")
            return analysis

        # Find parser.c (required)
        parser_c = src_dir / "parser.c"
        if not parser_c.exists():
            analysis.is_valid = False
            analysis.validation_errors.append(
                "parser.c not found. Run 'tree-sitter generate' to create it."
            )
            return analysis

        # Add parser.c dependency (always first)
        analysis.dependencies.append(
            BuildDependency(
                source_type=SourceType.PARSER,
                path=parser_c,
                compiler_requirement=CompilerRequirement.C_COMPILER,
                compilation_order=0
            )
        )
        analysis.required_compilers.add(CompilerRequirement.C_COMPILER)

        # Scan for external scanners
        scanner_dep = self._find_scanner(src_dir)
        if scanner_dep:
            analysis.dependencies.append(scanner_dep)
            analysis.required_compilers.add(scanner_dep.compiler_requirement)
            analysis.has_external_scanner = True
            analysis.needs_cpp = scanner_dep.compiler_requirement == CompilerRequirement.CPP_COMPILER

            logger.info(
                f"Grammar '{grammar_name}' has external scanner: {scanner_dep.path.name} "
                f"(requires {scanner_dep.compiler_requirement.value} compiler)"
            )

        return analysis

    def _find_scanner(self, src_dir: Path) -> Optional[BuildDependency]:
        """
        Find external scanner file in source directory.

        Args:
            src_dir: Source directory to search

        Returns:
            BuildDependency for scanner if found, None otherwise
        """
        for filename, (source_type, compiler_req) in self.SCANNER_PATTERNS.items():
            scanner_path = src_dir / filename
            if scanner_path.exists():
                return BuildDependency(
                    source_type=source_type,
                    path=scanner_path,
                    compiler_requirement=compiler_req,
                    compilation_order=1  # Compile after parser.c
                )

        return None

    def validate_dependencies(
        self,
        analysis: DependencyAnalysis,
        available_c_compiler: bool = False,
        available_cpp_compiler: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all dependencies can be satisfied.

        Args:
            analysis: Dependency analysis to validate
            available_c_compiler: Whether C compiler is available
            available_cpp_compiler: Whether C++ compiler is available

        Returns:
            Tuple of (is_valid, error_messages)

        Example:
            >>> resolver = DependencyResolver()
            >>> analysis = resolver.analyze_grammar(Path("/path/to/grammar"))
            >>> valid, errors = resolver.validate_dependencies(
            ...     analysis,
            ...     available_c_compiler=True,
            ...     available_cpp_compiler=False
            ... )
            >>> if not valid:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        errors = []

        # Check if grammar structure is valid
        if not analysis.is_valid:
            errors.extend(analysis.validation_errors)
            return False, errors

        # Check compiler availability
        if CompilerRequirement.C_COMPILER in analysis.required_compilers:
            if not available_c_compiler:
                errors.append(
                    "C compiler required but not available. "
                    "Install gcc, clang, or MSVC."
                )

        if CompilerRequirement.CPP_COMPILER in analysis.required_compilers:
            if not available_cpp_compiler:
                scanner_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_CPP)
                if scanner_deps:
                    scanner_name = scanner_deps[0].path.name
                    errors.append(
                        f"Grammar has C++ scanner ({scanner_name}) but no C++ compiler available. "
                        "Install g++, clang++, or MSVC."
                    )

        # Verify all source files exist
        for dep in analysis.dependencies:
            if not dep.path.exists():
                errors.append(f"Source file not found: {dep.path}")

        return len(errors) == 0, errors

    def get_compilation_summary(self, analysis: DependencyAnalysis) -> str:
        """
        Get human-readable summary of compilation requirements.

        Args:
            analysis: Dependency analysis

        Returns:
            Formatted summary string
        """
        lines = [
            f"Grammar: {analysis.grammar_name}",
            f"Source files: {len(analysis.dependencies)}",
        ]

        for dep in sorted(analysis.dependencies, key=lambda d: d.compilation_order):
            lines.append(f"  - {dep}")

        compiler_list = ", ".join(c.value for c in sorted(analysis.required_compilers, key=lambda x: x.value))
        lines.append(f"Required compilers: {compiler_list}")

        if analysis.has_external_scanner:
            scanner_type = "C++" if analysis.needs_cpp else "C"
            lines.append(f"External scanner: {scanner_type}")

        if not analysis.is_valid:
            lines.append("Status: INVALID")
            for error in analysis.validation_errors:
                lines.append(f"  ✗ {error}")
        else:
            lines.append("Status: Valid")

        return "\n".join(lines)


# Export main classes and enums
__all__ = [
    "DependencyResolver",
    "DependencyAnalysis",
    "BuildDependency",
    "SourceType",
    "CompilerRequirement",
]
