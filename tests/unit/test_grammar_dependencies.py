"""Unit tests for grammar dependency resolution system."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from common.core.grammar_dependencies import (
    DependencyResolver,
    DependencyAnalysis,
    BuildDependency,
    SourceType,
    CompilerRequirement,
)


@pytest.fixture
def temp_grammar_dir(tmp_path):
    """Create a temporary grammar directory structure."""
    grammar_dir = tmp_path / "tree-sitter-test"
    src_dir = grammar_dir / "src"
    src_dir.mkdir(parents=True)
    return grammar_dir


@pytest.fixture
def resolver():
    """Create a DependencyResolver instance."""
    return DependencyResolver()


class TestBuildDependency:
    """Tests for BuildDependency dataclass."""

    def test_build_dependency_creation(self, temp_grammar_dir):
        """Test creating a BuildDependency."""
        parser_path = temp_grammar_dir / "src" / "parser.c"
        dep = BuildDependency(
            source_type=SourceType.PARSER,
            path=parser_path,
            compiler_requirement=CompilerRequirement.C_COMPILER,
            compilation_order=0
        )

        assert dep.source_type == SourceType.PARSER
        assert dep.path == parser_path
        assert dep.compiler_requirement == CompilerRequirement.C_COMPILER
        assert dep.compilation_order == 0

    def test_build_dependency_string_representation(self, temp_grammar_dir):
        """Test string representation of BuildDependency."""
        parser_path = temp_grammar_dir / "src" / "parser.c"
        dep = BuildDependency(
            source_type=SourceType.PARSER,
            path=parser_path,
            compiler_requirement=CompilerRequirement.C_COMPILER
        )

        assert str(dep) == "parser: parser.c"


class TestDependencyAnalysis:
    """Tests for DependencyAnalysis dataclass."""

    def test_dependency_analysis_creation(self, temp_grammar_dir):
        """Test creating a DependencyAnalysis."""
        analysis = DependencyAnalysis(
            grammar_name="test",
            grammar_path=temp_grammar_dir
        )

        assert analysis.grammar_name == "test"
        assert analysis.grammar_path == temp_grammar_dir
        assert analysis.dependencies == []
        assert analysis.required_compilers == set()
        assert analysis.is_valid is True
        assert analysis.validation_errors == []

    def test_get_source_files(self, temp_grammar_dir):
        """Test getting source files in compilation order."""
        parser_path = temp_grammar_dir / "src" / "parser.c"
        scanner_path = temp_grammar_dir / "src" / "scanner.c"

        analysis = DependencyAnalysis(
            grammar_name="test",
            grammar_path=temp_grammar_dir,
            dependencies=[
                BuildDependency(
                    source_type=SourceType.SCANNER_C,
                    path=scanner_path,
                    compiler_requirement=CompilerRequirement.C_COMPILER,
                    compilation_order=1
                ),
                BuildDependency(
                    source_type=SourceType.PARSER,
                    path=parser_path,
                    compiler_requirement=CompilerRequirement.C_COMPILER,
                    compilation_order=0
                ),
            ]
        )

        source_files = analysis.get_source_files()
        assert source_files == [parser_path, scanner_path]

    def test_get_dependencies_by_type(self, temp_grammar_dir):
        """Test filtering dependencies by type."""
        parser_path = temp_grammar_dir / "src" / "parser.c"
        scanner_path = temp_grammar_dir / "src" / "scanner.c"

        analysis = DependencyAnalysis(
            grammar_name="test",
            grammar_path=temp_grammar_dir,
            dependencies=[
                BuildDependency(
                    source_type=SourceType.PARSER,
                    path=parser_path,
                    compiler_requirement=CompilerRequirement.C_COMPILER,
                    compilation_order=0
                ),
                BuildDependency(
                    source_type=SourceType.SCANNER_C,
                    path=scanner_path,
                    compiler_requirement=CompilerRequirement.C_COMPILER,
                    compilation_order=1
                ),
            ]
        )

        parser_deps = analysis.get_dependencies_by_type(SourceType.PARSER)
        assert len(parser_deps) == 1
        assert parser_deps[0].source_type == SourceType.PARSER

        scanner_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_C)
        assert len(scanner_deps) == 1
        assert scanner_deps[0].source_type == SourceType.SCANNER_C


class TestDependencyResolver:
    """Tests for DependencyResolver class."""

    def test_analyze_grammar_no_src_dir(self, resolver, tmp_path):
        """Test analysis when src directory doesn't exist."""
        # Create grammar dir without src directory
        grammar_dir = tmp_path / "tree-sitter-test"
        grammar_dir.mkdir()

        analysis = resolver.analyze_grammar(grammar_dir)

        assert not analysis.is_valid
        assert len(analysis.validation_errors) == 1
        assert "Source directory not found" in analysis.validation_errors[0]

    def test_analyze_grammar_no_parser_c(self, resolver, temp_grammar_dir):
        """Test analysis when parser.c doesn't exist."""
        # src dir exists but no parser.c
        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert not analysis.is_valid
        assert len(analysis.validation_errors) == 1
        assert "parser.c not found" in analysis.validation_errors[0]

    def test_analyze_grammar_parser_only(self, resolver, temp_grammar_dir):
        """Test analysis with only parser.c."""
        # Create parser.c
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        parser_c.write_text("// parser code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert analysis.is_valid
        assert len(analysis.dependencies) == 1
        assert analysis.dependencies[0].source_type == SourceType.PARSER
        assert analysis.dependencies[0].path == parser_c
        assert CompilerRequirement.C_COMPILER in analysis.required_compilers
        assert not analysis.has_external_scanner
        assert not analysis.needs_cpp

    def test_analyze_grammar_with_c_scanner(self, resolver, temp_grammar_dir):
        """Test analysis with C scanner."""
        # Create parser.c and scanner.c
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_c = src_dir / "scanner.c"
        parser_c.write_text("// parser code")
        scanner_c.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert analysis.is_valid
        assert len(analysis.dependencies) == 2
        assert CompilerRequirement.C_COMPILER in analysis.required_compilers
        assert analysis.has_external_scanner
        assert not analysis.needs_cpp

        # Check scanner dependency
        scanner_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_C)
        assert len(scanner_deps) == 1
        assert scanner_deps[0].path == scanner_c

    def test_analyze_grammar_with_cpp_scanner_cc(self, resolver, temp_grammar_dir):
        """Test analysis with C++ scanner (.cc)."""
        # Create parser.c and scanner.cc
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_cc = src_dir / "scanner.cc"
        parser_c.write_text("// parser code")
        scanner_cc.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert analysis.is_valid
        assert len(analysis.dependencies) == 2
        assert CompilerRequirement.C_COMPILER in analysis.required_compilers
        assert CompilerRequirement.CPP_COMPILER in analysis.required_compilers
        assert analysis.has_external_scanner
        assert analysis.needs_cpp

        # Check scanner dependency
        scanner_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_CPP)
        assert len(scanner_deps) == 1
        assert scanner_deps[0].path == scanner_cc

    def test_analyze_grammar_with_cpp_scanner_cpp(self, resolver, temp_grammar_dir):
        """Test analysis with C++ scanner (.cpp)."""
        # Create parser.c and scanner.cpp
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_cpp = src_dir / "scanner.cpp"
        parser_c.write_text("// parser code")
        scanner_cpp.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert analysis.is_valid
        assert analysis.has_external_scanner
        assert analysis.needs_cpp

    def test_analyze_grammar_name_extraction(self, resolver, temp_grammar_dir):
        """Test grammar name extraction from path."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        parser_c.write_text("// parser code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        assert analysis.grammar_name == "test"  # From "tree-sitter-test"

    def test_analyze_grammar_custom_name(self, resolver, temp_grammar_dir):
        """Test using custom grammar name."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        parser_c.write_text("// parser code")

        analysis = resolver.analyze_grammar(temp_grammar_dir, grammar_name="custom")

        assert analysis.grammar_name == "custom"

    def test_validate_dependencies_valid(self, resolver, temp_grammar_dir):
        """Test validation with all requirements met."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        parser_c.write_text("// parser code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)
        valid, errors = resolver.validate_dependencies(
            analysis,
            available_c_compiler=True,
            available_cpp_compiler=False
        )

        assert valid
        assert len(errors) == 0

    def test_validate_dependencies_no_c_compiler(self, resolver, temp_grammar_dir):
        """Test validation without C compiler."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        parser_c.write_text("// parser code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)
        valid, errors = resolver.validate_dependencies(
            analysis,
            available_c_compiler=False,
            available_cpp_compiler=False
        )

        assert not valid
        assert len(errors) == 1
        assert "C compiler required" in errors[0]

    def test_validate_dependencies_no_cpp_compiler(self, resolver, temp_grammar_dir):
        """Test validation without C++ compiler when needed."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_cc = src_dir / "scanner.cc"
        parser_c.write_text("// parser code")
        scanner_cc.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)
        valid, errors = resolver.validate_dependencies(
            analysis,
            available_c_compiler=True,
            available_cpp_compiler=False
        )

        assert not valid
        assert len(errors) == 1
        assert "C++ scanner" in errors[0]
        assert "no C++ compiler" in errors[0]

    def test_validate_dependencies_invalid_structure(self, resolver, temp_grammar_dir):
        """Test validation with invalid grammar structure."""
        # No src directory
        analysis = resolver.analyze_grammar(temp_grammar_dir)
        valid, errors = resolver.validate_dependencies(
            analysis,
            available_c_compiler=True,
            available_cpp_compiler=True
        )

        assert not valid
        assert len(errors) > 0

    def test_get_compilation_summary(self, resolver, temp_grammar_dir):
        """Test getting compilation summary."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_c = src_dir / "scanner.c"
        parser_c.write_text("// parser code")
        scanner_c.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)
        summary = resolver.get_compilation_summary(analysis)

        assert "Grammar: test" in summary
        assert "Source files: 2" in summary
        assert "parser: parser.c" in summary
        assert "scanner_c: scanner.c" in summary
        assert "Required compilers: c" in summary
        assert "External scanner: C" in summary
        assert "Status: Valid" in summary

    def test_get_compilation_summary_invalid(self, resolver, tmp_path):
        """Test compilation summary for invalid grammar."""
        # Create grammar dir without src directory
        grammar_dir = tmp_path / "tree-sitter-test"
        grammar_dir.mkdir()

        analysis = resolver.analyze_grammar(grammar_dir)
        summary = resolver.get_compilation_summary(analysis)

        assert "Status: INVALID" in summary
        assert "Source directory not found" in summary

    def test_scanner_priority_c_over_cpp(self, resolver, temp_grammar_dir):
        """Test that scanner.c takes priority over scanner.cc if both exist."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_c = src_dir / "scanner.c"
        scanner_cc = src_dir / "scanner.cc"
        parser_c.write_text("// parser code")
        scanner_c.write_text("// C scanner")
        scanner_cc.write_text("// C++ scanner")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        # Should detect scanner.c first
        scanner_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_C)
        assert len(scanner_deps) == 1
        assert scanner_deps[0].path.name == "scanner.c"

        cpp_deps = analysis.get_dependencies_by_type(SourceType.SCANNER_CPP)
        assert len(cpp_deps) == 0

    def test_compilation_order(self, resolver, temp_grammar_dir):
        """Test that compilation order is correctly set."""
        src_dir = temp_grammar_dir / "src"
        parser_c = src_dir / "parser.c"
        scanner_c = src_dir / "scanner.c"
        parser_c.write_text("// parser code")
        scanner_c.write_text("// scanner code")

        analysis = resolver.analyze_grammar(temp_grammar_dir)

        # Parser should be first
        assert analysis.dependencies[0].compilation_order == 0
        assert analysis.dependencies[0].source_type == SourceType.PARSER

        # Scanner should be second
        assert analysis.dependencies[1].compilation_order == 1
        assert analysis.dependencies[1].source_type == SourceType.SCANNER_C
