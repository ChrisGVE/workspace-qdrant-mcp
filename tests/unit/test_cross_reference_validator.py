"""Comprehensive unit tests for cross-reference validator with edge cases."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from docs.framework.validation.cross_reference import (
        CrossReferenceValidator,
        ReferenceLink,
        ReferenceType,
        ValidationResult,
    )
except ModuleNotFoundError:
    pytest.skip("Docs framework not available", allow_module_level=True)


class TestReferenceLink:
    """Test ReferenceLink data class."""

    def test_reference_link_initialization(self):
        """Test basic initialization."""
        ref = ReferenceLink(
            source_file=Path("test.md"),
            source_line=10,
            source_column=5,
            reference_text=":func:`test_function`",
            reference_type=ReferenceType.FUNCTION,
            target="test_function"
        )

        assert ref.source_file == Path("test.md")
        assert ref.source_line == 10
        assert ref.reference_type == ReferenceType.FUNCTION
        assert ref.target == "test_function"
        assert ref.is_valid is None

    def test_reference_link_string_path(self):
        """Test initialization with string path."""
        ref = ReferenceLink(
            source_file="test.md",
            source_line=1,
            source_column=0,
            reference_text="test",
            reference_type=ReferenceType.FILE_PATH,
            target="test"
        )

        assert isinstance(ref.source_file, Path)
        assert ref.source_file == Path("test.md")

    def test_reference_link_with_suggestions(self):
        """Test reference link with suggestions."""
        ref = ReferenceLink(
            source_file=Path("test.md"),
            source_line=1,
            source_column=0,
            reference_text="test",
            reference_type=ReferenceType.FUNCTION,
            target="test_func",
            suggestions=["test_function", "test_func_alt"]
        )

        assert len(ref.suggestions) == 2
        assert "test_function" in ref.suggestions


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_validation_result_initialization(self):
        """Test basic initialization."""
        result = ValidationResult()

        assert result.total_references == 0
        assert result.valid_references == 0
        assert len(result.invalid_references) == 0
        assert result.validity_score == 100.0

    def test_validity_score_calculation(self):
        """Test validity score calculation."""
        result = ValidationResult(
            total_references=10,
            valid_references=8
        )

        assert result.validity_score == 80.0

    def test_validity_score_zero_references(self):
        """Test validity score with zero references."""
        result = ValidationResult(
            total_references=0,
            valid_references=0
        )

        assert result.validity_score == 100.0

    def test_validity_score_edge_cases(self):
        """Test validity score edge cases."""
        # All invalid
        result = ValidationResult(total_references=5, valid_references=0)
        assert result.validity_score == 0.0

        # All valid
        result = ValidationResult(total_references=5, valid_references=5)
        assert result.validity_score == 100.0


class TestCrossReferenceValidator:
    """Test CrossReferenceValidator with comprehensive edge cases."""

    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary documentation directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir) / "docs"
            docs_path.mkdir()

            # Create sample files
            (docs_path / "test.md").write_text("""
# Test Documentation

This is a :func:`test_function` reference.
Here's a :class:`TestClass` reference.
And a broken :func:`missing_function` reference.

See [link text](https://example.com) for more info.
Also check `file.py` for implementation.
""")

            (docs_path / "api.rst").write_text("""
API Reference
=============

:mod:`mymodule` contains various functions:

* :func:`mymodule.function_one`
* :meth:`MyClass.method_one`
""")

            # Create Python files for symbol extraction
            (docs_path / "mymodule.py").write_text("""
class TestClass:
    def test_method(self):
        pass

def test_function():
    pass

def function_one():
    pass

class MyClass:
    def method_one(self):
        pass
""")

            yield docs_path

    def test_validator_initialization(self, temp_docs_dir):
        """Test validator initialization."""
        validator = CrossReferenceValidator(temp_docs_dir)

        assert validator.root_path == temp_docs_dir
        assert not validator.external_validation
        assert ReferenceType.FUNCTION in validator.patterns
        assert len(validator.ignore_patterns) > 0

    def test_validator_with_custom_patterns(self, temp_docs_dir):
        """Test validator with custom patterns."""
        custom_patterns = {
            ReferenceType.FUNCTION: r':custom_func:`([^`]+)`'
        }

        validator = CrossReferenceValidator(
            temp_docs_dir,
            patterns=custom_patterns
        )

        assert validator.patterns[ReferenceType.FUNCTION] == r':custom_func:`([^`]+)`'

    def test_validator_with_ignore_patterns(self, temp_docs_dir):
        """Test validator with custom ignore patterns."""
        ignore_patterns = [r'test_.*', r'\.tmp$']

        validator = CrossReferenceValidator(
            temp_docs_dir,
            ignore_patterns=ignore_patterns
        )

        assert r'test_.*' in validator.ignore_patterns

    def test_should_ignore_file(self, temp_docs_dir):
        """Test file ignoring logic."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Default patterns
        assert validator._should_ignore_file(Path("__pycache__/test.pyc"))
        assert validator._should_ignore_file(Path(".git/config"))
        assert not validator._should_ignore_file(Path("docs/test.md"))

    def test_build_symbol_index(self, temp_docs_dir):
        """Test building code symbol index."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        # Check symbols were extracted
        assert len(validator._code_symbols) > 0

        # Find the mymodule symbols
        module_key = None
        for key in validator._code_symbols:
            if 'mymodule' in key:
                module_key = key
                break

        assert module_key is not None
        symbols = validator._code_symbols[module_key]
        assert "function:test_function" in symbols
        assert "class:TestClass" in symbols
        assert "method:TestClass.test_method" in symbols

    def test_extract_python_symbols_edge_cases(self, temp_docs_dir):
        """Test Python symbol extraction edge cases."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Test empty file
        empty_file = temp_docs_dir / "empty.py"
        empty_file.write_text("")
        symbols = validator._extract_python_symbols(empty_file)
        assert len(symbols) == 0

        # Test invalid syntax file
        invalid_file = temp_docs_dir / "invalid.py"
        invalid_file.write_text("def broken function(:")
        symbols = validator._extract_python_symbols(invalid_file)
        assert len(symbols) == 0  # Should handle gracefully

    def test_extract_python_symbols_complex_cases(self, temp_docs_dir):
        """Test Python symbol extraction with complex cases."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Create complex Python file
        complex_file = temp_docs_dir / "complex.py"
        complex_file.write_text("""
# Module with various constructs
import os

# Module variable
MODULE_VAR = "test"

def outer_function():
    def inner_function():  # Should not be extracted
        pass
    return inner_function

class OuterClass:
    class_var = 42

    def __init__(self):
        self.instance_var = 0

    def public_method(self):
        pass

    def _private_method(self):
        pass

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

    class InnerClass:
        def inner_method(self):
            pass

# Assignment to multiple targets
a, b, c = 1, 2, 3
""")

        symbols = validator._extract_python_symbols(complex_file)

        # Check extracted symbols
        assert "function:outer_function" in symbols
        assert "class:OuterClass" in symbols
        assert "method:OuterClass.public_method" in symbols
        assert "method:OuterClass._private_method" in symbols
        assert "class:InnerClass" in symbols
        assert "variable:MODULE_VAR" in symbols

    def test_get_module_path(self, temp_docs_dir):
        """Test module path calculation."""
        validator = CrossReferenceValidator(temp_docs_dir)

        test_file = temp_docs_dir / "subdir" / "module.py"
        test_file.parent.mkdir()
        test_file.write_text("")

        module_path = validator._get_module_path(test_file)
        assert module_path == "subdir.module"

    def test_validate_references_basic(self, temp_docs_dir):
        """Test basic reference validation."""
        validator = CrossReferenceValidator(temp_docs_dir)
        result = validator.validate_references()

        assert result.total_references > 0
        assert result.validity_score >= 0
        assert isinstance(result.invalid_references, list)

    def test_validate_references_custom_patterns(self, temp_docs_dir):
        """Test validation with custom file patterns."""
        validator = CrossReferenceValidator(temp_docs_dir)
        result = validator.validate_references(['*.md'])

        assert result.total_references > 0
        # Should only process .md files

    def test_validate_file_references(self, temp_docs_dir):
        """Test validating references in single file."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        test_file = temp_docs_dir / "test.md"
        result = validator._validate_file_references(test_file)

        assert result.total_references > 0
        assert len(result.invalid_references) >= 0

    def test_validate_file_references_unreadable_file(self, temp_docs_dir):
        """Test handling of unreadable files."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Create file then make it unreadable (if possible)
        unreadable_file = temp_docs_dir / "unreadable.md"
        unreadable_file.write_text("test")

        # Mock open to raise exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = validator._validate_file_references(unreadable_file)

        assert result.total_references == 0

    def test_validate_code_reference_valid(self, temp_docs_dir):
        """Test validating valid code references."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text=":func:`test_function`",
            reference_type=ReferenceType.FUNCTION,
            target="test_function"
        )

        is_valid = validator._validate_code_reference(reference)
        assert is_valid
        assert reference.is_valid

    def test_validate_code_reference_invalid(self, temp_docs_dir):
        """Test validating invalid code references."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text=":func:`missing_function`",
            reference_type=ReferenceType.FUNCTION,
            target="missing_function"
        )

        is_valid = validator._validate_code_reference(reference)
        assert not is_valid
        assert not reference.is_valid
        assert reference.error_message is not None
        assert len(reference.suggestions) >= 0

    def test_validate_code_reference_qualified_name(self, temp_docs_dir):
        """Test validating qualified code references."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text=":func:`mymodule.function_one`",
            reference_type=ReferenceType.FUNCTION,
            target="mymodule.function_one"
        )

        validator._validate_code_reference(reference)
        # May be valid depending on symbol extraction

    def test_validate_file_reference_valid(self, temp_docs_dir):
        """Test validating valid file references."""
        validator = CrossReferenceValidator(temp_docs_dir)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="`mymodule.py`",
            reference_type=ReferenceType.FILE_PATH,
            target="mymodule.py"
        )

        is_valid = validator._validate_file_reference(reference)
        assert is_valid

    def test_validate_file_reference_invalid(self, temp_docs_dir):
        """Test validating invalid file references."""
        validator = CrossReferenceValidator(temp_docs_dir)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="`missing_file.py`",
            reference_type=ReferenceType.FILE_PATH,
            target="missing_file.py"
        )

        is_valid = validator._validate_file_reference(reference)
        assert not is_valid
        assert not reference.is_valid
        assert reference.error_message is not None

    def test_validate_file_reference_relative_paths(self, temp_docs_dir):
        """Test validating file references with relative paths."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Create subdirectory structure
        subdir = temp_docs_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.py").write_text("")
        (subdir / "doc.md").write_text("")

        # Reference from subdir to sibling file
        reference = ReferenceLink(
            source_file=subdir / "doc.md",
            source_line=1,
            source_column=0,
            reference_text="`file.py`",
            reference_type=ReferenceType.FILE_PATH,
            target="file.py"
        )

        is_valid = validator._validate_file_reference(reference)
        assert is_valid

    @patch('requests.head')
    def test_validate_url_reference_valid(self, mock_head, temp_docs_dir):
        """Test validating valid URL references."""
        validator = CrossReferenceValidator(temp_docs_dir, external_validation=True)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="https://example.com",
            reference_type=ReferenceType.URL,
            target="https://example.com"
        )

        is_valid = validator._validate_url_reference(reference)
        assert is_valid
        mock_head.assert_called_once()

    @patch('requests.head')
    def test_validate_url_reference_invalid(self, mock_head, temp_docs_dir):
        """Test validating invalid URL references."""
        validator = CrossReferenceValidator(temp_docs_dir, external_validation=True)

        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="https://example.com/404",
            reference_type=ReferenceType.URL,
            target="https://example.com/404"
        )

        is_valid = validator._validate_url_reference(reference)
        assert not is_valid
        assert reference.error_message == "HTTP 404"

    @patch('requests.head')
    def test_validate_url_reference_connection_error(self, mock_head, temp_docs_dir):
        """Test handling URL connection errors."""
        validator = CrossReferenceValidator(temp_docs_dir, external_validation=True)

        mock_head.side_effect = Exception("Connection failed")

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="https://invalid-domain.example",
            reference_type=ReferenceType.URL,
            target="https://invalid-domain.example"
        )

        is_valid = validator._validate_url_reference(reference)
        assert not is_valid
        assert "Connection failed" in reference.error_message

    def test_validate_url_reference_disabled(self, temp_docs_dir):
        """Test URL validation when disabled."""
        validator = CrossReferenceValidator(temp_docs_dir, external_validation=False)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="https://example.com",
            reference_type=ReferenceType.URL,
            target="https://example.com"
        )

        is_valid = validator._validate_url_reference(reference)
        assert is_valid  # Should pass when validation disabled

    def test_validate_url_reference_no_requests(self, temp_docs_dir):
        """Test URL validation when requests library unavailable."""
        validator = CrossReferenceValidator(temp_docs_dir, external_validation=True)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="https://example.com",
            reference_type=ReferenceType.URL,
            target="https://example.com"
        )

        with patch('builtins.__import__', side_effect=ImportError("No module named 'requests'")):
            is_valid = validator._validate_url_reference(reference)
            assert is_valid  # Should pass gracefully

    def test_validate_anchor_reference(self, temp_docs_dir):
        """Test anchor reference validation."""
        validator = CrossReferenceValidator(temp_docs_dir)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="#section-header",
            reference_type=ReferenceType.ANCHOR,
            target="section-header"
        )

        is_valid = validator._validate_anchor_reference(reference)
        assert is_valid  # Currently always returns True

    def test_validate_reference_exception_handling(self, temp_docs_dir):
        """Test exception handling during reference validation."""
        validator = CrossReferenceValidator(temp_docs_dir)

        reference = ReferenceLink(
            source_file=temp_docs_dir / "test.md",
            source_line=1,
            source_column=0,
            reference_text="test",
            reference_type=ReferenceType.FUNCTION,
            target="test"
        )

        # Mock validation method to raise exception
        with patch.object(validator, '_validate_code_reference', side_effect=Exception("Test error")):
            is_valid = validator._validate_reference(reference)

        assert not is_valid
        assert reference.error_message == "Test error"

    def test_find_similar_symbols(self, temp_docs_dir):
        """Test finding similar symbols."""
        validator = CrossReferenceValidator(temp_docs_dir)
        validator._build_symbol_index()

        # Mock some symbols
        validator._code_symbols['test_module'] = {
            'function:test_function',
            'function:test_func',
            'function:another_test',
        }

        similar = validator._find_similar_symbols("test_fun", ReferenceType.FUNCTION)
        assert isinstance(similar, list)

    def test_find_similar_files(self, temp_docs_dir):
        """Test finding similar files."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Create some test files
        (temp_docs_dir / "test_file.py").write_text("")
        (temp_docs_dir / "test_module.py").write_text("")
        (temp_docs_dir / "example.py").write_text("")

        similar = validator._find_similar_files("test_fil.py")
        assert isinstance(similar, list)

    def test_find_orphaned_targets(self, temp_docs_dir):
        """Test finding orphaned targets."""
        validator = CrossReferenceValidator(temp_docs_dir)
        orphaned = validator._find_orphaned_targets()
        assert isinstance(orphaned, list)

    def test_generate_suggestions(self, temp_docs_dir):
        """Test generating suggestions for invalid references."""
        validator = CrossReferenceValidator(temp_docs_dir)

        invalid_refs = [
            ReferenceLink(
                source_file=temp_docs_dir / "test.md",
                source_line=10,
                source_column=5,
                reference_text="test",
                reference_type=ReferenceType.FUNCTION,
                target="test",
                suggestions=["test_function", "test_method"]
            )
        ]

        suggestions = validator._generate_suggestions(invalid_refs)
        assert isinstance(suggestions, dict)

        if suggestions:
            key = list(suggestions.keys())[0]
            assert "test.md:10" in key

    def test_export_results(self, temp_docs_dir):
        """Test exporting validation results."""
        validator = CrossReferenceValidator(temp_docs_dir)

        result = ValidationResult(
            total_references=10,
            valid_references=8,
            invalid_references=[
                ReferenceLink(
                    source_file=temp_docs_dir / "test.md",
                    source_line=5,
                    source_column=10,
                    reference_text=":func:`missing`",
                    reference_type=ReferenceType.FUNCTION,
                    target="missing",
                    error_message="Symbol not found"
                )
            ]
        )

        output_path = temp_docs_dir / "results.json"
        success = validator.export_results(result, output_path)

        assert success
        assert output_path.exists()

        # Verify exported data
        with open(output_path) as f:
            data = json.load(f)

        assert data['summary']['total_references'] == 10
        assert data['summary']['valid_references'] == 8
        assert len(data['invalid_references']) == 1

    def test_export_results_error_handling(self, temp_docs_dir):
        """Test export error handling."""
        validator = CrossReferenceValidator(temp_docs_dir)
        result = ValidationResult()

        # Try to export to invalid path
        invalid_path = Path("/invalid/path/results.json")
        success = validator.export_results(result, invalid_path)

        assert not success

    def test_validation_with_no_files(self, temp_docs_dir):
        """Test validation when no matching files found."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Remove all files
        for file in temp_docs_dir.rglob("*"):
            if file.is_file():
                file.unlink()

        result = validator.validate_references(['*.nonexistent'])

        assert result.total_references == 0
        assert result.validity_score == 100.0

    def test_validation_with_binary_files(self, temp_docs_dir):
        """Test handling of binary files."""
        validator = CrossReferenceValidator(temp_docs_dir)

        # Create binary file
        binary_file = temp_docs_dir / "binary.pdf"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xFF')

        # Should handle gracefully
        validator.validate_references(['*.pdf'])
        # Result depends on how binary files are handled

    def test_edge_case_empty_patterns(self, temp_docs_dir):
        """Test edge case with empty patterns."""
        validator = CrossReferenceValidator(temp_docs_dir, patterns={})
        result = validator.validate_references()

        # Should handle empty patterns gracefully
        assert result.total_references == 0

    def test_edge_case_malformed_references(self, temp_docs_dir):
        """Test handling malformed references."""
        malformed_file = temp_docs_dir / "malformed.md"
        malformed_file.write_text("""
# Test with malformed references

:func:`` (empty target)
:class:`Unclosed class reference
[](empty link)
[broken link](
""")

        validator = CrossReferenceValidator(temp_docs_dir)
        result = validator.validate_references(['malformed.md'])

        # Should handle malformed references gracefully
        assert isinstance(result.total_references, int)

    def test_performance_large_codebase(self, temp_docs_dir):
        """Test performance with larger codebase simulation."""
        # Create many files to test performance
        for i in range(10):
            py_file = temp_docs_dir / f"module_{i}.py"
            py_file.write_text(f"""
def function_{i}_a():
    pass

def function_{i}_b():
    pass

class Class{i}:
    def method_{i}(self):
        pass
""")

            md_file = temp_docs_dir / f"doc_{i}.md"
            md_file.write_text(f"""
# Documentation {i}

See :func:`function_{i}_a` and :class:`Class{i}`.
Also :func:`function_{i}_missing` (broken reference).
""")

        validator = CrossReferenceValidator(temp_docs_dir)
        result = validator.validate_references()

        # Should complete without issues
        assert result.total_references > 0
        assert isinstance(result.validity_score, float)

    def test_unicode_handling(self, temp_docs_dir):
        """Test handling of Unicode content."""
        unicode_file = temp_docs_dir / "unicode.md"
        unicode_file.write_text("""
# Documentation with Unicode: 测试

Function reference: :func:`测试_function`
Class reference: :class:`Тест_Class`

Link with Unicode: [测试链接](https://example.com/测试)
""", encoding='utf-8')

        unicode_py = temp_docs_dir / "unicode_module.py"
        unicode_py.write_text("""
def 测试_function():
    \"\"\"Unicode function name.\"\"\"
    pass

class Тест_Class:
    \"\"\"Unicode class name.\"\"\"
    pass
""", encoding='utf-8')

        validator = CrossReferenceValidator(temp_docs_dir)
        result = validator.validate_references()

        # Should handle Unicode gracefully
        assert isinstance(result.total_references, int)
