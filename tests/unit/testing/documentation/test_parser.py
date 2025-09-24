"""
Comprehensive unit tests for TestFileParser.

Tests all edge cases including empty files, syntax errors, encoding issues,
malformed docstrings, and complex decorator scenarios.
"""

import ast
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import chardet

from src.python.workspace_qdrant_mcp.testing.documentation.parser import (
    TestFileParser,
    TestFileInfo,
    TestMetadata,
    TestType,
    ParameterInfo,
    DecoratorInfo,
    TestMetadataVisitor
)


class TestTestFileParser:
    """Test TestFileParser with comprehensive edge case coverage."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = TestFileParser(max_file_size=1024)  # Small size for testing

    def test_parse_simple_test_file(self):
        """Test parsing a simple test file."""
        test_content = '''
def test_simple():
    """A simple test function."""
    assert True

def test_async():
    """An async test function."""
    pass

@pytest.mark.parametrize("value", [1, 2, 3])
def test_parametrized(value):
    """A parametrized test."""
    assert value > 0
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            assert len(file_info.tests) == 3
            assert file_info.parse_errors == []
            assert file_info.encoding == 'utf-8'
            assert file_info.total_lines > 0

            # Check specific tests
            simple_test = next(t for t in file_info.tests if t.name == 'test_simple')
            assert simple_test.docstring == "A simple test function."
            assert simple_test.test_type == TestType.UNKNOWN
            assert not simple_test.is_async

            Path(f.name).unlink()

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            file_info = self.parser.parse_file(f.name)

            assert len(file_info.tests) == 0
            assert file_info.total_lines == 0
            assert file_info.parse_errors == []

            Path(f.name).unlink()

    def test_parse_file_with_syntax_error(self):
        """Test parsing file with syntax errors."""
        test_content = '''
def test_broken(
    # Missing closing parenthesis
    assert True
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            assert len(file_info.tests) == 0
            assert len(file_info.parse_errors) == 1
            assert "Syntax error" in file_info.parse_errors[0]

            Path(f.name).unlink()

    def test_file_too_large(self):
        """Test handling of files that exceed size limit."""
        large_content = "# " + "x" * 2000  # Exceeds our 1KB limit

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            f.flush()

            with pytest.raises(ValueError, match="File too large"):
                self.parser.parse_file(f.name)

            Path(f.name).unlink()

    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with pytest.raises(ValueError, match="File does not exist"):
            self.parser.parse_file("/nonexistent/file.py")

    def test_directory_instead_of_file(self):
        """Test handling when path points to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Path is not a file"):
                self.parser.parse_file(temp_dir)

    @patch('chardet.detect')
    def test_encoding_detection_failure(self, mock_detect):
        """Test fallback to utf-8 when encoding detection fails."""
        mock_detect.side_effect = Exception("Detection failed")

        test_content = "def test_simple(): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            assert file_info.encoding == 'utf-8'
            assert len(file_info.tests) == 1

            Path(f.name).unlink()

    @patch('chardet.detect')
    def test_low_confidence_encoding_detection(self, mock_detect):
        """Test fallback when encoding confidence is low."""
        mock_detect.return_value = {'encoding': 'latin-1', 'confidence': 0.5}

        test_content = "def test_simple(): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            assert file_info.encoding == 'utf-8'  # Should fallback due to low confidence

            Path(f.name).unlink()

    def test_complex_decorators(self):
        """Test parsing of complex pytest decorators."""
        test_content = '''
import pytest

@pytest.mark.parametrize("a,b,expected", [(1,2,3), (2,3,5)])
@pytest.mark.slow
@pytest.mark.integration
def test_complex_decorators(a, b, expected):
    """Test with multiple complex decorators."""
    assert a + b == expected

@pytest.mark.xfail(reason="Known bug")
def test_expected_failure():
    """This test is expected to fail."""
    assert False

@pytest.mark.skip(reason="Not implemented")
def test_skipped():
    """This test is skipped."""
    pass

@pytest.fixture
def sample_fixture():
    """A test fixture."""
    return "fixture_value"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            # Check fixture is not included in tests
            test_names = [t.name for t in file_info.tests]
            assert 'sample_fixture' not in test_names
            assert 'sample_fixture' in file_info.fixtures

            # Check parametrized test
            param_test = next(t for t in file_info.tests if t.name == 'test_complex_decorators')
            assert param_test.is_parametrized
            assert 'slow' in param_test.marks
            assert 'integration' in param_test.marks
            assert len(param_test.decorators) == 3

            # Check expected failure
            xfail_test = next(t for t in file_info.tests if t.name == 'test_expected_failure')
            assert xfail_test.expected_to_fail

            # Check skipped test
            skip_test = next(t for t in file_info.tests if t.name == 'test_skipped')
            assert skip_test.skip_reason == "Not implemented"

            Path(f.name).unlink()

    def test_async_functions(self):
        """Test parsing of async test functions."""
        test_content = '''
import asyncio
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """An async test function."""
    await asyncio.sleep(0.1)
    assert True

class TestAsyncClass:
    @pytest.mark.asyncio
    async def test_async_method(self):
        """An async test method."""
        assert True
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            async_test = next(t for t in file_info.tests if t.name == 'test_async_function')
            assert async_test.is_async
            assert 'asyncio' in async_test.marks

            async_method = next(t for t in file_info.tests if t.name == 'test_async_method')
            assert async_method.is_async

            Path(f.name).unlink()

    def test_test_type_classification(self):
        """Test automatic test type classification."""
        test_content = '''
def test_unit_something():
    """A unit test."""
    pass

def test_integration_database():
    """An integration test."""
    pass

def test_e2e_workflow():
    """An end-to-end test."""
    pass

def test_performance_benchmark():
    """A performance test."""
    pass

@pytest.mark.smoke
def test_basic_functionality():
    """A smoke test."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            unit_test = next(t for t in file_info.tests if t.name == 'test_unit_something')
            assert unit_test.test_type == TestType.UNIT

            integration_test = next(t for t in file_info.tests if t.name == 'test_integration_database')
            assert integration_test.test_type == TestType.INTEGRATION

            e2e_test = next(t for t in file_info.tests if t.name == 'test_e2e_workflow')
            assert e2e_test.test_type == TestType.E2E

            perf_test = next(t for t in file_info.tests if t.name == 'test_performance_benchmark')
            assert perf_test.test_type == TestType.PERFORMANCE

            Path(f.name).unlink()

    def test_complex_parameters_and_types(self):
        """Test parsing functions with complex parameters and type annotations."""
        test_content = '''
from typing import List, Dict, Optional

def test_typed_parameters(
    items: List[str],
    mapping: Dict[str, int],
    optional_value: Optional[bool] = None,
    *args,
    **kwargs
):
    """Test with complex type annotations."""
    pass

def test_default_values(
    count: int = 10,
    name: str = "default",
    enabled: bool = True
):
    """Test with default parameter values."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            typed_test = next(t for t in file_info.tests if t.name == 'test_typed_parameters')
            assert len(typed_test.parameters) >= 3  # May include *args, **kwargs

            items_param = next(p for p in typed_test.parameters if p.name == 'items')
            assert items_param.type_annotation == 'List[str]'

            default_test = next(t for t in file_info.tests if t.name == 'test_default_values')
            count_param = next(p for p in default_test.parameters if p.name == 'count')
            assert count_param.type_annotation == 'int'

            Path(f.name).unlink()

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        test_content = '''
def test_simple():
    """Simple test with complexity 1."""
    assert True

def test_complex():
    """Complex test with control flow."""
    for i in range(10):
        if i % 2 == 0:
            try:
                with open("test.txt") as f:
                    content = f.read()
                    if content:
                        assert len(content) > 0
            except FileNotFoundError:
                pass
            finally:
                pass
        else:
            while i > 0:
                i -= 1

@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.slow
@pytest.fixture
def test_with_decorators(a):
    """Test with multiple decorators affecting complexity."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            simple_test = next(t for t in file_info.tests if t.name == 'test_simple')
            assert simple_test.complexity_score == 1

            complex_test = next(t for t in file_info.tests if t.name == 'test_complex')
            assert complex_test.complexity_score > 1
            assert complex_test.complexity_score <= 10  # Capped at 10

            decorator_test = next(t for t in file_info.tests if t.name == 'test_with_decorators')
            assert decorator_test.complexity_score > 1  # Has decorators and parameters

            Path(f.name).unlink()

    def test_malformed_docstrings(self):
        """Test handling of malformed or missing docstrings."""
        test_content = '''
def test_no_docstring():
    pass

def test_empty_docstring():
    """"""
    pass

def test_multiline_docstring():
    """
    This is a multiline docstring
    with multiple lines of text.

    It includes blank lines and formatting.
    """
    pass

def test_weird_docstring():
    """Single line but with \n weird \\t characters."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            no_doc_test = next(t for t in file_info.tests if t.name == 'test_no_docstring')
            assert no_doc_test.docstring is None

            empty_doc_test = next(t for t in file_info.tests if t.name == 'test_empty_docstring')
            assert empty_doc_test.docstring == ""

            multiline_test = next(t for t in file_info.tests if t.name == 'test_multiline_docstring')
            assert "This is a multiline docstring" in multiline_test.docstring
            assert "multiple lines" in multiline_test.docstring

            weird_test = next(t for t in file_info.tests if t.name == 'test_weird_docstring')
            assert weird_test.docstring is not None
            assert "weird" in weird_test.docstring

            Path(f.name).unlink()

    def test_parse_directory(self):
        """Test parsing entire directory of test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple test files
            (temp_path / "test_file1.py").write_text('''
def test_one():
    """Test one."""
    pass
''')

            (temp_path / "test_file2.py").write_text('''
def test_two():
    """Test two."""
    pass

def test_three():
    """Test three."""
    pass
''')

            # Create non-test file (should be ignored)
            (temp_path / "not_a_test.py").write_text('''
def regular_function():
    pass
''')

            # Create file with syntax error
            (temp_path / "test_broken.py").write_text('''
def test_broken(
    # syntax error
    pass
''')

            file_infos = self.parser.parse_directory(temp_dir)

            # Should find 3 test files (including broken one)
            assert len(file_infos) == 3

            # Check that we found the correct files
            file_names = [fi.file_path.name for fi in file_infos]
            assert "test_file1.py" in file_names
            assert "test_file2.py" in file_names
            assert "test_broken.py" in file_names
            assert "not_a_test.py" not in file_names

            # Check test counts
            file1_info = next(fi for fi in file_infos if fi.file_path.name == "test_file1.py")
            assert len(file1_info.tests) == 1

            file2_info = next(fi for fi in file_infos if fi.file_path.name == "test_file2.py")
            assert len(file2_info.tests) == 2

            broken_info = next(fi for fi in file_infos if fi.file_path.name == "test_broken.py")
            assert len(broken_info.parse_errors) > 0

    def test_parse_directory_invalid_path(self):
        """Test parsing invalid directory paths."""
        with pytest.raises(ValueError, match="Invalid directory"):
            self.parser.parse_directory("/nonexistent/directory")

    @patch('builtins.open')
    def test_unicode_decode_error(self, mock_open_func):
        """Test handling of unicode decode errors."""
        mock_open_func.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            f.flush()

            with pytest.raises(UnicodeDecodeError):
                self.parser.parse_file(f.name)

            Path(f.name).unlink()

    def test_class_based_tests(self):
        """Test parsing class-based test structures."""
        test_content = '''
class TestUserManagement:
    """Tests for user management functionality."""

    def test_create_user(self):
        """Test user creation."""
        pass

    def test_delete_user(self):
        """Test user deletion."""
        pass

    def not_a_test_method(self):
        """This should not be detected as a test."""
        pass

class NotATestClass:
    """This is not a test class."""

    def some_method(self):
        pass

class TestEmpty:
    """Empty test class."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            # Should have 3 classes but only 2 tests
            assert len(file_info.classes) == 3
            assert len(file_info.tests) == 2

            # Check class names
            assert "TestUserManagement" in file_info.classes
            assert "NotATestClass" in file_info.classes
            assert "TestEmpty" in file_info.classes

            # Check test detection
            test_names = [t.name for t in file_info.tests]
            assert "test_create_user" in test_names
            assert "test_delete_user" in test_names
            assert "not_a_test_method" not in test_names
            assert "some_method" not in test_names

            Path(f.name).unlink()

    def test_imports_extraction(self):
        """Test extraction of import statements."""
        test_content = '''
import os
import sys
from pathlib import Path
from typing import List, Dict
from unittest.mock import patch, Mock
import pytest
from my_module import my_function

def test_something():
    """Test with imports."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            f.flush()

            file_info = self.parser.parse_file(f.name)

            # Check that imports were captured
            assert len(file_info.imports) > 0
            assert 'os' in file_info.imports
            assert 'sys' in file_info.imports
            assert 'from pathlib import Path' in file_info.imports
            assert 'from typing import List' in file_info.imports

            Path(f.name).unlink()


class TestTestMetadataVisitor:
    """Test TestMetadataVisitor directly for edge cases."""

    def test_visitor_with_complex_ast(self):
        """Test visitor with complex AST structures."""
        code = '''
import pytest
from typing import Optional

class TestComplexScenarios:
    @pytest.fixture(scope="session")
    def complex_fixture(self) -> Optional[str]:
        return "test_value"

    @pytest.mark.parametrize("param1,param2", [
        (1, "a"),
        (2, "b"),
        pytest.param(3, "c", marks=pytest.mark.slow)
    ])
    def test_with_complex_parametrize(self, param1: int, param2: str):
        """Complex parametrized test."""
        assert param1 > 0
        assert isinstance(param2, str)
'''

        tree = ast.parse(code)
        file_info = TestFileInfo(file_path=Path("test.py"))
        visitor = TestMetadataVisitor(file_info, {})
        visitor.visit(tree)

        # Should find the fixture
        assert "complex_fixture" in file_info.fixtures

        # Should find the test
        assert len(file_info.tests) == 1
        test = file_info.tests[0]
        assert test.name == "test_with_complex_parametrize"
        assert test.is_parametrized

        # Should have extracted decorators
        assert len(test.decorators) > 0

        # Should have parameters
        assert len(test.parameters) >= 2


class TestDataClasses:
    """Test dataclass structures for completeness."""

    def test_parameter_info_creation(self):
        """Test ParameterInfo dataclass."""
        param = ParameterInfo(
            name="test_param",
            type_annotation="str",
            default_value="'default'",
            is_fixture=True
        )

        assert param.name == "test_param"
        assert param.type_annotation == "str"
        assert param.default_value == "'default'"
        assert param.is_fixture is True

    def test_decorator_info_creation(self):
        """Test DecoratorInfo dataclass."""
        decorator = DecoratorInfo(
            name="pytest.mark.parametrize",
            args=["param1", "param2"],
            kwargs={"indirect": "True"}
        )

        assert decorator.name == "pytest.mark.parametrize"
        assert decorator.args == ["param1", "param2"]
        assert decorator.kwargs == {"indirect": "True"}

    def test_test_metadata_creation(self):
        """Test TestMetadata dataclass."""
        metadata = TestMetadata(
            name="test_function",
            docstring="Test docstring",
            file_path=Path("test.py"),
            line_number=10,
            test_type=TestType.UNIT,
            is_async=True,
            complexity_score=3
        )

        assert metadata.name == "test_function"
        assert metadata.docstring == "Test docstring"
        assert metadata.file_path == Path("test.py")
        assert metadata.line_number == 10
        assert metadata.test_type == TestType.UNIT
        assert metadata.is_async is True
        assert metadata.complexity_score == 3
        assert metadata.marks == set()  # Default empty set

    def test_test_file_info_creation(self):
        """Test TestFileInfo dataclass."""
        file_info = TestFileInfo(
            file_path=Path("test.py"),
            encoding="utf-8",
            total_lines=50
        )

        assert file_info.file_path == Path("test.py")
        assert file_info.encoding == "utf-8"
        assert file_info.total_lines == 50
        assert file_info.tests == []  # Default empty list
        assert file_info.parse_errors == []  # Default empty list