"""Unit tests for the AST-based documentation parser."""

import ast
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Add the docs framework to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../docs/framework'))

from generators.ast_parser import (
    DocstringParser,
    DocumentationNode,
    MemberType,
    Parameter,
    PythonASTParser,
    extract_module_info,
    extract_package_info,
)


class TestDocstringParser:
    """Test the docstring parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocstringParser()

    def test_parse_empty_docstring(self):
        """Test parsing empty/None docstring."""
        result = self.parser.parse(None)
        assert result == {}

        result = self.parser.parse("")
        assert result == {}

        result = self.parser.parse("   ")
        assert result == {}

    def test_parse_simple_docstring(self):
        """Test parsing a simple docstring."""
        docstring = "This is a simple function."
        result = self.parser.parse(docstring)

        assert result['summary'] == "This is a simple function."
        assert result['description'] == ""
        assert result['parameters'] == {}
        assert result['returns'] is None
        assert result['raises'] == []
        assert result['examples'] == []

    def test_parse_complex_docstring(self):
        """Test parsing a complex docstring with all sections."""
        docstring = """
        Process a document with advanced options.

        This function processes documents using various algorithms
        and returns the processed result.

        Args:
            document: The document to process
            options: Processing options dictionary
            verbose: Whether to enable verbose output

        Returns:
            ProcessedDocument: The processed document result

        Raises:
            ValueError: If document is invalid
            ProcessingError: If processing fails

        Examples:
            >>> result = process_document(doc, {'method': 'fast'})
            >>> print(result.status)
            'completed'

        Notes:
            This function requires at least 1GB of memory.
        """

        result = self.parser.parse(docstring)

        assert result['summary'] == "Process a document with advanced options."
        assert "This function processes documents" in result['description']
        assert 'document' in result['parameters']
        assert 'options' in result['parameters']
        assert 'verbose' in result['parameters']
        assert "The processed document result" in result['returns']
        assert len(result['raises']) == 2
        assert len(result['examples']) == 1
        assert "1GB of memory" in result['notes']

    def test_parse_parameters_without_types(self):
        """Test parsing parameters without type information."""
        docstring = """
        Test function.

        Args:
            param1: First parameter
            param2: Second parameter with
                multi-line description
            param3: Third parameter
        """

        result = self.parser.parse(docstring)
        params = result['parameters']

        assert 'param1' in params
        assert params['param1'] == "First parameter"
        assert 'param2' in params
        assert "multi-line description" in params['param2']
        assert 'param3' in params
        assert params['param3'] == "Third parameter"

    def test_parse_malformed_docstring(self):
        """Test parsing malformed docstring."""
        docstring = """
        This is malformed.
        Args:
        Returns: Something
        Raises:
            ValueError
        """

        result = self.parser.parse(docstring)
        # Should not crash and should parse what it can
        assert result['summary'] == "This is malformed."
        assert result['returns'] == "Something"

    def test_parse_edge_cases(self):
        """Test edge cases in docstring parsing."""
        # Docstring with only colons
        result = self.parser.parse("Args: : : Returns:")
        assert isinstance(result, dict)

        # Very long docstring
        long_docstring = "A" * 10000
        result = self.parser.parse(long_docstring)
        assert result['summary'] == long_docstring

        # Unicode characters
        unicode_docstring = "Function with émojis: 🚀🐍"
        result = self.parser.parse(unicode_docstring)
        assert "émojis" in result['summary']


class TestPythonASTParser:
    """Test the Python AST parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonASTParser(include_private=False)
        self.parser_with_private = PythonASTParser(include_private=True)

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = '''
def hello_world(name: str = "World") -> str:
    """Say hello to someone.

    Args:
        name: The name to greet

    Returns:
        A greeting string
    """
    return f"Hello, {name}!"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)

                assert result.member_type == MemberType.MODULE
                assert len(result.children) == 1

                func = result.children[0]
                assert func.name == "hello_world"
                assert func.member_type == MemberType.FUNCTION
                assert func.signature == "hello_world(name: str) -> str"
                assert len(func.parameters) == 1
                assert func.parameters[0].name == "name"
                assert func.parameters[0].annotation == "str"
                assert func.parameters[0].default == "'World'"
                assert func.return_annotation == "str"
                assert "Say hello to someone" in func.docstring
            finally:
                os.unlink(f.name)

    def test_parse_class_with_methods(self):
        """Test parsing a class with methods."""
        code = '''
class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2):
        """Initialize calculator.

        Args:
            precision: Number of decimal places
        """
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @property
    def current_precision(self) -> int:
        """Get current precision."""
        return self.precision

    def _private_method(self):
        """This is private."""
        pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)

                assert len(result.children) == 1
                calc_class = result.children[0]

                assert calc_class.name == "Calculator"
                assert calc_class.member_type == MemberType.CLASS
                assert len(calc_class.children) == 3  # __init__, add, current_precision (no private method)

                # Check __init__ method
                init_method = next(c for c in calc_class.children if c.name == "__init__")
                assert init_method.member_type == MemberType.METHOD
                assert len(init_method.parameters) == 1  # self is typically excluded or handled specially

                # Check property
                prop = next(c for c in calc_class.children if c.name == "current_precision")
                assert prop.member_type == MemberType.PROPERTY
                assert prop.is_property

                # Private method should not be included
                private_methods = [c for c in calc_class.children if c.name == "_private_method"]
                assert len(private_methods) == 0

            finally:
                os.unlink(f.name)

    def test_parse_with_private_members(self):
        """Test parsing with private members included."""
        code = '''
def _private_func():
    """Private function."""
    pass

def public_func():
    """Public function."""
    pass

class _PrivateClass:
    """Private class."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                # Without private members
                result = self.parser.parse_file(f.name)
                assert len(result.children) == 1  # Only public_func

                # With private members
                result_with_private = self.parser_with_private.parse_file(f.name)
                assert len(result_with_private.children) == 3  # All members

            finally:
                os.unlink(f.name)

    def test_parse_complex_annotations(self):
        """Test parsing complex type annotations."""
        code = '''
from typing import Optional, Union

def complex_function(
    items: List[Dict[str, Union[int, str]]],
    callback: Optional[callable] = None,
    options: Dict[str, Any] = None
) -> Tuple[List[str], Dict[str, int]]:
    """Function with complex annotations."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                func = result.children[0]

                assert len(func.parameters) == 3

                # Check complex parameter annotations
                items_param = func.parameters[0]
                assert items_param.name == "items"
                assert "List" in items_param.annotation
                assert "Dict" in items_param.annotation

                # Check return annotation
                assert "Tuple" in func.return_annotation
                assert "List" in func.return_annotation

            finally:
                os.unlink(f.name)

    def test_parse_decorators(self):
        """Test parsing function decorators."""
        code = '''
from functools import wraps

@property
@wraps(some_func)
def decorated_function():
    """Function with decorators."""
    pass

class MyClass:
    @staticmethod
    def static_method():
        """Static method."""
        pass

    @classmethod
    def class_method(cls):
        """Class method."""
        pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)

                func = result.children[0]
                assert "property" in func.decorators
                assert "wraps" in func.decorators[1] or "wraps(some_func)" in func.decorators

                cls = result.children[1]
                static_method = next(c for c in cls.children if c.name == "static_method")
                assert "staticmethod" in static_method.decorators

                class_method = next(c for c in cls.children if c.name == "class_method")
                assert "classmethod" in class_method.decorators

            finally:
                os.unlink(f.name)

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_file("/nonexistent/file.py")

    def test_parse_syntax_error(self):
        """Test parsing file with syntax errors."""
        code = '''
def broken_function(
    missing_closing_paren
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                with pytest.raises(SyntaxError):
                    self.parser.parse_file(f.name)
            finally:
                os.unlink(f.name)

    def test_parse_unicode_file(self):
        """Test parsing file with unicode content."""
        code = '''# -*- coding: utf-8 -*-
def función_con_tildes(parámetro: str) -> str:
    """Función con caracteres especiales: ñáéíóú."""
    return f"Hola, {parámetro}! 🚀"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                func = result.children[0]
                assert func.name == "función_con_tildes"
                assert "ñáéíóú" in func.docstring

            finally:
                os.unlink(f.name)

    def test_parse_directory(self):
        """Test parsing entire directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple Python files
            (temp_path / "module1.py").write_text('''
def func1():
    """Function in module 1."""
    pass
''')

            (temp_path / "module2.py").write_text('''
class Class2:
    """Class in module 2."""
    pass
''')

            # Create subdirectory
            subdir = temp_path / "subpackage"
            subdir.mkdir()
            (subdir / "module3.py").write_text('''
CONSTANT = "value"
''')

            # Non-Python file (should be ignored)
            (temp_path / "README.md").write_text("# Documentation")

            # Parse directory recursively
            results = self.parser.parse_directory(temp_path, recursive=True)

            # Should find all 3 Python files
            assert len(results) == 3

            module_names = [r.name for r in results]
            assert "module1" in module_names
            assert "module2" in module_names
            assert "module3" in module_names

    def test_parse_directory_non_recursive(self):
        """Test parsing directory non-recursively."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Python file in root
            (temp_path / "root_module.py").write_text('''
def root_func():
    """Root function."""
    pass
''')

            # Create subdirectory with Python file
            subdir = temp_path / "subpackage"
            subdir.mkdir()
            (subdir / "sub_module.py").write_text('''
def sub_func():
    """Sub function."""
    pass
''')

            # Parse directory non-recursively
            results = self.parser.parse_directory(temp_path, recursive=False)

            # Should only find root module
            assert len(results) == 1
            assert results[0].name == "root_module"

    def test_parse_directory_not_exists(self):
        """Test parsing non-existent directory."""
        with pytest.raises(NotADirectoryError):
            self.parser.parse_directory("/nonexistent/directory")

    def test_module_constants(self):
        """Test parsing module-level constants."""
        code = '''
"""Module with constants."""

PUBLIC_CONSTANT = "public value"
_PRIVATE_CONSTANT = "private value"
NUMERIC_CONSTANT = 42
BOOLEAN_CONSTANT = True
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)

                # Should find public constants only
                constants = [c for c in result.children if c.member_type == MemberType.CONSTANT]
                assert len(constants) >= 1

                public_const = next(c for c in constants if c.name == "PUBLIC_CONSTANT")
                assert not public_const.is_private

            finally:
                os.unlink(f.name)

    def test_edge_case_empty_file(self):
        """Test parsing empty Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                assert result.member_type == MemberType.MODULE
                assert len(result.children) == 0

            finally:
                os.unlink(f.name)

    def test_edge_case_only_comments(self):
        """Test parsing file with only comments."""
        code = '''
# This is a comment
# Another comment
"""This is a module docstring."""
# More comments
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                assert result.member_type == MemberType.MODULE
                assert result.docstring == "This is a module docstring."
                assert len(result.children) == 0

            finally:
                os.unlink(f.name)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_extract_module_info(self):
        """Test extract_module_info convenience function."""
        code = '''
def test_func():
    """Test function."""
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = extract_module_info(f.name)
                assert result.member_type == MemberType.MODULE
                assert len(result.children) == 1

            finally:
                os.unlink(f.name)

    def test_extract_package_info(self):
        """Test extract_package_info convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "test_module.py").write_text('''
def test_func():
    """Test function."""
    pass
''')

            results = extract_package_info(temp_path)
            assert len(results) == 1
            assert results[0].name == "test_module"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonASTParser()

    def test_binary_file_handling(self):
        """Test handling of binary files."""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            # Write binary data
            f.write(b'\x00\x01\x02\x03')
            f.flush()

            try:
                # Should handle encoding errors gracefully
                result = self.parser.parse_file(f.name)
                # If it doesn't crash, that's good
                assert result is not None

            except UnicodeDecodeError:
                # This is also acceptable
                pass
            finally:
                os.unlink(f.name)

    def test_very_large_file(self):
        """Test handling of large files."""
        # Create a large but valid Python file
        code = '''
def generated_function_{i}():
    """Generated function {i}."""
    return {i}
'''

        large_code = "\n".join([code.format(i=i) for i in range(100)])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                assert result.member_type == MemberType.MODULE
                assert len(result.children) == 100

            finally:
                os.unlink(f.name)

    def test_nested_classes_and_functions(self):
        """Test parsing nested classes and functions."""
        code = '''
class OuterClass:
    """Outer class."""

    class InnerClass:
        """Inner class."""

        def inner_method(self):
            """Inner method."""
            def nested_function():
                """Nested function."""
                pass
            return nested_function

    def outer_method(self):
        """Outer method."""
        pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = self.parser.parse_file(f.name)
                outer_class = result.children[0]

                # Should find outer class
                assert outer_class.name == "OuterClass"

                # Should find inner class and outer method
                inner_class = next(c for c in outer_class.children if c.name == "InnerClass")
                assert inner_class is not None

                outer_method = next(c for c in outer_class.children if c.name == "outer_method")
                assert outer_method is not None

                # Should find inner method (nested function typically not captured at this level)
                inner_method = next(c for c in inner_class.children if c.name == "inner_method")
                assert inner_method is not None

            finally:
                os.unlink(f.name)
