"""
Test File Parser

Provides AST-based parsing of Python test files to extract metadata
including test functions, docstrings, decorators, and parameters.
"""

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
import logging
from enum import Enum
import chardet

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of test functions."""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SMOKE = "smoke"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


@dataclass
class ParameterInfo:
    """Information about test function parameters."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_fixture: bool = False


@dataclass
class DecoratorInfo:
    """Information about test function decorators."""
    name: str
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestMetadata:
    """Complete metadata for a test function."""
    name: str
    docstring: Optional[str]
    file_path: Path
    line_number: int
    test_type: TestType
    decorators: List[DecoratorInfo] = field(default_factory=list)
    parameters: List[ParameterInfo] = field(default_factory=list)
    is_async: bool = False
    is_parametrized: bool = False
    expected_to_fail: bool = False
    skip_reason: Optional[str] = None
    marks: Set[str] = field(default_factory=set)
    complexity_score: int = 1


@dataclass
class TestFileInfo:
    """Information about a complete test file."""
    file_path: Path
    tests: List[TestMetadata] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    encoding: str = "utf-8"
    total_lines: int = 0
    test_coverage: Optional[float] = None


class TestFileParser:
    """
    AST-based parser for Python test files.

    Extracts comprehensive metadata from test files including:
    - Test function definitions and metadata
    - Decorators and their arguments
    - Function parameters and type hints
    - Docstrings and test categorization
    - Import statements and fixtures
    """

    def __init__(self, max_file_size: int = 10 * 1024 * 1024):  # 10MB limit
        """
        Initialize parser with safety limits.

        Args:
            max_file_size: Maximum file size to parse (bytes)
        """
        self.max_file_size = max_file_size
        self._test_type_keywords = {
            TestType.UNIT: {'unit', 'test_unit', 'unittest'},
            TestType.INTEGRATION: {'integration', 'test_integration', 'integr'},
            TestType.FUNCTIONAL: {'functional', 'test_functional', 'func'},
            TestType.E2E: {'e2e', 'end_to_end', 'test_e2e', 'endtoend'},
            TestType.PERFORMANCE: {'perf', 'performance', 'bench', 'benchmark'},
            TestType.SMOKE: {'smoke', 'test_smoke', 'sanity'},
            TestType.REGRESSION: {'regression', 'test_regression', 'regr'}
        }

    def parse_file(self, file_path: Union[str, Path]) -> TestFileInfo:
        """
        Parse a Python test file and extract metadata.

        Args:
            file_path: Path to the test file

        Returns:
            TestFileInfo containing parsed metadata

        Raises:
            ValueError: If file is too large or not accessible
            UnicodeDecodeError: If file encoding cannot be determined
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes > {self.max_file_size}")

        if file_size == 0:
            logger.warning(f"Empty file: {file_path}")
            return TestFileInfo(
                file_path=file_path,
                total_lines=0,
                encoding="utf-8"
            )

        # Detect encoding
        encoding = self._detect_encoding(file_path)

        try:
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            total_lines = len(content.splitlines())

            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return TestFileInfo(
                    file_path=file_path,
                    parse_errors=[f"Syntax error: {e}"],
                    encoding=encoding,
                    total_lines=total_lines
                )

            # Extract metadata
            file_info = TestFileInfo(
                file_path=file_path,
                encoding=encoding,
                total_lines=total_lines
            )

            self._extract_metadata(tree, file_info)
            return file_info

        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
            raise UnicodeDecodeError(f"Cannot decode file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing {file_path}: {e}")
            return TestFileInfo(
                file_path=file_path,
                parse_errors=[f"Unexpected error: {e}"],
                encoding=encoding,
                total_lines=0
            )

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding safely."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 8KB for encoding detection
                sample = f.read(8192)
                if not sample:
                    return "utf-8"

                result = chardet.detect(sample)
                confidence = result.get('confidence', 0)
                encoding = result.get('encoding', 'utf-8')

                # Only trust high-confidence results
                if confidence > 0.7 and encoding:
                    return encoding
                else:
                    return "utf-8"

        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8"

    def _extract_metadata(self, tree: ast.AST, file_info: TestFileInfo) -> None:
        """Extract all metadata from AST."""
        visitor = TestMetadataVisitor(file_info, self._test_type_keywords)
        visitor.visit(tree)

    def parse_directory(self, directory: Union[str, Path],
                       pattern: str = "test_*.py") -> List[TestFileInfo]:
        """
        Parse all test files in a directory.

        Args:
            directory: Directory to scan
            pattern: File pattern to match

        Returns:
            List of TestFileInfo for all parsed files
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        results = []
        test_files = list(directory.rglob(pattern))

        for file_path in test_files:
            try:
                file_info = self.parse_file(file_path)
                results.append(file_info)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                # Create error entry
                error_info = TestFileInfo(
                    file_path=file_path,
                    parse_errors=[f"Parse failed: {e}"]
                )
                results.append(error_info)

        return results


class TestMetadataVisitor(ast.NodeVisitor):
    """AST visitor to extract test metadata."""

    def __init__(self, file_info: TestFileInfo, test_type_keywords: Dict[TestType, Set[str]]):
        self.file_info = file_info
        self.test_type_keywords = test_type_keywords
        self.current_class = None

    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for alias in node.names:
            self.file_info.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from-import statements."""
        if node.module:
            for alias in node.names:
                import_name = f"from {node.module} import {alias.name}"
                self.file_info.imports.append(import_name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract test class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.file_info.classes.append(node.name)
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        """Extract test function metadata."""
        if self._is_test_function(node) or self._is_fixture(node):
            metadata = self._extract_function_metadata(node)
            if metadata:
                if self._is_fixture(node):
                    self.file_info.fixtures.append(node.name)
                else:
                    self.file_info.tests.append(metadata)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self.visit_FunctionDef(node)

    def _is_test_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a test function."""
        return (node.name.startswith('test_') or
                any(self._has_pytest_mark(d) for d in node.decorator_list) or
                (self.current_class and
                 (self.current_class.startswith('Test') or
                  node.name.startswith('test'))))

    def _is_fixture(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a fixture."""
        return any(
            isinstance(d, ast.Name) and d.id == 'fixture' or
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'fixture' or
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'fixture'
            for d in node.decorator_list
        )

    def _has_pytest_mark(self, decorator: ast.expr) -> bool:
        """Check if decorator is a pytest mark."""
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return (isinstance(decorator.func.value, ast.Name) and
                    decorator.func.value.id == 'pytest')
        return False

    def _extract_function_metadata(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[TestMetadata]:
        """Extract comprehensive metadata from a test function."""
        try:
            # Basic info
            metadata = TestMetadata(
                name=node.name,
                docstring=ast.get_docstring(node),
                file_path=self.file_info.file_path,
                line_number=node.lineno,
                test_type=self._determine_test_type(node),
                is_async=isinstance(node, ast.AsyncFunctionDef)
            )

            # Extract decorators
            for decorator in node.decorator_list:
                decorator_info = self._extract_decorator_info(decorator)
                if decorator_info:
                    metadata.decorators.append(decorator_info)

                    # Check for specific pytest decorators
                    if decorator_info.name in ['parametrize', 'pytest.mark.parametrize']:
                        metadata.is_parametrized = True
                    elif decorator_info.name in ['xfail', 'pytest.mark.xfail']:
                        metadata.expected_to_fail = True
                    elif decorator_info.name in ['skip', 'pytest.mark.skip']:
                        metadata.skip_reason = decorator_info.kwargs.get('reason', 'No reason provided')

                    # Extract marks
                    if decorator_info.name.startswith('pytest.mark.'):
                        mark_name = decorator_info.name.replace('pytest.mark.', '')
                        metadata.marks.add(mark_name)

            # Extract parameters
            for arg in node.args.args:
                param_info = ParameterInfo(
                    name=arg.arg,
                    type_annotation=self._extract_type_annotation(arg.annotation)
                )
                metadata.parameters.append(param_info)

            # Calculate complexity score
            metadata.complexity_score = self._calculate_complexity(node)

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata for function {node.name}: {e}")
            return None

    def _determine_test_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> TestType:
        """Determine the type of test based on name and decorators."""
        name_lower = node.name.lower()
        file_path_lower = str(self.file_info.file_path).lower()

        # Check file path first
        for test_type, keywords in self.test_type_keywords.items():
            if any(keyword in file_path_lower for keyword in keywords):
                return test_type

        # Check function name
        for test_type, keywords in self.test_type_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return test_type

        # Check decorators for marks
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if (isinstance(decorator.func.value, ast.Name) and
                        decorator.func.value.id == 'pytest' and
                        decorator.func.attr == 'mark'):
                        # Extract pytest mark name
                        if decorator.args and isinstance(decorator.args[0], ast.Name):
                            mark_name = decorator.args[0].id.lower()
                            for test_type, keywords in self.test_type_keywords.items():
                                if mark_name in keywords:
                                    return test_type

        return TestType.UNKNOWN

    def _extract_decorator_info(self, decorator: ast.expr) -> Optional[DecoratorInfo]:
        """Extract decorator information."""
        try:
            if isinstance(decorator, ast.Name):
                return DecoratorInfo(name=decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    name = decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    name = ast.unparse(decorator.func)
                else:
                    return None

                args = []
                kwargs = {}

                for arg in decorator.args:
                    try:
                        args.append(ast.unparse(arg))
                    except:
                        args.append(str(arg))

                for keyword in decorator.keywords:
                    try:
                        kwargs[keyword.arg] = ast.unparse(keyword.value)
                    except:
                        kwargs[keyword.arg] = str(keyword.value)

                return DecoratorInfo(name=name, args=args, kwargs=kwargs)
            elif isinstance(decorator, ast.Attribute):
                return DecoratorInfo(name=ast.unparse(decorator))

        except Exception as e:
            logger.warning(f"Failed to extract decorator info: {e}")

        return None

    def _extract_type_annotation(self, annotation: Optional[ast.expr]) -> Optional[str]:
        """Extract type annotation as string."""
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except:
            return str(annotation)

    def _calculate_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate complexity score for test function."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        # Add complexity for decorators
        complexity += len(node.decorator_list)

        # Add complexity for parameters
        complexity += len(node.args.args)

        return min(complexity, 10)  # Cap at 10