"""AST-based Python code parser for extracting documentation information."""

import ast
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum


class MemberType(Enum):
    """Types of code members that can be documented."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    ATTRIBUTE = "attribute"
    CONSTANT = "constant"


@dataclass
class Parameter:
    """Represents a function/method parameter."""
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    kind: str = "positional"  # positional, keyword-only, var-positional, var-keyword
    description: Optional[str] = None


@dataclass
class DocumentationNode:
    """Represents a documented code element."""
    name: str
    member_type: MemberType
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parameters: List[Parameter] = field(default_factory=list)
    return_annotation: Optional[str] = None
    return_description: Optional[str] = None
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    line_number: Optional[int] = None
    source_file: Optional[str] = None
    is_private: bool = False
    is_property: bool = False
    parent: Optional[str] = None
    children: List['DocumentationNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocstringParser:
    """Parses docstrings to extract structured information."""

    def parse(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Parse a docstring into structured components.

        Args:
            docstring: The raw docstring to parse

        Returns:
            Dictionary containing parsed components
        """
        if not docstring:
            return {}

        lines = docstring.strip().split('\n')
        result = {
            'summary': '',
            'description': '',
            'parameters': {},
            'returns': None,
            'raises': [],
            'examples': [],
            'notes': []
        }

        current_section = 'description'
        current_content = []

        for line in lines:
            stripped = line.strip()

            # Check for section headers
            if self._is_section_header(stripped):
                # Save previous section
                self._save_section(result, current_section, current_content)
                current_section = self._get_section_type(stripped)
                current_content = []
                continue

            current_content.append(line)

        # Save final section
        self._save_section(result, current_section, current_content)

        # Extract summary from description
        if result['description']:
            desc_lines = result['description'].strip().split('\n')
            if desc_lines:
                result['summary'] = desc_lines[0].strip()
                if len(desc_lines) > 1:
                    result['description'] = '\n'.join(desc_lines[1:]).strip()
                else:
                    result['description'] = ''

        return result

    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        headers = ['Args:', 'Arguments:', 'Parameters:', 'Returns:', 'Return:',
                  'Yields:', 'Raises:', 'Examples:', 'Example:', 'Notes:', 'Note:']
        return any(line.startswith(header) for header in headers)

    def _get_section_type(self, line: str) -> str:
        """Get the section type from a header line."""
        if line.startswith(('Args:', 'Arguments:', 'Parameters:')):
            return 'parameters'
        elif line.startswith(('Returns:', 'Return:')):
            return 'returns'
        elif line.startswith('Yields:'):
            return 'yields'
        elif line.startswith('Raises:'):
            return 'raises'
        elif line.startswith(('Examples:', 'Example:')):
            return 'examples'
        elif line.startswith(('Notes:', 'Note:')):
            return 'notes'
        return 'description'

    def _save_section(self, result: Dict, section: str, content: List[str]) -> None:
        """Save a section's content to the result dictionary."""
        if not content:
            return

        text = '\n'.join(content).strip()
        if not text:
            return

        if section == 'parameters':
            result[section].update(self._parse_parameters(text))
        elif section == 'raises':
            result[section].extend(self._parse_raises(text))
        elif section == 'examples':
            result[section].extend(self._parse_examples(text))
        elif section in ['returns', 'yields']:
            result[section] = text
        else:
            if result[section]:
                result[section] += '\n' + text
            else:
                result[section] = text

    def _parse_parameters(self, text: str) -> Dict[str, str]:
        """Parse parameter documentation."""
        params = {}
        lines = text.split('\n')
        current_param = None
        current_desc = []

        for line in lines:
            stripped = line.strip()
            if ':' in stripped and not line.startswith(' '):
                # Save previous parameter
                if current_param:
                    params[current_param] = '\n'.join(current_desc).strip()

                # Start new parameter
                parts = stripped.split(':', 1)
                current_param = parts[0].strip()
                if len(parts) > 1:
                    current_desc = [parts[1].strip()]
                else:
                    current_desc = []
            elif current_param and stripped:
                current_desc.append(stripped)

        # Save final parameter
        if current_param:
            params[current_param] = '\n'.join(current_desc).strip()

        return params

    def _parse_raises(self, text: str) -> List[str]:
        """Parse raises documentation."""
        raises = []
        lines = text.split('\n')

        for line in lines:
            stripped = line.strip()
            if stripped:
                raises.append(stripped)

        return raises

    def _parse_examples(self, text: str) -> List[str]:
        """Parse examples from documentation."""
        # For now, return the entire text as one example
        # Could be enhanced to detect code blocks
        return [text] if text.strip() else []


class PythonASTParser:
    """AST-based parser for extracting documentation from Python code."""

    def __init__(self, include_private: bool = False):
        """Initialize the parser.

        Args:
            include_private: Whether to include private members (starting with _)
        """
        self.include_private = include_private
        self.docstring_parser = DocstringParser()
        self._current_file: Optional[str] = None
        self._current_module: Optional[ast.Module] = None

    def parse_file(self, file_path: Union[str, Path]) -> DocumentationNode:
        """Parse a Python file and extract documentation.

        Args:
            file_path: Path to the Python file to parse

        Returns:
            DocumentationNode representing the module

        Raises:
            SyntaxError: If the file contains invalid Python syntax
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._current_file = str(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {file_path}: {e}")

        self._current_module = tree
        return self._parse_module(tree, file_path.stem)

    def parse_directory(self, directory_path: Union[str, Path],
                       recursive: bool = True) -> List[DocumentationNode]:
        """Parse all Python files in a directory.

        Args:
            directory_path: Path to the directory to parse
            recursive: Whether to parse subdirectories recursively

        Returns:
            List of DocumentationNode objects for each module
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        modules = []
        pattern = "**/*.py" if recursive else "*.py"

        for py_file in directory_path.glob(pattern):
            if py_file.name.startswith('.'):
                continue

            try:
                module_node = self.parse_file(py_file)
                modules.append(module_node)
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Warning: Could not parse {py_file}: {e}")
                continue

        return modules

    def _parse_module(self, node: ast.Module, name: str) -> DocumentationNode:
        """Parse a module AST node."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.docstring_parser.parse(docstring)

        module_node = DocumentationNode(
            name=name,
            member_type=MemberType.MODULE,
            docstring=docstring,
            source_file=self._current_file,
            line_number=1,
            metadata=parsed_doc
        )

        # Parse module-level elements
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                if self._should_include(child.name):
                    class_node = self._parse_class(child)
                    class_node.parent = name
                    module_node.children.append(class_node)
            elif isinstance(child, ast.FunctionDef):
                if self._should_include(child.name):
                    func_node = self._parse_function(child)
                    func_node.parent = name
                    module_node.children.append(func_node)
            elif isinstance(child, ast.Assign):
                # Parse module-level constants/variables
                constants = self._parse_assignments(child)
                for const_node in constants:
                    if self._should_include(const_node.name):
                        const_node.parent = name
                        module_node.children.append(const_node)

        return module_node

    def _parse_class(self, node: ast.ClassDef) -> DocumentationNode:
        """Parse a class AST node."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.docstring_parser.parse(docstring)

        class_node = DocumentationNode(
            name=node.name,
            member_type=MemberType.CLASS,
            docstring=docstring,
            line_number=node.lineno,
            source_file=self._current_file,
            is_private=node.name.startswith('_'),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            metadata=parsed_doc
        )

        # Parse class members
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                if self._should_include(child.name):
                    method_node = self._parse_function(child, is_method=True)
                    method_node.parent = node.name
                    class_node.children.append(method_node)
            elif isinstance(child, ast.Assign):
                # Parse class attributes
                attrs = self._parse_assignments(child)
                for attr_node in attrs:
                    if self._should_include(attr_node.name):
                        attr_node.parent = node.name
                        class_node.children.append(attr_node)

        return class_node

    def _parse_function(self, node: ast.FunctionDef,
                       is_method: bool = False) -> DocumentationNode:
        """Parse a function/method AST node."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.docstring_parser.parse(docstring)

        # Determine if it's a property
        is_property = any(
            self._get_decorator_name(d) == 'property'
            for d in node.decorator_list
        )

        # Parse function signature
        signature = self._build_signature(node)
        parameters = self._parse_function_parameters(node, parsed_doc.get('parameters', {}))

        # Determine member type
        if is_property:
            member_type = MemberType.PROPERTY
        elif is_method:
            member_type = MemberType.METHOD
        else:
            member_type = MemberType.FUNCTION

        return DocumentationNode(
            name=node.name,
            member_type=member_type,
            docstring=docstring,
            signature=signature,
            parameters=parameters,
            return_annotation=self._get_return_annotation(node),
            return_description=parsed_doc.get('returns'),
            raises=parsed_doc.get('raises', []),
            examples=parsed_doc.get('examples', []),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            line_number=node.lineno,
            source_file=self._current_file,
            is_private=node.name.startswith('_'),
            is_property=is_property,
            metadata=parsed_doc
        )

    def _parse_assignments(self, node: ast.Assign) -> List[DocumentationNode]:
        """Parse assignment nodes to extract constants/attributes."""
        assignments = []

        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple assignment: x = value
                const_node = DocumentationNode(
                    name=target.id,
                    member_type=MemberType.CONSTANT if target.id.isupper() else MemberType.ATTRIBUTE,
                    line_number=node.lineno,
                    source_file=self._current_file,
                    is_private=target.id.startswith('_')
                )
                assignments.append(const_node)

        return assignments

    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build a function signature string."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_string(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg += f": {self._get_annotation_string(node.args.vararg.annotation)}"
            args.append(vararg)

        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_string(arg.annotation)}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg += f": {self._get_annotation_string(node.args.kwarg.annotation)}"
            args.append(kwarg)

        signature = f"{node.name}({', '.join(args)})"

        # Add return annotation
        if node.returns:
            signature += f" -> {self._get_annotation_string(node.returns)}"

        return signature

    def _parse_function_parameters(self, node: ast.FunctionDef,
                                 param_docs: Dict[str, str]) -> List[Parameter]:
        """Parse function parameters with documentation."""
        parameters = []

        # Regular arguments
        defaults = node.args.defaults
        default_offset = len(node.args.args) - len(defaults)

        for i, arg in enumerate(node.args.args):
            default_value = None
            if i >= default_offset:
                default_idx = i - default_offset
                default_value = self._get_default_value(defaults[default_idx])

            param = Parameter(
                name=arg.arg,
                annotation=self._get_annotation_string(arg.annotation) if arg.annotation else None,
                default=default_value,
                kind="positional",
                description=param_docs.get(arg.arg)
            )
            parameters.append(param)

        # *args
        if node.args.vararg:
            param = Parameter(
                name=node.args.vararg.arg,
                annotation=self._get_annotation_string(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                kind="var-positional",
                description=param_docs.get(node.args.vararg.arg)
            )
            parameters.append(param)

        # Keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            default_value = None
            if i < len(node.args.kw_defaults) and node.args.kw_defaults[i]:
                default_value = self._get_default_value(node.args.kw_defaults[i])

            param = Parameter(
                name=arg.arg,
                annotation=self._get_annotation_string(arg.annotation) if arg.annotation else None,
                default=default_value,
                kind="keyword-only",
                description=param_docs.get(arg.arg)
            )
            parameters.append(param)

        # **kwargs
        if node.args.kwarg:
            param = Parameter(
                name=node.args.kwarg.arg,
                annotation=self._get_annotation_string(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                kind="var-keyword",
                description=param_docs.get(node.args.kwarg.arg)
            )
            parameters.append(param)

        return parameters

    def _get_annotation_string(self, annotation) -> str:
        """Convert an annotation AST node to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return repr(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation_string(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            value = self._get_annotation_string(annotation.value)
            slice_value = self._get_annotation_string(annotation.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(annotation, ast.Tuple):
            elements = [self._get_annotation_string(elt) for elt in annotation.elts]
            return f"({', '.join(elements)})"
        elif isinstance(annotation, ast.List):
            elements = [self._get_annotation_string(elt) for elt in annotation.elts]
            return f"[{', '.join(elements)}]"
        else:
            # Fallback to unparsing if available (Python 3.9+)
            try:
                return ast.unparse(annotation)
            except AttributeError:
                return "Any"

    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Get the return type annotation as a string."""
        if node.returns:
            return self._get_annotation_string(node.returns)
        return None

    def _get_default_value(self, node) -> str:
        """Get the default value of a parameter as a string."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_string(node.value)}.{node.attr}"
        else:
            try:
                return ast.unparse(node)
            except AttributeError:
                return "..."

    def _get_decorator_name(self, decorator) -> str:
        """Get the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_annotation_string(decorator.value)}.{decorator.attr}"
        else:
            try:
                return ast.unparse(decorator)
            except AttributeError:
                return "decorator"

    def _should_include(self, name: str) -> bool:
        """Determine if a member should be included based on privacy settings."""
        if not self.include_private and name.startswith('_'):
            return False
        return True


def extract_module_info(file_path: Union[str, Path],
                       include_private: bool = False) -> DocumentationNode:
    """Convenience function to extract documentation from a single file.

    Args:
        file_path: Path to the Python file
        include_private: Whether to include private members

    Returns:
        DocumentationNode for the module
    """
    parser = PythonASTParser(include_private=include_private)
    return parser.parse_file(file_path)


def extract_package_info(package_path: Union[str, Path],
                        include_private: bool = False,
                        recursive: bool = True) -> List[DocumentationNode]:
    """Convenience function to extract documentation from a package.

    Args:
        package_path: Path to the package directory
        include_private: Whether to include private members
        recursive: Whether to process subdirectories

    Returns:
        List of DocumentationNode objects for all modules
    """
    parser = PythonASTParser(include_private=include_private)
    return parser.parse_directory(package_path, recursive=recursive)