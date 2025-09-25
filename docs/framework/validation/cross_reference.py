"""Cross-reference validation for documentation consistency and accuracy."""

import ast
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from urllib.parse import urlparse, urljoin
import yaml
import json

logger = logging.getLogger(__name__)


class ReferenceType(Enum):
    """Types of references that can be validated."""

    # Code references
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    ATTRIBUTE = "attribute"
    MODULE = "module"
    VARIABLE = "variable"

    # Documentation references
    SECTION = "section"
    HEADING = "heading"
    FIGURE = "figure"
    TABLE = "table"
    EXAMPLE = "example"

    # External references
    URL = "url"
    API_ENDPOINT = "api_endpoint"
    FILE_PATH = "file_path"

    # Cross-document references
    CROSS_DOC = "cross_document"
    ANCHOR = "anchor"


@dataclass
class ReferenceLink:
    """Represents a reference link in documentation."""

    source_file: Path
    source_line: int
    source_column: int
    reference_text: str
    reference_type: ReferenceType
    target: str
    context: str = ""
    is_valid: Optional[bool] = None
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.source_file, str):
            self.source_file = Path(self.source_file)


@dataclass
class ValidationResult:
    """Result of cross-reference validation."""

    total_references: int = 0
    valid_references: int = 0
    invalid_references: List[ReferenceLink] = field(default_factory=list)
    broken_links: List[ReferenceLink] = field(default_factory=list)
    orphaned_targets: List[str] = field(default_factory=list)
    suggestions: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def validity_score(self) -> float:
        """Calculate validity score as percentage of valid references."""
        if self.total_references == 0:
            return 100.0
        return (self.valid_references / self.total_references) * 100.0


class CrossReferenceValidator:
    """Validates cross-references in documentation for consistency and accuracy."""

    def __init__(self,
                 root_path: Union[str, Path],
                 patterns: Optional[Dict[ReferenceType, str]] = None,
                 ignore_patterns: Optional[List[str]] = None,
                 external_validation: bool = False):
        """Initialize cross-reference validator.

        Args:
            root_path: Root directory for documentation
            patterns: Custom regex patterns for different reference types
            ignore_patterns: Patterns to ignore during validation
            external_validation: Whether to validate external URLs
        """
        self.root_path = Path(root_path)
        self.external_validation = external_validation
        self.ignore_patterns = ignore_patterns or [
            r'__pycache__',
            r'\.pyc$',
            r'\.git',
            r'node_modules'
        ]

        # Default patterns for different reference types
        self.patterns = {
            ReferenceType.FUNCTION: r':func:`([^`]+)`',
            ReferenceType.CLASS: r':class:`([^`]+)`',
            ReferenceType.METHOD: r':meth:`([^`]+)`',
            ReferenceType.ATTRIBUTE: r':attr:`([^`]+)`',
            ReferenceType.MODULE: r':mod:`([^`]+)`',
            ReferenceType.SECTION: r':ref:`([^`]+)`',
            ReferenceType.FIGURE: r':numref:`([^`]+)`',
            ReferenceType.URL: r'https?://[^\s\)]+',
            ReferenceType.FILE_PATH: r'`([^`]+\.(py|md|rst|txt))`',
            ReferenceType.ANCHOR: r'#([a-zA-Z0-9\-_]+)',
        }

        if patterns:
            self.patterns.update(patterns)

        # Caches for performance
        self._code_symbols: Dict[str, Set[str]] = {}
        self._file_anchors: Dict[Path, Set[str]] = {}
        self._doc_sections: Dict[Path, Set[str]] = {}

    def validate_references(self, file_patterns: List[str] = None) -> ValidationResult:
        """Validate all cross-references in documentation.

        Args:
            file_patterns: Patterns to match files for validation

        Returns:
            ValidationResult with detailed analysis
        """
        if file_patterns is None:
            file_patterns = ['**/*.md', '**/*.rst', '**/*.txt']

        logger.info(f"Starting cross-reference validation in {self.root_path}")

        # Build symbol index
        self._build_symbol_index()

        result = ValidationResult()

        for pattern in file_patterns:
            for file_path in self.root_path.glob(pattern):
                if self._should_ignore_file(file_path):
                    continue

                try:
                    file_result = self._validate_file_references(file_path)
                    result.total_references += file_result.total_references
                    result.valid_references += file_result.valid_references
                    result.invalid_references.extend(file_result.invalid_references)
                    result.broken_links.extend(file_result.broken_links)

                except Exception as e:
                    logger.error(f"Error validating {file_path}: {e}")

        # Check for orphaned targets
        result.orphaned_targets = self._find_orphaned_targets()

        # Generate suggestions for broken references
        result.suggestions = self._generate_suggestions(result.invalid_references)

        logger.info(f"Validation complete: {result.valid_references}/{result.total_references} "
                   f"valid references ({result.validity_score:.1f}%)")

        return result

    def _build_symbol_index(self):
        """Build index of available code symbols."""
        logger.debug("Building code symbol index")

        for py_file in self.root_path.glob('**/*.py'):
            if self._should_ignore_file(py_file):
                continue

            try:
                symbols = self._extract_python_symbols(py_file)
                module_path = self._get_module_path(py_file)
                self._code_symbols[module_path] = symbols

            except Exception as e:
                logger.warning(f"Could not parse {py_file}: {e}")

    def _extract_python_symbols(self, file_path: Path) -> Set[str]:
        """Extract symbols (functions, classes, etc.) from Python file."""
        symbols = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.add(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.add(f"class:{node.name}")
                    # Add methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            symbols.add(f"method:{node.name}.{item.name}")
                elif isinstance(node, ast.Assign):
                    # Module-level variables
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols.add(f"variable:{target.id}")

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return symbols

    def _get_module_path(self, file_path: Path) -> str:
        """Convert file path to module path."""
        relative_path = file_path.relative_to(self.root_path)
        module_parts = relative_path.with_suffix('').parts
        return '.'.join(module_parts)

    def _validate_file_references(self, file_path: Path) -> ValidationResult:
        """Validate references in a single file."""
        result = ValidationResult()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return result

        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Check each reference type
            for ref_type, pattern in self.patterns.items():
                matches = re.finditer(pattern, line)

                for match in matches:
                    reference = ReferenceLink(
                        source_file=file_path,
                        source_line=line_num,
                        source_column=match.start(),
                        reference_text=match.group(0),
                        reference_type=ref_type,
                        target=match.group(1) if match.groups() else match.group(0),
                        context=line.strip()
                    )

                    result.total_references += 1

                    if self._validate_reference(reference):
                        result.valid_references += 1
                    else:
                        result.invalid_references.append(reference)
                        if ref_type == ReferenceType.URL:
                            result.broken_links.append(reference)

        return result

    def _validate_reference(self, reference: ReferenceLink) -> bool:
        """Validate a single reference."""
        try:
            if reference.reference_type in [ReferenceType.FUNCTION, ReferenceType.CLASS,
                                          ReferenceType.METHOD, ReferenceType.MODULE]:
                return self._validate_code_reference(reference)

            elif reference.reference_type == ReferenceType.FILE_PATH:
                return self._validate_file_reference(reference)

            elif reference.reference_type == ReferenceType.URL:
                return self._validate_url_reference(reference)

            elif reference.reference_type in [ReferenceType.SECTION, ReferenceType.ANCHOR]:
                return self._validate_anchor_reference(reference)

            else:
                # For other types, assume valid unless proven otherwise
                reference.is_valid = True
                return True

        except Exception as e:
            reference.error_message = str(e)
            reference.is_valid = False
            return False

    def _validate_code_reference(self, reference: ReferenceLink) -> bool:
        """Validate references to code symbols."""
        target = reference.target

        # Handle fully qualified names (module.symbol)
        if '.' in target:
            parts = target.split('.')
            module_name = '.'.join(parts[:-1])
            symbol_name = parts[-1]
        else:
            # Try to infer module from current file
            module_name = self._get_module_path(reference.source_file.parent)
            symbol_name = target

        # Check if symbol exists in any known module
        symbol_key = f"{reference.reference_type.value}:{symbol_name}"

        for module, symbols in self._code_symbols.items():
            if symbol_key in symbols:
                reference.is_valid = True
                return True

        reference.is_valid = False
        reference.error_message = f"Symbol '{target}' not found in any module"

        # Generate suggestions
        similar_symbols = self._find_similar_symbols(symbol_name, reference.reference_type)
        reference.suggestions = similar_symbols[:5]  # Limit to top 5

        return False

    def _validate_file_reference(self, reference: ReferenceLink) -> bool:
        """Validate references to files."""
        target_path = Path(reference.target)

        # Try relative to current file's directory
        relative_path = reference.source_file.parent / target_path
        if relative_path.exists():
            reference.is_valid = True
            return True

        # Try relative to root
        root_path = self.root_path / target_path
        if root_path.exists():
            reference.is_valid = True
            return True

        reference.is_valid = False
        reference.error_message = f"File '{reference.target}' not found"

        # Suggest similar files
        similar_files = self._find_similar_files(target_path.name)
        reference.suggestions = similar_files[:3]

        return False

    def _validate_url_reference(self, reference: ReferenceLink) -> bool:
        """Validate URL references."""
        if not self.external_validation:
            # Skip external validation if disabled
            reference.is_valid = True
            return True

        try:
            import requests
            response = requests.head(reference.target, timeout=10, allow_redirects=True)

            if response.status_code < 400:
                reference.is_valid = True
                return True
            else:
                reference.is_valid = False
                reference.error_message = f"HTTP {response.status_code}"
                return False

        except ImportError:
            logger.warning("requests library not available for URL validation")
            reference.is_valid = True
            return True
        except Exception as e:
            reference.is_valid = False
            reference.error_message = str(e)
            return False

    def _validate_anchor_reference(self, reference: ReferenceLink) -> bool:
        """Validate anchor references within documents."""
        # This would need more sophisticated parsing of document structure
        # For now, assume valid
        reference.is_valid = True
        return True

    def _find_similar_symbols(self, target: str, ref_type: ReferenceType) -> List[str]:
        """Find symbols similar to the target."""
        from difflib import get_close_matches

        all_symbols = []
        prefix = f"{ref_type.value}:"

        for symbols in self._code_symbols.values():
            for symbol in symbols:
                if symbol.startswith(prefix):
                    symbol_name = symbol[len(prefix):]
                    all_symbols.append(symbol_name)

        return get_close_matches(target, all_symbols, n=5, cutoff=0.6)

    def _find_similar_files(self, target: str) -> List[str]:
        """Find files similar to the target."""
        from difflib import get_close_matches

        all_files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                all_files.append(file_path.name)

        return get_close_matches(target, all_files, n=3, cutoff=0.6)

    def _find_orphaned_targets(self) -> List[str]:
        """Find symbols or targets that are defined but never referenced."""
        # This is a simplified implementation
        orphaned = []

        # Check for defined functions/classes that are never referenced
        all_referenced = set()
        for symbols in self._code_symbols.values():
            all_referenced.update(symbols)

        # This would need more sophisticated analysis
        return orphaned

    def _generate_suggestions(self, invalid_refs: List[ReferenceLink]) -> Dict[str, List[str]]:
        """Generate suggestions for fixing invalid references."""
        suggestions = {}

        for ref in invalid_refs:
            key = f"{ref.source_file}:{ref.source_line}"
            if ref.suggestions:
                suggestions[key] = ref.suggestions

        return suggestions

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on patterns."""
        path_str = str(file_path)

        for pattern in self.ignore_patterns:
            if re.search(pattern, path_str):
                return True

        return False

    def export_results(self, result: ValidationResult, output_path: Path) -> bool:
        """Export validation results to file."""
        try:
            export_data = {
                'summary': {
                    'total_references': result.total_references,
                    'valid_references': result.valid_references,
                    'validity_score': result.validity_score,
                    'invalid_count': len(result.invalid_references),
                    'broken_links_count': len(result.broken_links)
                },
                'invalid_references': [
                    {
                        'file': str(ref.source_file),
                        'line': ref.source_line,
                        'column': ref.source_column,
                        'reference_text': ref.reference_text,
                        'type': ref.reference_type.value,
                        'target': ref.target,
                        'error': ref.error_message,
                        'suggestions': ref.suggestions,
                        'context': ref.context
                    }
                    for ref in result.invalid_references
                ],
                'broken_links': [
                    {
                        'file': str(ref.source_file),
                        'line': ref.source_line,
                        'url': ref.target,
                        'error': ref.error_message,
                        'context': ref.context
                    }
                    for ref in result.broken_links
                ],
                'orphaned_targets': result.orphaned_targets,
                'suggestions': result.suggestions
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Validation results exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False