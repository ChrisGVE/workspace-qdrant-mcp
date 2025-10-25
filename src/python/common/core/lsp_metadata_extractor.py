"""
LSP Code Metadata Extraction Pipeline

This module implements a comprehensive code metadata extraction system that leverages
Language Server Protocol (LSP) servers to extract rich code intelligence including
symbols, types, relationships, and documentation from source files.

Key Features:
    - Symbol extraction (functions, classes, variables, imports, exports)
    - Type information extraction (parameters, return types, annotations)
    - Documentation capture (docstrings, comments, JSDoc)
    - Relationship mapping (dependencies, inheritance, call graphs)
    - Interface + Minimal Context storage strategy
    - Language-specific extraction logic for Python, Rust, JS/TS, Java, C/C++, Go
    - Robust error handling for malformed code and LSP failures
    - Performance optimization with caching and batching

Example:
    ```python
    from workspace_qdrant_mcp.core.lsp_metadata_extractor import LspMetadataExtractor

    # Initialize extractor
    extractor = LspMetadataExtractor()
    await extractor.initialize()

    # Extract metadata from file
    metadata = await extractor.extract_file_metadata("/path/to/source.py")

    # Get relationship graph
    relationships = await extractor.build_relationship_graph(["/path/to/files"])
    ```
"""

import asyncio
import re
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from .language_filters import LanguageAwareFilter
from .lsp_client import AsyncioLspClient

# logger imported from loguru


class SymbolKind(Enum):
    """LSP Symbol Kinds with workspace-specific extensions"""
    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14
    STRING = 15
    NUMBER = 16
    BOOLEAN = 17
    ARRAY = 18
    OBJECT = 19
    KEY = 20
    NULL = 21
    ENUM_MEMBER = 22
    STRUCT = 23
    EVENT = 24
    OPERATOR = 25
    TYPE_PARAMETER = 26

    # Workspace-specific extensions
    IMPORT = 100
    EXPORT = 101
    DECLARATION = 102
    DEFINITION = 103


class RelationshipType(Enum):
    """Types of relationships between code symbols"""
    IMPORTS = "imports"
    EXPORTS = "exports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    REFERENCES = "references"
    DECLARES = "declares"
    DEFINES = "defines"
    CONTAINS = "contains"
    OVERRIDES = "overrides"
    USES = "uses"


@dataclass
class Position:
    """Source code position information"""
    line: int  # 0-based line number
    character: int  # 0-based character offset

    def to_dict(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_lsp(cls, lsp_position: dict[str, Any]) -> "Position":
        """Create Position from LSP position data"""
        return cls(
            line=lsp_position.get("line", 0),
            character=lsp_position.get("character", 0)
        )


@dataclass
class Range:
    """Source code range information"""
    start: Position
    end: Position

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict()
        }

    @classmethod
    def from_lsp(cls, lsp_range: dict[str, Any]) -> "Range":
        """Create Range from LSP range data"""
        return cls(
            start=Position.from_lsp(lsp_range.get("start", {})),
            end=Position.from_lsp(lsp_range.get("end", {}))
        )


@dataclass
class TypeInformation:
    """Type information for symbols"""
    type_name: str | None = None
    type_signature: str | None = None
    parameter_types: list[dict[str, str]] = field(default_factory=list)
    return_type: str | None = None
    generic_parameters: list[str] = field(default_factory=list)
    nullable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "type_signature": self.type_signature,
            "parameter_types": self.parameter_types,
            "return_type": self.return_type,
            "generic_parameters": self.generic_parameters,
            "nullable": self.nullable
        }


@dataclass
class Documentation:
    """Documentation information for symbols"""
    docstring: str | None = None
    inline_comments: list[str] = field(default_factory=list)
    leading_comments: list[str] = field(default_factory=list)
    trailing_comments: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)  # JSDoc-style tags

    def to_dict(self) -> dict[str, Any]:
        return {
            "docstring": self.docstring,
            "inline_comments": self.inline_comments,
            "leading_comments": self.leading_comments,
            "trailing_comments": self.trailing_comments,
            "tags": self.tags
        }


@dataclass
class CodeSymbol:
    """
    Represents a code symbol with comprehensive metadata.

    This implements the 'Interface + Minimal Context' storage strategy:
    - Complete symbol signatures and metadata
    - 1-3 lines of surrounding context for better understanding
    """

    name: str
    kind: SymbolKind
    file_uri: str
    range: Range
    selection_range: Range | None = None

    # Type information
    type_info: TypeInformation | None = None

    # Documentation
    documentation: Documentation | None = None

    # Context (1-3 lines around the symbol)
    context_before: list[str] = field(default_factory=list)  # 1-2 lines before
    context_after: list[str] = field(default_factory=list)   # 0-1 lines after

    # Symbol metadata
    visibility: str | None = None  # public, private, protected, etc.
    modifiers: list[str] = field(default_factory=list)  # static, final, async, etc.
    language: str | None = None

    # Symbol relationships
    parent_symbol: str | None = None  # Parent class/namespace
    children: list[str] = field(default_factory=list)  # Child symbols

    # Additional metadata
    deprecated: bool = False
    experimental: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "kind_name": self.kind.name,
            "file_uri": self.file_uri,
            "range": self.range.to_dict(),
            "selection_range": self.selection_range.to_dict() if self.selection_range else None,
            "type_info": self.type_info.to_dict() if self.type_info else None,
            "documentation": self.documentation.to_dict() if self.documentation else None,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "visibility": self.visibility,
            "modifiers": self.modifiers,
            "language": self.language,
            "parent_symbol": self.parent_symbol,
            "children": self.children,
            "deprecated": self.deprecated,
            "experimental": self.experimental,
            "tags": self.tags
        }

    def get_full_name(self) -> str:
        """Get fully qualified symbol name"""
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.name}"
        return self.name

    def get_signature(self) -> str:
        """Get symbol signature string"""
        # If we have a type signature, prepend modifiers if any
        if self.type_info and self.type_info.type_signature:
            if self.modifiers:
                return " ".join(self.modifiers) + " " + self.type_info.type_signature
            return self.type_info.type_signature

        # Build basic signature from available information
        signature_parts = []
        if self.modifiers:
            signature_parts.extend(self.modifiers)

        if self.kind == SymbolKind.FUNCTION or self.kind == SymbolKind.METHOD:
            params = []
            if self.type_info and self.type_info.parameter_types:
                for param in self.type_info.parameter_types:
                    param_str = param.get("name", "")
                    if param.get("type"):
                        param_str += f": {param['type']}"
                    params.append(param_str)

            signature = f"{self.name}({', '.join(params)})"
            if self.type_info and self.type_info.return_type:
                signature += f" -> {self.type_info.return_type}"

            signature_parts.append(signature)
        else:
            name_and_type = self.name
            if self.type_info and self.type_info.type_name:
                name_and_type += f": {self.type_info.type_name}"
            signature_parts.append(name_and_type)

        return " ".join(signature_parts)


@dataclass
class SymbolRelationship:
    """Relationship between two code symbols"""
    from_symbol: str  # Symbol identifier
    to_symbol: str    # Symbol identifier
    relationship_type: RelationshipType
    file_uri: str     # File where relationship is defined
    location: Range | None = None  # Location of relationship reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_symbol": self.from_symbol,
            "to_symbol": self.to_symbol,
            "relationship_type": self.relationship_type.value,
            "file_uri": self.file_uri,
            "location": self.location.to_dict() if self.location else None
        }


@dataclass
class FileMetadata:
    """Complete metadata extracted from a source file"""
    file_uri: str
    file_path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    relationships: list[SymbolRelationship] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    # File-level documentation and metadata
    file_docstring: str | None = None
    file_comments: list[str] = field(default_factory=list)

    # Extraction metadata
    extraction_timestamp: float | None = None
    lsp_server: str | None = None
    extraction_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_uri": self.file_uri,
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "imports": self.imports,
            "exports": self.exports,
            "file_docstring": self.file_docstring,
            "file_comments": self.file_comments,
            "extraction_timestamp": self.extraction_timestamp,
            "lsp_server": self.lsp_server,
            "extraction_errors": self.extraction_errors,
            "symbol_count": len(self.symbols),
            "relationship_count": len(self.relationships)
        }


@dataclass
class ExtractionStatistics:
    """Statistics for metadata extraction operations"""
    files_processed: int = 0
    files_failed: int = 0
    symbols_extracted: int = 0
    relationships_found: int = 0
    extraction_time_ms: float = 0.0
    lsp_requests_made: int = 0
    lsp_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "symbols_extracted": self.symbols_extracted,
            "relationships_found": self.relationships_found,
            "extraction_time_ms": self.extraction_time_ms,
            "lsp_requests_made": self.lsp_requests_made,
            "lsp_errors": self.lsp_errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "success_rate": self.files_processed / max(1, self.files_processed + self.files_failed),
            "symbols_per_file": self.symbols_extracted / max(1, self.files_processed),
            "relationships_per_file": self.relationships_found / max(1, self.files_processed)
        }


class LanguageSpecificExtractor(ABC):
    """
    Abstract base class for language-specific metadata extraction logic.

    Each supported language can implement specific extraction patterns
    for better code intelligence extraction.
    """

    @abstractmethod
    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract documentation for a symbol from source code"""
        pass

    @abstractmethod
    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract type information from LSP symbol and hover data"""
        pass

    @abstractmethod
    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract import and export statements from source code"""
        pass

    @abstractmethod
    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get 1-3 lines of context around a symbol (before, after)"""
        pass


class PythonExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for Python code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract Python docstrings and comments"""
        doc = Documentation()

        # Look for docstring after symbol definition
        start_line = symbol_range.start.line
        if start_line + 1 < len(source_lines):
            next_line = source_lines[start_line + 1].strip()

            # Check for docstring patterns
            if next_line.startswith('"""') or next_line.startswith("'''"):
                docstring_lines = []
                quote_type = '"""' if next_line.startswith('"""') else "'''"

                # Single line docstring
                if next_line.endswith(quote_type) and len(next_line) > 6:
                    doc.docstring = next_line[3:-3].strip()
                else:
                    # Multi-line docstring
                    docstring_lines.append(next_line[3:])
                    for i in range(start_line + 2, min(len(source_lines), start_line + 20)):
                        line = source_lines[i].strip()
                        if line.endswith(quote_type):
                            docstring_lines.append(line[:-3])
                            break
                        docstring_lines.append(line)

                    doc.docstring = "\n".join(docstring_lines).strip()

        # Extract inline and leading comments
        for i in range(max(0, start_line - 3), min(len(source_lines), start_line + 3)):
            line = source_lines[i].strip()
            if line.startswith('#'):
                comment = line[1:].strip()
                if i < start_line:
                    doc.leading_comments.append(comment)
                elif i == start_line and '#' in source_lines[i][symbol_range.start.character:]:
                    doc.inline_comments.append(comment)
                else:
                    doc.trailing_comments.append(comment)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract Python type information"""
        type_info = TypeInformation()

        # Extract from hover data if available
        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]

                # Parse Python type signatures
                if "def " in type_text or "class " in type_text:
                    type_info.type_signature = type_text.strip()

                    # Extract return type from function signature
                    if " -> " in type_text:
                        return_part = type_text.split(" -> ")[1].split(":")[0].strip()
                        type_info.return_type = return_part

                    # Extract parameter types
                    if "(" in type_text and ")" in type_text:
                        params_match = re.search(r'\((.*?)\)', type_text)
                        if params_match:
                            params_str = params_match.group(1)
                            params = []
                            for param in params_str.split(','):
                                param = param.strip()
                                if ':' in param:
                                    name, type_name = param.split(':', 1)
                                    params.append({
                                        "name": name.strip(),
                                        "type": type_name.strip()
                                    })
                                elif param and param != "self":
                                    params.append({"name": param.strip()})
                            type_info.parameter_types = params

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract Python import and export statements"""
        imports = []
        exports = []

        for line in source_lines:
            line = line.strip()

            # Import statements
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line)

            # Python doesn't have explicit exports, but __all__ defines public interface
            if line.startswith("__all__"):
                # Extract items from __all__ list
                all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', line, re.DOTALL)
                if all_match:
                    items = all_match.group(1)
                    for item in items.split(','):
                        item = item.strip().strip('"\'')
                        if item:
                            exports.append(item)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for Python symbols"""
        start_line = symbol_range.start.line

        # Get 1-2 lines before (excluding empty lines and comments)
        # Prioritize structural context (class/def) over imports
        context_before = []
        structural_context = None

        for i in range(start_line - 1, max(-1, start_line - 5), -1):
            if i < 0:
                break
            line = source_lines[i]
            line_stripped = line.strip()

            if line_stripped and not line_stripped.startswith('#'):
                # Check if this is structural context (class, def, etc.)
                if line_stripped.startswith(('class ', 'def ', 'async def ')):
                    structural_context = line
                    break
                # Otherwise just add as regular context
                context_before.insert(0, line)
                if len(context_before) >= 2:
                    break

        # If we found structural context, use it preferentially
        if structural_context:
            context_before = [structural_context]

        # Get 0-1 lines after (for context like decorators or continuations)
        context_after = []
        if start_line + 1 < len(source_lines):
            next_line = source_lines[start_line + 1].strip()
            if next_line and not next_line.startswith('"""') and not next_line.startswith("'''"):
                context_after.append(source_lines[start_line + 1])

        return context_before, context_after


# Additional language extractors would go here (Rust, JavaScript, Java, etc.)
class RustExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for Rust code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract Rust doc comments (/// and //!)"""
        doc = Documentation()
        start_line = symbol_range.start.line

        # Look for doc comments before the symbol
        doc_lines = []
        multiline_start = None
        multiline_end = None

        for i in range(start_line - 1, max(-1, start_line - 20), -1):
            if i < 0:
                break
            line = source_lines[i].strip()

            if line.startswith("///") or line.startswith("//!"):
                doc_lines.insert(0, line[3:].strip())
            elif line.endswith("*/") and multiline_end is None:
                # Found end of multi-line comment - mark it and keep searching for start
                multiline_end = i
            elif (line.startswith("/**") or line.startswith("/*!")) and multiline_end is not None:
                # Found start of multi-line doc comment
                multiline_start = i
                break
            elif not line or line.startswith("//"):
                continue
            else:
                # If we haven't found a multiline end yet, break
                if multiline_end is None:
                    break

        # Process multi-line doc comment if found
        if multiline_start is not None and multiline_end is not None:
            comment_lines = []
            for j in range(multiline_start, multiline_end + 1):
                comment_line = source_lines[j].strip()
                if comment_line.startswith("/**") or comment_line.startswith("/*!"):
                    content = comment_line[3:].strip()
                    # Only add if there's actual content and it's not just "*"
                    if content and content != "*" and not content.startswith("*/"):
                        comment_lines.append(content)
                elif comment_line.endswith("*/"):
                    content = comment_line[:-2].strip().lstrip("* ")
                    if content and content != "*":
                        comment_lines.append(content)
                    break
                else:
                    content = comment_line.lstrip("* ")
                    if content and content != "*":
                        comment_lines.append(content)
            doc_lines = comment_lines + doc_lines

        if doc_lines:
            doc.docstring = "\n".join(doc_lines)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract Rust type information"""
        type_info = TypeInformation()

        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()

                # Extract return type from function signature
                if " -> " in type_text:
                    return_part = type_text.split(" -> ")[1].split("{")[0].strip()
                    type_info.return_type = return_part

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract Rust use statements and pub items"""
        imports = []
        exports = []

        for line in source_lines:
            line = line.strip()

            # Use statements (imports)
            if line.startswith("use "):
                imports.append(line)

            # Public items (exports)
            if line.startswith("pub "):
                exports.append(line)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for Rust symbols"""
        start_line = symbol_range.start.line

        context_before = []
        structural_context = None

        for i in range(start_line - 1, max(-1, start_line - 5), -1):
            if i < 0:
                break
            line = source_lines[i]
            line_stripped = line.strip()

            if line_stripped and not line_stripped.startswith("//"):
                # Check if this is structural context (impl, struct, etc.)
                if line_stripped.startswith(('impl ', 'struct ', 'enum ', 'trait ')):
                    structural_context = line
                    break
                # Otherwise just add as regular context
                context_before.insert(0, line)
                if len(context_before) >= 1:
                    break

        # If we found structural context, use it preferentially
        if structural_context:
            context_before = [structural_context]

        context_after = []
        return context_before, context_after


class JavaScriptExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for JavaScript/TypeScript code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract JSDoc comments and regular comments"""
        doc = Documentation()
        start_line = symbol_range.start.line

        # Look for JSDoc comments before the symbol
        multiline_start = None
        multiline_end = None

        for i in range(start_line - 1, max(-1, start_line - 20), -1):
            if i < 0:
                break
            line = source_lines[i].strip()
            if line.endswith("*/") and multiline_end is None:
                # Found end of multi-line comment
                multiline_end = i
            elif line.startswith("/**") and multiline_end is not None:
                # Found start of JSDoc comment
                multiline_start = i
                break
            elif not line or line.startswith("//"):
                continue
            else:
                # If we haven't found a multiline end yet, break
                if multiline_end is None:
                    break

        # Process JSDoc comment if found
        if multiline_start is not None and multiline_end is not None:
            jsdoc_lines = []
            for j in range(multiline_start, multiline_end + 1):
                comment_line = source_lines[j].strip()
                if comment_line.startswith("/**"):
                    content = comment_line[3:].strip()
                    # Only add if there's actual content and not just "*" or "*/"
                    if content and content != "*" and not content.startswith("*/"):
                        jsdoc_lines.append(content)
                elif comment_line.startswith("*/"):
                    break
                elif comment_line.startswith("*"):
                    content = comment_line[1:].strip()
                    if content.startswith("@"):
                        # JSDoc tag
                        tag_match = re.match(r'@(\w+)\s*(.*)', content)
                        if tag_match:
                            tag_name, tag_content = tag_match.groups()
                            if tag_name not in doc.tags:
                                doc.tags[tag_name] = []
                            doc.tags[tag_name].append(tag_content.strip())
                    elif content and content != "*":
                        jsdoc_lines.append(content)

            if jsdoc_lines:
                doc.docstring = "\n".join(jsdoc_lines)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract JavaScript/TypeScript type information"""
        type_info = TypeInformation()

        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()

                # Extract TypeScript return type from function signature
                if "): " in type_text:
                    # function signature like "function name(...): returnType"
                    type_part = type_text.split("): ")[1].strip()
                    type_info.type_name = type_part
                elif ": " in type_text:
                    # variable type like "const name: type"
                    type_part = type_text.split(": ")[1].split("=")[0].strip()
                    type_info.type_name = type_part

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract JavaScript import/export statements"""
        imports = []
        exports = []

        for line in source_lines:
            line = line.strip()

            # Import statements
            if line.startswith("import ") or line.startswith("const ") and "require(" in line:
                imports.append(line)

            # Export statements
            if line.startswith("export ") or line.startswith("module.exports"):
                exports.append(line)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for JavaScript symbols"""
        start_line = symbol_range.start.line

        context_before = []
        for i in range(start_line - 1, max(0, start_line - 2), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line and not line.startswith("//") and not line.startswith("/*"):
                    context_before.insert(0, source_lines[i])
                    if len(context_before) >= 1:
                        break

        context_after = []
        return context_before, context_after


class JavaExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for Java code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract Java Javadoc comments and regular comments"""
        doc = Documentation()
        start_line = symbol_range.start.line

        # Look for Javadoc comments before the symbol
        javadoc_lines = []
        multiline_start = None
        multiline_end = None

        for i in range(start_line - 1, max(-1, start_line - 20), -1):
            if i < 0:
                break
            line = source_lines[i].strip()
            if line.endswith("*/") and multiline_end is None:
                # Found end of multi-line comment
                multiline_end = i
            elif line.startswith("/**") and multiline_end is not None:
                # Found start of Javadoc comment
                multiline_start = i
                break
            elif not line or line.startswith("//"):
                continue
            else:
                # If we haven't found a multiline end yet, break
                if multiline_end is None:
                    break

        # Process Javadoc comment if found
        if multiline_start is not None and multiline_end is not None:
            for j in range(multiline_start, multiline_end + 1):
                comment_line = source_lines[j].strip()
                if comment_line.startswith("/**"):
                    content = comment_line[3:].strip()
                    # Only add if there's actual content and not just "*" or "*/"
                    if content and content != "*" and not content.startswith("*/"):
                        javadoc_lines.append(content)
                elif comment_line.startswith("*/"):
                    break
                elif comment_line.startswith("*"):
                    content = comment_line[1:].strip()
                    if content.startswith("@"):
                        # Javadoc tag
                        tag_match = re.match(r'@(\w+)\s*(.*)', content)
                        if tag_match:
                            tag_name, tag_content = tag_match.groups()
                            if tag_name not in doc.tags:
                                doc.tags[tag_name] = []
                            doc.tags[tag_name].append(tag_content.strip())
                    elif content and content != "*":
                        javadoc_lines.append(content)

        if javadoc_lines:
            doc.docstring = "\n".join(javadoc_lines)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract Java type information"""
        type_info = TypeInformation()

        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()

                # Extract return type from method signature
                if " -> " in type_text or ":" in type_text:
                    # Look for return type patterns
                    return_match = re.search(r':\s*([A-Za-z_][A-Za-z0-9_<>[\],\s]*)', type_text)
                    if return_match:
                        type_info.return_type = return_match.group(1).strip()

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract Java import statements and public declarations"""
        imports = []
        exports = []

        for line in source_lines:
            line = line.strip()

            # Import statements
            if line.startswith("import "):
                imports.append(line)

            # Public declarations (exports)
            if line.startswith("public "):
                exports.append(line)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for Java symbols"""
        start_line = symbol_range.start.line

        context_before = []
        for i in range(start_line - 1, max(0, start_line - 2), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line and not line.startswith("//") and not line.startswith("/*"):
                    context_before.insert(0, source_lines[i])
                    if len(context_before) >= 1:
                        break

        context_after = []
        return context_before, context_after


class GoExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for Go code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract Go doc comments"""
        doc = Documentation()
        start_line = symbol_range.start.line

        # Look for doc comments before the symbol (// comments immediately before)
        doc_lines = []
        for i in range(start_line - 1, max(-1, start_line - 10), -1):
            if i < 0:
                break
            line = source_lines[i].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
            elif not line:
                # Empty lines can be part of doc comments in Go
                # But we stop if we already collected some lines and hit empty line
                if doc_lines:
                    break
                continue
            else:
                break

        if doc_lines:
            doc.docstring = "\n".join(doc_lines)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract Go type information"""
        type_info = TypeInformation()

        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()

                # Extract return type from function signature
                if "func " in type_text:
                    # Look for return type after closing parenthesis
                    paren_match = re.search(r'\)\s*([A-Za-z_][A-Za-z0-9_*\[\]]*)', type_text)
                    if paren_match:
                        type_info.return_type = paren_match.group(1).strip()

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract Go import statements and exported symbols"""
        imports = []
        exports = []

        in_import_block = False
        for line in source_lines:
            line_stripped = line.strip()

            # Import statements
            if line_stripped.startswith("import "):
                if "(" in line_stripped:
                    in_import_block = True
                imports.append(line_stripped)
            elif in_import_block:
                if ")" in line_stripped:
                    in_import_block = False
                    imports.append(line_stripped)
                elif line_stripped and not line_stripped.startswith("//"):
                    imports.append(line_stripped)

            # Exported symbols (start with capital letter)
            if (line_stripped.startswith("func ") or
                line_stripped.startswith("type ") or
                line_stripped.startswith("var ") or
                line_stripped.startswith("const ")):
                # Check if the symbol name starts with capital letter (exported)
                words = line_stripped.split()
                if len(words) >= 2:
                    symbol_name = words[1].split("(")[0]  # Remove parameters
                    if symbol_name and symbol_name[0].isupper():
                        exports.append(line_stripped)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for Go symbols"""
        start_line = symbol_range.start.line

        context_before = []
        for i in range(start_line - 1, max(0, start_line - 2), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line and not line.startswith("//"):
                    context_before.insert(0, source_lines[i])
                    if len(context_before) >= 1:
                        break

        context_after = []
        return context_before, context_after


class CppExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for C/C++ code"""

    def extract_documentation(self, source_lines: list[str], symbol_range: Range) -> Documentation:
        """Extract C/C++ doc comments (/// and /** */)"""
        doc = Documentation()
        start_line = symbol_range.start.line

        # Look for doc comments before the symbol
        doc_lines = []
        multiline_start = None

        for i in range(start_line - 1, max(-1, start_line - 15), -1):
            if i < 0:
                break
            line = source_lines[i].strip()

            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
            elif line.startswith("/**"):
                # Found start of multi-line doc comment
                multiline_start = i
                break
            elif line.startswith("//") or not line:
                continue
            else:
                break

        # Process multi-line doc comment if found
        if multiline_start is not None:
            comment_lines = []
            for j in range(multiline_start, start_line):
                comment_line = source_lines[j].strip()
                if comment_line.startswith("/**"):
                    content = comment_line[3:].strip()
                    # Only add if there's actual content and not just "*" or "*/"
                    if content and content != "*" and not content.startswith("*/"):
                        comment_lines.append(content)
                elif comment_line.endswith("*/"):
                    content = comment_line[:-2].strip().lstrip("* ")
                    if content and content != "*":
                        comment_lines.append(content)
                    break
                else:
                    content = comment_line.lstrip("* ")
                    if content and content != "*":
                        comment_lines.append(content)
            doc_lines = comment_lines + doc_lines

        if doc_lines:
            doc.docstring = "\n".join(doc_lines)

        return doc

    def extract_type_information(self, symbol_data: dict[str, Any], hover_data: dict[str, Any] | None) -> TypeInformation:
        """Extract C/C++ type information"""
        type_info = TypeInformation()

        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()

                # Extract return type from function signature
                if "(" in type_text and ")" in type_text:
                    # Try to find return type before function name
                    func_match = re.search(r'([A-Za-z_][A-Za-z0-9_*&:<>]+)\s+\w+\s*\(', type_text)
                    if func_match:
                        type_info.return_type = func_match.group(1).strip()

        return type_info

    def extract_imports_exports(self, source_lines: list[str]) -> tuple[list[str], list[str]]:
        """Extract C/C++ include statements and exported symbols"""
        imports = []
        exports = []

        for line in source_lines:
            line = line.strip()

            # Include statements
            if line.startswith("#include "):
                imports.append(line)

            # Exported symbols (extern or in header context)
            if (line.startswith("extern ") or
                (line.startswith("class ") or line.startswith("struct ")) or
                line.startswith("namespace ")):
                exports.append(line)

        return imports, exports

    def get_minimal_context(self, source_lines: list[str], symbol_range: Range) -> tuple[list[str], list[str]]:
        """Get minimal context for C/C++ symbols"""
        start_line = symbol_range.start.line

        context_before = []
        for i in range(start_line - 1, max(0, start_line - 2), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line and not line.startswith("//") and not line.startswith("/*"):
                    context_before.insert(0, source_lines[i])
                    if len(context_before) >= 1:
                        break

        context_after = []
        return context_before, context_after


class LspMetadataExtractor:
    """
    Main LSP-based code metadata extraction system.

    This class orchestrates the extraction of comprehensive code metadata from source files
    using Language Server Protocol (LSP) servers. It manages multiple LSP clients for
    different programming languages and implements the Interface + Minimal Context storage
    strategy for optimal searchability and storage efficiency.

    Key capabilities:
    - Multi-language LSP client management
    - Batch processing with performance optimization
    - Metadata caching for improved performance
    - Relationship graph construction
    - Robust error handling and recovery
    - Integration with file filtering system
    """

    def __init__(
        self,
        file_filter: LanguageAwareFilter | None = None,
        request_timeout: float = 30.0,
        max_concurrent_files: int = 10,
        cache_size: int = 1000,
        enable_relationship_mapping: bool = True
    ):
        """
        Initialize the LSP metadata extractor.

        Args:
            file_filter: File filtering system (created if None)
            request_timeout: LSP request timeout in seconds
            max_concurrent_files: Maximum files to process concurrently
            cache_size: Maximum number of cached file metadata entries
            enable_relationship_mapping: Whether to build relationship graphs
        """
        self.file_filter = file_filter or LanguageAwareFilter()
        self.request_timeout = request_timeout
        self.max_concurrent_files = max_concurrent_files
        self.enable_relationship_mapping = enable_relationship_mapping

        # LSP client management
        self.lsp_clients: dict[str, AsyncioLspClient] = {}
        self.language_extractors: dict[str, LanguageSpecificExtractor] = {
            "python": PythonExtractor(),
            "rust": RustExtractor(),
            "javascript": JavaScriptExtractor(),
            "typescript": JavaScriptExtractor(),  # TypeScript uses same extractor as JS
            "java": JavaExtractor(),
            "go": GoExtractor(),
            "c": CppExtractor(),
            "cpp": CppExtractor(),
        }

        # Language server configurations
        self.lsp_server_configs = {
            "python": {
                "command": ["pylsp"],  # Python LSP Server
                "file_extensions": [".py", ".pyi"],
                "language_id": "python"
            },
            "rust": {
                "command": ["rust-analyzer"],  # Rust Analyzer
                "file_extensions": [".rs"],
                "language_id": "rust"
            },
            "javascript": {
                "command": ["typescript-language-server", "--stdio"],  # TypeScript LSP for JS/TS
                "file_extensions": [".js", ".jsx", ".mjs"],
                "language_id": "javascript"
            },
            "typescript": {
                "command": ["typescript-language-server", "--stdio"],
                "file_extensions": [".ts", ".tsx"],
                "language_id": "typescript"
            },
            "java": {
                "command": ["jdtls"],  # Eclipse JDT Language Server
                "file_extensions": [".java"],
                "language_id": "java"
            },
            "go": {
                "command": ["gopls"],  # Go Language Server
                "file_extensions": [".go"],
                "language_id": "go"
            },
            "c": {
                "command": ["clangd"],  # Clang Language Server
                "file_extensions": [".c", ".h"],
                "language_id": "c"
            },
            "cpp": {
                "command": ["clangd"],
                "file_extensions": [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
                "language_id": "cpp"
            }
        }

        # Caching system
        self.metadata_cache: dict[str, tuple[FileMetadata, float]] = {}  # file_uri -> (metadata, timestamp)
        self.cache_size = cache_size
        self.cache_ttl = 3600.0  # 1 hour TTL

        # Statistics tracking
        self.statistics = ExtractionStatistics()

        # Background processing
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_files)
        self._shutdown_event = asyncio.Event()

        # Initialized state
        self._initialized = False

        logger.info(
            "LSP metadata extractor initialized",
            max_concurrent_files=max_concurrent_files,
            cache_size=cache_size,
            supported_languages=list(self.lsp_server_configs.keys())
        )

    async def initialize(self, workspace_root: str | Path | None = None) -> None:
        """
        Initialize the extractor and its dependencies.

        Args:
            workspace_root: Root directory of the workspace to extract from
        """
        if self._initialized:
            return

        logger.info("Initializing LSP metadata extractor", workspace_root=str(workspace_root) if workspace_root else None)

        # Initialize file filter
        if not self.file_filter._initialized:
            await self.file_filter.load_configuration()

        # Initialize language servers for languages we find in the workspace
        if workspace_root:
            workspace_path = Path(workspace_root)
            detected_languages = await self._detect_languages(workspace_path)

            for language in detected_languages:
                if language in self.lsp_server_configs:
                    try:
                        await self._initialize_lsp_client(language, workspace_path)
                    except Exception as e:
                        logger.warning(
                            "Failed to initialize LSP client",
                            language=language,
                            error=str(e)
                        )
                        self.statistics.lsp_errors += 1

        self._initialized = True
        logger.info("LSP metadata extractor initialization completed")

    async def _detect_languages(self, workspace_path: Path) -> set[str]:
        """Detect programming languages present in the workspace"""
        languages = set()

        for config_name, config in self.lsp_server_configs.items():
            for extension in config["file_extensions"]:
                if list(workspace_path.rglob(f"*{extension}")):
                    languages.add(config_name)
                    break

        logger.debug("Detected languages in workspace", languages=list(languages))
        return languages

    async def _initialize_lsp_client(self, language: str, workspace_path: Path) -> None:
        """Initialize LSP client for a specific language"""
        if language in self.lsp_clients:
            return

        config = self.lsp_server_configs[language]
        client = AsyncioLspClient(
            server_name=f"{language}-lsp",
            request_timeout=self.request_timeout
        )

        try:
            # Connect to LSP server
            await client.connect_stdio(
                server_command=config["command"],
                cwd=str(workspace_path)
            )

            # Initialize the server
            workspace_uri = f"file://{workspace_path.resolve()}"
            await client.initialize(
                root_uri=workspace_uri,
                client_name="workspace-qdrant-mcp",
                client_version="1.0.0"
            )

            self.lsp_clients[language] = client
            logger.info(
                "LSP client initialized",
                language=language,
                command=config["command"][0],
                workspace_uri=workspace_uri
            )

        except Exception as e:
            logger.error(
                "Failed to initialize LSP client",
                language=language,
                command=config["command"],
                error=str(e)
            )
            await client.disconnect()
            raise

    def _get_language_from_file(self, file_path: Path) -> str | None:
        """Determine programming language from file extension"""
        extension = file_path.suffix.lower()

        for language, config in self.lsp_server_configs.items():
            if extension in config["file_extensions"]:
                return language

        return None

    async def extract_file_metadata(
        self,
        file_path: str | Path,
        force_refresh: bool = False
    ) -> FileMetadata | None:
        """
        Extract comprehensive metadata from a single source file.

        Args:
            file_path: Path to the source file
            force_refresh: Whether to bypass cache and force fresh extraction

        Returns:
            FileMetadata object with extracted information or None if extraction failed
        """
        async with self._processing_semaphore:
            return await self._extract_file_metadata_impl(file_path, force_refresh)

    async def _extract_file_metadata_impl(
        self,
        file_path: str | Path,
        force_refresh: bool = False
    ) -> FileMetadata | None:
        """Implementation of file metadata extraction"""
        start_time = time.perf_counter()
        file_path = Path(file_path).resolve()
        file_uri = f"file://{file_path}"

        try:
            # Check file filter
            should_process, filter_reason = self.file_filter.should_process_file(file_path)
            if not should_process:
                logger.debug(
                    "File filtered out",
                    file_path=str(file_path),
                    reason=filter_reason
                )
                return None

            # Check cache
            if not force_refresh and file_uri in self.metadata_cache:
                cached_metadata, cache_time = self.metadata_cache[file_uri]
                if time.time() - cache_time < self.cache_ttl:
                    self.statistics.cache_hits += 1
                    logger.debug("Using cached metadata", file_path=str(file_path))
                    return cached_metadata
                else:
                    # Cache expired
                    del self.metadata_cache[file_uri]

            self.statistics.cache_misses += 1

            # Determine language
            language = self._get_language_from_file(file_path)
            if not language:
                logger.debug("Unknown language for file", file_path=str(file_path))
                return None

            # Get LSP client
            if language not in self.lsp_clients:
                logger.warning(
                    "No LSP client available for language",
                    language=language,
                    file_path=str(file_path)
                )
                return None

            client = self.lsp_clients[language]
            if not client.is_initialized:
                logger.warning(
                    "LSP client not initialized",
                    language=language,
                    file_path=str(file_path)
                )
                return None

            # Read file content
            try:
                content = file_path.read_text(encoding='utf-8')
                source_lines = content.splitlines()
            except Exception as e:
                logger.error(
                    "Failed to read file",
                    file_path=str(file_path),
                    error=str(e)
                )
                self.statistics.files_failed += 1
                return None

            # Create metadata object
            metadata = FileMetadata(
                file_uri=file_uri,
                file_path=str(file_path),
                language=language,
                extraction_timestamp=time.time(),
                lsp_server=f"{language}-lsp"
            )

            # Notify LSP server about file
            try:
                await client.sync_file_opened(
                    str(file_path),
                    content,
                    self.lsp_server_configs[language]["language_id"]
                )
            except Exception as e:
                logger.warning(
                    "Failed to sync file with LSP server",
                    file_path=str(file_path),
                    error=str(e)
                )
                metadata.extraction_errors.append(f"LSP sync failed: {e}")

            # Extract document symbols
            await self._extract_document_symbols(client, file_uri, source_lines, metadata)

            # Extract imports and exports using language-specific extractor
            if language in self.language_extractors:
                extractor = self.language_extractors[language]
                try:
                    imports, exports = extractor.extract_imports_exports(source_lines)
                    metadata.imports = imports
                    metadata.exports = exports
                except Exception as e:
                    logger.warning(
                        "Failed to extract imports/exports",
                        file_path=str(file_path),
                        error=str(e)
                    )
                    metadata.extraction_errors.append(f"Import/export extraction failed: {e}")

            # Extract file-level documentation
            await self._extract_file_documentation(source_lines, metadata)

            # Build relationships if enabled
            if self.enable_relationship_mapping:
                await self._extract_symbol_relationships(client, file_uri, metadata)

            # Update statistics
            self.statistics.files_processed += 1
            self.statistics.symbols_extracted += len(metadata.symbols)
            self.statistics.relationships_found += len(metadata.relationships)

            # Cache the result
            self._cache_metadata(file_uri, metadata)

            # Clean up LSP state
            try:
                await client.sync_file_closed(str(file_path))
            except Exception:
                pass  # Non-critical error

            extraction_time = (time.perf_counter() - start_time) * 1000
            self.statistics.extraction_time_ms += extraction_time

            logger.debug(
                "File metadata extraction completed",
                file_path=str(file_path),
                symbols_count=len(metadata.symbols),
                relationships_count=len(metadata.relationships),
                extraction_time_ms=extraction_time
            )

            return metadata

        except Exception as e:
            self.statistics.files_failed += 1
            extraction_time = (time.perf_counter() - start_time) * 1000
            self.statistics.extraction_time_ms += extraction_time

            logger.error(
                "File metadata extraction failed",
                file_path=str(file_path),
                error=str(e),
                traceback=traceback.format_exc()
            )
            return None

    async def _extract_document_symbols(
        self,
        client: AsyncioLspClient,
        file_uri: str,
        source_lines: list[str],
        metadata: FileMetadata
    ) -> None:
        """Extract symbols from document using LSP document symbol request"""
        try:
            self.statistics.lsp_requests_made += 1
            symbols_data = await client.document_symbol(file_uri)

            if not symbols_data:
                return

            language = metadata.language
            extractor = self.language_extractors.get(language)

            # Process symbols recursively (LSP can return nested symbols)
            await self._process_symbol_hierarchy(
                symbols_data,
                file_uri,
                source_lines,
                metadata,
                extractor,
                client
            )

        except Exception as e:
            logger.warning(
                "Failed to extract document symbols",
                file_uri=file_uri,
                error=str(e)
            )
            metadata.extraction_errors.append(f"Document symbols extraction failed: {e}")
            self.statistics.lsp_errors += 1

    async def _process_symbol_hierarchy(
        self,
        symbols_data: list[dict[str, Any]],
        file_uri: str,
        source_lines: list[str],
        metadata: FileMetadata,
        extractor: LanguageSpecificExtractor | None,
        client: AsyncioLspClient,
        parent_symbol: str | None = None
    ) -> None:
        """Process symbol hierarchy recursively"""
        for symbol_data in symbols_data:
            try:
                # Create CodeSymbol from LSP data
                symbol = await self._create_code_symbol(
                    symbol_data,
                    file_uri,
                    source_lines,
                    metadata,
                    extractor,
                    client,
                    parent_symbol
                )

                if symbol:
                    metadata.symbols.append(symbol)

                    # Process children if present
                    children = symbol_data.get("children", [])
                    if children:
                        await self._process_symbol_hierarchy(
                            children,
                            file_uri,
                            source_lines,
                            metadata,
                            extractor,
                            client,
                            symbol.get_full_name()
                        )

            except Exception as e:
                logger.warning(
                    "Failed to process symbol",
                    symbol_name=symbol_data.get("name", "unknown"),
                    error=str(e)
                )
                metadata.extraction_errors.append(f"Symbol processing failed: {e}")

    async def _create_code_symbol(
        self,
        symbol_data: dict[str, Any],
        file_uri: str,
        source_lines: list[str],
        metadata: FileMetadata,
        extractor: LanguageSpecificExtractor | None,
        client: AsyncioLspClient,
        parent_symbol: str | None = None
    ) -> CodeSymbol | None:
        """Create a CodeSymbol from LSP symbol data"""
        try:
            name = symbol_data.get("name", "")
            kind_value = symbol_data.get("kind", 1)

            # Validate symbol data - log errors for malformed data
            if not name:
                error_msg = f"Symbol missing name field: {symbol_data}"
                logger.debug(error_msg)
                metadata.extraction_errors.append(error_msg)

            if "range" not in symbol_data:
                error_msg = f"Symbol '{name}' missing range field"
                logger.debug(error_msg)
                metadata.extraction_errors.append(error_msg)

            # Convert LSP kind to our SymbolKind
            try:
                kind = SymbolKind(kind_value)
            except ValueError:
                kind = SymbolKind.VARIABLE  # Default fallback

            # Extract ranges
            range_data = symbol_data.get("range", {})
            selection_range_data = symbol_data.get("selectionRange")

            symbol_range = Range.from_lsp(range_data)
            selection_range = Range.from_lsp(selection_range_data) if selection_range_data else None

            # Create symbol object
            symbol = CodeSymbol(
                name=name,
                kind=kind,
                file_uri=file_uri,
                range=symbol_range,
                selection_range=selection_range,
                language=metadata.language,
                parent_symbol=parent_symbol
            )

            # Extract additional metadata using language-specific extractor
            if extractor:
                try:
                    # Extract documentation
                    symbol.documentation = extractor.extract_documentation(source_lines, symbol_range)

                    # Extract minimal context
                    context_before, context_after = extractor.get_minimal_context(source_lines, symbol_range)
                    symbol.context_before = context_before
                    symbol.context_after = context_after

                except Exception as e:
                    logger.debug(
                        "Language-specific extraction failed",
                        symbol_name=name,
                        error=str(e)
                    )

            # Get hover information for type data
            if selection_range:
                try:
                    self.statistics.lsp_requests_made += 1
                    hover_data = await client.hover(
                        file_uri,
                        selection_range.start.line,
                        selection_range.start.character
                    )

                    if hover_data and extractor:
                        symbol.type_info = extractor.extract_type_information(symbol_data, hover_data)

                except Exception as e:
                    logger.debug(
                        "Failed to get hover information",
                        symbol_name=name,
                        error=str(e)
                    )

            # Extract additional symbol metadata
            symbol.deprecated = symbol_data.get("deprecated", False)
            if symbol_data.get("tags"):
                symbol.tags = [str(tag) for tag in symbol_data["tags"]]

            return symbol

        except Exception as e:
            logger.warning(
                "Failed to create symbol",
                symbol_data=symbol_data,
                error=str(e)
            )
            return None

    async def _extract_file_documentation(
        self,
        source_lines: list[str],
        metadata: FileMetadata
    ) -> None:
        """Extract file-level documentation"""
        try:
            # Look for file-level docstring or header comments in first 20 lines
            doc_lines = []
            in_docstring = False
            docstring_quote = None

            for _i, line in enumerate(source_lines[:20]):
                line = line.strip()

                # Python-style module docstring
                if not in_docstring and (line.startswith('"""') or line.startswith("'''")):
                    docstring_quote = '"""' if line.startswith('"""') else "'''"
                    if line.endswith(docstring_quote) and len(line) > 6:
                        # Single line docstring
                        doc_lines.append(line[3:-3].strip())
                        break
                    else:
                        in_docstring = True
                        doc_lines.append(line[3:])
                elif in_docstring:
                    if line.endswith(docstring_quote):
                        doc_lines.append(line[:-3])
                        break
                    else:
                        doc_lines.append(line)

                # File header comments
                elif line.startswith('#') or line.startswith('//') or line.startswith('/*'):
                    comment_text = line.lstrip('#/ *').strip()
                    if comment_text:
                        metadata.file_comments.append(comment_text)

                # Stop at first non-comment, non-docstring line
                elif line and not line.startswith(('import ', 'from ', 'use ', 'package ', 'namespace ')):
                    break

            if doc_lines:
                metadata.file_docstring = "\n".join(doc_lines).strip()

        except Exception as e:
            logger.debug("Failed to extract file documentation", error=str(e))

    async def _extract_symbol_relationships(
        self,
        client: AsyncioLspClient,
        file_uri: str,
        metadata: FileMetadata
    ) -> None:
        """Extract relationships between symbols"""
        try:
            # For each symbol, try to find references and definitions
            for symbol in metadata.symbols:
                if not symbol.selection_range:
                    continue

                try:
                    # Find references to this symbol
                    self.statistics.lsp_requests_made += 1
                    references = await client.references(
                        file_uri,
                        symbol.selection_range.start.line,
                        symbol.selection_range.start.character,
                        include_declaration=False
                    )

                    if references:
                        for ref in references:
                            ref_uri = ref.get("uri", "")
                            if ref_uri != file_uri:  # Cross-file reference
                                relationship = SymbolRelationship(
                                    from_symbol=symbol.get_full_name(),
                                    to_symbol=ref_uri,  # Could be refined with symbol name
                                    relationship_type=RelationshipType.REFERENCES,
                                    file_uri=file_uri,
                                    location=Range.from_lsp(ref.get("range", {}))
                                )
                                metadata.relationships.append(relationship)

                    # Find definitions
                    self.statistics.lsp_requests_made += 1
                    definitions = await client.definition(
                        file_uri,
                        symbol.selection_range.start.line,
                        symbol.selection_range.start.character
                    )

                    if definitions:
                        for definition in definitions:
                            def_uri = definition.get("uri", "")
                            if def_uri != file_uri:  # External definition
                                relationship = SymbolRelationship(
                                    from_symbol=symbol.get_full_name(),
                                    to_symbol=def_uri,
                                    relationship_type=RelationshipType.DEFINES,
                                    file_uri=file_uri,
                                    location=Range.from_lsp(definition.get("range", {}))
                                )
                                metadata.relationships.append(relationship)

                except Exception as e:
                    logger.debug(
                        "Failed to extract relationships for symbol",
                        symbol_name=symbol.name,
                        error=str(e)
                    )

        except Exception as e:
            logger.warning(
                "Failed to extract symbol relationships",
                file_uri=file_uri,
                error=str(e)
            )
            metadata.extraction_errors.append(f"Relationship extraction failed: {e}")

    def _cache_metadata(self, file_uri: str, metadata: FileMetadata) -> None:
        """Cache extracted metadata with TTL"""
        # Limit cache size
        if len(self.metadata_cache) >= self.cache_size:
            # Remove oldest entries
            oldest_entries = sorted(
                self.metadata_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )[:100]  # Remove 100 oldest

            for uri, _ in oldest_entries:
                del self.metadata_cache[uri]

        self.metadata_cache[file_uri] = (metadata, time.time())

    async def extract_directory_metadata(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        file_pattern: str = "*"
    ) -> list[FileMetadata]:
        """
        Extract metadata from all eligible files in a directory.

        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            file_pattern: Glob pattern for file matching

        Returns:
            List of FileMetadata objects for successfully processed files
        """
        if not self._initialized:
            await self.initialize(directory_path)

        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        logger.info(
            "Starting directory metadata extraction",
            directory=str(directory_path),
            recursive=recursive,
            pattern=file_pattern
        )

        # Find all files to process
        if recursive:
            files = list(directory_path.rglob(file_pattern))
        else:
            files = list(directory_path.glob(file_pattern))

        # Filter files
        eligible_files = []
        for file_path in files:
            if file_path.is_file():
                should_process, _ = self.file_filter.should_process_file(file_path)
                if should_process:
                    eligible_files.append(file_path)

        logger.info(
            "Found eligible files for processing",
            total_files=len(files),
            eligible_files=len(eligible_files)
        )

        # Process files concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent_files)

        async def process_file(file_path: Path) -> FileMetadata | None:
            async with semaphore:
                return await self.extract_file_metadata(file_path)

        # Execute batch processing
        tasks = [process_file(file_path) for file_path in eligible_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        successful_metadata = []
        for result in results:
            if isinstance(result, FileMetadata):
                successful_metadata.append(result)
            elif isinstance(result, Exception):
                logger.error("File processing failed with exception", error=str(result))
                self.statistics.files_failed += 1

        logger.info(
            "Directory metadata extraction completed",
            directory=str(directory_path),
            files_processed=len(successful_metadata),
            files_failed=self.statistics.files_failed
        )

        return successful_metadata

    async def build_relationship_graph(
        self,
        file_paths: list[str | Path]
    ) -> dict[str, list[SymbolRelationship]]:
        """
        Build comprehensive relationship graph across multiple files.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary mapping symbol names to their relationships
        """
        if not self.enable_relationship_mapping:
            logger.warning("Relationship mapping is disabled")
            return {}

        logger.info("Building relationship graph", files_count=len(file_paths))

        # Extract metadata from all files
        all_metadata = []
        for file_path in file_paths:
            metadata = await self.extract_file_metadata(file_path)
            if metadata:
                all_metadata.append(metadata)

        # Build comprehensive relationship map
        relationship_graph: dict[str, list[SymbolRelationship]] = {}

        # Collect all symbols
        all_symbols: dict[str, CodeSymbol] = {}
        for metadata in all_metadata:
            for symbol in metadata.symbols:
                full_name = symbol.get_full_name()
                all_symbols[full_name] = symbol

        # Process import/export relationships
        for metadata in all_metadata:
            for import_stmt in metadata.imports:
                # Parse import to find relationships
                # This is language-specific but simplified here
                imported_names = self._parse_import_statement(import_stmt, metadata.language)
                for imported_name in imported_names:
                    if imported_name in all_symbols:
                        relationship = SymbolRelationship(
                            from_symbol=metadata.file_uri,
                            to_symbol=imported_name,
                            relationship_type=RelationshipType.IMPORTS,
                            file_uri=metadata.file_uri
                        )

                        if metadata.file_uri not in relationship_graph:
                            relationship_graph[metadata.file_uri] = []
                        relationship_graph[metadata.file_uri].append(relationship)

        # Add symbol-level relationships
        for metadata in all_metadata:
            for relationship in metadata.relationships:
                symbol_name = relationship.from_symbol
                if symbol_name not in relationship_graph:
                    relationship_graph[symbol_name] = []
                relationship_graph[symbol_name].append(relationship)

        logger.info(
            "Relationship graph completed",
            symbols_count=len(all_symbols),
            relationships_count=sum(len(rels) for rels in relationship_graph.values())
        )

        return relationship_graph

    def _parse_import_statement(self, import_stmt: str, language: str) -> list[str]:
        """Parse import statement to extract imported symbol names"""
        # Simplified parsing - could be enhanced with AST parsing
        imported = []

        if language == "python":
            if import_stmt.startswith("from "):
                # from module import name1, name2
                match = re.search(r'from\s+[\w.]+\s+import\s+(.+)', import_stmt)
                if match:
                    imports = match.group(1)
                    for name in imports.split(','):
                        name = name.strip().split(' as ')[0].strip()
                        imported.append(name)
            elif import_stmt.startswith("import "):
                # import module.name
                match = re.search(r'import\s+([\w.]+)', import_stmt)
                if match:
                    imported.append(match.group(1))

        elif language in ["javascript", "typescript"]:
            # import { name1, name2 } from 'module'
            if "from" in import_stmt:
                match = re.search(r'import\s*{([^}]+)}\s*from', import_stmt)
                if match:
                    imports = match.group(1)
                    for name in imports.split(','):
                        name = name.strip().split(' as ')[0].strip()
                        imported.append(name)

        return imported

    def get_statistics(self) -> ExtractionStatistics:
        """Get current extraction statistics"""
        return self.statistics

    def reset_statistics(self) -> None:
        """Reset extraction statistics"""
        self.statistics = ExtractionStatistics()

    def clear_cache(self) -> None:
        """Clear metadata cache"""
        self.metadata_cache.clear()
        logger.info("Metadata cache cleared")

    async def shutdown(self) -> None:
        """Shutdown the extractor and clean up resources"""
        logger.info("Shutting down LSP metadata extractor")
        self._shutdown_event.set()

        # Disconnect all LSP clients
        for language, client in self.lsp_clients.items():
            try:
                await client.disconnect()
                logger.debug("LSP client disconnected", language=language)
            except Exception as e:
                logger.warning(
                    "Error disconnecting LSP client",
                    language=language,
                    error=str(e)
                )

        self.lsp_clients.clear()
        self.metadata_cache.clear()
        self._initialized = False

        logger.info("LSP metadata extractor shutdown completed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
