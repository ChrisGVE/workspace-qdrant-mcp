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
import json
import re
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import structlog

from .error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from .language_filters import LanguageAwareFilter
from .lsp_client import AsyncioLspClient, LspError, ConnectionState

logger = structlog.get_logger(__name__)


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
    
    def to_dict(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}
    
    @classmethod
    def from_lsp(cls, lsp_position: Dict[str, Any]) -> "Position":
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict()
        }
    
    @classmethod
    def from_lsp(cls, lsp_range: Dict[str, Any]) -> "Range":
        """Create Range from LSP range data"""
        return cls(
            start=Position.from_lsp(lsp_range.get("start", {})),
            end=Position.from_lsp(lsp_range.get("end", {}))
        )


@dataclass
class TypeInformation:
    """Type information for symbols"""
    type_name: Optional[str] = None
    type_signature: Optional[str] = None
    parameter_types: List[Dict[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    generic_parameters: List[str] = field(default_factory=list)
    nullable: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
    docstring: Optional[str] = None
    inline_comments: List[str] = field(default_factory=list)
    leading_comments: List[str] = field(default_factory=list)
    trailing_comments: List[str] = field(default_factory=list)
    tags: Dict[str, List[str]] = field(default_factory=dict)  # JSDoc-style tags
    
    def to_dict(self) -> Dict[str, Any]:
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
    selection_range: Optional[Range] = None
    
    # Type information
    type_info: Optional[TypeInformation] = None
    
    # Documentation
    documentation: Optional[Documentation] = None
    
    # Context (1-3 lines around the symbol)
    context_before: List[str] = field(default_factory=list)  # 1-2 lines before
    context_after: List[str] = field(default_factory=list)   # 0-1 lines after
    
    # Symbol metadata
    visibility: Optional[str] = None  # public, private, protected, etc.
    modifiers: List[str] = field(default_factory=list)  # static, final, async, etc.
    language: Optional[str] = None
    
    # Symbol relationships
    parent_symbol: Optional[str] = None  # Parent class/namespace
    children: List[str] = field(default_factory=list)  # Child symbols
    
    # Additional metadata
    deprecated: bool = False
    experimental: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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
        if self.type_info and self.type_info.type_signature:
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
            signature_parts.append(self.name)
            if self.type_info and self.type_info.type_name:
                signature_parts.append(f": {self.type_info.type_name}")
        
        return " ".join(signature_parts)


@dataclass
class SymbolRelationship:
    """Relationship between two code symbols"""
    from_symbol: str  # Symbol identifier
    to_symbol: str    # Symbol identifier  
    relationship_type: RelationshipType
    file_uri: str     # File where relationship is defined
    location: Optional[Range] = None  # Location of relationship reference
    
    def to_dict(self) -> Dict[str, Any]:
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
    symbols: List[CodeSymbol] = field(default_factory=list)
    relationships: List[SymbolRelationship] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    
    # File-level documentation and metadata
    file_docstring: Optional[str] = None
    file_comments: List[str] = field(default_factory=list)
    
    # Extraction metadata
    extraction_timestamp: Optional[float] = None
    lsp_server: Optional[str] = None
    extraction_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    def extract_documentation(self, source_lines: List[str], symbol_range: Range) -> Documentation:
        """Extract documentation for a symbol from source code"""
        pass
    
    @abstractmethod
    def extract_type_information(self, symbol_data: Dict[str, Any], hover_data: Optional[Dict[str, Any]]) -> TypeInformation:
        """Extract type information from LSP symbol and hover data"""
        pass
    
    @abstractmethod
    def extract_imports_exports(self, source_lines: List[str]) -> Tuple[List[str], List[str]]:
        """Extract import and export statements from source code"""
        pass
    
    @abstractmethod
    def get_minimal_context(self, source_lines: List[str], symbol_range: Range) -> Tuple[List[str], List[str]]:
        """Get 1-3 lines of context around a symbol (before, after)"""
        pass


class PythonExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for Python code"""
    
    def extract_documentation(self, source_lines: List[str], symbol_range: Range) -> Documentation:
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
    
    def extract_type_information(self, symbol_data: Dict[str, Any], hover_data: Optional[Dict[str, Any]]) -> TypeInformation:
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
    
    def extract_imports_exports(self, source_lines: List[str]) -> Tuple[List[str], List[str]]:
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
    
    def get_minimal_context(self, source_lines: List[str], symbol_range: Range) -> Tuple[List[str], List[str]]:
        """Get minimal context for Python symbols"""
        start_line = symbol_range.start.line
        
        # Get 1-2 lines before (excluding empty lines and comments)
        context_before = []
        for i in range(start_line - 1, max(0, start_line - 3), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line and not line.startswith('#'):
                    context_before.insert(0, source_lines[i])
                    if len(context_before) >= 2:
                        break
        
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
    
    def extract_documentation(self, source_lines: List[str], symbol_range: Range) -> Documentation:
        """Extract Rust doc comments (/// and //!)"""
        doc = Documentation()
        start_line = symbol_range.start.line
        
        # Look for doc comments before the symbol
        doc_lines = []
        for i in range(start_line - 1, max(0, start_line - 20), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line.startswith("///") or line.startswith("//!"):
                    doc_lines.insert(0, line[3:].strip())
                elif line.startswith("/**") or line.startswith("/*!"):
                    # Multi-line doc comment
                    comment_lines = [line[3:].strip()]
                    for j in range(i + 1, start_line):
                        comment_line = source_lines[j].strip()
                        if comment_line.endswith("*/"):
                            comment_lines.append(comment_line[:-2].strip())
                            break
                        else:
                            comment_lines.append(comment_line.lstrip("* "))
                    doc_lines = comment_lines + doc_lines
                    break
                elif not line or line.startswith("//"):
                    continue
                else:
                    break
        
        if doc_lines:
            doc.docstring = "\n".join(doc_lines)
        
        return doc
    
    def extract_type_information(self, symbol_data: Dict[str, Any], hover_data: Optional[Dict[str, Any]]) -> TypeInformation:
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
    
    def extract_imports_exports(self, source_lines: List[str]) -> Tuple[List[str], List[str]]:
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
    
    def get_minimal_context(self, source_lines: List[str], symbol_range: Range) -> Tuple[List[str], List[str]]:
        """Get minimal context for Rust symbols"""
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


class JavaScriptExtractor(LanguageSpecificExtractor):
    """Language-specific extractor for JavaScript/TypeScript code"""
    
    def extract_documentation(self, source_lines: List[str], symbol_range: Range) -> Documentation:
        """Extract JSDoc comments and regular comments"""
        doc = Documentation()
        start_line = symbol_range.start.line
        
        # Look for JSDoc comments before the symbol
        for i in range(start_line - 1, max(0, start_line - 20), -1):
            if i >= 0:
                line = source_lines[i].strip()
                if line.startswith("/**"):
                    # Multi-line JSDoc comment
                    jsdoc_lines = []
                    for j in range(i, start_line):
                        comment_line = source_lines[j].strip()
                        if comment_line.startswith("/**"):
                            jsdoc_lines.append(comment_line[3:].strip())
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
                            else:
                                jsdoc_lines.append(content)
                    
                    if jsdoc_lines:
                        doc.docstring = "\n".join(jsdoc_lines)
                    break
                elif not line or line.startswith("//"):
                    continue
                else:
                    break
        
        return doc
    
    def extract_type_information(self, symbol_data: Dict[str, Any], hover_data: Optional[Dict[str, Any]]) -> TypeInformation:
        """Extract JavaScript/TypeScript type information"""
        type_info = TypeInformation()
        
        if hover_data and hover_data.get("contents"):
            contents = hover_data["contents"]
            if isinstance(contents, dict) and contents.get("value"):
                type_text = contents["value"]
                type_info.type_signature = type_text.strip()
                
                # Extract TypeScript type annotations
                if ": " in type_text:
                    type_part = type_text.split(": ")[1].split("=")[0].strip()
                    type_info.type_name = type_part
        
        return type_info
    
    def extract_imports_exports(self, source_lines: List[str]) -> Tuple[List[str], List[str]]:
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
    
    def get_minimal_context(self, source_lines: List[str], symbol_range: Range) -> Tuple[List[str], List[str]]:
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