"""
Comprehensive unit tests for LSP metadata extraction system.

This test suite achieves high coverage of the lsp_metadata_extractor module
by testing all major classes, methods, and code paths with extensive mocking.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.python.common.core.language_filters import LanguageAwareFilter
from src.python.common.core.lsp_client import (
    AsyncioLspClient,
    ConnectionState,
    LspError,
)
from src.python.common.core.lsp_metadata_extractor import (
    CodeSymbol,
    CppExtractor,
    Documentation,
    ExtractionStatistics,
    FileMetadata,
    GoExtractor,
    JavaExtractor,
    JavaScriptExtractor,
    LanguageSpecificExtractor,
    # Core classes
    LspMetadataExtractor,
    # Data classes
    Position,
    # Language extractors
    PythonExtractor,
    Range,
    RelationshipType,
    RustExtractor,
    # Enums
    SymbolKind,
    SymbolRelationship,
    TypeInformation,
)


class TestPosition:
    """Test Position data class"""

    def test_position_creation(self):
        """Test Position object creation and methods"""
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5

    def test_position_to_dict(self):
        """Test Position to_dict method"""
        pos = Position(line=15, character=8)
        result = pos.to_dict()
        expected = {"line": 15, "character": 8}
        assert result == expected

    def test_position_from_lsp(self):
        """Test Position creation from LSP data"""
        lsp_data = {"line": 20, "character": 12}
        pos = Position.from_lsp(lsp_data)
        assert pos.line == 20
        assert pos.character == 12

    def test_position_from_lsp_defaults(self):
        """Test Position creation from LSP data with defaults"""
        lsp_data = {}
        pos = Position.from_lsp(lsp_data)
        assert pos.line == 0
        assert pos.character == 0

    def test_position_from_lsp_partial(self):
        """Test Position creation from partial LSP data"""
        lsp_data = {"line": 25}
        pos = Position.from_lsp(lsp_data)
        assert pos.line == 25
        assert pos.character == 0


class TestRange:
    """Test Range data class"""

    def test_range_creation(self):
        """Test Range object creation"""
        start = Position(10, 5)
        end = Position(15, 20)
        range_obj = Range(start=start, end=end)
        assert range_obj.start == start
        assert range_obj.end == end

    def test_range_to_dict(self):
        """Test Range to_dict method"""
        start = Position(10, 5)
        end = Position(15, 20)
        range_obj = Range(start=start, end=end)
        result = range_obj.to_dict()
        expected = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 15, "character": 20}
        }
        assert result == expected

    def test_range_from_lsp(self):
        """Test Range creation from LSP data"""
        lsp_data = {
            "start": {"line": 5, "character": 10},
            "end": {"line": 8, "character": 25}
        }
        range_obj = Range.from_lsp(lsp_data)
        assert range_obj.start.line == 5
        assert range_obj.start.character == 10
        assert range_obj.end.line == 8
        assert range_obj.end.character == 25

    def test_range_from_lsp_empty(self):
        """Test Range creation from empty LSP data"""
        lsp_data = {}
        range_obj = Range.from_lsp(lsp_data)
        assert range_obj.start.line == 0
        assert range_obj.start.character == 0
        assert range_obj.end.line == 0
        assert range_obj.end.character == 0


class TestTypeInformation:
    """Test TypeInformation data class"""

    def test_type_information_creation(self):
        """Test TypeInformation creation with all fields"""
        type_info = TypeInformation(
            type_name="str",
            type_signature="def func(x: int) -> str",
            parameter_types=[{"name": "x", "type": "int"}],
            return_type="str",
            generic_parameters=["T"],
            nullable=True
        )
        assert type_info.type_name == "str"
        assert type_info.type_signature == "def func(x: int) -> str"
        assert type_info.parameter_types == [{"name": "x", "type": "int"}]
        assert type_info.return_type == "str"
        assert type_info.generic_parameters == ["T"]
        assert type_info.nullable is True

    def test_type_information_defaults(self):
        """Test TypeInformation with default values"""
        type_info = TypeInformation()
        assert type_info.type_name is None
        assert type_info.type_signature is None
        assert type_info.parameter_types == []
        assert type_info.return_type is None
        assert type_info.generic_parameters == []
        assert type_info.nullable is None

    def test_type_information_to_dict(self):
        """Test TypeInformation to_dict method"""
        type_info = TypeInformation(
            type_name="int",
            return_type="bool",
            nullable=False
        )
        result = type_info.to_dict()
        expected = {
            "type_name": "int",
            "type_signature": None,
            "parameter_types": [],
            "return_type": "bool",
            "generic_parameters": [],
            "nullable": False
        }
        assert result == expected


class TestDocumentation:
    """Test Documentation data class"""

    def test_documentation_creation(self):
        """Test Documentation creation with all fields"""
        doc = Documentation(
            docstring="Main function docstring",
            inline_comments=["inline comment"],
            leading_comments=["leading comment"],
            trailing_comments=["trailing comment"],
            tags={"param": ["x: input value"], "returns": ["result"]}
        )
        assert doc.docstring == "Main function docstring"
        assert doc.inline_comments == ["inline comment"]
        assert doc.leading_comments == ["leading comment"]
        assert doc.trailing_comments == ["trailing comment"]
        assert doc.tags == {"param": ["x: input value"], "returns": ["result"]}

    def test_documentation_defaults(self):
        """Test Documentation with default values"""
        doc = Documentation()
        assert doc.docstring is None
        assert doc.inline_comments == []
        assert doc.leading_comments == []
        assert doc.trailing_comments == []
        assert doc.tags == {}

    def test_documentation_to_dict(self):
        """Test Documentation to_dict method"""
        doc = Documentation(docstring="Test doc", tags={"test": ["value"]})
        result = doc.to_dict()
        expected = {
            "docstring": "Test doc",
            "inline_comments": [],
            "leading_comments": [],
            "trailing_comments": [],
            "tags": {"test": ["value"]}
        }
        assert result == expected


class TestCodeSymbol:
    """Test CodeSymbol data class"""

    def test_code_symbol_creation(self):
        """Test CodeSymbol creation with required fields"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        symbol = CodeSymbol(
            name="test_function",
            kind=SymbolKind.FUNCTION,
            file_uri="file:///test.py",
            range=range_obj
        )
        assert symbol.name == "test_function"
        assert symbol.kind == SymbolKind.FUNCTION
        assert symbol.file_uri == "file:///test.py"
        assert symbol.range == range_obj

    def test_code_symbol_get_full_name_without_parent(self):
        """Test get_full_name without parent symbol"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        symbol = CodeSymbol(
            name="test_function",
            kind=SymbolKind.FUNCTION,
            file_uri="file:///test.py",
            range=range_obj
        )
        assert symbol.get_full_name() == "test_function"

    def test_code_symbol_get_full_name_with_parent(self):
        """Test get_full_name with parent symbol"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        symbol = CodeSymbol(
            name="method",
            kind=SymbolKind.METHOD,
            file_uri="file:///test.py",
            range=range_obj,
            parent_symbol="TestClass"
        )
        assert symbol.get_full_name() == "TestClass.method"

    def test_code_symbol_get_signature_function(self):
        """Test get_signature for function with type info"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        type_info = TypeInformation(
            type_signature="def func(x: int, y: str) -> bool",
            parameter_types=[
                {"name": "x", "type": "int"},
                {"name": "y", "type": "str"}
            ],
            return_type="bool"
        )
        symbol = CodeSymbol(
            name="func",
            kind=SymbolKind.FUNCTION,
            file_uri="file:///test.py",
            range=range_obj,
            type_info=type_info,
            modifiers=["async"]
        )
        signature = symbol.get_signature()
        expected = "async def func(x: int, y: str) -> bool"
        assert signature == expected

    def test_code_symbol_get_signature_basic_function(self):
        """Test get_signature for function without detailed type info"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        symbol = CodeSymbol(
            name="simple_func",
            kind=SymbolKind.FUNCTION,
            file_uri="file:///test.py",
            range=range_obj
        )
        signature = symbol.get_signature()
        assert signature == "simple_func()"

    def test_code_symbol_get_signature_variable(self):
        """Test get_signature for variable with type"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        type_info = TypeInformation(type_name="int")
        symbol = CodeSymbol(
            name="count",
            kind=SymbolKind.VARIABLE,
            file_uri="file:///test.py",
            range=range_obj,
            type_info=type_info,
            modifiers=["static"]
        )
        signature = symbol.get_signature()
        assert signature == "static count: int"

    def test_code_symbol_to_dict(self):
        """Test CodeSymbol to_dict method"""
        range_obj = Range(Position(10, 0), Position(10, 10))
        symbol = CodeSymbol(
            name="test_symbol",
            kind=SymbolKind.CLASS,
            file_uri="file:///test.py",
            range=range_obj,
            deprecated=True,
            tags=["important"]
        )
        result = symbol.to_dict()
        assert result["name"] == "test_symbol"
        assert result["kind"] == SymbolKind.CLASS.value
        assert result["kind_name"] == SymbolKind.CLASS.name
        assert result["file_uri"] == "file:///test.py"
        assert result["deprecated"] is True
        assert result["tags"] == ["important"]


class TestSymbolRelationship:
    """Test SymbolRelationship data class"""

    def test_symbol_relationship_creation(self):
        """Test SymbolRelationship creation"""
        relationship = SymbolRelationship(
            from_symbol="ClassA.method1",
            to_symbol="ClassB.method2",
            relationship_type=RelationshipType.CALLS,
            file_uri="file:///test.py"
        )
        assert relationship.from_symbol == "ClassA.method1"
        assert relationship.to_symbol == "ClassB.method2"
        assert relationship.relationship_type == RelationshipType.CALLS
        assert relationship.file_uri == "file:///test.py"

    def test_symbol_relationship_to_dict(self):
        """Test SymbolRelationship to_dict method"""
        range_obj = Range(Position(5, 10), Position(5, 20))
        relationship = SymbolRelationship(
            from_symbol="main",
            to_symbol="helper",
            relationship_type=RelationshipType.CALLS,
            file_uri="file:///main.py",
            location=range_obj
        )
        result = relationship.to_dict()
        expected = {
            "from_symbol": "main",
            "to_symbol": "helper",
            "relationship_type": "calls",
            "file_uri": "file:///main.py",
            "location": {
                "start": {"line": 5, "character": 10},
                "end": {"line": 5, "character": 20}
            }
        }
        assert result == expected


class TestFileMetadata:
    """Test FileMetadata data class"""

    def test_file_metadata_creation(self):
        """Test FileMetadata creation"""
        metadata = FileMetadata(
            file_uri="file:///test.py",
            file_path="/path/to/test.py",
            language="python"
        )
        assert metadata.file_uri == "file:///test.py"
        assert metadata.file_path == "/path/to/test.py"
        assert metadata.language == "python"
        assert metadata.symbols == []
        assert metadata.relationships == []

    def test_file_metadata_to_dict(self):
        """Test FileMetadata to_dict method"""
        range_obj = Range(Position(0, 0), Position(1, 0))
        symbol = CodeSymbol("test", SymbolKind.FUNCTION, "file:///test.py", range_obj)
        relationship = SymbolRelationship("test", "other", RelationshipType.CALLS, "file:///test.py")

        metadata = FileMetadata(
            file_uri="file:///test.py",
            file_path="/test.py",
            language="python",
            symbols=[symbol],
            relationships=[relationship],
            imports=["import os"],
            exports=["test"],
            file_docstring="Test module",
            extraction_timestamp=1234567890.0,
            lsp_server="python-lsp"
        )

        result = metadata.to_dict()
        assert result["file_uri"] == "file:///test.py"
        assert result["language"] == "python"
        assert result["symbol_count"] == 1
        assert result["relationship_count"] == 1
        assert result["imports"] == ["import os"]
        assert result["exports"] == ["test"]
        assert result["file_docstring"] == "Test module"


class TestExtractionStatistics:
    """Test ExtractionStatistics data class"""

    def test_extraction_statistics_creation(self):
        """Test ExtractionStatistics creation with defaults"""
        stats = ExtractionStatistics()
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.symbols_extracted == 0
        assert stats.cache_hits == 0

    def test_extraction_statistics_to_dict(self):
        """Test ExtractionStatistics to_dict method with calculations"""
        stats = ExtractionStatistics(
            files_processed=10,
            files_failed=2,
            symbols_extracted=50,
            relationships_found=25,
            extraction_time_ms=1500.0
        )
        result = stats.to_dict()
        assert result["files_processed"] == 10
        assert result["files_failed"] == 2
        assert result["success_rate"] == 10 / 12  # 10 / (10 + 2)
        assert result["symbols_per_file"] == 5.0  # 50 / 10
        assert result["relationships_per_file"] == 2.5  # 25 / 10

    def test_extraction_statistics_edge_cases(self):
        """Test ExtractionStatistics edge cases with zero values"""
        stats = ExtractionStatistics()
        result = stats.to_dict()
        assert result["success_rate"] == 0.0
        assert result["symbols_per_file"] == 0.0
        assert result["relationships_per_file"] == 0.0


class TestPythonExtractor:
    """Test PythonExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = PythonExtractor()

    def test_extract_documentation_single_line_docstring(self):
        """Test extraction of single-line docstring"""
        source_lines = [
            "def test_function():",
            '    """This is a test function."""',
            "    pass"
        ]
        range_obj = Range(Position(0, 0), Position(0, 20))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        assert doc.docstring == "This is a test function."

    def test_extract_documentation_multi_line_docstring(self):
        """Test extraction of multi-line docstring"""
        source_lines = [
            "def complex_function():",
            '    """',
            "    This is a complex function",
            "    that does multiple things.",
            '    """',
            "    pass"
        ]
        range_obj = Range(Position(0, 0), Position(0, 22))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        expected = "This is a complex function\nthat does multiple things."
        assert doc.docstring == expected

    def test_extract_documentation_comments(self):
        """Test extraction of comments around symbol"""
        source_lines = [
            "# This is a leading comment",
            "# Another leading comment",
            "def function():  # inline comment",
            "    pass",
            "# trailing comment"
        ]
        range_obj = Range(Position(2, 0), Position(2, 14))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        assert len(doc.leading_comments) == 2
        assert "This is a leading comment" in doc.leading_comments
        assert "Another leading comment" in doc.leading_comments

    def test_extract_type_information_with_hover(self):
        """Test type information extraction from hover data"""
        symbol_data = {"name": "test_func"}
        hover_data = {
            "contents": {
                "value": "def test_func(x: int, y: str) -> bool:\n    ..."
            }
        }
        type_info = self.extractor.extract_type_information(symbol_data, hover_data)
        assert type_info.type_signature == "def test_func(x: int, y: str) -> bool:\n    ..."
        assert type_info.return_type == "bool"
        assert len(type_info.parameter_types) == 2
        assert type_info.parameter_types[0]["name"] == "x"
        assert type_info.parameter_types[0]["type"] == "int"
        assert type_info.parameter_types[1]["name"] == "y"
        assert type_info.parameter_types[1]["type"] == "str"

    def test_extract_imports_exports(self):
        """Test extraction of Python imports and exports"""
        source_lines = [
            "import os",
            "from typing import Dict",
            "import json as js",
            "",
            "__all__ = ['function1', 'Class1']",
            "",
            "def function1():",
            "    pass"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "import os" in imports
        assert "from typing import Dict" in imports
        assert "import json as js" in imports
        assert "function1" in exports
        assert "Class1" in exports

    def test_get_minimal_context(self):
        """Test minimal context extraction around symbol"""
        source_lines = [
            "# File header",
            "import os",
            "class TestClass:",
            "    def method(self):",
            "        return True",
            "    def another_method(self):",
            "        pass"
        ]
        range_obj = Range(Position(3, 4), Position(3, 18))  # method definition
        context_before, context_after = self.extractor.get_minimal_context(source_lines, range_obj)
        assert len(context_before) >= 1
        assert "class TestClass:" in context_before[0]
        assert len(context_after) <= 1


class TestRustExtractor:
    """Test RustExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = RustExtractor()

    def test_extract_documentation_doc_comments(self):
        """Test extraction of Rust doc comments"""
        source_lines = [
            "/// This function does something important",
            "/// It takes a parameter and returns a result",
            "pub fn important_function(param: i32) -> String {",
            '    "result".to_string()',
            "}"
        ]
        range_obj = Range(Position(2, 0), Position(2, 32))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        expected = "This function does something important\nIt takes a parameter and returns a result"
        assert doc.docstring == expected

    def test_extract_documentation_multiline_doc_comment(self):
        """Test extraction of multi-line Rust doc comment"""
        source_lines = [
            "/**",
            " * Multi-line documentation",
            " * with detailed explanation",
            " */",
            "pub struct TestStruct {",
            "    field: i32,",
            "}"
        ]
        range_obj = Range(Position(4, 0), Position(4, 23))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        assert "Multi-line documentation" in doc.docstring
        assert "with detailed explanation" in doc.docstring

    def test_extract_type_information_with_return_type(self):
        """Test Rust type information extraction"""
        symbol_data = {"name": "rust_func"}
        hover_data = {
            "contents": {
                "value": "fn rust_func(x: i32) -> String { ... }"
            }
        }
        type_info = self.extractor.extract_type_information(symbol_data, hover_data)
        assert type_info.type_signature == "fn rust_func(x: i32) -> String { ... }"
        assert type_info.return_type == "String"

    def test_extract_imports_exports_rust(self):
        """Test extraction of Rust use statements and pub items"""
        source_lines = [
            "use std::collections::HashMap;",
            "use serde::{Serialize, Deserialize};",
            "",
            "pub struct PublicStruct {",
            "    pub field: String,",
            "}",
            "",
            "pub fn public_function() -> i32 {",
            "    42",
            "}"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "use std::collections::HashMap;" in imports
        assert "use serde::{Serialize, Deserialize};" in imports
        assert "pub struct PublicStruct {" in exports
        assert "pub fn public_function() -> i32 {" in exports

    def test_get_minimal_context_rust(self):
        """Test minimal context extraction for Rust"""
        source_lines = [
            "impl TestStruct {",
            "    /// Method documentation",
            "    pub fn method(&self) -> i32 {",
            "        self.field",
            "    }",
            "}"
        ]
        range_obj = Range(Position(2, 4), Position(2, 32))
        context_before, context_after = self.extractor.get_minimal_context(source_lines, range_obj)
        assert len(context_before) >= 1
        assert "impl TestStruct {" in context_before[0]


class TestJavaScriptExtractor:
    """Test JavaScriptExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = JavaScriptExtractor()

    def test_extract_documentation_jsdoc(self):
        """Test extraction of JSDoc comments"""
        source_lines = [
            "/**",
            " * This function processes data",
            " * @param {string} input - The input string",
            " * @param {number} count - Number of times to process",
            " * @returns {boolean} Success status",
            " */",
            "function processData(input, count) {",
            "    return true;",
            "}"
        ]
        range_obj = Range(Position(6, 0), Position(6, 35))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        assert "This function processes data" in doc.docstring
        assert "param" in doc.tags
        assert len(doc.tags["param"]) == 2
        assert "returns" in doc.tags

    def test_extract_type_information_typescript(self):
        """Test TypeScript type information extraction"""
        symbol_data = {"name": "tsFunction"}
        hover_data = {
            "contents": {
                "value": "function tsFunction(x: string): number"
            }
        }
        type_info = self.extractor.extract_type_information(symbol_data, hover_data)
        assert type_info.type_signature == "function tsFunction(x: string): number"
        assert type_info.type_name == "number"

    def test_extract_imports_exports_js(self):
        """Test JavaScript/ES6 import/export extraction"""
        source_lines = [
            "import { Component, useState } from 'react';",
            "import axios from 'axios';",
            "const fs = require('fs');",
            "",
            "export default function App() {",
            "    return null;",
            "}",
            "",
            "export const utils = {};",
            "module.exports = { helper };"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "import { Component, useState } from 'react';" in imports
        assert "import axios from 'axios';" in imports
        assert "const fs = require('fs');" in imports
        assert "export default function App() {" in exports
        assert "export const utils = {};" in exports
        assert "module.exports = { helper };" in exports


class TestJavaExtractor:
    """Test JavaExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = JavaExtractor()

    def test_extract_documentation_javadoc(self):
        """Test extraction of Javadoc comments"""
        source_lines = [
            "/**",
            " * Processes the input data",
            " * @param input the input string",
            " * @param count number of iterations",
            " * @return processing result",
            " */",
            "public boolean processData(String input, int count) {",
            "    return true;",
            "}"
        ]
        range_obj = Range(Position(6, 0), Position(6, 47))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        assert "Processes the input data" in doc.docstring
        assert "param" in doc.tags
        assert "return" in doc.tags

    def test_extract_imports_exports_java(self):
        """Test Java import and public declaration extraction"""
        source_lines = [
            "package com.example;",
            "",
            "import java.util.List;",
            "import java.util.ArrayList;",
            "",
            "public class TestClass {",
            "    private String field;",
            "    public void method() {}",
            "}"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "import java.util.List;" in imports
        assert "import java.util.ArrayList;" in imports
        assert "public class TestClass {" in exports
        assert "public void method() {}" in exports


class TestGoExtractor:
    """Test GoExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = GoExtractor()

    def test_extract_documentation_go_comments(self):
        """Test extraction of Go doc comments"""
        source_lines = [
            "// ProcessData processes the input data",
            "// and returns a boolean result.",
            "//",
            "// This function is exported.",
            "func ProcessData(input string, count int) bool {",
            "    return true",
            "}"
        ]
        range_obj = Range(Position(4, 0), Position(4, 45))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        expected = "ProcessData processes the input data\nand returns a boolean result.\n\nThis function is exported."
        assert doc.docstring == expected

    def test_extract_type_information_go_function(self):
        """Test Go function type information extraction"""
        symbol_data = {"name": "GoFunc"}
        hover_data = {
            "contents": {
                "value": "func GoFunc(x int, y string) error"
            }
        }
        type_info = self.extractor.extract_type_information(symbol_data, hover_data)
        assert type_info.type_signature == "func GoFunc(x int, y string) error"
        assert type_info.return_type == "error"

    def test_extract_imports_exports_go(self):
        """Test Go import and exported symbol extraction"""
        source_lines = [
            "package main",
            "",
            "import (",
            "    \"fmt\"",
            "    \"os\"",
            "    \"github.com/example/package\"",
            ")",
            "",
            "// ExportedFunction is public",
            "func ExportedFunction() {}",
            "",
            "func unexportedFunction() {}",
            "",
            "type ExportedStruct struct {",
            "    Field string",
            "}",
            "",
            "var ExportedVar = 42"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "\"fmt\"" in imports
        assert "\"os\"" in imports
        assert "\"github.com/example/package\"" in imports
        assert "func ExportedFunction() {}" in exports
        assert "type ExportedStruct struct {" in exports
        assert "var ExportedVar = 42" in exports

        # Unexported function should not be in exports
        unexported_found = any("unexportedFunction" in export for export in exports)
        assert not unexported_found


class TestCppExtractor:
    """Test CppExtractor language-specific functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.extractor = CppExtractor()

    def test_extract_documentation_cpp_doc_comments(self):
        """Test extraction of C++ doc comments"""
        source_lines = [
            "/// Brief description of the function",
            "/// Detailed description follows",
            "/// @param value Input parameter",
            "int process(int value) {",
            "    return value * 2;",
            "}"
        ]
        range_obj = Range(Position(3, 0), Position(3, 23))
        doc = self.extractor.extract_documentation(source_lines, range_obj)
        expected = "Brief description of the function\nDetailed description follows\n@param value Input parameter"
        assert doc.docstring == expected

    def test_extract_imports_exports_cpp(self):
        """Test C++ include and export extraction"""
        source_lines = [
            "#include <iostream>",
            "#include <vector>",
            "#include \"local_header.h\"",
            "",
            "class ExportedClass {",
            "public:",
            "    void method();",
            "};",
            "",
            "extern int global_variable;",
            "namespace ExportedNamespace {",
            "    void function();",
            "}"
        ]
        imports, exports = self.extractor.extract_imports_exports(source_lines)
        assert "#include <iostream>" in imports
        assert "#include <vector>" in imports
        assert "#include \"local_header.h\"" in imports
        assert "class ExportedClass {" in exports
        assert "extern int global_variable;" in exports
        assert "namespace ExportedNamespace {" in exports


class TestLspMetadataExtractor:
    """Test main LspMetadataExtractor class"""

    def setup_method(self):
        """Setup for each test method"""
        self.mock_filter = Mock(spec=LanguageAwareFilter)
        self.mock_filter._initialized = True
        self.mock_filter.should_process_file.return_value = (True, "allowed")

        self.extractor = LspMetadataExtractor(
            file_filter=self.mock_filter,
            request_timeout=10.0,
            max_concurrent_files=5,
            cache_size=100
        )

    def test_initialization(self):
        """Test LspMetadataExtractor initialization"""
        assert self.extractor.file_filter == self.mock_filter
        assert self.extractor.request_timeout == 10.0
        assert self.extractor.max_concurrent_files == 5
        assert not self.extractor._initialized
        assert len(self.extractor.language_extractors) > 0
        assert "python" in self.extractor.language_extractors
        assert "rust" in self.extractor.language_extractors

    def test_get_language_from_file(self):
        """Test language detection from file extensions"""
        assert self.extractor._get_language_from_file(Path("test.py")) == "python"
        assert self.extractor._get_language_from_file(Path("main.rs")) == "rust"
        assert self.extractor._get_language_from_file(Path("app.js")) == "javascript"
        assert self.extractor._get_language_from_file(Path("component.tsx")) == "typescript"
        assert self.extractor._get_language_from_file(Path("Main.java")) == "java"
        assert self.extractor._get_language_from_file(Path("main.go")) == "go"
        assert self.extractor._get_language_from_file(Path("source.cpp")) == "cpp"
        assert self.extractor._get_language_from_file(Path("header.h")) == "c"
        assert self.extractor._get_language_from_file(Path("unknown.xyz")) is None

    @pytest.mark.asyncio
    async def test_detect_languages(self):
        """Test language detection in workspace"""
        with patch('pathlib.Path.rglob') as mock_rglob:
            # Mock finding Python and Rust files
            mock_rglob.side_effect = lambda pattern: [
                Path("test.py")] if pattern == "*.py" else (
                [Path("main.rs")] if pattern == "*.rs" else []
            )

            workspace_path = Path("/test/workspace")
            languages = await self.extractor._detect_languages(workspace_path)
            assert "python" in languages
            assert "rust" in languages
            assert len(languages) == 2

    @pytest.mark.asyncio
    async def test_initialize_with_workspace(self):
        """Test extractor initialization with workspace"""
        with patch.object(self.extractor, '_detect_languages') as mock_detect, \
             patch.object(self.extractor, '_initialize_lsp_client') as mock_init_client:

            mock_detect.return_value = {"python", "rust"}
            mock_init_client.return_value = None

            workspace_root = Path("/test/workspace")
            await self.extractor.initialize(workspace_root)

            assert self.extractor._initialized
            mock_detect.assert_called_once_with(workspace_root)
            # Should try to initialize LSP clients for detected languages
            assert mock_init_client.call_count == 2

    @pytest.mark.asyncio
    async def test_initialize_lsp_client_success(self):
        """Test successful LSP client initialization"""
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.connect_stdio = AsyncMock()
        mock_client.initialize = AsyncMock()

        workspace_path = Path("/test/workspace")
        language = "python"

        with patch('src.python.common.core.lsp_metadata_extractor.AsyncioLspClient') as mock_client_class:
            mock_client_class.return_value = mock_client

            await self.extractor._initialize_lsp_client(language, workspace_path)

            assert language in self.extractor.lsp_clients
            assert self.extractor.lsp_clients[language] == mock_client
            mock_client.connect_stdio.assert_called_once()
            mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_lsp_client_failure(self):
        """Test LSP client initialization failure"""
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.connect_stdio = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.disconnect = AsyncMock()

        workspace_path = Path("/test/workspace")
        language = "python"

        with patch('src.python.common.core.lsp_metadata_extractor.AsyncioLspClient') as mock_client_class:
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception, match="Connection failed"):
                await self.extractor._initialize_lsp_client(language, workspace_path)

            assert language not in self.extractor.lsp_clients
            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_file_metadata_filtered_file(self):
        """Test file metadata extraction for filtered file"""
        self.mock_filter.should_process_file.return_value = (False, "excluded by filter")

        result = await self.extractor.extract_file_metadata("/test/file.py")
        assert result is None
        self.mock_filter.should_process_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_file_metadata_unknown_language(self):
        """Test file metadata extraction for unknown language"""
        result = await self.extractor.extract_file_metadata("/test/file.unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_file_metadata_no_lsp_client(self):
        """Test file metadata extraction without LSP client"""
        result = await self.extractor.extract_file_metadata("/test/file.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_file_metadata_success(self):
        """Test successful file metadata extraction"""
        # Setup mock LSP client
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.is_initialized = True
        mock_client.sync_file_opened = AsyncMock()
        mock_client.sync_file_closed = AsyncMock()
        mock_client.document_symbol = AsyncMock(return_value=[
            {
                "name": "test_function",
                "kind": 12,  # FUNCTION
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 2, "character": 0}
                },
                "selectionRange": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 17}
                }
            }
        ])
        mock_client.hover = AsyncMock(return_value={
            "contents": {"value": "def test_function() -> None"}
        })
        mock_client.references = AsyncMock(return_value=[])
        mock_client.definition = AsyncMock(return_value=[])

        self.extractor.lsp_clients["python"] = mock_client

        # Mock file content
        file_content = 'def test_function():\n    """Test function."""\n    pass\n'

        with patch('pathlib.Path.read_text', return_value=file_content), \
             patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):

            result = await self.extractor.extract_file_metadata("/test/file.py")

            assert result is not None
            assert result.language == "python"
            assert result.file_path == "/test/file.py"
            assert len(result.symbols) == 1
            assert result.symbols[0].name == "test_function"
            assert result.symbols[0].kind == SymbolKind.FUNCTION

            # Verify LSP calls were made
            mock_client.sync_file_opened.assert_called_once()
            mock_client.document_symbol.assert_called_once()
            mock_client.sync_file_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_file_metadata_read_error(self):
        """Test file metadata extraction with file read error"""
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.is_initialized = True
        self.extractor.lsp_clients["python"] = mock_client

        with patch('pathlib.Path.read_text', side_effect=OSError("Permission denied")), \
             patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):

            result = await self.extractor.extract_file_metadata("/test/file.py")
            assert result is None
            assert self.extractor.statistics.files_failed == 1

    @pytest.mark.asyncio
    async def test_extract_file_metadata_with_cache(self):
        """Test file metadata extraction with cache hit"""
        # Create cached metadata
        cached_metadata = FileMetadata(
            file_uri="file:///test/file.py",
            file_path="/test/file.py",
            language="python"
        )
        current_time = time.time()
        self.extractor.metadata_cache["file:///test/file.py"] = (cached_metadata, current_time)

        with patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):
            result = await self.extractor.extract_file_metadata("/test/file.py")

            assert result == cached_metadata
            assert self.extractor.statistics.cache_hits == 1

    @pytest.mark.asyncio
    async def test_extract_file_metadata_expired_cache(self):
        """Test file metadata extraction with expired cache"""
        # Create expired cached metadata
        cached_metadata = FileMetadata(
            file_uri="file:///test/file.py",
            file_path="/test/file.py",
            language="python"
        )
        expired_time = time.time() - 7200  # 2 hours ago
        self.extractor.metadata_cache["file:///test/file.py"] = (cached_metadata, expired_time)

        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.is_initialized = True
        mock_client.sync_file_opened = AsyncMock()
        mock_client.sync_file_closed = AsyncMock()
        mock_client.document_symbol = AsyncMock(return_value=[])
        self.extractor.lsp_clients["python"] = mock_client

        file_content = 'print("hello")\n'

        with patch('pathlib.Path.read_text', return_value=file_content), \
             patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):

            result = await self.extractor.extract_file_metadata("/test/file.py")

            # Should process file since cache expired
            assert result is not None
            assert self.extractor.statistics.cache_misses == 1
            # Cache should be updated
            assert "file:///test/file.py" in self.extractor.metadata_cache

    @pytest.mark.asyncio
    async def test_create_code_symbol_success(self):
        """Test successful code symbol creation"""
        symbol_data = {
            "name": "TestClass",
            "kind": 5,  # CLASS
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 10, "character": 0}
            },
            "selectionRange": {
                "start": {"line": 0, "character": 6},
                "end": {"line": 0, "character": 15}
            },
            "deprecated": True,
            "tags": ["test"]
        }

        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.hover = AsyncMock(return_value={
            "contents": {"value": "class TestClass:\n    pass"}
        })

        metadata = FileMetadata("file:///test.py", "/test.py", "python")
        extractor = self.extractor.language_extractors["python"]
        source_lines = ["class TestClass:", "    pass"]

        symbol = await self.extractor._create_code_symbol(
            symbol_data, "file:///test.py", source_lines, metadata, extractor, mock_client
        )

        assert symbol is not None
        assert symbol.name == "TestClass"
        assert symbol.kind == SymbolKind.CLASS
        assert symbol.deprecated is True
        assert symbol.tags == ["test"]
        assert symbol.documentation is not None
        assert symbol.type_info is not None

    @pytest.mark.asyncio
    async def test_create_code_symbol_invalid_kind(self):
        """Test code symbol creation with invalid kind"""
        symbol_data = {
            "name": "test",
            "kind": 999,  # Invalid kind
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 1, "character": 0}
            }
        }

        mock_client = AsyncMock(spec=AsyncioLspClient)
        metadata = FileMetadata("file:///test.py", "/test.py", "python")
        extractor = self.extractor.language_extractors["python"]
        source_lines = ["test = 42"]

        symbol = await self.extractor._create_code_symbol(
            symbol_data, "file:///test.py", source_lines, metadata, extractor, mock_client
        )

        assert symbol is not None
        assert symbol.kind == SymbolKind.VARIABLE  # Default fallback

    def test_parse_import_statement_python(self):
        """Test Python import statement parsing"""
        # Test 'from ... import ...' statement
        imports1 = self.extractor._parse_import_statement(
            "from typing import Dict", "python"
        )
        assert "List" in imports1
        assert "Dict" in imports1

        # Test 'import ...' statement
        imports2 = self.extractor._parse_import_statement(
            "import os.path", "python"
        )
        assert "os.path" in imports2

        # Test 'from ... import ... as ...' statement
        imports3 = self.extractor._parse_import_statement(
            "from json import loads as json_loads, dumps", "python"
        )
        assert "loads" in imports3  # 'as' part stripped
        assert "dumps" in imports3

    def test_parse_import_statement_javascript(self):
        """Test JavaScript import statement parsing"""
        imports = self.extractor._parse_import_statement(
            "import { Component, useState } from 'react'", "javascript"
        )
        assert "Component" in imports
        assert "useState" in imports

    def test_cache_metadata_size_limit(self):
        """Test metadata caching with size limit"""
        # Fill cache to near capacity
        for i in range(self.extractor.cache_size + 50):
            metadata = FileMetadata(f"file:///test{i}.py", f"/test{i}.py", "python")
            self.extractor._cache_metadata(f"file:///test{i}.py", metadata)

        # Cache should be limited to cache_size
        assert len(self.extractor.metadata_cache) <= self.extractor.cache_size

    def test_get_statistics(self):
        """Test getting extraction statistics"""
        self.extractor.statistics.files_processed = 10
        self.extractor.statistics.symbols_extracted = 50

        stats = self.extractor.get_statistics()
        assert stats.files_processed == 10
        assert stats.symbols_extracted == 50
        assert isinstance(stats, ExtractionStatistics)

    def test_reset_statistics(self):
        """Test resetting extraction statistics"""
        self.extractor.statistics.files_processed = 10
        self.extractor.statistics.files_failed = 2

        self.extractor.reset_statistics()

        assert self.extractor.statistics.files_processed == 0
        assert self.extractor.statistics.files_failed == 0

    def test_clear_cache(self):
        """Test clearing metadata cache"""
        # Add some cached data
        metadata = FileMetadata("file:///test.py", "/test.py", "python")
        self.extractor._cache_metadata("file:///test.py", metadata)

        assert len(self.extractor.metadata_cache) == 1

        self.extractor.clear_cache()
        assert len(self.extractor.metadata_cache) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test extractor shutdown"""
        # Add mock LSP clients
        mock_client1 = AsyncMock(spec=AsyncioLspClient)
        mock_client2 = AsyncMock(spec=AsyncioLspClient)
        mock_client1.disconnect = AsyncMock()
        mock_client2.disconnect = AsyncMock()

        self.extractor.lsp_clients["python"] = mock_client1
        self.extractor.lsp_clients["rust"] = mock_client2
        self.extractor._initialized = True

        # Add some cache data
        metadata = FileMetadata("file:///test.py", "/test.py", "python")
        self.extractor._cache_metadata("file:///test.py", metadata)

        await self.extractor.shutdown()

        # Verify cleanup
        assert len(self.extractor.lsp_clients) == 0
        assert len(self.extractor.metadata_cache) == 0
        assert not self.extractor._initialized
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        with patch.object(self.extractor, 'shutdown') as mock_shutdown:
            async with self.extractor as extractor:
                assert extractor is self.extractor

            mock_shutdown.assert_called_once()


class TestSymbolKindEnum:
    """Test SymbolKind enum"""

    def test_symbol_kind_values(self):
        """Test SymbolKind enum values"""
        assert SymbolKind.FILE.value == 1
        assert SymbolKind.FUNCTION.value == 12
        assert SymbolKind.CLASS.value == 5
        assert SymbolKind.VARIABLE.value == 13
        assert SymbolKind.IMPORT.value == 100  # Workspace extension
        assert SymbolKind.EXPORT.value == 101  # Workspace extension


class TestRelationshipTypeEnum:
    """Test RelationshipType enum"""

    def test_relationship_type_values(self):
        """Test RelationshipType enum values"""
        assert RelationshipType.IMPORTS.value == "imports"
        assert RelationshipType.CALLS.value == "calls"
        assert RelationshipType.EXTENDS.value == "extends"
        assert RelationshipType.REFERENCES.value == "references"


class TestLanguageSpecificExtractorABC:
    """Test LanguageSpecificExtractor abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated"""
        with pytest.raises(TypeError):
            LanguageSpecificExtractor()

    def test_abstract_methods_exist(self):
        """Test that abstract methods are properly defined"""
        abstract_methods = LanguageSpecificExtractor.__abstractmethods__
        expected_methods = {
            'extract_documentation',
            'extract_type_information',
            'extract_imports_exports',
            'get_minimal_context'
        }
        assert abstract_methods == expected_methods


# Integration tests for complex scenarios
class TestIntegrationScenarios:
    """Integration tests for complex extraction scenarios"""

    def setup_method(self):
        """Setup for integration tests"""
        self.mock_filter = Mock(spec=LanguageAwareFilter)
        self.mock_filter._initialized = True
        self.mock_filter.should_process_file.return_value = (True, "allowed")

        self.extractor = LspMetadataExtractor(
            file_filter=self.mock_filter,
            enable_relationship_mapping=True
        )

    @pytest.mark.asyncio
    async def test_file_extraction_error_handling(self):
        """Test file extraction with various error conditions"""
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.is_initialized = True
        mock_client.sync_file_opened = AsyncMock(side_effect=Exception("LSP sync failed"))
        mock_client.sync_file_closed = AsyncMock()
        mock_client.document_symbol = AsyncMock(return_value=[])

        self.extractor.lsp_clients["python"] = mock_client

        file_content = 'def test():\n    pass\n'

        with patch('pathlib.Path.read_text', return_value=file_content), \
             patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):

            result = await self.extractor.extract_file_metadata("/test/file.py")

            # Should still succeed despite sync error
            assert result is not None
            assert len(result.extraction_errors) > 0
            assert "LSP sync failed" in result.extraction_errors[0]

    @pytest.mark.asyncio
    async def test_symbol_processing_error_handling(self):
        """Test symbol processing with malformed LSP data"""
        mock_client = AsyncMock(spec=AsyncioLspClient)
        mock_client.is_initialized = True
        mock_client.sync_file_opened = AsyncMock()
        mock_client.sync_file_closed = AsyncMock()

        # Malformed symbol data missing required fields
        malformed_symbols = [
            {"name": "good_symbol", "kind": 12, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 0}}},
            {"name": "bad_symbol"},  # Missing kind and range
            {"kind": 5}  # Missing name
        ]
        mock_client.document_symbol = AsyncMock(return_value=malformed_symbols)

        self.extractor.lsp_clients["python"] = mock_client

        file_content = 'def good_symbol():\n    pass\n'

        with patch('pathlib.Path.read_text', return_value=file_content), \
             patch('pathlib.Path.resolve', return_value=Path("/test/file.py")):

            result = await self.extractor.extract_file_metadata("/test/file.py")

            assert result is not None
            # Should process valid symbols and handle errors for invalid ones
            assert len(result.symbols) >= 1  # At least the good symbol
            assert len(result.extraction_errors) > 0  # Errors for malformed symbols

    def test_all_language_extractors_completeness(self):
        """Test that all language extractors implement required methods"""
        extractors = [
            PythonExtractor(),
            RustExtractor(),
            JavaScriptExtractor(),
            JavaExtractor(),
            GoExtractor(),
            CppExtractor()
        ]

        required_methods = [
            'extract_documentation',
            'extract_type_information',
            'extract_imports_exports',
            'get_minimal_context'
        ]

        for extractor in extractors:
            for method_name in required_methods:
                assert hasattr(extractor, method_name)
                assert callable(getattr(extractor, method_name))
