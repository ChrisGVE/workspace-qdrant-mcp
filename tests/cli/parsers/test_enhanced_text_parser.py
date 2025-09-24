"""
Comprehensive test suite for the enhanced TextParser.

This module provides extensive unit tests covering all functionality of the enhanced
text document processor including edge cases for encoding detection, structured
format parsing, and error handling.
"""

import asyncio
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

# Import the enhanced TextParser
from src.python.wqm_cli.cli.parsers.text_parser import TextParser
from src.python.wqm_cli.cli.parsers.base import ParsedDocument
from src.python.wqm_cli.cli.parsers.exceptions import ParsingError


class TestEnhancedTextParser:
    """Test suite for enhanced TextParser functionality."""

    @pytest.fixture
    def text_parser(self):
        """Create TextParser instance for testing."""
        return TextParser()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # Core structured format tests

    def test_csv_parsing_with_headers(self, text_parser, temp_dir):
        """Test CSV parsing with header detection and type analysis."""
        csv_content = """Name,Age,Score,Date
John Doe,25,95.5,2024-01-01
Jane Smith,30,87.2,2024-01-02
Bob Johnson,22,92.1,2024-01-03"""

        test_file = temp_dir / "test.csv"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        result = asyncio.run(text_parser.parse(test_file))

        assert isinstance(result, ParsedDocument)
        assert result.content == csv_content
        assert result.metadata["content_type"] == "tabular_data"
        assert "headers" in result.parsing_info
        assert result.parsing_info["headers"] == ["Name", "Age", "Score", "Date"]
        assert result.parsing_info["row_count"] == 4
        assert result.parsing_info["column_count"] == 4

    def test_csv_delimiter_detection(self, text_parser):
        """Test automatic CSV delimiter detection."""
        content_comma = "a,b,c\n1,2,3"
        content_tab = "a\tb\tc\n1\t2\t3"

        # Test comma detection
        metadata = text_parser._parse_csv_metadata(content_comma)
        assert metadata["delimiter"] == ","
        assert metadata["delimiter_detected"] is True

        # Test tab detection
        metadata = text_parser._parse_csv_metadata(content_tab)
        assert metadata["delimiter"] == "\t"

    def test_log_file_parsing(self, text_parser, temp_dir):
        """Test structured log file parsing with pattern detection."""
        log_content = """2024-01-01 12:34:56 INFO Starting application
2024-01-01 12:34:57 DEBUG Initializing database connection
2024-01-01 12:34:58 ERROR Failed to connect to database: Connection timeout
2024-01-01 12:34:59 WARN Retrying connection in 5 seconds
2024-01-01 12:35:04 INFO Database connection established"""

        test_file = temp_dir / "app.log"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(log_content)

        result = asyncio.run(text_parser.parse(test_file))

        assert result.metadata["content_type"] == "log_file"
        assert result.parsing_info["format_type"] == "log"
        assert result.parsing_info["timestamp_detection_rate"] == 1.0  # 100% lines have timestamps
        assert "INFO" in result.parsing_info["detected_log_levels"]
        assert "ERROR" in result.parsing_info["detected_log_levels"]
        assert result.parsing_info["appears_structured"] is True

    def test_jsonl_parsing(self, text_parser, temp_dir):
        """Test JSON Lines format parsing."""
        jsonl_content = '''{"id": 1, "name": "John", "age": 25}
{"id": 2, "name": "Jane", "age": 30}
{"id": 3, "name": "Bob", "age": 22}'''

        test_file = temp_dir / "data.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(jsonl_content)

        result = asyncio.run(text_parser.parse(test_file))

        assert result.metadata["content_type"] == "structured_data"
        assert result.parsing_info["format_type"] == "jsonl"
        assert result.parsing_info["valid_json_rate"] == 1.0
        assert "id" in result.parsing_info["sample_keys"]
        assert "name" in result.parsing_info["sample_keys"]

    def test_ini_config_parsing(self, text_parser, temp_dir):
        """Test INI configuration file parsing."""
        ini_content = """[database]
host = localhost
port = 5432
user = admin

[logging]
level = INFO
file = app.log

# This is a comment
[cache]
enabled = true
ttl = 3600"""

        test_file = temp_dir / "config.ini"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(ini_content)

        result = asyncio.run(text_parser.parse(test_file))

        assert result.metadata["content_type"] == "configuration"
        assert result.parsing_info["format_type"] == "config"
        assert result.parsing_info["section_count"] == 3
        assert result.parsing_info["key_value_pairs"] == 7
        assert result.parsing_info["comment_lines"] == 1

    # Encoding tests

    def test_encoding_fallback_chain(self, text_parser, temp_dir):
        """Test robust encoding reading with fallback chain."""
        content = "Hello, world!"
        test_file = temp_dir / "test_fallback.txt"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content)

        result = asyncio.run(text_parser._read_with_encoding_robust(test_file, "invalid-encoding"))
        read_content, actual_encoding = result

        assert read_content == content
        assert actual_encoding == "utf-8"  # Should fall back to UTF-8

    # Content type classification tests

    def test_content_type_classification(self, text_parser):
        """Test content type classification for various file types."""
        test_cases = [
            (".py", "print('hello')", ("code", "python")),
            (".csv", "a,b,c\\n1,2,3", ("tabular_data", "csv")),
            (".log", "2024-01-01 INFO test", ("log_file", "application_log")),
            (".ini", "[section]\\nkey=value", ("configuration", "ini")),
            (".json", '{"key": "value"}', ("structured_data", "json")),
            (".txt", "plain text", ("plain_text", "generic")),
        ]

        for extension, content, expected in test_cases:
            result = text_parser._classify_content_type(extension, content)
            assert result == expected

    def test_csv_data_type_analysis(self, text_parser):
        """Test CSV column data type analysis."""
        test_rows = [
            ["123", "45.6", "2024-01-01", "hello"],
            ["456", "78.9", "2024-01-02", "world"],
            ["789", "12.3", "2024-01-03", "test"]
        ]

        result = text_parser._analyze_csv_data_types(test_rows)

        assert result == ["integer", "float", "date", "string"]

    def test_date_pattern_recognition(self, text_parser):
        """Test date pattern recognition in CSV data."""
        assert text_parser._is_date_like("2024-01-01") is True
        assert text_parser._is_date_like("01/01/2024") is True
        assert text_parser._is_date_like("01-01-2024") is True
        assert text_parser._is_date_like("not-a-date") is False
        assert text_parser._is_date_like("12345") is False

    def test_numeric_type_detection(self, text_parser):
        """Test integer and float detection."""
        assert text_parser._is_integer("123") is True
        assert text_parser._is_integer("123.45") is False
        assert text_parser._is_integer("abc") is False

        assert text_parser._is_float("123.45") is True
        assert text_parser._is_float("123") is True  # Integers are also valid floats
        assert text_parser._is_float("abc") is False

    # Edge case tests

    def test_empty_file_handling(self, text_parser, temp_dir):
        """Test handling of empty files."""
        test_file = temp_dir / "empty.txt"
        test_file.touch()

        result = asyncio.run(text_parser.parse(test_file))

        assert result.content == ""
        assert result.metadata["content_length"] == 0
        assert result.parsing_info["word_count"] == 0

    def test_corrupted_csv_handling(self, text_parser, temp_dir):
        """Test handling of malformed CSV data."""
        corrupted_csv = '''Name,Age,Score
"John Doe,25,95.5
Jane "Smith",30,"87.2
Bob,22,'''  # Unbalanced quotes and incomplete data

        test_file = temp_dir / "corrupted.csv"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(corrupted_csv)

        result = asyncio.run(text_parser.parse(test_file))

        # Should still parse but might have parsing errors noted
        assert result.content == corrupted_csv
        # Error information might be captured in parsing_info
        # This is acceptable - we want graceful degradation

    # Integration tests

    def test_end_to_end_csv_workflow(self, text_parser, temp_dir):
        """Test complete CSV parsing workflow."""
        csv_content = """Product,Price,Category,In_Stock,Rating
Laptop,999.99,Electronics,true,4.5
Book,19.99,Education,true,4.2
Chair,149.50,Furniture,false,3.8"""

        test_file = temp_dir / "products.csv"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        result = asyncio.run(text_parser.parse(test_file, enable_structured_parsing=True))

        # Verify all aspects of the parsing
        assert result.content == csv_content
        assert result.file_type == "text"
        assert result.metadata["content_type"] == "tabular_data"
        assert result.metadata["content_subtype"] == "csv"
        assert result.parsing_info["format_type"] == "csv"
        assert result.parsing_info["row_count"] == 4
        assert result.parsing_info["column_count"] == 5
        assert "Price" in result.parsing_info["headers"]
        assert len(result.parsing_info["sample_data_types"]) == 5

    def test_end_to_end_multilingual_text(self, text_parser, temp_dir):
        """Test parsing of multilingual text content."""
        multilingual_content = """English: Hello, world!
Spanish: ¡Hola, mundo!
French: Bonjour le monde!
German: Hallo, Welt!
Chinese: 你好，世界！
Japanese: こんにちは、世界！
Russian: Привет, мир!"""

        test_file = temp_dir / "multilingual.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(multilingual_content)

        result = asyncio.run(text_parser.parse(test_file))

        assert result.content == multilingual_content
        assert result.metadata["content_type"] == "plain_text"
        assert result.parsing_info["encoding"] == "utf-8"

    # Configuration and options tests

    def test_parsing_options_completeness(self, text_parser):
        """Test that all parsing options are properly defined."""
        options = text_parser.get_parsing_options()

        required_options = [
            "encoding", "detect_encoding", "clean_content", "preserve_whitespace",
            "enable_structured_parsing", "csv_delimiter", "csv_has_header", "max_file_size"
        ]

        for option in required_options:
            assert option in options
            assert "type" in options[option]
            assert "default" in options[option]
            assert "description" in options[option]

    def test_all_supported_extensions(self, text_parser):
        """Test that all advertised extensions are properly handled."""
        supported_extensions = text_parser.supported_extensions

        # Verify we have comprehensive coverage
        expected_extensions = [
            ".txt", ".text", ".log", ".csv", ".tsv", ".rtf", ".json", ".jsonl",
            ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".properties", ".env",
            ".py", ".js", ".html", ".css", ".sql", ".sh", ".bash", ".zsh", ".fish",
            ".ps1", ".bat", ".cmd"
        ]

        for ext in expected_extensions:
            assert ext in supported_extensions

    # Error handling tests

    def test_file_not_found_error(self, text_parser, temp_dir):
        """Test file not found error handling."""
        non_existent_file = temp_dir / "does_not_exist.txt"

        with pytest.raises(ParsingError):
            asyncio.run(text_parser.parse(non_existent_file))

    def test_invalid_csv_graceful_handling(self, text_parser):
        """Test graceful handling of invalid CSV content."""
        invalid_csv = "This is not CSV content at all!"

        metadata = text_parser._parse_csv_metadata(invalid_csv)

        # Should return metadata, potentially with error information
        assert "format_type" in metadata
        assert metadata["format_type"] == "csv"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])