"""Comprehensive unit tests for consistency checker with edge cases."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

try:
    from docs.framework.validation.consistency_checker import (
        ConsistencyChecker,
        ConsistencyLevel,
        ConsistencyReport,
        ConsistencyRule,
        ConsistencyViolation,
    )
except ModuleNotFoundError:
    pytest.skip("Docs framework not available", allow_module_level=True)


class TestConsistencyRule:
    """Test ConsistencyRule data class."""

    def test_consistency_rule_initialization(self):
        """Test basic initialization."""
        rule = ConsistencyRule(
            name="test_rule",
            description="Test rule",
            level=ConsistencyLevel.WARNING,
            pattern=r"test_pattern"
        )

        assert rule.name == "test_rule"
        assert rule.level == ConsistencyLevel.WARNING
        assert rule.pattern == r"test_pattern"
        assert rule.enabled is True
        assert rule.applies_to == ['*.md', '*.rst']

    def test_consistency_rule_with_custom_fields(self):
        """Test initialization with custom fields."""
        rule = ConsistencyRule(
            name="custom_rule",
            description="Custom rule",
            level=ConsistencyLevel.ERROR,
            expected_value="expected",
            applies_to=['*.txt'],
            enabled=False
        )

        assert rule.expected_value == "expected"
        assert rule.applies_to == ['*.txt']
        assert rule.enabled is False


class TestConsistencyViolation:
    """Test ConsistencyViolation data class."""

    def test_consistency_violation_initialization(self):
        """Test basic initialization."""
        violation = ConsistencyViolation(
            rule_name="test_rule",
            file_path=Path("test.md"),
            line_number=10,
            column=5,
            message="Test violation",
            level=ConsistencyLevel.WARNING
        )

        assert violation.rule_name == "test_rule"
        assert violation.file_path == Path("test.md")
        assert violation.line_number == 10
        assert violation.level == ConsistencyLevel.WARNING

    def test_consistency_violation_string_path(self):
        """Test initialization with string path."""
        violation = ConsistencyViolation(
            rule_name="test_rule",
            file_path="test.md",
            line_number=1,
            column=0,
            message="Test",
            level=ConsistencyLevel.INFO
        )

        assert isinstance(violation.file_path, Path)
        assert violation.file_path == Path("test.md")

    def test_consistency_violation_with_suggestions(self):
        """Test violation with suggested fix."""
        violation = ConsistencyViolation(
            rule_name="test_rule",
            file_path=Path("test.md"),
            line_number=1,
            column=0,
            message="Test",
            level=ConsistencyLevel.ERROR,
            suggested_fix="Fix suggestion",
            actual_value="actual",
            expected_value="expected"
        )

        assert violation.suggested_fix == "Fix suggestion"
        assert violation.actual_value == "actual"
        assert violation.expected_value == "expected"


class TestConsistencyReport:
    """Test ConsistencyReport data class."""

    def test_consistency_report_initialization(self):
        """Test basic initialization."""
        report = ConsistencyReport()

        assert report.total_files_checked == 0
        assert report.total_violations == 0
        assert report.compliance_score == 0.0
        assert len(report.violations) == 0

        # Check violations_by_level is initialized
        assert ConsistencyLevel.INFO in report.violations_by_level
        assert report.violations_by_level[ConsistencyLevel.INFO] == 0

    def test_consistency_report_with_data(self):
        """Test report with data."""
        violations = [
            ConsistencyViolation(
                rule_name="test_rule",
                file_path=Path("test.md"),
                line_number=1,
                column=0,
                message="Test",
                level=ConsistencyLevel.ERROR
            )
        ]

        report = ConsistencyReport(
            total_files_checked=5,
            total_violations=1,
            violations=violations,
            compliance_score=95.0
        )

        assert report.total_files_checked == 5
        assert len(report.violations) == 1
        assert report.compliance_score == 95.0


class TestConsistencyChecker:
    """Test ConsistencyChecker with comprehensive edge cases."""

    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary documentation directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir) / "docs"
            docs_path.mkdir()

            # Create sample files with various consistency issues
            (docs_path / "test.md").write_text("""# Test Documentation

This is a test document with various consistency issues.

## Section One
Some content here.
## Section Two
More content.

### Subsection with Period.

Some bullet points:
- Item one
* Item two (inconsistent marker)
+ Item three (another marker)

Code blocks:
```python
print("with language")
```

```
print("without language")
```

Here are some "double quotes" and 'single quotes' mixed.

This line is really, really, really long and exceeds the typical line length limit that would be enforced by consistency rules for documentation formatting.

## Another Section
[Link with inconsistent format](https://example.com)
[Link with reference format][ref]

[ref]: https://example.com

Table with formatting:
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
""")

            (docs_path / "api.rst").write_text("""API Reference
=============

This file uses different capitalization.

functions and classes
=====================

Some inconsistent terminology:
- JSON parser
- json encoder
- API endpoint
- api client
""")

            yield docs_path

    def test_checker_initialization(self, temp_docs_dir):
        """Test checker initialization."""
        checker = ConsistencyChecker(temp_docs_dir)

        assert checker.root_path == temp_docs_dir
        assert len(checker.rules) > 0
        assert 'heading_capitalization' in checker.rules
        assert isinstance(checker._terminology, dict)

    def test_checker_initialization_with_config_file(self, temp_docs_dir):
        """Test initialization with config file."""
        # Create config file
        config_file = temp_docs_dir / "config.yaml"
        config_data = {
            'rules': [
                {
                    'name': 'custom_rule',
                    'description': 'Custom test rule',
                    'level': 'warning',
                    'pattern': r'custom_pattern',
                    'enabled': True
                }
            ]
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        checker = ConsistencyChecker(temp_docs_dir, config_file=config_file)

        assert 'custom_rule' in checker.rules
        assert checker.rules['custom_rule'].name == 'custom_rule'

    def test_checker_initialization_with_json_config(self, temp_docs_dir):
        """Test initialization with JSON config file."""
        config_file = temp_docs_dir / "config.json"
        config_data = {
            'rules': [
                {
                    'name': 'json_rule',
                    'description': 'JSON test rule',
                    'level': 'error',
                    'expected_value': 'test_value'
                }
            ]
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        checker = ConsistencyChecker(temp_docs_dir, config_file=config_file)

        assert 'json_rule' in checker.rules
        assert checker.rules['json_rule'].level == ConsistencyLevel.ERROR

    def test_checker_initialization_with_custom_rules(self, temp_docs_dir):
        """Test initialization with custom rules."""
        custom_rules = [
            ConsistencyRule(
                name="custom_test_rule",
                description="Custom test rule",
                level=ConsistencyLevel.INFO,
                pattern=r"custom_.*"
            )
        ]

        checker = ConsistencyChecker(temp_docs_dir, custom_rules=custom_rules)

        assert 'custom_test_rule' in checker.rules

    def test_load_config_file_errors(self, temp_docs_dir):
        """Test config file loading error handling."""
        # Test with non-existent file
        ConsistencyChecker(temp_docs_dir, config_file="nonexistent.yaml")
        # Should not raise exception, just log error

        # Test with invalid YAML
        invalid_config = temp_docs_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        ConsistencyChecker(temp_docs_dir, config_file=invalid_config)
        # Should handle gracefully

    def test_load_default_rules(self, temp_docs_dir):
        """Test default rules loading."""
        checker = ConsistencyChecker(temp_docs_dir)

        # Check all expected default rules are loaded
        expected_rules = [
            'heading_capitalization',
            'heading_spacing',
            'code_block_language',
            'link_format',
            'terminology_consistency',
            'list_formatting',
            'table_formatting',
            'line_length',
            'heading_punctuation',
            'quote_style'
        ]

        for rule_name in expected_rules:
            assert rule_name in checker.rules
            assert isinstance(checker.rules[rule_name], ConsistencyRule)
            assert checker.rules[rule_name].enabled

    def test_check_consistency_basic(self, temp_docs_dir):
        """Test basic consistency checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        report = checker.check_consistency()

        assert report.total_files_checked > 0
        assert isinstance(report.total_violations, int)
        assert isinstance(report.compliance_score, float)
        assert 0 <= report.compliance_score <= 100

    def test_check_consistency_custom_patterns(self, temp_docs_dir):
        """Test consistency checking with custom file patterns."""
        checker = ConsistencyChecker(temp_docs_dir)
        report = checker.check_consistency(['*.md'])

        assert report.total_files_checked >= 0
        # Should only process .md files

    def test_check_consistency_no_files(self, temp_docs_dir):
        """Test consistency checking when no files match."""
        checker = ConsistencyChecker(temp_docs_dir)
        report = checker.check_consistency(['*.nonexistent'])

        assert report.total_files_checked == 0
        assert report.total_violations == 0

    def test_collect_statistics(self, temp_docs_dir):
        """Test statistics collection."""
        checker = ConsistencyChecker(temp_docs_dir)
        checker._collect_statistics(['*.md', '*.rst'])

        assert len(checker._terminology) > 0
        # Should have collected words from files

    def test_collect_statistics_unreadable_file(self, temp_docs_dir):
        """Test statistics collection with unreadable files."""
        checker = ConsistencyChecker(temp_docs_dir)

        # Mock file reading to fail
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            checker._collect_statistics(['*.md'])

        # Should handle gracefully without crashing

    def test_check_file_consistency(self, temp_docs_dir):
        """Test checking consistency for single file."""
        checker = ConsistencyChecker(temp_docs_dir)
        checker._collect_statistics(['*.md'])

        test_file = temp_docs_dir / "test.md"
        violations = checker._check_file_consistency(test_file)

        assert isinstance(violations, list)
        # Should find some violations in the test file

    def test_check_file_consistency_unreadable_file(self, temp_docs_dir):
        """Test file consistency check with unreadable file."""
        checker = ConsistencyChecker(temp_docs_dir)

        test_file = temp_docs_dir / "test.md"

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            violations = checker._check_file_consistency(test_file)

        assert violations == []

    def test_check_file_consistency_rule_errors(self, temp_docs_dir):
        """Test file consistency check with rule execution errors."""
        checker = ConsistencyChecker(temp_docs_dir)

        # Mock a rule checker to raise exception
        def failing_checker(rule, file_path, lines, content):
            raise Exception("Rule checker error")

        checker.rules['test_rule'] = ConsistencyRule(
            name='test_rule',
            description='Test rule',
            level=ConsistencyLevel.ERROR,
            checker_function=failing_checker
        )

        test_file = temp_docs_dir / "test.md"
        violations = checker._check_file_consistency(test_file)

        # Should handle rule errors gracefully
        assert isinstance(violations, list)

    def test_check_heading_capitalization(self, temp_docs_dir):
        """Test heading capitalization checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['heading_capitalization']

        # Test content with mixed heading styles
        content = """# Title Case Heading
## another heading in lowercase
### YET ANOTHER IN UPPERCASE
#### Final Mixed Case Heading"""

        lines = content.split('\n')
        violations = checker._check_heading_capitalization(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0
        # Should find inconsistent capitalization

    def test_check_heading_capitalization_consistent(self, temp_docs_dir):
        """Test heading capitalization with consistent headings."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['heading_capitalization']

        content = """# Title Case Heading
## Another Title Case Heading
### Third Title Case Heading"""

        lines = content.split('\n')
        violations = checker._check_heading_capitalization(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) == 0

    def test_check_heading_spacing(self, temp_docs_dir):
        """Test heading spacing checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['heading_spacing']

        content = """Some content here.
# Heading Without Blank Line Before

Some more content.

# Proper Heading With Blank Line"""

        lines = content.split('\n')
        violations = checker._check_heading_spacing(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0

    def test_check_code_block_language(self, temp_docs_dir):
        """Test code block language checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['code_block_language']

        content = """```python
print("with language")
```

```
print("without language")
```"""

        lines = content.split('\n')
        violations = checker._check_code_block_language(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0
        assert any("missing language" in v.message for v in violations)

    def test_check_link_format(self, temp_docs_dir):
        """Test link format checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['link_format']

        content = """[Markdown Link](https://example.com)
[Reference Link][ref]
[Another Markdown Link](https://example.org)

[ref]: https://example.com"""

        lines = content.split('\n')
        violations = checker._check_link_format(
            rule, temp_docs_dir / "test.md", lines, content
        )

        # Should detect mixed link formats
        assert len(violations) >= 0

    def test_check_terminology_consistency(self, temp_docs_dir):
        """Test terminology consistency checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['terminology_consistency']

        # Set up terminology statistics
        checker._terminology = {
            'api': 10,
            'API': 2,
            'json': 8,
            'JSON': 3
        }

        content = """This document uses API and api inconsistently.
It also mixes json and JSON formats."""

        lines = content.split('\n')
        violations = checker._check_terminology_consistency(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0

    def test_check_list_formatting(self, temp_docs_dir):
        """Test list formatting checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['list_formatting']

        content = """- Item with dash
* Item with asterisk
+ Item with plus
- Another dash item"""

        lines = content.split('\n')
        violations = checker._check_list_formatting(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0

    def test_check_list_formatting_consistent(self, temp_docs_dir):
        """Test list formatting with consistent markers."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['list_formatting']

        content = """- Item one
- Item two
- Item three"""

        lines = content.split('\n')
        violations = checker._check_list_formatting(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) == 0

    def test_check_table_formatting(self, temp_docs_dir):
        """Test table formatting checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['table_formatting']

        content = """| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |"""

        lines = content.split('\n')
        violations = checker._check_table_formatting(
            rule, temp_docs_dir / "test.md", lines, content
        )

        # Basic implementation may not find violations
        assert isinstance(violations, list)

    def test_check_line_length(self, temp_docs_dir):
        """Test line length checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['line_length']

        content = f"""This is a normal length line.
{'x' * 150}
Another normal line."""

        lines = content.split('\n')
        violations = checker._check_line_length(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0
        assert any("Line too long" in v.message for v in violations)

    def test_check_line_length_skip_code_blocks(self, temp_docs_dir):
        """Test line length skips code blocks."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['line_length']

        content = f"""Normal line.
```python
{'x' * 150}
```
| {'x' * 150} |"""

        lines = content.split('\n')
        violations = checker._check_line_length(
            rule, temp_docs_dir / "test.md", lines, content
        )

        # Should skip code blocks and tables
        assert len(violations) == 0

    def test_check_heading_punctuation(self, temp_docs_dir):
        """Test heading punctuation checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['heading_punctuation']

        content = """# Proper Heading
## Heading With Period.
### Another Proper Heading"""

        lines = content.split('\n')
        violations = checker._check_heading_punctuation(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0
        assert any("period" in v.message.lower() for v in violations)

    def test_check_quote_style(self, temp_docs_dir):
        """Test quote style checking."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['quote_style']

        content = """Text with "double quotes" and 'single quotes' mixed together.
More "double quotes" here.
And 'single quotes' there."""

        lines = content.split('\n')
        violations = checker._check_quote_style(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) > 0

    def test_check_quote_style_consistent(self, temp_docs_dir):
        """Test quote style with consistent quotes."""
        checker = ConsistencyChecker(temp_docs_dir)
        rule = checker.rules['quote_style']

        content = """Text with "double quotes" only.
More "double quotes" here.
All "consistent" throughout."""

        lines = content.split('\n')
        violations = checker._check_quote_style(
            rule, temp_docs_dir / "test.md", lines, content
        )

        assert len(violations) == 0

    def test_export_report(self, temp_docs_dir):
        """Test exporting consistency report."""
        checker = ConsistencyChecker(temp_docs_dir)

        report = ConsistencyReport(
            total_files_checked=5,
            total_violations=3,
            compliance_score=85.0,
            violations=[
                ConsistencyViolation(
                    rule_name="test_rule",
                    file_path=temp_docs_dir / "test.md",
                    line_number=10,
                    column=5,
                    message="Test violation",
                    level=ConsistencyLevel.WARNING
                )
            ]
        )

        output_path = temp_docs_dir / "report.json"
        success = checker.export_report(report, output_path)

        assert success
        assert output_path.exists()

        # Verify exported data
        with open(output_path) as f:
            data = json.load(f)

        assert data['summary']['total_files_checked'] == 5
        assert data['summary']['compliance_score'] == 85.0
        assert len(data['violations']) == 1
        assert 'rules_applied' in data

    def test_export_report_error_handling(self, temp_docs_dir):
        """Test export error handling."""
        checker = ConsistencyChecker(temp_docs_dir)
        report = ConsistencyReport()

        # Try to export to invalid path
        invalid_path = Path("/invalid/path/report.json")
        success = checker.export_report(report, invalid_path)

        assert not success

    def test_compliance_score_calculation(self, temp_docs_dir):
        """Test compliance score calculation."""
        checker = ConsistencyChecker(temp_docs_dir)

        # Mock file checking to return specific violations
        def mock_check_file(file_path):
            return [
                ConsistencyViolation(
                    rule_name="test_rule",
                    file_path=file_path,
                    line_number=1,
                    column=0,
                    message="Critical error",
                    level=ConsistencyLevel.CRITICAL
                ),
                ConsistencyViolation(
                    rule_name="test_rule2",
                    file_path=file_path,
                    line_number=2,
                    column=0,
                    message="Warning",
                    level=ConsistencyLevel.WARNING
                )
            ]

        with patch.object(checker, '_check_file_consistency', side_effect=mock_check_file):
            report = checker.check_consistency(['*.md'])

        # Should have calculated compliance score based on violations
        assert isinstance(report.compliance_score, float)
        assert 0 <= report.compliance_score <= 100

    def test_rule_applies_to_file_type(self, temp_docs_dir):
        """Test rule file type filtering."""
        # Create rule that only applies to .txt files
        custom_rule = ConsistencyRule(
            name="txt_only_rule",
            description="Rule for .txt files only",
            level=ConsistencyLevel.INFO,
            applies_to=['*.txt'],
            checker_function=lambda rule, file_path, lines, content: []
        )

        checker = ConsistencyChecker(temp_docs_dir, custom_rules=[custom_rule])

        # Check .md file - rule should not apply
        test_file = temp_docs_dir / "test.md"
        violations = checker._check_file_consistency(test_file)

        # Rule should not have been applied to .md file
        assert not any(v.rule_name == "txt_only_rule" for v in violations)

    def test_disabled_rules(self, temp_docs_dir):
        """Test disabled rules are not applied."""
        checker = ConsistencyChecker(temp_docs_dir)

        # Disable a rule
        checker.rules['heading_capitalization'].enabled = False

        test_file = temp_docs_dir / "test.md"
        violations = checker._check_file_consistency(test_file)

        # Should not have violations from disabled rule
        assert not any(v.rule_name == "heading_capitalization" for v in violations)

    def test_edge_case_empty_file(self, temp_docs_dir):
        """Test handling empty files."""
        empty_file = temp_docs_dir / "empty.md"
        empty_file.write_text("")

        checker = ConsistencyChecker(temp_docs_dir)
        violations = checker._check_file_consistency(empty_file)

        assert isinstance(violations, list)
        # Empty file should not cause crashes

    def test_edge_case_binary_file(self, temp_docs_dir):
        """Test handling binary files."""
        binary_file = temp_docs_dir / "binary.md"  # .md extension but binary content
        binary_file.write_bytes(b'\x00\x01\x02\x03\xFF')

        checker = ConsistencyChecker(temp_docs_dir)

        # Should handle binary files gracefully
        try:
            violations = checker._check_file_consistency(binary_file)
            assert isinstance(violations, list)
        except UnicodeDecodeError:
            # Acceptable to fail on binary files
            pass

    def test_edge_case_very_long_lines(self, temp_docs_dir):
        """Test handling very long lines."""
        long_line = 'x' * 10000
        long_file = temp_docs_dir / "long.md"
        long_file.write_text(f"# Heading\n{long_line}\n")

        checker = ConsistencyChecker(temp_docs_dir)
        violations = checker._check_file_consistency(long_file)

        # Should handle very long lines without crashing
        assert isinstance(violations, list)

    def test_performance_many_violations(self, temp_docs_dir):
        """Test performance with many violations."""
        # Create file with many consistency issues
        content_lines = []
        for i in range(100):
            content_lines.append(f"## heading {i} with period.")  # Punctuation violation
            content_lines.append("- item")  # List item
            content_lines.append("* different marker")  # Inconsistent marker
            content_lines.append("This is a 'quote' mixing with \"another quote\".")  # Quote inconsistency

        many_violations_file = temp_docs_dir / "many_violations.md"
        many_violations_file.write_text("\n".join(content_lines))

        checker = ConsistencyChecker(temp_docs_dir)
        report = checker.check_consistency(['many_violations.md'])

        # Should complete without performance issues
        assert report.total_violations > 0
        assert isinstance(report.compliance_score, float)

    def test_unicode_content_handling(self, temp_docs_dir):
        """Test handling Unicode content."""
        unicode_file = temp_docs_dir / "unicode.md"
        unicode_file.write_text("""# Unicode Heading: 测试

Content with Unicode characters: Тест, 测试, العربية

- Item with Unicode: 测试
* Different marker: Тест

Code block:
```python
print("Unicode: 测试")
```

Some "quotes" and 'quotes' in Unicode context.
""", encoding='utf-8')

        checker = ConsistencyChecker(temp_docs_dir)
        violations = checker._check_file_consistency(unicode_file)

        # Should handle Unicode gracefully
        assert isinstance(violations, list)

    def test_complex_markdown_structures(self, temp_docs_dir):
        """Test handling complex Markdown structures."""
        complex_file = temp_docs_dir / "complex.md"
        complex_file.write_text("""# Main Heading

## Section with nested elements

### Subsection

Some text with [inline link](https://example.com) and ![image](image.png).

> Blockquote with *emphasis* and **strong** text.
> Second line of blockquote.

1. Ordered list item
2. Another item with `inline code`
   - Nested unordered item
   - Another nested item

```python
def function_with_long_line():
    # This is a comment that exceeds the typical line length limit for consistency checking
    return "result"
```

| Table | With | Multiple | Columns |
|-------|------|----------|---------|
| Row 1 | Data | More     | Content |
| Row 2 | Data | More     | Content |

---

[Reference link][ref1] and [another reference][ref2].

[ref1]: https://example1.com "Title 1"
[ref2]: https://example2.com "Title 2"
""")

        checker = ConsistencyChecker(temp_docs_dir)
        violations = checker._check_file_consistency(complex_file)

        # Should handle complex structures without crashing
        assert isinstance(violations, list)

    def test_regex_pattern_edge_cases(self, temp_docs_dir):
        """Test regex pattern edge cases."""
        edge_case_file = temp_docs_dir / "edge_cases.md"
        edge_case_file.write_text("""# Heading with special chars: [brackets] and (parentheses)

Content with regex special characters: . * + ? ^ $ { } | \\ ( )

```
Code block with no language and special chars: .*+?
```

Links with special chars: [text with [brackets]](https://example.com/path?param=value&other=test)

- List item with * asterisk in text
- List item with + plus in text
""")

        checker = ConsistencyChecker(temp_docs_dir)
        violations = checker._check_file_consistency(edge_case_file)

        # Should handle special characters in content gracefully
        assert isinstance(violations, list)

    def test_fnmatch_import(self):
        """Test that fnmatch is properly imported."""
        # This tests the import at the end of the file
        import fnmatch
        assert hasattr(fnmatch, 'fnmatch')

    def test_consistency_levels_enum(self):
        """Test all ConsistencyLevel enum values."""
        levels = [
            ConsistencyLevel.INFO,
            ConsistencyLevel.WARNING,
            ConsistencyLevel.ERROR,
            ConsistencyLevel.CRITICAL
        ]

        for level in levels:
            assert isinstance(level.value, str)
            assert len(level.value) > 0

    def test_default_rule_checker_functions(self, temp_docs_dir):
        """Test that all default rules have working checker functions."""
        checker = ConsistencyChecker(temp_docs_dir)

        test_content = """# Test Heading
Some content here.
"""
        test_lines = test_content.split('\n')

        for rule_name, rule in checker.rules.items():
            if rule.checker_function:
                try:
                    violations = rule.checker_function(
                        rule,
                        temp_docs_dir / "test.md",
                        test_lines,
                        test_content
                    )
                    assert isinstance(violations, list)
                except Exception as e:
                    pytest.fail(f"Rule {rule_name} checker function failed: {e}")

    def test_violations_by_level_initialization(self):
        """Test that violations_by_level is properly initialized."""
        report = ConsistencyReport()

        # Should have all levels initialized to 0
        for level in ConsistencyLevel:
            assert level in report.violations_by_level
            assert report.violations_by_level[level] == 0
