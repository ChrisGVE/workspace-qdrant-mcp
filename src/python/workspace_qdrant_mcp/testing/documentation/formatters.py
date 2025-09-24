"""
Test Documentation Formatters

Provides multiple output formats for test documentation including
Markdown, HTML, and JSON with customizable templates and styling.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from .parser import TestFileInfo, TestMetadata, TestType

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """Base class for test documentation formatters."""

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize formatter with optional template directory.

        Args:
            template_dir: Directory containing custom templates
        """
        self.template_dir = template_dir
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Setup Jinja2 environment with templates."""
        if self.template_dir and self.template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            # Use string templates
            self.env = Environment(autoescape=True)

    @abstractmethod
    def format_test_file(self, file_info: TestFileInfo) -> str:
        """Format a single test file's documentation."""
        pass

    @abstractmethod
    def format_test_suite(self, files: List[TestFileInfo],
                         title: str = "Test Suite Documentation") -> str:
        """Format documentation for multiple test files."""
        pass

    def _safe_format(self, template_str: str, context: Dict[str, Any]) -> str:
        """Safely format template with error handling."""
        try:
            template = self.env.from_string(template_str)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template formatting error: {e}")
            return f"<!-- Template Error: {e} -->"


class MarkdownFormatter(BaseFormatter):
    """Markdown formatter for test documentation."""

    def __init__(self, template_dir: Optional[Path] = None):
        super().__init__(template_dir)
        self.file_template = self._get_file_template()
        self.suite_template = self._get_suite_template()

    def format_test_file(self, file_info: TestFileInfo) -> str:
        """Format a single test file as Markdown."""
        context = self._build_file_context(file_info)
        return self._safe_format(self.file_template, context)

    def format_test_suite(self, files: List[TestFileInfo],
                         title: str = "Test Suite Documentation") -> str:
        """Format multiple test files as a comprehensive Markdown document."""
        context = self._build_suite_context(files, title)
        return self._safe_format(self.suite_template, context)

    def _build_file_context(self, file_info: TestFileInfo) -> Dict[str, Any]:
        """Build context for single file template."""
        # Group tests by type
        tests_by_type = {}
        for test in file_info.tests:
            test_type = test.test_type.value
            if test_type not in tests_by_type:
                tests_by_type[test_type] = []
            tests_by_type[test_type].append(test)

        # Calculate statistics
        total_tests = len(file_info.tests)
        async_tests = sum(1 for test in file_info.tests if test.is_async)
        parametrized_tests = sum(1 for test in file_info.tests if test.is_parametrized)
        avg_complexity = (
            sum(test.complexity_score for test in file_info.tests) / total_tests
            if total_tests > 0 else 0
        )

        return {
            'file_info': file_info,
            'tests_by_type': tests_by_type,
            'stats': {
                'total_tests': total_tests,
                'async_tests': async_tests,
                'parametrized_tests': parametrized_tests,
                'avg_complexity': round(avg_complexity, 1),
                'total_lines': file_info.total_lines,
                'test_coverage': file_info.test_coverage or 0,
                'parse_errors': len(file_info.parse_errors)
            },
            'generated_at': datetime.now().isoformat(),
            'has_errors': bool(file_info.parse_errors)
        }

    def _build_suite_context(self, files: List[TestFileInfo], title: str) -> Dict[str, Any]:
        """Build context for test suite template."""
        # Calculate suite-wide statistics
        total_tests = sum(len(f.tests) for f in files)
        total_files = len(files)
        files_with_errors = sum(1 for f in files if f.parse_errors)

        # Group by test type across all files
        suite_tests_by_type = {}
        all_marks = set()

        for file_info in files:
            for test in file_info.tests:
                test_type = test.test_type.value
                if test_type not in suite_tests_by_type:
                    suite_tests_by_type[test_type] = 0
                suite_tests_by_type[test_type] += 1
                all_marks.update(test.marks)

        # Calculate average complexity across all tests
        all_tests = [test for file_info in files for test in file_info.tests]
        avg_complexity = (
            sum(test.complexity_score for test in all_tests) / len(all_tests)
            if all_tests else 0
        )

        return {
            'title': title,
            'files': files,
            'suite_stats': {
                'total_files': total_files,
                'total_tests': total_tests,
                'files_with_errors': files_with_errors,
                'avg_complexity': round(avg_complexity, 1),
                'test_types': suite_tests_by_type,
                'all_marks': sorted(all_marks)
            },
            'generated_at': datetime.now().isoformat()
        }

    def _get_file_template(self) -> str:
        """Get template for single file documentation."""
        return '''# Test File: {{ file_info.file_path.name }}

**Path:** `{{ file_info.file_path }}`
**Encoding:** {{ file_info.encoding }}
**Generated:** {{ generated_at }}

## Summary

- **Total Tests:** {{ stats.total_tests }}
- **Async Tests:** {{ stats.async_tests }}
- **Parametrized Tests:** {{ stats.parametrized_tests }}
- **Average Complexity:** {{ stats.avg_complexity }}
- **Total Lines:** {{ stats.total_lines }}
{% if stats.test_coverage %}
- **Test Coverage:** {{ "%.1f"|format(stats.test_coverage) }}%
{% endif %}
{% if has_errors %}
- **Parse Errors:** {{ stats.parse_errors }}
{% endif %}

{% if file_info.parse_errors %}
## ⚠️ Parse Errors

{% for error in file_info.parse_errors %}
- {{ error }}
{% endfor %}

{% endif %}

{% if file_info.imports %}
## Imports

```python
{% for import in file_info.imports[:10] %}
{{ import }}
{% endfor %}
{% if file_info.imports|length > 10 %}
... and {{ file_info.imports|length - 10 }} more
{% endif %}
```

{% endif %}

{% for test_type, tests in tests_by_type.items() %}
## {{ test_type.title() }} Tests ({{ tests|length }})

{% for test in tests %}
### `{{ test.name }}`

**Line:** {{ test.line_number }}{% if test.is_async %} | **Async**{% endif %}{% if test.is_parametrized %} | **Parametrized**{% endif %}
**Complexity:** {{ test.complexity_score }}/10

{% if test.docstring %}
{{ test.docstring }}
{% else %}
*No documentation provided*
{% endif %}

{% if test.decorators %}
**Decorators:**
{% for decorator in test.decorators %}
- `{{ decorator.name }}`{% if decorator.args or decorator.kwargs %}({% if decorator.args %}{{ decorator.args|join(', ') }}{% endif %}{% if decorator.kwargs %}{% for k, v in decorator.kwargs.items() %}{{ k }}={{ v }}{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}){% endif %}
{% endfor %}
{% endif %}

{% if test.parameters %}
**Parameters:**
{% for param in test.parameters %}
- `{{ param.name }}`{% if param.type_annotation %}: {{ param.type_annotation }}{% endif %}{% if param.default_value %} = {{ param.default_value }}{% endif %}
{% endfor %}
{% endif %}

{% if test.marks %}
**Marks:** {{ test.marks|join(', ') }}
{% endif %}

{% if test.expected_to_fail %}
⚠️ **Expected to fail**
{% endif %}

{% if test.skip_reason %}
⏭️ **Skipped:** {{ test.skip_reason }}
{% endif %}

---

{% endfor %}
{% endfor %}

{% if file_info.fixtures %}
## Fixtures

{% for fixture in file_info.fixtures %}
- `{{ fixture }}`
{% endfor %}
{% endif %}

{% if file_info.classes %}
## Test Classes

{% for class_name in file_info.classes %}
- `{{ class_name }}`
{% endfor %}
{% endif %}
'''

    def _get_suite_template(self) -> str:
        """Get template for test suite documentation."""
        return '''# {{ title }}

**Generated:** {{ generated_at }}

## Suite Overview

- **Total Files:** {{ suite_stats.total_files }}
- **Total Tests:** {{ suite_stats.total_tests }}
- **Files with Errors:** {{ suite_stats.files_with_errors }}
- **Average Complexity:** {{ suite_stats.avg_complexity }}

### Test Distribution by Type

{% for test_type, count in suite_stats.test_types.items() %}
- **{{ test_type.title() }}:** {{ count }} tests
{% endfor %}

{% if suite_stats.all_marks %}
### Available Test Marks

{{ suite_stats.all_marks|join(', ') }}
{% endif %}

---

## Test Files

{% for file_info in files %}
### 📁 {{ file_info.file_path.name }}

**Path:** `{{ file_info.file_path }}`
**Tests:** {{ file_info.tests|length }}{% if file_info.parse_errors %} | ⚠️ **{{ file_info.parse_errors|length }} errors**{% endif %}

{% if file_info.tests %}
{% for test in file_info.tests[:5] %}
- `{{ test.name }}` ({{ test.test_type.value }}){% if test.is_async %} [async]{% endif %}{% if test.is_parametrized %} [parametrized]{% endif %}
{% endfor %}
{% if file_info.tests|length > 5 %}
- ... and {{ file_info.tests|length - 5 }} more tests
{% endif %}
{% endif %}

---

{% endfor %}
'''


class HTMLFormatter(BaseFormatter):
    """HTML formatter for test documentation with interactive features."""

    def __init__(self, template_dir: Optional[Path] = None, include_css: bool = True):
        super().__init__(template_dir)
        self.include_css = include_css

    def format_test_file(self, file_info: TestFileInfo) -> str:
        """Format a single test file as HTML."""
        context = self._build_html_context(file_info)
        template_str = self._get_html_file_template()
        return self._safe_format(template_str, context)

    def format_test_suite(self, files: List[TestFileInfo],
                         title: str = "Test Suite Documentation") -> str:
        """Format multiple test files as an interactive HTML document."""
        context = self._build_html_suite_context(files, title)
        template_str = self._get_html_suite_template()
        return self._safe_format(template_str, context)

    def _build_html_context(self, file_info: TestFileInfo) -> Dict[str, Any]:
        """Build HTML-specific context."""
        context = {
            'file_info': file_info,
            'include_css': self.include_css,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Add HTML-specific data
        context['css'] = self._get_css() if self.include_css else ""
        context['javascript'] = self._get_javascript()

        return context

    def _build_html_suite_context(self, files: List[TestFileInfo], title: str) -> Dict[str, Any]:
        """Build HTML suite context."""
        return {
            'title': title,
            'files': files,
            'include_css': self.include_css,
            'css': self._get_css() if self.include_css else "",
            'javascript': self._get_javascript(),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def _get_css(self) -> str:
        """Get CSS styles for HTML output."""
        return '''
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { border-bottom: 2px solid #e1e5e9; padding-bottom: 20px; margin-bottom: 30px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }
            .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 14px; color: #6c757d; margin-top: 5px; }
            .test-section { margin: 30px 0; }
            .test-item { border: 1px solid #e1e5e9; border-radius: 6px; margin: 15px 0; padding: 20px; }
            .test-header { display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }
            .test-name { font-size: 18px; font-weight: bold; color: #2c3e50; }
            .test-meta { font-size: 12px; color: #7f8c8d; }
            .badges { margin: 10px 0; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px; }
            .badge-async { background: #e3f2fd; color: #1976d2; }
            .badge-parametrized { background: #f3e5f5; color: #7b1fa2; }
            .badge-skip { background: #fff3e0; color: #f57c00; }
            .badge-xfail { background: #ffebee; color: #c62828; }
            .docstring { background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; font-style: italic; }
            .decorators, .parameters { margin: 10px 0; }
            .code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 90%; }
            .error { background: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 15px 0; }
            .collapsible { cursor: pointer; padding: 10px; background: #f1f3f4; border: none; width: 100%; text-align: left; font-weight: bold; }
            .collapsible:hover { background: #e8eaed; }
            .collapsible-content { display: none; padding: 10px; }
            .collapsible.active + .collapsible-content { display: block; }
            .filter-section { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }
            .filter-input { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            .hidden { display: none !important; }
        </style>
        '''

    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return '''
        <script>
            // Toggle collapsible sections
            document.addEventListener('DOMContentLoaded', function() {
                const collapsibles = document.querySelectorAll('.collapsible');
                collapsibles.forEach(function(collapsible) {
                    collapsible.addEventListener('click', function() {
                        this.classList.toggle('active');
                        const content = this.nextElementSibling;
                        if (content.style.display === 'block') {
                            content.style.display = 'none';
                        } else {
                            content.style.display = 'block';
                        }
                    });
                });

                // Filter functionality
                const filterInput = document.getElementById('test-filter');
                if (filterInput) {
                    filterInput.addEventListener('input', function() {
                        const filterText = this.value.toLowerCase();
                        const testItems = document.querySelectorAll('.test-item');

                        testItems.forEach(function(item) {
                            const testName = item.querySelector('.test-name').textContent.toLowerCase();
                            const docstring = item.querySelector('.docstring');
                            const docText = docstring ? docstring.textContent.toLowerCase() : '';

                            if (testName.includes(filterText) || docText.includes(filterText)) {
                                item.classList.remove('hidden');
                            } else {
                                item.classList.add('hidden');
                            }
                        });
                    });
                }
            });
        </script>
        '''

    def _get_html_file_template(self) -> str:
        """Get HTML template for single file."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Documentation: {{ file_info.file_path.name }}</title>
    {{ css|safe }}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 {{ file_info.file_path.name }}</h1>
            <p><strong>Path:</strong> <code>{{ file_info.file_path }}</code></p>
            <p><strong>Generated:</strong> {{ generated_at }}</p>
        </div>

        {% if file_info.parse_errors %}
        <div class="error">
            <h3>⚠️ Parse Errors</h3>
            {% for error in file_info.parse_errors %}
            <p>{{ error }}</p>
            {% endfor %}
        </div>
        {% endif %}

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ file_info.tests|length }}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ file_info.tests|selectattr("is_async")|list|length }}</div>
                <div class="stat-label">Async Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ file_info.tests|selectattr("is_parametrized")|list|length }}</div>
                <div class="stat-label">Parametrized Tests</div>
            </div>
        </div>

        {% if file_info.tests %}
        <div class="filter-section">
            <input type="text" id="test-filter" class="filter-input" placeholder="Filter tests by name or description...">
        </div>

        {% for test in file_info.tests %}
        <div class="test-item">
            <div class="test-header">
                <div class="test-name">{{ test.name }}</div>
                <div class="test-meta">Line {{ test.line_number }} | Complexity {{ test.complexity_score }}/10</div>
            </div>

            <div class="badges">
                {% if test.is_async %}<span class="badge badge-async">Async</span>{% endif %}
                {% if test.is_parametrized %}<span class="badge badge-parametrized">Parametrized</span>{% endif %}
                {% if test.skip_reason %}<span class="badge badge-skip">Skipped</span>{% endif %}
                {% if test.expected_to_fail %}<span class="badge badge-xfail">Expected Failure</span>{% endif %}
            </div>

            {% if test.docstring %}
            <div class="docstring">{{ test.docstring }}</div>
            {% endif %}

            {% if test.decorators or test.parameters %}
            <button class="collapsible">Details</button>
            <div class="collapsible-content">
                {% if test.decorators %}
                <div class="decorators">
                    <strong>Decorators:</strong>
                    {% for decorator in test.decorators %}
                    <div><code>{{ decorator.name }}</code></div>
                    {% endfor %}
                </div>
                {% endif %}

                {% if test.parameters %}
                <div class="parameters">
                    <strong>Parameters:</strong>
                    {% for param in test.parameters %}
                    <div><code>{{ param.name }}{% if param.type_annotation %}: {{ param.type_annotation }}{% endif %}</code></div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
        {% endif %}
    </div>

    {{ javascript|safe }}
</body>
</html>'''

    def _get_html_suite_template(self) -> str:
        """Get HTML template for test suite."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    {{ css|safe }}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {{ title }}</h1>
            <p><strong>Generated:</strong> {{ generated_at }}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ files|length }}</div>
                <div class="stat-label">Test Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ files|sum(attribute='tests')|length }}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ files|selectattr("parse_errors")|list|length }}</div>
                <div class="stat-label">Files with Errors</div>
            </div>
        </div>

        <div class="test-section">
            <h2>Test Files</h2>
            {% for file_info in files %}
            <div class="test-item">
                <div class="test-header">
                    <div class="test-name">📁 {{ file_info.file_path.name }}</div>
                    <div class="test-meta">{{ file_info.tests|length }} tests{% if file_info.parse_errors %} | {{ file_info.parse_errors|length }} errors{% endif %}</div>
                </div>

                <p><code>{{ file_info.file_path }}</code></p>

                {% if file_info.tests %}
                <button class="collapsible">View Tests ({{ file_info.tests|length }})</button>
                <div class="collapsible-content">
                    {% for test in file_info.tests %}
                    <div style="margin: 10px 0; padding: 10px; border-left: 3px solid #007bff;">
                        <strong>{{ test.name }}</strong> ({{ test.test_type.value }})
                        {% if test.docstring %}<br><em>{{ test.docstring[:100] }}{% if test.docstring|length > 100 %}...{% endif %}</em>{% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                {% if file_info.parse_errors %}
                <div class="error">
                    <strong>Parse Errors:</strong>
                    {% for error in file_info.parse_errors %}
                    <div>{{ error }}</div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>

    {{ javascript|safe }}
</body>
</html>'''


class JSONFormatter(BaseFormatter):
    """JSON formatter for programmatic access to test documentation."""

    def format_test_file(self, file_info: TestFileInfo) -> str:
        """Format a single test file as JSON."""
        try:
            # Convert to dict and handle non-serializable types
            data = self._convert_to_json_safe(file_info)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON formatting error: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    def format_test_suite(self, files: List[TestFileInfo],
                         title: str = "Test Suite Documentation") -> str:
        """Format multiple test files as JSON."""
        try:
            suite_data = {
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_files": len(files),
                    "total_tests": sum(len(f.tests) for f in files),
                    "files_with_errors": sum(1 for f in files if f.parse_errors)
                },
                "files": [self._convert_to_json_safe(file_info) for file_info in files]
            }
            return json.dumps(suite_data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON suite formatting error: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    def _convert_to_json_safe(self, obj: Any) -> Any:
        """Convert objects to JSON-safe format."""
        if hasattr(obj, '__dict__'):
            # Handle dataclasses and objects
            result = {}
            for key, value in asdict(obj).items() if hasattr(obj, '__dataclass_fields__') else obj.__dict__.items():
                result[key] = self._convert_value(value)
            return result
        else:
            return self._convert_value(obj)

    def _convert_value(self, value: Any) -> Any:
        """Convert individual values to JSON-safe types."""
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, set):
            return list(value)
        elif hasattr(value, '__dict__') or hasattr(value, '__dataclass_fields__'):
            return self._convert_to_json_safe(value)
        elif isinstance(value, list):
            return [self._convert_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        else:
            return value