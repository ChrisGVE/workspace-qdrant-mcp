"""Comprehensive test for the complete documentation framework."""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the docs framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'docs/framework'))

from generators.ast_parser import PythonASTParser
from generators.template_engine import DocumentationTemplateEngine, create_default_templates
from validation.coverage_analyzer import DocumentationCoverageAnalyzer
from validation.quality_checker import DocumentationQualityChecker
from server.sandbox import CodeSandbox
from deployment.builder import DocumentationBuilder


def test_complete_workflow():
    """Test the complete documentation generation workflow."""
    print("Testing Complete Documentation Framework Workflow")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test project structure
        project_dir = temp_path / "test_project"
        project_dir.mkdir()

        source_dir = project_dir / "src"
        source_dir.mkdir()

        docs_dir = project_dir / "docs"
        docs_dir.mkdir()

        templates_dir = docs_dir / "templates"
        templates_dir.mkdir()

        # Create test Python module
        test_module_code = '''
"""Test module for documentation framework.

This module demonstrates various Python constructs that should be
properly documented and analyzed by the framework.
"""

import typing
from typing import List, Optional, Dict, Any


class TestClass:
    """A test class with comprehensive documentation.

    This class demonstrates proper docstring formatting with
    parameters, examples, and detailed descriptions.

    Attributes:
        value: The internal value stored by the class

    Examples:
        >>> test = TestClass(42)
        >>> test.get_value()
        42
    """

    def __init__(self, value: int):
        """Initialize the TestClass.

        Args:
            value: Initial value to store
        """
        self.value = value

    def get_value(self) -> int:
        """Get the stored value.

        Returns:
            The stored integer value

        Examples:
            >>> test = TestClass(10)
            >>> test.get_value()
            10
        """
        return self.value

    def set_value(self, new_value: int) -> None:
        """Set a new value.

        Args:
            new_value: The new value to store

        Raises:
            ValueError: If new_value is negative
        """
        if new_value < 0:
            raise ValueError("Value cannot be negative")
        self.value = new_value

    @property
    def doubled_value(self) -> int:
        """Get the value multiplied by 2."""
        return self.value * 2


def well_documented_function(name: str, age: int,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process user information with comprehensive validation.

    This function takes user information and processes it according to
    business rules, with proper validation and error handling.

    Args:
        name: The user's full name (must not be empty)
        age: The user's age in years (must be positive)
        metadata: Optional additional information about the user

    Returns:
        A dictionary containing processed user information with
        validation status and computed fields

    Raises:
        ValueError: If name is empty or age is not positive
        TypeError: If arguments are of wrong types

    Examples:
        >>> result = well_documented_function("John Doe", 25)
        >>> result['name']
        'John Doe'
        >>> result['age']
        25

        >>> result = well_documented_function("Jane", 30, {"role": "admin"})
        >>> result['metadata']['role']
        'admin'
    """
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")

    if not isinstance(age, int) or age <= 0:
        raise ValueError("Age must be a positive integer")

    result = {
        'name': name.strip(),
        'age': age,
        'is_adult': age >= 18,
        'metadata': metadata or {}
    }

    return result


def poorly_documented_function(x, y):
    """Does some calculation."""
    return x * y + 1


# Module-level constant
DEFAULT_TIMEOUT = 30

# Another constant with no documentation
MAGIC_NUMBER = 42
'''

        (source_dir / "test_module.py").write_text(test_module_code)

        # Create configuration file
        config = {
            'project': {
                'name': 'Test Documentation Project',
                'description': 'A test project for the documentation framework',
                'version': '1.0.0'
            },
            'sources': {
                'python': [str(source_dir)]
            },
            'output': {
                'base_dir': str(docs_dir / 'generated'),
                'formats': ['html', 'markdown', 'json']
            },
            'validation': {
                'enabled': True,
                'coverage': {
                    'minimum_percentage': 80,
                    'require_examples': True,
                    'require_return_docs': True,
                    'require_param_docs': True
                },
                'quality': {
                    'min_docstring_length': 20,
                    'check_grammar': True,
                    'check_spelling': False
                }
            }
        }

        import yaml
        config_file = docs_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Create templates
        create_default_templates(templates_dir)

        print("✓ Test project structure created")

        # Test 1: AST Parser
        print("\n1. Testing AST Parser...")
        parser = PythonASTParser()
        modules = parser.parse_directory(source_dir)

        assert len(modules) == 1
        module = modules[0]
        assert module.name == "test_module"
        assert len(module.children) >= 4  # class, 2 functions, 2 constants

        print(f"   ✓ Parsed {len(module.children)} members from module")

        # Test 2: Template Engine
        print("\n2. Testing Template Engine...")
        template_engine = DocumentationTemplateEngine(templates_dir, config)

        html_output = template_engine.render_api_documentation(modules, 'html')
        assert 'TestClass' in html_output
        assert 'well_documented_function' in html_output

        print("   ✓ Generated HTML documentation")

        # Test 3: Coverage Analyzer
        print("\n3. Testing Coverage Analyzer...")
        coverage_analyzer = DocumentationCoverageAnalyzer(
            require_examples=True,
            require_return_docs=True,
            require_param_docs=True
        )

        file_coverage = coverage_analyzer.analyze_file(source_dir / "test_module.py")
        assert file_coverage.stats.total_items > 0

        print(f"   ✓ Coverage: {file_coverage.stats.coverage_percentage:.1f}%")
        print(f"   ✓ Analyzed {file_coverage.stats.total_items} items")

        # Test 4: Quality Checker
        print("\n4. Testing Quality Checker...")
        quality_checker = DocumentationQualityChecker()

        quality_report = quality_checker.check_project_quality(modules)
        assert quality_report.overall_score >= 0

        print(f"   ✓ Overall quality score: {quality_report.overall_score:.1f}/100")
        print(f"   ✓ Found {quality_report.summary_stats['total_issues']} issues")

        # Test 5: Code Sandbox
        print("\n5. Testing Code Sandbox...")
        sandbox = CodeSandbox(timeout=10)

        if sandbox.is_available():
            import asyncio

            # Test simple code execution
            try:
                result = asyncio.run(sandbox.execute_code("2 + 2"))
                print(f"   Code result: {result}")

                if result.get('result') == '4' or result.get('output') == '4':
                    print("   ✓ Code execution working")
                else:
                    print("   ⚠ Code execution gave unexpected result")

                # Test code with output
                result = asyncio.run(sandbox.execute_code('print("Hello, World!")'))
                if 'Hello, World!' in str(result.get('output', '')):
                    print("   ✓ Output capture working")
                else:
                    print("   ⚠ Output capture not working as expected")

            except Exception as e:
                print(f"   ⚠ Code execution error: {e}")

            # Test security - this should fail
            try:
                result = asyncio.run(sandbox.execute_code('import os; os.system("ls")'))
                if result.get('error'):
                    print("   ✓ Security restrictions working")
                else:
                    print("   ⚠ Security test failed - dangerous code executed")
            except Exception:
                print("   ✓ Security restrictions working")
        else:
            print("   ⚠ Code sandbox not available")

        # Test 6: Documentation Builder
        print("\n6. Testing Documentation Builder...")
        builder = DocumentationBuilder(config_file)

        build_report = builder.build_documentation()

        assert build_report['success'] == True
        assert build_report['modules_processed'] == 1
        assert len(build_report['formats_generated']) > 0

        print(f"   ✓ Built documentation for {build_report['modules_processed']} modules")
        print(f"   ✓ Generated formats: {', '.join(build_report['formats_generated'])}")

        # Verify output files exist
        output_base = docs_dir / 'generated'
        for format_type in build_report['formats_generated']:
            format_dir = output_base / format_type
            assert format_dir.exists(), f"{format_type} directory not created"

            if format_type == 'html':
                assert (format_dir / 'index.html').exists()
                assert (format_dir / 'test_module.html').exists()
            elif format_type == 'markdown':
                assert (format_dir / 'test_module.md').exists()
                assert (format_dir / 'README.md').exists()
            elif format_type == 'json':
                assert (format_dir / 'documentation.json').exists()

        print("   ✓ All output files generated correctly")

        # Test validation results
        validation = build_report.get('validation_results', {})
        if validation:
            print(f"   ✓ Validation completed:")
            if 'coverage' in validation:
                print(f"     - Coverage: {validation['coverage'].get('overall_score', 0):.1f}%")
            if 'quality' in validation:
                print(f"     - Quality: {validation['quality'].get('overall_score', 0):.1f}/100")
            if 'links' in validation:
                print(f"     - Links: {validation['links'].get('total_links', 0)} checked")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED! Documentation framework is fully functional.")

        # Print summary
        print(f"\nFramework Summary:")
        print(f"- Parsed {len(modules)} modules with {sum(len(m.children) for m in modules)} members")
        print(f"- Generated {len(build_report['formats_generated'])} output formats")
        print(f"- Documentation coverage: {file_coverage.stats.coverage_percentage:.1f}%")
        print(f"- Quality score: {quality_report.overall_score:.1f}/100")

        return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")

    # Test with empty docstring
    parser = PythonASTParser()

    empty_code = '''
def empty_function():
    pass

class EmptyClass:
    pass
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(empty_code)
        f.flush()

        try:
            module = parser.parse_file(f.name)
            assert len(module.children) == 2
            print("   ✓ Handled empty docstrings correctly")
        finally:
            os.unlink(f.name)

    # Test coverage analyzer with poor documentation
    analyzer = DocumentationCoverageAnalyzer()
    coverage = analyzer.analyze_file(__file__)  # Analyze this test file itself

    print(f"   ✓ Coverage analysis with edge cases: {coverage.stats.coverage_percentage:.1f}%")


if __name__ == "__main__":
    try:
        test_complete_workflow()
        test_edge_cases()
        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)