"""Test script for the documentation validation system."""

import sys
import os
from pathlib import Path

# Add the docs framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'docs/framework'))

from validation.coverage_analyzer import DocumentationCoverageAnalyzer, CoverageStats
from validation.quality_checker import DocumentationQualityChecker
from generators.ast_parser import PythonASTParser, DocumentationNode, MemberType

def test_coverage_stats():
    """Test coverage stats calculation."""
    print("Testing CoverageStats...")
    stats = CoverageStats(total_items=10, documented_items=7)
    stats.calculate_percentage()
    assert stats.coverage_percentage == 70.0
    print("✓ CoverageStats working correctly")

def test_ast_parser():
    """Test AST parser functionality."""
    print("Testing AST parser...")

    # Test with a simple code sample
    code = '''
"""Test module."""

def test_function(param: str) -> str:
    """Test function with documentation.

    Args:
        param: Input parameter

    Returns:
        Processed string
    """
    return param.upper()

class TestClass:
    """Test class."""

    def method(self):
        """Test method."""
        pass
'''

    # Write to temporary file and parse
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()

        try:
            parser = PythonASTParser()
            module = parser.parse_file(f.name)

            assert module.name == 'tmp' or 'tmp' in module.name
            assert module.member_type == MemberType.MODULE
            assert len(module.children) >= 2  # function and class

            print("✓ AST parser working correctly")
        finally:
            os.unlink(f.name)

def test_coverage_analyzer():
    """Test coverage analyzer."""
    print("Testing Coverage Analyzer...")

    analyzer = DocumentationCoverageAnalyzer()

    # Test with sample code
    code = '''
def well_documented(param: str) -> str:
    """Well documented function.

    Args:
        param: Input parameter

    Returns:
        Result string
    """
    return param

def poorly_documented():
    pass
'''

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()

        try:
            coverage = analyzer.analyze_file(f.name)
            assert coverage.stats.total_items > 0
            assert coverage.stats.coverage_percentage >= 0

            print(f"✓ Coverage analyzer working: {coverage.stats.coverage_percentage:.1f}% coverage")
        finally:
            os.unlink(f.name)

def test_quality_checker():
    """Test quality checker."""
    print("Testing Quality Checker...")

    checker = DocumentationQualityChecker()

    # Create a test node
    node = DocumentationNode(
        name="test_function",
        member_type=MemberType.FUNCTION,
        docstring="This is a well-written docstring that explains the function purpose clearly."
    )

    report = checker.check_member_quality(node)
    assert report.member_name == "test_function"
    assert report.quality_score >= 0

    print(f"✓ Quality checker working: score {report.quality_score:.1f}/100")

def main():
    """Run all tests."""
    print("Testing Documentation Framework Components")
    print("=" * 50)

    try:
        test_coverage_stats()
        test_ast_parser()
        test_coverage_analyzer()
        test_quality_checker()

        print("\n" + "=" * 50)
        print("✓ All tests passed! Documentation framework is working.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()