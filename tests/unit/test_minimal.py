"""
Ultra-minimal test to diagnose execution hanging issue.
No project imports, just basic Python functionality.
"""
import sys


def test_basic_math():
    """Test basic arithmetic - should execute in milliseconds."""
    assert 2 + 2 == 4


def test_system_info():
    """Test basic system access - should execute in milliseconds."""
    assert sys.version_info.major >= 3
