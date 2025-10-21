"""
Test result parsers for different formats.

Provides parsers for:
- pytest (JSON, JUnit XML)
- cargo test (text output)
- Custom benchmark JSON
- gRPC integration tests
"""

from .base import BaseParser

__all__ = ["BaseParser"]
