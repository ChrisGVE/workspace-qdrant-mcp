"""
Base parser interface for test result formats.

All format-specific parsers inherit from BaseParser and implement parse() method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ..models import TestRun


class BaseParser(ABC):
    """Base class for test result parsers."""

    @abstractmethod
    def parse(self, source: str | Path | dict) -> TestRun:
        """
        Parse test results from a source into a TestRun object.

        Args:
            source: Can be:
                - File path (str or Path) to test results
                - Dictionary with pre-loaded test data
                - String content of test results

        Returns:
            TestRun object with parsed results

        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If file path doesn't exist
        """
        pass

    def _ensure_path(self, source: str | Path) -> Path:
        """
        Convert source to Path and verify it exists.

        Args:
            source: File path as string or Path

        Returns:
            Path object

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Test result file not found: {path}")
        return path
