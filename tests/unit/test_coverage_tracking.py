"""
Unit tests for coverage tracking system.

Tests coverage parsers, storage, threshold checking, and report generation.
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.reporting.coverage_checker import (
    CoverageChecker,
    CoverageThresholds,
    ThresholdStatus,
)
from tests.reporting.models import CoverageMetrics, FileCoverage, TestRun, TestSource
from tests.reporting.parsers.coverage_py_parser import CoveragePyParser
from tests.reporting.parsers.tarpaulin_parser import TarpaulinParser
from tests.reporting.storage import TestResultStorage


class TestCoverageDataModels:
    """Test coverage data models."""

    def test_file_coverage_creation(self):
        """Test FileCoverage creation and serialization."""
        file_cov = FileCoverage(
            file_path="src/test.py",
            lines_covered=80,
            lines_total=100,
            line_coverage_percent=80.0,
            uncovered_lines=[10, 20, 30],
        )

        assert file_cov.file_path == "src/test.py"
        assert file_cov.lines_covered == 80
        assert file_cov.line_coverage_percent == 80.0
        assert len(file_cov.uncovered_lines) == 3

        # Test serialization
        data = file_cov.to_dict()
        assert data["file_path"] == "src/test.py"
        assert data["lines_covered"] == 80

        # Test deserialization
        restored = FileCoverage.from_dict(data)
        assert restored.file_path == file_cov.file_path
        assert restored.lines_covered == file_cov.lines_covered

    def test_coverage_metrics_creation(self):
        """Test CoverageMetrics creation and serialization."""
        file_cov = FileCoverage(
            file_path="src/test.py",
            lines_covered=80,
            lines_total=100,
            line_coverage_percent=80.0,
        )

        coverage = CoverageMetrics(
            line_coverage_percent=85.0,
            lines_covered=850,
            lines_total=1000,
            function_coverage_percent=75.0,
            functions_covered=30,
            functions_total=40,
            file_coverage=[file_cov],
            coverage_tool="coverage.py",
        )

        assert coverage.line_coverage_percent == 85.0
        assert coverage.lines_covered == 850
        assert len(coverage.file_coverage) == 1

        # Test serialization
        data = coverage.to_dict()
        assert data["line_coverage_percent"] == 85.0
        assert data["coverage_tool"] == "coverage.py"
        assert len(data["file_coverage"]) == 1

        # Test deserialization
        restored = CoverageMetrics.from_dict(data)
        assert restored.line_coverage_percent == coverage.line_coverage_percent
        assert len(restored.file_coverage) == 1


class TestCoveragePyParser:
    """Test coverage.py XML parser."""

    def test_parse_simple_coverage_xml(self, tmp_path):
        """Test parsing a simple coverage.xml file."""
        # Create minimal coverage.xml
        coverage_xml = tmp_path / "coverage.xml"
        xml_content = """<?xml version="1.0" ?>
<coverage line-rate="0.85" branch-rate="0.70">
    <packages>
        <package name="src.test">
            <classes>
                <class name="module.py" filename="src/test/module.py">
                    <lines>
                        <line number="1" hits="1"/>
                        <line number="2" hits="1"/>
                        <line number="3" hits="0"/>
                        <line number="4" hits="1"/>
                        <line number="5" hits="1"/>
                    </lines>
                </class>
            </classes>
        </package>
    </packages>
</coverage>"""
        coverage_xml.write_text(xml_content)

        # Parse
        parser = CoveragePyParser()
        coverage = parser.parse(coverage_xml)

        # Verify
        assert coverage.line_coverage_percent == 85.0
        assert coverage.branch_coverage_percent == 70.0
        assert coverage.coverage_tool == "coverage.py"
        assert len(coverage.file_coverage) == 1

        file_cov = coverage.file_coverage[0]
        assert file_cov.lines_total == 5
        assert file_cov.lines_covered == 4
        assert 3 in file_cov.uncovered_lines


class TestTarpaulinParser:
    """Test tarpaulin JSON parser."""

    def test_parse_tarpaulin_json(self, tmp_path):
        """Test parsing tarpaulin JSON format."""
        # Create minimal tarpaulin JSON
        coverage_json = tmp_path / "coverage.json"
        json_data = {
            "files": {
                "src/main.rs": {
                    "covered": [1, 2, 5, 6, 7],
                    "uncovered": [3, 4],
                }
            }
        }
        coverage_json.write_text(json.dumps(json_data))

        # Parse
        parser = TarpaulinParser()
        coverage = parser.parse(coverage_json)

        # Verify
        assert coverage.coverage_tool == "tarpaulin"
        assert coverage.lines_total == 7
        assert coverage.lines_covered == 5
        assert abs(coverage.line_coverage_percent - 71.43) < 0.1
        assert len(coverage.file_coverage) == 1

        file_cov = coverage.file_coverage[0]
        assert file_cov.file_path == "src/main.rs"
        assert file_cov.lines_covered == 5
        assert 3 in file_cov.uncovered_lines
        assert 4 in file_cov.uncovered_lines


class TestCoverageStorage:
    """Test coverage storage in database."""

    def test_save_and_load_coverage(self, tmp_path):
        """Test saving and loading coverage with test run."""
        # Create storage
        db_path = tmp_path / "test.db"
        storage = TestResultStorage(db_path)

        # Create test run with coverage
        test_run = TestRun.create(source=TestSource.PYTEST)

        file_cov = FileCoverage(
            file_path="src/test.py",
            lines_covered=80,
            lines_total=100,
            line_coverage_percent=80.0,
            uncovered_lines=[10, 20],
        )

        test_run.coverage = CoverageMetrics(
            line_coverage_percent=85.0,
            lines_covered=850,
            lines_total=1000,
            file_coverage=[file_cov],
            coverage_tool="coverage.py",
        )

        # Save
        storage.save_test_run(test_run)

        # Load
        loaded_run = storage.get_test_run(test_run.run_id)

        # Verify
        assert loaded_run is not None
        assert loaded_run.coverage is not None
        assert loaded_run.coverage.line_coverage_percent == 85.0
        assert loaded_run.coverage.lines_covered == 850
        assert loaded_run.coverage.coverage_tool == "coverage.py"
        assert len(loaded_run.coverage.file_coverage) == 1

        loaded_file_cov = loaded_run.coverage.file_coverage[0]
        assert loaded_file_cov.file_path == "src/test.py"
        assert loaded_file_cov.lines_covered == 80
        assert len(loaded_file_cov.uncovered_lines) == 2


class TestCoverageThresholds:
    """Test coverage threshold checking."""

    def test_default_thresholds(self):
        """Test default threshold creation."""
        thresholds = CoverageThresholds.default()
        assert thresholds.line_coverage_min == 80.0
        assert thresholds.function_coverage_min == 70.0

    def test_strict_thresholds(self):
        """Test strict threshold creation."""
        thresholds = CoverageThresholds.strict()
        assert thresholds.line_coverage_min == 90.0
        assert thresholds.function_coverage_min == 85.0

    def test_threshold_pass(self):
        """Test coverage passing thresholds."""
        coverage = CoverageMetrics(
            line_coverage_percent=90.0,
            lines_covered=900,
            lines_total=1000,
            function_coverage_percent=85.0,
            functions_covered=34,
            functions_total=40,
        )

        thresholds = CoverageThresholds(
            line_coverage_min=80.0,
            function_coverage_min=75.0,
        )

        checker = CoverageChecker(thresholds)
        result = checker.check(coverage)

        assert result.status == ThresholdStatus.PASSED
        assert result.passed
        assert len(result.violations) == 0

    def test_threshold_fail(self):
        """Test coverage failing thresholds."""
        coverage = CoverageMetrics(
            line_coverage_percent=70.0,
            lines_covered=700,
            lines_total=1000,
        )

        thresholds = CoverageThresholds(line_coverage_min=80.0)

        checker = CoverageChecker(thresholds)
        result = checker.check(coverage)

        assert result.status == ThresholdStatus.FAILED
        assert not result.passed
        assert len(result.violations) == 1
        assert result.violations[0].metric == "line_coverage"
        assert result.violations[0].actual == 70.0
        assert result.violations[0].threshold == 80.0

    def test_threshold_warning(self):
        """Test coverage with warnings but no failures."""
        coverage = CoverageMetrics(
            line_coverage_percent=85.0,
            lines_covered=850,
            lines_total=1000,
        )

        thresholds = CoverageThresholds(
            line_coverage_min=80.0,
            line_coverage_warning=90.0,
        )

        checker = CoverageChecker(thresholds)
        result = checker.check(coverage)

        assert result.status == ThresholdStatus.WARNING
        assert result.passed  # Warnings don't cause failure
        assert result.has_warnings
        assert len(result.violations) == 0
        assert len(result.warnings) == 1

    def test_per_file_threshold(self):
        """Test per-file coverage thresholds."""
        file_cov1 = FileCoverage(
            file_path="src/good.py",
            lines_covered=90,
            lines_total=100,
            line_coverage_percent=90.0,
        )

        file_cov2 = FileCoverage(
            file_path="src/bad.py",
            lines_covered=50,
            lines_total=100,
            line_coverage_percent=50.0,
        )

        coverage = CoverageMetrics(
            line_coverage_percent=70.0,
            lines_covered=140,
            lines_total=200,
            file_coverage=[file_cov1, file_cov2],
        )

        thresholds = CoverageThresholds(
            line_coverage_min=60.0,  # Overall passes
            file_line_coverage_min=80.0,  # But per-file fails
        )

        checker = CoverageChecker(thresholds)
        result = checker.check(coverage)

        assert result.status == ThresholdStatus.FAILED
        assert len(result.violations) == 1
        assert result.violations[0].file_path == "src/bad.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
