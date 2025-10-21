"""
Main test result aggregator.

Orchestrates parsing test results from multiple sources and storing them
in the unified database. Supports incremental result collection for CI/CD pipelines.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import TestRun, TestSource
from .parsers.base import BaseParser
from .parsers.benchmark_parser import BenchmarkJsonParser
from .parsers.cargo_parser import CargoTestParser
from .parsers.pytest_parser import PytestParser
from .storage import TestResultStorage


class TestResultAggregator:
    """
    Main aggregator for test results from multiple sources.

    Provides a unified interface for parsing and storing test results
    from pytest, cargo test, benchmarks, and other sources.
    """

    def __init__(self, storage: Optional[TestResultStorage] = None):
        """
        Initialize aggregator.

        Args:
            storage: Storage backend. If None, creates default storage.
        """
        self.storage = storage or TestResultStorage()

        # Register parsers
        self.parsers: Dict[TestSource, BaseParser] = {
            TestSource.PYTEST: PytestParser(),
            TestSource.CARGO: CargoTestParser(),
            TestSource.BENCHMARK_JSON: BenchmarkJsonParser(),
            TestSource.GRPC_TEST: BenchmarkJsonParser(),  # Uses same parser
        }

    def aggregate_from_file(
        self,
        file_path: Union[str, Path],
        source: TestSource,
        run_id: Optional[str] = None,
    ) -> TestRun:
        """
        Parse and store test results from a file.

        Args:
            file_path: Path to test result file
            source: Type of test results (pytest, cargo, benchmark, etc.)
            run_id: Optional run ID for incremental aggregation

        Returns:
            TestRun object with parsed results

        Raises:
            ValueError: If source type is not supported or file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        parser = self._get_parser(source)
        test_run = parser.parse(file_path)

        # Override run_id if provided (for incremental aggregation)
        if run_id:
            test_run.run_id = run_id

        # Store in database
        self.storage.save_test_run(test_run)

        return test_run

    def aggregate_from_data(
        self,
        data: Union[str, dict],
        source: TestSource,
        run_id: Optional[str] = None,
    ) -> TestRun:
        """
        Parse and store test results from raw data.

        Args:
            data: String content or dictionary with test results
            source: Type of test results
            run_id: Optional run ID for incremental aggregation

        Returns:
            TestRun object with parsed results

        Raises:
            ValueError: If source type is not supported or data format is invalid
        """
        parser = self._get_parser(source)
        test_run = parser.parse(data)

        # Override run_id if provided
        if run_id:
            test_run.run_id = run_id

        # Store in database
        self.storage.save_test_run(test_run)

        return test_run

    def aggregate_multiple(
        self,
        sources: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> TestRun:
        """
        Aggregate test results from multiple sources into a single test run.

        This is useful for CI/CD pipelines where different test stages
        produce separate result files (unit tests, integration tests, benchmarks, etc.).

        Args:
            sources: List of source specifications, each with:
                - 'file': Path to result file (optional if 'data' provided)
                - 'data': Raw result data (optional if 'file' provided)
                - 'source': TestSource type
            run_id: Optional run ID to group all results under

        Returns:
            Combined TestRun with all results

        Example:
            >>> aggregator.aggregate_multiple([
            ...     {'file': 'junit.xml', 'source': TestSource.PYTEST},
            ...     {'file': 'cargo_output.txt', 'source': TestSource.CARGO},
            ...     {'file': 'benchmarks.json', 'source': TestSource.BENCHMARK_JSON},
            ... ], run_id='ci-pipeline-123')
        """
        if not run_id:
            # Generate run_id from first source
            from uuid import uuid4

            run_id = str(uuid4())

        combined_run: Optional[TestRun] = None

        for source_spec in sources:
            source_type = source_spec.get("source")
            if not source_type:
                raise ValueError("Each source must specify 'source' type")

            # Parse the source
            if "file" in source_spec:
                test_run = self.aggregate_from_file(
                    source_spec["file"], source_type, run_id=run_id
                )
            elif "data" in source_spec:
                test_run = self.aggregate_from_data(
                    source_spec["data"], source_type, run_id=run_id
                )
            else:
                raise ValueError("Each source must specify either 'file' or 'data'")

            # Merge into combined run
            if combined_run is None:
                combined_run = test_run
            else:
                # Add suites from this run to combined run
                for suite in test_run.suites:
                    combined_run.add_suite(suite)

        # Update the combined run in storage
        if combined_run:
            self.storage.save_test_run(combined_run)

        return combined_run

    def get_test_run(self, run_id: str) -> Optional[TestRun]:
        """
        Retrieve a test run by ID.

        Args:
            run_id: Test run ID

        Returns:
            TestRun object or None if not found
        """
        return self.storage.get_test_run(run_id)

    def list_test_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        source: Optional[TestSource] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List test runs with optional filtering.

        Args:
            limit: Maximum number of runs to return
            offset: Offset for pagination
            source: Filter by test source
            start_date: Filter runs after this date
            end_date: Filter runs before this date

        Returns:
            List of test run summaries
        """
        return self.storage.list_test_runs(
            limit=limit,
            offset=offset,
            source=source,
            start_date=start_date,
            end_date=end_date,
        )

    def delete_test_run(self, run_id: str) -> bool:
        """
        Delete a test run and all associated data.

        Args:
            run_id: Test run ID

        Returns:
            True if deleted, False if not found
        """
        return self.storage.delete_test_run(run_id)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics from all test runs.

        Returns:
            Dictionary with statistics
        """
        return self.storage.get_statistics()

    def _get_parser(self, source: TestSource) -> BaseParser:
        """
        Get parser for the specified test source.

        Args:
            source: Test source type

        Returns:
            Parser instance

        Raises:
            ValueError: If source type is not supported
        """
        parser = self.parsers.get(source)
        if not parser:
            raise ValueError(f"Unsupported test source: {source}")
        return parser

    def register_parser(self, source: TestSource, parser: BaseParser) -> None:
        """
        Register a custom parser for a test source.

        Args:
            source: Test source type
            parser: Parser instance implementing BaseParser
        """
        self.parsers[source] = parser


# Convenience function for quick aggregation
def aggregate_test_results(
    file_path: Union[str, Path],
    source: TestSource,
    storage_path: Optional[Path] = None,
) -> TestRun:
    """
    Quick helper to aggregate test results from a file.

    Args:
        file_path: Path to test result file
        source: Type of test results
        storage_path: Optional path to SQLite database

    Returns:
        TestRun object with parsed and stored results

    Example:
        >>> from tests.reporting.aggregator import aggregate_test_results
        >>> from tests.reporting.models import TestSource
        >>> run = aggregate_test_results('results.xml', TestSource.PYTEST)
        >>> print(f"Aggregated {run.total_tests} tests")
    """
    storage = TestResultStorage(storage_path) if storage_path else TestResultStorage()
    aggregator = TestResultAggregator(storage)
    return aggregator.aggregate_from_file(file_path, source)
