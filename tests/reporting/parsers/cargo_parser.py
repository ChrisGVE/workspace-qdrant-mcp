"""
Parser for cargo test output.

Parses Rust test results from cargo test text output or JSON output
(cargo test -- --format json).
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from ..models import (
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from .base import BaseParser


class CargoTestParser(BaseParser):
    """Parser for cargo test output."""

    def parse(self, source: str | Path | dict) -> TestRun:
        """
        Parse cargo test results into TestRun.

        Args:
            source: Path to cargo test output file, text output, or dict with JSON data

        Returns:
            TestRun object with cargo test results
        """
        # If dict, treat as JSON format
        if isinstance(source, dict):
            return self._parse_json(source)

        # If Path, read file content
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                with open(path) as f:
                    content = f.read()
            else:
                # Treat as string content
                content = str(source)
        else:
            content = str(source)

        # Try to parse as JSON first (cargo test -- --format json)
        try:
            data = json.loads(content)
            return self._parse_json(data)
        except (json.JSONDecodeError, ValueError):
            # Parse as text output
            return self._parse_text(content)

    def _parse_text(self, content: str) -> TestRun:
        """Parse cargo test text output."""
        timestamp = datetime.now()

        # Create test run
        test_run = TestRun.create(
            source=TestSource.CARGO,
            timestamp=timestamp,
            metadata={"format": "cargo_text"},
        )

        # Parse output line by line
        lines = content.split("\n")

        # Look for test results
        # Format: "test module::test_name ... ok"
        #         "test module::test_name ... FAILED"
        #         "test module::test_name ... ignored"

        test_results: list[dict[str, Any]] = []

        for line in lines:
            line = line.strip()

            # Skip empty lines and non-test lines
            if not line or not line.startswith("test "):
                continue

            # Parse test line
            match = re.match(
                r"test\s+([^\s]+)\s+\.\.\.\s+(\w+)(?:\s+in\s+([0-9.]+)(s|ms))?", line
            )
            if match:
                test_name = match.group(1)
                outcome = match.group(2).lower()
                duration_val = match.group(3)
                duration_unit = match.group(4)

                # Convert duration to ms
                duration_ms = 0.0
                if duration_val and duration_unit:
                    val = float(duration_val)
                    if duration_unit == "s":
                        duration_ms = val * 1000
                    else:  # ms
                        duration_ms = val

                # Map outcome to status
                status_map = {
                    "ok": TestStatus.PASSED,
                    "failed": TestStatus.FAILED,
                    "ignored": TestStatus.SKIPPED,
                }
                status = status_map.get(outcome, TestStatus.ERROR)

                test_results.append(
                    {
                        "name": test_name,
                        "status": status,
                        "duration_ms": duration_ms,
                    }
                )

        # Group tests by module into suites
        suite_map: dict[str, TestSuite] = {}

        for test_data in test_results:
            name = test_data["name"]

            # Extract module name (everything before last ::)
            if "::" in name:
                module_name = name.rsplit("::", 1)[0]
                test_name = name.rsplit("::", 1)[1]
            else:
                module_name = "root"
                test_name = name

            # Get or create suite
            if module_name not in suite_map:
                suite = TestSuite(
                    suite_id=str(uuid4()),
                    name=module_name,
                    test_type=TestType.UNIT,  # Cargo tests are typically unit tests
                )
                suite_map[module_name] = suite
                test_run.add_suite(suite)
            else:
                suite = suite_map[module_name]

            # Create test case
            case = TestCase(
                case_id=str(uuid4()),
                name=test_name,
                test_type=TestType.UNIT,
                metadata={"full_name": name, "module": module_name},
            )

            # Create result
            result = TestResult(
                test_id=str(uuid4()),
                name=test_name,
                status=test_data["status"],
                duration_ms=test_data["duration_ms"],
                timestamp=timestamp,
            )

            case.add_result(result)
            suite.add_test_case(case)

        # Parse summary line
        # Format: "test result: ok. 123 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.23s"
        summary_match = re.search(
            r"test result:\s+(\w+)\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored",
            content,
        )
        if summary_match:
            test_run.metadata.update(
                {
                    "result": summary_match.group(1),
                    "passed": int(summary_match.group(2)),
                    "failed": int(summary_match.group(3)),
                    "ignored": int(summary_match.group(4)),
                }
            )

        return test_run

    def _parse_json(self, data: dict[str, Any] | list[dict[str, Any]]) -> TestRun:
        """
        Parse cargo test JSON output (cargo test -- --format json).

        The JSON format is line-delimited JSON (NDJSON) with events like:
        {"type": "suite", "event": "started", ...}
        {"type": "test", "event": "started", "name": "test_name"}
        {"type": "test", "event": "ok", "name": "test_name", "exec_time": 0.001}
        """
        timestamp = datetime.now()

        # Create test run
        test_run = TestRun.create(
            source=TestSource.CARGO,
            timestamp=timestamp,
            metadata={"format": "cargo_json"},
        )

        # If data is a list, process each event
        # If data is a dict, process single event or assume it's a collection
        events = []
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            # Check if it's a single event or a collection
            if "type" in data:
                events = [data]
            else:
                # Assume it's keyed by test name
                # Convert to events
                for name, result in data.items():
                    events.append(
                        {
                            "type": "test",
                            "name": name,
                            "event": result.get("event", "ok"),
                            "exec_time": result.get("exec_time", 0),
                        }
                    )

        # Process events
        suite_map: dict[str, TestSuite] = {}
        test_events = {}

        for event in events:
            event_type = event.get("type")
            event_name = event.get("event")

            if event_type == "test":
                test_name = event.get("name", "unknown")

                # Store event by name
                if test_name not in test_events:
                    test_events[test_name] = {}

                test_events[test_name][event_name] = event

        # Create test cases from events
        for test_name, events_dict in test_events.items():
            # Determine status from event
            if "ok" in events_dict:
                status = TestStatus.PASSED
                duration_sec = events_dict["ok"].get("exec_time", 0)
            elif "failed" in events_dict:
                status = TestStatus.FAILED
                duration_sec = events_dict["failed"].get("exec_time", 0)
            elif "ignored" in events_dict:
                status = TestStatus.SKIPPED
                duration_sec = 0
            else:
                status = TestStatus.ERROR
                duration_sec = 0

            duration_ms = duration_sec * 1000

            # Extract module name
            if "::" in test_name:
                module_name = test_name.rsplit("::", 1)[0]
                short_name = test_name.rsplit("::", 1)[1]
            else:
                module_name = "root"
                short_name = test_name

            # Get or create suite
            if module_name not in suite_map:
                suite = TestSuite(
                    suite_id=str(uuid4()),
                    name=module_name,
                    test_type=TestType.UNIT,
                )
                suite_map[module_name] = suite
                test_run.add_suite(suite)
            else:
                suite = suite_map[module_name]

            # Create test case
            case = TestCase(
                case_id=str(uuid4()),
                name=short_name,
                test_type=TestType.UNIT,
                metadata={"full_name": test_name, "module": module_name},
            )

            # Create result
            result = TestResult(
                test_id=str(uuid4()),
                name=short_name,
                status=status,
                duration_ms=duration_ms,
                timestamp=timestamp,
                metadata={"events": events_dict},
            )

            case.add_result(result)
            suite.add_test_case(case)

        return test_run
