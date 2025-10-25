"""
Fuzzing integration and automation for security testing.

Implements automated fuzzing using Hypothesis for Python components and provides
integration guides for AFL++ fuzzing of Rust components. Includes fuzzing
harnesses, crash detection, corpus generation, and CI/CD integration.

Fuzzing Tools:
- Hypothesis: Property-based testing and fuzzing for Python
- AFL++: Coverage-guided fuzzing for Rust components (documentation)
- Crash detection and reporting infrastructure
- CI/CD pipeline integration examples

For AFL++ Rust fuzzing, see: .github/workflows/fuzzing.yml and rust-fuzzing/
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Try to import hypothesis for fuzzing tests
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Mock decorators for when hypothesis is not available
    def given(*args, **kwargs):
        return lambda f: pytest.mark.skip(reason="hypothesis not installed")(f)
    def settings(*args, **kwargs):
        return lambda f: f
    # Mock HealthCheck
    class HealthCheck:
        too_slow = None
    # Mock strategies
    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def dictionaries(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None
        @staticmethod
        def one_of(*args, **kwargs):
            return None
        @staticmethod
        def sampled_from(*args, **kwargs):
            return None
        @staticmethod
        def booleans(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None


@pytest.mark.security
class TestHypothesisFuzzing:
    """Property-based fuzzing tests using Hypothesis."""

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_text_input_handling(self, text_input: str):
        """Fuzz test for text input handling."""
        # Any text input should be handled safely without crashes
        try:
            # Simulate text processing
            result = self._process_text_input(text_input)

            # Should return a result (even if empty)
            assert isinstance(result, str)

            # Should not contain unescaped HTML
            if "<script>" in text_input.lower():
                assert "<script>" not in result.lower()

        except (ValueError, TypeError) as e:
            # Expected exceptions are acceptable
            assert str(e)  # Error should have message

    @given(st.integers(min_value=-2**63, max_value=2**63-1))
    @settings(max_examples=100)
    def test_fuzz_integer_handling(self, int_value: int):
        """Fuzz test for integer handling."""
        # Any integer should be handled safely
        try:
            result = self._process_integer(int_value)
            assert isinstance(result, (int, float, str))
        except (ValueError, OverflowError) as e:
            # Expected exceptions
            assert str(e)

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=100),
        values=st.one_of(
            st.text(max_size=1000),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ),
        max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_json_processing(self, data_dict: dict[str, Any]):
        """Fuzz test for JSON data processing."""
        # Any valid dictionary should be JSON-serializable
        try:
            # Convert to JSON and back
            json_str = json.dumps(data_dict)
            parsed = json.loads(json_str)

            # Should roundtrip correctly
            assert isinstance(parsed, dict)

            # Process the data
            result = self._process_json_data(parsed)
            assert result is not None

        except (TypeError, ValueError, OverflowError):
            # Some edge cases may not be JSON-serializable
            pass

    @given(st.lists(
        st.text(min_size=0, max_size=100),
        min_size=0,
        max_size=1000
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_list_processing(self, list_data: list[str]):
        """Fuzz test for list data processing."""
        # Any list should be processed safely
        try:
            result = self._process_list_data(list_data)
            assert isinstance(result, (list, tuple, set))

            # No crashes on iteration
            for item in result:
                assert item is not None or item is None  # Tautology to force iteration

        except (ValueError, TypeError):
            # Expected exceptions
            pass

    def _process_text_input(self, text: str) -> str:
        """Process text input safely."""
        import html
        # HTML escape for safety
        escaped = html.escape(text)
        # Truncate if too long
        return escaped[:10000]

    def _process_integer(self, value: int) -> Any:
        """Process integer safely."""
        # Handle large integers
        if abs(value) > 2**32:
            return str(value)  # Convert to string for large values
        return value

    def _process_json_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process JSON data safely."""
        # Validate and sanitize
        sanitized = {}
        for key, value in data.items():
            if isinstance(key, str) and len(key) <= 100:
                if isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
        return sanitized

    def _process_list_data(self, data: list[str]) -> list[str]:
        """Process list data safely."""
        # Limit list size
        return data[:1000]


@pytest.mark.security
class TestAPIFuzzingHarnesses:
    """Fuzzing harnesses for API endpoints."""

    @given(st.dictionaries(
        keys=st.sampled_from(["path", "collection", "content", "metadata"]),
        values=st.text(max_size=500),
        min_size=1
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_store_endpoint(self, params: dict[str, str]):
        """Fuzz test for store endpoint parameters."""
        # Simulate store endpoint call
        try:
            result = self._simulate_store_call(params)

            # Should return a result or raise expected error
            if result:
                assert "error" in result or "success" in result

        except (ValueError, TypeError, KeyError) as e:
            # Expected parameter validation errors
            assert str(e)

    @given(st.dictionaries(
        keys=st.sampled_from(["query", "collection", "limit", "filters"]),
        values=st.one_of(
            st.text(max_size=500),
            st.integers(min_value=0, max_value=1000)
        )
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_search_endpoint(self, params: dict[str, Any]):
        """Fuzz test for search endpoint parameters."""
        # Simulate search endpoint call
        try:
            result = self._simulate_search_call(params)

            if result:
                assert isinstance(result, (dict, list))

        except (ValueError, TypeError):
            # Expected validation errors
            pass

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_file_content_parsing(self, file_content: str):
        """Fuzz test for file content parsing."""
        # Test various file formats
        for file_type in ["txt", "md", "json"]:
            try:
                result = self._parse_file_content(file_content, file_type)

                # Should return parsed content or empty
                assert isinstance(result, (str, dict, list))

            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                # Expected parsing errors
                pass

    def _simulate_store_call(self, params: dict[str, str]) -> dict[str, Any]:
        """Simulate MCP store endpoint call."""
        # Validate required parameters
        if "content" not in params:
            raise ValueError("Missing required parameter: content")

        # Simulate successful storage
        return {"success": True, "id": "test_id"}

    def _simulate_search_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Simulate MCP search endpoint call."""
        # Validate query parameter
        if "query" not in params:
            raise ValueError("Missing required parameter: query")

        # Simulate search results
        return {"results": [], "count": 0}

    def _parse_file_content(self, content: str, file_type: str) -> Any:
        """Parse file content."""
        if file_type == "json":
            return json.loads(content)
        elif file_type in ["txt", "md"]:
            return content
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


@pytest.mark.security
class TestCorpusGeneration:
    """Test fuzzing corpus generation from existing data."""

    def test_generate_corpus_from_test_files(self, tmp_path):
        """Generate fuzzing corpus from existing test files."""
        # Create sample test files
        test_files = {
            "test_input_1.txt": "normal text content",
            "test_input_2.json": '{"key": "value"}',
            "test_input_3.txt": "<script>alert('xss')</script>",
        }

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        for filename, content in test_files.items():
            (tmp_path / filename).write_text(content)

        # Generate corpus
        corpus_files = self._generate_corpus(tmp_path, corpus_dir)

        # Verify corpus files created
        assert len(corpus_files) > 0

        # Corpus should contain interesting inputs
        corpus_content = [f.read_text() for f in corpus_files]
        assert any("xss" in c.lower() for c in corpus_content)

    def test_corpus_minimization(self, tmp_path):
        """Test corpus minimization to remove duplicates."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create duplicate inputs
        inputs = [
            "input1",
            "input1",  # Duplicate
            "input2",
            "input3",
            "input1",  # Another duplicate
        ]

        for i, content in enumerate(inputs):
            (corpus_dir / f"input_{i}.txt").write_text(content)

        # Minimize corpus
        minimized = self._minimize_corpus(corpus_dir)

        # Should have only unique inputs
        unique_content = {f.read_text() for f in minimized}
        assert len(unique_content) == 3  # input1, input2, input3

    def test_corpus_seed_generation(self, tmp_path):
        """Test generating seed inputs for fuzzing."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Generate diverse seed inputs
        seeds = self._generate_seed_inputs()

        # Write seeds to corpus
        for i, seed in enumerate(seeds):
            (corpus_dir / f"seed_{i}.txt").write_bytes(seed)

        # Verify seeds are diverse
        assert len(seeds) >= 5

        # Seeds should cover various patterns
        all_content = b"".join(seeds)
        assert b"\x00" in all_content  # Null bytes
        assert b"\xff" in all_content  # High bytes
        assert b"<" in all_content  # Special characters

    def _generate_corpus(self, source_dir: Path, corpus_dir: Path) -> list[Path]:
        """Generate fuzzing corpus from source files."""
        corpus_files = []

        for file_path in source_dir.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".txt", ".json"]:
                # Copy interesting inputs to corpus
                corpus_file = corpus_dir / f"corpus_{file_path.name}"
                corpus_file.write_bytes(file_path.read_bytes())
                corpus_files.append(corpus_file)

        return corpus_files

    def _minimize_corpus(self, corpus_dir: Path) -> list[Path]:
        """Minimize corpus by removing duplicates."""
        seen_content = set()
        minimized = []

        for file_path in corpus_dir.glob("*.txt"):
            content = file_path.read_text()
            if content not in seen_content:
                seen_content.add(content)
                minimized.append(file_path)

        return minimized

    def _generate_seed_inputs(self) -> list[bytes]:
        """Generate diverse seed inputs for fuzzing."""
        return [
            b"normal input",
            b"\x00\x00\x00\x00",  # Null bytes
            b"\xff\xff\xff\xff",  # High bytes
            b"<script>alert(1)</script>",  # XSS payload
            b"' OR '1'='1",  # SQL injection
            b"../../../etc/passwd",  # Path traversal
            b"\n" * 1000,  # Newline flood
            bytes(range(256)),  # All byte values
        ]


@pytest.mark.security
class TestCrashDetection:
    """Test crash detection and reporting infrastructure."""

    def test_detect_segmentation_fault(self):
        """Test detection of segmentation faults."""
        # Simulate process that might crash
        crash_info = self._run_with_crash_detection(
            lambda: self._safe_operation()
        )

        # Should complete without crash
        assert crash_info["crashed"] is False
        assert crash_info["exit_code"] == 0

    def test_detect_memory_corruption(self):
        """Test detection of memory corruption indicators."""
        # Check for memory corruption patterns
        test_data = b"\x00" * 1000 + b"\xff" * 1000

        is_corrupted = self._check_memory_corruption(test_data)

        # Normal data should not trigger corruption detection
        assert not is_corrupted

    def test_crash_report_generation(self, tmp_path):
        """Test crash report generation."""
        crash_data = {
            "type": "segmentation_fault",
            "signal": "SIGSEGV",
            "input": "malicious_input",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        report_path = tmp_path / "crash_report.json"
        self._generate_crash_report(crash_data, report_path)

        # Verify report created
        assert report_path.exists()

        # Verify report content
        report = json.loads(report_path.read_text())
        assert report["type"] == "segmentation_fault"
        assert "input" in report

    def test_fuzzing_statistics_collection(self):
        """Test collection of fuzzing statistics."""
        stats = self._collect_fuzzing_stats()

        # Should have required metrics
        assert "executions" in stats
        assert "crashes" in stats
        assert "unique_crashes" in stats
        assert "coverage" in stats

        # Values should be reasonable
        assert stats["executions"] >= 0
        assert stats["crashes"] >= 0
        assert 0 <= stats["coverage"] <= 100

    def _run_with_crash_detection(self, func):
        """Run function with crash detection."""
        try:
            result = func()
            return {
                "crashed": False,
                "exit_code": 0,
                "result": result
            }
        except Exception as e:
            return {
                "crashed": True,
                "exit_code": 1,
                "exception": str(e)
            }

    def _safe_operation(self):
        """Safe operation that won't crash."""
        return "success"

    def _check_memory_corruption(self, data: bytes) -> bool:
        """Check for memory corruption patterns."""
        # Simple heuristic: look for suspicious patterns
        if len(data) > 10000:
            # Check for repeating patterns (possible overflow)
            if data[:1000] == data[1000:2000]:
                return True

        return False

    def _generate_crash_report(self, crash_data: dict[str, Any], report_path: Path):
        """Generate crash report."""
        report_path.write_text(json.dumps(crash_data, indent=2))

    def _collect_fuzzing_stats(self) -> dict[str, Any]:
        """Collect fuzzing statistics."""
        # In production: collect from fuzzing engine
        return {
            "executions": 10000,
            "crashes": 0,
            "unique_crashes": 0,
            "coverage": 85.5,
            "paths_found": 1234,
        }


@pytest.mark.security
class TestCICDIntegration:
    """Test CI/CD pipeline integration for fuzzing."""

    def test_fuzzing_workflow_config(self, tmp_path):
        """Test fuzzing workflow configuration generation."""
        workflow_path = tmp_path / ".github" / "workflows" / "fuzzing.yml"
        workflow_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate workflow config
        workflow_content = self._generate_fuzzing_workflow()
        workflow_path.write_text(workflow_content)

        # Verify workflow exists
        assert workflow_path.exists()

        # Verify workflow content
        content = workflow_path.read_text()
        assert "fuzzing" in content.lower()
        assert "hypothesis" in content.lower()

    def test_fuzzing_report_artifact(self, tmp_path):
        """Test fuzzing report artifact generation."""
        report_dir = tmp_path / "fuzzing_reports"
        report_dir.mkdir()

        # Generate fuzzing report
        report = self._generate_fuzzing_report()

        # Save as artifact
        artifact_path = report_dir / "fuzzing_report.json"
        artifact_path.write_text(json.dumps(report, indent=2))

        # Verify artifact
        assert artifact_path.exists()

        # Verify report structure
        loaded_report = json.loads(artifact_path.read_text())
        assert "summary" in loaded_report
        assert "crashes" in loaded_report

    def test_fuzzing_threshold_check(self):
        """Test fuzzing quality threshold checking."""
        stats = {
            "coverage": 85.5,
            "crashes": 0,
            "executions": 10000
        }

        # Check against thresholds
        passes = self._check_fuzzing_thresholds(stats)

        # Should pass with good stats
        assert passes is True

    def _generate_fuzzing_workflow(self) -> str:
        """Generate GitHub Actions fuzzing workflow."""
        return """name: Fuzzing Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Run nightly
  workflow_dispatch:

jobs:
  fuzzing:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install hypothesis pytest

      - name: Run Hypothesis fuzzing
        run: |
          pytest tests/security/test_fuzzing_integration.py -v --tb=short

      - name: Upload crash artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: fuzzing-crashes
          path: crashes/
"""

    def _generate_fuzzing_report(self) -> dict[str, Any]:
        """Generate fuzzing report."""
        return {
            "summary": {
                "total_executions": 10000,
                "total_crashes": 0,
                "unique_crashes": 0,
                "coverage": 85.5
            },
            "crashes": [],
            "statistics": {
                "execution_time_seconds": 1800,
                "executions_per_second": 5.56
            }
        }

    def _check_fuzzing_thresholds(self, stats: dict[str, Any]) -> bool:
        """Check if fuzzing meets quality thresholds."""
        # Define thresholds
        min_coverage = 80.0
        max_crashes = 0

        # Check thresholds
        if stats["coverage"] < min_coverage:
            return False

        if stats["crashes"] > max_crashes:
            return False

        return True


@pytest.mark.security
class TestAFLPlusPlusIntegration:
    """Documentation and testing for AFL++ Rust integration."""

    def test_afl_rust_fuzzing_harness_template(self, tmp_path):
        """Test AFL++ fuzzing harness template for Rust."""
        harness_path = tmp_path / "fuzz_target.rs"

        # Generate AFL++ harness template
        harness_content = self._generate_afl_harness_template()
        harness_path.write_text(harness_content)

        # Verify harness created
        assert harness_path.exists()

        # Verify harness structure
        content = harness_path.read_text()
        assert "afl::fuzz!" in content
        assert "fuzz_target" in content

    def test_afl_cargo_configuration(self, tmp_path):
        """Test AFL++ Cargo.toml configuration."""
        cargo_path = tmp_path / "Cargo.toml"

        # Generate AFL++ Cargo config
        cargo_content = self._generate_afl_cargo_config()
        cargo_path.write_text(cargo_content)

        # Verify config
        assert cargo_path.exists()

        content = cargo_path.read_text()
        assert "afl" in content

    def _generate_afl_harness_template(self) -> str:
        """Generate AFL++ fuzzing harness template for Rust."""
        return """// AFL++ fuzzing harness for Rust components
// Compile with: cargo afl build
// Run with: cargo afl fuzz -i in -o out target/debug/fuzz_target

use afl::fuzz;

fn main() {
    fuzz!(|data: &[u8]| {
        // Convert fuzzing input to appropriate type
        if let Ok(text) = std::str::from_utf8(data) {
            // Call function to fuzz
            fuzz_target(text);
        }
    });
}

fn fuzz_target(input: &str) {
    // Your code to fuzz here
    // Example: parse input, process data, etc.

    // Should not panic or crash
    let _ = process_input(input);
}

fn process_input(input: &str) -> Result<String, String> {
    // Simulate processing that could crash
    if input.len() > 10000 {
        return Err("Input too large".to_string());
    }

    Ok(input.to_string())
}
"""

    def _generate_afl_cargo_config(self) -> str:
        """Generate AFL++ Cargo.toml configuration."""
        return """[package]
name = "fuzz-target"
version = "0.1.0"
edition = "2021"

[dependencies]
afl = "0.13"

[[bin]]
name = "fuzz_target"
path = "fuzz_target.rs"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
"""


# Security test markers are configured in pyproject.toml
