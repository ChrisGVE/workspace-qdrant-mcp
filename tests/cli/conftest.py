"""
CLI test fixtures and configuration.

Provides fixtures specific to CLI testing including command
execution, output capture, and configuration management.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Provide Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_config_dir(tmp_path: Path) -> Path:
    """Provide temporary configuration directory for CLI tests."""
    config_dir = tmp_path / ".wqm"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def cli_test_environment(cli_config_dir: Path, tmp_path: Path) -> dict[str, str]:
    """Provide test environment variables for CLI."""
    return {
        "WQM_CONFIG_DIR": str(cli_config_dir),
        "WQM_DATA_DIR": str(tmp_path / "data"),
        "QDRANT_URL": "http://localhost:6333",
        "WQM_TEST_MODE": "true",
    }


@pytest.fixture
def cli_config_file(cli_config_dir: Path) -> Path:
    """Provide test configuration file."""
    config_path = cli_config_dir / "config.yaml"
    config_content = """
qdrant:
  url: http://localhost:6333
  api_key: null
  timeout: 30

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32

collections:
  - project

global_collections:
  - global

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_cli_commands() -> list[dict[str, Any]]:
    """Provide sample CLI commands for testing."""
    return [
        {
            "command": "wqm status",
            "description": "Show server status",
            "expected_exit_code": 0,
        },
        {
            "command": "wqm collections list",
            "description": "List all collections",
            "expected_exit_code": 0,
        },
        {
            "command": "wqm ingest --path ./docs",
            "description": "Ingest documents from directory",
            "expected_exit_code": 0,
        },
        {
            "command": "wqm search --query 'test'",
            "description": "Search for documents",
            "expected_exit_code": 0,
        },
    ]


@pytest.fixture
def mock_cli_output():
    """Provide mock CLI output capture utilities."""

    class MockOutput:
        """Capture and validate CLI output."""

        def __init__(self):
            self.stdout = []
            self.stderr = []

        def capture_stdout(self, text: str):
            """Capture stdout output."""
            self.stdout.append(text)

        def capture_stderr(self, text: str):
            """Capture stderr output."""
            self.stderr.append(text)

        def get_stdout(self) -> str:
            """Get combined stdout."""
            return "\n".join(self.stdout)

        def get_stderr(self) -> str:
            """Get combined stderr."""
            return "\n".join(self.stderr)

        def contains_stdout(self, text: str) -> bool:
            """Check if stdout contains text."""
            return text in self.get_stdout()

        def contains_stderr(self, text: str) -> bool:
            """Check if stderr contains text."""
            return text in self.get_stderr()

    return MockOutput()


@pytest.fixture
def test_documents_dir(tmp_path: Path) -> Path:
    """Create temporary directory with test documents."""
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()

    # Create sample documents
    (docs_dir / "readme.md").write_text("# Test Project\n\nTest documentation.")
    (docs_dir / "api.md").write_text("# API Reference\n\nAPI documentation.")
    (docs_dir / "guide.md").write_text("# User Guide\n\nUser guide content.")

    # Create subdirectory
    sub_dir = docs_dir / "advanced"
    sub_dir.mkdir()
    (sub_dir / "advanced.md").write_text("# Advanced Topics\n\nAdvanced content.")

    return docs_dir


@pytest.fixture
def cli_ingestion_config() -> dict[str, Any]:
    """Provide ingestion configuration for testing."""
    return {
        "batch_size": 10,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "file_patterns": ["*.md", "*.txt", "*.py"],
        "exclude_patterns": ["*.pyc", "__pycache__", ".git"],
        "recursive": True,
        "follow_symlinks": False,
    }


@pytest.fixture
def cli_search_config() -> dict[str, Any]:
    """Provide search configuration for testing."""
    return {
        "default_limit": 10,
        "max_limit": 100,
        "min_score": 0.5,
        "use_hybrid": True,
        "rerank": False,
    }


@pytest.fixture
def cli_output_formatter():
    """Provide CLI output formatting utilities."""

    class OutputFormatter:
        """Format CLI output for different formats."""

        @staticmethod
        def format_json(data: Any) -> str:
            """Format data as JSON."""
            import json

            return json.dumps(data, indent=2)

        @staticmethod
        def format_table(data: list[dict[str, Any]], headers: list[str]) -> str:
            """Format data as table."""
            from io import StringIO

            from rich.console import Console
            from rich.table import Table

            table = Table()
            for header in headers:
                table.add_column(header)

            for row in data:
                table.add_row(*[str(row.get(h, "")) for h in headers])

            console = Console(file=StringIO())
            console.print(table)
            return console.file.getvalue()

        @staticmethod
        def format_yaml(data: Any) -> str:
            """Format data as YAML."""
            import yaml

            return yaml.dump(data, default_flow_style=False)

    return OutputFormatter()


@pytest.fixture
async def mock_cli_services():
    """Provide mock services for CLI testing."""

    class MockServices:
        """Mock external services for CLI tests."""

        def __init__(self):
            self.qdrant_available = True
            self.daemon_available = False
            self.network_available = True

        def set_qdrant_available(self, available: bool):
            """Set Qdrant availability."""
            self.qdrant_available = available

        def set_daemon_available(self, available: bool):
            """Set daemon availability."""
            self.daemon_available = available

        def set_network_available(self, available: bool):
            """Set network availability."""
            self.network_available = available

        def check_qdrant(self) -> bool:
            """Check if Qdrant is available."""
            return self.qdrant_available

        def check_daemon(self) -> bool:
            """Check if daemon is available."""
            return self.daemon_available

        def check_network(self) -> bool:
            """Check if network is available."""
            return self.network_available

    return MockServices()


@pytest.fixture
def cli_performance_monitor():
    """Provide performance monitoring for CLI operations."""

    class PerformanceMonitor:
        """Monitor CLI command performance."""

        def __init__(self):
            self.command_times = {}

        def record_command(self, command: str, duration_ms: float):
            """Record command execution time."""
            if command not in self.command_times:
                self.command_times[command] = []
            self.command_times[command].append(duration_ms)

        def get_average_time(self, command: str) -> float:
            """Get average execution time for command."""
            times = self.command_times.get(command, [])
            return sum(times) / len(times) if times else 0.0

        def get_slowest_command(self) -> tuple:
            """Get slowest command and its time."""
            if not self.command_times:
                return None, 0.0

            slowest = max(
                self.command_times.items(), key=lambda x: max(x[1] if x[1] else [0])
            )
            return slowest[0], max(slowest[1]) if slowest[1] else 0.0

    return PerformanceMonitor()
