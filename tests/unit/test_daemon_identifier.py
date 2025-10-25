"""Unit tests for DaemonIdentifier in project detection utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from workspace_qdrant_mcp.utils.project_detection import DaemonIdentifier


@pytest.fixture(autouse=True)
def stub_project_detection_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the module logger with a mock that accepts structured kwargs."""
    monkeypatch.setattr(
        "workspace_qdrant_mcp.utils.project_detection.logger",
        MagicMock(),
        raising=True,
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Ensure the DaemonIdentifier registry is isolated between tests."""
    DaemonIdentifier.clear_registry()
    yield
    DaemonIdentifier.clear_registry()


def test_generate_identifier_registers_project(tmp_path: Path) -> None:
    """Generating an identifier stores it in the registry with path metadata."""
    project_path = tmp_path / "example"
    project_path.mkdir()

    identifier = DaemonIdentifier("my-project", str(project_path))
    value = identifier.generate_identifier()

    expected_hash = hashlib.sha256(str(project_path.resolve()).encode("utf-8")).hexdigest()[:8]
    assert value == f"my-project_{expected_hash}"
    assert identifier.get_identifier() == value
    assert identifier.get_path_hash() == expected_hash

    active = DaemonIdentifier.get_active_identifiers()
    assert value in active

    info = DaemonIdentifier.get_identifier_info(value)
    assert info is not None
    assert info["project_name"] == "my-project"
    assert Path(info["project_path"]) == project_path.resolve()


def test_release_identifier_removes_from_registry(tmp_path: Path) -> None:
    """Releasing an identifier removes it from active tracking structures."""
    project_path = tmp_path / "release"
    project_path.mkdir()

    identifier = DaemonIdentifier("release-project", str(project_path))
    value = identifier.generate_identifier()
    assert value in DaemonIdentifier.get_active_identifiers()

    identifier.release_identifier()
    assert value not in DaemonIdentifier.get_active_identifiers()
    assert DaemonIdentifier.get_identifier_info(value) is None


def test_collision_extends_hash_length(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When a collision occurs the identifier increases the hash length."""
    first = DaemonIdentifier("collision", str(tmp_path / "first"))
    second = DaemonIdentifier("collision", str(tmp_path / "second"))

    def fake_hash(self, _path: str, length: int) -> str:  # type: ignore[override]
        # Return the same short hash to trigger collision, then a longer unique hash.
        return {8: "deadbeef", 12: "deadbeefcafe"}.get(length, "deadbeefcafefeed")[:length]

    monkeypatch.setattr(DaemonIdentifier, "_generate_path_hash", fake_hash, raising=True)

    first_value = first.generate_identifier()
    second_value = second.generate_identifier()

    assert first_value == "collision_deadbeef"
    assert second_value == "collision_deadbeefcafe"
    assert second.get_path_hash() == "deadbeefcafe"


def test_validate_identifier_uses_expected_path_hash(tmp_path: Path) -> None:
    """Validation succeeds only when the hash matches the project path."""
    project_path = tmp_path / "validate"
    project_path.mkdir()

    identifier = DaemonIdentifier("validate", str(project_path))
    value = identifier.generate_identifier()

    assert identifier.validate_identifier(value) is True
    assert identifier.validate_identifier("validate_deadbeef") is False
