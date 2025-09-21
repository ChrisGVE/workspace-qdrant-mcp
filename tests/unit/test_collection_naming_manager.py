"""Unit tests for the CollectionNamingManager."""

from __future__ import annotations

import pytest

from common.core.collection_naming import (
    CollectionNamingManager,
    CollectionType,
)


@pytest.fixture
def manager() -> CollectionNamingManager:
    """Provide a manager with predictable suffixes and legacy collections."""
    return CollectionNamingManager(
        global_collections=["legacy-global"],
        valid_project_suffixes=["docs", "notes"],
    )


def test_empty_name_rejected(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("   ")
    assert result.is_valid is False
    assert result.error_message == "Collection name cannot be empty"


def test_reserved_name_rejected(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("_memory")
    assert result.is_valid is False
    assert "reserved" in result.error_message


def test_memory_collection_valid(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("memory")
    assert result.is_valid is True
    assert result.collection_info.collection_type is CollectionType.MEMORY
    assert result.collection_info.is_readonly_from_mcp is False


def test_system_memory_validation(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("__user_rules")
    assert result.is_valid is True
    info = result.collection_info
    assert info.collection_type is CollectionType.SYSTEM_MEMORY
    assert info.display_name == "user_rules"
    assert info.is_readonly_from_mcp is True


def test_invalid_system_memory_rejected(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("__1invalid")
    assert result.is_valid is False
    assert "System memory name '1invalid'" in result.error_message


def test_library_collection_validation(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("_library")
    assert result.is_valid is True
    info = result.collection_info
    assert info.collection_type is CollectionType.LIBRARY
    assert info.display_name == "library"
    assert info.is_readonly_from_mcp is True


def test_invalid_library_rejected(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("_")
    assert result.is_valid is False
    assert "Library name cannot be empty" in result.error_message


def test_project_collection_validation(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("sample-docs")
    assert result.is_valid is True
    info = result.collection_info
    assert info.collection_type is CollectionType.PROJECT
    assert info.project_name == "sample"
    assert info.collection_suffix == "docs"


def test_project_suffix_must_be_valid(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name("sample-code")
    assert result.is_valid is False
    assert "Invalid project collection suffix 'code'" in result.error_message


def test_intended_type_mismatch_reports_error(manager: CollectionNamingManager) -> None:
    result = manager.validate_collection_name(
        "unknown-set", intended_type=CollectionType.PROJECT
    )
    assert result.is_valid is False
    assert "suggests type" in result.error_message


def test_check_naming_conflicts_with_existing_library(manager: CollectionNamingManager) -> None:
    result = manager.check_naming_conflicts("_reports", ["reports"])
    assert result.is_valid is False
    assert "naming conflict" in result.error_message


def test_check_naming_conflicts_with_existing_display_name(manager: CollectionNamingManager) -> None:
    result = manager.check_naming_conflicts("reports", ["_reports"])
    assert result.is_valid is False
    assert "library collection '_reports'" in result.error_message


def test_check_naming_conflicts_allows_unique_name(manager: CollectionNamingManager) -> None:
    result = manager.check_naming_conflicts("unique", ["existing", "_other"])
    assert result.is_valid is True
    assert result.collection_info.display_name == "unique"


def test_get_display_and_actual_name_roundtrip(manager: CollectionNamingManager) -> None:
    display = manager.get_display_name("_knowledge")
    assert display == "knowledge"
    actual = manager.get_actual_name(display, CollectionType.LIBRARY)
    assert actual == "_knowledge"
    system_actual = manager.get_actual_name("rules", CollectionType.SYSTEM_MEMORY)
    assert system_actual == "__rules"


def test_is_readonly_and_system_memory_helpers(manager: CollectionNamingManager) -> None:
    assert manager.is_mcp_readonly("_library") is True
    assert manager.is_mcp_readonly("memory") is False
    assert manager.is_system_memory_collection("__user_rules") is True
    assert manager.is_system_memory_collection("memory") is False


def test_filter_workspace_collections(manager: CollectionNamingManager) -> None:
    collections = [
        "memory",
        "__user_rules",
        "_library",
        "project-docs",
        "project-code",  # should be excluded (memexd pattern)
        "legacy-global",
    ]
    filtered = manager.filter_workspace_collections(collections)
    expected = sorted(["memory", "__user_rules", "_library", "project-docs", "legacy-global"])
    # project-code filtered, others retained and sorted
    assert filtered == expected


def test_generate_project_collection_names(manager: CollectionNamingManager) -> None:
    names = set(manager.generate_project_collection_names("workspace"))
    assert names == {"workspace-docs", "workspace-notes"}


def test_get_collection_info_legacy(manager: CollectionNamingManager) -> None:
    info = manager.get_collection_info("global-archive")
    assert info.collection_type is CollectionType.LEGACY
    assert info.display_name == "global-archive"
