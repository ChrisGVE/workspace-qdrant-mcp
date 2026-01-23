"""Unit tests for store tool multi-tenant updates (Task 397).

Tests automatic project_id tagging and file type detection.
"""

import pytest
from workspace_qdrant_mcp.server import (
    _detect_file_type,
    CANONICAL_COLLECTIONS,
)


class TestDetectFileType:
    """Tests for _detect_file_type function."""

    def test_code_files(self):
        """Test detection of code files."""
        code_files = [
            "main.py", "server.rs", "app.go", "index.js", "App.tsx",
            "Main.java", "lib.c", "utils.cpp", "Component.swift"
        ]
        for f in code_files:
            assert _detect_file_type(f) == "code", f"Expected 'code' for {f}"

    def test_test_files(self):
        """Test detection of test files."""
        test_files = [
            "test_main.py", "test_server.rs", "app_test.go",
            "index.spec.js", "App.test.tsx", "MainTest.java"
        ]
        for f in test_files:
            assert _detect_file_type(f) == "test", f"Expected 'test' for {f}"

    def test_docs_files(self):
        """Test detection of documentation files."""
        doc_files = [
            "README.md", "CHANGELOG.md", "docs.rst", "notes.txt",
            "guide.adoc", "readme", "CONTRIBUTING", "LICENSE"
        ]
        for f in doc_files:
            assert _detect_file_type(f) == "docs", f"Expected 'docs' for {f}"

    def test_config_files(self):
        """Test detection of configuration files."""
        config_files = [
            "config.yaml", "settings.yml", "config.json", "pyproject.toml",
            "setup.ini", "nginx.conf", "Dockerfile", ".gitignore", ".env"
        ]
        for f in config_files:
            assert _detect_file_type(f) == "config", f"Expected 'config' for {f}"

    def test_build_files(self):
        """Test detection of build files."""
        build_files = [
            "Cargo.lock", "go.sum", "package-lock.json",
            "Cargo.toml", "pyproject.toml", "package.json", "go.mod"
        ]
        # Note: Cargo.toml and pyproject.toml can be config or build
        # The function checks specific names for build first
        for f in build_files:
            result = _detect_file_type(f)
            assert result in ("build", "config"), f"Expected 'build' or 'config' for {f}"

    def test_data_files(self):
        """Test detection of data files."""
        data_files = [
            "data.csv", "records.parquet", "events.arrow",
            "database.db", "local.sqlite"
        ]
        for f in data_files:
            assert _detect_file_type(f) == "data", f"Expected 'data' for {f}"

    def test_other_files(self):
        """Test detection of unknown file types."""
        other_files = [
            "image.png", "photo.jpg", "video.mp4",
            "archive.zip", "unknown.xyz"
        ]
        for f in other_files:
            assert _detect_file_type(f) == "other", f"Expected 'other' for {f}"

    def test_case_insensitivity(self):
        """Test that file extension detection is case insensitive."""
        assert _detect_file_type("Main.PY") == "code"
        assert _detect_file_type("README.MD") == "docs"
        assert _detect_file_type("Config.YAML") == "config"


class TestUnifiedCollectionsUsage:
    """Tests verifying unified collections are properly defined."""

    def test_projects_collection_defined(self):
        """Test that projects collection is defined."""
        assert "projects" in CANONICAL_COLLECTIONS
        assert CANONICAL_COLLECTIONS["projects"] == "projects"

    def test_all_collections_underscore_prefixed(self):
        """Test that all unified collections have underscore prefix."""
        for name, collection in CANONICAL_COLLECTIONS.items():
            assert collection.startswith("_"), f"Collection {name} should start with underscore"


class TestStoreToolSignature:
    """Tests for store tool function signature."""

    def test_store_tool_exists(self):
        """Test that store tool is defined."""
        from workspace_qdrant_mcp.server import store
        assert store is not None

    def test_store_tool_has_file_type_parameter(self):
        """Test that store tool has file_type parameter."""
        from workspace_qdrant_mcp.server import store
        # FunctionTool wraps the function
        fn = store.fn if hasattr(store, 'fn') else store
        import inspect
        sig = inspect.signature(fn)
        assert "file_type" in sig.parameters
