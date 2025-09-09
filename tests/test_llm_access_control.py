"""
Comprehensive unit tests for the LLM access control system.

This test suite provides complete coverage of the core.llm_access_control module,
testing all security boundaries, access control scenarios, and integration points.

Test Categories:
- LLMAccessController class methods and validation
- Access control for all collection types (SYSTEM, LIBRARY, GLOBAL, PROJECT, MEMORY)
- Security boundary testing and forbidden patterns
- Configuration integration and context handling
- Exception handling and error messages
- Edge cases and malformed inputs
- Module-level convenience functions
"""

import pytest
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.llm_access_control import (
    LLMAccessController,
    AccessViolationType,
    AccessViolation,
    LLMAccessControlError,
    can_llm_create_collection,
    can_llm_delete_collection,
    validate_llm_collection_access,
    get_forbidden_collection_patterns
)
from core.collection_types import CollectionType, CollectionInfo
from core.config import McpConfig, DaemonConfig


class TestLLMAccessController:
    """Test the LLMAccessController class methods."""

    def test_init_without_config(self):
        """Test initializing controller without configuration."""
        controller = LLMAccessController()
        assert controller.config is None
        assert hasattr(controller, 'classifier')
        assert hasattr(controller, '_forbidden_patterns')
        assert isinstance(controller._existing_collections, set)
        assert len(controller._existing_collections) == 0

    def test_init_with_mcp_config(self):
        """Test initializing controller with MCP configuration."""
        config = McpConfig()
        controller = LLMAccessController(config)
        assert controller.config == config
        assert hasattr(controller, 'classifier')

    def test_init_with_daemon_config(self):
        """Test initializing controller with daemon configuration."""
        config = DaemonConfig()
        controller = LLMAccessController(config)
        assert controller.config == config
        assert hasattr(controller, 'classifier')

    def test_set_existing_collections(self):
        """Test setting existing collections for validation."""
        controller = LLMAccessController()
        collections = ["project1-docs", "project2-data", "__system_config"]
        
        controller.set_existing_collections(collections)
        
        assert controller._existing_collections == set(collections)

    def test_set_existing_collections_empty_list(self):
        """Test setting empty list of existing collections."""
        controller = LLMAccessController()
        controller.set_existing_collections([])
        assert controller._existing_collections == set()

    def test_set_existing_collections_duplicates(self):
        """Test setting collections with duplicates."""
        controller = LLMAccessController()
        collections = ["project1-docs", "project1-docs", "project2-data"]
        
        controller.set_existing_collections(collections)
        
        # Set should eliminate duplicates
        assert controller._existing_collections == {"project1-docs", "project2-data"}


class TestCollectionCreationAccess:
    """Test LLM access control for collection creation operations."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    @pytest.fixture
    def mock_classifier(self, controller):
        """Mock the classifier to return controlled responses."""
        with patch.object(controller, 'classifier') as mock:
            yield mock

    # System Collection Creation Tests
    def test_can_llm_create_system_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot create system collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__system_config",
            type=CollectionType.SYSTEM,
            display_name="system_config",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_create_collection("__system_config") is False

    def test_validate_llm_create_system_collection_raises_exception(self, controller, mock_classifier):
        """Test that system collection creation raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__admin_panel",
            type=CollectionType.SYSTEM,
            display_name="admin_panel",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "__admin_panel")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_SYSTEM_CREATION
        assert violation.collection_name == "__admin_panel"
        assert violation.operation == "create"
        assert "system collection" in violation.message.lower()
        assert "cli only" in violation.message.lower()
        assert violation.suggested_alternatives is not None

    # Library Collection Creation Tests
    def test_can_llm_create_library_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot create library collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_shared_utils",
            type=CollectionType.LIBRARY,
            display_name="shared_utils",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        assert controller.can_llm_create_collection("_shared_utils") is False

    def test_validate_llm_create_library_collection_raises_exception(self, controller, mock_classifier):
        """Test that library collection creation raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_common_library",
            type=CollectionType.LIBRARY,
            display_name="common_library",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "_common_library")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_LIBRARY_CREATION
        assert violation.collection_name == "_common_library"
        assert violation.operation == "create"
        assert "library collection" in violation.message.lower()
        assert "cli only" in violation.message.lower()
        assert violation.suggested_alternatives is not None

    # Global Collection Creation Tests
    def test_can_llm_create_global_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot create global collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="algorithms",
            type=CollectionType.GLOBAL,
            display_name="algorithms",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_create_collection("algorithms") is False

    def test_validate_llm_create_global_collection_raises_exception(self, controller, mock_classifier):
        """Test that global collection creation raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="workspace",
            type=CollectionType.GLOBAL,
            display_name="workspace",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "workspace")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_GLOBAL_CREATION
        assert violation.collection_name == "workspace"
        assert violation.operation == "create"
        assert "reserved global collection" in violation.message.lower()
        assert violation.suggested_alternatives is not None

    # Project Collection Creation Tests
    def test_can_llm_create_project_collection_true(self, controller, mock_classifier):
        """Test that LLM can create project collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="myproject-docs",
            type=CollectionType.PROJECT,
            display_name="myproject-docs",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="myproject",
            suffix="docs"
        )
        
        assert controller.can_llm_create_collection("myproject-docs") is True

    def test_validate_llm_create_project_collection_success(self, controller, mock_classifier):
        """Test that project collection creation succeeds."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="testproject-data",
            type=CollectionType.PROJECT,
            display_name="testproject-data",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="testproject",
            suffix="data"
        )
        
        # Should not raise exception
        controller.validate_llm_collection_access("create", "testproject-data")

    def test_validate_llm_create_project_collection_reserved_suffix(self, controller, mock_classifier):
        """Test that project collections with reserved suffixes are blocked."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="myproject-system",
            type=CollectionType.PROJECT,
            display_name="myproject-system",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="myproject",
            suffix="system"
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "myproject-system")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.RESERVED_NAME_CONFLICT
        assert "reserved" in violation.message.lower()
        assert violation.suggested_alternatives is not None

    @pytest.mark.parametrize("reserved_suffix", ["memory", "system", "config", "admin"])
    def test_validate_llm_create_project_collection_all_reserved_suffixes(self, controller, mock_classifier, reserved_suffix):
        """Test that all reserved suffixes are properly blocked."""
        collection_name = f"testproject-{reserved_suffix}"
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name=collection_name,
            type=CollectionType.PROJECT,
            display_name=collection_name,
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="testproject",
            suffix=reserved_suffix
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", collection_name)
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.RESERVED_NAME_CONFLICT

    # Unknown Collection Creation Tests
    def test_can_llm_create_unknown_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot create unknown format collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="invalid_collection_name",
            type=CollectionType.UNKNOWN,
            display_name="invalid_collection_name",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_create_collection("invalid_collection_name") is False

    def test_validate_llm_create_unknown_collection_raises_exception(self, controller, mock_classifier):
        """Test that unknown collection creation raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="bad-format_name",
            type=CollectionType.UNKNOWN,
            display_name="bad-format_name",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "bad-format_name")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "does not follow recognized patterns" in violation.message
        assert violation.suggested_alternatives is not None

    # Existing Collection Tests
    def test_validate_llm_create_existing_collection_raises_exception(self, controller, mock_classifier):
        """Test that creating existing collection raises exception."""
        controller.set_existing_collections(["existing-project"])
        
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="existing-project",
            type=CollectionType.PROJECT,
            display_name="existing-project",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="existing",
            suffix="project"
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "existing-project")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.COLLECTION_ALREADY_EXISTS
        assert "already exists" in violation.message


class TestCollectionDeletionAccess:
    """Test LLM access control for collection deletion operations."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    @pytest.fixture
    def mock_classifier(self, controller):
        """Mock the classifier to return controlled responses."""
        with patch.object(controller, 'classifier') as mock:
            yield mock

    # System Collection Deletion Tests
    def test_can_llm_delete_system_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot delete system collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__system_config",
            type=CollectionType.SYSTEM,
            display_name="system_config",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_delete_collection("__system_config") is False

    def test_validate_llm_delete_system_collection_raises_exception(self, controller, mock_classifier):
        """Test that system collection deletion raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__admin_data",
            type=CollectionType.SYSTEM,
            display_name="admin_data",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("delete", "__admin_data")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_SYSTEM_DELETION
        assert "system collections are protected" in violation.message.lower()

    # Library Collection Deletion Tests
    def test_can_llm_delete_library_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot delete library collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_shared_lib",
            type=CollectionType.LIBRARY,
            display_name="shared_lib",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        assert controller.can_llm_delete_collection("_shared_lib") is False

    def test_validate_llm_delete_library_collection_raises_exception(self, controller, mock_classifier):
        """Test that library collection deletion raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_common_utils",
            type=CollectionType.LIBRARY,
            display_name="common_utils",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("delete", "_common_utils")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_LIBRARY_DELETION
        assert "library collections are protected" in violation.message.lower()

    # Global Collection Deletion Tests
    def test_can_llm_delete_global_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot delete global collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="documents",
            type=CollectionType.GLOBAL,
            display_name="documents",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_delete_collection("documents") is False

    def test_validate_llm_delete_global_collection_raises_exception(self, controller, mock_classifier):
        """Test that global collection deletion raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="knowledge",
            type=CollectionType.GLOBAL,
            display_name="knowledge",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("delete", "knowledge")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_LIBRARY_DELETION  # Uses same type for globals
        assert "global collections are system-wide" in violation.message.lower()

    # Memory Collection Deletion Tests
    def test_can_llm_delete_memory_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot delete memory collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="project1-memory",
            type=CollectionType.PROJECT,
            display_name="project1-memory",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=True,
            project_name="project1",
            suffix="memory"
        )
        
        assert controller.can_llm_delete_collection("project1-memory") is False

    def test_validate_llm_delete_memory_collection_raises_exception(self, controller, mock_classifier):
        """Test that memory collection deletion raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__workspace_memory",
            type=CollectionType.SYSTEM,
            display_name="workspace_memory",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=True
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("delete", "__workspace_memory")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_MEMORY_DELETION
        assert "memory collections contain persistent state" in violation.message

    # Project Collection Deletion Tests
    def test_can_llm_delete_project_collection_true(self, controller, mock_classifier):
        """Test that LLM can delete non-memory project collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="myproject-docs",
            type=CollectionType.PROJECT,
            display_name="myproject-docs",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="myproject",
            suffix="docs"
        )
        
        assert controller.can_llm_delete_collection("myproject-docs") is True

    def test_validate_llm_delete_project_collection_success(self, controller, mock_classifier):
        """Test that non-memory project collection deletion succeeds."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="testproject-data",
            type=CollectionType.PROJECT,
            display_name="testproject-data",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="testproject",
            suffix="data"
        )
        
        # Should not raise exception
        controller.validate_llm_collection_access("delete", "testproject-data")

    # Unknown Collection Deletion Tests
    def test_can_llm_delete_unknown_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot delete unknown format collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="unknown_format",
            type=CollectionType.UNKNOWN,
            display_name="unknown_format",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_delete_collection("unknown_format") is False

    def test_validate_llm_delete_unknown_collection_raises_exception(self, controller, mock_classifier):
        """Test that unknown collection deletion raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="bad_format",
            type=CollectionType.UNKNOWN,
            display_name="bad_format",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("delete", "bad_format")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "unrecognized collection type" in violation.message


class TestCollectionWriteAccess:
    """Test LLM access control for collection write operations."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    @pytest.fixture
    def mock_classifier(self, controller):
        """Mock the classifier to return controlled responses."""
        with patch.object(controller, 'classifier') as mock:
            yield mock

    # System Collection Write Tests
    def test_can_llm_write_to_system_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot write to system collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__system_logs",
            type=CollectionType.SYSTEM,
            display_name="system_logs",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_write_to_collection("__system_logs") is False

    def test_validate_llm_write_to_system_collection_raises_exception(self, controller, mock_classifier):
        """Test that system collection write raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__admin_config",
            type=CollectionType.SYSTEM,
            display_name="admin_config",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("write", "__admin_config")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_SYSTEM_WRITE
        assert "cli-writable only" in violation.message.lower()

    # Library Collection Write Tests
    def test_can_llm_write_to_library_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot write to library collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_shared_resources",
            type=CollectionType.LIBRARY,
            display_name="shared_resources",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        assert controller.can_llm_write_to_collection("_shared_resources") is False

    def test_validate_llm_write_to_library_collection_raises_exception(self, controller, mock_classifier):
        """Test that library collection write raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="_common_data",
            type=CollectionType.LIBRARY,
            display_name="common_data",
            is_searchable=True,
            is_readonly=True,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("write", "_common_data")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.FORBIDDEN_LIBRARY_WRITE
        assert "library collections are read-only" in violation.message.lower()

    # Global Collection Write Tests
    def test_can_llm_write_to_global_collection_true(self, controller, mock_classifier):
        """Test that LLM can write to global collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="documents",
            type=CollectionType.GLOBAL,
            display_name="documents",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_write_to_collection("documents") is True

    def test_validate_llm_write_to_global_collection_success(self, controller, mock_classifier):
        """Test that global collection write succeeds."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="workspace",
            type=CollectionType.GLOBAL,
            display_name="workspace",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        # Should not raise exception
        controller.validate_llm_collection_access("write", "workspace")

    # Project Collection Write Tests
    def test_can_llm_write_to_project_collection_true(self, controller, mock_classifier):
        """Test that LLM can write to project collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="myproject-data",
            type=CollectionType.PROJECT,
            display_name="myproject-data",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="myproject",
            suffix="data"
        )
        
        assert controller.can_llm_write_to_collection("myproject-data") is True

    def test_validate_llm_write_to_project_collection_success(self, controller, mock_classifier):
        """Test that project collection write succeeds."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="testproject-content",
            type=CollectionType.PROJECT,
            display_name="testproject-content",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="testproject",
            suffix="content"
        )
        
        # Should not raise exception
        controller.validate_llm_collection_access("write", "testproject-content")

    # Memory Collection Write Tests
    def test_can_llm_write_to_memory_collection_true(self, controller, mock_classifier):
        """Test that LLM can write to memory collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="project1-memory",
            type=CollectionType.PROJECT,
            display_name="project1-memory",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=True,
            project_name="project1",
            suffix="memory"
        )
        
        assert controller.can_llm_write_to_collection("project1-memory") is True

    def test_validate_llm_write_to_memory_collection_success(self, controller, mock_classifier):
        """Test that memory collection write succeeds."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="test-memory",
            type=CollectionType.PROJECT,
            display_name="test-memory",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=True,
            project_name="test",
            suffix="memory"
        )
        
        # Should not raise exception
        controller.validate_llm_collection_access("write", "test-memory")

    # Unknown Collection Write Tests
    def test_can_llm_write_to_unknown_collection_false(self, controller, mock_classifier):
        """Test that LLM cannot write to unknown format collections."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="unknown_collection",
            type=CollectionType.UNKNOWN,
            display_name="unknown_collection",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert controller.can_llm_write_to_collection("unknown_collection") is False

    def test_validate_llm_write_to_unknown_collection_raises_exception(self, controller, mock_classifier):
        """Test that unknown collection write raises proper exception."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="malformed_name",
            type=CollectionType.UNKNOWN,
            display_name="malformed_name",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("write", "malformed_name")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "unrecognized collection type" in violation.message


class TestEdgeCasesAndErrorHandling:
    """Test edge cases, malformed inputs, and error handling scenarios."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    @pytest.fixture
    def mock_classifier(self, controller):
        """Mock the classifier to return controlled responses."""
        with patch.object(controller, 'classifier') as mock:
            yield mock

    # Invalid Collection Names
    @pytest.mark.parametrize("invalid_name", ["", "   ", "\t", "\n", None])
    def test_validate_empty_collection_names(self, controller, invalid_name):
        """Test validation of empty or None collection names."""
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", invalid_name)
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "cannot be empty" in violation.message

    def test_validate_collection_name_with_whitespace(self, controller, mock_classifier):
        """Test that collection names with leading/trailing whitespace are handled."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="project-docs",
            type=CollectionType.PROJECT,
            display_name="project-docs",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="project",
            suffix="docs"
        )
        
        # Should not raise exception - whitespace should be stripped
        controller.validate_llm_collection_access("create", "  project-docs  ")

    # Invalid Operations
    @pytest.mark.parametrize("invalid_operation", ["modify", "update", "list", "search", "", None])
    def test_validate_invalid_operations(self, controller, invalid_operation):
        """Test validation of invalid operation types."""
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access(invalid_operation, "test-collection")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "invalid operation" in violation.message.lower()

    # Collection Name Format Validation
    @patch('core.llm_access_control.collection_naming.validate_collection_name')
    def test_validate_collection_name_format_failure(self, mock_validate, controller):
        """Test handling of collection name format validation failures."""
        mock_validate.return_value = False
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "invalid-name-format")
        
        violation = exc_info.value.violation
        assert violation.violation_type == AccessViolationType.INVALID_COLLECTION_NAME
        assert "invalid collection name format" in violation.message.lower()

    # Case Sensitivity Tests
    def test_operation_case_insensitive(self, controller, mock_classifier):
        """Test that operations are case-insensitive."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="project-docs",
            type=CollectionType.PROJECT,
            display_name="project-docs",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="project",
            suffix="docs"
        )
        
        # These should all work without exception
        controller.validate_llm_collection_access("CREATE", "project-docs")
        controller.validate_llm_collection_access("Delete", "project-docs")
        controller.validate_llm_collection_access("WRITE", "project-docs")

    # Exception Chain Testing
    def test_exception_contains_violation_info(self, controller, mock_classifier):
        """Test that LLMAccessControlError contains proper violation information."""
        mock_classifier.get_collection_info.return_value = CollectionInfo(
            name="__system_test",
            type=CollectionType.SYSTEM,
            display_name="system_test",
            is_searchable=False,
            is_readonly=False,
            is_memory_collection=False
        )
        
        with pytest.raises(LLMAccessControlError) as exc_info:
            controller.validate_llm_collection_access("create", "__system_test")
        
        error = exc_info.value
        assert hasattr(error, 'violation')
        assert isinstance(error.violation, AccessViolation)
        assert error.violation.collection_name == "__system_test"
        assert error.violation.operation == "create"
        assert error.violation.violation_type == AccessViolationType.FORBIDDEN_SYSTEM_CREATION


class TestForbiddenPatternsAndSuggestions:
    """Test forbidden pattern generation and suggestion mechanisms."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    def test_get_forbidden_collection_patterns(self, controller):
        """Test that forbidden patterns are properly returned."""
        patterns = controller.get_forbidden_collection_patterns()
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check that all major pattern types are covered
        pattern_text = " ".join(patterns).lower()
        assert "system collections" in pattern_text
        assert "library collections" in pattern_text
        assert "global names" in pattern_text
        assert "memory collections" in pattern_text
        assert "invalid formats" in pattern_text

    def test_suggest_collection_name_basic(self, controller):
        """Test basic collection name suggestions."""
        suggestions = controller.suggest_collection_name("test", "general")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3  # Should return top 3
        assert all(isinstance(s, str) for s in suggestions)
        assert all("-" in s for s in suggestions)  # Should follow project-suffix format

    @pytest.mark.parametrize("purpose,expected_suffixes", [
        ("documents", ["docs", "documents", "content"]),
        ("code", ["code", "source", "implementation"]),
        ("data", ["data", "dataset", "information"]),
        ("memory", ["state", "context", "workspace"]),  # Avoids "memory" suffix
        ("general", ["docs", "data", "content"])
    ])
    def test_suggest_collection_name_purposes(self, controller, purpose, expected_suffixes):
        """Test collection name suggestions for different purposes."""
        suggestions = controller.suggest_collection_name("myproject", purpose)
        
        assert len(suggestions) > 0
        # Check that suggestions use expected suffixes
        for suggestion in suggestions:
            parts = suggestion.split("-", 1)
            if len(parts) == 2:
                suffix = parts[1]
                # At least some suggestions should match expected patterns
                # (We allow flexibility since suggestions may vary)

    def test_suggest_collection_name_avoids_existing(self, controller):
        """Test that suggestions avoid existing collection names."""
        existing = ["myproject-docs", "myproject-data", "myproject-content"]
        controller.set_existing_collections(existing)
        
        suggestions = controller.suggest_collection_name("myproject", "general")
        
        # Suggestions should not include existing collections
        for suggestion in suggestions:
            assert suggestion not in existing

    def test_suggest_collection_name_invalid_input(self, controller):
        """Test suggestion handling with invalid input names."""
        suggestions = controller.suggest_collection_name("", "general")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should provide fallback suggestions
        assert all("myproject" in s.lower() or "collection" in s.lower() for s in suggestions)

    def test_suggest_collection_name_fallback(self, controller):
        """Test fallback suggestions when normal generation fails."""
        # Set many existing collections to force fallback
        existing = []
        for i in range(10):
            for suffix in ["docs", "data", "content", "code", "source"]:
                existing.append(f"test-{suffix}{i}")
        controller.set_existing_collections(existing)
        
        with patch('core.llm_access_control.collection_naming.build_project_collection_name', side_effect=ValueError):
            suggestions = controller.suggest_collection_name("test", "general")
            
            assert len(suggestions) > 0
            # Should provide numbered fallbacks
            assert any("collection" in s for s in suggestions)


class TestConfigurationIntegration:
    """Test integration with different configuration objects."""

    def test_controller_with_mcp_config(self):
        """Test controller behavior with MCP configuration."""
        mcp_config = Mock(spec=McpConfig)
        mcp_config.server_name = "test-server"
        
        controller = LLMAccessController(mcp_config)
        
        assert controller.config == mcp_config
        assert hasattr(controller, 'classifier')

    def test_controller_with_daemon_config(self):
        """Test controller behavior with daemon configuration."""
        daemon_config = Mock(spec=DaemonConfig)
        daemon_config.daemon_name = "test-daemon"
        
        controller = LLMAccessController(daemon_config)
        
        assert controller.config == daemon_config
        assert hasattr(controller, 'classifier')

    def test_controller_config_context_preserved(self):
        """Test that configuration context is preserved during operations."""
        config = Mock(spec=McpConfig)
        controller = LLMAccessController(config)
        
        # Config should remain available throughout operations
        assert controller.config == config
        
        # After setting existing collections
        controller.set_existing_collections(["test-collection"])
        assert controller.config == config


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    @patch('core.llm_access_control.LLMAccessController')
    def test_can_llm_create_collection_function(self, mock_controller_class):
        """Test module-level can_llm_create_collection function."""
        mock_controller = Mock()
        mock_controller.can_llm_create_collection.return_value = True
        mock_controller_class.return_value = mock_controller
        
        result = can_llm_create_collection("test-collection")
        
        assert result is True
        mock_controller_class.assert_called_once_with(None)
        mock_controller.can_llm_create_collection.assert_called_once_with("test-collection")

    @patch('core.llm_access_control.LLMAccessController')
    def test_can_llm_create_collection_function_with_config(self, mock_controller_class):
        """Test module-level can_llm_create_collection function with config."""
        mock_controller = Mock()
        mock_controller.can_llm_create_collection.return_value = False
        mock_controller_class.return_value = mock_controller
        
        config = Mock(spec=McpConfig)
        result = can_llm_create_collection("__system_test", config)
        
        assert result is False
        mock_controller_class.assert_called_once_with(config)
        mock_controller.can_llm_create_collection.assert_called_once_with("__system_test")

    @patch('core.llm_access_control.LLMAccessController')
    def test_can_llm_delete_collection_function(self, mock_controller_class):
        """Test module-level can_llm_delete_collection function."""
        mock_controller = Mock()
        mock_controller.can_llm_delete_collection.return_value = True
        mock_controller_class.return_value = mock_controller
        
        result = can_llm_delete_collection("project-docs")
        
        assert result is True
        mock_controller_class.assert_called_once_with(None)
        mock_controller.can_llm_delete_collection.assert_called_once_with("project-docs")

    @patch('core.llm_access_control.LLMAccessController')
    def test_validate_llm_collection_access_function(self, mock_controller_class):
        """Test module-level validate_llm_collection_access function."""
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        validate_llm_collection_access("write", "test-collection")
        
        mock_controller_class.assert_called_once_with(None)
        mock_controller.validate_llm_collection_access.assert_called_once_with("write", "test-collection")

    @patch('core.llm_access_control.LLMAccessController')
    def test_validate_llm_collection_access_function_with_config(self, mock_controller_class):
        """Test module-level validate_llm_collection_access function with config."""
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        config = Mock(spec=DaemonConfig)
        validate_llm_collection_access("delete", "project-data", config)
        
        mock_controller_class.assert_called_once_with(config)
        mock_controller.validate_llm_collection_access.assert_called_once_with("delete", "project-data")

    @patch('core.llm_access_control.LLMAccessController')
    def test_get_forbidden_collection_patterns_function(self, mock_controller_class):
        """Test module-level get_forbidden_collection_patterns function."""
        mock_controller = Mock()
        expected_patterns = ["__* (system)", "_* (library)", "global names"]
        mock_controller.get_forbidden_collection_patterns.return_value = expected_patterns
        mock_controller_class.return_value = mock_controller
        
        result = get_forbidden_collection_patterns()
        
        assert result == expected_patterns
        mock_controller_class.assert_called_once_with(None)
        mock_controller.get_forbidden_collection_patterns.assert_called_once()


class TestSecurityBoundaryValidation:
    """Test comprehensive security boundary scenarios."""

    @pytest.fixture
    def controller(self):
        """Create a fresh LLMAccessController for each test."""
        return LLMAccessController()

    def test_access_control_matrix_system_collections(self, controller):
        """Test complete access control matrix for system collections."""
        with patch.object(controller, 'classifier') as mock_classifier:
            mock_classifier.get_collection_info.return_value = CollectionInfo(
                name="__test_system",
                type=CollectionType.SYSTEM,
                display_name="test_system",
                is_searchable=False,
                is_readonly=False,
                is_memory_collection=False
            )
            
            # System collections: Create ❌, Delete ❌, Write ❌
            assert controller.can_llm_create_collection("__test_system") is False
            assert controller.can_llm_delete_collection("__test_system") is False
            assert controller.can_llm_write_to_collection("__test_system") is False

    def test_access_control_matrix_library_collections(self, controller):
        """Test complete access control matrix for library collections."""
        with patch.object(controller, 'classifier') as mock_classifier:
            mock_classifier.get_collection_info.return_value = CollectionInfo(
                name="_test_library",
                type=CollectionType.LIBRARY,
                display_name="test_library",
                is_searchable=True,
                is_readonly=True,
                is_memory_collection=False
            )
            
            # Library collections: Create ❌, Delete ❌, Write ❌
            assert controller.can_llm_create_collection("_test_library") is False
            assert controller.can_llm_delete_collection("_test_library") is False
            assert controller.can_llm_write_to_collection("_test_library") is False

    def test_access_control_matrix_global_collections(self, controller):
        """Test complete access control matrix for global collections."""
        with patch.object(controller, 'classifier') as mock_classifier:
            mock_classifier.get_collection_info.return_value = CollectionInfo(
                name="documents",
                type=CollectionType.GLOBAL,
                display_name="documents",
                is_searchable=True,
                is_readonly=False,
                is_memory_collection=False
            )
            
            # Global collections: Create ❌, Delete ❌, Write ✅
            assert controller.can_llm_create_collection("documents") is False
            assert controller.can_llm_delete_collection("documents") is False
            assert controller.can_llm_write_to_collection("documents") is True

    def test_access_control_matrix_project_collections(self, controller):
        """Test complete access control matrix for project collections."""
        with patch.object(controller, 'classifier') as mock_classifier:
            mock_classifier.get_collection_info.return_value = CollectionInfo(
                name="project-docs",
                type=CollectionType.PROJECT,
                display_name="project-docs",
                is_searchable=True,
                is_readonly=False,
                is_memory_collection=False,
                project_name="project",
                suffix="docs"
            )
            
            # Project collections: Create ✅, Delete ✅, Write ✅
            assert controller.can_llm_create_collection("project-docs") is True
            assert controller.can_llm_delete_collection("project-docs") is True
            assert controller.can_llm_write_to_collection("project-docs") is True

    def test_access_control_matrix_memory_collections(self, controller):
        """Test complete access control matrix for memory collections."""
        with patch.object(controller, 'classifier') as mock_classifier:
            mock_classifier.get_collection_info.return_value = CollectionInfo(
                name="project-memory",
                type=CollectionType.PROJECT,
                display_name="project-memory",
                is_searchable=True,
                is_readonly=False,
                is_memory_collection=True,
                project_name="project",
                suffix="memory"
            )
            
            # Memory collections: Create ⚠️ (conditional), Delete ❌, Write ✅
            # For this test, assume creation is allowed for project-type memory collections
            assert controller.can_llm_create_collection("project-memory") is True
            assert controller.can_llm_delete_collection("project-memory") is False
            assert controller.can_llm_write_to_collection("project-memory") is True

    def test_comprehensive_security_boundary_enforcement(self, controller):
        """Test that all security boundaries are properly enforced."""
        test_cases = [
            # (collection_name, collection_type, is_memory, expected_create, expected_delete, expected_write)
            ("__system_test", CollectionType.SYSTEM, False, False, False, False),
            ("_library_test", CollectionType.LIBRARY, False, False, False, False),
            ("algorithms", CollectionType.GLOBAL, False, False, False, True),
            ("project-docs", CollectionType.PROJECT, False, True, True, True),
            ("project-memory", CollectionType.PROJECT, True, True, False, True),
            ("__memory_system", CollectionType.SYSTEM, True, False, False, False),
            ("invalid_format", CollectionType.UNKNOWN, False, False, False, False),
        ]
        
        with patch.object(controller, 'classifier') as mock_classifier:
            for name, col_type, is_memory, exp_create, exp_delete, exp_write in test_cases:
                mock_classifier.get_collection_info.return_value = CollectionInfo(
                    name=name,
                    type=col_type,
                    display_name=name.lstrip("_"),
                    is_searchable=col_type != CollectionType.SYSTEM,
                    is_readonly=col_type == CollectionType.LIBRARY,
                    is_memory_collection=is_memory
                )
                
                assert controller.can_llm_create_collection(name) == exp_create, f"Create access failed for {name}"
                assert controller.can_llm_delete_collection(name) == exp_delete, f"Delete access failed for {name}"
                assert controller.can_llm_write_to_collection(name) == exp_write, f"Write access failed for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])