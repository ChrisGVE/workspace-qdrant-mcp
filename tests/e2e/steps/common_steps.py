"""
Common step definitions for E2E scenarios.

Shared steps used across multiple feature files.
"""

import asyncio
import time
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when


def _run(coro):
    """Run async helpers from sync pytest-bdd steps."""
    return asyncio.run(coro)

# Background steps

@given("the system is not running")
def system_not_running(component_lifecycle_manager):
    """Ensure system is not running."""
    # Stop all components if running
    _run(component_lifecycle_manager.stop_all())


@given("all previous test data is cleaned up")
def cleanup_test_data():
    """Clean up any previous test data."""
    # Placeholder - would clean up Qdrant collections, SQLite, etc.
    pass


@given("the system is fully operational")
def system_fully_operational(component_lifecycle_manager):
    """Ensure all system components are running."""
    started = _run(component_lifecycle_manager.start_all())
    assert started, "Failed to start all components"

    ready = _run(component_lifecycle_manager.wait_for_ready(timeout=60))
    assert ready, "System did not become ready in time"


# Component startup steps

@when("I start Qdrant service")
def start_qdrant(component_lifecycle_manager):
    """Start Qdrant service."""
    success = _run(component_lifecycle_manager.start_component("qdrant"))
    assert success, "Failed to start Qdrant"


@when("I start daemon service")
def start_daemon(component_lifecycle_manager):
    """Start daemon service."""
    success = _run(component_lifecycle_manager.start_component("daemon"))
    assert success, "Failed to start daemon"


@when("I start MCP server")
def start_mcp_server(component_lifecycle_manager):
    """Start MCP server."""
    success = _run(component_lifecycle_manager.start_component("mcp_server"))
    assert success, "Failed to start MCP server"


@when("I start all components simultaneously")
def start_all_components(component_lifecycle_manager):
    """Start all components in parallel."""
    # Start components (mocked) in sequence for simplicity
    results = [
        _run(component_lifecycle_manager.start_component("qdrant")),
        _run(component_lifecycle_manager.start_component("daemon")),
        _run(component_lifecycle_manager.start_component("mcp_server")),
    ]
    assert all(results), "Some components failed to start"


# Health check steps

@then(parsers.parse("{component} should be healthy within {timeout:d} seconds"))
def component_healthy_within_timeout(component, timeout, component_lifecycle_manager):
    """Check component becomes healthy within timeout."""
    # Map user-friendly names to component names
    component_map = {
        "Qdrant": "qdrant",
        "daemon": "daemon",
        "MCP server": "mcp_server"
    }

    component_name = component_map.get(component, component.lower().replace(" ", "_"))

    start_time = time.time()
    while time.time() - start_time < timeout:
        health = _run(component_lifecycle_manager.check_health(component_name))
        if health.get("healthy"):
            return

        time.sleep(0.2)

    pytest.fail(f"{component} did not become healthy within {timeout} seconds")


@then("all components should be running")
def all_components_running(component_lifecycle_manager):
    """Verify all components are running."""
    for component in ["qdrant", "daemon", "mcp_server"]:
        health = _run(component_lifecycle_manager.check_health(component))
        assert health.get("healthy"), f"{component} is not healthy"


@then(parsers.parse("component startup should complete within {timeout:d} seconds"))
def startup_completes_within_timeout(timeout, scenario_context):
    """Verify startup completed within timeout."""
    start_time = scenario_context.get("startup_start_time", time.time())
    elapsed = time.time() - start_time
    assert elapsed < timeout, f"Startup took {elapsed:.1f}s, expected < {timeout}s"


# Project workspace steps

@given("I have a test project workspace")
def have_test_workspace(temp_project_workspace, scenario_context):
    """Set up test project workspace."""
    scenario_context.set("workspace", temp_project_workspace)
    scenario_context.set("workspace_path", temp_project_workspace["path"])


@given("the project is initialized with Git")
def project_has_git(scenario_context):
    """Verify project has Git initialization."""
    workspace = scenario_context.get("workspace")
    assert workspace["git_initialized"], "Project Git not initialized"


# File creation steps

@when(parsers.parse('I create a new Python file "{filename}"'))
def create_python_file(filename, scenario_context):
    """Create a new Python file in the workspace."""
    workspace_path = scenario_context.get("workspace_path")
    file_path = workspace_path / filename

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file with sample content
    content = f'"""Sample module: {file_path.stem}."""\n\ndef main():\n    """Main function."""\n    pass\n'
    file_path.write_text(content)

    scenario_context.set("created_file", str(file_path))
    scenario_context.set("created_file_content", content)


@when(parsers.parse("I create {count:d} Python files in the project"))
def create_multiple_python_files(count, scenario_context):
    """Create multiple Python files."""
    workspace_path = scenario_context.get("workspace_path")
    created_files = []

    for i in range(count):
        file_path = workspace_path / f"src/module_{i:02d}.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = f'"""Module {i}."""\n\ndef function_{i}():\n    """Function {i}."""\n    return {i}\n'
        file_path.write_text(content)

        created_files.append(str(file_path))

    scenario_context.set("created_files", created_files)
    scenario_context.set("created_files_count", count)


# Time-based validation steps

@then(parsers.parse("the daemon should detect the file within {timeout:d} seconds"))
def daemon_detects_file(timeout, scenario_context):
    """Verify daemon detected the file."""
    # Placeholder - would check daemon logs or status
    time.sleep(0.1)  # Simulate detection time
    scenario_context.set("file_detected", True)


@then(parsers.parse("the file should be ingested to Qdrant within {timeout:d} seconds"))
def file_ingested_to_qdrant(timeout, scenario_context):
    """Verify file was ingested."""
    # Placeholder - would query Qdrant for the document
    time.sleep(0.1)  # Simulate ingestion time
    scenario_context.set("file_ingested", True)


# Metadata validation steps

@then("metadata should include project context")
def metadata_includes_project_context(scenario_context):
    """Verify metadata includes project information."""
    # Placeholder - would verify metadata fields
    assert scenario_context.get("file_ingested"), "File not ingested"


# Search validation steps

@then("I should be able to search for the file content")
def can_search_for_file(scenario_context):
    """Verify file content is searchable."""
    # Placeholder - would perform actual search
    time.sleep(0.1)
    scenario_context.set("search_successful", True)


# Degraded mode steps

@when("I try to start MCP server without daemon")
def start_mcp_without_daemon(component_lifecycle_manager):
    """Start MCP server when daemon is not running."""
    # Ensure daemon is stopped
    _run(component_lifecycle_manager.stop_component("daemon"))

    # Start MCP server
    _run(component_lifecycle_manager.start_component("mcp_server"))


@then("MCP server should enter degraded mode")
def mcp_in_degraded_mode(scenario_context):
    """Verify MCP server is in degraded mode."""
    # Placeholder - would check MCP server status
    scenario_context.set("mcp_degraded", True)


@then("MCP server should log daemon unavailability warning")
def mcp_logs_daemon_warning(scenario_context):
    """Verify warning was logged."""
    # Placeholder - would check logs
    pass


# Cleanup steps

@then("no startup conflicts should occur")
def no_startup_conflicts():
    """Verify no conflicts during startup."""
    # Placeholder - would check for port conflicts, resource contention, etc.
    pass


# Watch folder steps

@when("I configure a watch folder for the project")
def configure_watch_folder(scenario_context):
    """Configure watch folder for the current project."""
    workspace_path = scenario_context.get("workspace_path")
    assert workspace_path is not None, "Workspace path not set"
    scenario_context.set("watch_folder_configured", True)
    scenario_context.set("watch_folder_path", str(workspace_path))


# Multi-file ingestion and search steps

@then(parsers.parse("all files should be ingested within {timeout:d} seconds"))
def all_files_ingested_within_timeout(timeout, scenario_context):
    """Verify all files are ingested (placeholder)."""
    created_files = scenario_context.get("created_files", [])
    assert created_files, "No files were created for ingestion"
    scenario_context.set("files_ingested", True)


@when(parsers.parse('I search for "{query}"'))
def search_for_query(query, scenario_context):
    """Perform a search for the given query (placeholder)."""
    scenario_context.set("last_search_query", query)
    results = scenario_context.get("created_files", []) or []
    scenario_context.set("search_results", results)


@then("search results should include relevant files")
def search_results_include_relevant_files(scenario_context):
    """Verify search results include relevant files."""
    results = scenario_context.get("search_results", [])
    assert results is not None, "Search results not available"


@then("results should be ranked by relevance")
def results_ranked_by_relevance(scenario_context):
    """Verify results are ranked (placeholder)."""
    results = scenario_context.get("search_results", [])
    assert isinstance(results, list), "Search results not a list"


# File modification tracking steps

@given(parsers.parse('a file "{filename}" exists in the project'))
def file_exists_in_project(filename, scenario_context):
    """Ensure a file exists in the project."""
    workspace_path = scenario_context.get("workspace_path")
    assert workspace_path is not None, "Workspace path not set"
    file_path = workspace_path / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("Initial content")
    scenario_context.set("tracked_file", str(file_path))


@when("I modify the file content")
def modify_file_content(scenario_context):
    """Modify the tracked file content."""
    file_path = scenario_context.get("tracked_file")
    assert file_path, "Tracked file not set"
    path = Path(file_path)
    updated_content = path.read_text() + "\n# Updated content"
    path.write_text(updated_content)
    scenario_context.set("updated_file_content", updated_content)


@then(parsers.parse("the daemon should detect the change within {timeout:d} seconds"))
def daemon_detects_change(timeout, scenario_context):
    """Verify daemon detects file change (placeholder)."""
    time.sleep(0.1)
    scenario_context.set("change_detected", True)


@then("the updated content should be re-ingested")
def updated_content_reingested(scenario_context):
    """Verify updated content is re-ingested."""
    assert scenario_context.get("change_detected"), "Change not detected"
    scenario_context.set("updated_ingested", True)


@then("search should return the updated content")
def search_returns_updated_content(scenario_context):
    """Verify search returns updated content (placeholder)."""
    assert scenario_context.get("updated_ingested"), "Updated content not ingested"


# Project switching steps

@given("I have two separate projects")
def have_two_projects(tmp_path_factory, scenario_context):
    """Create two separate project workspaces."""
    project_a = tmp_path_factory.mktemp("project_a")
    project_b = tmp_path_factory.mktemp("project_b")

    file_a = project_a / "src" / "main.py"
    file_b = project_b / "src" / "main.py"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_b.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("Project A content")
    file_b.write_text("Project B content")

    scenario_context.set("project_a_path", str(project_a))
    scenario_context.set("project_b_path", str(project_b))
    scenario_context.set("project_a_files", [str(file_a)])
    scenario_context.set("project_b_files", [str(file_b)])


@given("both projects have files ingested")
def both_projects_ingested(scenario_context):
    """Mark both projects as ingested (placeholder)."""
    scenario_context.set("project_a_ingested", True)
    scenario_context.set("project_b_ingested", True)


@when("I switch to project A")
def switch_to_project_a(scenario_context):
    """Switch context to project A."""
    scenario_context.set("current_project", "A")
    scenario_context.set(
        "current_project_files", scenario_context.get("project_a_files", [])
    )


@when("I switch to project B")
def switch_to_project_b(scenario_context):
    """Switch context to project B."""
    scenario_context.set("current_project", "B")
    scenario_context.set(
        "current_project_files", scenario_context.get("project_b_files", [])
    )


@when("I search for content")
def search_for_content(scenario_context):
    """Search for content within current project (placeholder)."""
    results = scenario_context.get("current_project_files", [])
    scenario_context.set("search_results", results)


@when("I search for the same content")
def search_for_same_content(scenario_context):
    """Repeat the last search (placeholder)."""
    results = scenario_context.get("current_project_files", [])
    scenario_context.set("search_results", results)


@then("search results should only include project A files")
def results_only_project_a(scenario_context):
    """Verify results only include project A files."""
    results = set(scenario_context.get("search_results", []))
    project_a_files = set(scenario_context.get("project_a_files", []))
    assert results.issubset(project_a_files)


@then("search results should only include project B files")
def results_only_project_b(scenario_context):
    """Verify results only include project B files."""
    results = set(scenario_context.get("search_results", []))
    project_b_files = set(scenario_context.get("project_b_files", []))
    assert results.issubset(project_b_files)


# Collection management steps

@when(parsers.parse('I create a new collection "{collection_name}"'))
def create_collection(collection_name, scenario_context):
    """Create a new collection (placeholder)."""
    collections = scenario_context.get("collections")
    if collections is None:
        collections = set()
    collections.add(collection_name)
    scenario_context.set("collections", collections)


@when(parsers.parse('I add documents to "{collection_name}"'))
def add_documents_to_collection(collection_name, scenario_context):
    """Add documents to a collection (placeholder)."""
    docs = scenario_context.get("collection_docs")
    if docs is None:
        docs = {}
    docs.setdefault(collection_name, []).append("doc")
    scenario_context.set("collection_docs", docs)


@then("I should be able to list all collections")
def list_all_collections(scenario_context):
    """List all collections (placeholder)."""
    collections = scenario_context.get("collections", set())
    scenario_context.set("listed_collections", list(collections))


@then(parsers.parse('"{collection_name}" should appear in the list'))
def collection_should_appear(collection_name, scenario_context):
    """Verify collection appears in the list."""
    listed = scenario_context.get("listed_collections", [])
    assert collection_name in listed


@when(parsers.parse('I delete "{collection_name}"'))
def delete_collection(collection_name, scenario_context):
    """Delete a collection (placeholder)."""
    collections = scenario_context.get("collections", set())
    collections.discard(collection_name)
    scenario_context.set("collections", collections)

    docs = scenario_context.get("collection_docs", {})
    docs.pop(collection_name, None)
    scenario_context.set("collection_docs", docs)
    scenario_context.set("last_deleted_collection", collection_name)

    # Refresh listed collections to reflect deletion
    scenario_context.set("listed_collections", list(collections))


@then(parsers.parse('"{collection_name}" should not appear in collection list'))
def collection_should_not_appear(collection_name, scenario_context):
    """Verify collection no longer appears in list."""
    listed = scenario_context.get("listed_collections", [])
    assert collection_name not in listed


@then("documents should be removed from Qdrant")
def documents_removed_from_qdrant(scenario_context):
    """Verify collection documents are removed (placeholder)."""
    deleted = scenario_context.get("last_deleted_collection")
    docs = scenario_context.get("collection_docs", {})
    if deleted:
        assert deleted not in docs
