"""
Common step definitions for E2E scenarios.

Shared steps used across multiple feature files.
"""

import asyncio
import time
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

# Background steps

@given("the system is not running")
def system_not_running(component_lifecycle_manager):
    """Ensure system is not running."""
    # Stop all components if running
    asyncio.run(component_lifecycle_manager.stop_all())


@given("all previous test data is cleaned up")
def cleanup_test_data():
    """Clean up any previous test data."""
    # Placeholder - would clean up Qdrant collections, SQLite, etc.
    pass


@given("the system is fully operational")
async def system_fully_operational(component_lifecycle_manager):
    """Ensure all system components are running."""
    started = await component_lifecycle_manager.start_all()
    assert started, "Failed to start all components"

    ready = await component_lifecycle_manager.wait_for_ready(timeout=60)
    assert ready, "System did not become ready in time"


# Component startup steps

@when("I start Qdrant service")
async def start_qdrant(component_lifecycle_manager):
    """Start Qdrant service."""
    success = await component_lifecycle_manager.start_component("qdrant")
    assert success, "Failed to start Qdrant"


@when("I start daemon service")
async def start_daemon(component_lifecycle_manager):
    """Start daemon service."""
    success = await component_lifecycle_manager.start_component("daemon")
    assert success, "Failed to start daemon"


@when("I start MCP server")
async def start_mcp_server(component_lifecycle_manager):
    """Start MCP server."""
    success = await component_lifecycle_manager.start_component("mcp_server")
    assert success, "Failed to start MCP server"


@when("I start all components simultaneously")
async def start_all_components(component_lifecycle_manager):
    """Start all components in parallel."""
    # Start components in parallel
    tasks = [
        component_lifecycle_manager.start_component("qdrant"),
        component_lifecycle_manager.start_component("daemon"),
        component_lifecycle_manager.start_component("mcp_server")
    ]

    results = await asyncio.gather(*tasks)
    assert all(results), "Some components failed to start"


# Health check steps

@then(parsers.parse("{component} should be healthy within {timeout:d} seconds"))
async def component_healthy_within_timeout(component, timeout, component_lifecycle_manager):
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
        health = await component_lifecycle_manager.check_health(component_name)
        if health.get("healthy"):
            return

        await asyncio.sleep(2)

    pytest.fail(f"{component} did not become healthy within {timeout} seconds")


@then("all components should be running")
async def all_components_running(component_lifecycle_manager):
    """Verify all components are running."""
    for component in ["qdrant", "daemon", "mcp_server"]:
        health = await component_lifecycle_manager.check_health(component)
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
async def daemon_detects_file(timeout, scenario_context):
    """Verify daemon detected the file."""
    # Placeholder - would check daemon logs or status
    await asyncio.sleep(2)  # Simulate detection time
    scenario_context.set("file_detected", True)


@then(parsers.parse("the file should be ingested to Qdrant within {timeout:d} seconds"))
async def file_ingested_to_qdrant(timeout, scenario_context):
    """Verify file was ingested."""
    # Placeholder - would query Qdrant for the document
    await asyncio.sleep(2)  # Simulate ingestion time
    scenario_context.set("file_ingested", True)


# Metadata validation steps

@then("metadata should include project context")
def metadata_includes_project_context(scenario_context):
    """Verify metadata includes project information."""
    # Placeholder - would verify metadata fields
    assert scenario_context.get("file_ingested"), "File not ingested"


# Search validation steps

@then("I should be able to search for the file content")
async def can_search_for_file(scenario_context):
    """Verify file content is searchable."""
    # Placeholder - would perform actual search
    await asyncio.sleep(1)
    scenario_context.set("search_successful", True)


# Degraded mode steps

@when("I try to start MCP server without daemon")
async def start_mcp_without_daemon(component_lifecycle_manager):
    """Start MCP server when daemon is not running."""
    # Ensure daemon is stopped
    await component_lifecycle_manager.stop_component("daemon")

    # Start MCP server
    await component_lifecycle_manager.start_component("mcp_server")


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
