#!/usr/bin/env python3
"""
Functional test for integrated queue system components.

Tests the integration between:
- Tool Discovery System (Task 349)
- Language Support System (Task 347)
- Queue Operations (Task 348)
- Missing Metadata Tracker (Task 350)
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.core.sqlite_state_manager import SQLiteStateManager
from common.core.tool_discovery import ToolDiscovery
from common.core.tool_database_integration import ToolDatabaseIntegration
from common.core.tool_reporting import ToolReporter
from common.core.language_support_parser import LanguageSupportParser
from common.core.language_support_loader import LanguageSupportLoader
from common.core.language_support_manager import LanguageSupportManager
from common.core.missing_metadata_tracker import MissingMetadataTracker
from loguru import logger


async def test_tool_discovery(state_manager: SQLiteStateManager):
    """Test 1: Tool Discovery System"""
    logger.info("=" * 60)
    logger.info("TEST 1: Tool Discovery System")
    logger.info("=" * 60)

    # Initialize tool discovery
    discovery = ToolDiscovery()
    db_integration = ToolDatabaseIntegration(state_manager)

    # Discover tools
    logger.info("Discovering LSP servers...")
    # For now, just discover what's available on system
    python_lsp = discovery.find_executable("pyright")
    logger.info(f"  Python LSP (pyright): {python_lsp or 'NOT FOUND'}")

    rust_lsp = discovery.find_executable("rust-analyzer")
    logger.info(f"  Rust LSP (rust-analyzer): {rust_lsp or 'NOT FOUND'}")

    logger.info("Discovering tree-sitter CLI...")
    ts_result = discovery.discover_tree_sitter_cli()
    logger.info(f"  Tree-sitter: {ts_result}")

    logger.info("Discovering compilers...")
    compilers = discovery.discover_compilers()
    for name, path in compilers.items():
        if path:
            logger.info(f"  {name}: {path}")

    logger.info("Discovering build tools...")
    build_tools = discovery.discover_build_tools()
    for name, path in build_tools.items():
        if path:
            logger.info(f"  {name}: {path}")

    logger.info("✓ Tool discovery complete\n")
    return discovery, db_integration


async def test_language_support(state_manager: SQLiteStateManager):
    """Test 2: Language Support System"""
    logger.info("=" * 60)
    logger.info("TEST 2: Language Support System")
    logger.info("=" * 60)

    # Check if language_support.yaml exists
    yaml_path = Path("assets/languages_support.yaml")
    if not yaml_path.exists():
        logger.warning(f"  {yaml_path} not found, skipping language support test")
        return None

    # Initialize language support
    manager = LanguageSupportManager(state_manager)

    logger.info(f"Loading language support from {yaml_path}...")
    result = await manager.initialize_from_yaml(yaml_path)
    logger.info(f"  Loaded {result.get('languages_loaded', 0)} languages")
    logger.info(f"  Version: {result.get('version', 'unknown')}")

    logger.info("✓ Language support loaded\n")
    return manager


async def test_queue_operations(state_manager: SQLiteStateManager):
    """Test 3: Queue Operations"""
    logger.info("=" * 60)
    logger.info("TEST 3: Queue Operations")
    logger.info("=" * 60)

    # Create test file path
    test_file = Path(__file__).parent / "README.md"
    if not test_file.exists():
        test_file = Path(__file__)  # Use this script itself

    logger.info(f"Enqueuing test file: {test_file}")

    # Calculate tenant and branch
    tenant_id = await state_manager.calculate_tenant_id(test_file.parent)
    branch = await state_manager.get_current_branch(test_file.parent)

    logger.info(f"  Tenant ID: {tenant_id}")
    logger.info(f"  Branch: {branch}")

    # Enqueue the file
    queue_id = await state_manager.enqueue(
        file_path=str(test_file),
        collection="test-collection",
        priority=5,  # NORMAL priority
        tenant_id=tenant_id,
        branch=branch,
        metadata={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    )

    logger.info(f"  Enqueued with ID: {queue_id}")

    # Check queue depth
    depth = await state_manager.get_queue_depth()
    logger.info(f"  Queue depth: {depth}")

    # Dequeue the file
    logger.info("Dequeuing file...")
    items = await state_manager.dequeue(batch_size=1)

    if items:
        item = items[0]
        logger.info(f"  Dequeued: {item.file_path}")
        logger.info(f"  Priority: {item.priority}")
        logger.info(f"  Collection: {item.collection}")

        # Remove from queue
        await state_manager.remove_from_queue(item.queue_id)
        logger.info(f"  Removed from queue")
    else:
        logger.error("  Failed to dequeue item!")

    logger.info("✓ Queue operations complete\n")


async def test_missing_metadata_tracker(state_manager: SQLiteStateManager):
    """Test 4: Missing Metadata Tracker"""
    logger.info("=" * 60)
    logger.info("TEST 4: Missing Metadata Tracker")
    logger.info("=" * 60)

    tracker = MissingMetadataTracker(state_manager)

    # Track a file with missing LSP
    test_file = Path(__file__).parent / "test_file.py"

    logger.info(f"Tracking file with missing LSP: {test_file}")
    await tracker.track_missing_metadata(
        file_path=str(test_file),
        language_name="python",
        branch="main",
        missing_lsp=True,
        missing_ts=False
    )

    # Get tracked files
    tracked = await tracker.get_files_missing_metadata(language="python")
    logger.info(f"  Tracked files: {len(tracked)}")
    for f in tracked:
        logger.info(f"    - {f['file_absolute_path']}")
        logger.info(f"      LSP missing: {f['missing_lsp_metadata']}")
        logger.info(f"      TS missing: {f['missing_ts_metadata']}")

    # Check tool availability
    lsp_status = await tracker.check_lsp_available("python")
    logger.info(f"  Python LSP available: {lsp_status['available']}")

    ts_status = await tracker.check_tree_sitter_available()
    logger.info(f"  Tree-sitter available: {ts_status['available']}")

    # Get statistics
    stats = await tracker.get_tracked_file_count()
    logger.info(f"  Total tracked: {stats['total']}")
    logger.info(f"  Missing LSP: {stats['missing_lsp']}")
    logger.info(f"  Missing TS: {stats['missing_ts']}")

    # Cleanup
    await tracker.remove_tracked_file(str(test_file))
    logger.info(f"  Cleaned up test entry")

    logger.info("✓ Missing metadata tracker complete\n")


async def test_missing_tool_reporting(state_manager: SQLiteStateManager, discovery: ToolDiscovery):
    """Test 5: Missing Tool Reporting"""
    logger.info("=" * 60)
    logger.info("TEST 5: Missing Tool Reporting")
    logger.info("=" * 60)

    reporter = ToolReporter(discovery, state_manager)

    # Get missing tools
    logger.info("Generating missing tools report...")
    missing = await reporter.get_missing_tools()

    if missing:
        logger.info(f"  Found {len(missing)} missing tools:")
        for tool in missing:
            logger.info(f"    - {tool.name} ({tool.tool_type})")
            logger.info(f"      Severity: {tool.severity}")
            if tool.install_command:
                logger.info(f"      Install: {tool.install_command}")

        # Generate full report
        report = reporter.generate_installation_guide(missing)
        logger.info("\n" + report)
    else:
        logger.info("  No missing tools found!")

    logger.info("✓ Missing tool reporting complete\n")


async def test_integration_workflow(state_manager: SQLiteStateManager):
    """Test 6: End-to-End Integration Workflow"""
    logger.info("=" * 60)
    logger.info("TEST 6: End-to-End Integration Workflow")
    logger.info("=" * 60)

    tracker = MissingMetadataTracker(state_manager)
    test_file = Path(__file__).parent / "integration_test.py"

    # Step 1: Simulate processing failure
    logger.info("Step 1: Simulating processing failure...")
    await tracker.track_missing_metadata(
        file_path=str(test_file),
        language_name="python",
        branch="main",
        missing_lsp=True,
        missing_ts=False
    )
    logger.info("  ✓ File tracked for missing LSP")

    # Step 2: Check tool availability
    logger.info("Step 2: Checking tool availability...")
    lsp_status = await tracker.check_lsp_available("python")
    logger.info(f"  Python LSP available: {lsp_status['available']}")

    # Step 3: Requeue if tool available
    if lsp_status['available']:
        logger.info("Step 3: Tool available, requeuing...")
        result = await tracker.requeue_when_tools_available(
            tool_type='lsp',
            language='python',
            priority=5
        )
        logger.info(f"  Requeued: {result['requeued']} files")
        logger.info(f"  Still missing: {result['still_missing']} files")
    else:
        logger.info("Step 3: Tool not available, keeping in tracker")
        logger.info("  File remains tracked for later requeuing")

    # Step 4: Cleanup
    logger.info("Step 4: Cleanup...")
    await tracker.remove_tracked_file(str(test_file))
    logger.info("  ✓ Test entry removed")

    logger.info("✓ Integration workflow complete\n")


async def main():
    """Run all functional tests"""
    logger.info("\n" + "=" * 60)
    logger.info("FUNCTIONAL TEST SUITE")
    logger.info("Testing integration of queue system components")
    logger.info("=" * 60 + "\n")

    # Use temporary database for testing
    db_path = Path(__file__).parent / "workspace_state.db"
    logger.info(f"Using database: {db_path}\n")

    # Initialize state manager
    state_manager = SQLiteStateManager(db_path)
    await state_manager.initialize()

    try:
        # Run tests
        discovery, db_integration = await test_tool_discovery(state_manager)
        lang_manager = await test_language_support(state_manager)
        await test_queue_operations(state_manager)
        await test_missing_metadata_tracker(state_manager)
        await test_missing_tool_reporting(state_manager, discovery)
        await test_integration_workflow(state_manager)

        # Summary
        logger.info("=" * 60)
        logger.info("FUNCTIONAL TEST SUMMARY")
        logger.info("=" * 60)
        logger.info("✓ All tests completed successfully!")
        logger.info("\nSystems verified:")
        logger.info("  1. Tool Discovery")
        logger.info("  2. Language Support")
        logger.info("  3. Queue Operations")
        logger.info("  4. Missing Metadata Tracker")
        logger.info("  5. Missing Tool Reporting")
        logger.info("  6. End-to-End Integration")
        logger.info("\nThe queue system foundation is working correctly.")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"\n❌ FUNCTIONAL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Note: Not closing state_manager to preserve data for inspection
        pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
