#!/usr/bin/env python3
"""
Temporary script to identify and fix missing module imports.
This script analyzes import errors and copies missing modules from common to workspace_qdrant_mcp.
"""

import os
import shutil
from pathlib import Path

def copy_missing_modules():
    """Copy missing modules from common to workspace_qdrant_mcp package."""

    source_base = Path("src/python/common")
    target_base = Path("src/python/workspace_qdrant_mcp")

    # List of modules that tests are trying to import but are missing
    missing_modules = [
        # Core modules
        "core/pattern_manager.py",
        "core/admin_cli.py",
        "core/incremental_processor.py",
        "core/monitoring_integration.py",
        "core/priority_queue_integration.py",
        "core/priority_queue_manager.py",
        "core/unified_config.py",
        "core/automatic_recovery.py",
        "core/collision_detection.py",
        "core/component_lifecycle.py",
        "core/daemon_integration_workflows.py",
        "core/daemon_ipc_communication.py",
        "core/daemon_storage_integration.py",
        "core/document_processing_workflow.py",
        "core/file_watching_core.py",
        "core/graceful_degradation.py",
        "core/metadata_filtering.py",
        "core/metadata_schema.py",
        "core/performance_monitoring.py",
        "core/priority_queue_manager_core.py",
        "core/workspace_integration.py",

        # Utils modules
        "utils/admin_cli.py",
        "utils/os_directories.py",

        # Memory modules (create if needed)
        "memory/",

        # Tools modules
        "tools/performance_benchmark_cli.py",
    ]

    copied_count = 0
    errors = []

    for module_path in missing_modules:
        source_path = source_base / module_path
        target_path = target_base / module_path

        try:
            if source_path.exists():
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if source_path.is_dir():
                    # Copy directory
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
                    print(f"Copied directory: {module_path}")
                else:
                    # Copy file
                    shutil.copy2(source_path, target_path)
                    print(f"Copied file: {module_path}")

                copied_count += 1
            else:
                print(f"Source not found: {source_path}")
                errors.append(f"Missing source: {module_path}")

        except Exception as e:
            error_msg = f"Error copying {module_path}: {e}"
            print(error_msg)
            errors.append(error_msg)

    print(f"\nCopied {copied_count} modules")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")

    return copied_count, errors

if __name__ == "__main__":
    copy_missing_modules()