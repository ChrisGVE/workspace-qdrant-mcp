#!/usr/bin/env python3
"""
Task 215: Validation script for logging migration progress

Validates that the unified logging system migration has been successful
and identifies remaining work.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python")

class LoggingMigrationValidator:
    """Validates the logging migration progress for Task 215."""

    def __init__(self):
        self.patterns = {
            'direct_logging_getlogger': r'logging\.getLogger\s*\(',
            'import_logging': r'^import logging$',
            'from_logging': r'^from logging import',
            'stderr_write': r'sys\.stderr\.write\s*\(',
            'stderr_dunder_write': r'sys\.__stderr__\.write\s*\(',
            'traceback_print_exc': r'traceback\.print_exc\s*\(',
            'unified_get_logger': r'get_logger\s*\(',
            'unified_import': r'from.*\.logging import get_logger',
        }

    def scan_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """Scan a file for logging patterns."""
        results = {pattern: [] for pattern in self.patterns.keys()}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_regex in self.patterns.items():
                    if re.search(pattern_regex, line, re.MULTILINE):
                        results[pattern_name].append((line_num, line.strip()))

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        return results

    def validate_project(self) -> Dict:
        """Validate the entire project logging migration."""
        print("Task 215: Logging Migration Validation")
        print("=" * 60)

        results = {
            'files_scanned': 0,
            'files_with_issues': 0,
            'files_migrated': 0,
            'total_direct_logging': 0,
            'total_stderr_usage': 0,
            'total_unified_usage': 0,
            'critical_issues': [],
            'migrated_files': [],
            'remaining_files': []
        }

        # Scan all Python files
        for py_file in BASE_DIR.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".pytest_cache" in str(py_file):
                continue

            results['files_scanned'] += 1
            file_results = self.scan_file(py_file)
            rel_path = str(py_file.relative_to(BASE_DIR))

            # Count issues
            direct_logging_count = (
                len(file_results['direct_logging_getlogger']) +
                len(file_results['import_logging']) +
                len(file_results['from_logging'])
            )

            stderr_count = (
                len(file_results['stderr_write']) +
                len(file_results['stderr_dunder_write'])
            )

            unified_count = (
                len(file_results['unified_get_logger']) +
                len(file_results['unified_import'])
            )

            traceback_count = len(file_results['traceback_print_exc'])

            # Classify files
            has_issues = direct_logging_count > 0 or stderr_count > 0 or traceback_count > 0
            is_migrated = unified_count > 0

            if has_issues:
                results['files_with_issues'] += 1
                results['remaining_files'].append({
                    'file': rel_path,
                    'direct_logging': direct_logging_count,
                    'stderr_usage': stderr_count,
                    'traceback_usage': traceback_count,
                })

                # Critical files that need immediate attention
                critical_files = [
                    'workspace_qdrant_mcp/server.py',
                    'workspace_qdrant_mcp/stdio_server.py',
                    'wqm_cli/cli_wrapper.py'
                ]
                if rel_path in critical_files:
                    results['critical_issues'].append(rel_path)

            if is_migrated:
                results['files_migrated'] += 1
                results['migrated_files'].append(rel_path)

            results['total_direct_logging'] += direct_logging_count
            results['total_stderr_usage'] += stderr_count
            results['total_unified_usage'] += unified_count

        return results

    def report_validation(self, results: Dict):
        """Generate validation report."""
        print(f"\nVALIDATION RESULTS:")
        print(f"Files scanned: {results['files_scanned']}")
        print(f"Files with issues: {results['files_with_issues']}")
        print(f"Files migrated: {results['files_migrated']}")
        print(f"Total direct logging calls: {results['total_direct_logging']}")
        print(f"Total stderr usage: {results['total_stderr_usage']}")
        print(f"Total unified logging usage: {results['total_unified_usage']}")

        print(f"\nCRITICAL FILES STATUS:")
        critical_files = [
            'workspace_qdrant_mcp/server.py',
            'workspace_qdrant_mcp/stdio_server.py',
            'wqm_cli/cli_wrapper.py'
        ]
        for critical_file in critical_files:
            if critical_file in results['critical_issues']:
                print(f"✗ {critical_file} - NEEDS MIGRATION")
            else:
                print(f"✓ {critical_file} - MIGRATED")

        print(f"\nTOP PRIORITY REMAINING FILES:")
        # Sort by issue count
        remaining_sorted = sorted(
            results['remaining_files'],
            key=lambda x: x['direct_logging'] + x['stderr_usage'] + x['traceback_usage'],
            reverse=True
        )

        for i, file_info in enumerate(remaining_sorted[:10], 1):
            total_issues = file_info['direct_logging'] + file_info['stderr_usage'] + file_info['traceback_usage']
            print(f"{i:2d}. {file_info['file']} ({total_issues} issues)")

        print(f"\nSUCCESSFULLY MIGRATED FILES:")
        for migrated_file in results['migrated_files'][:5]:  # Show first 5
            print(f"✓ {migrated_file}")
        if len(results['migrated_files']) > 5:
            print(f"   ... and {len(results['migrated_files']) - 5} more")

        # Migration progress
        if results['files_scanned'] > 0:
            migration_progress = (results['files_migrated'] / results['files_scanned']) * 100
            print(f"\nMIGRATION PROGRESS: {migration_progress:.1f}% ({results['files_migrated']}/{results['files_scanned']} files)")

        # Task 215 completion estimate
        remaining_issues = results['total_direct_logging'] + results['total_stderr_usage']
        print(f"\nTASK 215 STATUS:")
        if remaining_issues == 0:
            print("✅ TASK 215 COMPLETED - No direct logging calls remain")
        else:
            print(f"🔄 IN PROGRESS - {remaining_issues} logging calls still need migration")


def main():
    """Run the validation."""
    validator = LoggingMigrationValidator()
    results = validator.validate_project()
    validator.report_validation(results)

    # Check specific success criteria
    print(f"\n" + "=" * 60)
    print("TASK 215 SUCCESS CRITERIA CHECK:")
    print("=" * 60)

    success_criteria = [
        ("Zero direct logging.getLogger() calls remain", results['total_direct_logging'] == 0),
        ("All sys.stderr.write() removed or redirected", results['total_stderr_usage'] == 0),
        ("All modules use unified logging system", results['total_unified_usage'] > 0),
        ("Critical files migrated", len(results['critical_issues']) == 0),
    ]

    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {criterion}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 TASK 215 SUCCESSFULLY COMPLETED!")
        print("All success criteria have been met.")
    else:
        print(f"\n⚠️  TASK 215 IN PROGRESS")
        print("Some success criteria still need to be addressed.")

if __name__ == "__main__":
    main()