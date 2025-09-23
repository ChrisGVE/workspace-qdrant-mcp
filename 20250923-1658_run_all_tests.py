#!/usr/bin/env python3
"""
Run all client tests and gather coverage information without pytest hanging issues
"""
import subprocess
import sys
import time
from pathlib import Path

def run_test_batch(test_patterns, timeout=30):
    """Run a batch of tests with timeout."""
    results = []

    for pattern in test_patterns:
        try:
            print(f"Running: {pattern}")
            start_time = time.time()

            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"tests/unit/test_core_client.py::{pattern}",
                "-v", "--tb=short", "--no-cov"  # Disable coverage for now
            ],
            cwd=Path.cwd(),
            timeout=timeout,
            capture_output=True,
            text=True
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                results.append((pattern, "PASS", duration, ""))
                print(f"  ✓ PASS ({duration:.1f}s)")
            else:
                results.append((pattern, "FAIL", duration, result.stdout + result.stderr))
                print(f"  ✗ FAIL ({duration:.1f}s)")

        except subprocess.TimeoutExpired:
            results.append((pattern, "TIMEOUT", timeout, "Test timed out"))
            print(f"  ⏰ TIMEOUT ({timeout}s)")
        except Exception as e:
            results.append((pattern, "ERROR", 0, str(e)))
            print(f"  ❌ ERROR: {e}")

    return results

def main():
    """Run comprehensive test coverage."""

    # Define test batches (simple tests first)
    simple_tests = [
        "TestQdrantWorkspaceClient::test_init",
        "TestQdrantWorkspaceClient::test_get_project_info",
        "TestQdrantWorkspaceClient::test_get_project_context_no_project",
        "TestQdrantWorkspaceClient::test_get_project_context_success",
        "TestQdrantWorkspaceClient::test_generate_project_id",
        "TestQdrantWorkspaceClient::test_refresh_project_detection",
        "TestQdrantWorkspaceClient::test_get_embedding_service",
        "TestQdrantWorkspaceClient::test_list_collections_not_initialized",
        "TestQdrantWorkspaceClient::test_list_collections_success",
        "TestQdrantWorkspaceClient::test_list_collections_exception",
        "TestCreateQdrantClient::test_create_qdrant_client",
    ]

    non_async_tests = [
        "TestQdrantWorkspaceClient::test_get_enhanced_collection_selector_not_initialized",
        "TestQdrantWorkspaceClient::test_get_enhanced_collection_selector_success",
        "TestQdrantWorkspaceClient::test_select_collections_by_type_not_initialized",
        "TestQdrantWorkspaceClient::test_select_collections_by_type_success",
        "TestQdrantWorkspaceClient::test_get_searchable_collections_not_initialized",
        "TestQdrantWorkspaceClient::test_get_searchable_collections_success",
        "TestQdrantWorkspaceClient::test_validate_collection_access_not_initialized",
        "TestQdrantWorkspaceClient::test_validate_collection_access_success",
    ]

    edge_case_tests = [
        "TestEdgeCases::test_ensure_collection_exists_exception",
        "TestEdgeCases::test_get_project_context_empty_project",
        "TestEdgeCases::test_select_collections_by_type_exception",
        "TestEdgeCases::test_get_searchable_collections_exception_fallback",
        "TestEdgeCases::test_validate_collection_access_exception",
    ]

    async_tests = [
        "TestQdrantWorkspaceClient::test_get_status_not_initialized",
        "TestQdrantWorkspaceClient::test_get_status_success",
        "TestQdrantWorkspaceClient::test_get_status_exception",
        "TestQdrantWorkspaceClient::test_search_with_project_context_not_initialized",
        "TestQdrantWorkspaceClient::test_search_with_project_context_success",
        "TestQdrantWorkspaceClient::test_search_with_project_context_exception",
        "TestQdrantWorkspaceClient::test_ensure_collection_exists_not_initialized",
        "TestQdrantWorkspaceClient::test_ensure_collection_exists_empty_name",
        "TestQdrantWorkspaceClient::test_ensure_collection_exists_success",
        "TestQdrantWorkspaceClient::test_create_collection_not_initialized",
        "TestQdrantWorkspaceClient::test_create_collection_legacy_mode",
        "TestQdrantWorkspaceClient::test_close",
        "TestQdrantWorkspaceClient::test_close_with_none_services",
    ]

    # Complex async tests that might hang
    complex_async_tests = [
        "TestQdrantWorkspaceClient::test_initialize_success",
        "TestQdrantWorkspaceClient::test_initialize_already_initialized",
        "TestQdrantWorkspaceClient::test_initialize_connection_error",
    ]

    print("="*80)
    print("COMPREHENSIVE CLIENT TEST EXECUTION")
    print("="*80)

    all_results = []

    # Run test batches
    print("\n1. Running simple tests...")
    all_results.extend(run_test_batch(simple_tests, timeout=15))

    print("\n2. Running non-async tests...")
    all_results.extend(run_test_batch(non_async_tests, timeout=15))

    print("\n3. Running edge case tests...")
    all_results.extend(run_test_batch(edge_case_tests, timeout=15))

    print("\n4. Running async tests...")
    all_results.extend(run_test_batch(async_tests, timeout=20))

    print("\n5. Running complex async tests...")
    all_results.extend(run_test_batch(complex_async_tests, timeout=30))

    # Summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)

    passed = sum(1 for _, status, _, _ in all_results if status == "PASS")
    failed = sum(1 for _, status, _, _ in all_results if status == "FAIL")
    timeout = sum(1 for _, status, _, _ in all_results if status == "TIMEOUT")
    error = sum(1 for _, status, _, _ in all_results if status == "ERROR")
    total = len(all_results)

    print(f"TOTAL TESTS: {total}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TIMEOUT: {timeout}")
    print(f"ERROR: {error}")
    print(f"SUCCESS RATE: {passed/total*100:.1f}%")

    # Show failures
    if failed > 0 or timeout > 0 or error > 0:
        print("\n" + "="*80)
        print("FAILED/TIMEOUT/ERROR TESTS")
        print("="*80)
        for test_name, status, duration, output in all_results:
            if status != "PASS":
                print(f"\n{status}: {test_name}")
                if output and len(output) < 500:
                    print(f"Output: {output[:500]}")

    print("\n" + "="*80)
    print("NEXT STEPS FOR 100% COVERAGE")
    print("="*80)

    if passed >= 35:  # Most tests passing
        print("✓ Most tests are passing! Ready for coverage analysis.")
        print("✓ Focus on adding tests for uncovered edge cases:")
        print("  - SSL localhost detection in initialize()")
        print("  - Authentication token handling")
        print("  - Exception paths in complex methods")
        print("  - Import error fallbacks")
        print("  - Validation failures")
    else:
        print("❗ Need to fix failing tests before proceeding to coverage analysis")

if __name__ == "__main__":
    main()