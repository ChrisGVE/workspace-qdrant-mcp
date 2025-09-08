#!/usr/bin/env python3
"""
Simplified Tools Interface Test Suite

This test suite validates the simplified tools interface implementation
and ensures backward compatibility with existing tool calls.

Test Coverage:
1. Core simplified tools functionality
2. Backward compatibility routing  
3. Parameter validation and conversion
4. Error handling and recovery
5. Performance comparison between modes
6. Migration path validation

Usage:
    python 20250908-1635_simplified_tools_test.py
    
Environment Variables:
    QDRANT_MCP_MODE=basic|standard|full|compatible
    QDRANT_URL=http://localhost:6333
    OPENAI_API_KEY=your_api_key_here
"""

import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

# Add the source directory to Python path
sys.path.insert(0, '/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src')

try:
    from workspace_qdrant_mcp.tools.simplified_interface import (
        SimplifiedToolsMode,
        SimplifiedToolsRouter,
        register_simplified_tools
    )
    from workspace_qdrant_mcp.tools.compatibility_layer import (
        CompatibilityMapping,
        create_compatibility_wrapper,
        should_disable_tool
    )
    from workspace_qdrant_mcp.core.config import Config
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from workspace_qdrant_mcp.observability import get_logger
except ImportError as e:
    print(f"Failed to import workspace_qdrant_mcp modules: {e}")
    print("Make sure the source path is correct and dependencies are installed")
    sys.exit(1)

logger = get_logger(__name__)


class SimplifiedToolsTestSuite:
    """Test suite for simplified tools interface."""
    
    def __init__(self):
        self.test_results = []
        self.mock_client = None
        self.mock_watch_manager = None
        self.router = None
        
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {test_name}: {details}")
        
    async def setup_test_environment(self):
        """Set up test environment with mock objects."""
        print("Setting up test environment...")
        
        # Create mock workspace client
        self.mock_client = AsyncMock()
        self.mock_client.get_status.return_value = {
            "connected": True,
            "current_project": "test-project",
            "collections_count": 3,
            "workspace_collections": ["test-project", "scratchbook", "shared"]
        }
        self.mock_client.list_collections.return_value = [
            "test-project", "scratchbook", "shared"
        ]
        
        # Create mock watch manager
        self.mock_watch_manager = AsyncMock()
        
        # Initialize router
        self.router = SimplifiedToolsRouter(self.mock_client, self.mock_watch_manager)
        
        print("Test environment setup complete")

    async def test_mode_configuration(self):
        """Test mode configuration and tool availability."""
        print("\n=== Testing Mode Configuration ===")
        
        # Test mode detection
        original_mode = os.environ.get("QDRANT_MCP_MODE", "")
        
        try:
            # Test basic mode
            os.environ["QDRANT_MCP_MODE"] = "basic"
            mode = SimplifiedToolsMode.get_mode()
            enabled_tools = SimplifiedToolsMode.get_enabled_tools()
            
            self.log_test_result(
                "basic_mode_config",
                mode == "basic" and enabled_tools == ["qdrant_store", "qdrant_find"],
                f"Mode: {mode}, Tools: {enabled_tools}"
            )
            
            # Test standard mode
            os.environ["QDRANT_MCP_MODE"] = "standard"
            mode = SimplifiedToolsMode.get_mode()
            enabled_tools = SimplifiedToolsMode.get_enabled_tools()
            
            self.log_test_result(
                "standard_mode_config",
                mode == "standard" and len(enabled_tools) == 4,
                f"Mode: {mode}, Tools: {enabled_tools}"
            )
            
            # Test full mode  
            os.environ["QDRANT_MCP_MODE"] = "full"
            mode = SimplifiedToolsMode.get_mode()
            is_simplified = SimplifiedToolsMode.is_simplified_mode()
            
            self.log_test_result(
                "full_mode_config",
                mode == "full" and not is_simplified,
                f"Mode: {mode}, Simplified: {is_simplified}"
            )
            
        finally:
            # Restore original mode
            if original_mode:
                os.environ["QDRANT_MCP_MODE"] = original_mode
            else:
                os.environ.pop("QDRANT_MCP_MODE", None)

    async def test_qdrant_store_functionality(self):
        """Test qdrant_store simplified tool."""
        print("\n=== Testing qdrant_store Functionality ===")
        
        # Mock successful document storage
        async def mock_add_document(*args, **kwargs):
            return {
                "success": True,
                "document_id": "doc-123",
                "collection": kwargs.get("collection", "test-project"),
                "chunks_added": 1
            }
            
        async def mock_update_scratchbook(*args, **kwargs):
            return {
                "success": True,
                "note_id": "note-456", 
                "collection": "scratchbook",
                "note_type": kwargs.get("note_type", "scratchbook")
            }
        
        # Test basic document storage
        try:
            # Mock the document add function
            import workspace_qdrant_mcp.tools.documents as docs_module
            original_add_document = docs_module.add_document
            docs_module.add_document = mock_add_document
            
            result = await self.router.qdrant_store(
                information="Test document content",
                collection="test-project",
                note_type="document"
            )
            
            self.log_test_result(
                "qdrant_store_document",
                result.get("success") and result.get("document_id"),
                f"Result: {result}"
            )
            
            # Restore original function
            docs_module.add_document = original_add_document
            
        except Exception as e:
            self.log_test_result(
                "qdrant_store_document",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test scratchbook storage
        try:
            # Mock the scratchbook function
            import workspace_qdrant_mcp.tools.scratchbook as scratchbook_module
            original_update_scratchbook = scratchbook_module.update_scratchbook
            scratchbook_module.update_scratchbook = mock_update_scratchbook
            
            result = await self.router.qdrant_store(
                information="Test note content",
                note_type="scratchbook",
                title="Test Note",
                tags=["test", "validation"]
            )
            
            self.log_test_result(
                "qdrant_store_scratchbook", 
                result.get("success") and result.get("note_id"),
                f"Result: {result}"
            )
            
            # Restore original function
            scratchbook_module.update_scratchbook = original_update_scratchbook
            
        except Exception as e:
            self.log_test_result(
                "qdrant_store_scratchbook",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test parameter validation
        try:
            result = await self.router.qdrant_store(
                information=""  # Empty content should still work
            )
            
            self.log_test_result(
                "qdrant_store_validation",
                "error" not in result or "empty" not in result.get("error", "").lower(),
                "Empty content handled gracefully"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_store_validation",
                False,
                f"Exception: {str(e)}"
            )

    async def test_qdrant_find_functionality(self):
        """Test qdrant_find simplified tool."""
        print("\n=== Testing qdrant_find Functionality ===")
        
        # Mock successful search results
        async def mock_search_workspace(*args, **kwargs):
            return {
                "query": kwargs.get("query", "test"),
                "mode": kwargs.get("mode", "hybrid"),
                "collections_searched": ["test-project"],
                "total_results": 2,
                "results": [
                    {
                        "id": "doc-1",
                        "score": 0.95,
                        "payload": {"content": "Test document 1"},
                        "collection": "test-project"
                    },
                    {
                        "id": "doc-2", 
                        "score": 0.87,
                        "payload": {"content": "Test document 2"},
                        "collection": "test-project"
                    }
                ]
            }
        
        # Test basic search
        try:
            # Mock the search function
            import workspace_qdrant_mcp.tools.search as search_module
            original_search_workspace = search_module.search_workspace
            search_module.search_workspace = mock_search_workspace
            
            result = await self.router.qdrant_find(
                query="test search",
                collection="test-project",
                search_mode="hybrid",
                limit=10
            )
            
            self.log_test_result(
                "qdrant_find_basic",
                result.get("total_results", 0) > 0 and len(result.get("results", [])) > 0,
                f"Found {result.get('total_results', 0)} results"
            )
            
            # Restore original function
            search_module.search_workspace = original_search_workspace
            
        except Exception as e:
            self.log_test_result(
                "qdrant_find_basic",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test parameter validation
        test_cases = [
            {"limit": 0, "should_fail": True, "error_contains": "limit must be greater than 0"},
            {"score_threshold": -0.1, "should_fail": True, "error_contains": "score_threshold must be between"},
            {"score_threshold": 1.1, "should_fail": True, "error_contains": "score_threshold must be between"},
            {"limit": "invalid", "should_fail": True, "error_contains": "Invalid parameter types"},
            {"limit": 5, "score_threshold": 0.8, "should_fail": False, "error_contains": ""}
        ]
        
        for i, case in enumerate(test_cases):
            try:
                result = await self.router.qdrant_find(
                    query="test",
                    limit=case.get("limit", 10),
                    score_threshold=case.get("score_threshold", 0.7)
                )
                
                has_error = "error" in result
                if case["should_fail"]:
                    test_passed = has_error and case["error_contains"] in result.get("error", "")
                else:
                    test_passed = not has_error
                
                self.log_test_result(
                    f"qdrant_find_validation_{i}",
                    test_passed,
                    f"Case: {case}, Result: {result.get('error', 'Success')}"
                )
                
            except Exception as e:
                self.log_test_result(
                    f"qdrant_find_validation_{i}",
                    case["should_fail"],
                    f"Exception: {str(e)}"
                )

    async def test_qdrant_manage_functionality(self):
        """Test qdrant_manage simplified tool."""
        print("\n=== Testing qdrant_manage Functionality ===")
        
        # Test status action
        try:
            result = await self.router.qdrant_manage(action="status")
            
            self.log_test_result(
                "qdrant_manage_status",
                result.get("connected") is True and "current_project" in result,
                f"Status: {result}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_manage_status",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test collections action
        try:
            result = await self.router.qdrant_manage(action="collections")
            
            self.log_test_result(
                "qdrant_manage_collections",
                "collections" in result and len(result.get("collections", [])) > 0,
                f"Collections: {result}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_manage_collections", 
                False,
                f"Exception: {str(e)}"
            )
        
        # Test invalid action
        try:
            result = await self.router.qdrant_manage(action="invalid_action")
            
            self.log_test_result(
                "qdrant_manage_invalid_action",
                "error" in result and "Unknown management action" in result.get("error", ""),
                f"Error handled: {result.get('error', '')}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_manage_invalid_action",
                False,
                f"Exception: {str(e)}"
            )

    async def test_qdrant_watch_functionality(self):
        """Test qdrant_watch simplified tool."""
        print("\n=== Testing qdrant_watch Functionality ===")
        
        # Mock watch manager responses
        self.mock_watch_manager.add_watch_folder.return_value = {
            "success": True,
            "watch_id": "watch-123",
            "path": "/test/path",
            "collection": "test-project"
        }
        
        self.mock_watch_manager.list_watched_folders.return_value = {
            "summary": {"total_watches": 1},
            "watches": [{"id": "watch-123", "path": "/test/path"}]
        }
        
        # Test add action
        try:
            result = await self.router.qdrant_watch(
                action="add",
                path="/test/path",
                collection="test-project",
                patterns=["*.pdf", "*.txt"],
                recursive=True
            )
            
            self.log_test_result(
                "qdrant_watch_add",
                result.get("success") and result.get("watch_id"),
                f"Watch added: {result}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_watch_add",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test list action
        try:
            result = await self.router.qdrant_watch(action="list")
            
            self.log_test_result(
                "qdrant_watch_list",
                "summary" in result and result.get("summary", {}).get("total_watches", 0) > 0,
                f"Watch list: {result}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_watch_list",
                False,
                f"Exception: {str(e)}"
            )
        
        # Test parameter validation
        try:
            result = await self.router.qdrant_watch(
                action="add"
                # Missing required path and collection
            )
            
            self.log_test_result(
                "qdrant_watch_validation",
                "error" in result and "required" in result.get("error", ""),
                f"Validation error: {result.get('error', '')}"
            )
            
        except Exception as e:
            self.log_test_result(
                "qdrant_watch_validation",
                False,
                f"Exception: {str(e)}"
            )

    async def test_compatibility_mapping(self):
        """Test backward compatibility mapping."""
        print("\n=== Testing Compatibility Mapping ===")
        
        # Test tool mapping existence
        mapped_tools = list(CompatibilityMapping.TOOL_MAPPINGS.keys())
        expected_tools = [
            "add_document_tool",
            "search_workspace_tool", 
            "workspace_status",
            "add_watch_folder"
        ]
        
        missing_tools = [tool for tool in expected_tools if tool not in mapped_tools]
        
        self.log_test_result(
            "compatibility_mapping_coverage",
            len(missing_tools) == 0,
            f"Mapped tools: {len(mapped_tools)}, Missing: {missing_tools}"
        )
        
        # Test parameter mapping
        add_doc_mapping = CompatibilityMapping.TOOL_MAPPINGS.get("add_document_tool", {})
        param_mapping = add_doc_mapping.get("param_mapping", {})
        
        self.log_test_result(
            "compatibility_parameter_mapping",
            "content" in param_mapping and param_mapping["content"] == "information",
            f"Parameter mapping: {param_mapping}"
        )
        
        # Test tool disabling in simplified mode
        os.environ["QDRANT_MCP_MODE"] = "basic"
        
        should_disable_mapped = should_disable_tool("add_document_tool")
        should_disable_unmapped = should_disable_tool("unknown_tool")
        
        self.log_test_result(
            "compatibility_tool_disabling",
            should_disable_mapped and not should_disable_unmapped,
            f"Mapped tool disabled: {should_disable_mapped}, Unmapped: {should_disable_unmapped}"
        )

    async def test_performance_comparison(self):
        """Test performance differences between modes."""
        print("\n=== Testing Performance Comparison ===")
        
        # Simple performance test - tool registration time
        modes = ["basic", "standard", "full"]
        performance_results = {}
        
        for mode in modes:
            os.environ["QDRANT_MCP_MODE"] = mode
            
            start_time = time.time()
            
            # Simulate tool registration process
            enabled_tools = SimplifiedToolsMode.get_enabled_tools()
            is_simplified = SimplifiedToolsMode.is_simplified_mode()
            
            registration_time = time.time() - start_time
            
            performance_results[mode] = {
                "registration_time": registration_time,
                "tool_count": len(enabled_tools) if enabled_tools else 30,  # Assume 30 for full mode
                "is_simplified": is_simplified
            }
        
        # Compare basic vs full mode
        basic_time = performance_results["basic"]["registration_time"]
        full_time = performance_results["full"]["registration_time"] 
        
        self.log_test_result(
            "performance_registration_time",
            basic_time <= full_time * 2,  # Basic should not be significantly slower
            f"Basic: {basic_time:.4f}s, Full: {full_time:.4f}s, Results: {performance_results}"
        )

    async def test_error_handling(self):
        """Test error handling in simplified interface."""
        print("\n=== Testing Error Handling ===")
        
        # Test router without client
        router_no_client = SimplifiedToolsRouter(None, None)
        
        try:
            result = await router_no_client.qdrant_store(
                information="test"
            )
            
            self.log_test_result(
                "error_handling_no_client",
                "error" in result and "not initialized" in result.get("error", ""),
                f"Error handled: {result.get('error', '')}"
            )
            
        except Exception as e:
            self.log_test_result(
                "error_handling_no_client",
                False,
                f"Exception not handled: {str(e)}"
            )
        
        # Test invalid parameters
        try:
            result = await self.router.qdrant_find(
                query="test",
                limit="invalid_number"
            )
            
            self.log_test_result(
                "error_handling_invalid_params",
                "error" in result and "Invalid parameter types" in result.get("error", ""),
                f"Parameter error handled: {result.get('error', '')}"
            )
            
        except Exception as e:
            self.log_test_result(
                "error_handling_invalid_params",
                False,
                f"Exception not handled: {str(e)}"
            )

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "recommendations": []
        }
        
        # Add recommendations based on results
        if failed_tests > 0:
            failed_categories = {}
            for result in self.test_results:
                if not result["passed"]:
                    category = result["test"].split("_")[0]
                    failed_categories[category] = failed_categories.get(category, 0) + 1
            
            report["recommendations"].append(
                f"Review failed tests in categories: {list(failed_categories.keys())}"
            )
        
        if passed_tests / total_tests < 0.8:
            report["recommendations"].append(
                "Success rate below 80% - review implementation before production deployment"
            )
        
        return report

    async def run_all_tests(self):
        """Run complete test suite."""
        print("=== Simplified Tools Interface Test Suite ===")
        print(f"Python version: {sys.version}")
        print(f"Current mode: {SimplifiedToolsMode.get_mode()}")
        print("")
        
        await self.setup_test_environment()
        
        # Run all test categories
        test_methods = [
            self.test_mode_configuration,
            self.test_qdrant_store_functionality,
            self.test_qdrant_find_functionality, 
            self.test_qdrant_manage_functionality,
            self.test_qdrant_watch_functionality,
            self.test_compatibility_mapping,
            self.test_performance_comparison,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"Test method {test_method.__name__} failed with exception: {e}")
                self.log_test_result(
                    f"{test_method.__name__}_exception",
                    False,
                    f"Unhandled exception: {str(e)}"
                )
        
        # Generate and display report
        report = self.generate_test_report()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if report["recommendations"]:
            print("\nRECOMMENDations:")
            for rec in report["recommendations"]:
                print(f"- {rec}")
        
        # Save detailed report
        report_file = "simplified_tools_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
        
        return report


async def main():
    """Main test execution."""
    # Set default test mode if not specified
    if "QDRANT_MCP_MODE" not in os.environ:
        os.environ["QDRANT_MCP_MODE"] = "standard"
    
    test_suite = SimplifiedToolsTestSuite()
    report = await test_suite.run_all_tests()
    
    # Exit with error code if tests failed
    exit_code = 0 if report["summary"]["success_rate"] >= 80.0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())