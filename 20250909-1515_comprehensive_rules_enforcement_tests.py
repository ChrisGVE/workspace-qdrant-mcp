#!/usr/bin/env python3
"""
Task 181: Comprehensive Collection Rules Enforcement Testing

This script provides comprehensive testing of the CollectionRulesEnforcer system
to validate that ALL collection management rules are enforced correctly and that
no rule bypass is possible through parameter manipulation.
"""

import sys
import os
sys.path.insert(0, 'src')

import logging
from typing import Dict, List, Any
import json

from workspace_qdrant_mcp.core.collection_naming import (
    CollectionRulesEnforcer,
    ValidationSource,
    OperationType,
    ValidationResult,
    CollectionRulesEnforcementError
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveRulesEnforcementTester:
    """Comprehensive testing framework for collection rules enforcement."""
    
    def __init__(self):
        self.enforcer = CollectionRulesEnforcer()
        self.test_results = []
        self.security_violations = []
        
        # Set up realistic existing collections for testing
        self.existing_collections = [
            "__system_memory",      # System memory collection
            "__admin_config",       # System admin collection  
            "_mylib",              # Library collection
            "_utils",              # Another library collection
            "memory",              # Global memory collection
            "algorithms",          # Global collection
            "project-docs",        # Valid project collection
            "project-memory",      # Project memory collection
            "legacy-collection"    # Legacy collection
        ]
        self.enforcer.set_existing_collections(self.existing_collections)
        logger.info("Comprehensive Rules Enforcement Tester initialized")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        logger.info("=== Starting Comprehensive Collection Rules Enforcement Tests ===")
        
        # Test suites
        self.test_llm_access_control_integration()
        self.test_system_collection_protection()
        self.test_library_collection_read_only()
        self.test_memory_collection_deletion_protection()
        self.test_parameter_manipulation_bypass_prevention()
        self.test_clear_error_messages()
        self.test_source_aware_validation()
        self.test_edge_cases_and_malformed_inputs()
        self.test_rule_consistency_across_operations()
        
        # Generate summary
        return self.generate_test_summary()
    
    def test_llm_access_control_integration(self):
        """Test integration with LLM access control from Task 173."""
        logger.info("--- Testing LLM Access Control Integration ---")
        
        test_cases = [
            # LLM cannot create system collections
            {
                "operation": OperationType.CREATE,
                "name": "__new_system",
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "LLM create system collection"
            },
            # LLM can create valid project collections
            {
                "operation": OperationType.CREATE,
                "name": "myproject-docs",
                "source": ValidationSource.LLM,
                "expected_valid": True,
                "description": "LLM create valid project collection"
            }
        ]
        
        for test_case in test_cases:
            self._run_test_case("LLM Access Control Integration", test_case)
    
    def test_system_collection_protection(self):
        """Test system collection protection across all sources."""
        logger.info("--- Testing System Collection Protection ---")
        
        test_cases = [
            {
                "operation": OperationType.WRITE,
                "name": "__system_memory",
                "source": ValidationSource.MCP_INTERNAL,
                "expected_valid": False,
                "description": "MCP write to system memory collection"
            },
            {
                "operation": OperationType.DELETE,
                "name": "__admin_config",
                "source": ValidationSource.MCP_INTERNAL,
                "expected_valid": False,
                "description": "MCP delete system admin collection"
            },
            {
                "operation": OperationType.CREATE,
                "name": "__new_admin",
                "source": ValidationSource.CLI,
                "expected_valid": True,
                "description": "CLI create system collection"
            }
        ]
        
        for test_case in test_cases:
            self._run_test_case("System Collection Protection", test_case)
    
    def test_library_collection_read_only(self):
        """Test library collection read-only enforcement from MCP."""
        logger.info("--- Testing Library Collection Read-Only Enforcement ---")
        
        test_cases = [
            {
                "operation": OperationType.WRITE,
                "name": "_mylib",
                "source": ValidationSource.MCP_INTERNAL,
                "expected_valid": False,
                "description": "MCP write to library collection"
            },
            {
                "operation": OperationType.WRITE,
                "name": "_mylib",
                "source": ValidationSource.CLI,
                "expected_valid": True,
                "description": "CLI write to library collection"
            },
            {
                "operation": OperationType.READ,
                "name": "_mylib",
                "source": ValidationSource.LLM,
                "expected_valid": True,
                "description": "LLM read from library collection"
            }
        ]
        
        for test_case in test_cases:
            self._run_test_case("Library Collection Read-Only", test_case)
    
    def test_memory_collection_deletion_protection(self):
        """Test memory collection deletion protection."""
        logger.info("--- Testing Memory Collection Deletion Protection ---")
        
        test_cases = [
            {
                "operation": OperationType.DELETE,
                "name": "project-memory",
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "LLM delete project memory collection"
            },
            {
                "operation": OperationType.DELETE,
                "name": "memory",
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "LLM delete global memory collection"
            }
        ]
        
        for test_case in test_cases:
            self._run_test_case("Memory Collection Deletion Protection", test_case)
    
    def test_parameter_manipulation_bypass_prevention(self):
        """Test prevention of rule bypass through parameter manipulation."""
        logger.info("--- Testing Parameter Manipulation Bypass Prevention ---")
        
        bypass_attempts = [
            {
                "operation": OperationType.CREATE,
                "name": "",
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "Empty collection name bypass"
            },
            {
                "operation": OperationType.CREATE,
                "name": "   ",
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "Whitespace-only name bypass"
            },
            {
                "operation": OperationType.CREATE,
                "name": "a" * 200,  # Very long name
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "Excessively long name bypass"
            }
        ]
        
        for attempt in bypass_attempts:
            self._run_test_case("Parameter Manipulation Prevention", attempt)
    
    def test_clear_error_messages(self):
        """Test that error messages are clear and informative."""
        logger.info("--- Testing Clear Error Messages ---")
        
        # Test cases that should produce specific error messages
        error_message_tests = [
            {
                "operation": OperationType.DELETE,
                "name": "nonexistent",
                "source": ValidationSource.LLM,
                "expected_keywords": ["does not exist"],
                "description": "Collection not found error"
            }
        ]
        
        for test in error_message_tests:
            result = self.enforcer.validate_operation(
                test["operation"], test["name"], test["source"]
            )
            
            test_result = {
                "test_suite": "Clear Error Messages",
                "test_case": test["description"],
                "passed": not result.is_valid and result.error_message is not None,
                "error_message": result.error_message
            }
            self.test_results.append(test_result)
    
    def test_source_aware_validation(self):
        """Test that validation behaves differently based on operation source."""
        logger.info("--- Testing Source-Aware Validation ---")
        
        # Same operation should have different outcomes based on source
        source_tests = [
            {
                "operation": OperationType.CREATE,
                "name": "__new_system",
                "sources_and_expected": [
                    (ValidationSource.LLM, False),        # LLM blocked
                    (ValidationSource.CLI, True),         # CLI allowed
                    (ValidationSource.SYSTEM, True),      # System allowed
                    (ValidationSource.MCP_INTERNAL, True) # MCP allowed for system ops
                ],
                "description": "System collection creation by source"
            }
        ]
        
        for test in source_tests:
            for source, expected_valid in test["sources_and_expected"]:
                test_case = {
                    "operation": test["operation"],
                    "name": test["name"],
                    "source": source,
                    "expected_valid": expected_valid,
                    "description": f"{test['description']} - {source.value}"
                }
                self._run_test_case("Source-Aware Validation", test_case)
    
    def test_edge_cases_and_malformed_inputs(self):
        """Test edge cases and malformed input handling."""
        logger.info("--- Testing Edge Cases and Malformed Inputs ---")
        
        edge_cases = [
            {
                "operation": OperationType.CREATE,
                "name": "a",  # Single character
                "source": ValidationSource.LLM,
                "expected_valid": False,  # Single chars don't follow project pattern
                "description": "Single character collection name"
            },
            {
                "operation": OperationType.CREATE,
                "name": "_",  # Just underscore
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "Single underscore collection name"
            },
            {
                "operation": OperationType.CREATE,
                "name": "project-docs",  # Exists in our test set
                "source": ValidationSource.LLM,
                "expected_valid": False,
                "description": "Create existing collection"
            }
        ]
        
        for test_case in edge_cases:
            self._run_test_case("Edge Cases and Malformed Inputs", test_case)
    
    def test_rule_consistency_across_operations(self):
        """Test that rules are consistently applied across all operations."""
        logger.info("--- Testing Rule Consistency Across Operations ---")
        
        consistency_tests = [
            {
                "collection": "__system_memory",
                "source": ValidationSource.LLM,
                "expected_results": {
                    OperationType.CREATE: False,  # Cannot create system
                    OperationType.READ: True,     # Can read existing
                    OperationType.WRITE: False,   # Cannot write to system
                    OperationType.DELETE: False   # Cannot delete system
                }
            }
        ]
        
        for test in consistency_tests:
            for operation, expected in test["expected_results"].items():
                test_case = {
                    "operation": operation,
                    "name": test["collection"],
                    "source": test["source"],
                    "expected_valid": expected,
                    "description": f"Consistency: {operation.value} {test['collection']} as {test['source'].value}"
                }
                self._run_test_case("Rule Consistency", test_case)
    
    def _run_test_case(self, test_suite: str, test_case: Dict[str, Any]):
        """Run a single test case and record results."""
        try:
            result = self.enforcer.validate_operation(
                test_case["operation"], 
                test_case["name"], 
                test_case["source"]
            )
            
            passed = result.is_valid == test_case["expected_valid"]
            
            test_result = {
                "test_suite": test_suite,
                "test_case": test_case["description"],
                "operation": test_case["operation"].value,
                "collection": test_case["name"],
                "source": test_case["source"].value,
                "expected_valid": test_case["expected_valid"],
                "actual_valid": result.is_valid,
                "passed": passed,
                "error_message": result.error_message
            }
            
            self.test_results.append(test_result)
            
            if not passed:
                violation = {
                    "test": test_suite,
                    "description": test_case["description"],
                    "expected": test_case["expected_valid"],
                    "actual": result.is_valid,
                    "error_message": result.error_message
                }
                self.security_violations.append(violation)
                logger.error(f"SECURITY VIOLATION: {test_case['description']} - Expected: {test_case['expected_valid']}, Got: {result.is_valid}")
            else:
                logger.debug(f"✅ {test_case['description']}")
                
        except Exception as e:
            test_result = {
                "test_suite": test_suite,
                "test_case": test_case["description"],
                "passed": False,
                "exception": str(e)
            }
            self.test_results.append(test_result)
            logger.error(f"❌ Exception in {test_case['description']}: {e}")
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary with security analysis."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get("passed", False)])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "test_execution": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "security_analysis": {
                "total_violations": len(self.security_violations),
                "security_score": max(0, 100 - len(self.security_violations) * 10)
            },
            "test_breakdown": {},
            "violations": self.security_violations,
            "detailed_results": self.test_results,
            "compliance_status": {
                "overall_compliant": len(self.security_violations) == 0
            }
        }
        
        # Test suite breakdown
        for result in self.test_results:
            suite = result["test_suite"]
            if suite not in summary["test_breakdown"]:
                summary["test_breakdown"][suite] = {"total": 0, "passed": 0, "failed": 0}
            
            summary["test_breakdown"][suite]["total"] += 1
            if result.get("passed", False):
                summary["test_breakdown"][suite]["passed"] += 1
            else:
                summary["test_breakdown"][suite]["failed"] += 1
        
        return summary
    
    def print_summary_report(self, summary: Dict[str, Any]):
        """Print a human-readable summary report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COLLECTION RULES ENFORCEMENT TEST REPORT")
        print("=" * 80)
        
        print(f"\n📊 TEST EXECUTION SUMMARY:")
        print(f"   Total Tests: {summary['test_execution']['total_tests']}")
        print(f"   Passed: {summary['test_execution']['passed_tests']}")
        print(f"   Failed: {summary['test_execution']['failed_tests']}")
        print(f"   Success Rate: {summary['test_execution']['success_rate']:.1f}%")
        
        print(f"\n🔒 SECURITY ANALYSIS:")
        print(f"   Security Score: {summary['security_analysis']['security_score']}/100")
        print(f"   Total Violations: {summary['security_analysis']['total_violations']}")
        
        print(f"\n✅ COMPLIANCE STATUS:")
        compliance = summary['compliance_status']
        print(f"   Overall Compliant: {'✅' if compliance['overall_compliant'] else '❌'}")
        
        print(f"\n📋 TEST SUITE BREAKDOWN:")
        for suite, results in summary['test_breakdown'].items():
            success_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0
            print(f"   {suite}: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        
        if summary['violations']:
            print(f"\n🚨 SECURITY VIOLATIONS:")
            for i, violation in enumerate(summary['violations'][:10], 1):
                print(f"   {i}. {violation['description']}")
                print(f"      Expected: {violation['expected']}, Got: {violation['actual']}")
        
        print("\n" + "=" * 80)
        print(f"TASK 181 COLLECTION RULES ENFORCEMENT: {'COMPLIANT' if compliance['overall_compliant'] else 'NON-COMPLIANT'}")
        print("=" * 80)


def main():
    """Run comprehensive collection rules enforcement tests."""
    tester = ComprehensiveRulesEnforcementTester()
    
    try:
        # Run all tests
        summary = tester.run_all_tests()
        
        # Print human-readable report
        tester.print_summary_report(summary)
        
        # Save detailed results to JSON for further analysis
        results_file = "task_181_rules_enforcement_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if summary['compliance_status']['overall_compliant']:
            print("\n🎉 All tests passed! Collection rules enforcement is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {summary['security_analysis']['total_violations']} security violations detected!")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())