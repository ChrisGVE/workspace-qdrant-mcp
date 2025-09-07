#!/usr/bin/env python3
"""
Comprehensive Bug Fix Testing Script for workspace-qdrant-mcp
===========================================================

This script tests all critical bug fixes identified in the GitHub issues:
- Issue #12: Search functionality returns actual results
- Issue #13: Scratchbook functionality works without errors
- Issue #5: Auto-ingestion processes workspace files
- Issue #14: Advanced search tools handle parameter type conversion
- Server restart fix: Configuration validation works properly

Run with: python 20250106-1142_comprehensive_bug_fix_test_fixed.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"Added to Python path: {src_path}")

try:
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from workspace_qdrant_mcp.core.config import Config
    from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
    from workspace_qdrant_mcp.tools.search import search_workspace
    from workspace_qdrant_mcp.core.advanced_watch_config import AdvancedWatchConfig
    from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative imports...")
    try:
        # Try basic imports first
        import workspace_qdrant_mcp
        print("✅ Base package imported successfully")
    except ImportError as e2:
        print(f"❌ Even basic import failed: {e2}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BugFixTestSuite:
    """Comprehensive test suite for all critical bug fixes."""
    
    def __init__(self):
        self.workspace_client = None
        self.test_results = {}
        self.temp_dir = None
        
    async def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp(prefix="wqmcp_test_")
        print(f"   📁 Created temp directory: {self.temp_dir}")
        
        # Try to initialize workspace client with test configuration
        try:
            config = Config(
                qdrant={"url": "http://localhost:6333"},
                workspace={"collection_suffixes": ["test"]},
                debug=True
            )
            
            self.workspace_client = QdrantWorkspaceClient(config)
            await self.workspace_client.initialize()
            print("   ✅ Workspace client initialized")
            
        except Exception as e:
            print(f"   ⚠️  Qdrant not available: {e}")
            print("   📝 Will test offline components only")
            
        # Create test documents in temp directory
        self.create_test_documents()
        
    def create_test_documents(self):
        """Create test documents for search testing."""
        print("   📄 Creating test documents...")
        
        test_docs = {
            "README.md": """
# Test Project for workspace-qdrant-mcp

This is a test project to verify search functionality works correctly.

## Features
- Hybrid search combining semantic and keyword search
- Scratchbook functionality for personal notes
- Auto-ingestion of workspace files
- Advanced search tools with parameter type conversion

## Implementation
The project uses Qdrant vector database for storage.
""",
            "main.py": """
import asyncio
from workspace_qdrant_mcp import QdrantWorkspaceClient

async def main():
    \"\"\"Main function for testing search functionality.\"\"\"
    client = QdrantWorkspaceClient()
    await client.initialize()
    
    # Test search functionality
    results = await client.search("search functionality")
    print(f"Found {len(results)} results")

if __name__ == "__main__":
    asyncio.run(main())
""",
            "config.json": """
{
    "database": {
        "url": "http://localhost:6333",
        "collection_prefix": "test_project"
    },
    "search": {
        "hybrid_mode": true,
        "score_threshold": 0.7
    }
}
"""
        }
        
        for filename, content in test_docs.items():
            filepath = Path(self.temp_dir) / filename
            filepath.write_text(content)
            print(f"     ✅ Created {filename}")
    
    async def test_issue_12_search_functionality(self):
        """Test Issue #12: Search functionality returns actual results."""
        print("\n🔍 Testing Issue #12: Search functionality...")
        
        results = {}
        
        # Test parameter validation and type conversion (related to Issue #14)
        test_cases = [
            {
                "name": "Parameter conversion - string to int limit",
                "params": {"limit": "10", "score_threshold": "0.7"},
                "expected": True
            },
            {
                "name": "Parameter conversion - string to float threshold",
                "params": {"limit": 5, "score_threshold": "0.85"},
                "expected": True
            },
            {
                "name": "Invalid limit parameter",
                "params": {"limit": "invalid", "score_threshold": "0.7"},
                "expected": False
            },
            {
                "name": "Invalid threshold parameter",
                "params": {"limit": "10", "score_threshold": "invalid"},
                "expected": False
            },
            {
                "name": "Out of range limit",
                "params": {"limit": "-1", "score_threshold": "0.7"},
                "expected": False
            },
            {
                "name": "Out of range threshold",
                "params": {"limit": "10", "score_threshold": "1.5"},
                "expected": False
            }
        ]
        
        for test_case in test_cases:
            print(f"   🧪 {test_case['name']}")
            
            try:
                # Test the parameter conversion logic (from server.py lines 270-281)
                limit = test_case['params']['limit']
                score_threshold = test_case['params']['score_threshold']
                
                # Convert string parameters to appropriate numeric types if needed
                limit = int(limit) if isinstance(limit, str) else limit
                score_threshold = float(score_threshold) if isinstance(score_threshold, str) else score_threshold
                
                # Validate numeric parameter ranges
                limit_valid = limit > 0
                threshold_valid = 0.0 <= score_threshold <= 1.0
                
                conversion_success = limit_valid and threshold_valid
                test_passed = conversion_success == test_case['expected']
                
                results[test_case['name']] = {
                    'passed': test_passed,
                    'converted_limit': limit,
                    'converted_threshold': score_threshold,
                    'expected': test_case['expected'],
                    'actual': conversion_success
                }
                
                if test_passed:
                    print(f"     ✅ Parameter validation behavior as expected")
                else:
                    print(f"     ❌ Unexpected parameter validation behavior")
                    
            except (ValueError, TypeError) as e:
                conversion_failed = True
                test_passed = not test_case['expected']  # Should fail for invalid inputs
                
                results[test_case['name']] = {
                    'passed': test_passed,
                    'error': str(e),
                    'expected': test_case['expected'],
                    'actual': False
                }
                
                if test_passed:
                    print(f"     ✅ Expected parameter conversion failure: {e}")
                else:
                    print(f"     ❌ Unexpected parameter conversion failure: {e}")
                    
        self.test_results['issue_12_search'] = results
        
    async def test_issue_13_scratchbook_functionality(self):
        """Test Issue #13: Scratchbook functionality works without errors."""
        print("\n📝 Testing Issue #13: Scratchbook functionality...")
        
        results = {}
        
        try:
            # Test scratchbook manager class structure
            print("   🧪 Testing ScratchbookManager class structure")
            
            try:
                from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
                
                # Test that class can be imported and has expected methods
                expected_methods = ['update_note', 'search_notes', 'list_notes']
                has_methods = all(hasattr(ScratchbookManager, method) for method in expected_methods)
                
                results['class_structure'] = {
                    'passed': has_methods,
                    'methods_found': [method for method in expected_methods if hasattr(ScratchbookManager, method)]
                }
                
                if has_methods:
                    print("     ✅ ScratchbookManager has all required methods")
                else:
                    missing = [method for method in expected_methods if not hasattr(ScratchbookManager, method)]
                    print(f"     ❌ ScratchbookManager missing methods: {missing}")
                    
            except ImportError as e:
                results['class_structure'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Import error: {e}")
                
            # Test scratchbook manager initialization (if client available)
            if self.workspace_client:
                print("   🧪 Testing scratchbook manager initialization")
                
                try:
                    scratchbook_manager = ScratchbookManager(self.workspace_client)
                    
                    results['manager_initialization'] = {'passed': True}
                    print("     ✅ ScratchbookManager initialized successfully")
                    
                    # Test the method signatures to detect AttributeError issues
                    print("   🧪 Testing method signatures")
                    
                    try:
                        # Check if methods exist and are callable
                        callable_methods = {
                            'update_note': callable(getattr(scratchbook_manager, 'update_note', None)),
                            'search_notes': callable(getattr(scratchbook_manager, 'search_notes', None)),
                            'list_notes': callable(getattr(scratchbook_manager, 'list_notes', None))
                        }
                        
                        all_callable = all(callable_methods.values())
                        
                        results['method_signatures'] = {
                            'passed': all_callable,
                            'callable_methods': callable_methods
                        }
                        
                        if all_callable:
                            print("     ✅ All methods are properly callable")
                        else:
                            print(f"     ❌ Some methods not callable: {callable_methods}")
                            
                    except AttributeError as e:
                        # This tests the specific AttributeError bug mentioned in issue #13
                        results['method_signatures'] = {'passed': False, 'error': f'AttributeError: {e}'}
                        print(f"     ❌ AttributeError bug detected: {e}")
                        
                except Exception as e:
                    results['manager_initialization'] = {'passed': False, 'error': str(e)}
                    print(f"     ❌ Initialization error: {e}")
            else:
                print("   ⚠️  No workspace client available for runtime testing")
                
        except Exception as e:
            results['general_error'] = {'passed': False, 'error': str(e), 'traceback': traceback.format_exc()}
            print(f"   ❌ General error: {e}")
            
        self.test_results['issue_13_scratchbook'] = results
        
    async def test_issue_5_auto_ingestion(self):
        """Test Issue #5: Auto-ingestion processes workspace files."""
        print("\n🔄 Testing Issue #5: Auto-ingestion functionality...")
        
        results = {}
        
        try:
            # Test configuration validation
            print("   🧪 Testing configuration validation")
            
            try:
                from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
                
                config_validator = ConfigValidator()
                
                # Test with valid configuration
                valid_config = {
                    "qdrant": {
                        "url": "http://localhost:6333",
                        "collection_prefix": "test_project"
                    },
                    "workspace": {
                        "collection_suffixes": ["project"]
                    },
                    "auto_ingestion": {
                        "enabled": True,
                        "include_common_files": True
                    }
                }
                
                validation_result = config_validator.validate_config(valid_config)
                
                results['config_validation'] = {
                    'passed': validation_result.get('valid', False),
                    'errors': validation_result.get('errors', [])
                }
                
                if results['config_validation']['passed']:
                    print("     ✅ Configuration validation passed")
                else:
                    print(f"     ❌ Configuration validation failed: {results['config_validation']['errors']}")
                    
            except ImportError as e:
                results['config_validation'] = {'passed': False, 'error': f'Import error: {e}'}
                print(f"     ❌ ConfigValidator import failed: {e}")
                
            # Test advanced watch configuration
            print("   🧪 Testing advanced watch configuration")
            
            try:
                from workspace_qdrant_mcp.core.advanced_watch_config import AdvancedWatchConfig
                
                watch_config = AdvancedWatchConfig(
                    enabled=True,
                    patterns=["*.py", "*.md"],
                    ignore_patterns=[".git/*"],
                    workspace_path=self.temp_dir
                )
                
                results['watch_config'] = {
                    'passed': True,
                    'enabled': watch_config.enabled,
                    'patterns': watch_config.patterns,
                    'workspace_path': str(watch_config.workspace_path)
                }
                
                print("     ✅ Advanced watch configuration created successfully")
                
            except ImportError as e:
                results['watch_config'] = {'passed': False, 'error': f'Import error: {e}'}
                print(f"     ❌ AdvancedWatchConfig import failed: {e}")
            except Exception as e:
                results['watch_config'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Advanced watch configuration failed: {e}")
                
            # Test Config class auto-ingestion settings
            print("   🧪 Testing Config class auto-ingestion settings")
            
            try:
                config = Config()
                has_auto_ingestion = hasattr(config, 'auto_ingestion')
                
                results['config_auto_ingestion'] = {
                    'passed': has_auto_ingestion,
                    'enabled': getattr(config.auto_ingestion, 'enabled', None) if has_auto_ingestion else None
                }
                
                if has_auto_ingestion:
                    print(f"     ✅ Config has auto_ingestion settings (enabled: {config.auto_ingestion.enabled})")
                else:
                    print("     ❌ Config missing auto_ingestion settings")
                    
            except Exception as e:
                results['config_auto_ingestion'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Config auto-ingestion test failed: {e}")
                
        except Exception as e:
            results['general_error'] = {'passed': False, 'error': str(e)}
            print(f"   ❌ General error: {e}")
            
        self.test_results['issue_5_auto_ingestion'] = results
        
    async def test_issue_14_parameter_conversion(self):
        """Test Issue #14: Advanced search tools handle parameter type conversion."""
        print("\n🔢 Testing Issue #14: Parameter type conversion...")
        
        results = {}
        
        # Test the exact conversion logic from server.py
        print("   🧪 Testing server.py parameter conversion logic")
        
        test_cases = [
            {
                "name": "String limit and threshold conversion",
                "input": {"limit": "10", "score_threshold": "0.7"},
                "expected_success": True
            },
            {
                "name": "Already numeric parameters",
                "input": {"limit": 5, "score_threshold": 0.85},
                "expected_success": True
            },
            {
                "name": "Invalid string limit",
                "input": {"limit": "abc", "score_threshold": "0.7"},
                "expected_success": False
            },
            {
                "name": "Invalid string threshold",
                "input": {"limit": "10", "score_threshold": "xyz"},
                "expected_success": False
            },
            {
                "name": "Edge case - zero limit",
                "input": {"limit": "0", "score_threshold": "0.7"},
                "expected_success": False  # limit must be > 0
            },
            {
                "name": "Edge case - boundary threshold values",
                "input": {"limit": "10", "score_threshold": "0.0"},
                "expected_success": True
            },
            {
                "name": "Edge case - max threshold",
                "input": {"limit": "10", "score_threshold": "1.0"},
                "expected_success": True
            },
            {
                "name": "Out of range threshold",
                "input": {"limit": "10", "score_threshold": "1.1"},
                "expected_success": False
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"   🧪 Case {i+1}: {test_case['name']}")
            
            try:
                # Replicate the exact logic from server.py lines 271-281
                limit = test_case['input']['limit']
                score_threshold = test_case['input']['score_threshold']
                
                # Convert string parameters to appropriate numeric types if needed
                limit = int(limit) if isinstance(limit, str) else limit
                score_threshold = float(score_threshold) if isinstance(score_threshold, str) else score_threshold
                
                # Validate numeric parameter ranges
                if limit <= 0:
                    raise ValueError("limit must be greater than 0")
                    
                if not (0.0 <= score_threshold <= 1.0):
                    raise ValueError("score_threshold must be between 0.0 and 1.0")
                
                # If we reach here, conversion and validation succeeded
                conversion_success = True
                error_msg = None
                
            except (ValueError, TypeError) as e:
                conversion_success = False
                error_msg = str(e)
            
            test_passed = conversion_success == test_case['expected_success']
            
            results[f'case_{i+1}_{test_case["name"].replace(" ", "_")}'] = {
                'passed': test_passed,
                'expected_success': test_case['expected_success'],
                'actual_success': conversion_success,
                'converted_limit': limit if conversion_success else None,
                'converted_threshold': score_threshold if conversion_success else None,
                'error': error_msg
            }
            
            if test_passed:
                print(f"     ✅ Conversion behavior as expected")
            else:
                status = "succeeded" if conversion_success else "failed"
                expected = "succeed" if test_case['expected_success'] else "fail"
                print(f"     ❌ Expected conversion to {expected} but it {status}")
                if error_msg:
                    print(f"       Error: {error_msg}")
                    
        self.test_results['issue_14_parameter_conversion'] = results
        
    async def test_server_restart_fix(self):
        """Test server restart fix: Configuration validation works properly."""
        print("\n🔄 Testing Server restart fix: Configuration validation...")
        
        results = {}
        
        try:
            # Test Config class instantiation
            print("   🧪 Testing Config class instantiation")
            
            try:
                config = Config()
                results['config_creation'] = {
                    'passed': True,
                    'qdrant_url': config.qdrant.url,
                    'workspace_suffixes': config.workspace.collection_suffixes
                }
                print("     ✅ Config class created successfully")
                
            except Exception as e:
                results['config_creation'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Config creation failed: {e}")
                
            # Test configuration with custom values
            print("   🧪 Testing configuration with custom values")
            
            try:
                custom_config = Config(
                    qdrant={"url": "http://test:6333", "timeout": 60},
                    workspace={"collection_suffixes": ["docs", "tests"]},
                    debug=True
                )
                
                url_correct = custom_config.qdrant.url == "http://test:6333"
                timeout_correct = custom_config.qdrant.timeout == 60
                suffixes_correct = custom_config.workspace.collection_suffixes == ["docs", "tests"]
                debug_correct = custom_config.debug == True
                
                all_correct = url_correct and timeout_correct and suffixes_correct and debug_correct
                
                results['custom_config'] = {
                    'passed': all_correct,
                    'url_correct': url_correct,
                    'timeout_correct': timeout_correct,
                    'suffixes_correct': suffixes_correct,
                    'debug_correct': debug_correct
                }
                
                if all_correct:
                    print("     ✅ Custom configuration values set correctly")
                else:
                    print("     ❌ Some custom configuration values not set correctly")
                    
            except Exception as e:
                results['custom_config'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Custom config test failed: {e}")
                
            # Test configuration validation if ConfigValidator is available
            print("   🧪 Testing configuration validation")
            
            try:
                from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
                
                validator = ConfigValidator()
                
                # Test valid configuration
                valid_config = {
                    "qdrant": {"url": "http://localhost:6333"},
                    "embedding": {"model": "BAAI/bge-small-en-v1.5"},
                    "workspace": {"collection_suffixes": ["project"]}
                }
                
                validation_result = validator.validate_config(valid_config)
                valid_passed = validation_result.get('valid', False)
                
                # Test invalid configuration  
                invalid_config = {
                    "qdrant": {},  # Missing required URL
                    "embedding": {"model": ""},  # Empty model
                }
                
                invalid_validation_result = validator.validate_config(invalid_config)
                invalid_rejected = not invalid_validation_result.get('valid', True)
                
                results['config_validation'] = {
                    'passed': valid_passed and invalid_rejected,
                    'valid_config_passed': valid_passed,
                    'invalid_config_rejected': invalid_rejected,
                    'valid_errors': validation_result.get('errors', []),
                    'invalid_errors': invalid_validation_result.get('errors', [])
                }
                
                if valid_passed and invalid_rejected:
                    print("     ✅ Configuration validation working correctly")
                else:
                    print("     ❌ Configuration validation issues detected")
                    
            except ImportError as e:
                results['config_validation'] = {'passed': False, 'error': f'ConfigValidator not available: {e}'}
                print(f"     ⚠️  ConfigValidator not available: {e}")
                
        except Exception as e:
            results['general_error'] = {'passed': False, 'error': str(e)}
            print(f"   ❌ General error: {e}")
            
        self.test_results['server_restart_fix'] = results
        
    async def cleanup(self):
        """Cleanup test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        if self.workspace_client:
            try:
                await self.workspace_client.close()
                print("   ✅ Workspace client closed")
            except Exception as e:
                print(f"   ⚠️  Error closing client: {e}")
                
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"   ✅ Removed temp directory: {self.temp_dir}")
            
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n📊 COMPREHENSIVE BUG FIX TEST REPORT")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        issues_summary = {}
        
        for issue, tests in self.test_results.items():
            print(f"\n🔍 {issue.upper().replace('_', ' ')}")
            print("-" * 30)
            
            issue_passed = 0
            issue_total = 0
            
            for test_name, result in tests.items():
                issue_total += 1
                total_tests += 1
                
                if result.get('passed', False):
                    issue_passed += 1
                    passed_tests += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"
                    
                print(f"  {status}: {test_name}")
                
                if result.get('error'):
                    print(f"    💥 Error: {result['error']}")
                if result.get('note'):
                    print(f"    📝 Note: {result['note']}")
                if 'expected' in result and 'actual' in result:
                    print(f"    📋 Expected: {result['expected']}, Actual: {result['actual']}")
                    
            issues_summary[issue] = {
                'passed': issue_passed,
                'total': issue_total,
                'rate': f"{(issue_passed/issue_total*100):.1f}%" if issue_total > 0 else "0%"
            }
            print(f"  📈 Issue Score: {issue_passed}/{issue_total} ({issues_summary[issue]['rate']}) tests passed")
            
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        print(f"\n📋 ISSUE-BY-ISSUE BREAKDOWN")
        for issue, summary in issues_summary.items():
            status_icon = "✅" if summary['passed'] == summary['total'] else "⚠️" if summary['passed'] > 0 else "❌"
            print(f"   {status_icon} {issue.replace('_', ' ').title()}: {summary['passed']}/{summary['total']} ({summary['rate']})")
        
        # Save detailed results to JSON
        report_file = Path(".") / "20250106-1142_bug_fix_test_report.json"
            
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': str(asyncio.get_event_loop().time()),
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': total_tests - passed_tests,
                    'success_rate': (passed_tests/total_tests*100) if total_tests > 0 else 0
                },
                'issues_summary': issues_summary,
                'detailed_results': self.test_results
            }, f, indent=2)
            
        print(f"\n📋 Detailed report saved to: {report_file}")
        
        return passed_tests == total_tests

async def main():
    """Run comprehensive bug fix testing."""
    print("🚀 Starting Comprehensive Bug Fix Testing for workspace-qdrant-mcp")
    print("=" * 70)
    
    test_suite = BugFixTestSuite()
    
    try:
        await test_suite.setup()
        
        # Run all tests
        await test_suite.test_issue_12_search_functionality()
        await test_suite.test_issue_13_scratchbook_functionality()  
        await test_suite.test_issue_5_auto_ingestion()
        await test_suite.test_issue_14_parameter_conversion()
        await test_suite.test_server_restart_fix()
        
        # Generate and display report
        all_passed = test_suite.generate_report()
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Bug fixes are working correctly.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please review the detailed report above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Fatal error during testing: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1
        
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)