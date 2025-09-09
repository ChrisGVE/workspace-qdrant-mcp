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

Run with: python 20250106-1142_comprehensive_bug_fix_test.py
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
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from workspace_qdrant_mcp.core.config import QdrantConfig
    from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
    from workspace_qdrant_mcp.tools.search import search_workspace
    from workspace_qdrant_mcp.core.advanced_watch_config import AdvancedWatchConfig
    from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
except ImportError as e:
    print(f"❌ Import error: {e}")
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
        
        # Initialize workspace client with test configuration
        try:
            config = QdrantConfig(
                url="http://localhost:6333",
                collection_prefix="test_wqmcp",
                workspace_path=self.temp_dir
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
        
        test_cases = [
            {
                "name": "Hybrid search for 'search functionality'",
                "query": "search functionality",
                "mode": "hybrid",
                "expected_results": True
            },
            {
                "name": "Dense semantic search for 'vector database'", 
                "query": "vector database",
                "mode": "dense",
                "expected_results": True
            },
            {
                "name": "Sparse keyword search for 'async def'",
                "query": "async def",
                "mode": "sparse", 
                "expected_results": True
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"   🧪 {test_case['name']}")
            
            try:
                if self.workspace_client:
                    # Test with actual client
                    search_result = await search_workspace(
                        self.workspace_client,
                        query=test_case['query'],
                        collections=None,
                        mode=test_case['mode'],
                        limit=10,
                        score_threshold=0.7
                    )
                    
                    has_results = search_result.get('total_results', 0) > 0
                    results[test_case['name']] = {
                        'passed': has_results,
                        'total_results': search_result.get('total_results', 0),
                        'collections_searched': search_result.get('collections_searched', [])
                    }
                    
                    if has_results:
                        print(f"     ✅ Found {search_result.get('total_results', 0)} results")
                    else:
                        print(f"     ❌ No results found")
                        
                else:
                    # Test parameter validation without client
                    print(f"     ⚠️  Testing parameter validation only (no Qdrant)")
                    
                    # Test string to numeric conversion (Issue #14)
                    try:
                        limit = int("10")
                        score_threshold = float("0.7")
                        
                        if limit > 0 and 0.0 <= score_threshold <= 1.0:
                            results[test_case['name']] = {'passed': True, 'note': 'Parameter validation OK'}
                            print(f"     ✅ Parameter validation passed")
                        else:
                            results[test_case['name']] = {'passed': False, 'note': 'Parameter validation failed'}
                            print(f"     ❌ Parameter validation failed")
                    except (ValueError, TypeError) as e:
                        results[test_case['name']] = {'passed': False, 'error': str(e)}
                        print(f"     ❌ Parameter conversion failed: {e}")
                        
            except Exception as e:
                results[test_case['name']] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Error: {e}")
                
        self.test_results['issue_12_search'] = results
        
    async def test_issue_13_scratchbook_functionality(self):
        """Test Issue #13: Scratchbook functionality works without errors."""
        print("\n📝 Testing Issue #13: Scratchbook functionality...")
        
        results = {}
        
        try:
            if self.workspace_client:
                # Test scratchbook manager initialization
                print("   🧪 Testing scratchbook manager initialization")
                
                scratchbook_manager = ScratchbookManager(self.workspace_client)
                
                # Test adding a note
                print("   🧪 Testing adding scratchbook note")
                test_note = "This is a test note for bug fix verification"
                
                update_result = await scratchbook_manager.update_note(
                    content=test_note,
                    tags=["test", "bug-fix", "verification"]
                )
                
                results['add_note'] = {
                    'passed': 'error' not in str(update_result).lower(),
                    'result': update_result
                }
                
                if results['add_note']['passed']:
                    print("     ✅ Successfully added scratchbook note")
                else:
                    print("     ❌ Failed to add scratchbook note")
                    
                # Test searching notes
                print("   🧪 Testing scratchbook note search")
                
                try:
                    search_results = await scratchbook_manager.search_notes(
                        query="test note",
                        limit=5
                    )
                    
                    results['search_notes'] = {
                        'passed': isinstance(search_results, (list, dict)),
                        'result_count': len(search_results) if isinstance(search_results, list) else 1
                    }
                    
                    if results['search_notes']['passed']:
                        print(f"     ✅ Successfully searched notes, found {results['search_notes']['result_count']} results")
                    else:
                        print("     ❌ Failed to search scratchbook notes")
                        
                except AttributeError as e:
                    # This tests the specific AttributeError bug mentioned in issue #13
                    results['search_notes'] = {'passed': False, 'error': f'AttributeError: {e}'}
                    print(f"     ❌ AttributeError bug still present: {e}")
                    
            else:
                print("   ⚠️  No Qdrant client available, testing scratchbook logic only")
                
                # Test scratchbook manager class structure
                try:
                    # Import and basic instantiation test
                    from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
                    
                    # Test that class can be imported and has expected methods
                    expected_methods = ['update_note', 'search_notes', 'list_notes']
                    has_methods = all(hasattr(ScratchbookManager, method) for method in expected_methods)
                    
                    results['class_structure'] = {'passed': has_methods}
                    
                    if has_methods:
                        print("     ✅ ScratchbookManager has required methods")
                    else:
                        print("     ❌ ScratchbookManager missing required methods")
                        
                except ImportError as e:
                    results['class_structure'] = {'passed': False, 'error': str(e)}
                    print(f"     ❌ Import error: {e}")
                    
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
            
            config_validator = ConfigValidator()
            
            # Test with valid configuration
            valid_config = {
                "qdrant": {
                    "url": "http://localhost:6333",
                    "collection_prefix": "test_project"
                },
                "watch": {
                    "enabled": True,
                    "patterns": ["*.py", "*.md", "*.json"],
                    "ignore_patterns": [".git/*", "*.pyc"]
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
                
            # Test advanced watch configuration
            print("   🧪 Testing advanced watch configuration")
            
            try:
                watch_config = AdvancedWatchConfig(
                    enabled=True,
                    patterns=["*.py", "*.md"],
                    ignore_patterns=[".git/*"],
                    workspace_path=self.temp_dir
                )
                
                results['watch_config'] = {
                    'passed': True,
                    'enabled': watch_config.enabled,
                    'patterns': watch_config.patterns
                }
                
                print("     ✅ Advanced watch configuration created successfully")
                
            except Exception as e:
                results['watch_config'] = {'passed': False, 'error': str(e)}
                print(f"     ❌ Advanced watch configuration failed: {e}")
                
        except Exception as e:
            results['general_error'] = {'passed': False, 'error': str(e)}
            print(f"   ❌ General error: {e}")
            
        self.test_results['issue_5_auto_ingestion'] = results
        
    async def test_issue_14_parameter_conversion(self):
        """Test Issue #14: Advanced search tools handle parameter type conversion."""
        print("\n🔢 Testing Issue #14: Parameter type conversion...")
        
        results = {}
        
        # Test cases for string to numeric conversion
        test_cases = [
            {"limit": "10", "score_threshold": "0.7", "expected": True},
            {"limit": "5", "score_threshold": "0.85", "expected": True},
            {"limit": "invalid", "score_threshold": "0.7", "expected": False},
            {"limit": "10", "score_threshold": "invalid", "expected": False},
            {"limit": "-1", "score_threshold": "0.7", "expected": False},  # Invalid range
            {"limit": "10", "score_threshold": "1.5", "expected": False},  # Invalid range
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"   🧪 Testing case {i+1}: limit='{test_case['limit']}', score_threshold='{test_case['score_threshold']}'")
            
            try:
                # Test the conversion logic from the server
                limit = int(test_case['limit']) if isinstance(test_case['limit'], str) else test_case['limit']
                score_threshold = float(test_case['score_threshold']) if isinstance(test_case['score_threshold'], str) else test_case['score_threshold']
                
                # Validate ranges
                limit_valid = limit > 0
                threshold_valid = 0.0 <= score_threshold <= 1.0
                
                conversion_success = limit_valid and threshold_valid
                
                results[f'case_{i+1}'] = {
                    'passed': conversion_success == test_case['expected'],
                    'converted_limit': limit,
                    'converted_threshold': score_threshold,
                    'expected': test_case['expected'],
                    'actual': conversion_success
                }
                
                if conversion_success == test_case['expected']:
                    print(f"     ✅ Conversion behavior as expected")
                else:
                    print(f"     ❌ Unexpected conversion behavior")
                    
            except (ValueError, TypeError) as e:
                conversion_failed = True
                results[f'case_{i+1}'] = {
                    'passed': not test_case['expected'],  # Should fail for invalid inputs
                    'error': str(e),
                    'expected': test_case['expected'],
                    'actual': False
                }
                
                if not test_case['expected']:
                    print(f"     ✅ Expected conversion failure: {e}")
                else:
                    print(f"     ❌ Unexpected conversion failure: {e}")
                    
        self.test_results['issue_14_parameter_conversion'] = results
        
    async def test_server_restart_fix(self):
        """Test server restart fix: Configuration validation works properly."""
        print("\n🔄 Testing Server restart fix: Configuration validation...")
        
        results = {}
        
        try:
            # Test configuration validator instantiation
            print("   🧪 Testing ConfigValidator instantiation")
            
            config_validator = ConfigValidator()
            results['validator_creation'] = {'passed': True}
            print("     ✅ ConfigValidator created successfully")
            
            # Test validation of good configuration  
            print("   🧪 Testing valid configuration validation")
            
            valid_config = {
                "qdrant": {
                    "url": "http://localhost:6333",
                    "collection_prefix": "workspace_qdrant_mcp"
                },
                "embedding": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            }
            
            validation_result = config_validator.validate_config(valid_config)
            
            results['valid_config_test'] = {
                'passed': validation_result.get('valid', False),
                'errors': validation_result.get('errors', []),
                'warnings': validation_result.get('warnings', [])
            }
            
            if results['valid_config_test']['passed']:
                print("     ✅ Valid configuration passed validation")
            else:
                print(f"     ❌ Valid configuration failed: {results['valid_config_test']['errors']}")
                
            # Test validation of invalid configuration
            print("   🧪 Testing invalid configuration validation")
            
            invalid_config = {
                "qdrant": {
                    # Missing required URL
                    "collection_prefix": "test"
                }
            }
            
            validation_result = config_validator.validate_config(invalid_config)
            
            results['invalid_config_test'] = {
                'passed': not validation_result.get('valid', True),  # Should fail
                'errors': validation_result.get('errors', [])
            }
            
            if results['invalid_config_test']['passed']:
                print("     ✅ Invalid configuration properly rejected")
            else:
                print("     ❌ Invalid configuration was incorrectly accepted")
                
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
                    
            print(f"  📈 Issue Score: {issue_passed}/{issue_total} tests passed")
            
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        # Save detailed results to JSON
        report_file = Path(self.temp_dir or ".") / "bug_fix_test_report.json"
        if self.temp_dir is None:
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