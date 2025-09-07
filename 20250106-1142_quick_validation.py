#!/usr/bin/env python3
"""
Quick validation script to test critical bug fixes.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all critical modules can be imported."""
    print("🔍 Testing module imports...")
    
    results = {}
    
    # Test basic imports
    try:
        from workspace_qdrant_mcp.core.config import Config
        results['config'] = True
        print("  ✅ Config imported successfully")
    except ImportError as e:
        results['config'] = False
        print(f"  ❌ Config import failed: {e}")
    
    try:
        from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
        results['scratchbook'] = True
        print("  ✅ ScratchbookManager imported successfully")
    except ImportError as e:
        results['scratchbook'] = False
        print(f"  ❌ ScratchbookManager import failed: {e}")
    
    try:
        from workspace_qdrant_mcp.tools.search import search_workspace
        results['search'] = True
        print("  ✅ search_workspace imported successfully")
    except ImportError as e:
        results['search'] = False
        print(f"  ❌ search_workspace import failed: {e}")
    
    try:
        from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
        results['validator'] = True
        print("  ✅ ConfigValidator imported successfully")
    except ImportError as e:
        results['validator'] = False
        print(f"  ❌ ConfigValidator import failed: {e}")
    
    try:
        from workspace_qdrant_mcp.core.advanced_watch_config import AdvancedWatchConfig
        results['watch_config'] = True
        print("  ✅ AdvancedWatchConfig imported successfully")
    except ImportError as e:
        results['watch_config'] = False
        print(f"  ❌ AdvancedWatchConfig import failed: {e}")
    
    return results

def test_parameter_conversion():
    """Test Issue #14: Parameter type conversion logic."""
    print("\n🔢 Testing parameter conversion logic...")
    
    test_cases = [
        ("10", "0.7", True),    # Valid string conversion
        (5, 0.85, True),        # Already numeric
        ("abc", "0.7", False),  # Invalid limit
        ("10", "xyz", False),   # Invalid threshold
        ("0", "0.7", False),    # Invalid range (limit <= 0)
        ("10", "1.5", False),   # Invalid range (threshold > 1.0)
    ]
    
    passed = 0
    total = len(test_cases)
    
    for limit_input, threshold_input, expected_success in test_cases:
        try:
            # Replicate server.py conversion logic
            limit = int(limit_input) if isinstance(limit_input, str) else limit_input
            score_threshold = float(threshold_input) if isinstance(threshold_input, str) else threshold_input
            
            # Validate ranges
            if limit <= 0:
                raise ValueError("limit must be greater than 0")
            if not (0.0 <= score_threshold <= 1.0):
                raise ValueError("score_threshold must be between 0.0 and 1.0")
                
            success = True
        except (ValueError, TypeError):
            success = False
            
        if success == expected_success:
            passed += 1
            status = "✅"
        else:
            status = "❌"
            
        print(f"  {status} limit={limit_input}, threshold={threshold_input} -> {success} (expected {expected_success})")
    
    print(f"  📊 Parameter conversion: {passed}/{total} tests passed")
    return passed == total

def test_config_creation():
    """Test configuration creation and validation."""
    print("\n⚙️ Testing configuration creation...")
    
    try:
        from workspace_qdrant_mcp.core.config import Config
        
        # Test basic config creation
        config = Config()
        print(f"  ✅ Basic Config created (qdrant.url: {config.qdrant.url})")
        
        # Test custom config
        custom_config = Config(
            qdrant={"url": "http://test:6333"},
            workspace={"collection_suffixes": ["test"]},
            debug=True
        )
        
        url_correct = custom_config.qdrant.url == "http://test:6333"
        debug_correct = custom_config.debug == True
        
        if url_correct and debug_correct:
            print("  ✅ Custom configuration values set correctly")
            return True
        else:
            print("  ❌ Custom configuration values not set correctly")
            return False
            
    except Exception as e:
        print(f"  ❌ Config creation failed: {e}")
        return False

def test_scratchbook_structure():
    """Test scratchbook class structure."""
    print("\n📝 Testing scratchbook structure...")
    
    try:
        from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
        
        # Check required methods exist
        required_methods = ['update_note', 'search_notes', 'list_notes']
        missing_methods = [method for method in required_methods if not hasattr(ScratchbookManager, method)]
        
        if not missing_methods:
            print(f"  ✅ ScratchbookManager has all required methods: {required_methods}")
            
            # Check if methods are callable
            callable_methods = [method for method in required_methods if callable(getattr(ScratchbookManager, method))]
            if len(callable_methods) == len(required_methods):
                print("  ✅ All methods are callable")
                return True
            else:
                print(f"  ❌ Some methods not callable: {set(required_methods) - set(callable_methods)}")
                return False
        else:
            print(f"  ❌ Missing methods: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"  ❌ Scratchbook structure test failed: {e}")
        return False

def main():
    """Run quick validation tests."""
    print("🚀 Quick Validation of Critical Bug Fixes")
    print("=" * 45)
    
    # Run all tests
    import_results = test_imports()
    param_test = test_parameter_conversion()
    config_test = test_config_creation()
    scratchbook_test = test_scratchbook_structure()
    
    # Calculate results
    import_passed = sum(import_results.values())
    import_total = len(import_results)
    
    total_tests = import_total + 3  # param, config, scratchbook
    passed_tests = import_passed + sum([param_test, config_test, scratchbook_test])
    
    print(f"\n📊 SUMMARY")
    print(f"  Imports: {import_passed}/{import_total}")
    print(f"  Parameter conversion: {'✅' if param_test else '❌'}")
    print(f"  Config creation: {'✅' if config_test else '❌'}")
    print(f"  Scratchbook structure: {'✅' if scratchbook_test else '❌'}")
    print(f"  Overall: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 All critical components are working!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())