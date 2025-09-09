#!/usr/bin/env python3
"""
Quick Sync Validation - Focused test to demonstrate real-time sync concepts
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from pathlib import Path

def quick_sync_validation():
    """Quick validation of sync concepts"""
    print("🚀 Quick Real-Time Sync Validation - Task #147")
    print("=" * 50)
    
    # Setup
    test_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/LT_20250906-1822_testing_sandbox/quick_sync_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    qdrant_url = "http://localhost:6333"
    
    # Get available collections
    try:
        response = requests.get(f"{qdrant_url}/collections", timeout=5)
        if response.status_code == 200:
            collections = response.json().get('result', {}).get('collections', [])
            test_collections = [c['name'] for c in collections if 'workspace' in c['name'] or 'qmk' in c['name']]
            print(f"✅ Found {len(test_collections)} test collections")
        else:
            print(f"❌ Cannot access collections: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False
    
    results = {
        'test_name': 'Quick Real-Time Sync Validation',
        'task_number': 147,
        'timestamp': datetime.now().isoformat(),
        'scenarios': []
    }
    
    # Scenario 1: File Creation and Search Test
    print(f"\n🔄 Scenario 1: File Creation and Search")
    scenario_start = time.time()
    
    # Create test file with unique marker
    unique_id = f"quick-sync-{int(time.time() * 1000)}"
    test_file = test_dir / "quick_sync_test.c"
    
    content = f"""
// Quick sync test file
// Generated: {datetime.now().isoformat()}
// Unique marker: {unique_id}

#include <stdio.h>

void quick_sync_test_function(void) {{
    printf("Quick sync test: {unique_id}\\n");
}}

// SYNC_MARKER: {unique_id}
"""
    
    print(f"   📝 Creating file with marker: {unique_id}")
    
    file_start = time.time()
    with open(test_file, 'w') as f:
        f.write(content)
    file_creation_time = time.time() - file_start
    
    print(f"   ✅ File created in {file_creation_time*1000:.1f}ms")
    
    # Wait and search for the marker
    search_delays = [1, 5, 10, 15]  # Test at different delays
    search_results = []
    
    for delay in search_delays:
        print(f"   🔍 Searching after {delay}s delay...")
        time.sleep(delay - sum(search_delays[:search_delays.index(delay)]) if search_delays.index(delay) > 0 else delay)
        
        search_found = False
        search_time = 0
        
        for collection in test_collections:
            try:
                search_start = time.time()
                
                search_data = {
                    "filter": {
                        "must": [
                            {
                                "key": "content", 
                                "match": {"text": unique_id}
                            }
                        ]
                    },
                    "limit": 1,
                    "with_payload": True
                }
                
                response = requests.post(
                    f"{qdrant_url}/collections/{collection}/points/scroll",
                    json=search_data,
                    timeout=5
                )
                
                search_time = time.time() - search_start
                
                if response.status_code == 200:
                    points = response.json().get('result', {}).get('points', [])
                    if points:
                        search_found = True
                        print(f"      ✅ Found in collection '{collection}' ({search_time*1000:.1f}ms)")
                        break
                
            except Exception as e:
                continue
        
        if not search_found:
            print(f"      ❌ Not found after {delay}s delay")
        
        search_results.append({
            'delay_seconds': delay,
            'found': search_found,
            'search_time': search_time
        })
    
    scenario_time = time.time() - scenario_start
    
    scenario_1_result = {
        'scenario': 'file_creation_and_search',
        'duration': scenario_time,
        'unique_marker': unique_id,
        'file_creation_time': file_creation_time,
        'search_results': search_results,
        'collections_available': len(test_collections)
    }
    
    results['scenarios'].append(scenario_1_result)
    
    # Scenario 2: Rapid File Modifications
    print(f"\n🔄 Scenario 2: Rapid File Modifications")
    scenario_start = time.time()
    
    modification_results = []
    
    for i in range(3):  # 3 rapid modifications
        mod_unique_id = f"rapid-mod-{i}-{int(time.time() * 1000)}"
        
        mod_content = f"""
// Rapid modification test #{i+1}
// Generated: {datetime.now().isoformat()}
// Modification marker: {mod_unique_id}

#include <stdio.h>

void modification_test_{i}(void) {{
    printf("Modification test {i+1}: {mod_unique_id}\\n");
}}

// RAPID_MOD_MARKER: {mod_unique_id}
"""
        
        print(f"   🔧 Modification {i+1} with marker: {mod_unique_id}")
        
        mod_start = time.time()
        with open(test_file, 'w') as f:
            f.write(mod_content)
        mod_time = time.time() - mod_start
        
        # Quick search test
        time.sleep(2)  # Brief wait
        
        mod_found = False
        for collection in test_collections[:2]:  # Test first 2 collections only
            try:
                search_data = {
                    "filter": {
                        "must": [
                            {
                                "key": "content",
                                "match": {"text": mod_unique_id}
                            }
                        ]
                    },
                    "limit": 1
                }
                
                response = requests.post(
                    f"{qdrant_url}/collections/{collection}/points/scroll",
                    json=search_data,
                    timeout=3
                )
                
                if response.status_code == 200:
                    points = response.json().get('result', {}).get('points', [])
                    if points:
                        mod_found = True
                        break
                        
            except:
                continue
        
        modification_results.append({
            'modification_number': i + 1,
            'marker': mod_unique_id,
            'modification_time': mod_time,
            'found_in_search': mod_found
        })
        
        if mod_found:
            print(f"      ✅ Modification {i+1} found in search")
        else:
            print(f"      ❌ Modification {i+1} not found")
        
        time.sleep(1)  # Brief pause between modifications
    
    scenario_2_time = time.time() - scenario_start
    
    scenario_2_result = {
        'scenario': 'rapid_file_modifications',
        'duration': scenario_2_time,
        'total_modifications': len(modification_results),
        'modifications': modification_results,
        'successful_searches': sum(1 for m in modification_results if m['found_in_search']),
        'success_rate': sum(1 for m in modification_results if m['found_in_search']) / len(modification_results) * 100
    }
    
    results['scenarios'].append(scenario_2_result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"quick_sync_validation_results_{timestamp}.json"
    filepath = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration") / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    if test_file.exists():
        test_file.unlink()
    test_dir.rmdir()
    
    # Analysis
    print(f"\n{'='*60}")
    print("QUICK SYNC VALIDATION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"📊 Test completed in {time.time() - scenario_start:.1f} seconds")
    print(f"📄 Results saved to: {filepath}")
    
    # Scenario 1 analysis
    scenario_1 = results['scenarios'][0]
    successful_searches = [r for r in scenario_1['search_results'] if r['found']]
    
    print(f"\n🎯 SCENARIO 1 - File Creation & Search:")
    print(f"   File creation: {scenario_1['file_creation_time']*1000:.1f}ms")
    print(f"   Successful searches: {len(successful_searches)}/{len(scenario_1['search_results'])}")
    
    if successful_searches:
        fastest_search = min(successful_searches, key=lambda x: x['delay_seconds'])
        print(f"   Fastest detection: {fastest_search['delay_seconds']}s delay")
    else:
        print(f"   ❌ No searches found the marker")
    
    # Scenario 2 analysis  
    scenario_2 = results['scenarios'][1]
    print(f"\n🎯 SCENARIO 2 - Rapid Modifications:")
    print(f"   Total modifications: {scenario_2['total_modifications']}")
    print(f"   Search success rate: {scenario_2['success_rate']:.1f}%")
    print(f"   Scenario duration: {scenario_2['duration']:.1f}s")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    overall_success_rate = (len(successful_searches) / 4 * 50) + (scenario_2['success_rate'] * 0.5)
    
    print(f"   Overall sync performance: {overall_success_rate:.1f}%")
    
    if overall_success_rate >= 80:
        print("   🎉 EXCELLENT - Real-time sync working well")
        print("   ✅ Ready for development workflows")
    elif overall_success_rate >= 60:
        print("   ⚠️  GOOD - Sync working with minor delays")
        print("   ✅ Acceptable for most development use cases")
    elif overall_success_rate >= 40:
        print("   ⚠️  MODERATE - Sync has noticeable delays")
        print("   ❌ May impact development experience")
    else:
        print("   ❌ POOR - Sync not working reliably")
        print("   ❌ Needs significant improvement")
    
    print(f"\n📋 KEY FINDINGS:")
    print(f"   • File operations are fast ({scenario_1['file_creation_time']*1000:.1f}ms)")
    print(f"   • Search system has {len(test_collections)} available collections")
    print(f"   • Rapid modifications success: {scenario_2['success_rate']:.1f}%")
    
    if len(successful_searches) > 0:
        avg_detection_delay = sum(r['delay_seconds'] for r in successful_searches) / len(successful_searches)
        print(f"   • Average detection delay: {avg_detection_delay:.1f}s")
    
    print(f"\n🔬 Task #147 Quick Sync Validation Complete")
    return results

if __name__ == "__main__":
    try:
        results = quick_sync_validation()
        print("\n✅ Quick sync validation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()