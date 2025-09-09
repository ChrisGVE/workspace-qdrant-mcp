#!/usr/bin/env python3
"""
Quick Stress Test Validation - Task #148
Simple validation of stress testing capabilities and basic breaking point detection
"""

import json
import requests
import psutil
import random
import time
from datetime import datetime

def quick_stress_test():
    """Quick validation of stress testing framework"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests_completed': [],
        'system_info': {},
        'breaking_points': {},
        'recommendations': {}
    }
    
    print("🎯 QUICK RESOURCE STRESS TEST VALIDATION")
    print("="*50)
    
    # System info
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    results['system_info'] = {
        'memory_total_gb': memory.total / 1024 / 1024 / 1024,
        'memory_used_percent': memory.percent,
        'memory_available_gb': memory.available / 1024 / 1024 / 1024,
        'cpu_percent': cpu,
        'cpu_count': psutil.cpu_count()
    }
    
    print(f"🖥️  System: {results['system_info']['cpu_count']} cores, {results['system_info']['memory_total_gb']:.1f}GB RAM")
    print(f"📊 Current: Memory {memory.percent:.1f}%, CPU {cpu:.1f}%")
    
    # Test 1: Basic Qdrant connectivity and collection operations
    print(f"\n🔗 TEST 1: Qdrant Connectivity")
    try:
        response = requests.get("http://localhost:6333/cluster", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant accessible")
            results['tests_completed'].append('qdrant_connectivity')
        else:
            print(f"❌ Qdrant not accessible: {response.status_code}")
            return results
    except Exception as e:
        print(f"❌ Qdrant connection error: {e}")
        return results
    
    # Test 2: Collection creation and basic operations
    collection_name = f"quick_test_{int(time.time())}"
    print(f"\n📁 TEST 2: Collection Operations")
    
    try:
        # Create collection
        config = {"vectors": {"size": 384, "distance": "Cosine"}}
        response = requests.put(f"http://localhost:6333/collections/{collection_name}", json=config)
        
        if response.status_code in [200, 201]:
            print(f"✅ Created collection: {collection_name}")
            
            # Add test documents
            docs = []
            for i in range(10):
                docs.append({
                    "id": i,
                    "vector": [random.random() for _ in range(384)],
                    "payload": {"test": True, "id": i}
                })
            
            response = requests.put(
                f"http://localhost:6333/collections/{collection_name}/points",
                json={"points": docs}
            )
            
            if response.status_code in [200, 201]:
                print("✅ Added test documents")
                results['tests_completed'].append('basic_document_operations')
            else:
                print(f"❌ Failed to add documents: {response.status_code}")
                
        else:
            print(f"❌ Failed to create collection: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Collection operations error: {e}")
    
    # Test 3: Memory stress with progressively larger documents
    print(f"\n🧠 TEST 3: Memory Stress (Progressive Document Sizes)")
    
    memory_before = psutil.virtual_memory()
    print(f"   Memory before: {memory_before.percent:.1f}% ({memory_before.used/1024/1024/1024:.1f}GB)")
    
    # Try different document sizes
    size_tests = [
        (20, 100),  # 20 docs × 100KB = 2MB
        (10, 500),  # 10 docs × 500KB = 5MB  
        (5, 1000),  # 5 docs × 1MB = 5MB
    ]
    
    memory_results = []
    
    for doc_count, size_kb in size_tests:
        print(f"   Testing: {doc_count} documents × {size_kb}KB = {doc_count * size_kb / 1024:.1f}MB")
        
        # Create large content
        large_content = 'A' * (size_kb * 1024)
        large_docs = []
        
        for i in range(doc_count):
            large_docs.append({
                "id": 1000 + i,
                "vector": [random.random() for _ in range(384)],
                "payload": {
                    "content": large_content,
                    "size_kb": size_kb,
                    "test_id": i
                }
            })
        
        memory_test_before = psutil.virtual_memory()
        
        try:
            response = requests.put(
                f"http://localhost:6333/collections/{collection_name}/points",
                json={"points": large_docs},
                timeout=30
            )
            
            memory_test_after = psutil.virtual_memory()
            
            test_result = {
                'doc_count': doc_count,
                'size_kb': size_kb,
                'total_mb': doc_count * size_kb / 1024,
                'memory_before': memory_test_before.percent,
                'memory_after': memory_test_after.percent,
                'memory_delta': memory_test_after.percent - memory_test_before.percent,
                'upsert_success': response.status_code in [200, 201],
                'status_code': response.status_code
            }
            
            memory_results.append(test_result)
            
            print(f"      Memory: {memory_test_before.percent:.1f}% → {memory_test_after.percent:.1f}% (Δ{test_result['memory_delta']:+.1f}%)")
            print(f"      Upsert: {'✅' if test_result['upsert_success'] else '❌'} ({response.status_code})")
            
            # Check for potential breaking point
            if memory_test_after.percent > 80:
                print(f"      ⚠️  Memory approaching 80% threshold")
                results['breaking_points']['memory_warning'] = test_result
                
            if not test_result['upsert_success']:
                print(f"      🔶 Upsert failure - potential breaking point")
                results['breaking_points']['memory_failure'] = test_result
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            memory_results.append({
                'doc_count': doc_count,
                'size_kb': size_kb,
                'error': str(e)
            })
        
        time.sleep(2)  # Brief pause
    
    results['memory_stress_results'] = memory_results
    results['tests_completed'].append('memory_stress_basic')
    
    # Test 4: CPU stress with concurrent search queries  
    print(f"\n⚡ TEST 4: CPU Stress (Concurrent Searches)")
    
    cpu_before = psutil.cpu_percent(interval=1)
    print(f"   CPU before: {cpu_before:.1f}%")
    
    import concurrent.futures
    
    def search_query():
        try:
            query = {
                "vector": [random.random() for _ in range(384)],
                "limit": 5
            }
            response = requests.post(
                f"http://localhost:6333/collections/{collection_name}/points/search",
                json=query,
                timeout=10
            )
            return {
                'success': response.status_code == 200,
                'time': response.elapsed.total_seconds()
            }
        except:
            return {'success': False, 'time': 0}
    
    # Progressive CPU test
    cpu_tests = [
        (20, 2),   # 20 queries × 2 threads
        (50, 5),   # 50 queries × 5 threads
        (100, 10), # 100 queries × 10 threads
    ]
    
    cpu_results = []
    
    for query_count, threads in cpu_tests:
        print(f"   Testing: {query_count} queries × {threads} threads")
        
        cpu_test_before = psutil.cpu_percent(interval=1)
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(search_query) for _ in range(query_count)]
            search_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        duration = time.time() - start_time
        cpu_test_after = psutil.cpu_percent(interval=1)
        
        successful = len([r for r in search_results if r['success']])
        success_rate = successful / query_count * 100 if query_count > 0 else 0
        
        test_result = {
            'query_count': query_count,
            'threads': threads,
            'cpu_before': cpu_test_before,
            'cpu_after': cpu_test_after,
            'cpu_delta': cpu_test_after - cpu_test_before,
            'duration': duration,
            'successful_queries': successful,
            'success_rate': success_rate,
            'queries_per_second': query_count / duration if duration > 0 else 0
        }
        
        cpu_results.append(test_result)
        
        print(f"      CPU: {cpu_test_before:.1f}% → {cpu_test_after:.1f}% (Δ{test_result['cpu_delta']:+.1f}%)")
        print(f"      Success: {success_rate:.1f}% ({successful}/{query_count})")
        print(f"      QPS: {test_result['queries_per_second']:.1f}")
        
        # Check for potential breaking point
        if cpu_test_after > 80:
            print(f"      ⚠️  CPU approaching 80% threshold")
            results['breaking_points']['cpu_warning'] = test_result
            
        if success_rate < 90:
            print(f"      ⚠️  Success rate below 90%")
            results['breaking_points']['cpu_performance'] = test_result
        
        time.sleep(1)
    
    results['cpu_stress_results'] = cpu_results  
    results['tests_completed'].append('cpu_stress_basic')
    
    # Cleanup
    try:
        requests.delete(f"http://localhost:6333/collections/{collection_name}")
        print(f"\n🧹 Cleaned up collection: {collection_name}")
    except:
        pass
    
    # Generate summary
    memory_after = psutil.virtual_memory()
    cpu_after = psutil.cpu_percent(interval=1)
    
    print(f"\n📊 FINAL SYSTEM STATE:")
    print(f"   Memory: {memory_after.percent:.1f}% ({memory_after.used/1024/1024/1024:.1f}GB)")
    print(f"   CPU: {cpu_after:.1f}%")
    
    # Generate basic recommendations
    results['recommendations'] = {
        'memory_monitoring_threshold': 75,
        'cpu_monitoring_threshold': 75,
        'tests_successful': len(results['tests_completed']),
        'breaking_points_found': len(results['breaking_points']),
        'system_stable': memory_after.percent < 85 and cpu_after < 85
    }
    
    print(f"\n✅ STRESS TEST VALIDATION COMPLETED")
    print(f"   Tests completed: {len(results['tests_completed'])}")
    print(f"   Breaking points identified: {len(results['breaking_points'])}")
    print(f"   System stable: {'Yes' if results['recommendations']['system_stable'] else 'No'}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"quick_stress_validation_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    results = quick_stress_test()
    if results.get('recommendations', {}).get('system_stable', False):
        print("🎉 System passed stress validation")
    else:
        print("⚠️ System stress concerns identified")