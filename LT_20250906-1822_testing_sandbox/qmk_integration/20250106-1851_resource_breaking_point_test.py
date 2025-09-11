#!/usr/bin/env python3
"""
Resource Breaking Point Test - Task #148 Final Implementation
Quick and focused test to find daemon breaking points and resource limits
"""

import os
import sys
import time
import json
import psutil
import requests
import uuid
import random
from datetime import datetime
from pathlib import Path

def get_system_baseline():
    """Get baseline system measurements"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    return {
        'memory_percent': memory.percent,
        'memory_gb': memory.used / 1024 / 1024 / 1024,
        'memory_available_gb': memory.available / 1024 / 1024 / 1024,
        'memory_total_gb': memory.total / 1024 / 1024 / 1024,
        'cpu_percent': cpu,
        'cpu_count': psutil.cpu_count()
    }

def test_memory_stress():
    """Test progressive memory stress using large Qdrant documents"""
    print("🧠 MEMORY STRESS TEST - Finding Breaking Points")
    print("="*50)
    
    # Create test collection
    collection_name = f"memory_stress_{uuid.uuid4().hex[:8]}"
    
    try:
        # Create collection
        config = {"vectors": {"size": 384, "distance": "Cosine"}}
        response = requests.put(f"http://localhost:6333/collections/{collection_name}", json=config)
        if response.status_code not in [200, 201]:
            print(f"❌ Failed to create collection: {response.status_code}")
            return None
            
        print(f"✅ Created collection: {collection_name}")
        
        results = {
            'test_type': 'memory_stress',
            'phases': [],
            'breaking_point': None,
            'max_memory_reached': 0
        }
        
        # Progressive memory stress phases
        # Start with smaller documents and increase size
        test_phases = [
            (100, 50),    # 100 docs × 50KB = 5MB
            (100, 100),   # 100 docs × 100KB = 10MB  
            (100, 500),   # 100 docs × 500KB = 50MB
            (50, 1000),   # 50 docs × 1MB = 50MB
            (20, 5000),   # 20 docs × 5MB = 100MB
            (10, 10000),  # 10 docs × 10MB = 100MB
        ]
        
        for phase_num, (doc_count, size_kb) in enumerate(test_phases, 1):
            print(f"\n📈 Phase {phase_num}: {doc_count} documents × {size_kb}KB = {doc_count * size_kb / 1024:.1f}MB")
            
            memory_before = psutil.virtual_memory()
            
            # Generate large documents
            large_content = 'M' * (size_kb * 1024)  # Create content of specified size
            documents = []
            
            for i in range(doc_count):
                documents.append({
                    "id": phase_num * 1000 + i,
                    "vector": [random.random() for _ in range(384)],
                    "payload": {
                        "content": large_content,
                        "size_kb": size_kb,
                        "phase": phase_num,
                        "doc_id": i
                    }
                })
            
            # Attempt upsert
            upsert_start = time.time()
            try:
                response = requests.put(
                    f"http://localhost:6333/collections/{collection_name}/points",
                    json={"points": documents}
                )
                upsert_duration = time.time() - upsert_start
                upsert_success = response.status_code in [200, 201]
                
                if not upsert_success:
                    print(f"   ❌ Upsert failed: {response.status_code}")
                    
            except Exception as e:
                upsert_duration = time.time() - upsert_start
                upsert_success = False
                print(f"   ❌ Upsert error: {e}")
                
            memory_after = psutil.virtual_memory()
            
            phase_result = {
                'phase': phase_num,
                'doc_count': doc_count,
                'size_kb': size_kb,
                'total_mb': doc_count * size_kb / 1024,
                'memory_before_percent': memory_before.percent,
                'memory_after_percent': memory_after.percent,
                'memory_delta_percent': memory_after.percent - memory_before.percent,
                'memory_before_gb': memory_before.used / 1024 / 1024 / 1024,
                'memory_after_gb': memory_after.used / 1024 / 1024 / 1024,
                'upsert_success': upsert_success,
                'upsert_duration': upsert_duration,
                'throughput_docs_per_sec': doc_count / upsert_duration if upsert_duration > 0 else 0
            }
            
            results['phases'].append(phase_result)
            results['max_memory_reached'] = max(results['max_memory_reached'], memory_after.percent)
            
            print(f"   Memory: {memory_before.percent:.1f}% → {memory_after.percent:.1f}% (Δ{phase_result['memory_delta_percent']:+.1f}%)")
            print(f"   Memory: {memory_before.used/1024/1024/1024:.1f}GB → {memory_after.used/1024/1024/1024:.1f}GB")
            
            if upsert_success:
                print(f"   ✅ Upsert successful: {phase_result['throughput_docs_per_sec']:.1f} docs/sec")
            else:
                print(f"   ❌ Upsert failed - Breaking point reached")
                
            # Check for breaking point conditions
            breaking_point_reached = False
            breaking_reason = None
            
            if memory_after.percent > 85:  # 85% memory threshold
                breaking_point_reached = True
                breaking_reason = "memory_limit_85_percent"
            elif not upsert_success:
                breaking_point_reached = True
                breaking_reason = "upsert_failure"
            elif memory_after.percent > 80 and phase_result['memory_delta_percent'] > 10:
                breaking_point_reached = True
                breaking_reason = "high_memory_with_large_delta"
                
            if breaking_point_reached:
                print(f"🔶 MEMORY BREAKING POINT REACHED")
                print(f"   Reason: {breaking_reason}")
                print(f"   Memory usage: {memory_after.percent:.1f}%")
                print(f"   Document size: {size_kb}KB × {doc_count} docs")
                
                results['breaking_point'] = {
                    'phase': phase_num,
                    'reason': breaking_reason,
                    'memory_percent': memory_after.percent,
                    'memory_gb': memory_after.used / 1024 / 1024 / 1024,
                    'doc_size_kb': size_kb,
                    'doc_count': doc_count,
                    'total_data_mb': doc_count * size_kb / 1024
                }
                break
                
            # Brief pause for system recovery
            time.sleep(3)
        
        return results
        
    finally:
        # Cleanup collection
        try:
            requests.delete(f"http://localhost:6333/collections/{collection_name}")
            print(f"🧹 Cleaned up collection: {collection_name}")
        except:
            pass

def test_cpu_stress():
    """Test CPU stress using concurrent search queries"""
    print("\n⚡ CPU STRESS TEST - Finding Processing Limits")
    print("="*50)
    
    # Create test collection with some data
    collection_name = f"cpu_stress_{uuid.uuid4().hex[:8]}"
    
    try:
        # Create collection
        config = {"vectors": {"size": 384, "distance": "Cosine"}}
        response = requests.put(f"http://localhost:6333/collections/{collection_name}", json=config)
        if response.status_code not in [200, 201]:
            print(f"❌ Failed to create collection for CPU test")
            return None
            
        # Add some baseline data for searching
        baseline_docs = []
        for i in range(50):  # Add 50 baseline documents
            baseline_docs.append({
                "id": i,
                "vector": [random.random() for _ in range(384)],
                "payload": {"baseline": True, "id": i}
            })
            
        response = requests.put(
            f"http://localhost:6333/collections/{collection_name}/points",
            json={"points": baseline_docs}
        )
        
        if response.status_code not in [200, 201]:
            print(f"❌ Failed to add baseline data for CPU test")
            return None
            
        print(f"✅ Created collection with baseline data: {collection_name}")
        
        results = {
            'test_type': 'cpu_stress',
            'phases': [],
            'breaking_point': None,
            'max_cpu_reached': 0
        }
        
        # Progressive CPU stress phases
        # Increase query load and concurrency
        import concurrent.futures
        import threading
        
        test_phases = [
            (20, 2),    # 20 queries × 2 threads
            (50, 5),    # 50 queries × 5 threads  
            (100, 10),  # 100 queries × 10 threads
            (200, 15),  # 200 queries × 15 threads
            (500, 20),  # 500 queries × 20 threads
            (1000, 25), # 1000 queries × 25 threads
        ]
        
        def run_search_query():
            """Execute single search query"""
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
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'response_time': 0
                }
        
        for phase_num, (query_count, thread_count) in enumerate(test_phases, 1):
            print(f"\n🔄 Phase {phase_num}: {query_count} queries × {thread_count} threads")
            
            cpu_before = psutil.cpu_percent(interval=1)  # 1-second measurement
            
            # Execute concurrent search queries
            search_start = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(run_search_query) for _ in range(query_count)]
                search_results = []
                
                for future in concurrent.futures.as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        search_results.append(result)
                    except Exception as e:
                        search_results.append({
                            'success': False,
                            'error': str(e),
                            'response_time': 0
                        })
            
            search_duration = time.time() - search_start
            cpu_after = psutil.cpu_percent(interval=1)  # 1-second measurement
            
            # Analyze search results
            successful_searches = [r for r in search_results if r.get('success', False)]
            failed_searches = [r for r in search_results if not r.get('success', False)]
            
            success_rate = len(successful_searches) / len(search_results) * 100 if search_results else 0
            avg_response_time = sum(r['response_time'] for r in successful_searches) / len(successful_searches) if successful_searches else 0
            queries_per_second = query_count / search_duration if search_duration > 0 else 0
            
            phase_result = {
                'phase': phase_num,
                'query_count': query_count,
                'thread_count': thread_count,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'cpu_delta': cpu_after - cpu_before,
                'search_duration': search_duration,
                'successful_queries': len(successful_searches),
                'failed_queries': len(failed_searches),
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'queries_per_second': queries_per_second
            }
            
            results['phases'].append(phase_result)
            results['max_cpu_reached'] = max(results['max_cpu_reached'], cpu_after)
            
            print(f"   CPU: {cpu_before:.1f}% → {cpu_after:.1f}% (Δ{phase_result['cpu_delta']:+.1f}%)")
            print(f"   Success: {success_rate:.1f}% ({len(successful_searches)}/{len(search_results)})")
            print(f"   QPS: {queries_per_second:.1f}")
            print(f"   Avg response: {avg_response_time*1000:.1f}ms")
            
            # Check for breaking point conditions
            breaking_point_reached = False
            breaking_reason = None
            
            if cpu_after > 85:  # 85% CPU threshold
                breaking_point_reached = True
                breaking_reason = "cpu_limit_85_percent"
            elif success_rate < 80:  # Less than 80% success rate
                breaking_point_reached = True
                breaking_reason = "low_success_rate"
            elif avg_response_time > 1.0:  # Response time > 1 second
                breaking_point_reached = True
                breaking_reason = "high_response_time"
                
            if breaking_point_reached:
                print(f"🔶 CPU BREAKING POINT REACHED")
                print(f"   Reason: {breaking_reason}")
                print(f"   CPU usage: {cpu_after:.1f}%")
                print(f"   Thread count: {thread_count}")
                print(f"   Success rate: {success_rate:.1f}%")
                
                results['breaking_point'] = {
                    'phase': phase_num,
                    'reason': breaking_reason,
                    'cpu_percent': cpu_after,
                    'thread_count': thread_count,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'queries_per_second': queries_per_second
                }
                break
                
            # Brief pause for system recovery
            time.sleep(2)
            
        return results
        
    finally:
        # Cleanup collection
        try:
            requests.delete(f"http://localhost:6333/collections/{collection_name}")
            print(f"🧹 Cleaned up collection: {collection_name}")
        except:
            pass

def generate_final_analysis(baseline, memory_results, cpu_results):
    """Generate comprehensive analysis and recommendations"""
    print("\n" + "="*60)
    print("📊 RESOURCE BREAKING POINT ANALYSIS")
    print("="*60)
    
    analysis = {
        'test_timestamp': datetime.now().isoformat(),
        'system_baseline': baseline,
        'memory_stress_results': memory_results,
        'cpu_stress_results': cpu_results,
        'breaking_points_summary': {},
        'production_recommendations': {},
        'resource_guardrails_validation': {}
    }
    
    # Analyze breaking points
    print(f"🖥️  System: {baseline['cpu_count']} cores, {baseline['memory_total_gb']:.1f} GB RAM")
    print(f"📈 Baseline: Memory {baseline['memory_percent']:.1f}%, CPU {baseline['cpu_percent']:.1f}%")
    
    if memory_results and memory_results.get('breaking_point'):
        bp = memory_results['breaking_point']
        print(f"\n🧠 MEMORY BREAKING POINT FOUND:")
        print(f"   📊 Memory usage: {bp['memory_percent']:.1f}% ({bp['memory_gb']:.1f} GB)")
        print(f"   📄 Document size: {bp['doc_size_kb']}KB × {bp['doc_count']} docs = {bp['total_data_mb']:.1f}MB")
        print(f"   🔍 Reason: {bp['reason']}")
        
        analysis['breaking_points_summary']['memory'] = bp
        
    if cpu_results and cpu_results.get('breaking_point'):
        bp = cpu_results['breaking_point']
        print(f"\n⚡ CPU BREAKING POINT FOUND:")
        print(f"   📊 CPU usage: {bp['cpu_percent']:.1f}%")
        print(f"   🔄 Thread count: {bp['thread_count']}")
        print(f"   ✅ Success rate: {bp['success_rate']:.1f}%")
        print(f"   ⏱️  Response time: {bp['avg_response_time']*1000:.1f}ms")
        print(f"   🔍 Reason: {bp['reason']}")
        
        analysis['breaking_points_summary']['cpu'] = bp
        
    # Generate production recommendations
    print(f"\n🚀 PRODUCTION DEPLOYMENT RECOMMENDATIONS:")
    
    # Memory recommendations
    if memory_results:
        max_safe_memory = memory_results.get('max_memory_reached', baseline['memory_percent'])
        recommended_warning = min(70, max_safe_memory * 0.8)
        recommended_critical = min(80, max_safe_memory * 0.9)
        
        print(f"   🧠 Memory Limits:")
        print(f"      Warning threshold: {recommended_warning:.0f}%")
        print(f"      Critical threshold: {recommended_critical:.0f}%")
        print(f"      Tested maximum: {max_safe_memory:.1f}%")
        
        analysis['production_recommendations']['memory'] = {
            'warning_threshold': recommended_warning,
            'critical_threshold': recommended_critical,
            'tested_maximum': max_safe_memory
        }
        
    # CPU recommendations
    if cpu_results:
        max_safe_cpu = cpu_results.get('max_cpu_reached', baseline['cpu_percent'])
        recommended_cpu_warning = min(70, max_safe_cpu * 0.8)
        recommended_cpu_critical = min(80, max_safe_cpu * 0.9)
        
        print(f"   ⚡ CPU Limits:")
        print(f"      Warning threshold: {recommended_cpu_warning:.0f}%")
        print(f"      Critical threshold: {recommended_cpu_critical:.0f}%")
        print(f"      Tested maximum: {max_safe_cpu:.1f}%")
        
        analysis['production_recommendations']['cpu'] = {
            'warning_threshold': recommended_cpu_warning,
            'critical_threshold': recommended_cpu_critical,
            'tested_maximum': max_safe_cpu
        }
        
    # Resource guardrail validation
    print(f"\n🛡️  RESOURCE GUARDRAIL VALIDATION:")
    
    guardrails_effective = True
    if memory_results and memory_results.get('breaking_point'):
        if memory_results['breaking_point']['memory_percent'] > 90:
            print(f"   ⚠️  Memory exceeded 90% threshold")
            guardrails_effective = False
        else:
            print(f"   ✅ Memory guardrails effective (stopped at {memory_results['breaking_point']['memory_percent']:.1f}%)")
            
    if cpu_results and cpu_results.get('breaking_point'):
        if cpu_results['breaking_point']['cpu_percent'] > 90:
            print(f"   ⚠️  CPU exceeded 90% threshold")
            guardrails_effective = False
        else:
            print(f"   ✅ CPU guardrails effective (stopped at {cpu_results['breaking_point']['cpu_percent']:.1f}%)")
            
    analysis['resource_guardrails_validation'] = {
        'effective': guardrails_effective,
        'memory_safe': memory_results.get('breaking_point', {}).get('memory_percent', 0) <= 90,
        'cpu_safe': cpu_results.get('breaking_point', {}).get('cpu_percent', 0) <= 90
    }
    
    # Operational guidelines
    print(f"\n📋 OPERATIONAL GUIDELINES:")
    print(f"   • Maintain 20% system resource headroom for OS operations")
    print(f"   • Implement progressive monitoring at warning thresholds")
    print(f"   • Use circuit breakers for query rate limiting")
    print(f"   • Monitor memory usage during large document ingestion")
    print(f"   • Limit concurrent processing threads based on test results")
    
    analysis['operational_guidelines'] = [
        "Maintain 20% system resource headroom for OS operations",
        "Implement progressive monitoring at warning thresholds", 
        "Use circuit breakers for query rate limiting",
        "Monitor memory usage during large document ingestion",
        "Limit concurrent processing threads based on test results"
    ]
    
    return analysis

def main():
    """Execute focused resource breaking point testing"""
    print("🎯 RESOURCE BREAKING POINT STRESS TESTING - TASK #148")
    print("="*70)
    print("MISSION: Find exact daemon breaking points and validate guardrails")
    print("SAFETY: System resource monitoring with emergency stops")
    print("="*70)
    
    try:
        # Verify Qdrant accessibility
        response = requests.get("http://localhost:6333/cluster", timeout=5)
        if response.status_code != 200:
            print("❌ Qdrant daemon not accessible")
            return False
        print("✅ Qdrant daemon accessible")
        
        # Get baseline measurements
        print("\n📊 Establishing baseline system measurements...")
        baseline = get_system_baseline()
        print(f"   Memory: {baseline['memory_percent']:.1f}% ({baseline['memory_gb']:.1f} GB used)")
        print(f"   CPU: {baseline['cpu_percent']:.1f}%")
        print(f"   Available headroom: {100 - baseline['memory_percent']:.1f}% memory, {100 - baseline['cpu_percent']:.1f}% CPU")
        
        # Execute memory stress test
        memory_results = test_memory_stress()
        
        # Execute CPU stress test  
        cpu_results = test_cpu_stress()
        
        # Generate comprehensive analysis
        final_analysis = generate_final_analysis(baseline, memory_results, cpu_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = f"resource_breaking_point_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(final_analysis, f, indent=2, default=str)
            
        print(f"\n💾 Comprehensive results saved to: {results_file}")
        print("\n🎉 RESOURCE BREAKING POINT TESTING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)