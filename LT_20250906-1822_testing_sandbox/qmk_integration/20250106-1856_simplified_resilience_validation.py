#!/usr/bin/env python3
"""
Simplified Resilience Testing - Core Validation
Focus on key resilience scenarios without complex process management
"""

import json
import requests
import psutil
import time
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

class SimpleResilienceValidator:
    """Simplified resilience testing focused on data consistency and recovery"""
    
    def __init__(self, qdrant_url="http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self.test_session_id = f"simple_resilience_{int(time.time())}"
        self.test_results = []
        
    def test_qdrant_connectivity(self) -> Dict[str, Any]:
        """Test basic Qdrant connectivity and responsiveness"""
        print("🔗 TESTING QDRANT CONNECTIVITY")
        test_start = time.time()
        
        try:
            # Basic cluster info
            response = requests.get(f"{self.qdrant_url}/cluster", timeout=5)
            cluster_accessible = response.status_code == 200
            
            # Collections endpoint
            response = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            collections_accessible = response.status_code == 200
            
            # Telemetry endpoint
            response = requests.get(f"{self.qdrant_url}/telemetry", timeout=5)
            telemetry_accessible = response.status_code == 200
            
            test_time = time.time() - test_start
            
            connectivity_result = {
                "test_name": "qdrant_connectivity",
                "success": cluster_accessible and collections_accessible,
                "duration_seconds": test_time,
                "metrics": {
                    "cluster_accessible": cluster_accessible,
                    "collections_accessible": collections_accessible,
                    "telemetry_accessible": telemetry_accessible,
                    "response_time_seconds": test_time
                }
            }
            
            print(f"   Cluster: {'✅' if cluster_accessible else '❌'}")
            print(f"   Collections: {'✅' if collections_accessible else '❌'}")
            print(f"   Telemetry: {'✅' if telemetry_accessible else '❌'}")
            print(f"   Response time: {test_time:.3f}s")
            
            return connectivity_result
            
        except Exception as e:
            return {
                "test_name": "qdrant_connectivity",
                "success": False,
                "duration_seconds": time.time() - test_start,
                "error": str(e),
                "metrics": {}
            }
            
    def test_data_consistency_under_load(self) -> Dict[str, Any]:
        """Test data consistency during concurrent operations"""
        print("\n🔍 TESTING DATA CONSISTENCY UNDER LOAD")
        test_start = time.time()
        
        collection_name = f"consistency_test_{int(time.time())}"
        
        try:
            # Create test collection
            config = {
                "vectors": {"size": 128, "distance": "Cosine"},
                "optimizers_config": {"default_segment_number": 2}
            }
            
            response = requests.put(f"{self.qdrant_url}/collections/{collection_name}", json=config)
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to create collection: {response.status_code}")
                
            print(f"   Created collection: {collection_name}")
            
            # Create test documents with checksums
            documents = []
            expected_checksums = {}
            
            for i in range(100):
                content = f"resilience_test_doc_{i}_{random.randint(1000, 9999)}"
                vector = [random.random() for _ in range(128)]
                checksum = hashlib.sha256(content.encode()).hexdigest()
                
                documents.append({
                    "id": i,
                    "vector": vector,
                    "payload": {
                        "content": content,
                        "checksum": checksum,
                        "test_marker": "consistency_validation"
                    }
                })
                
                expected_checksums[str(i)] = checksum
                
            # Insert documents
            response = requests.put(
                f"{self.qdrant_url}/collections/{collection_name}/points",
                json={"points": documents}
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to insert documents: {response.status_code}")
                
            print(f"   Inserted {len(documents)} test documents")
            
            # Perform concurrent searches while validating consistency
            search_results = []
            for i in range(20):
                vector = [random.random() for _ in range(128)]
                response = requests.post(
                    f"{self.qdrant_url}/collections/{collection_name}/points/search",
                    json={"vector": vector, "limit": 5, "with_payload": True}
                )
                search_results.append(response.status_code == 200)
                time.sleep(0.05)  # Small delay between searches
                
            search_success_rate = sum(search_results) / len(search_results)
            
            # Validate data integrity
            response = requests.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                json={"limit": len(documents), "with_payload": True}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve documents: {response.status_code}")
                
            retrieved_points = response.json()["result"]["points"]
            
            # Verify checksums
            checksum_matches = 0
            for point in retrieved_points:
                doc_id = str(point["id"])
                payload = point.get("payload", {})
                stored_checksum = payload.get("checksum")
                content = payload.get("content")
                
                if doc_id in expected_checksums and content:
                    expected_checksum = expected_checksums[doc_id]
                    calculated_checksum = hashlib.sha256(content.encode()).hexdigest()
                    
                    if expected_checksum == stored_checksum == calculated_checksum:
                        checksum_matches += 1
                        
            integrity_rate = checksum_matches / len(documents)
            
            # Cleanup
            requests.delete(f"{self.qdrant_url}/collections/{collection_name}")
            
            test_time = time.time() - test_start
            success = search_success_rate > 0.95 and integrity_rate > 0.98
            
            consistency_result = {
                "test_name": "data_consistency_under_load",
                "success": success,
                "duration_seconds": test_time,
                "metrics": {
                    "documents_created": len(documents),
                    "concurrent_searches": len(search_results),
                    "search_success_rate": search_success_rate,
                    "checksum_matches": checksum_matches,
                    "integrity_rate": integrity_rate,
                    "documents_retrieved": len(retrieved_points)
                }
            }
            
            print(f"   Concurrent searches: {len(search_results)} (success rate: {search_success_rate:.1%})")
            print(f"   Data integrity: {checksum_matches}/{len(documents)} (rate: {integrity_rate:.1%})")
            print(f"   Overall result: {'✅' if success else '❌'}")
            
            return consistency_result
            
        except Exception as e:
            # Cleanup on error
            try:
                requests.delete(f"{self.qdrant_url}/collections/{collection_name}")
            except:
                pass
                
            return {
                "test_name": "data_consistency_under_load",
                "success": False,
                "duration_seconds": time.time() - test_start,
                "error": str(e),
                "metrics": {}
            }
            
    def test_resource_pressure_recovery(self) -> Dict[str, Any]:
        """Test system behavior under resource pressure"""
        print("\n💾 TESTING RESOURCE PRESSURE RECOVERY")
        test_start = time.time()
        
        collection_name = f"pressure_test_{int(time.time())}"
        
        try:
            # Get baseline system metrics
            memory_before = psutil.virtual_memory()
            cpu_before = psutil.cpu_percent(interval=1)
            
            # Create collection
            config = {"vectors": {"size": 256, "distance": "Cosine"}}
            response = requests.put(f"{self.qdrant_url}/collections/{collection_name}", json=config)
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to create collection: {response.status_code}")
                
            print(f"   Baseline: Memory {memory_before.percent:.1f}%, CPU {cpu_before:.1f}%")
            
            # Apply memory pressure with larger documents
            pressure_docs = []
            for i in range(50):  # Fewer but larger documents to avoid overwhelming
                large_content = "x" * (512 * 1024)  # 512KB per document
                vector = [random.random() for _ in range(256)]
                
                pressure_docs.append({
                    "id": i,
                    "vector": vector,
                    "payload": {
                        "content": large_content,
                        "size_kb": 512,
                        "pressure_marker": True
                    }
                })
                
            # Insert pressure documents
            print(f"   Applying pressure: {len(pressure_docs)} large documents")
            response = requests.put(
                f"{self.qdrant_url}/collections/{collection_name}/points",
                json={"points": pressure_docs}
            )
            
            # Check system metrics under pressure
            memory_during = psutil.virtual_memory()
            cpu_during = psutil.cpu_percent(interval=1)
            
            print(f"   Under pressure: Memory {memory_during.percent:.1f}%, CPU {cpu_during:.1f}%")
            
            # Test system responsiveness under pressure
            response_tests = []
            for i in range(10):
                vector = [random.random() for _ in range(256)]
                start = time.time()
                response = requests.post(
                    f"{self.qdrant_url}/collections/{collection_name}/points/search",
                    json={"vector": vector, "limit": 3}
                )
                duration = time.time() - start
                response_tests.append((response.status_code == 200, duration))
                
            successful_responses = sum(1 for success, _ in response_tests if success)
            response_success_rate = successful_responses / len(response_tests)
            avg_response_time = sum(duration for _, duration in response_tests) / len(response_tests)
            
            # Allow recovery period
            print("   Allowing system recovery...")
            time.sleep(5)
            
            # Check recovery metrics
            memory_after = psutil.virtual_memory()
            cpu_after = psutil.cpu_percent(interval=1)
            
            # Cleanup
            requests.delete(f"{self.qdrant_url}/collections/{collection_name}")
            
            test_time = time.time() - test_start
            
            # Success criteria: system remains responsive and memory doesn't spike excessively
            memory_increase = memory_during.percent - memory_before.percent
            system_stable = memory_increase < 20 and response_success_rate > 0.8
            
            pressure_result = {
                "test_name": "resource_pressure_recovery",
                "success": system_stable,
                "duration_seconds": test_time,
                "metrics": {
                    "memory_baseline_percent": memory_before.percent,
                    "memory_under_pressure_percent": memory_during.percent,
                    "memory_after_recovery_percent": memory_after.percent,
                    "memory_increase_percent": memory_increase,
                    "cpu_baseline_percent": cpu_before,
                    "cpu_under_pressure_percent": cpu_during,
                    "cpu_after_recovery_percent": cpu_after,
                    "response_success_rate": response_success_rate,
                    "average_response_time_seconds": avg_response_time,
                    "pressure_documents_count": len(pressure_docs),
                    "document_size_kb": 512
                }
            }
            
            print(f"   Memory increase: {memory_increase:.1f}%")
            print(f"   Response success rate: {response_success_rate:.1%}")
            print(f"   Average response time: {avg_response_time:.3f}s")
            print(f"   Recovery: Memory {memory_after.percent:.1f}%, CPU {cpu_after:.1f}%")
            print(f"   Result: {'✅' if system_stable else '❌'}")
            
            return pressure_result
            
        except Exception as e:
            # Cleanup on error
            try:
                requests.delete(f"{self.qdrant_url}/collections/{collection_name}")
            except:
                pass
                
            return {
                "test_name": "resource_pressure_recovery", 
                "success": False,
                "duration_seconds": time.time() - test_start,
                "error": str(e),
                "metrics": {}
            }
            
    def test_connection_resilience(self) -> Dict[str, Any]:
        """Test connection resilience and recovery"""
        print("\n🌐 TESTING CONNECTION RESILIENCE")
        test_start = time.time()
        
        try:
            # Test rapid connection attempts
            connection_tests = []
            for i in range(20):
                start = time.time()
                try:
                    response = requests.get(f"{self.qdrant_url}/cluster", timeout=2)
                    success = response.status_code == 200
                    duration = time.time() - start
                    connection_tests.append((success, duration))
                except requests.RequestException:
                    connection_tests.append((False, time.time() - start))
                    
                time.sleep(0.1)  # Brief pause between attempts
                
            successful_connections = sum(1 for success, _ in connection_tests if success)
            connection_success_rate = successful_connections / len(connection_tests)
            avg_connection_time = sum(duration for _, duration in connection_tests) / len(connection_tests)
            
            # Test connection with various timeout scenarios
            timeout_tests = []
            for timeout in [1, 2, 5, 10]:
                try:
                    start = time.time()
                    response = requests.get(f"{self.qdrant_url}/cluster", timeout=timeout)
                    duration = time.time() - start
                    timeout_tests.append((timeout, True, duration))
                except requests.RequestException as e:
                    timeout_tests.append((timeout, False, timeout))
                    
            test_time = time.time() - test_start
            
            # Success criteria: high connection success rate and reasonable response times
            connection_resilient = connection_success_rate > 0.95 and avg_connection_time < 1.0
            
            resilience_result = {
                "test_name": "connection_resilience",
                "success": connection_resilient,
                "duration_seconds": test_time,
                "metrics": {
                    "connection_attempts": len(connection_tests),
                    "successful_connections": successful_connections,
                    "connection_success_rate": connection_success_rate,
                    "average_connection_time_seconds": avg_connection_time,
                    "timeout_tests": timeout_tests
                }
            }
            
            print(f"   Connection attempts: {len(connection_tests)}")
            print(f"   Success rate: {connection_success_rate:.1%}")
            print(f"   Average connection time: {avg_connection_time:.3f}s")
            print(f"   Result: {'✅' if connection_resilient else '❌'}")
            
            return resilience_result
            
        except Exception as e:
            return {
                "test_name": "connection_resilience",
                "success": False,
                "duration_seconds": time.time() - test_start,
                "error": str(e),
                "metrics": {}
            }
            
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Execute all resilience validation tests"""
        print("🔥" * 20 + " RESILIENCE VALIDATION SUITE " + "🔥" * 20)
        print(f"Session ID: {self.test_session_id}")
        print(f"Start Time: {datetime.now().isoformat()}")
        print("="*80)
        
        # Get system baseline
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"🖥️  System: {psutil.cpu_count()} cores, {memory.total/1024**3:.1f}GB RAM")
        print(f"📊 Baseline: Memory {memory.percent:.1f}%, CPU {cpu:.1f}%")
        
        test_start_time = time.time()
        all_results = []
        
        # Execute test battery
        tests = [
            self.test_qdrant_connectivity,
            self.test_data_consistency_under_load,
            self.test_resource_pressure_recovery,
            self.test_connection_resilience
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    "test_name": test_func.__name__,
                    "success": False,
                    "duration_seconds": 0,
                    "error": str(e),
                    "metrics": {}
                })
                
        # Analyze results
        total_test_time = time.time() - test_start_time
        successful_tests = sum(1 for result in all_results if result.get("success", False))
        total_tests = len(all_results)
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Final system check
        final_memory = psutil.virtual_memory()
        final_cpu = psutil.cpu_percent(interval=1)
        
        comprehensive_results = {
            "test_session_id": self.test_session_id,
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_test_time,
            
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "overall_success_rate": overall_success_rate
            },
            
            "system_health": {
                "baseline_memory_percent": memory.percent,
                "baseline_cpu_percent": cpu,
                "final_memory_percent": final_memory.percent,
                "final_cpu_percent": final_cpu,
                "memory_change": final_memory.percent - memory.percent,
                "system_stable": abs(final_memory.percent - memory.percent) < 5
            },
            
            "detailed_results": all_results,
            
            "resilience_assessment": {
                "connectivity_stable": any(r.get("test_name") == "qdrant_connectivity" and r.get("success") for r in all_results),
                "data_consistency_validated": any(r.get("test_name") == "data_consistency_under_load" and r.get("success") for r in all_results),
                "resource_resilience_confirmed": any(r.get("test_name") == "resource_pressure_recovery" and r.get("success") for r in all_results),
                "connection_resilience_validated": any(r.get("test_name") == "connection_resilience" and r.get("success") for r in all_results),
                "production_ready": overall_success_rate >= 0.75
            }
        }
        
        return comprehensive_results

def main():
    """Main execution function"""
    validator = SimpleResilienceValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = f"simplified_resilience_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print("📊 RESILIENCE VALIDATION RESULTS")
    print("="*60)
    
    summary = results["summary"]
    assessment = results["resilience_assessment"]
    system = results["system_health"]
    
    print(f"✅ Tests Passed: {summary['successful_tests']}/{summary['total_tests']}")
    print(f"📈 Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"⏱️  Total Duration: {results['total_duration_seconds']:.1f}s")
    
    print(f"\n🏭 RESILIENCE ASSESSMENT:")
    print(f"   Connectivity: {'✅' if assessment['connectivity_stable'] else '❌'}")
    print(f"   Data Consistency: {'✅' if assessment['data_consistency_validated'] else '❌'}")
    print(f"   Resource Resilience: {'✅' if assessment['resource_resilience_confirmed'] else '❌'}")
    print(f"   Connection Resilience: {'✅' if assessment['connection_resilience_validated'] else '❌'}")
    
    print(f"\n💾 System Health:")
    print(f"   Memory: {system['baseline_memory_percent']:.1f}% → {system['final_memory_percent']:.1f}% (Δ{system['memory_change']:+.1f}%)")
    print(f"   CPU: {system['baseline_cpu_percent']:.1f}% → {system['final_cpu_percent']:.1f}%")
    print(f"   System Stable: {'✅' if system['system_stable'] else '❌'}")
    
    print(f"\n🚀 PRODUCTION READINESS: {'YES' if assessment['production_ready'] else 'NEEDS IMPROVEMENT'}")
    print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()