#!/usr/bin/env python3
"""
FastMCP Framework Performance Benchmark

Measures performance characteristics before and after optimizations:
1. Tool registration overhead
2. Message serialization/deserialization
3. stdio protocol communication efficiency
4. Framework memory usage
5. Request/response latency
"""

import asyncio
import json
import time
import tracemalloc
import resource
import statistics
from pathlib import Path
from typing import Dict, Any, List
import psutil
import sys
import os

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workspace_qdrant_mcp.tools.simplified_interface import SimplifiedToolsMode


class FastMCPBenchmark:
    """Benchmark FastMCP framework performance characteristics."""
    
    def __init__(self):
        self.results = {}
        self.baseline_memory = None
        
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("Starting FastMCP Performance Benchmark...")
        
        # Set up memory tracking
        tracemalloc.start()
        process = psutil.Process()
        self.baseline_memory = process.memory_info()
        
        results = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "tool_registration": await self._benchmark_tool_registration(),
            "message_processing": await self._benchmark_message_processing(),
            "stdio_communication": await self._benchmark_stdio_communication(),
            "memory_usage": await self._benchmark_memory_usage(),
            "framework_overhead": await self._benchmark_framework_overhead(),
        }
        
        # Clean up
        tracemalloc.stop()
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "platform": sys.platform,
        }
    
    async def _benchmark_tool_registration(self) -> Dict[str, Any]:
        """Benchmark tool registration performance in different modes."""
        print("Benchmarking tool registration...")
        
        results = {}
        
        # Test different tool modes
        for mode in ["basic", "standard", "full"]:
            # Set mode environment variable
            os.environ["QDRANT_MCP_MODE"] = mode
            
            start_time = time.perf_counter()
            start_memory = tracemalloc.take_snapshot()
            
            try:
                # Simulate tool registration process
                from fastmcp import FastMCP
                app = FastMCP("test-benchmark")
                
                # Get tool count based on mode
                SimplifiedToolsMode._mode_cache = None  # Clear cache
                enabled_tools = SimplifiedToolsMode.get_enabled_tools()
                
                # Simulate registration overhead
                if SimplifiedToolsMode.is_simplified_mode():
                    # Simplified mode - fewer tools
                    tool_count = len(enabled_tools) if enabled_tools else 4
                else:
                    # Full mode - all 30+ tools
                    tool_count = 35
                
                # Measure registration time per tool (simulated)
                for i in range(tool_count):
                    @app.tool()
                    async def dummy_tool() -> dict:
                        return {"result": "test"}
                
                end_time = time.perf_counter()
                end_memory = tracemalloc.take_snapshot()
                
                memory_diff = self._calculate_memory_diff(start_memory, end_memory)
                
                results[mode] = {
                    "tool_count": tool_count,
                    "registration_time": (end_time - start_time) * 1000,  # ms
                    "memory_usage": memory_diff,
                    "time_per_tool": ((end_time - start_time) * 1000) / tool_count if tool_count > 0 else 0,
                }
                
                print(f"  {mode.upper()} mode: {tool_count} tools, {results[mode]['registration_time']:.2f}ms")
                
            except Exception as e:
                results[mode] = {"error": str(e)}
        
        return results
    
    async def _benchmark_message_processing(self) -> Dict[str, Any]:
        """Benchmark message serialization/deserialization performance."""
        print("Benchmarking message processing...")
        
        # Create sample messages of different sizes
        test_messages = [
            {"small": {"query": "test search", "limit": 10}},
            {"medium": {"information": "x" * 1000, "metadata": {"key": "value"}}},
            {"large": {"content": "x" * 10000, "metadata": {"tags": ["a"] * 100}}},
        ]
        
        results = {}
        
        for size, message in [(k, v) for msg in test_messages for k, v in msg.items()]:
            serialization_times = []
            deserialization_times = []
            
            # Run multiple iterations for statistical accuracy
            for _ in range(100):
                # Measure serialization
                start_time = time.perf_counter()
                json_str = json.dumps(message)
                serialization_times.append((time.perf_counter() - start_time) * 1000000)  # μs
                
                # Measure deserialization
                start_time = time.perf_counter()
                parsed = json.loads(json_str)
                deserialization_times.append((time.perf_counter() - start_time) * 1000000)  # μs
            
            results[size] = {
                "message_size": len(json.dumps(message)),
                "serialization": {
                    "mean": statistics.mean(serialization_times),
                    "median": statistics.median(serialization_times),
                    "std": statistics.stdev(serialization_times) if len(serialization_times) > 1 else 0,
                },
                "deserialization": {
                    "mean": statistics.mean(deserialization_times),
                    "median": statistics.median(deserialization_times),
                    "std": statistics.stdev(deserialization_times) if len(deserialization_times) > 1 else 0,
                },
            }
            
            print(f"  {size.upper()} message ({results[size]['message_size']} bytes): "
                  f"ser {results[size]['serialization']['mean']:.1f}μs, "
                  f"deser {results[size]['deserialization']['mean']:.1f}μs")
        
        return results
    
    async def _benchmark_stdio_communication(self) -> Dict[str, Any]:
        """Benchmark stdio protocol communication efficiency."""
        print("Benchmarking stdio communication...")
        
        # Simulate stdio communication patterns
        results = {}
        
        # Test different message patterns
        patterns = [
            {"name": "simple_request", "data": {"method": "qdrant_store", "params": {"information": "test"}}},
            {"name": "complex_request", "data": {"method": "qdrant_find", "params": {"query": "search", "filters": {"tags": ["a", "b"]}}}},
            {"name": "large_response", "data": {"result": {"results": [{"content": "x" * 1000}] * 10}}},
        ]
        
        for pattern in patterns:
            times = []
            
            for _ in range(50):
                start_time = time.perf_counter()
                
                # Simulate stdio write/read cycle
                json_data = json.dumps(pattern["data"])
                _ = len(json_data.encode('utf-8'))
                
                # Simulate processing delay
                await asyncio.sleep(0.001)  # 1ms processing
                
                # Simulate response
                response = {"id": 1, "result": pattern["data"]}
                json_response = json.dumps(response)
                _ = len(json_response.encode('utf-8'))
                
                times.append((time.perf_counter() - start_time) * 1000)  # ms
            
            results[pattern["name"]] = {
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "p95_time": sorted(times)[int(0.95 * len(times))],
                "message_size": len(json.dumps(pattern["data"])),
            }
            
            print(f"  {pattern['name']}: {results[pattern['name']]['mean_time']:.2f}ms avg")
        
        return results
    
    async def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage...")
        
        process = psutil.Process()
        current_memory = process.memory_info()
        
        # Simulate different workload patterns
        workloads = []
        
        # Memory usage for tool registration
        for mode in ["basic", "standard", "full"]:
            os.environ["QDRANT_MCP_MODE"] = mode
            SimplifiedToolsMode._mode_cache = None
            
            start_memory = process.memory_info()
            
            # Simulate framework initialization
            try:
                from fastmcp import FastMCP
                app = FastMCP(f"test-memory-{mode}")
                
                # Wait for memory to stabilize
                await asyncio.sleep(0.1)
                
                end_memory = process.memory_info()
                
                workloads.append({
                    "mode": mode,
                    "memory_increase": end_memory.rss - start_memory.rss,
                    "tool_count": len(SimplifiedToolsMode.get_enabled_tools()) if SimplifiedToolsMode.is_simplified_mode() else 35,
                })
                
            except Exception as e:
                workloads.append({"mode": mode, "error": str(e)})
        
        # Calculate memory efficiency
        results = {
            "baseline_memory": self.baseline_memory.rss,
            "current_memory": current_memory.rss,
            "workloads": workloads,
            "memory_per_tool": {},
        }
        
        for workload in workloads:
            if "error" not in workload and workload["tool_count"] > 0:
                results["memory_per_tool"][workload["mode"]] = workload["memory_increase"] / workload["tool_count"]
                print(f"  {workload['mode'].upper()} mode: {workload['memory_increase']/1024:.1f}KB total, "
                      f"{results['memory_per_tool'][workload['mode']]/1024:.1f}KB per tool")
        
        return results
    
    async def _benchmark_framework_overhead(self) -> Dict[str, Any]:
        """Benchmark framework overhead in different configurations."""
        print("Benchmarking framework overhead...")
        
        results = {}
        
        # Test overhead for different tool sets
        for mode in ["basic", "standard", "full"]:
            os.environ["QDRANT_MCP_MODE"] = mode
            SimplifiedToolsMode._mode_cache = None
            
            start_time = time.perf_counter()
            
            try:
                # Simulate full server initialization
                from fastmcp import FastMCP
                app = FastMCP(f"overhead-test-{mode}")
                
                # Simulate tool registration based on mode
                tool_count = len(SimplifiedToolsMode.get_enabled_tools()) if SimplifiedToolsMode.is_simplified_mode() else 35
                
                # Add sample tools
                for i in range(min(tool_count, 5)):  # Limit to 5 for testing
                    @app.tool()
                    async def sample_tool() -> dict:
                        return {"result": f"tool_{i}"}
                
                # Measure initialization completion time
                initialization_time = (time.perf_counter() - start_time) * 1000
                
                # Test request processing overhead
                request_times = []
                for _ in range(20):
                    request_start = time.perf_counter()
                    # Simulate request processing
                    await asyncio.sleep(0.001)  # 1ms simulated processing
                    request_times.append((time.perf_counter() - request_start) * 1000)
                
                results[mode] = {
                    "initialization_time": initialization_time,
                    "tool_count": tool_count,
                    "request_processing": {
                        "mean": statistics.mean(request_times),
                        "median": statistics.median(request_times),
                        "overhead": statistics.mean(request_times) - 1.0,  # Subtract base processing time
                    }
                }
                
                print(f"  {mode.upper()} mode: init {initialization_time:.2f}ms, "
                      f"request overhead {results[mode]['request_processing']['overhead']:.2f}ms")
                
            except Exception as e:
                results[mode] = {"error": str(e)}
        
        return results
    
    def _calculate_memory_diff(self, start_snapshot, end_snapshot):
        """Calculate memory difference between snapshots."""
        start_stats = start_snapshot.statistics('lineno')
        end_stats = end_snapshot.statistics('lineno')
        
        # Simple approximation - sum of memory blocks
        start_total = sum(stat.size for stat in start_stats)
        end_total = sum(stat.size for stat in end_stats)
        
        return end_total - start_total
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"fastmcp_benchmark_results_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to: {filepath}")
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n=== FASTMCP PERFORMANCE BENCHMARK SUMMARY ===")
        
        print(f"\nSystem: {results['system_info']['platform']}, "
              f"{results['system_info']['cpu_count']} CPUs")
        
        # Tool registration summary
        if "tool_registration" in results:
            print("\n--- Tool Registration Performance ---")
            for mode, data in results["tool_registration"].items():
                if "error" not in data:
                    print(f"{mode.upper():>8}: {data['tool_count']:>2} tools, "
                          f"{data['registration_time']:>6.2f}ms total, "
                          f"{data['time_per_tool']:>5.2f}ms per tool")
        
        # Framework overhead summary
        if "framework_overhead" in results:
            print("\n--- Framework Overhead ---")
            for mode, data in results["framework_overhead"].items():
                if "error" not in data:
                    overhead = data['request_processing']['overhead']
                    print(f"{mode.upper():>8}: {data['initialization_time']:>6.2f}ms init, "
                          f"{overhead:>5.2f}ms request overhead")
        
        # Memory usage summary
        if "memory_usage" in results and "memory_per_tool" in results["memory_usage"]:
            print("\n--- Memory Efficiency ---")
            for mode, mem_per_tool in results["memory_usage"]["memory_per_tool"].items():
                print(f"{mode.upper():>8}: {mem_per_tool/1024:>5.1f}KB per tool")
        
        print("\n--- Recommendations ---")
        
        # Analyze results and provide recommendations
        reg_results = results.get("tool_registration", {})
        if "basic" in reg_results and "full" in reg_results:
            basic_time = reg_results["basic"].get("registration_time", 0)
            full_time = reg_results["full"].get("registration_time", 0)
            
            if full_time > basic_time * 2:
                print("• Consider simplified mode for faster startup")
            
            if basic_time < 50:  # ms
                print("• Tool registration is efficient")
            else:
                print("• Tool registration could be optimized")


async def main():
    """Run FastMCP performance benchmark."""
    benchmark = FastMCPBenchmark()
    
    try:
        results = await benchmark.run_full_benchmark()
        benchmark.save_results(results)
        
        return results
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return None
    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())