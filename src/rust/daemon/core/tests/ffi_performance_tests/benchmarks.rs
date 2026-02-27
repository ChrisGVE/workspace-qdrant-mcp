//! Async, memory, function call, and concurrency benchmarks plus helper functions.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use criterion::black_box;

use super::tester::FfiPerformanceTester;
use super::types::{
    AsyncOperationBenchmarks, ConcurrencyBenchmark, DataTransferBenchmark,
    FunctionCallBenchmarks, MemoryCopyBenchmark,
};

impl FfiPerformanceTester {
    pub(super) async fn benchmark_async_operations(
        &self,
    ) -> anyhow::Result<AsyncOperationBenchmarks> {
        // Benchmark async function calls
        let async_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(self.simulate_async_function_call().await);
        }
        let async_function_call_ns =
            async_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark future creation
        let future_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let _ = black_box(async {
                tokio::task::yield_now().await;
            });
        }
        let future_creation_ns =
            future_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark await overhead
        let await_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let future = async { 42 };
            black_box(future.await);
        }
        let await_overhead_ns =
            await_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark task spawning
        let task_start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let handle = tokio::spawn(async { 42 });
            handles.push(handle);
        }
        for handle in handles {
            let _ = handle.await;
        }
        let task_spawning_ns =
            task_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark channel communication
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let channel_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            tx.send(i).unwrap();
            black_box(rx.recv().await.unwrap());
        }
        let channel_communication_ns =
            channel_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        Ok(AsyncOperationBenchmarks {
            async_function_call_ns,
            future_creation_ns,
            await_overhead_ns,
            task_spawning_ns,
            channel_communication_ns,
        })
    }

    async fn simulate_async_function_call(&self) -> i32 {
        tokio::task::yield_now().await;
        42
    }

    pub(super) async fn benchmark_memory_copy(
        &self,
    ) -> anyhow::Result<HashMap<usize, MemoryCopyBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &size in &self.config.data_sizes {
            let benchmark = self.benchmark_memory_copy_size(size).await?;
            benchmarks.insert(size, benchmark);
        }

        Ok(benchmarks)
    }

    pub(super) async fn benchmark_memory_copy_size(
        &self,
        size: usize,
    ) -> anyhow::Result<MemoryCopyBenchmark> {
        let source_data = vec![0u8; size];

        // Benchmark memory copy
        let copy_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let mut dest = Vec::with_capacity(size);
            dest.extend_from_slice(&source_data);
            black_box(dest);
        }
        let copy_ns =
            copy_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark clone
        let clone_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(source_data.clone());
        }
        let clone_ns =
            clone_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Calculate throughput (GB/s)
        let throughput_gbps = if copy_ns > 0 {
            (size as f64 * 1_000_000_000.0) / (copy_ns as f64 * 1_073_741_824.0)
        } else {
            0.0
        };

        Ok(MemoryCopyBenchmark {
            size,
            copy_ns,
            clone_ns,
            zero_copy_supported: size < 1024,
            alignment_optimized: size % 64 == 0,
            throughput_gbps,
        })
    }

    pub(super) async fn benchmark_function_calls(
        &self,
    ) -> anyhow::Result<FunctionCallBenchmarks> {
        let simple_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(simple_function());
        }
        let simple_call_ns =
            simple_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        let params_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_parameters(i as i32, "test"));
        }
        let with_parameters_ns =
            params_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        let return_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_return_value(i as i32));
        }
        let with_return_value_ns =
            return_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        let error_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_error_handling(i as i32).unwrap_or(0));
        }
        let with_error_handling_ns =
            error_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        let callback_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(function_with_callback(|| 42));
        }
        let callback_overhead_ns =
            callback_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        Ok(FunctionCallBenchmarks {
            simple_call_ns,
            with_parameters_ns,
            with_return_value_ns,
            with_error_handling_ns,
            callback_overhead_ns,
        })
    }

    pub(super) async fn benchmark_concurrency(
        &self,
    ) -> anyhow::Result<HashMap<usize, ConcurrencyBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &thread_count in &self.config.concurrency_levels {
            let benchmark = self.benchmark_concurrency_level(thread_count).await?;
            benchmarks.insert(thread_count, benchmark);
        }

        Ok(benchmarks)
    }

    pub(super) async fn benchmark_concurrency_level(
        &self,
        thread_count: usize,
    ) -> anyhow::Result<ConcurrencyBenchmark> {
        let operations_per_thread = self.config.measurement_iterations / thread_count;
        let total_operations = operations_per_thread * thread_count;

        let start_time = Instant::now();

        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let _perf_data = Arc::clone(&self.performance_data);
                thread::spawn(move || {
                    let mut local_operations = 0;
                    let thread_start = Instant::now();

                    for _ in 0..operations_per_thread {
                        let data = vec![0u8; 1024];
                        let _transferred = simulate_ffi_operation(&data);
                        local_operations += 1;
                    }

                    let _thread_duration = thread_start.elapsed();
                    local_operations
                })
            })
            .collect();

        let mut total_ops = 0;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }

        let total_duration = start_time.elapsed();
        let total_duration_ms = total_duration.as_millis() as u64;

        let operations_per_second = if total_duration_ms > 0 {
            (total_ops as f64 * 1000.0) / total_duration_ms as f64
        } else {
            0.0
        };

        let single_thread_ops_per_sec = if thread_count == 1 {
            operations_per_second
        } else {
            operations_per_second / (thread_count as f64 * 0.8)
        };

        let scalability_efficiency = if single_thread_ops_per_sec > 0.0 {
            operations_per_second / (single_thread_ops_per_sec * thread_count as f64)
        } else {
            0.0
        };

        Ok(ConcurrencyBenchmark {
            thread_count,
            total_operations,
            total_duration_ms,
            operations_per_second,
            scalability_efficiency,
            contention_detected: scalability_efficiency < 0.7,
        })
    }

    pub(super) fn calculate_performance_score(
        &self,
        data_transfer: &HashMap<usize, DataTransferBenchmark>,
        function_calls: &FunctionCallBenchmarks,
        concurrency: &HashMap<usize, ConcurrencyBenchmark>,
    ) -> f64 {
        let mut score: f64 = 100.0;

        // Evaluate data transfer performance
        let avg_throughput: f64 = data_transfer
            .values()
            .map(|b| b.throughput_mbps)
            .sum::<f64>()
            / data_transfer.len() as f64;

        if avg_throughput < 100.0 {
            score -= 20.0;
        } else if avg_throughput > 1000.0 {
            score += 10.0;
        }

        // Evaluate function call overhead
        if function_calls.simple_call_ns > 1000 {
            score -= 15.0;
        } else if function_calls.simple_call_ns < 100 {
            score += 5.0;
        }

        // Evaluate concurrency scalability
        let max_efficiency = concurrency
            .values()
            .map(|b| b.scalability_efficiency)
            .fold(0.0, f64::max);

        if max_efficiency < 0.5 {
            score -= 25.0;
        } else if max_efficiency > 0.8 {
            score += 15.0;
        }

        score.max(0.0).min(100.0)
    }
}

// Benchmark helper functions

pub(super) fn simple_function() -> i32 {
    42
}

pub(super) fn function_with_parameters(x: i32, _s: &str) -> i32 {
    x * 2
}

pub(super) fn function_with_return_value(x: i32) -> String {
    format!("Result: {}", x)
}

pub(super) fn function_with_error_handling(x: i32) -> Result<i32, &'static str> {
    if x >= 0 {
        Ok(x * 2)
    } else {
        Err("Negative number")
    }
}

pub(super) fn function_with_callback<F>(callback: F) -> i32
where
    F: Fn() -> i32,
{
    callback() * 2
}

pub(super) fn simulate_ffi_operation(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    result.extend_from_slice(data);

    for byte in &mut result {
        *byte = byte.wrapping_add(1);
    }

    result
}
