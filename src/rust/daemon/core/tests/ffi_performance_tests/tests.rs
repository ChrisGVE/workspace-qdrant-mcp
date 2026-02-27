//! Unit tests for FFI performance benchmarking.

use std::collections::HashMap;
use std::time::Duration;

use super::benchmarks::simulate_ffi_operation;
use super::tester::FfiPerformanceTester;
use super::types::{
    ConcurrencyBenchmark, DataTransferBenchmark, FfiPerformanceConfig, FunctionCallBenchmarks,
};

#[tokio::test]
async fn test_ffi_performance_tester_creation() {
    let tester = FfiPerformanceTester::new();
    assert_eq!(tester.config.measurement_iterations, 1000);
}

#[tokio::test]
async fn test_data_transfer_simulation() {
    let tester = FfiPerformanceTester::new();
    let test_data = vec![1, 2, 3, 4, 5];

    let python_data = tester.simulate_rust_to_python_transfer(&test_data);
    assert!(!python_data.is_empty());

    let rust_data = tester.simulate_python_to_rust_transfer(&python_data);
    assert!(!rust_data.is_empty());
}

#[tokio::test]
async fn test_serialization_benchmark() {
    let tester = FfiPerformanceTester::new();
    let result = tester.benchmark_serialization_format("json").await;
    assert!(result.is_ok());

    let benchmark = result.unwrap();
    assert_eq!(benchmark.format, "json");
    assert!(benchmark.serialize_ns > 0);
    assert!(benchmark.deserialize_ns > 0);
}

#[tokio::test]
async fn test_async_operations_benchmark() {
    let tester = FfiPerformanceTester::new();
    let result = tester.benchmark_async_operations().await;
    assert!(result.is_ok());

    let benchmark = result.unwrap();
    assert!(benchmark.async_function_call_ns > 0);
}

#[tokio::test]
async fn test_memory_copy_benchmark() {
    let tester = FfiPerformanceTester::new();
    let result = tester.benchmark_memory_copy_size(1024).await;
    assert!(result.is_ok());

    let benchmark = result.unwrap();
    assert_eq!(benchmark.size, 1024);
    assert!(benchmark.copy_ns > 0);
    assert!(benchmark.clone_ns > 0);
}

#[tokio::test]
async fn test_function_calls_benchmark() {
    let tester = FfiPerformanceTester::new();
    let result = tester.benchmark_function_calls().await;
    assert!(result.is_ok());

    let _benchmark = result.unwrap();
}

#[tokio::test]
async fn test_concurrency_benchmark() {
    let tester = FfiPerformanceTester::new();
    let result = tester.benchmark_concurrency_level(2).await;
    assert!(result.is_ok());

    let benchmark = result.unwrap();
    assert_eq!(benchmark.thread_count, 2);
    assert!(benchmark.total_operations > 0);
    assert!(benchmark.operations_per_second >= 0.0);
}

#[test]
fn test_function_call_primitives() {
    use super::benchmarks::{
        function_with_callback, function_with_error_handling, function_with_parameters,
        function_with_return_value, simple_function,
    };

    assert_eq!(simple_function(), 42);
    assert_eq!(function_with_parameters(10, "test"), 20);
    assert_eq!(function_with_return_value(5), "Result: 5");
    assert_eq!(function_with_error_handling(5), Ok(10));
    assert_eq!(function_with_error_handling(-1), Err("Negative number"));
    assert_eq!(function_with_callback(|| 21), 42);
}

#[test]
fn test_simulate_ffi_operation() {
    let input = vec![1, 2, 3, 4, 5];
    let output = simulate_ffi_operation(&input);

    assert_eq!(output.len(), input.len());
    // Each byte should be incremented by 1
    for (i, &byte) in output.iter().enumerate() {
        assert_eq!(byte, input[i].wrapping_add(1));
    }
}

#[tokio::test]
async fn test_performance_data_recording() {
    let tester = FfiPerformanceTester::new();

    tester.record_performance_data("test_op", Duration::from_millis(10), 1024);

    let data = tester.performance_data.lock().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].operation, "test_op");
    assert_eq!(data[0].data_size, 1024);
}

#[tokio::test]
async fn test_performance_score_calculation() {
    let tester = FfiPerformanceTester::new();

    let mut data_transfer = HashMap::new();
    data_transfer.insert(
        1024,
        DataTransferBenchmark {
            data_size: 1024,
            rust_to_python_ns: 1000,
            python_to_rust_ns: 1000,
            roundtrip_ns: 2000,
            throughput_mbps: 500.0,
            memory_overhead: 100,
            cpu_overhead_percent: 5.0,
        },
    );

    let function_calls = FunctionCallBenchmarks {
        simple_call_ns: 50,
        with_parameters_ns: 100,
        with_return_value_ns: 150,
        with_error_handling_ns: 200,
        callback_overhead_ns: 300,
    };

    let mut concurrency = HashMap::new();
    concurrency.insert(
        4,
        ConcurrencyBenchmark {
            thread_count: 4,
            total_operations: 1000,
            total_duration_ms: 1000,
            operations_per_second: 1000.0,
            scalability_efficiency: 0.75,
            contention_detected: false,
        },
    );

    let score = tester.calculate_performance_score(&data_transfer, &function_calls, &concurrency);

    // Should be a good score due to low function call overhead and good scalability
    assert!(score >= 80.0);
    assert!(score <= 100.0);
}

#[tokio::test]
async fn test_full_performance_test_suite() {
    let mut config = FfiPerformanceConfig::default();
    config.measurement_iterations = 10; // Reduce for faster test
    config.data_sizes = vec![100, 1000]; // Reduce sizes for faster test
    config.concurrency_levels = vec![1, 2]; // Reduce levels for faster test

    let tester = FfiPerformanceTester::with_config(config);
    let results = tester.run_performance_tests().await;

    assert!(results.is_ok());
    let performance_results = results.unwrap();

    assert!(!performance_results.data_transfer_benchmarks.is_empty());
    assert!(!performance_results.serialization_benchmarks.is_empty());
    assert!(!performance_results.memory_copy_benchmarks.is_empty());
    assert!(!performance_results.concurrency_benchmarks.is_empty());
    assert!(performance_results.overall_performance_score >= 0.0);
    assert!(performance_results.overall_performance_score <= 100.0);
}
