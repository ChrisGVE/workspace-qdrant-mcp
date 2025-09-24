//! FFI Performance Benchmarks for Rust-Python Interface
//!
//! This module provides comprehensive performance benchmarking for the
//! Rust-Python Foreign Function Interface (FFI) boundary, measuring:
//! - Function call overhead
//! - Data serialization/deserialization performance
//! - Memory allocation patterns across FFI boundary
//! - Large data transfer efficiency
//! - Error handling performance impact
//! - Async/await bridge performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

use workspace_qdrant_daemon::{
    daemon::Daemon,
    config::DaemonConfig,
    error::WorkspaceError,
};

/// FFI benchmark data structures
#[repr(C)]
struct FFITestStruct {
    id: u64,
    name: *const c_char,
    data: *const u8,
    data_len: usize,
    score: f64,
}

#[repr(C)]
struct FFIBulkData {
    items: *const FFITestStruct,
    count: usize,
    metadata: *const c_char,
}

/// Simulated Python-to-Rust FFI functions
extern "C" fn rust_process_simple_data(id: u64, value: f64) -> f64 {
    // Simulate simple computation
    id as f64 + value * 2.0
}

extern "C" fn rust_process_string_data(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let c_str = CStr::from_ptr(input);
        if let Ok(str_slice) = c_str.to_str() {
            let processed = format!("processed_{}", str_slice);
            let c_string = CString::new(processed).unwrap();
            c_string.into_raw()
        } else {
            ptr::null_mut()
        }
    }
}

extern "C" fn rust_process_bulk_data(bulk: *const FFIBulkData) -> u64 {
    if bulk.is_null() {
        return 0;
    }

    unsafe {
        let bulk_data = &*bulk;
        let items = std::slice::from_raw_parts(bulk_data.items, bulk_data.count);

        let mut total = 0u64;
        for item in items {
            total += item.id;
            total += (item.score as u64);
        }
        total
    }
}

extern "C" fn rust_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Benchmark helper functions
fn create_test_strings(count: usize, length: usize) -> Vec<CString> {
    (0..count)
        .map(|i| {
            let content = format!("test_string_{}_{}", i, "x".repeat(length - 20));
            CString::new(content).unwrap()
        })
        .collect()
}

fn create_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn create_ffi_test_structs(count: usize) -> (Vec<FFITestStruct>, Vec<CString>, Vec<Vec<u8>>) {
    let strings = create_test_strings(count, 64);
    let data_vecs = (0..count)
        .map(|i| create_test_data(i % 1000 + 100))
        .collect::<Vec<_>>();

    let structs = (0..count)
        .map(|i| FFITestStruct {
            id: i as u64,
            name: strings[i].as_ptr(),
            data: data_vecs[i].as_ptr(),
            data_len: data_vecs[i].len(),
            score: i as f64 * 1.23,
        })
        .collect();

    (structs, strings, data_vecs)
}

/// Core FFI performance benchmarks
fn bench_simple_ffi_calls(c: &mut Criterion) {
    c.bench_function("ffi_simple_call", |b| {
        b.iter(|| {
            let result = rust_process_simple_data(black_box(123), black_box(45.67));
            black_box(result);
        })
    });

    let mut group = c.benchmark_group("ffi_simple_call_batch");
    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut total = 0.0;
                    for i in 0..batch_size {
                        let result = rust_process_simple_data(black_box(i), black_box(i as f64));
                        total += result;
                    }
                    black_box(total);
                })
            },
        );
    }
    group.finish();
}

fn bench_string_ffi_performance(c: &mut Criterion) {
    let test_strings = create_test_strings(100, 128);

    c.bench_function("ffi_string_processing", |b| {
        b.iter(|| {
            let input = black_box(&test_strings[42]);
            let result = rust_process_string_data(input.as_ptr());
            if !result.is_null() {
                rust_free_string(result);
            }
        })
    });

    let mut group = c.benchmark_group("ffi_string_length_impact");
    for string_length in [16, 64, 256, 1024, 4096].iter() {
        let test_string = CString::new("x".repeat(*string_length)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("length", string_length),
            string_length,
            |b, _| {
                b.iter(|| {
                    let result = rust_process_string_data(black_box(test_string.as_ptr()));
                    if !result.is_null() {
                        rust_free_string(result);
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_bulk_data_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_bulk_data_transfer");

    for item_count in [10, 100, 1000, 10000].iter() {
        let (structs, _strings, _data) = create_ffi_test_structs(*item_count);
        let metadata = CString::new("bulk_test_metadata").unwrap();
        let bulk_data = FFIBulkData {
            items: structs.as_ptr(),
            count: structs.len(),
            metadata: metadata.as_ptr(),
        };

        group.bench_with_input(
            BenchmarkId::new("items", item_count),
            item_count,
            |b, _| {
                b.iter(|| {
                    let result = rust_process_bulk_data(black_box(&bulk_data));
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    c.bench_function("ffi_allocation_deallocation", |b| {
        b.iter(|| {
            // Simulate typical FFI allocation pattern
            let test_string = CString::new("allocation_test").unwrap();
            let result = rust_process_string_data(test_string.as_ptr());

            if !result.is_null() {
                // Simulate Python reading the result
                unsafe {
                    let _processed = CStr::from_ptr(result);
                }
                rust_free_string(result);
            }
        })
    });

    let mut group = c.benchmark_group("ffi_allocation_sizes");
    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.bench_with_input(
            BenchmarkId::new("size", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let test_string = CString::new("x".repeat(size)).unwrap();
                    let result = rust_process_string_data(black_box(test_string.as_ptr()));

                    if !result.is_null() {
                        rust_free_string(result);
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_error_handling_overhead(c: &mut Criterion) {
    c.bench_function("ffi_error_handling_success", |b| {
        let valid_string = CString::new("valid_input").unwrap();
        b.iter(|| {
            let result = rust_process_string_data(black_box(valid_string.as_ptr()));
            if !result.is_null() {
                rust_free_string(result);
            }
        })
    });

    c.bench_function("ffi_error_handling_failure", |b| {
        b.iter(|| {
            let result = rust_process_string_data(black_box(ptr::null()));
            // Result should be null for error case
            assert!(result.is_null());
        })
    });
}

fn bench_async_ffi_bridge(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("async_ffi_simple", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate async FFI call
            let start = Instant::now();

            // Simulate Python calling into Rust async function
            tokio::task::yield_now().await;
            let result = rust_process_simple_data(black_box(42), black_box(3.14));

            let _duration = start.elapsed();
            black_box(result);
        })
    });

    let temp_dir = tempfile::tempdir().unwrap();
    let config = DaemonConfig {
        workspace_root: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    c.bench_function("async_ffi_daemon_operation", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate creating and using daemon through FFI
            let daemon_result = Daemon::new(config.clone()).await;
            if let Ok(_daemon) = daemon_result {
                // Simulate daemon operation that would be called from Python
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        })
    });
}

/// Specialized benchmarks for optimization analysis
fn bench_data_serialization_formats(c: &mut Criterion) {
    let test_data = (0..1000).map(|i| (i, format!("item_{}", i), i as f64 * 1.5)).collect::<Vec<_>>();

    let mut group = c.benchmark_group("ffi_serialization_formats");

    // JSON serialization (common Python-Rust bridge format)
    group.bench_function("json_serialization", |b| {
        b.iter(|| {
            let json = serde_json::to_string(&black_box(&test_data)).unwrap();
            let _deserialized: Vec<(i32, String, f64)> = serde_json::from_str(&json).unwrap();
        })
    });

    // Binary serialization with bincode
    group.bench_function("bincode_serialization", |b| {
        b.iter(|| {
            let encoded = bincode::serialize(&black_box(&test_data)).unwrap();
            let _decoded: Vec<(i32, String, f64)> = bincode::deserialize(&encoded).unwrap();
        })
    });

    group.finish();
}

fn bench_ffi_call_frequency_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_call_frequency");

    for calls_per_batch in [1, 10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("calls", calls_per_batch),
            calls_per_batch,
            |b, &calls| {
                b.iter(|| {
                    let mut total = 0.0;
                    for i in 0..calls {
                        // Simulate frequent small FFI calls vs batched calls
                        let result = rust_process_simple_data(i, i as f64);
                        total += result;
                    }
                    black_box(total);
                })
            },
        );
    }
    group.finish();
}

fn bench_concurrent_ffi_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("ffi_concurrent_access");

    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            thread_count,
            |b, &threads| {
                b.to_async(&rt).iter(|| async {
                    let handles = (0..threads).map(|i| {
                        tokio::spawn(async move {
                            let mut total = 0.0;
                            for j in 0..100 {
                                let result = rust_process_simple_data(
                                    black_box((i * 100 + j) as u64),
                                    black_box((i * 100 + j) as f64)
                                );
                                total += result;
                            }
                            total
                        })
                    }).collect::<Vec<_>>();

                    let mut grand_total = 0.0;
                    for handle in handles {
                        grand_total += handle.await.unwrap();
                    }
                    black_box(grand_total);
                })
            },
        );
    }
    group.finish();
}

/// Performance regression detection utilities
pub struct FFIPerformanceBaselines {
    simple_call_ns: f64,
    string_processing_ns: f64,
    bulk_transfer_ns_per_item: f64,
    allocation_ns: f64,
}

impl FFIPerformanceBaselines {
    pub fn measure_current() -> Self {
        let iterations = 10000;

        // Measure simple call performance
        let start = Instant::now();
        for i in 0..iterations {
            rust_process_simple_data(i, i as f64);
        }
        let simple_call_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        // Measure string processing
        let test_string = CString::new("benchmark_test").unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            let result = rust_process_string_data(test_string.as_ptr());
            if !result.is_null() {
                rust_free_string(result);
            }
        }
        let string_processing_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        // Measure bulk transfer
        let (structs, _strings, _data) = create_ffi_test_structs(100);
        let metadata = CString::new("benchmark_metadata").unwrap();
        let bulk_data = FFIBulkData {
            items: structs.as_ptr(),
            count: structs.len(),
            metadata: metadata.as_ptr(),
        };

        let start = Instant::now();
        for _ in 0..1000 {
            rust_process_bulk_data(&bulk_data);
        }
        let bulk_total_ns = start.elapsed().as_nanos() as f64 / 1000.0;
        let bulk_transfer_ns_per_item = bulk_total_ns / structs.len() as f64;

        // Measure allocation overhead
        let start = Instant::now();
        for i in 0..iterations {
            let test_str = format!("allocation_test_{}", i);
            let c_string = CString::new(test_str).unwrap();
            let result = rust_process_string_data(c_string.as_ptr());
            if !result.is_null() {
                rust_free_string(result);
            }
        }
        let allocation_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        Self {
            simple_call_ns,
            string_processing_ns,
            bulk_transfer_ns_per_item,
            allocation_ns,
        }
    }

    pub fn detect_regressions(&self, current: &Self, threshold_percent: f64) -> Vec<String> {
        let mut regressions = Vec::new();

        macro_rules! check_regression {
            ($field:ident, $name:literal) => {
                let regression = (current.$field - self.$field) / self.$field * 100.0;
                if regression > threshold_percent {
                    regressions.push(format!("{}: {:.2}% slower", $name, regression));
                }
            };
        }

        check_regression!(simple_call_ns, "Simple FFI calls");
        check_regression!(string_processing_ns, "String processing");
        check_regression!(bulk_transfer_ns_per_item, "Bulk data transfer");
        check_regression!(allocation_ns, "Memory allocation");

        regressions
    }

    pub fn print_report(&self) {
        println!("FFI Performance Baseline Report:");
        println!("  Simple call: {:.2} ns", self.simple_call_ns);
        println!("  String processing: {:.2} ns", self.string_processing_ns);
        println!("  Bulk transfer per item: {:.2} ns", self.bulk_transfer_ns_per_item);
        println!("  Allocation overhead: {:.2} ns", self.allocation_ns);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_performance_regression_detection() {
        let baseline = FFIPerformanceBaselines::measure_current();
        baseline.print_report();

        // Simulate a regression
        let mut regressed = FFIPerformanceBaselines::measure_current();
        regressed.simple_call_ns *= 1.15; // 15% slower

        let regressions = baseline.detect_regressions(&regressed, 10.0);
        assert!(!regressions.is_empty());
        assert!(regressions[0].contains("Simple FFI calls"));
    }
}

// Additional dependency for bincode serialization benchmark
use bincode;

criterion_group!(
    ffi_benchmarks,
    bench_simple_ffi_calls,
    bench_string_ffi_performance,
    bench_bulk_data_transfer,
    bench_memory_allocation_patterns,
    bench_error_handling_overhead,
    bench_async_ffi_bridge,
    bench_data_serialization_formats,
    bench_ffi_call_frequency_impact,
    bench_concurrent_ffi_access
);

criterion_main!(ffi_benchmarks);