//! FFI (Foreign Function Interface) performance benchmarking tests
//!
//! This module provides comprehensive performance testing for Rust-Python FFI
//! operations, measuring overhead and identifying optimization opportunities.

mod benchmarks;
mod tester;
mod types;

pub use tester::FfiPerformanceTester;
pub use types::{
    AsyncOperationBenchmarks, ConcurrencyBenchmark, DataTransferBenchmark,
    FfiPerformanceConfig, FfiPerformanceResults, FunctionCallBenchmarks,
    MemoryCopyBenchmark, SerializationBenchmark,
};

#[cfg(test)]
mod tests;
