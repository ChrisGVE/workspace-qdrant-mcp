//! Cross-platform testing and memory safety validation
//!
//! This module provides comprehensive testing for:
//! 1. Cross-platform behavior validation (Windows/macOS/Linux)
//! 2. Memory safety validation with leak detection
//! 3. Rust-Python FFI performance benchmarking
//! 4. Unsafe code validation
//! 5. Thread safety testing
//! 6. Performance regression detection

pub mod types;
pub mod suite;
mod platform_tests;
mod benchmarks;
mod tests;

// Re-export public types for use by parent module and integration tests
pub use suite::CrossPlatformTestSuite;
pub use types::{
    CrossPlatformTestConfig, CrossPlatformResults, MemorySafetyResults,
    FFIPerformanceResults, ThreadSafetyResults, PerformanceRegressionResults,
};
