//! Cross-platform testing and memory safety validation
//!
//! This module provides comprehensive testing for:
//! 1. Cross-platform behavior validation (Windows/macOS/Linux)
//! 2. Memory safety validation with leak detection
//! 3. Rust-Python FFI performance benchmarking
//! 4. Unsafe code validation
//! 5. Thread safety testing
//! 6. Performance regression detection

mod benchmarks;
mod platform_tests;
pub mod suite;
mod tests;
pub mod types;

// Re-export public types for use by parent module and integration tests
pub use suite::CrossPlatformTestSuite;
pub use types::{
    CrossPlatformResults, CrossPlatformTestConfig, FFIPerformanceResults, MemorySafetyResults,
    PerformanceRegressionResults, ThreadSafetyResults,
};
