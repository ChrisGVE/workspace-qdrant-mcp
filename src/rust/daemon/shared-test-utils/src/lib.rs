//! Shared test utilities for the workspace-qdrant-mcp Rust workspace
//!
//! This crate provides common testing infrastructure, mock implementations,
//! and test helpers that can be used across all workspace members.

// pub mod containers; // Temporarily disabled due to dependency conflicts
pub mod fixtures;
pub mod matchers;
// pub mod mocks; // Temporarily disabled due to dependency conflicts
pub mod proptest_generators;
pub mod test_helpers;

// Re-export commonly used testing types and functions
// pub use containers::*; // Temporarily disabled
pub use fixtures::*;
pub use matchers::*;
// pub use mocks::*; // Temporarily disabled
pub use proptest_generators::*;
pub use test_helpers::*;

// Re-export testing dependencies for convenience
pub use proptest;
pub use serial_test;
pub use tempfile;
pub use test_log;
// pub use testcontainers; // Temporarily disabled
// pub use testcontainers_modules; // Temporarily disabled
pub use tokio_test;
pub use tracing_test;
// pub use wiremock; // Temporarily disabled

/// Common test configuration and constants
pub mod config {
    use std::time::Duration;

    /// Default timeout for test operations
    pub const DEFAULT_TEST_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default timeout for async operations in tests
    pub const ASYNC_OPERATION_TIMEOUT: Duration = Duration::from_secs(10);

    /// Default timeout for container startup
    pub const CONTAINER_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

    /// Default number of items for stress tests
    pub const STRESS_TEST_ITEMS: usize = 1000;

    /// Default collection name for tests
    pub const TEST_COLLECTION: &str = "test_collection";

    /// Test project name
    pub const TEST_PROJECT: &str = "test_project";

    /// Test embedding dimension
    pub const TEST_EMBEDDING_DIM: usize = 384;
}

/// Common test result type
pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Initialize test environment with proper logging and tracing
pub fn init_test_env() {
    use tracing_subscriber::{fmt, EnvFilter};
    use std::sync::Once;

    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"));

        fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

/// Macro for creating async test functions with proper setup
#[macro_export]
macro_rules! async_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        async fn $name() -> $crate::TestResult {
            $crate::init_test_env();
            $body
        }
    };
}

/// Macro for creating serial async tests that cannot run in parallel
#[macro_export]
macro_rules! serial_async_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        #[serial_test::serial]
        async fn $name() -> $crate::TestResult {
            $crate::init_test_env();
            $body
        }
    };
}

/// Macro for creating property-based tests
#[macro_export]
macro_rules! property_test {
    ($name:ident, $strategy:expr, $test_fn:expr) => {
        #[test]
        fn $name() {
            $crate::init_test_env();
            proptest::proptest!($strategy, $test_fn);
        }
    };
}