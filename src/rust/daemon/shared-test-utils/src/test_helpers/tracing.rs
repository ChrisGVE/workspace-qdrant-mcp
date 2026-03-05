//! Test tracing and logging initialisation helper

use std::sync::Once;
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize test tracing/logging (call once per test)
pub fn init_test_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

        fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init()
            .ok(); // Ignore error if already initialized
    });
}
