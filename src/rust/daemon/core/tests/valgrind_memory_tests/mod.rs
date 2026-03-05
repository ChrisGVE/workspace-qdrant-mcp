//! Valgrind integration tests for memory safety validation
//!
//! This module provides Valgrind-based memory safety testing including:
//! 1. Memory leak detection with Memcheck
//! 2. Cache performance analysis with Cachegrind
//! 3. Heap profiling with Massif
//! 4. Thread error detection with Helgrind
//! 5. Data race detection with DRD

mod suite;
mod types;

// Re-export all public types for external consumers
pub use suite::ValgrindTestSuite;
pub use types::{
    AllocationSnapshot, BarrierReuse, CachegrindResults, ConditionVariableError, DataRace,
    DrdResults, ErrorSeverity, HeapAllocation, HelgrindResults, LockContention, LockOrderViolation,
    MassifResults, MemcheckResults, MemoryError, PerformanceHotspot, RaceCondition,
    ThreadApiMisuse, ValgrindConfig, ValgrindError, ValgrindLeakCheck, ValgrindResults,
    ValgrindStatus,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn test_valgrind_availability() {
        // This test checks if Valgrind is available
        let available = ValgrindTestSuite::is_valgrind_available();

        if cfg!(target_os = "linux") {
            // On Linux, Valgrind might be available
            println!("Valgrind available on Linux: {}", available);
        } else {
            // On other platforms, it should not be available
            assert!(
                !available,
                "Valgrind should not be available on non-Linux platforms"
            );
        }
    }

    #[test]
    fn test_valgrind_config_default() {
        let config = ValgrindConfig::default();
        assert!(config.enable_memcheck);
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(matches!(config.leak_check, ValgrindLeakCheck::Full));
    }

    #[test]
    fn test_leak_check_to_string() {
        assert_eq!(suite::leak_check_to_string(&ValgrindLeakCheck::No), "no");
        assert_eq!(
            suite::leak_check_to_string(&ValgrindLeakCheck::Summary),
            "summary"
        );
        assert_eq!(suite::leak_check_to_string(&ValgrindLeakCheck::Yes), "yes");
        assert_eq!(
            suite::leak_check_to_string(&ValgrindLeakCheck::Full),
            "full"
        );
    }

    #[test]
    fn test_extract_number() {
        let xml_content = r#"<definitely_lost>1024</definitely_lost>"#;
        let result = suite::extract_number(xml_content, "definitely_lost");
        assert_eq!(result, Some(1024));
    }

    #[tokio::test]
    async fn test_valgrind_suite_creation() {
        let binary_path = PathBuf::from("/bin/echo");

        if ValgrindTestSuite::is_valgrind_available() {
            let suite = ValgrindTestSuite::new(binary_path);
            assert!(suite.is_ok());
        } else {
            let suite = ValgrindTestSuite::new(binary_path);
            assert!(matches!(suite.unwrap_err(), ValgrindError::NotAvailable));
        }
    }

    #[test]
    fn test_valgrind_results_serialization() {
        let results = ValgrindResults {
            memcheck_results: Some(MemcheckResults {
                definitely_lost: 0,
                indirectly_lost: 0,
                possibly_lost: 0,
                still_reachable: 1024,
                suppressed: 0,
                invalid_reads: 0,
                invalid_writes: 0,
                invalid_frees: 0,
                mismatched_frees: 0,
                error_summary: Vec::new(),
            }),
            cachegrind_results: None,
            massif_results: None,
            helgrind_results: None,
            drd_results: None,
            overall_status: ValgrindStatus::Passed,
        };

        let serialized = serde_json::to_string(&results).unwrap();
        let deserialized: ValgrindResults = serde_json::from_str(&serialized).unwrap();

        assert!(matches!(
            deserialized.overall_status,
            ValgrindStatus::Passed
        ));
        assert!(deserialized.memcheck_results.is_some());
    }
}
