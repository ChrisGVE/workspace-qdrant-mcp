//! Unit tests for the unsafe code audit infrastructure
//!
//! Validates the auditor, memory tracker, concurrency tracker,
//! safety score calculation, and UTF-16 edge cases.

use std::thread;

use serial_test::serial;

use super::auditor::{AccessType, ConcurrencyTracker, MemoryTracker, UnsafeCodeAuditor};
use super::types::{
    ConcurrencySafety, FfiSafety, SafetyViolation, ViolationSeverity, ViolationType,
};

#[tokio::test]
async fn test_unsafe_auditor_creation() {
    let auditor = UnsafeCodeAuditor::new();
    let violations = auditor.violations.lock().unwrap();
    assert_eq!(violations.len(), 0);
}

#[tokio::test]
async fn test_memory_tracker() {
    let mut tracker = MemoryTracker::new();

    tracker.track_allocation(0x1000, 1024, "test_location".to_string());
    assert_eq!(tracker.total_allocated, 1024);

    let info = tracker.track_deallocation(0x1000);
    assert!(info.is_some());
    assert_eq!(tracker.total_allocated, 0);
}

#[tokio::test]
async fn test_concurrency_tracker() {
    let mut tracker = ConcurrencyTracker::new();
    let thread_id = thread::current().id();

    tracker.register_thread(thread_id);
    tracker.record_access(0x2000, AccessType::Read);

    assert!(tracker.active_threads.contains_key(&thread_id));
    assert!(tracker.shared_accesses.contains_key(&0x2000));
}

#[tokio::test]
async fn test_violation_recording() {
    let auditor = UnsafeCodeAuditor::new();

    let violation = SafetyViolation {
        location: "test.rs:1".to_string(),
        violation_type: ViolationType::MemoryLeak,
        severity: ViolationSeverity::Medium,
        description: "Test violation".to_string(),
        suggested_fix: "Fix the test".to_string(),
        stack_trace: None,
    };

    auditor.record_violation(violation);

    let violations = auditor.violations.lock().unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].location, "test.rs:1");
}

#[tokio::test]
async fn test_safety_score_calculation() {
    let auditor = UnsafeCodeAuditor::new();

    auditor.record_violation(SafetyViolation {
        location: "test.rs:1".to_string(),
        violation_type: ViolationType::MemoryLeak,
        severity: ViolationSeverity::Medium,
        description: "Test violation".to_string(),
        suggested_fix: "Fix the test".to_string(),
        stack_trace: None,
    });

    let concurrency_safety = ConcurrencySafety {
        thread_safety_verified: true,
        data_race_free: true,
        send_sync_correctness: true,
        atomic_operations_correct: true,
        lock_free_correctness: true,
        aba_problem_prevention: true,
        memory_ordering_correct: true,
    };

    let ffi_safety = FfiSafety {
        c_string_handling: true,
        null_termination_verified: true,
        encoding_safety: true,
        lifetime_management: true,
        callback_safety: true,
        abi_compatibility: true,
        error_propagation: true,
    };

    let score = auditor.calculate_safety_score(&concurrency_safety, &ffi_safety);

    // 100 - 5 (medium) + 5 (concurrency) + 5 (ffi) = 105, capped at 100
    assert_eq!(score, 100.0);
}

#[tokio::test]
#[serial]
async fn test_fd_duplication_safety() {
    let auditor = UnsafeCodeAuditor::new();
    let result = auditor.test_fd_duplication_safety().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_concurrent_operations_safety() {
    let auditor = UnsafeCodeAuditor::new();

    let fd_result = auditor.test_concurrent_fd_access().await;
    assert!(fd_result.is_ok());

    let string_result = auditor.test_concurrent_string_operations().await;
    assert!(string_result.is_ok());
}

#[test]
fn test_utf16_conversion_edge_cases() {
    // Test empty string
    let empty: Vec<u16> = "".encode_utf16().chain(std::iter::once(0)).collect();
    assert_eq!(empty, vec![0]);

    // Test Unicode string
    let unicode: Vec<u16> = "\u{6D4B}\u{8BD5}"
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    assert!(unicode.len() > 2);
    assert_eq!(unicode.last(), Some(&0));

    // Test very long string
    let long_string = "a".repeat(10000);
    let long_wide: Vec<u16> = long_string
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    assert_eq!(long_wide.len(), 10001);
    assert_eq!(long_wide.last(), Some(&0));
}

#[tokio::test]
async fn test_full_unsafe_audit() {
    let auditor = UnsafeCodeAuditor::new();
    let results = auditor.audit_unsafe_code().await;

    assert!(results.is_ok());
    let audit_results = results.unwrap();

    assert_eq!(audit_results.total_unsafe_blocks, 8);
    assert_eq!(audit_results.blocks_audited, 8);
    assert!(audit_results.overall_safety_score >= 0.0);
    assert!(audit_results.overall_safety_score <= 100.0);
}
