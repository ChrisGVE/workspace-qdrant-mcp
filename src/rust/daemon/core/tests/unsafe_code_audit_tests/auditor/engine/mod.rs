//! Core audit engine (`UnsafeCodeAuditor`) for unsafe code validation
//!
//! Orchestrates platform-specific, storage, concurrency, and FFI safety checks
//! and aggregates results into `UnsafeAuditResults`.

mod analysis;
mod platform_checks;

use std::sync::{Arc, Mutex, RwLock};

use super::super::types::{
    ConcurrencySafety, FfiSafety, SafetyViolation, UnsafeAuditError, UnsafeAuditResults,
};
use super::trackers::{ConcurrencyTracker, MemoryTracker};

/// Main unsafe code audit suite
pub struct UnsafeCodeAuditor {
    pub(crate) violations: Arc<Mutex<Vec<SafetyViolation>>>,
    _memory_tracker: Arc<RwLock<MemoryTracker>>,
    _concurrency_tracker: Arc<Mutex<ConcurrencyTracker>>,
}

impl Default for UnsafeCodeAuditor {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsafeCodeAuditor {
    pub fn new() -> Self {
        Self {
            violations: Arc::new(Mutex::new(Vec::new())),
            _memory_tracker: Arc::new(RwLock::new(MemoryTracker::new())),
            _concurrency_tracker: Arc::new(Mutex::new(ConcurrencyTracker::new())),
        }
    }

    /// Run comprehensive unsafe code audit
    pub async fn audit_unsafe_code(&self) -> Result<UnsafeAuditResults, UnsafeAuditError> {
        self.audit_platform_watching_unsafe().await?;
        self.audit_storage_unsafe().await?;
        self.audit_service_discovery_unsafe().await?;

        let memory_access_patterns = self.test_memory_access_patterns().await?;
        let invariant_validations = self.validate_invariants().await?;
        let boundary_tests = self.test_boundary_conditions().await?;
        let concurrency_safety = self.analyze_concurrency_safety().await?;
        let ffi_safety = self.analyze_ffi_safety().await?;
        let overall_safety_score = self.calculate_safety_score(&concurrency_safety, &ffi_safety);

        let violations = self.violations.lock().unwrap();

        Ok(UnsafeAuditResults {
            total_unsafe_blocks: 8,
            blocks_audited: 8,
            safety_violations: violations.clone(),
            memory_access_patterns,
            invariant_validations,
            boundary_condition_tests: boundary_tests,
            concurrency_safety,
            ffi_safety,
            overall_safety_score,
        })
    }

    // -----------------------------------------------------------------------
    // Platform-specific audit dispatchers
    // -----------------------------------------------------------------------

    /// Audit unsafe code in platform watching module
    async fn audit_platform_watching_unsafe(&self) -> Result<(), UnsafeAuditError> {
        #[cfg(target_os = "windows")]
        self.test_windows_file_watching_safety().await?;

        #[cfg(target_os = "linux")]
        self.test_linux_file_watching_safety().await?;

        #[cfg(target_os = "macos")]
        self.test_macos_file_watching_safety().await?;

        Ok(())
    }

    /// Audit unsafe code in storage module
    async fn audit_storage_unsafe(&self) -> Result<(), UnsafeAuditError> {
        self.test_fd_duplication_safety().await?;
        self.test_stdio_redirection_safety().await?;
        Ok(())
    }

    /// Audit unsafe code in service discovery module
    async fn audit_service_discovery_unsafe(&self) -> Result<(), UnsafeAuditError> {
        self.test_service_discovery_safety().await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Scoring and violation recording
    // -----------------------------------------------------------------------

    pub(crate) fn calculate_safety_score(
        &self,
        concurrency: &ConcurrencySafety,
        ffi: &FfiSafety,
    ) -> f64 {
        let violations = self.violations.lock().unwrap();

        let mut score: f64 = 100.0;

        for violation in violations.iter() {
            use super::super::types::ViolationSeverity;
            let deduction = match violation.severity {
                ViolationSeverity::Low => 1.0,
                ViolationSeverity::Medium => 5.0,
                ViolationSeverity::High => 15.0,
                ViolationSeverity::Critical => 30.0,
            };
            score -= deduction;
        }

        if concurrency.thread_safety_verified && concurrency.data_race_free {
            score += 5.0;
        }

        if ffi.c_string_handling && ffi.null_termination_verified {
            score += 5.0;
        }

        score.max(0.0).min(100.0)
    }

    pub(crate) fn record_violation(&self, violation: SafetyViolation) {
        let mut violations = self.violations.lock().unwrap();
        violations.push(violation);
    }
}
