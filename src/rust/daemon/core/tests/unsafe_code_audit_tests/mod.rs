//! Unsafe code audit and validation tests
//!
//! This module provides comprehensive testing and validation for all unsafe code
//! blocks in the codebase, ensuring memory safety and correctness under all
//! conditions.
//!
//! Submodules:
//! - `types` -- shared type definitions and error types
//! - `auditor` -- core audit engine and helper trackers
//! - `proptest_checks` -- property-based tests for unsafe patterns
//! - `unit_tests` -- deterministic unit tests for the audit infrastructure

pub mod types;
pub mod auditor;
mod proptest_checks;
#[cfg(test)]
mod unit_tests;

// Re-export public API expected by the parent tests/mod.rs
pub use types::{
    UnsafeAuditResults, SafetyViolation, ViolationType, ViolationSeverity,
    MemoryAccessPattern, InvariantValidation, BoundaryTest, ConcurrencySafety, FfiSafety,
};
pub use auditor::UnsafeCodeAuditor;
