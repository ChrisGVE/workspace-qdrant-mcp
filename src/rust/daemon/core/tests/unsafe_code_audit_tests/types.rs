//! Type definitions for unsafe code audit tests
//!
//! Shared types used across audit, property-based, and unit test modules.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Unsafe code audit results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsafeAuditResults {
    pub total_unsafe_blocks: usize,
    pub blocks_audited: usize,
    pub safety_violations: Vec<SafetyViolation>,
    pub memory_access_patterns: HashMap<String, MemoryAccessPattern>,
    pub invariant_validations: HashMap<String, InvariantValidation>,
    pub boundary_condition_tests: HashMap<String, BoundaryTest>,
    pub concurrency_safety: ConcurrencySafety,
    pub ffi_safety: FfiSafety,
    pub overall_safety_score: f64,
}

/// Safety violation detected in unsafe code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub location: String,
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub description: String,
    pub suggested_fix: String,
    pub stack_trace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    MemoryLeak,
    UseAfterFree,
    DoubleFree,
    BufferOverflow,
    BufferUnderflow,
    NullPointerDereference,
    DanglingPointer,
    DataRace,
    InvalidCast,
    AlignmentViolation,
    LifetimeViolation,
    UndefinedBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,      // Style or performance issue
    Medium,   // Potential problem under specific conditions
    High,     // Likely to cause issues in production
    Critical, // Immediate memory safety violation
}

/// Memory access pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessPattern {
    pub read_operations: usize,
    pub write_operations: usize,
    pub allocation_operations: usize,
    pub deallocation_operations: usize,
    pub pointer_arithmetic: usize,
    pub bounds_checked: bool,
    pub null_checked: bool,
    pub alignment_verified: bool,
}

/// Invariant validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantValidation {
    pub invariant_description: String,
    pub validation_method: String,
    pub pre_conditions_met: bool,
    pub post_conditions_met: bool,
    pub loop_invariants_maintained: bool,
    pub exception_safety: ExceptionSafety,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionSafety {
    NoGuarantee, // No guarantees about state if panic occurs
    Basic,       // Object remains in valid state but may have changed
    Strong,      // Operation either succeeds or has no effect
    NoThrow,     // Operation guaranteed not to panic
}

/// Boundary condition test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryTest {
    pub test_name: String,
    pub edge_cases_tested: Vec<String>,
    pub null_pointer_handling: bool,
    pub zero_size_handling: bool,
    pub max_size_handling: bool,
    pub negative_size_handling: bool,
    pub alignment_boundary_tests: bool,
    pub all_tests_passed: bool,
}

/// Concurrency safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencySafety {
    pub thread_safety_verified: bool,
    pub data_race_free: bool,
    pub send_sync_correctness: bool,
    pub atomic_operations_correct: bool,
    pub lock_free_correctness: bool,
    pub aba_problem_prevention: bool,
    pub memory_ordering_correct: bool,
}

/// FFI (Foreign Function Interface) safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiSafety {
    pub c_string_handling: bool,
    pub null_termination_verified: bool,
    pub encoding_safety: bool,
    pub lifetime_management: bool,
    pub callback_safety: bool,
    pub abi_compatibility: bool,
    pub error_propagation: bool,
}

/// Unsafe code audit errors
#[derive(Error, Debug)]
pub enum UnsafeAuditError {
    #[error("Memory safety violation: {message}")]
    MemorySafetyViolation { message: String },

    #[error("Concurrency safety violation: {message}")]
    ConcurrencySafetyViolation { message: String },

    #[error("FFI safety violation: {message}")]
    FfiSafetyViolation { message: String },

    #[error("Invariant violation: {message}")]
    InvariantViolation { message: String },

    #[error("Test execution failed: {message}")]
    TestExecutionFailed { message: String },
}
