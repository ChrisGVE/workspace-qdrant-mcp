//! Valgrind type definitions, configuration, results, and error types
//!
//! This module contains all the data structures used by the Valgrind
//! integration test suite, including configuration, result types for
//! each Valgrind tool, and error handling.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Valgrind test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValgrindConfig {
    pub enable_memcheck: bool,
    pub enable_cachegrind: bool,
    pub enable_massif: bool,
    pub enable_helgrind: bool,
    pub enable_drd: bool,
    pub timeout: Duration,
    pub leak_check: ValgrindLeakCheck,
    pub show_reachable: bool,
    pub track_origins: bool,
    pub suppressions_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValgrindLeakCheck {
    No,
    Summary,
    Yes,
    Full,
}

impl Default for ValgrindConfig {
    fn default() -> Self {
        Self {
            enable_memcheck: true,
            enable_cachegrind: cfg!(target_os = "linux"),
            enable_massif: cfg!(target_os = "linux"),
            enable_helgrind: cfg!(target_os = "linux"),
            enable_drd: cfg!(target_os = "linux"),
            timeout: Duration::from_secs(300), // 5 minutes
            leak_check: ValgrindLeakCheck::Full,
            show_reachable: true,
            track_origins: true,
            suppressions_file: None,
        }
    }
}

/// Valgrind test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValgrindResults {
    pub memcheck_results: Option<MemcheckResults>,
    pub cachegrind_results: Option<CachegrindResults>,
    pub massif_results: Option<MassifResults>,
    pub helgrind_results: Option<HelgrindResults>,
    pub drd_results: Option<DrdResults>,
    pub overall_status: ValgrindStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValgrindStatus {
    Passed,
    Failed,
    Timeout,
    NotAvailable,
}

/// Memcheck results for memory error detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemcheckResults {
    pub definitely_lost: usize,
    pub indirectly_lost: usize,
    pub possibly_lost: usize,
    pub still_reachable: usize,
    pub suppressed: usize,
    pub invalid_reads: usize,
    pub invalid_writes: usize,
    pub invalid_frees: usize,
    pub mismatched_frees: usize,
    pub error_summary: Vec<MemoryError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryError {
    pub error_type: String,
    pub error_count: usize,
    pub stack_trace: String,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Cachegrind results for cache performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachegrindResults {
    pub instruction_cache_refs: u64,
    pub instruction_cache_misses: u64,
    pub data_cache_refs: u64,
    pub data_cache_misses: u64,
    pub l2_cache_refs: u64,
    pub l2_cache_misses: u64,
    pub cache_miss_rate: f64,
    pub performance_hotspots: Vec<PerformanceHotspot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHotspot {
    pub function_name: String,
    pub instruction_count: u64,
    pub cache_misses: u64,
    pub percentage_of_total: f64,
}

/// Massif results for heap profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassifResults {
    pub peak_heap_usage: usize,
    pub peak_extra_heap_usage: usize,
    pub peak_stacks_usage: usize,
    pub heap_tree: Vec<HeapAllocation>,
    pub allocation_timeline: Vec<AllocationSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapAllocation {
    pub bytes: usize,
    pub percentage: f64,
    pub function: String,
    pub children: Vec<HeapAllocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationSnapshot {
    pub time: f64,
    pub heap_size: usize,
    pub useful_heap_size: usize,
    pub extra_heap_size: usize,
    pub stacks_size: usize,
}

/// Helgrind results for thread error detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelgrindResults {
    pub race_conditions: Vec<RaceCondition>,
    pub lock_order_violations: Vec<LockOrderViolation>,
    pub api_misuse: Vec<ThreadApiMisuse>,
    pub total_errors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceCondition {
    pub address: String,
    pub size: usize,
    pub thread1_stack: String,
    pub thread2_stack: String,
    pub access_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockOrderViolation {
    pub lock1: String,
    pub lock2: String,
    pub thread1_stack: String,
    pub thread2_stack: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadApiMisuse {
    pub api_call: String,
    pub error_description: String,
    pub stack_trace: String,
}

/// DRD (Data Race Detector) results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrdResults {
    pub data_races: Vec<DataRace>,
    pub lock_contention: Vec<LockContention>,
    pub barrier_reuse: Vec<BarrierReuse>,
    pub condition_variable_errors: Vec<ConditionVariableError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRace {
    pub address: String,
    pub size: usize,
    pub conflicting_access: String,
    pub first_access_stack: String,
    pub second_access_stack: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockContention {
    pub lock_address: String,
    pub contention_count: usize,
    pub max_wait_time: Duration,
    pub average_wait_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierReuse {
    pub barrier_address: String,
    pub reuse_error: String,
    pub stack_trace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionVariableError {
    pub cv_address: String,
    pub mutex_address: String,
    pub error_type: String,
    pub stack_trace: String,
}

/// Valgrind test errors
#[derive(Error, Debug)]
pub enum ValgrindError {
    #[error("Valgrind not available on this platform")]
    NotAvailable,

    #[error("Valgrind executable not found")]
    NotInstalled,

    #[error("Test timeout after {seconds} seconds")]
    Timeout { seconds: u64 },

    #[error("Valgrind execution failed: {message}")]
    ExecutionFailed { message: String },

    #[error("Failed to parse Valgrind output: {message}")]
    ParseError { message: String },

    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
}
