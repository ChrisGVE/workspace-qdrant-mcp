//! Daemon startup operations: recovery, reconciliation, and rules backfill.
//!
//! This module consolidates all startup-phase logic that runs after schema
//! migrations and before the unified queue processor starts processing.
//!
//! # Submodules
//! - [`recovery`] — Filesystem recovery: reconciles `tracked_files` with disk
//! - [`reconciliation`] — Stale state cleanup and watch folder validation
//! - [`rules_backfill`] — Rules mirror backfill from Qdrant

pub mod reconciliation;
pub mod recovery;
pub mod rules_backfill;

// Re-export all public items for backward-compatible imports via `crate::startup::*`
pub use recovery::{
    check_base_point_migration, run_startup_recovery,
    RecoveryStats, FullRecoveryStats,
};
pub use reconciliation::{
    clean_stale_state, validate_watch_folders,
    StaleCleanupStats, WatchValidationStats,
};
pub use rules_backfill::{
    backfill_rules_mirror, RulesBackfillStats,
};
