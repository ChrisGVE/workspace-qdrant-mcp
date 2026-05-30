//! SQLite read-only access to the daemon's `state.db`.
//!
//! This module provides:
//! - [`manager::StateManager`] — connection lifecycle and degraded-mode
//! - [`SharedStateManager`] — `Send + Sync` wrapper for cross-thread dispatch
//! - [`queue_stats`] — unified_queue statistics
//! - [`rules_mirror`] — rules_mirror reads
//! - [`scratchpad_mirror`] — scratchpad_mirror reads
//! - [`project_queries`] — watch_folders / instance queries
//! - [`tag_queries`] — tags / keyword_baskets / canonical_tags queries
//! - [`tracked_files`] — tracked_files / submodules / project_components queries

pub mod manager;
pub mod project_queries;
pub mod queue_stats;
pub mod rules_mirror;
pub mod scratchpad_mirror;
pub mod tag_queries;
pub mod tracked_files;

// Convenience re-exports of the most commonly used types.
pub use manager::{DegradedReason, QueryResult, QueryStatus, StateManager};

// ---------------------------------------------------------------------------
// SharedStateManager — Send + Sync wrapper
// ---------------------------------------------------------------------------
//
// `rusqlite::Connection` contains `RefCell` internally and is therefore NOT
// `Sync`.  Wrapping `StateManager` in a `std::sync::Mutex` makes it `Sync`
// (since `Mutex<T>: Sync` whenever `T: Send`, and `StateManager: Send`),
// allowing the dispatcher to satisfy the `ServerHandler: Send + Sync` bound
// without holding a bare `&StateManager` across any `.await` point.
//
// # Usage contract
//
// All callers MUST lock, perform synchronous SQLite queries, then DROP the
// guard before any `.await`.  Holding a `MutexGuard<StateManager>` across an
// await point risks deadlock and violates this contract.

use std::sync::{Mutex, MutexGuard};

/// `Send + Sync` wrapper around `StateManager`.
///
/// Handlers receive `&SharedStateManager` (which IS `Send` because
/// `SharedStateManager: Sync`).  They call [`SharedStateManager::lock`] to
/// get a guard, perform synchronous SQLite reads, then DROP the guard before
/// the first `.await` in the caller.
pub struct SharedStateManager(Mutex<StateManager>);

impl SharedStateManager {
    /// Wrap an existing `StateManager`.
    pub fn new(state: StateManager) -> Self {
        Self(Mutex::new(state))
    }

    /// Lock the inner `StateManager` for synchronous access.
    ///
    /// # Panics
    /// Panics if the mutex was poisoned (only possible if a previous holder
    /// panicked while holding the lock — should never occur in production).
    pub fn lock(&self) -> MutexGuard<'_, StateManager> {
        self.0.lock().expect("StateManager mutex was poisoned")
    }
}

// SAFETY contract: callers MUST drop the guard before any `.await`.
// All production callers in this crate follow that contract.
// `Mutex<StateManager>: Sync` because `StateManager: Send`.
// `&SharedStateManager` is therefore `Send`.

// `RulesReader for SharedStateManager` is implemented in
// `tools/rules/traits.rs` to avoid a `sqlite → tools → sqlite` cycle.
// See: `impl RulesReader for SharedStateManager` in that file.
