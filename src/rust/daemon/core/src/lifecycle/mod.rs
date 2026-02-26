//! State Machine: watch folder and project lifecycle management.
//!
//! Centralizes all `watch_folders.is_active` mutations to fix the split-brain
//! where `DaemonStateManager`, `PriorityManager`, `SystemService`, and
//! `startup::reconciliation` previously mutated the column independently.
//!
//! # Submodules
//! - [`watch_folder`] — `WatchFolderLifecycle` state machine (all `is_active` writes)

pub mod watch_folder;

pub use watch_folder::{WatchFolderLifecycle, WatchFolderLifecycleError};
