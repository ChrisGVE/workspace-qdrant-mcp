//! Branch Switch Protocol (Task 9)
//!
//! Handles branch switch and commit events detected by the git watcher.
//! On branch switch:
//!   1. Uses `diff_tree` to find changed files between old and new commits
//!   2. Batch-updates unchanged files (branch column only, no re-ingestion)
//!   3. Enqueues changed files for re-ingestion via the unified queue
//!   4. Updates `last_commit_hash` in `watch_folders`
//!
//! On new commit (same branch):
//!   1. Uses `diff_tree` to find changed files since last known commit
//!   2. Enqueues changed files for update

mod db;
mod handlers;
mod queue;
mod types;

#[cfg(test)]
mod tests;

pub use handlers::handle_git_event;
pub use types::BranchSwitchStats;
