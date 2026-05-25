//! Branch discovery: resilient population of branch membership data.
//!
//! When the daemon encounters a branch with no tracked_files entries, the
//! discovery algorithm scans the filesystem, compares content hashes against
//! known tracked files, and classifies files as SHARED (add branch to existing
//! points) or NOVEL (queue for embedding). This handles daemon restarts,
//! worktree additions, and database wipes without relying on creation-time events.

mod db;
mod scanner;

#[cfg(test)]
mod tests;

pub use scanner::{BranchDiscoveryResult, DiscoveryScanner};
