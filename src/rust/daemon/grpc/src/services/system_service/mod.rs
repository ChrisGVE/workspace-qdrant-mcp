//! SystemService gRPC implementation
//!
//! Handles system health monitoring, status reporting, refresh signaling,
//! and lifecycle management operations.
//! Provides 10 RPCs: Health, GetStatus, GetMetrics, GetQueueStats, Shutdown,
//! SendRefreshSignal, NotifyServerStatus, PauseAllWatchers, ResumeAllWatchers,
//! RebuildIndex

mod rebuild;
mod rpc_handlers;
mod service_impl;
mod types;

// Re-export primary types
pub use service_impl::SystemServiceImpl;
pub use types::{ServerStatusEntry, ServerStatusStore};

#[cfg(test)]
mod tests;
