//! Request queue with timeout, deduplication, and priority boosting
//!
//! Manages per-priority queues for incoming tasks, providing content-hash
//! deduplication, configurable timeouts, and automatic priority boosting
//! for aged requests.

mod config;
mod queue;
mod types;

pub use config::{QueueConfig, QueueConfigBuilder};
pub use queue::RequestQueue;
pub use types::QueueStats;
