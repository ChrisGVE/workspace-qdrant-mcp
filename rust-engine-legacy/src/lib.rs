//! Workspace Qdrant Daemon Library
//!
//! This library provides the core functionality for the workspace document processing
//! and vector search daemon.

pub mod config;
pub mod error;
pub mod daemon;
pub mod grpc;
pub mod memory;
pub mod qdrant;

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("workspace_daemon");
}

// Re-export commonly used types
pub use config::*;
pub use error::*;