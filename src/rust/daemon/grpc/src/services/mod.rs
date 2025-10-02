//! Modular gRPC service implementations
//!
//! This module contains the individual service implementations for the
//! workspace_daemon proto: SystemService, CollectionService, DocumentService

pub mod collection_service;

// Re-export service implementations
pub use collection_service::CollectionServiceImpl;
