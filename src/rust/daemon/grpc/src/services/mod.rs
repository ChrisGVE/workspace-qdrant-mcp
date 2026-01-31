//! Modular gRPC service implementations
//!
//! This module contains the individual service implementations for the
//! workspace_daemon proto: SystemService, CollectionService, DocumentService,
//! EmbeddingService, ProjectService

pub mod collection_service;
pub mod document_service;
pub mod embedding_service;
pub mod project_service;
pub mod system_service;

// Re-export service implementations
pub use collection_service::CollectionServiceImpl;
pub use document_service::DocumentServiceImpl;
pub use embedding_service::EmbeddingServiceImpl;
pub use project_service::ProjectServiceImpl;
pub use system_service::SystemServiceImpl;
