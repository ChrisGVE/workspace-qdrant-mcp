//! Modular gRPC service implementations
//!
//! This module contains the individual service implementations for the
//! workspace_daemon proto: SystemService, CollectionService, DocumentService,
//! EmbeddingService, ProjectService, TextSearchService, GraphService

pub mod collection_service;
pub mod document_service;
pub mod embedding_service;
pub mod graph_service;
pub mod project_service;
mod rules_rebuild;
pub mod system_service;
pub mod text_search_service;

// Re-export service implementations
pub use collection_service::CollectionServiceImpl;
pub use document_service::DocumentServiceImpl;
pub use embedding_service::EmbeddingServiceImpl;
pub use graph_service::GraphServiceImpl;
pub use project_service::ProjectServiceImpl;
pub use system_service::SystemServiceImpl;
pub use text_search_service::TextSearchServiceImpl;
