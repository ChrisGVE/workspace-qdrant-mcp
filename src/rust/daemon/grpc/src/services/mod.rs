//! Modular gRPC service implementations
//!
//! This module contains the individual service implementations for the
//! workspace_daemon proto: 7 core services + 5 write services.

// Core services
pub mod collection_service;
pub mod document_service;
pub mod embedding_service;
pub mod graph_service;
pub mod language_service;
pub mod project_service;
mod rules_payload_backfill;
mod rules_rebuild;
mod scratchpad_rebuild;
pub mod system_service;
pub mod text_search_service;

// Re-embed pipeline (used by AdminWriteService.TriggerReembed)
pub mod reembed;

// Write services (daemon-exclusive SQLite mutations)
pub mod admin_write_service;
pub mod library_write_service;
pub mod queue_write_service;
pub mod tracking_write_service;
pub mod watch_write_service;

// Re-export core service implementations
pub use collection_service::CollectionServiceImpl;
pub use document_service::DocumentServiceImpl;
pub use embedding_service::EmbeddingServiceImpl;
pub use graph_service::GraphServiceImpl;
pub use language_service::LanguageServiceImpl;
pub use project_service::ProjectServiceImpl;
pub use system_service::SystemServiceImpl;
pub use text_search_service::TextSearchServiceImpl;

// Re-export write service implementations
pub use admin_write_service::AdminWriteServiceImpl;
pub use library_write_service::LibraryWriteServiceImpl;
pub use queue_write_service::QueueWriteServiceImpl;
pub use tracking_write_service::TrackingWriteServiceImpl;
pub use watch_write_service::WatchWriteServiceImpl;
