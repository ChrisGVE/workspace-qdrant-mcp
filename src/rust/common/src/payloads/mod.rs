//! Queue payload structs shared between daemon and CLI
//!
//! These structs represent the JSON payloads for different queue item types.

mod content;
mod filesystem;
mod library;
mod operations;
mod web;

pub use content::{ContentPayload, MemoryPayload, ScratchpadPayload};
pub use filesystem::{FilePayload, FolderPayload};
pub use library::{ChunkingConfigPayload, LibraryDocumentPayload, LibraryPayload, ProjectPayload};
pub use operations::{CollectionPayload, DeleteDocumentPayload, DeleteTenantPayload};
pub use web::{UrlPayload, WebsitePayload};
