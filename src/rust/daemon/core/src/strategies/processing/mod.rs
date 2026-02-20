//! Per-item-type processing strategies.
//!
//! Each module implements `ProcessingStrategy` for a specific `ItemType`,
//! extracting the processing logic from `unified_queue_processor.rs` into
//! focused, testable units.

pub mod collection;
pub mod file;
pub mod folder;
pub mod tenant;
pub mod text;
pub mod url;
pub mod website;
