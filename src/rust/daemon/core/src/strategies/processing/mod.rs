//! Per-item-type processing strategies.
//!
//! Each module implements `ProcessingStrategy` for a specific `ItemType`,
//! extracting the processing logic from `unified_queue_processor.rs` into
//! focused, testable units.

pub mod text;
