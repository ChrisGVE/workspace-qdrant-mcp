//! Full re-embed pipeline executed under `AdminWriteService.TriggerReembed`.
//!
//! Implements PRD §6.6 (embedding-providers): pause → drain → flush →
//! recreate canonical Qdrant collections at the configured `output_dim` →
//! enqueue new ingestion items → resume queue. The handler stays
//! enqueue-only on the queue side: the `'reembed'` collection items are
//! recorded in `unified_queue` for traceability, and the actual file/rule
//! /scratchpad re-ingestion goes through normal `add` items so existing
//! queue strategies pick them up.

mod context;
mod enqueue;
mod pipeline;
mod recreator;

pub use context::{ReembedContext, CANONICAL_COLLECTIONS};
pub use pipeline::execute_reembed;
pub use recreator::{CollectionRecreator, StorageClientRecreator};

#[cfg(test)]
mod tests;
