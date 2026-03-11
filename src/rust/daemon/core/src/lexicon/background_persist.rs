//! Background persistence task for LexiconManager.
//!
//! Decouples lexicon SQLite writes from the document-processing hot path.
//! The `add_document()` path sends a `PersistRequest::Persist` message instead
//! of blocking on SQLite I/O. A single tokio task drains the channel and
//! calls `persist()` for each collection, deduplicating queued requests.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use super::manager::LexiconManager;

/// Message sent to the background persistence task.
pub(super) enum PersistRequest {
    /// Persist the specified collection at the next opportunity.
    Persist { collection: String },
    /// Flush all pending work and reply when done (used for graceful shutdown).
    Flush { reply: oneshot::Sender<()> },
}

/// Spawn the background persistence loop for the given `LexiconManager`.
///
/// Returns the sender half of the channel. The receiver is consumed by the
/// spawned task. The task exits when the sender is dropped (channel closed).
pub(super) fn spawn_background_persister(
    manager: Arc<LexiconManager>,
) -> mpsc::Sender<PersistRequest> {
    let (tx, rx) = mpsc::channel::<PersistRequest>(64);
    tokio::spawn(background_persist_loop(manager, rx));
    tx
}

/// Background task: drain PersistRequests and call persist() for each collection.
async fn background_persist_loop(
    manager: Arc<LexiconManager>,
    mut rx: mpsc::Receiver<PersistRequest>,
) {
    let mut pending: HashSet<String> = HashSet::new();

    while let Some(request) = rx.recv().await {
        match request {
            PersistRequest::Persist { collection } => {
                // Drain all immediately available messages to batch collections
                pending.insert(collection);
                while let Ok(extra) = rx.try_recv() {
                    match extra {
                        PersistRequest::Persist { collection: c } => {
                            pending.insert(c);
                        }
                        PersistRequest::Flush { reply } => {
                            // Flush all pending before replying
                            flush_pending(&manager, &mut pending).await;
                            let _ = reply.send(());
                        }
                    }
                }
                // Persist all batched collections
                flush_pending(&manager, &mut pending).await;
            }
            PersistRequest::Flush { reply } => {
                flush_pending(&manager, &mut pending).await;
                let _ = reply.send(());
            }
        }
    }
    debug!("Background persist loop exiting (channel closed)");
}

/// Persist all collections in `pending` and clear the set.
async fn flush_pending(manager: &Arc<LexiconManager>, pending: &mut HashSet<String>) {
    for collection in pending.drain() {
        if let Err(e) = manager.persist(&collection).await {
            warn!("Background persist failed for '{}': {}", collection, e);
        }
    }
}
