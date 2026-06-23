//! `Embedder` trait — the lazy embed seam for the dedup ladder (arch §4.1).
//!
//! File: `wqm-storage-write/src/blob/embed.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: The dedup ladder ([`crate::blob::ladder`]) embeds a chunk's raw text
//!   ONLY on a `content_key` MISS — a genuinely new blob. A hit reuses the stored
//!   vectors and never re-embeds. This trait is the seam between the ladder and the
//!   concrete embedding engine (fastembed/ONNX), which lives in the daemon, NOT in
//!   this crate. The write crate keeps NO embedder dependency (fastembed/ort/onnx are
//!   absent from its Cargo.toml) so the read/write split (Guard-3) stays intact: the
//!   ladder holds an injected `Arc<dyn Embedder>` and invokes it lazily on the miss
//!   branch only.
//!
//!   WHY here and not in `wqm-storage` (read crate): embedding a chunk is a WRITE-path
//!   operation (it produces durable blob vectors persisted before any Qdrant enqueue).
//!   The read-side search path does not need an injected ingest-embedder — `SearchQuery`
//!   already carries optional precomputed query vectors. Defining the trait in the
//!   write crate keeps the embed seam out of every read-only binary by construction.
//!
//! Neighbors: [`crate::blob::ladder`] (the sole caller — invokes `embed` on a miss),
//!   [`crate::blob::dedup`] (orchestrates the ingest, holds the `Arc<dyn Embedder>`).

use std::collections::HashMap;
use wqm_common::error::StorageError;

/// The dense + sparse vectors produced for one chunk of raw text.
///
/// `dense` is the dense embedding (the model's fixed-width float vector). `sparse`
/// maps a term id to its weight (the sparse/lexical embedding). Both are persisted
/// verbatim into `blobs.dense_vec` / `blobs.sparse_vec` and projected to Qdrant; one
/// embed call per new blob (never re-embedded on a hit).
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedChunk {
    /// Dense embedding vector for the chunk.
    pub dense: Vec<f32>,
    /// Sparse embedding: term id -> weight.
    pub sparse: HashMap<u32, f32>,
}

/// Produces dense + sparse vectors from raw chunk text.
///
/// Implemented in the daemon by the concrete embedding engine and injected into the
/// write facade as `Arc<dyn Embedder>`. The trait is `async` (via `#[async_trait]`)
/// so the implementation may call out to a model runtime; it is `Send + Sync` so the
/// `Arc<dyn Embedder>` can be shared across the daemon's worker tasks.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Embed one chunk's raw text into its dense + sparse vectors.
    ///
    /// Invoked LAZILY — only on a `content_key` miss (a new blob). A hit never calls
    /// this. An embedding failure surfaces as a [`StorageError`] and aborts the chunk's
    /// write (no partial blob is persisted).
    async fn embed(&self, text: &str) -> Result<EmbeddedChunk, StorageError>;
}

#[cfg(test)]
pub(crate) mod mock {
    //! A deterministic mock embedder for ladder unit tests.
    //!
    //! It never touches a model runtime: it derives reproducible vectors from the
    //! text's bytes and counts its calls, so a test can assert exactly how many times
    //! the ladder embedded (a hit must add zero calls).

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Deterministic fake embedder: same text -> same vectors, with a call counter.
    pub struct MockEmbedder {
        calls: Arc<AtomicUsize>,
    }

    impl MockEmbedder {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Arc::new(AtomicUsize::new(0)),
            })
        }

        /// How many times `embed` has been invoked — a hit must NOT increment this.
        pub fn call_count(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, text: &str) -> Result<EmbeddedChunk, StorageError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            // Deterministic, content-derived vectors: a tiny dense vector seeded from
            // the byte sum, and one sparse term per distinct byte value.
            let byte_sum: u32 = text.bytes().map(u32::from).sum();
            let dense = vec![
                byte_sum as f32,
                text.len() as f32,
                text.bytes().next().unwrap_or(0) as f32,
            ];
            let mut sparse = HashMap::new();
            for b in text.bytes() {
                *sparse.entry(u32::from(b)).or_insert(0.0) += 1.0;
            }
            Ok(EmbeddedChunk { dense, sparse })
        }
    }
}
