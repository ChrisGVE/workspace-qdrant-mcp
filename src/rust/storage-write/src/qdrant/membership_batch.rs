//! Batched-outside-lock survivor-membership PUT fallback (AC-F19.3).
//!
//! File: `wqm-storage-write/src/qdrant/membership_batch.rs`
//! Location: `src/rust/storage-write/src/qdrant/` (write-crate qdrant layer)
//! Context: The pessimistic-PUT fallback selected when F19's benchmark shows that
//!   a synchronous per-content_key `overwrite_payload` (PUT) cannot meet the F9
//!   deletion SLA of 5 s (AC-F19.2). Strategy selection is recorded in the
//!   committed benchmark report `docs/benchmarks/F19-put-vs-upsert.md`.
//!
//! ## Design invariant (AC-F19.3)
//!
//! The membership RECOMPUTE (`compute_membership` / `build_membership_payload`) stays
//! INSIDE the per-content_key lock -- that is where SQLite truth is committed and
//! the full branch set is authoritative. Only the Qdrant PUT is moved outside the
//! lock, into a batch flush.
//!
//! This preserves the F04 race-freedom property: the SQLite source of truth (the
//! `blob_refs` DELETE and the `compute_membership` result) is fully committed before
//! the batched PUT fires. Qdrant is eventually consistent with SQLite truth, but the
//! CONTENT of the payload (which branches survive) is determined atomically under the
//! lock before the batch is enqueued.
//!
//! ## Idempotency (AC-F19.3)
//!
//! Building the batch twice from the same SQLite state produces an identical
//! `(point_id, BlobPayload)` set -- the payload is derived purely from SQLite truth
//! via `compute_membership`. Re-running the batch flush against Qdrant yields the
//! same Qdrant state.
//!
//! ## What is NOT here
//!
//! - No `get_points` call: Qdrant is never a membership source of truth (arch §6.3).
//! - No `set_payload` (POST): it has no append mode and drops prior memberships.
//!
//! Neighbors: [`crate::qdrant::membership`] (the synchronous sibling),
//!   [`crate::blob::membership`] (the canonical SELECT DISTINCT producer),
//!   [`crate::qdrant::write_client::QdrantWriteClient`] (the PUT transport).

use qdrant_client::qdrant::{point_id, PointId, SetPayloadPointsBuilder};
use wqm_common::error::StorageError;

use crate::blob::ladder::BlobPayload;
use crate::qdrant::membership::blob_payload_to_qdrant;
use crate::qdrant::write_client::QdrantWriteClient;

/// One pending membership PUT: the Qdrant point UUID and the full three-field
/// payload computed from SQLite truth under the per-content_key lock.
///
/// Both fields are immutable once enqueued -- the payload was derived inside the
/// lock and does not change when the batch fires outside it.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingMembershipPut {
    /// The Qdrant point UUID (read from `blobs.point_id`, never recomputed).
    pub point_id: String,
    /// The full three-field payload (`tenant_id`, `branch_id[]`, `collection_id`)
    /// computed by `build_membership_payload` inside the lock.
    pub payload: BlobPayload,
}

/// Accumulator for survivor-membership PUTs that are flushed OUTSIDE the lock.
///
/// ## Usage pattern (AC-F19.3)
///
/// Inside the per-content_key lock:
/// 1. Commit the `blob_refs` DELETE to SQLite.
/// 2. Call `build_membership_payload` to get the surviving branch set.
/// 3. Call `MembershipPutBatch::push` to enqueue the (point_id, payload) tuple.
/// 4. Release the lock.
///
/// Outside all locks:
/// 5. Call `MembershipPutBatch::flush` to issue all `overwrite_payload` PUTs.
///
/// The batch issues one `overwrite_payload` call per point. Qdrant's
/// `SetPayloadPoints` does not support per-point distinct payloads in a single
/// batch call, so sequential per-point calls are the only correct path.
#[derive(Debug, Default)]
pub struct MembershipPutBatch {
    entries: Vec<(String, PendingMembershipPut)>,
}

impl MembershipPutBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue one pending PUT (called INSIDE the lock after the SQLite DELETE
    /// has committed and `build_membership_payload` has been called).
    ///
    /// `collection_name` is the Qdrant collection to target; it is stored alongside
    /// the payload so the batch flush does not need external context.
    pub fn push(&mut self, collection_name: impl Into<String>, pending: PendingMembershipPut) {
        self.entries.push((collection_name.into(), pending));
    }

    /// Return the number of enqueued PUTs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no PUTs have been enqueued.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return an iterator over all enqueued `(collection_name, PendingMembershipPut)` pairs.
    ///
    /// Used in tests to assert the exact set produced by a given SQLite state without
    /// requiring a live Qdrant client (AC-F19.3 idempotency test).
    pub fn iter(&self) -> impl Iterator<Item = &(String, PendingMembershipPut)> {
        self.entries.iter()
    }

    /// Consume the batch and flush all enqueued PUTs to Qdrant.
    ///
    /// Called OUTSIDE all per-content_key locks. Each PUT issues one
    /// `overwrite_payload` call; errors are collected and the first error is
    /// returned after attempting all entries (best-effort flush).
    ///
    /// ## Idempotency
    ///
    /// Re-running this method (if the caller retries) is safe: `overwrite_payload`
    /// replaces the full payload, so a duplicate flush produces the same Qdrant state.
    pub async fn flush(self, client: &QdrantWriteClient) -> Result<(), StorageError> {
        let mut first_err: Option<StorageError> = None;

        for (collection_name, pending) in self.entries {
            let qdrant_payload = blob_payload_to_qdrant(&pending.payload);

            let point = PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(pending.point_id.clone())),
            };

            let request = SetPayloadPointsBuilder::new(&collection_name, qdrant_payload)
                .points_selector(vec![point])
                .wait(true);

            if let Err(e) = client.overwrite_payload(request).await {
                if first_err.is_none() {
                    first_err = Some(StorageError::from(e));
                }
            }
        }

        match first_err {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use qdrant_client::qdrant::{value::Kind, ListValue, Value as QdrantValue};

    use super::*;
    use crate::blob::ladder::BlobPayload;

    // Helper: build a BlobPayload with the given branch list.
    fn make_payload(branches: &[&str]) -> BlobPayload {
        BlobPayload {
            tenant_id: "tenant-x".to_string(),
            branch_id: branches.iter().map(|b| b.to_string()).collect(),
            collection_id: "projects".to_string(),
        }
    }

    // Helper: build a PendingMembershipPut.
    fn make_pending(point_id: &str, branches: &[&str]) -> PendingMembershipPut {
        PendingMembershipPut {
            point_id: point_id.to_string(),
            payload: make_payload(branches),
        }
    }

    // AC-F19.3 (idempotency): building the batch twice from identical inputs
    // yields identical (point_id, payload) sets. This tests the accumulator +
    // payload logic without a live Qdrant client, mirroring F7's
    // build_membership_payload idempotency test.
    #[test]
    fn batch_built_twice_from_same_state_is_identical() {
        let pending_a = make_pending(
            "550e8400-e29b-41d4-a716-446655440000",
            &["branch-a", "branch-c"],
        );
        let pending_b = make_pending(
            "550e8400-e29b-41d4-a716-446655440001",
            &["branch-a", "branch-b"],
        );

        // First construction.
        let mut batch1 = MembershipPutBatch::new();
        batch1.push("projects", pending_a.clone());
        batch1.push("projects", pending_b.clone());

        // Second construction (same inputs -- simulates idempotent re-build from
        // the same SQLite state).
        let mut batch2 = MembershipPutBatch::new();
        batch2.push("projects", pending_a.clone());
        batch2.push("projects", pending_b.clone());

        // Both batches must produce the same ordered (collection, pending) sequence.
        let pairs1: Vec<_> = batch1.iter().cloned().collect();
        let pairs2: Vec<_> = batch2.iter().cloned().collect();

        assert_eq!(
            pairs1, pairs2,
            "AC-F19.3: two batches from same state must be equal"
        );
    }

    // AC-F19.3 (accumulator semantics): len/is_empty track correctly.
    #[test]
    fn batch_len_and_is_empty() {
        let mut batch = MembershipPutBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        batch.push("projects", make_pending("pt-1", &["b1"]));
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);

        batch.push("projects", make_pending("pt-2", &["b2"]));
        assert_eq!(batch.len(), 2);
    }

    // AC-F19.3 (payload content): the payload that enters the batch is exactly
    // the BlobPayload that was computed inside the lock (no mutation on push).
    #[test]
    fn push_preserves_payload_unchanged() {
        let payload = make_payload(&["branch-a", "branch-b"]);
        let pending = PendingMembershipPut {
            point_id: "test-uuid".to_string(),
            payload: payload.clone(),
        };

        let mut batch = MembershipPutBatch::new();
        batch.push("projects", pending);

        let stored = &batch.iter().next().expect("entry").1;
        assert_eq!(
            stored.payload, payload,
            "payload must be preserved verbatim"
        );
    }

    // Verify blob_payload_to_qdrant (re-used from membership.rs) converts all
    // three fields -- cross-module integration check.
    #[test]
    fn qdrant_payload_map_has_all_three_fields() {
        let payload = make_payload(&["b1", "b2"]);
        let map: HashMap<String, QdrantValue> = blob_payload_to_qdrant(&payload);

        assert!(map.contains_key("tenant_id"));
        assert!(map.contains_key("branch_id"));
        assert!(map.contains_key("collection_id"));

        // branch_id must be a list of two entries.
        if let Some(Kind::ListValue(ListValue { values })) =
            map.get("branch_id").and_then(|v| v.kind.as_ref())
        {
            assert_eq!(values.len(), 2);
        } else {
            panic!("branch_id must be a ListValue");
        }
    }
}
