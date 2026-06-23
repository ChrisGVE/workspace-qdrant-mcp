//! Qdrant membership PUT producer (arch §6.3, AC-F7.1 / AC-F7.2 / AC-F7.3).
//!
//! File: `wqm-storage-write/src/qdrant/membership.rs`
//! Location: `src/rust/storage-write/src/qdrant/` (write-crate qdrant layer)
//! Context: The CALLER side of the unified membership producer. This module
//!   builds and executes the `overwrite_payload` (PUT) for a blob point's
//!   `branch_id[]` payload field. It MUST NOT re-implement the SELECT DISTINCT
//!   query; it delegates to [`crate::blob::membership::compute_membership`] for
//!   the authoritative branch set (FP-2 / DR GP-1 / AC-F7.6).
//!
//! ## Invariants (arch §5.3 / §6.3 / AC-F7.2)
//!
//! Every PUT supplies ALL three payload fields: `tenant_id`, `branch_id`, and
//! `collection_id`. Supplying only `branch_id` would silently delete `tenant_id`
//! from the point's payload, breaking keyword filter searches that require
//! `tenant_id`. The `build_membership_payload` helper enforces this by
//! construction: the returned `BlobPayload` always carries all three fields.
//!
//! ## What is NOT here (AC-F7.3)
//!
//! - No `set_payload` (POST) for `branch_id[]`: it merges keys rather than
//!   replacing them, but it still replaces the field VALUE with the supplied
//!   array, silently dropping all prior memberships on every call.
//! - No `get_points` call: Qdrant is NEVER a source of membership truth
//!   (arch §6.3). The branch set is always derived from SQLite via
//!   `compute_membership`.
//!
//! ## ADD vs REMOVE timing (arch §6.3, AC-F7.1)
//!
//! - **ADD (existing-blob):** the blob_refs INSERT has already committed inside
//!   the ContentKeyLock. Call `compute_membership`, then enqueue the returned
//!   `BlobPayload` for batch flush (the ladder's `CaptureSink` / real sink).
//!   `put_membership` is the synchronous form intended for REMOVE and any
//!   context where live network access is acceptable; the ADD path uses the
//!   enqueue model via `build_membership_payload`.
//! - **REMOVE (F9):** call AFTER the blob_refs DELETE and INSIDE the lock, then
//!   call `put_membership` synchronously (batching REMOVE re-introduces the
//!   F04 race).
//!
//! Neighbors: [`crate::blob::membership`] (the SELECT DISTINCT producer this
//!   module calls), [`crate::blob::ladder`] (the ADD enqueue path),
//!   [`crate::qdrant::write_client::QdrantWriteClient`] (the PUT transport).

use std::collections::HashMap;

use qdrant_client::qdrant::{
    point_id, value::Kind, ListValue, PointId, SetPayloadPointsBuilder, Value as QdrantValue,
};
use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::blob::ladder::BlobPayload;
use crate::qdrant::write_client::QdrantWriteClient;

/// Build the full three-field `BlobPayload` for a point, fetching the branch
/// membership set from SQLite via the canonical single producer.
///
/// This is a pure-payload builder: it calls `compute_membership` and populates
/// all three required payload fields. Callers that need the `BlobPayload` for
/// testing or for the ladder's enqueue path can use this without a live Qdrant
/// client (AC-F7.2 — testability by construction).
///
/// The returned `BlobPayload.branch_id` vector is in SQLite's natural DISTINCT
/// order. Sort before asserting equality in tests.
pub async fn build_membership_payload(
    pool: &SqlitePool,
    blob_id: i64,
    tenant_id: impl Into<String>,
    collection_id: impl Into<String>,
) -> Result<BlobPayload, StorageError> {
    let branch_ids = crate::blob::membership::compute_membership(pool, blob_id).await?;
    Ok(BlobPayload {
        tenant_id: tenant_id.into(),
        branch_id: branch_ids,
        collection_id: collection_id.into(),
    })
}

/// Execute an `overwrite_payload` (PUT) for a blob point against a live Qdrant
/// collection.
///
/// The full three-field payload is built by calling [`build_membership_payload`],
/// which in turn calls [`crate::blob::membership::compute_membership`] (the single
/// SELECT DISTINCT producer). The PUT replaces the entire payload of the target
/// point so all three fields (`tenant_id`, `branch_id[]`, `collection_id`) are
/// always present after the call.
///
/// ## Precondition (arch §6.3)
///
/// The caller MUST hold the `ContentKeyLock` for `blob_id`'s content_key and
/// MUST have already committed the `blob_refs` mutation (INSERT or DELETE) before
/// calling this. The REMOVE path (F9) calls this synchronously inside the lock;
/// the ADD path enqueues via `build_membership_payload` instead.
///
/// ## Errors
///
/// Returns `StorageError::Sqlite` if `compute_membership` fails, or
/// `StorageError::Qdrant` if the Qdrant PUT fails.
pub async fn put_membership(
    client: &QdrantWriteClient,
    pool: &SqlitePool,
    blob_id: i64,
    point_id_str: &str,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
) -> Result<(), StorageError> {
    let payload = build_membership_payload(pool, blob_id, tenant_id, collection_id).await?;
    let qdrant_payload = blob_payload_to_qdrant(&payload);

    let point = PointId {
        point_id_options: Some(point_id::PointIdOptions::Uuid(point_id_str.to_string())),
    };

    let request = SetPayloadPointsBuilder::new(collection_name, qdrant_payload)
        .points_selector(vec![point])
        .wait(true);

    // `From<QdrantError> for StorageError` is implemented in wqm_common::error,
    // so `?` converts automatically without a manual map_err.
    client.overwrite_payload(request).await?;

    Ok(())
}

/// Convert a `BlobPayload` to the `HashMap<String, QdrantValue>` form expected
/// by `SetPayloadPointsBuilder`. All three fields are always included.
///
/// This helper is pub(crate) so it can be used directly in unit tests to assert
/// the payload shape without a live Qdrant client.
pub(crate) fn blob_payload_to_qdrant(payload: &BlobPayload) -> HashMap<String, QdrantValue> {
    let branch_values: Vec<QdrantValue> = payload
        .branch_id
        .iter()
        .map(|b| QdrantValue {
            kind: Some(Kind::StringValue(b.clone())),
        })
        .collect();

    let mut map = HashMap::new();
    map.insert(
        "tenant_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue(payload.tenant_id.clone())),
        },
    );
    map.insert(
        "branch_id".to_string(),
        QdrantValue {
            kind: Some(Kind::ListValue(ListValue {
                values: branch_values,
            })),
        },
    );
    map.insert(
        "collection_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue(payload.collection_id.clone())),
        },
    );
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::test_support::{add_branch, fixture, TENANT};
    use sqlx::SqlitePool;

    const BRANCH_A: &str = "branch-a";
    const BRANCH_B: &str = "branch-b";
    const COLLECTION: &str = "projects";

    async fn insert_blob_and_ref(pool: &SqlitePool, branch: &str, path: &str) -> (i64, i64) {
        let blob_result = sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES ('ck-qdrant-mem','hash1','pt-qdrant-mem',?,'hello',X'',X'','2024-01-01')",
        )
        .bind(TENANT)
        .execute(pool)
        .await
        .expect("blob insert");
        let blob_id = blob_result.last_insert_rowid();

        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(branch)
        .bind(path)
        .execute(pool)
        .await
        .expect("file insert");
        let file_id: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?",
        )
        .bind(branch)
        .bind(path)
        .fetch_one(pool)
        .await
        .expect("file_id");

        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES (?, ?, 0, ?) ON CONFLICT DO NOTHING",
        )
        .bind(branch)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("blob_ref insert");

        (blob_id, file_id)
    }

    // AC-F7.2: build_membership_payload always populates all three required
    // payload fields -- tenant_id, branch_id (non-empty), collection_id.
    // Omitting tenant_id would break tenant-scoped searches.
    #[tokio::test]
    async fn build_membership_payload_populates_all_three_fields() {
        let fx = fixture(BRANCH_A).await;
        let (blob_id, _) = insert_blob_and_ref(&fx.pool, BRANCH_A, "a.rs").await;

        let payload = build_membership_payload(&fx.pool, blob_id, TENANT, COLLECTION)
            .await
            .expect("build_membership_payload");

        assert!(
            !payload.tenant_id.is_empty(),
            "tenant_id must be present (AC-F7.2)"
        );
        assert!(
            !payload.collection_id.is_empty(),
            "collection_id must be present (AC-F7.2)"
        );
        assert_eq!(
            payload.branch_id,
            vec![BRANCH_A.to_string()],
            "branch_id must reflect SQLite blob_refs (AC-F7.2)"
        );
    }

    // AC-F7.4: build_membership_payload is idempotent -- two calls on the same
    // SQLite state yield identical sorted branch sets (B1).
    #[tokio::test]
    async fn build_membership_payload_is_idempotent() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let (blob_id, _) = insert_blob_and_ref(&fx.pool, BRANCH_A, "a.rs").await;

        // Add a second referrer from BRANCH_B.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, 'a.rs', 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(BRANCH_B)
        .execute(&fx.pool)
        .await
        .expect("file b insert");
        let file_b: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = 'a.rs'",
        )
        .bind(BRANCH_B)
        .fetch_one(&fx.pool)
        .await
        .expect("file_b id");
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES (?, ?, 0, ?) ON CONFLICT DO NOTHING",
        )
        .bind(BRANCH_B)
        .bind(file_b)
        .bind(blob_id)
        .execute(&fx.pool)
        .await
        .expect("ref b");

        let mut first = build_membership_payload(&fx.pool, blob_id, TENANT, COLLECTION)
            .await
            .expect("first call");
        first.branch_id.sort();

        let mut second = build_membership_payload(&fx.pool, blob_id, TENANT, COLLECTION)
            .await
            .expect("second call");
        second.branch_id.sort();

        assert_eq!(
            first, second,
            "idempotent: two calls on same state are equal (AC-F7.4)"
        );
    }

    // AC-F7.2 (payload shape): blob_payload_to_qdrant emits all three fields
    // as correctly-typed Qdrant values.
    #[test]
    fn blob_payload_to_qdrant_includes_all_three_fields() {
        let payload = BlobPayload {
            tenant_id: "tenant-x".to_string(),
            branch_id: vec!["branch-a".to_string(), "branch-b".to_string()],
            collection_id: "projects".to_string(),
        };

        let map = blob_payload_to_qdrant(&payload);

        assert!(map.contains_key("tenant_id"), "tenant_id must be present");
        assert!(map.contains_key("branch_id"), "branch_id must be present");
        assert!(
            map.contains_key("collection_id"),
            "collection_id must be present"
        );

        // tenant_id must be a StringValue.
        match &map["tenant_id"].kind {
            Some(Kind::StringValue(s)) => assert_eq!(s, "tenant-x"),
            other => panic!("tenant_id should be StringValue, got {:?}", other),
        }

        // branch_id must be a ListValue of StringValues.
        match &map["branch_id"].kind {
            Some(Kind::ListValue(list)) => {
                assert_eq!(list.values.len(), 2);
                let strings: Vec<String> = list
                    .values
                    .iter()
                    .filter_map(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    strings,
                    vec!["branch-a".to_string(), "branch-b".to_string()]
                );
            }
            other => panic!("branch_id should be ListValue, got {:?}", other),
        }

        // collection_id must be a StringValue.
        match &map["collection_id"].kind {
            Some(Kind::StringValue(s)) => assert_eq!(s, "projects"),
            other => panic!("collection_id should be StringValue, got {:?}", other),
        }
    }

    // AC-F7.6: qdrant/membership.rs does NOT contain the SELECT DISTINCT producer
    // query (it delegates to blob::membership::compute_membership instead).
    #[test]
    fn qdrant_membership_does_not_reimplement_select_distinct() {
        use std::path::Path;

        let qdrant_mem_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qdrant/membership.rs");
        let src = std::fs::read_to_string(&qdrant_mem_path).expect("read qdrant/membership.rs");

        let live_query_needle =
            "sqlx::query(\"SELECT DISTINCT branch_id FROM blob_refs WHERE blob_id";

        for line in src.lines() {
            assert!(
                !line.contains(live_query_needle),
                "qdrant/membership.rs must NOT contain the SELECT DISTINCT query; \
                 it must delegate to blob::membership::compute_membership. \
                 Found on line: {:?}",
                line
            );
        }
    }

    // AC-F7.3: qdrant/membership.rs must not call set_payload for branch_id
    // and must not call get_points anywhere.
    #[test]
    fn qdrant_membership_no_set_payload_no_get_points() {
        use std::path::Path;

        let qdrant_mem_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qdrant/membership.rs");
        let src = std::fs::read_to_string(&qdrant_mem_path).expect("read qdrant/membership.rs");

        // Only check production code: stop at the #[cfg(test)] boundary so the
        // test module's own assert-lines cannot self-trip the guard.
        let production_src: String = src
            .lines()
            .take_while(|l| !l.trim().starts_with("#[cfg(test)]"))
            .collect::<Vec<_>>()
            .join("\n");

        let set_payload_call = [".set_pay", "load("].concat();
        let overwrite_call = ".overwrite_payload(";
        for line in production_src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            if trimmed.contains(&set_payload_call) && !trimmed.contains(overwrite_call) {
                panic!(
                    "qdrant/membership.rs production code must not call .set_payload() \
                     (AC-F7.3). Found: {:?}",
                    line
                );
            }
        }

        let get_points_call = [".get_po", "ints("].concat();
        for line in production_src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            if trimmed.contains(&get_points_call) {
                panic!(
                    "qdrant/membership.rs production code must not call .get_points() \
                     (AC-F7.3): {:?}",
                    line
                );
            }
        }
    }
}
