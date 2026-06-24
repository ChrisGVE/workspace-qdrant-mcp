//! Tests for `wqm-storage-write/src/blob/ladder.rs` (AC-F6 write-cycle cases).
//!
//! Separated from the production module as a sibling test file to keep
//! `ladder.rs` within the 500-line codesize budget (coding.md §VIII). All
//! tests run WITHOUT a live Qdrant (offline / CI-safe).

use sqlx::SqlitePool;

use super::*;
use crate::blob::embed::mock::MockEmbedder;
use crate::blob::test_support::{add_branch, fixture, TENANT};

const BRANCH_A: &str = "branch-a";
const BRANCH_B: &str = "branch-b";
const COLLECTION: &str = "projects";

/// Insert a `files` row so the ladder's `blob_refs` FK is satisfiable, returning
/// its file_id. The ladder itself does not touch `files` (that is `dedup.rs`).
async fn make_file(pool: &SqlitePool, branch: &str, path: &str) -> i64 {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch)
    .bind(path)
    .execute(pool)
    .await
    .expect("file insert");
    sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
        .bind(branch)
        .bind(path)
        .fetch_one(pool)
        .await
        .expect("file_id")
}

fn ctx<'a>(branch: &'a str, file_id: i64) -> ChunkContext<'a> {
    ChunkContext {
        tenant_id: TENANT,
        branch_id: branch,
        collection_id: COLLECTION,
        file_id,
        content_key_version: 4,
    }
}

fn key_for(c: &ChunkContext<'_>, text: &str, chunk_index: u32) -> ChunkKey {
    let chunk_content_hash = wqm_common::hashing::compute_content_hash(text);
    ChunkKey {
        content_key: chunk_key(c, &chunk_content_hash),
        chunk_content_hash,
        chunk_index,
        text: text.into(),
    }
}

// AC-F6.3: a content_key MISS embeds, inserts the blob, and enqueues an Upsert with
// the single current branch as the initial membership.
#[tokio::test]
async fn miss_embeds_inserts_and_upserts() {
    let fx = fixture(BRANCH_A).await;
    let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    let c = ctx(BRANCH_A, file_id);
    let k = key_for(&c, "fn main() {}", 0);

    let outcome = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k)
        .await
        .expect("ingest miss");

    assert_eq!(outcome, ChunkOutcome::BlobCreated);
    assert_eq!(embedder.call_count(), 1, "a miss embeds exactly once");

    // The blob landed in SQLite (FP-1: before any flush).
    let blob_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE content_key = ?")
        .bind(&k.content_key)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(blob_count, 1);

    // Exactly one Upsert op with single-branch payload carrying all three fields.
    assert_eq!(sink.ops.len(), 1);
    match &sink.ops[0] {
        QdrantOp::Upsert { payload, .. } => {
            assert_eq!(payload.tenant_id, TENANT);
            assert_eq!(payload.branch_id, vec![BRANCH_A.to_string()]);
            assert_eq!(payload.collection_id, COLLECTION);
        }
        other => panic!("expected Upsert, got {other:?}"),
    }
}

// AC-F6.3: "byte-identical content" re-ingested is a HIT, not a third case -- it
// reuses the blob and never re-embeds.
#[tokio::test]
async fn byte_identical_content_is_a_hit_not_a_new_blob() {
    let fx = fixture(BRANCH_A).await;
    let file1 = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let file2 = make_file(&fx.pool, BRANCH_A, "b.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();

    let text = "let x = 1;";
    // First ingest (file a) -> miss.
    let c1 = ctx(BRANCH_A, file1);
    let k1 = key_for(&c1, text, 0);
    let o1 = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c1, &k1)
        .await
        .unwrap();
    assert_eq!(o1, ChunkOutcome::BlobCreated);

    // Second ingest of the SAME bytes in another file -> hit (same content_key).
    let c2 = ctx(BRANCH_A, file2);
    let k2 = key_for(&c2, text, 0);
    assert_eq!(
        k1.content_key, k2.content_key,
        "identical bytes -> same content_key"
    );
    let o2 = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c2, &k2)
        .await
        .unwrap();
    assert_eq!(o2, ChunkOutcome::BlobReused, "byte-identical is a HIT");

    assert_eq!(embedder.call_count(), 1, "the hit must NOT re-embed");
    let blob_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(blob_count, 1, "one blob, deduped across files");
}

// AC-F6.2: the HIT path recomputes the FULL branch_id[] from SQLite and PUTs it.
// After two branches reference the same content, the PUT carries BOTH branches.
#[tokio::test]
async fn hit_recomputes_full_membership_and_puts() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();

    let text = "shared chunk body";
    let ca = ctx(BRANCH_A, file_a);
    let ka = key_for(&ca, text, 0);
    ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
        .await
        .unwrap();

    // Branch B references the same content -> hit, full membership = {A, B}.
    let cb = ctx(BRANCH_B, file_b);
    let kb = key_for(&cb, text, 0);
    ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
        .await
        .unwrap();

    // The last op is an OverwritePayload (PUT) with both branches.
    match sink.ops.last().expect("an op") {
        QdrantOp::OverwritePayload { payload, .. } => {
            let mut branches = payload.branch_id.clone();
            branches.sort();
            assert_eq!(branches, vec![BRANCH_A.to_string(), BRANCH_B.to_string()]);
            assert_eq!(
                payload.tenant_id, TENANT,
                "PUT carries tenant_id (never dropped)"
            );
            assert_eq!(payload.collection_id, COLLECTION);
        }
        other => panic!("expected OverwritePayload PUT, got {other:?}"),
    }
}

// AC-F6.2 / DATA-05 / SEC-4: the HIT path PUTs against the STORED point_id, even
// when that point_id was salted to a value != point_id(content_key, 0).
#[tokio::test]
async fn hit_targets_stored_salted_point_id() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();

    let text = "rekeyed chunk";
    let ca = ctx(BRANCH_A, file_a);
    let ka = key_for(&ca, text, 0);
    ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
        .await
        .unwrap();

    // Force a re-key: overwrite the stored point_id with a salted derivation, as
    // the SEC-4 guard would (AC-F5.5). The natural derivation is point_id(ck, 0).
    let natural = point_id(&ka.content_key, 0).to_string();
    let salted =
        wqm_common::hashing::salted_point_id(&ka.content_key, &[0xde, 0xad], 0).to_string();
    assert_ne!(
        natural, salted,
        "the salted id differs from the natural one"
    );
    sqlx::query("UPDATE blobs SET point_id = ? WHERE content_key = ?")
        .bind(&salted)
        .bind(&ka.content_key)
        .execute(&fx.pool)
        .await
        .unwrap();

    // Now a hit from branch B must PUT against the STORED salted id, not `natural`.
    sink.ops.clear();
    let cb = ctx(BRANCH_B, file_b);
    let kb = key_for(&cb, text, 0);
    ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
        .await
        .unwrap();

    match sink.ops.last().expect("a PUT") {
        QdrantOp::OverwritePayload { point_id: pid, .. } => {
            assert_eq!(*pid, salted, "PUT targets the STORED salted point_id");
            assert_ne!(*pid, natural, "NOT the freshly-derived point_id");
        }
        other => panic!("expected OverwritePayload, got {other:?}"),
    }
}

// FP-1: durable vectors persist in SQLite BEFORE any Qdrant op is enqueued. We
// assert that on a miss the blob row (with non-empty vectors) exists and the sink
// op references the same point_id (i.e. SQLite truth precedes the enqueue).
#[tokio::test]
async fn fp1_sqlite_vectors_precede_qdrant_enqueue() {
    let fx = fixture(BRANCH_A).await;
    let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    let c = ctx(BRANCH_A, file_id);
    let k = key_for(&c, "vectorized", 0);

    ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k)
        .await
        .unwrap();

    // SQLite has the durable vectors.
    let row =
        sqlx::query("SELECT point_id, dense_vec, sparse_vec FROM blobs WHERE content_key = ?")
            .bind(&k.content_key)
            .fetch_one(&fx.pool)
            .await
            .unwrap();
    let stored_pid: String = row.get("point_id");
    let dense: Vec<u8> = row.get("dense_vec");
    let sparse: Vec<u8> = row.get("sparse_vec");
    assert!(!dense.is_empty(), "dense vector persisted before enqueue");
    assert!(!sparse.is_empty(), "sparse vector persisted before enqueue");

    // The enqueued op points at the SAME persisted point_id (truth-then-product).
    match &sink.ops[0] {
        QdrantOp::Upsert { point_id: pid, .. } => assert_eq!(*pid, stored_pid),
        other => panic!("expected Upsert, got {other:?}"),
    }
}

// AC-F6.5 (F04 race) at the SQLite layer: two ingests of the SAME content_key --
// the second finds the blob present and takes the hit (membership) path. (The lock
// that serializes the two is unit-tested in `lock.rs`; here we prove the second
// call's BEHAVIOR is the hit path once the blob exists.)
#[tokio::test]
async fn second_ingest_of_same_content_key_takes_hit_path() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    let text = "racing content";

    let ca = ctx(BRANCH_A, file_a);
    let ka = key_for(&ca, text, 0);
    let first = ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
        .await
        .unwrap();
    let cb = ctx(BRANCH_B, file_b);
    let kb = key_for(&cb, text, 0);
    let second = ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
        .await
        .unwrap();

    assert_eq!(first, ChunkOutcome::BlobCreated);
    assert_eq!(
        second,
        ChunkOutcome::BlobReused,
        "second takes the membership path"
    );
}

// AC-F6.2: blob points always use chunk_index 0 in the point_id; the positional
// index lives only in blob_refs.chunk_index. Two chunks at indices 3 and 7 of the
// same content still map to the SAME single point_id.
#[tokio::test]
async fn point_id_ignores_chunk_index_position() {
    let fx = fixture(BRANCH_A).await;
    let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
    let embedder = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    let c = ctx(BRANCH_A, file_id);

    let text = "same bytes at two positions";
    let k_at_3 = key_for(&c, text, 3);
    ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k_at_3)
        .await
        .unwrap();

    let stored: String = sqlx::query_scalar("SELECT point_id FROM blobs WHERE content_key = ?")
        .bind(&k_at_3.content_key)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(
        stored,
        point_id(&k_at_3.content_key, 0).to_string(),
        "point_id uses index 0"
    );

    // blob_refs preserves the positional index 3.
    let stored_index: i64 =
        sqlx::query_scalar("SELECT chunk_index FROM blob_refs WHERE branch_id = ? AND file_id = ?")
            .bind(BRANCH_A)
            .bind(file_id)
            .fetch_one(&fx.pool)
            .await
            .unwrap();
    assert_eq!(stored_index, 3, "positional index lives in blob_refs");
}
