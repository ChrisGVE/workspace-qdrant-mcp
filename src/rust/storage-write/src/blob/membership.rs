//! Unified branch membership producer (arch §6.3, AC-F7.1 / AC-F7.6).
//!
//! File: `wqm-storage-write/src/blob/membership.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: The SINGLE canonical producer of a blob's branch membership set.
//!   The query `SELECT DISTINCT branch_id FROM blob_refs WHERE blob_id = ?`
//!   appears EXACTLY ONCE in the write crate, in [`compute_membership`] below
//!   (FP-2 / DR GP-1 / AC-F7.6). All call sites that need the full membership
//!   set MUST delegate here; they MUST NOT re-implement the SELECT DISTINCT query.
//!
//! ## Call-site contract (arch §6.3 ADD/REMOVE timing)
//!
//! - **ADD (existing-blob):** call AFTER the blob_refs INSERT and INSIDE the
//!   ContentKeyLock, then enqueue the resulting PUT for batch flush. The insert
//!   happens first so the query naturally includes the new referrer.
//! - **ADD (new-blob):** the upsert carries `branch_id:[current_branch_id]`
//!   directly (single referrer, lock held) without calling this producer.
//! - **REMOVE (F9):** call AFTER the blob_refs DELETE and INSIDE the
//!   ContentKeyLock, then execute `overwrite_payload` (PUT) SYNCHRONOUSLY.
//!   Not enqueued (batching REMOVE re-introduces the F04 race).
//!
//! The Qdrant PUT caller lives in `crate::qdrant::membership::put_membership`
//! (the CALLER, not a re-implementer of this query).
//!
//! Neighbors: [`super::ladder`] (ADD existing-blob call site, rewired from the
//!   F6 inline form), [`crate::qdrant::membership`] (the PUT producer that calls
//!   this), F9 delete path (future synchronous REMOVE call site).

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

/// Return the FULL branch membership set for `blob_id`.
///
/// Executes `SELECT DISTINCT branch_id FROM blob_refs WHERE blob_id = ?`
/// (the canonical single producer, FP-2 / DR GP-1). The result is the
/// authoritative membership set to use in the Qdrant `overwrite_payload` (PUT)
/// — it reflects exactly what SQLite records at call time.
///
/// ## Ordering
///
/// SQLite does not guarantee a stable row order for `DISTINCT` queries. The
/// returned `Vec` is in SQLite's natural DISTINCT order, which is non-deterministic
/// across runs. Callers that need a stable set for comparison (e.g. idempotency
/// tests) MUST sort before asserting equality.
///
/// ## Timing invariant (arch §6.3)
///
/// The caller MUST execute the relevant `blob_refs` mutation (INSERT or DELETE)
/// inside the same ContentKeyLock BEFORE calling this function, so the query
/// reflects the post-mutation state. This is the single-producer discipline: the
/// mutation happens in SQLite first, then the membership is derived from SQLite.
pub async fn compute_membership(
    pool: &SqlitePool,
    blob_id: i64,
) -> Result<Vec<String>, StorageError> {
    let rows = sqlx::query("SELECT DISTINCT branch_id FROM blob_refs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("compute_membership: {e}")))?;
    Ok(rows
        .into_iter()
        .map(|r| r.get::<String, _>("branch_id"))
        .collect())
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

    /// Insert one blob row and return its blob_id.
    async fn insert_blob(pool: &SqlitePool) -> i64 {
        let result = sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES ('ck-mem-test','hash1','pt-mem-test',?,'hello',X'',X'','2024-01-01')",
        )
        .bind(TENANT)
        .execute(pool)
        .await
        .expect("blob insert");
        result.last_insert_rowid()
    }

    /// Insert a file row and return its file_id.
    async fn insert_file(pool: &SqlitePool, branch: &str, path: &str) -> i64 {
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

    /// Insert a blob_ref linking blob to file+branch.
    async fn insert_ref(pool: &SqlitePool, branch: &str, file_id: i64, blob_id: i64) {
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
    }

    // AC-F7.1 (ADD existing-blob): compute_membership returns only the branch that
    // has a referrer row, reflecting the post-INSERT state. No referrers -> empty.
    #[tokio::test]
    async fn returns_empty_when_no_referrers() {
        let fx = fixture(BRANCH_A).await;
        let blob_id = insert_blob(&fx.pool).await;
        let result = compute_membership(&fx.pool, blob_id)
            .await
            .expect("compute_membership");
        assert!(result.is_empty(), "no refs -> empty membership");
    }

    // compute_membership returns the single branch when one referrer exists.
    #[tokio::test]
    async fn returns_single_branch() {
        let fx = fixture(BRANCH_A).await;
        let blob_id = insert_blob(&fx.pool).await;
        let file_id = insert_file(&fx.pool, BRANCH_A, "a.rs").await;
        insert_ref(&fx.pool, BRANCH_A, file_id, blob_id).await;

        let mut result = compute_membership(&fx.pool, blob_id)
            .await
            .expect("compute_membership");
        result.sort();
        assert_eq!(result, vec![BRANCH_A.to_string()]);
    }

    // compute_membership returns DISTINCT branches when two branches reference the blob.
    #[tokio::test]
    async fn returns_distinct_branches() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let blob_id = insert_blob(&fx.pool).await;
        let file_a = insert_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = insert_file(&fx.pool, BRANCH_B, "a.rs").await;
        insert_ref(&fx.pool, BRANCH_A, file_a, blob_id).await;
        insert_ref(&fx.pool, BRANCH_B, file_b, blob_id).await;

        let mut result = compute_membership(&fx.pool, blob_id)
            .await
            .expect("compute_membership");
        result.sort();
        assert_eq!(
            result,
            vec![BRANCH_A.to_string(), BRANCH_B.to_string()],
            "two distinct branches"
        );
    }

    // AC-F7.4 — idempotency: calling compute_membership twice with the same SQLite
    // state yields identical sorted branch sets (B1).
    #[tokio::test]
    async fn idempotent_same_state_yields_identical_set() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let blob_id = insert_blob(&fx.pool).await;
        let file_a = insert_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = insert_file(&fx.pool, BRANCH_B, "b.rs").await;
        insert_ref(&fx.pool, BRANCH_A, file_a, blob_id).await;
        insert_ref(&fx.pool, BRANCH_B, file_b, blob_id).await;

        let mut first = compute_membership(&fx.pool, blob_id)
            .await
            .expect("first call");
        first.sort();

        let mut second = compute_membership(&fx.pool, blob_id)
            .await
            .expect("second call");
        second.sort();

        assert_eq!(
            first, second,
            "idempotent: two calls on same state are equal"
        );
    }

    // REMOVE timing (arch §6.3): after a blob_ref DELETE, compute_membership excludes
    // the removed branch naturally (no special logic required).
    #[tokio::test]
    async fn excludes_deleted_ref_naturally() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let blob_id = insert_blob(&fx.pool).await;
        let file_a = insert_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = insert_file(&fx.pool, BRANCH_B, "a.rs").await;
        insert_ref(&fx.pool, BRANCH_A, file_a, blob_id).await;
        insert_ref(&fx.pool, BRANCH_B, file_b, blob_id).await;

        // Delete branch_b's referrer (simulating F9 Step 4).
        sqlx::query("DELETE FROM blob_refs WHERE branch_id = ? AND blob_id = ?")
            .bind(BRANCH_B)
            .bind(blob_id)
            .execute(&fx.pool)
            .await
            .expect("delete ref");

        let mut result = compute_membership(&fx.pool, blob_id)
            .await
            .expect("compute_membership after delete");
        result.sort();
        assert_eq!(
            result,
            vec![BRANCH_A.to_string()],
            "deleted branch excluded from membership"
        );
    }

    // AC-F7.6 structural: the live SQL invocation of the SELECT DISTINCT producer
    // appears exactly once in the write crate source -- in this file. We match only
    // actual sqlx::query call lines (not doc-comments, not test-needle assignments).
    // The pattern is the sqlx::query string opening that contains the query text.
    #[test]
    fn select_distinct_producer_appears_exactly_once_in_write_crate() {
        use std::path::Path;

        let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        // Match lines that contain the live sqlx::query invocation of the producer.
        // This excludes doc-comment lines (which start with //!) and the test-needle
        // variable assignment in this same test module.
        let needle = "sqlx::query(\"SELECT DISTINCT branch_id FROM blob_refs WHERE blob_id";

        let mut matches: Vec<String> = Vec::new();
        collect_live_query_lines(&src_root, needle, &mut matches);

        assert_eq!(
            matches.len(),
            1,
            "expected the sqlx::query SELECT DISTINCT producer to appear EXACTLY ONCE \
             in src/rust/storage-write/src/, found {} match(es): {:?}",
            matches.len(),
            matches
        );

        // The single occurrence must be in blob/membership.rs (this file).
        assert!(
            matches[0].contains("blob/membership.rs"),
            "the sole live query must be in blob/membership.rs, but found it in: {}",
            matches[0]
        );
    }

    // AC-F7.3 structural: qdrant/membership.rs must NOT contain set_payload for
    // branch_id or a get_points call. We verify by checking that the qdrant membership
    // source contains neither "set_payload" nor "get_points".
    #[test]
    fn qdrant_membership_contains_no_set_payload_or_get_points() {
        use std::path::Path;

        let qdrant_mem = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qdrant/membership.rs");

        let src = std::fs::read_to_string(&qdrant_mem).unwrap_or_else(|_| {
            panic!(
                "could not read qdrant/membership.rs at {}; was it created?",
                qdrant_mem.display()
            )
        });

        // Only check production code: stop scanning at the #[cfg(test)] boundary
        // so assert-lines in the test module cannot self-trip the guard.
        let production_src: String = src
            .lines()
            .take_while(|l| !l.trim().starts_with("#[cfg(test)]"))
            .collect::<Vec<_>>()
            .join("\n");

        // Forbidden: bare ".set_payload(" call (not ".overwrite_payload(").
        // We search production_src directly; no need for a per-line loop.
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
                     (AC-F7.3); use overwrite_payload. Found: {:?}",
                    line
                );
            }
        }

        // Forbidden: any ".get_points(" call in production code.
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

    fn collect_live_query_lines(dir: &std::path::Path, needle: &str, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_live_query_lines(&path, needle, out);
            } else if path.extension().map_or(false, |e| e == "rs") {
                if let Ok(src) = std::fs::read_to_string(&path) {
                    for line in src.lines() {
                        if line.contains(needle) {
                            out.push(format!("{}:{}", path.display(), line.trim()));
                        }
                    }
                }
            }
        }
    }
}
