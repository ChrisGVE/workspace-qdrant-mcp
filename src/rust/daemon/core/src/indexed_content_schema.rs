//! Indexed Content Cache Table Schema
//!
//! Stores last-indexed file content with SHA256 hash for diff-based skip detection.
//! When a file changes, the daemon compares the new content hash against the stored
//! hash. If equal, the expensive diff/re-index pipeline is skipped entirely.
//!
//! - `file_id`: FK to `tracked_files(file_id)` with CASCADE delete
//! - `content`: Last-indexed file content as BLOB (raw bytes)
//! - `hash`: SHA256 hex digest of `content` (64 chars)
//! - `updated_at`: ISO 8601 timestamp with Z suffix

use sqlx::{Row, SqlitePool};
use wqm_common::timestamps;

// ---------------------------------------------------------------------------
// SQL constants — indexed_content
// ---------------------------------------------------------------------------

/// SQL to create the indexed_content table
pub const CREATE_INDEXED_CONTENT_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS indexed_content (
    file_id INTEGER PRIMARY KEY REFERENCES tracked_files(file_id) ON DELETE CASCADE,
    content BLOB NOT NULL,
    hash TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// SQL to create indexes for the indexed_content table
pub const CREATE_INDEXED_CONTENT_INDEXES_SQL: &[&str] = &[
    // Index for hash lookups (skip detection)
    r#"CREATE INDEX IF NOT EXISTS idx_indexed_content_hash
       ON indexed_content(hash)"#,
];

// ---------------------------------------------------------------------------
// Database operations
// ---------------------------------------------------------------------------

/// Upsert indexed content for a file.
///
/// Inserts or replaces the cached content and hash for the given file_id.
/// Uses INSERT OR REPLACE since file_id is the PRIMARY KEY.
pub async fn upsert_indexed_content(
    pool: &SqlitePool,
    file_id: i64,
    content: &[u8],
    hash: &str,
) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "INSERT OR REPLACE INTO indexed_content (file_id, content, hash, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(file_id)
    .bind(content)
    .bind(hash)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get the stored hash for a file (for skip detection).
///
/// Returns `Some(hash)` if the file has cached content, `None` otherwise.
pub async fn get_indexed_hash(
    pool: &SqlitePool,
    file_id: i64,
) -> Result<Option<String>, sqlx::Error> {
    let row = sqlx::query("SELECT hash FROM indexed_content WHERE file_id = ?1")
        .bind(file_id)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|r| r.get("hash")))
}

/// Get the stored content for a file (for diff computation).
///
/// Returns `Some((content, hash))` if the file has cached content, `None` otherwise.
pub async fn get_indexed_content(
    pool: &SqlitePool,
    file_id: i64,
) -> Result<Option<(Vec<u8>, String)>, sqlx::Error> {
    let row = sqlx::query("SELECT content, hash FROM indexed_content WHERE file_id = ?1")
        .bind(file_id)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|r| (r.get("content"), r.get("hash"))))
}

/// Delete indexed content for a file (explicit, CASCADE also handles this).
pub async fn delete_indexed_content(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM indexed_content WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Get the total number of cached entries and their combined size in bytes.
///
/// Useful for monitoring storage usage of the content cache.
pub async fn get_cache_stats(pool: &SqlitePool) -> Result<(i64, i64), sqlx::Error> {
    let row = sqlx::query(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(LENGTH(content)), 0) as total_bytes FROM indexed_content"
    )
    .fetch_one(pool)
    .await?;
    Ok((row.get("cnt"), row.get("total_bytes")))
}

// ---------------------------------------------------------------------------
// Transaction-aware operations
// ---------------------------------------------------------------------------

/// Upsert indexed content within a transaction.
pub async fn upsert_indexed_content_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    content: &[u8],
    hash: &str,
) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "INSERT OR REPLACE INTO indexed_content (file_id, content, hash, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(file_id)
    .bind(content)
    .bind(hash)
    .bind(&now)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;
    use wqm_common::hashing::compute_content_hash;

    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    async fn setup_tables(pool: &SqlitePool) {
        // Enable foreign keys
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool)
            .await
            .unwrap();

        // Create watch_folders (needed for FK chain)
        sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(pool)
            .await
            .unwrap();

        // Create tracked_files
        sqlx::query(crate::tracked_files_schema::CREATE_TRACKED_FILES_SQL)
            .execute(pool)
            .await
            .unwrap();

        // Create indexed_content
        sqlx::query(CREATE_INDEXED_CONTENT_SQL)
            .execute(pool)
            .await
            .unwrap();
        for idx in CREATE_INDEXED_CONTENT_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }

        // Insert test watch_folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/home/user/project', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(pool).await.unwrap();
    }

    async fn insert_test_file(pool: &SqlitePool, path: &str) -> i64 {
        let result = sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', ?1, '2025-01-01T00:00:00Z', 'abc123', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(path)
        .execute(pool)
        .await
        .unwrap();
        result.last_insert_rowid()
    }

    #[test]
    fn test_create_sql_is_valid() {
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("CREATE TABLE"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("indexed_content"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("file_id INTEGER PRIMARY KEY"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("REFERENCES tracked_files(file_id)"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("ON DELETE CASCADE"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("content BLOB NOT NULL"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("hash TEXT NOT NULL"));
        assert!(CREATE_INDEXED_CONTENT_SQL.contains("updated_at TEXT NOT NULL"));
    }

    #[test]
    fn test_indexes_sql_is_valid() {
        assert_eq!(CREATE_INDEXED_CONTENT_INDEXES_SQL.len(), 1);
        assert!(CREATE_INDEXED_CONTENT_INDEXES_SQL[0].contains("idx_indexed_content_hash"));
    }

    #[tokio::test]
    async fn test_table_creation() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_content')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "indexed_content table should exist");
    }

    #[tokio::test]
    async fn test_index_creation() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_indexed_content_hash'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_index, "hash index should exist");
    }

    #[tokio::test]
    async fn test_upsert_and_get_hash() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/main.rs").await;
        let content = b"fn main() { println!(\"hello\"); }";
        let hash = compute_content_hash(std::str::from_utf8(content).unwrap());

        upsert_indexed_content(&pool, file_id, content, &hash)
            .await
            .unwrap();

        let stored_hash = get_indexed_hash(&pool, file_id).await.unwrap();
        assert_eq!(stored_hash, Some(hash));
    }

    #[tokio::test]
    async fn test_upsert_and_get_content() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/lib.rs").await;
        let content = b"pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let hash = compute_content_hash(std::str::from_utf8(content).unwrap());

        upsert_indexed_content(&pool, file_id, content, &hash)
            .await
            .unwrap();

        let result = get_indexed_content(&pool, file_id).await.unwrap();
        assert!(result.is_some());
        let (stored_content, stored_hash) = result.unwrap();
        assert_eq!(stored_content, content);
        assert_eq!(stored_hash, hash);
    }

    #[tokio::test]
    async fn test_upsert_replaces_on_conflict() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/update.rs").await;
        let content_v1 = b"version 1";
        let hash_v1 = compute_content_hash("version 1");

        upsert_indexed_content(&pool, file_id, content_v1, &hash_v1)
            .await
            .unwrap();

        let content_v2 = b"version 2";
        let hash_v2 = compute_content_hash("version 2");

        upsert_indexed_content(&pool, file_id, content_v2, &hash_v2)
            .await
            .unwrap();

        let result = get_indexed_content(&pool, file_id).await.unwrap().unwrap();
        assert_eq!(result.0, b"version 2");
        assert_eq!(result.1, hash_v2);

        // Only one row should exist for this file_id
        let count: i32 =
            sqlx::query_scalar("SELECT COUNT(*) FROM indexed_content WHERE file_id = ?1")
                .bind(file_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_hash_comparison_skip_detection() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/skip.rs").await;
        let content = b"unchanged content";
        let hash = compute_content_hash("unchanged content");

        upsert_indexed_content(&pool, file_id, content, &hash)
            .await
            .unwrap();

        // Simulate new file content arriving — compute hash and compare
        let new_content = "unchanged content";
        let new_hash = compute_content_hash(new_content);

        let stored_hash = get_indexed_hash(&pool, file_id).await.unwrap().unwrap();
        assert_eq!(
            stored_hash, new_hash,
            "Same content should produce matching hash (skip)"
        );

        // Different content should NOT match
        let changed_hash = compute_content_hash("changed content");
        assert_ne!(
            stored_hash, changed_hash,
            "Different content should produce different hash (process)"
        );
    }

    #[tokio::test]
    async fn test_get_nonexistent_file() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let hash = get_indexed_hash(&pool, 99999).await.unwrap();
        assert!(hash.is_none());

        let content = get_indexed_content(&pool, 99999).await.unwrap();
        assert!(content.is_none());
    }

    #[tokio::test]
    async fn test_delete_indexed_content() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/delete.rs").await;
        upsert_indexed_content(&pool, file_id, b"content", "hash123")
            .await
            .unwrap();

        assert!(get_indexed_hash(&pool, file_id).await.unwrap().is_some());

        delete_indexed_content(&pool, file_id).await.unwrap();

        assert!(get_indexed_hash(&pool, file_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_cascade_delete_from_tracked_files() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/cascade.rs").await;
        upsert_indexed_content(&pool, file_id, b"cascade test", "chash")
            .await
            .unwrap();

        // Verify content exists
        assert!(get_indexed_hash(&pool, file_id).await.unwrap().is_some());

        // Delete tracked_file — should CASCADE to indexed_content
        sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .execute(&pool)
            .await
            .unwrap();

        // Verify content is gone
        assert!(get_indexed_hash(&pool, file_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        // Empty cache
        let (count, bytes) = get_cache_stats(&pool).await.unwrap();
        assert_eq!(count, 0);
        assert_eq!(bytes, 0);

        // Add some entries
        let f1 = insert_test_file(&pool, "src/a.rs").await;
        let f2 = insert_test_file(&pool, "src/b.rs").await;
        let content1 = b"short";
        let content2 = b"a bit longer content here";

        upsert_indexed_content(&pool, f1, content1, "h1")
            .await
            .unwrap();
        upsert_indexed_content(&pool, f2, content2, "h2")
            .await
            .unwrap();

        let (count, bytes) = get_cache_stats(&pool).await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(bytes, (content1.len() + content2.len()) as i64);
    }

    #[tokio::test]
    async fn test_upsert_indexed_content_tx_commit() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/tx.rs").await;

        let mut tx = pool.begin().await.unwrap();
        upsert_indexed_content_tx(&mut tx, file_id, b"tx content", "txhash")
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let hash = get_indexed_hash(&pool, file_id).await.unwrap();
        assert_eq!(hash, Some("txhash".to_string()));
    }

    #[tokio::test]
    async fn test_upsert_indexed_content_tx_rollback() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/tx_rollback.rs").await;

        {
            let mut tx = pool.begin().await.unwrap();
            upsert_indexed_content_tx(&mut tx, file_id, b"rollback content", "rbhash")
                .await
                .unwrap();
            // Drop without commit = rollback
        }

        let hash = get_indexed_hash(&pool, file_id).await.unwrap();
        assert!(hash.is_none(), "Rolled-back upsert should not be visible");
    }

    #[tokio::test]
    async fn test_large_content_blob() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/large.rs").await;

        // 100KB content
        let content: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let hash =
            compute_content_hash(std::str::from_utf8(&vec![b'x'; 100_000]).unwrap_or("fallback"));

        upsert_indexed_content(&pool, file_id, &content, &hash)
            .await
            .unwrap();

        let result = get_indexed_content(&pool, file_id).await.unwrap().unwrap();
        assert_eq!(result.0.len(), 100_000);
        assert_eq!(result.0, content);
    }

    #[tokio::test]
    async fn test_binary_content_blob() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_test_file(&pool, "src/binary.rs").await;

        // Content with null bytes and non-UTF8 sequences
        let content: Vec<u8> = vec![0x00, 0xFF, 0xFE, 0x01, 0x80, 0x90, 0x00, 0xAB];

        upsert_indexed_content(&pool, file_id, &content, "binhash")
            .await
            .unwrap();

        let result = get_indexed_content(&pool, file_id).await.unwrap().unwrap();
        assert_eq!(
            result.0, content,
            "Binary content should round-trip exactly"
        );
    }

    #[tokio::test]
    async fn test_hash_computation_performance() {
        // Verify SHA256 hash computation is fast for various file sizes
        use std::time::Instant;

        for size in [1_000, 10_000, 100_000] {
            let content = "x".repeat(size);
            let start = Instant::now();
            let _hash = compute_content_hash(&content);
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_millis() < 100,
                "Hash computation for {}B took {}ms (target: <100ms)",
                size,
                elapsed.as_millis()
            );
        }
    }
}
