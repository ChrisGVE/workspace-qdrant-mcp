//! Migration v40: Add `chunker_version` to `tracked_files`.
//!
//! The unchanged-hash skip in the file ingest gate made re-ingestion of
//! unmodified files free — and also made extractor upgrades unreachable:
//! after the registry gained `semantic_patterns` for `.proto`, a full
//! per-tenant re-embed still left every unchanged `.proto` file on its old
//! text chunks (the skip fires before the chunker runs).
//!
//! `chunker_version` stores the chunking-configuration fingerprint
//! (`tree_sitter::chunker::chunking_fingerprint`: chunker logic version +
//! language + digest of the language's registry `semantic_patterns`) that
//! produced the row's chunks. The gate now skips only when BOTH the file
//! hash AND the fingerprint are unchanged, so registry/extractor upgrades
//! propagate on the next scan or re-embed that visits the file.
//!
//! NULL (legacy rows, zero-byte rows) is grandfathered — it never triggers
//! a re-chunk by itself; a forced re-embed stamps it.
//!
//! Idempotent: skips the ALTER when the column already exists (fresh DBs
//! get it from the v37 rebuild DDL, which carries the current table shape).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V40Migration;

#[async_trait]
impl Migration for V40Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v40: Adding chunker_version column to tracked_files");

        let has_column: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'chunker_version'",
        )
        .fetch_one(pool)
        .await?;

        if has_column {
            debug!("chunker_version column already exists, skipping ALTER TABLE");
        } else {
            sqlx::query(crate::tracked_files_schema::MIGRATE_V40_ADD_CHUNKER_VERSION_SQL)
                .execute(pool)
                .await?;
        }

        info!("Migration v40 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        40
    }

    fn description(&self) -> &'static str {
        "Add chunker_version (chunking-configuration fingerprint) to tracked_files"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn fresh_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    /// Pre-v40 table shape: the v37 DDL without the chunker_version column,
    /// mirroring a production DB that ran the v37 rebuild before this
    /// migration existed.
    async fn setup_pre_v40(pool: &SqlitePool) {
        sqlx::query(
            r#"CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_folder_id TEXT NOT NULL,
                branch TEXT,
                file_type TEXT,
                language TEXT,
                file_mtime TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                chunking_method TEXT,
                lsp_status TEXT DEFAULT 'none',
                treesitter_status TEXT DEFAULT 'none',
                last_error TEXT,
                needs_reconcile INTEGER DEFAULT 0,
                reconcile_reason TEXT,
                extension TEXT,
                is_test INTEGER DEFAULT 0,
                collection TEXT NOT NULL DEFAULT 'projects',
                base_point TEXT,
                relative_path TEXT NOT NULL,
                incremental INTEGER DEFAULT 0,
                component TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(watch_folder_id, relative_path, branch)
            )"#,
        )
        .execute(pool)
        .await
        .unwrap();
    }

    async fn has_chunker_version(pool: &SqlitePool) -> bool {
        sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'chunker_version'",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn v40_adds_the_column_and_preserves_rows() {
        let pool = fresh_pool().await;
        setup_pre_v40(&pool).await;
        sqlx::query(
            "INSERT INTO tracked_files
               (watch_folder_id, relative_path, branch, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'src/a.proto', 'main', 't', 'h1', 't', 't')",
        )
        .execute(&pool)
        .await
        .unwrap();

        V40Migration.up(&pool).await.unwrap();

        assert!(has_chunker_version(&pool).await);
        // Existing rows decode as NULL (grandfathered).
        let stored: Option<String> = sqlx::query_scalar(
            "SELECT chunker_version FROM tracked_files WHERE relative_path = 'src/a.proto'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(stored, None);
    }

    #[tokio::test]
    async fn v40_is_idempotent_when_column_already_exists() {
        let pool = fresh_pool().await;
        // Fresh-DB path: the v37 rebuild DDL already carries the column.
        sqlx::query(crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL)
            .execute(&pool)
            .await
            .unwrap();
        assert!(has_chunker_version(&pool).await);

        V40Migration.up(&pool).await.unwrap();
        V40Migration.up(&pool).await.unwrap();

        assert!(has_chunker_version(&pool).await);
    }
}
