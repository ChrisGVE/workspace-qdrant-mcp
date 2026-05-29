//! Schema migration implementations for search.db (v1 through v9).

use sqlx::SqlitePool;
use tracing::info;

use super::types::{SearchDbError, SearchDbResult};

/// Migration v1: database initialization.
///
/// Establishes search.db with WAL mode and schema versioning.
pub(super) async fn migrate_v1(_pool: &SqlitePool) -> SearchDbResult<()> {
    info!("Search DB migration v1: database initialized");
    Ok(())
}

/// Migration v2: Create code_lines table for line-level code index.
///
/// Uses gap-based seq ordering (REAL) for efficient line insertions.
/// Line numbers derived via ROW_NUMBER() at query time.
pub(super) async fn migrate_v2(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{CREATE_CODE_LINES_INDEXES_SQL, CREATE_CODE_LINES_SQL};

    info!("Search DB migration v2: creating code_lines table");

    sqlx::query(CREATE_CODE_LINES_SQL).execute(pool).await?;

    for index_sql in CREATE_CODE_LINES_INDEXES_SQL {
        sqlx::query(index_sql).execute(pool).await?;
    }

    Ok(())
}

/// Migration v3: Create FTS5 trigram virtual table for substring search.
///
/// External content mode links to `code_lines` via `line_id`.
/// Trigram tokenizer enables fast substring matching.
pub(super) async fn migrate_v3(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::CREATE_CODE_LINES_FTS_SQL;

    info!("Search DB migration v3: creating code_lines_fts virtual table (FTS5 trigram)");

    sqlx::query(CREATE_CODE_LINES_FTS_SQL).execute(pool).await?;

    Ok(())
}

/// Migration v4: Create file_metadata table for project/branch/path scoping.
///
/// Denormalizes tenant_id, branch, and file_path into search.db so
/// FTS5 queries can be scoped without cross-database JOINs.
pub(super) async fn migrate_v4(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{CREATE_FILE_METADATA_INDEXES_SQL, CREATE_FILE_METADATA_SQL};

    info!("Search DB migration v4: creating file_metadata table for project/branch/path scoping");

    sqlx::query(CREATE_FILE_METADATA_SQL).execute(pool).await?;

    for index_sql in CREATE_FILE_METADATA_INDEXES_SQL {
        sqlx::query(index_sql).execute(pool).await?;
    }

    Ok(())
}

/// Migration v5: Add base_point columns to file_metadata.
///
/// Aligns file_metadata with the base_point identity model used by
/// Qdrant and tracked_files. Adds base_point, relative_path, file_hash.
pub(super) async fn migrate_v5(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{
        ALTER_FILE_METADATA_V5_SQL, CREATE_FILE_METADATA_BASE_POINT_INDEX_SQL,
    };

    info!("Search DB migration v5: adding base_point columns to file_metadata");

    for alter_sql in ALTER_FILE_METADATA_V5_SQL {
        sqlx::query(alter_sql).execute(pool).await?;
    }

    sqlx::query(CREATE_FILE_METADATA_BASE_POINT_INDEX_SQL)
        .execute(pool)
        .await?;

    Ok(())
}

/// Migration v6: Add materialized line_number column to code_lines.
///
/// Eliminates the correlated subquery for line number computation in
/// FTS5 search queries, making line number lookups O(1) per row.
/// Idempotent: skips ALTER if column already exists (fresh databases
/// create the table with line_number in v2).
pub(super) async fn migrate_v6(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{ALTER_CODE_LINES_V6_SQL, POPULATE_LINE_NUMBERS_V6_SQL};

    info!("Search DB migration v6: adding materialized line_number column to code_lines");

    // Check if line_number column already exists (fresh DBs include it in CREATE TABLE)
    let has_column: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('code_lines') WHERE name = 'line_number'",
    )
    .fetch_one(pool)
    .await?;

    if !has_column {
        sqlx::query(ALTER_CODE_LINES_V6_SQL).execute(pool).await?;

        // Populate line_number for existing rows from seq ordering
        sqlx::query(POPULATE_LINE_NUMBERS_V6_SQL)
            .execute(pool)
            .await?;
    }

    Ok(())
}

/// Migration v7: Add `size_bytes` to `file_metadata`.
///
/// Token-economy support (spec 20 §3.2): persisted file size lets `grep`
/// compute `bytes_in` against real numbers instead of a per-file proxy.
/// Idempotent — skips ALTER if the column is already present (fresh DBs
/// created at v7+ have it in CREATE TABLE).
pub(super) async fn migrate_v7(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::ALTER_FILE_METADATA_V7_SQL;

    info!("Search DB migration v7: adding size_bytes to file_metadata");

    let has_column: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('file_metadata') WHERE name = 'size_bytes'",
    )
    .fetch_one(pool)
    .await?;

    if !has_column {
        sqlx::query(ALTER_FILE_METADATA_V7_SQL)
            .execute(pool)
            .await?;
    }

    Ok(())
}

/// Migration v8: Add `fts5_skipped` to `file_metadata`.
///
/// Marker column for files where FTS5 ingestion was bypassed by the
/// `WQM_FTS5_HARD_CAP` guard. The hard-cap path writes the metadata row
/// (so it shows up in admin UI / Grafana panels) but skips inserting any
/// `code_lines` for the file — necessary to prevent RSS spikes from
/// 600k-line generated files (CSV dumps, proto-generated Java, lockfiles,
/// etc.) being held entirely in memory by the FTS5 batch processor's
/// Phase 1 diff materialization.
///
/// Idempotent: skips ALTER if the column already exists (fresh v8+ DBs
/// create it via `CREATE_FILE_METADATA_SQL`). Also creates a partial
/// index on `fts5_skipped = 1` for cheap admin-UI / metrics queries.
pub(super) async fn migrate_v8(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{
        ALTER_FILE_METADATA_V8_SQL, CREATE_FILE_METADATA_FTS5_SKIPPED_INDEX_SQL,
    };

    info!("Search DB migration v8: adding fts5_skipped to file_metadata");

    let has_column: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('file_metadata') WHERE name = 'fts5_skipped'",
    )
    .fetch_one(pool)
    .await?;

    if !has_column {
        sqlx::query(ALTER_FILE_METADATA_V8_SQL)
            .execute(pool)
            .await?;
    }

    // Always create the partial index (IF NOT EXISTS guarantees idempotency).
    sqlx::query(CREATE_FILE_METADATA_FTS5_SKIPPED_INDEX_SQL)
        .execute(pool)
        .await?;

    Ok(())
}

/// Migration v9: Add churn tracking (`reindex_count`, `first_indexed_at`) to
/// `file_metadata`.
///
/// Lets the admin layer rank files by how often their content is re-indexed
/// (`reindex_count / age(first_indexed_at)`) to surface IDE/build-generated
/// churn as ignore candidates. Idempotent: skips the ALTERs when the column
/// already exists (fresh v9+ DBs create them via `CREATE_FILE_METADATA_SQL`).
/// Both columns are added together, so the `reindex_count` probe gates both.
pub(super) async fn migrate_v9(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{
        ALTER_FILE_METADATA_V9_SQL, CREATE_FILE_METADATA_CHURN_INDEX_SQL,
    };

    info!("Search DB migration v9: adding reindex_count + first_indexed_at to file_metadata");

    let has_column: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('file_metadata') WHERE name = 'reindex_count'",
    )
    .fetch_one(pool)
    .await?;

    if !has_column {
        for alter_sql in ALTER_FILE_METADATA_V9_SQL {
            sqlx::query(alter_sql).execute(pool).await?;
        }
    }

    // Always create the churn index (IF NOT EXISTS guarantees idempotency).
    sqlx::query(CREATE_FILE_METADATA_CHURN_INDEX_SQL)
        .execute(pool)
        .await?;

    Ok(())
}

/// Dispatch a single migration by version number.
pub(super) async fn run_migration(pool: &SqlitePool, version: i32) -> SearchDbResult<()> {
    match version {
        1 => migrate_v1(pool).await,
        2 => migrate_v2(pool).await,
        3 => migrate_v3(pool).await,
        4 => migrate_v4(pool).await,
        5 => migrate_v5(pool).await,
        6 => migrate_v6(pool).await,
        7 => migrate_v7(pool).await,
        8 => migrate_v8(pool).await,
        9 => migrate_v9(pool).await,
        _ => Err(SearchDbError::Migration(format!(
            "Unknown search DB migration version: {}",
            version
        ))),
    }
}
