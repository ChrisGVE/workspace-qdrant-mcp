//! Schema migration implementations for search.db (v1 through v6).

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
    use crate::code_lines_schema::{CREATE_CODE_LINES_SQL, CREATE_CODE_LINES_INDEXES_SQL};

    info!("Search DB migration v2: creating code_lines table");

    sqlx::query(CREATE_CODE_LINES_SQL)
        .execute(pool)
        .await?;

    for index_sql in CREATE_CODE_LINES_INDEXES_SQL {
        sqlx::query(index_sql)
            .execute(pool)
            .await?;
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

    sqlx::query(CREATE_CODE_LINES_FTS_SQL)
        .execute(pool)
        .await?;

    Ok(())
}

/// Migration v4: Create file_metadata table for project/branch/path scoping.
///
/// Denormalizes tenant_id, branch, and file_path into search.db so
/// FTS5 queries can be scoped without cross-database JOINs.
pub(super) async fn migrate_v4(pool: &SqlitePool) -> SearchDbResult<()> {
    use crate::code_lines_schema::{CREATE_FILE_METADATA_SQL, CREATE_FILE_METADATA_INDEXES_SQL};

    info!("Search DB migration v4: creating file_metadata table for project/branch/path scoping");

    sqlx::query(CREATE_FILE_METADATA_SQL)
        .execute(pool)
        .await?;

    for index_sql in CREATE_FILE_METADATA_INDEXES_SQL {
        sqlx::query(index_sql)
            .execute(pool)
            .await?;
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
        sqlx::query(alter_sql)
            .execute(pool)
            .await?;
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
        sqlx::query(ALTER_CODE_LINES_V6_SQL)
            .execute(pool)
            .await?;

        // Populate line_number for existing rows from seq ordering
        sqlx::query(POPULATE_LINE_NUMBERS_V6_SQL)
            .execute(pool)
            .await?;
    }

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
        _ => Err(SearchDbError::Migration(format!(
            "Unknown search DB migration version: {}",
            version
        ))),
    }
}
