mod backfill;
mod detection;
mod unit;

/// Helper to create the in-memory test schema for backfill tests.
pub(super) async fn create_backfill_schema(pool: &sqlx::SqlitePool) {
    sqlx::query(
        "CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL DEFAULT 'projects',
            tenant_id TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_archived INTEGER NOT NULL DEFAULT 0
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            relative_path TEXT,
            component TEXT,
            UNIQUE(watch_folder_id, file_path)
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE project_components (
            component_id TEXT PRIMARY KEY,
            watch_folder_id TEXT NOT NULL,
            component_name TEXT NOT NULL,
            base_path TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto',
            patterns TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(watch_folder_id, component_name)
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}
