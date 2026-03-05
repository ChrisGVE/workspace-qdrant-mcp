//! Operational state free functions for reading/writing key-value state in SQLite.

use sqlx::SqlitePool;
use wqm_common::timestamps;

use super::DaemonStateResult;

/// Set a value in the operational_state table (upsert).
///
/// `project_id`: None for global entries, Some for project-scoped entries.
/// Internally stored as empty string for global, actual ID for project-scoped.
pub async fn set_operational_state(
    pool: &SqlitePool,
    key: &str,
    component: &str,
    value: &str,
    project_id: Option<&str>,
) -> DaemonStateResult<()> {
    let now = timestamps::now_utc();
    let pid = project_id.unwrap_or("");
    sqlx::query(
        r#"INSERT INTO operational_state (key, component, value, project_id, updated_at)
           VALUES (?1, ?2, ?3, ?4, ?5)
           ON CONFLICT (key, component, project_id) DO UPDATE SET
               value = excluded.value,
               updated_at = excluded.updated_at"#,
    )
    .bind(key)
    .bind(component)
    .bind(value)
    .bind(pid)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get a value from the operational_state table.
///
/// When `project_id` is None, returns the global entry (where project_id = '').
/// When `project_id` is Some, returns the project-scoped entry.
pub async fn get_operational_state(
    pool: &SqlitePool,
    key: &str,
    component: &str,
    project_id: Option<&str>,
) -> DaemonStateResult<Option<String>> {
    let pid = project_id.unwrap_or("");
    let row: Option<String> = sqlx::query_scalar(
        "SELECT value FROM operational_state WHERE key = ?1 AND component = ?2 AND project_id = ?3",
    )
    .bind(key)
    .bind(component)
    .bind(pid)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Poll the database for pause state and sync to a shared AtomicBool.
///
/// This function queries whether any enabled watch folder is paused and sets
/// the provided `pause_flag` accordingly. Used by the daemon to detect
/// CLI-driven pause/resume changes that bypass the gRPC endpoint.
///
/// Returns `true` if the flag value changed.
pub async fn poll_pause_state(
    pool: &SqlitePool,
    pause_flag: &std::sync::atomic::AtomicBool,
) -> DaemonStateResult<bool> {
    let count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1 AND enabled = 1",
    )
    .fetch_one(pool)
    .await?;
    let db_paused = count > 0;
    let previous = pause_flag.swap(db_paused, std::sync::atomic::Ordering::SeqCst);
    Ok(previous != db_paused)
}
