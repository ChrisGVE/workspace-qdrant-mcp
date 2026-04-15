//! Mtime tracking for `.gitignore` and `.wqmignore` files.
//!
//! Stores the last-seen modification time per ignore file per project root
//! in the `ignore_file_mtimes` table. The file watcher compares against
//! these stored values to skip reconciliation when an inotify event fires
//! but the file hasn't actually changed.

use sqlx::SqlitePool;

/// Get the stored mtime (unix seconds) for an ignore file.
///
/// Returns `None` if no entry exists for the given project root + file path.
pub async fn get_ignore_mtime(
    pool: &SqlitePool,
    project_root: &str,
    file_path: &str,
) -> Result<Option<i64>, sqlx::Error> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT mtime_unix FROM ignore_file_mtimes \
         WHERE project_root = ?1 AND file_path = ?2",
    )
    .bind(project_root)
    .bind(file_path)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|(mtime,)| mtime))
}

/// Insert or update the stored mtime for an ignore file.
pub async fn set_ignore_mtime(
    pool: &SqlitePool,
    project_root: &str,
    file_path: &str,
    mtime_unix: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO ignore_file_mtimes (project_root, file_path, mtime_unix) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(project_root, file_path) \
         DO UPDATE SET mtime_unix = excluded.mtime_unix",
    )
    .bind(project_root)
    .bind(file_path)
    .bind(mtime_unix)
    .execute(pool)
    .await?;
    Ok(())
}

/// Remove all mtime entries for a project root (used on project unregister).
pub async fn clear_ignore_mtimes(
    pool: &SqlitePool,
    project_root: &str,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM ignore_file_mtimes WHERE project_root = ?1")
        .bind(project_root)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::SqlitePool;

    async fn setup_test_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE ignore_file_mtimes (
                project_root TEXT NOT NULL,
                file_path TEXT NOT NULL,
                mtime_unix INTEGER NOT NULL,
                PRIMARY KEY (project_root, file_path)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let pool = setup_test_pool().await;
        let result = get_ignore_mtime(&pool, "/project", ".gitignore")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_and_get_mtime() {
        let pool = setup_test_pool().await;
        set_ignore_mtime(&pool, "/project", ".gitignore", 1700000000)
            .await
            .unwrap();
        let mtime = get_ignore_mtime(&pool, "/project", ".gitignore")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(mtime, 1700000000);
    }

    #[tokio::test]
    async fn update_mtime_overwrites() {
        let pool = setup_test_pool().await;
        set_ignore_mtime(&pool, "/project", ".gitignore", 1700000000)
            .await
            .unwrap();
        set_ignore_mtime(&pool, "/project", ".gitignore", 1700001000)
            .await
            .unwrap();
        let mtime = get_ignore_mtime(&pool, "/project", ".gitignore")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(mtime, 1700001000);
    }

    #[tokio::test]
    async fn different_files_independent() {
        let pool = setup_test_pool().await;
        set_ignore_mtime(&pool, "/project", ".gitignore", 100)
            .await
            .unwrap();
        set_ignore_mtime(&pool, "/project", ".wqmignore", 200)
            .await
            .unwrap();

        assert_eq!(
            get_ignore_mtime(&pool, "/project", ".gitignore")
                .await
                .unwrap(),
            Some(100)
        );
        assert_eq!(
            get_ignore_mtime(&pool, "/project", ".wqmignore")
                .await
                .unwrap(),
            Some(200)
        );
    }

    #[tokio::test]
    async fn clear_removes_all_for_project() {
        let pool = setup_test_pool().await;
        set_ignore_mtime(&pool, "/project-a", ".gitignore", 100)
            .await
            .unwrap();
        set_ignore_mtime(&pool, "/project-a", ".wqmignore", 200)
            .await
            .unwrap();
        set_ignore_mtime(&pool, "/project-b", ".gitignore", 300)
            .await
            .unwrap();

        let deleted = clear_ignore_mtimes(&pool, "/project-a").await.unwrap();
        assert_eq!(deleted, 2);

        // project-a entries gone
        assert!(get_ignore_mtime(&pool, "/project-a", ".gitignore")
            .await
            .unwrap()
            .is_none());
        // project-b untouched
        assert_eq!(
            get_ignore_mtime(&pool, "/project-b", ".gitignore")
                .await
                .unwrap(),
            Some(300)
        );
    }
}
