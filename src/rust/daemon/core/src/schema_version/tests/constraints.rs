use super::super::*;
use super::create_test_pool;

#[tokio::test]
async fn test_qdrant_chunks_cascade_delete() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, chunk_count, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', '2025-01-01T00:00:00Z', 'abc123', 2, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let file_id: i64 = sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'src/main.rs'")
        .fetch_one(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
         VALUES (?1, 'point-1', 0, 'hash1', '2025-01-01T00:00:00Z')"
    ).bind(file_id).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
         VALUES (?1, 'point-2', 1, 'hash2', '2025-01-01T00:00:00Z')"
    ).bind(file_id).execute(&pool).await.unwrap();

    let chunk_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
        .fetch_one(&pool).await.unwrap();
    assert_eq!(chunk_count, 2);

    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id).execute(&pool).await.unwrap();

    let chunk_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
        .fetch_one(&pool).await.unwrap();
    assert_eq!(chunk_count_after, 0, "qdrant_chunks should be deleted via CASCADE");
}

#[tokio::test]
async fn test_tracked_files_unique_constraint() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let result = sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-02T00:00:00Z', 'hash2', '2025-01-02T00:00:00Z', '2025-01-02T00:00:00Z')"
    ).execute(&pool).await;

    assert!(result.is_err(), "Duplicate (watch_folder_id, file_path, branch) should violate UNIQUE constraint");
}
