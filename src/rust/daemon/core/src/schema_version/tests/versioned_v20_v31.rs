use super::super::*;
use super::create_test_pool;

#[sqlx::test]
async fn test_migration_v20_destination_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let columns: Vec<String> =
        sqlx::query_scalar("SELECT name FROM pragma_table_info('unified_queue') ORDER BY name")
            .fetch_all(&pool)
            .await
            .unwrap();

    assert!(
        columns.contains(&"qdrant_status".to_string()),
        "qdrant_status column missing"
    );
    assert!(
        columns.contains(&"search_status".to_string()),
        "search_status column missing"
    );
    assert!(
        columns.contains(&"decision_json".to_string()),
        "decision_json column missing"
    );
}

#[sqlx::test]
async fn test_migration_v20_defaults_and_constraints() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, created_at, updated_at)
         VALUES ('q1', 'k1', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let qdrant_status: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = 'q1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        qdrant_status, "pending",
        "qdrant_status should default to 'pending'"
    );

    let search_status: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = 'q1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        search_status, "pending",
        "search_status should default to 'pending'"
    );

    let decision: Option<String> =
        sqlx::query_scalar("SELECT decision_json FROM unified_queue WHERE queue_id = 'q1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(decision.is_none(), "decision_json should default to NULL");

    let invalid = sqlx::query(
        "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, qdrant_status, created_at, updated_at)
         VALUES ('q2', 'k2', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', 'bogus', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await;
    assert!(
        invalid.is_err(),
        "CHECK constraint should reject invalid qdrant_status"
    );
}

#[sqlx::test]
async fn test_migration_v21_git_tracking_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let columns: Vec<String> =
        sqlx::query_scalar("SELECT name FROM pragma_table_info('watch_folders') ORDER BY name")
            .fetch_all(&pool)
            .await
            .unwrap();

    assert!(
        columns.contains(&"last_commit_hash".to_string()),
        "last_commit_hash column missing"
    );
    assert!(
        columns.contains(&"is_git_tracked".to_string()),
        "is_git_tracked column missing"
    );

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w-test', '/tmp/test-project', 'projects', 'tenant_test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let is_git_tracked: i32 =
        sqlx::query_scalar("SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'w-test'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(is_git_tracked, 0, "is_git_tracked should default to 0");

    let commit_hash: Option<String> =
        sqlx::query_scalar("SELECT last_commit_hash FROM watch_folders WHERE watch_id = 'w-test'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(
        commit_hash.is_none(),
        "last_commit_hash should default to NULL"
    );
}

#[sqlx::test]
async fn test_migration_v21_rules_mirror_table() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let exists: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='rules_mirror'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(exists, "rules_mirror table should exist");

    sqlx::query(
        "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
         VALUES ('m1', 'Always use snake_case', 'global', NULL, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let rule: String =
        sqlx::query_scalar("SELECT rule_text FROM rules_mirror WHERE rule_id = 'm1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(rule, "Always use snake_case");
}

#[sqlx::test]
async fn test_migration_v21_submodule_junction_table() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let exists: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='watch_folder_submodules'"
    ).fetch_one(&pool).await.unwrap();
    assert!(exists, "watch_folder_submodules table should exist");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w-parent', '/tmp/parent', 'projects', 'tenant1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path, created_at, updated_at)
         VALUES ('w-child', '/tmp/parent/lib', 'projects', 'tenant2', 'w-parent', 'lib', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, submodule_path, created_at)
         VALUES ('w-parent', 'w-child', 'lib', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM watch_folder_submodules WHERE parent_watch_id = 'w-parent'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count, 1);

    sqlx::query("DELETE FROM watch_folders WHERE watch_id = 'w-parent'")
        .execute(&pool)
        .await
        .unwrap();

    let count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM watch_folder_submodules")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count_after, 0, "CASCADE delete should remove junction rows");
}

#[sqlx::test]
async fn test_migration_v31_worktree_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let has_is_worktree: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'is_worktree'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(has_is_worktree, "is_worktree column should exist");

    let has_main_worktree_watch_id: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'main_worktree_watch_id'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_main_worktree_watch_id,
        "main_worktree_watch_id column should exist"
    );

    let has_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master \
         WHERE type='index' AND name='idx_watch_main_worktree'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(has_index, "idx_watch_main_worktree index should exist");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_worktree, created_at, updated_at)
         VALUES ('w-main', '/tmp/repo', 'projects', 't1', 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_worktree, main_worktree_watch_id, created_at, updated_at)
         VALUES ('w-wt', '/tmp/repo-wt', 'projects', 't1', 1, 'w-main', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let is_worktree: i32 =
        sqlx::query_scalar("SELECT is_worktree FROM watch_folders WHERE watch_id = 'w-main'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(is_worktree, 0, "is_worktree should be 0 for main tree");

    let is_worktree_wt: i32 =
        sqlx::query_scalar("SELECT is_worktree FROM watch_folders WHERE watch_id = 'w-wt'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(is_worktree_wt, 1, "is_worktree should be 1 for worktree");

    let main_id: Option<String> = sqlx::query_scalar(
        "SELECT main_worktree_watch_id FROM watch_folders WHERE watch_id = 'w-wt'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(main_id.as_deref(), Some("w-main"));

    let invalid = sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_worktree, created_at, updated_at)
         VALUES ('w-bad', '/tmp/bad', 'projects', 't1', 2, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await;
    assert!(
        invalid.is_err(),
        "CHECK constraint should reject is_worktree = 2"
    );
}

#[sqlx::test]
async fn test_migration_v31_idempotent() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.unwrap();

    manager
        .run_migrations()
        .await
        .expect("First migration run should succeed");

    manager
        .run_migration(31)
        .await
        .expect("Running v31 again should not fail");
}
