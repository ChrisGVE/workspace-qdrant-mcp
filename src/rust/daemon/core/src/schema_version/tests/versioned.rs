use super::super::*;
use super::create_test_pool;

#[tokio::test]
async fn test_search_behavior_view_classification() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.unwrap();
    manager.run_migrations().await.unwrap();

    sqlx::query(
        "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e1', 'sess-a', 'claude', 'rg', 'search', '2025-01-01T00:00:00.000Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e2', 'sess-b', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:01:00.000Z')"
    ).execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e3', 'sess-b', 'claude', 'mcp_qdrant', 'open', '2025-01-01T00:01:30.000Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e4', 'sess-c', 'claude', 'rg', 'search', '2025-01-01T00:02:00.000Z')"
    ).execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e5', 'sess-c', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:02:30.000Z')"
    ).execute(&pool).await.unwrap();

    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT session_id, tool, behavior FROM search_behavior ORDER BY ts"
    )
    .fetch_all(&pool).await.unwrap();

    assert!(!rows.is_empty(), "search_behavior view should return rows");

    let bypass = rows.iter().find(|(s, _, b)| s == "sess-a" && b == "bypass");
    assert!(bypass.is_some(), "Should detect bypass pattern (rg as first event)");

    let success = rows.iter().find(|(s, _, b)| s == "sess-b" && b == "success");
    assert!(success.is_some(), "Should detect success pattern (mcp_qdrant followed by open)");

    let fallback = rows.iter().find(|(s, t, b)| s == "sess-c" && t == "mcp_qdrant" && b == "fallback");
    assert!(fallback.is_some(), "Should detect fallback pattern (mcp_qdrant after rg within 2 min)");
}

#[tokio::test]
async fn test_migration_v16_keywords_tables() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    for table in &["keywords", "tags", "keyword_baskets", "canonical_tags", "tag_hierarchy_edges"] {
        let exists: bool = sqlx::query_scalar(
            &format!("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{}')", table)
        )
        .fetch_one(&pool).await.unwrap();
        assert!(exists, "{} table should exist after v16 migration", table);
    }

    let idx_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND (
            name LIKE 'idx_keywords_%' OR
            name LIKE 'idx_tags_%' OR
            name LIKE 'idx_keyword_baskets_%' OR
            name LIKE 'idx_canonical_tags_%' OR
            name LIKE 'idx_hierarchy_edges_%'
        )"
    )
    .fetch_one(&pool).await.unwrap();
    assert_eq!(idx_count, 14, "Should have 14 keyword/tag indexes (3+3+2+3+3)");
}

#[tokio::test]
async fn test_migration_v16_cascade_deletes() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tags (doc_id, tag, collection, tenant_id) VALUES ('doc1', 'vector-search', 'projects', 't1')"
    ).execute(&pool).await.unwrap();

    let tag_id: i64 = sqlx::query_scalar("SELECT tag_id FROM tags WHERE doc_id = 'doc1'")
        .fetch_one(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO keyword_baskets (tag_id, keywords_json, tenant_id) VALUES (?1, '[\"qdrant\",\"embedding\"]', 't1')"
    ).bind(tag_id).execute(&pool).await.unwrap();

    let basket_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
        .fetch_one(&pool).await.unwrap();
    assert_eq!(basket_count, 1);

    sqlx::query("DELETE FROM tags WHERE tag_id = ?1").bind(tag_id).execute(&pool).await.unwrap();

    let basket_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
        .fetch_one(&pool).await.unwrap();
    assert_eq!(basket_count_after, 0, "keyword_baskets should cascade delete when tag is deleted");
}

#[tokio::test]
async fn test_migration_v16_multi_tenant_isolation() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query(
        "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc1', 'qdrant', 0.9, 'projects', 't1')"
    ).execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc2', 'redis', 0.8, 'projects', 't2')"
    ).execute(&pool).await.unwrap();

    let t1_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM keywords WHERE tenant_id = 't1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(t1_count, 1);

    let t2_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM keywords WHERE tenant_id = 't2'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(t2_count, 1);
}

#[tokio::test]
async fn test_migration_v17_operational_state() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='operational_state')"
    ).fetch_one(&pool).await.unwrap();
    assert!(exists, "operational_state table should exist after v17 migration");

    let has_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_operational_state_project'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_index, "project index should exist on operational_state");

    sqlx::query(
        "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'test_val', '2026-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let val: String = sqlx::query_scalar(
        "SELECT value FROM operational_state WHERE key = 'test_key' AND component = 'daemon'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(val, "test_val");

    let result = sqlx::query(
        "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('k2', 'invalid', 'v', '2026-01-01T00:00:00Z')"
    ).execute(&pool).await;
    assert!(result.is_err(), "Invalid component should violate CHECK constraint");

    let result = sqlx::query(
        "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'dup', '2026-01-01T00:00:00Z')"
    ).execute(&pool).await;
    assert!(result.is_err(), "Duplicate (key, component, project_id) should violate PRIMARY KEY");
}

#[tokio::test]
async fn test_migration_v18_indexed_content() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_content')"
    ).fetch_one(&pool).await.unwrap();
    assert!(exists, "indexed_content table should exist after v18 migration");

    let has_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_indexed_content_hash'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_index, "hash index should exist on indexed_content");

    sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at)
         VALUES ('w1', 'test.rs', '2025-01-01T00:00:00Z', 'h1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let file_id: i64 = sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'test.rs'")
        .fetch_one(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO indexed_content (file_id, content, hash, updated_at) VALUES (?1, X'48454C4C4F', 'testhash', '2025-01-01T00:00:00Z')"
    ).bind(file_id).execute(&pool).await.unwrap();

    let hash: String = sqlx::query_scalar("SELECT hash FROM indexed_content WHERE file_id = ?1")
        .bind(file_id).fetch_one(&pool).await.unwrap();
    assert_eq!(hash, "testhash");

    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id).execute(&pool).await.unwrap();

    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM indexed_content")
        .fetch_one(&pool).await.unwrap();
    assert_eq!(count, 0, "indexed_content should cascade delete with tracked_files");
}

#[tokio::test]
async fn test_migration_v19_base_point_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let has_base_point: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'base_point'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_base_point, "base_point column should exist");

    let has_relative_path: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'relative_path'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_relative_path, "relative_path column should exist");

    let has_incremental: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'incremental'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_incremental, "incremental column should exist");

    let has_bp_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_base_point'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_bp_index, "base_point index should exist");

    let has_refcount_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_refcount'"
    ).fetch_one(&pool).await.unwrap();
    assert!(has_refcount_index, "refcount index should exist");
}

#[tokio::test]
async fn test_migration_v19_backfill() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/project', 'projects', 'tenant_abc', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'deadbeef', 'projects', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let bp: Option<String> = sqlx::query_scalar(
        "SELECT base_point FROM tracked_files WHERE file_path = 'src/main.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert!(bp.is_none(), "base_point should be NULL for insert without explicit value");

    let incr: i32 = sqlx::query_scalar(
        "SELECT incremental FROM tracked_files WHERE file_path = 'src/main.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(incr, 0, "incremental should default to 0");
}

#[sqlx::test]
async fn test_migration_v20_destination_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let columns: Vec<String> = sqlx::query_scalar(
        "SELECT name FROM pragma_table_info('unified_queue') ORDER BY name"
    ).fetch_all(&pool).await.unwrap();

    assert!(columns.contains(&"qdrant_status".to_string()), "qdrant_status column missing");
    assert!(columns.contains(&"search_status".to_string()), "search_status column missing");
    assert!(columns.contains(&"decision_json".to_string()), "decision_json column missing");
}

#[sqlx::test]
async fn test_migration_v20_defaults_and_constraints() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, created_at, updated_at)
         VALUES ('q1', 'k1', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let qdrant_status: String = sqlx::query_scalar(
        "SELECT qdrant_status FROM unified_queue WHERE queue_id = 'q1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(qdrant_status, "pending", "qdrant_status should default to 'pending'");

    let search_status: String = sqlx::query_scalar(
        "SELECT search_status FROM unified_queue WHERE queue_id = 'q1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(search_status, "pending", "search_status should default to 'pending'");

    let decision: Option<String> = sqlx::query_scalar(
        "SELECT decision_json FROM unified_queue WHERE queue_id = 'q1'"
    ).fetch_one(&pool).await.unwrap();
    assert!(decision.is_none(), "decision_json should default to NULL");

    let invalid = sqlx::query(
        "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, qdrant_status, created_at, updated_at)
         VALUES ('q2', 'k2', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', 'bogus', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await;
    assert!(invalid.is_err(), "CHECK constraint should reject invalid qdrant_status");
}

#[sqlx::test]
async fn test_migration_v21_git_tracking_columns() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let columns: Vec<String> = sqlx::query_scalar(
        "SELECT name FROM pragma_table_info('watch_folders') ORDER BY name"
    ).fetch_all(&pool).await.unwrap();

    assert!(columns.contains(&"last_commit_hash".to_string()), "last_commit_hash column missing");
    assert!(columns.contains(&"is_git_tracked".to_string()), "is_git_tracked column missing");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w-test', '/tmp/test-project', 'projects', 'tenant_test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let is_git_tracked: i32 = sqlx::query_scalar(
        "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'w-test'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(is_git_tracked, 0, "is_git_tracked should default to 0");

    let commit_hash: Option<String> = sqlx::query_scalar(
        "SELECT last_commit_hash FROM watch_folders WHERE watch_id = 'w-test'"
    ).fetch_one(&pool).await.unwrap();
    assert!(commit_hash.is_none(), "last_commit_hash should default to NULL");
}

#[sqlx::test]
async fn test_migration_v21_rules_mirror_table() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    let exists: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='rules_mirror'"
    ).fetch_one(&pool).await.unwrap();
    assert!(exists, "rules_mirror table should exist");

    sqlx::query(
        "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
         VALUES ('m1', 'Always use snake_case', 'global', NULL, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    let rule: String = sqlx::query_scalar(
        "SELECT rule_text FROM rules_mirror WHERE rule_id = 'm1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(rule, "Always use snake_case");
}

#[sqlx::test]
async fn test_migration_v21_submodule_junction_table() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

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
        "SELECT COUNT(*) FROM watch_folder_submodules WHERE parent_watch_id = 'w-parent'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(count, 1);

    sqlx::query("DELETE FROM watch_folders WHERE watch_id = 'w-parent'")
        .execute(&pool).await.unwrap();

    let count_after: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM watch_folder_submodules"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(count_after, 0, "CASCADE delete should remove junction rows");
}
