//! Integration tests for branch management end-to-end scenarios.
//!
//! Verifies the SQLite-side branch management logic using in-memory databases.
//! Qdrant-side behavior is covered by unit tests in each module.

use std::collections::HashMap;
use std::time::Duration;

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

use workspace_qdrant_core::tracked_files_schema::{
    CREATE_TRACKED_FILES_V40_INDEXES_SQL, CREATE_TRACKED_FILES_V40_SQL,
};
use workspace_qdrant_core::watch_folders_schema;

async fn create_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("pool")
}

async fn setup(pool: &SqlitePool) {
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(CREATE_TRACKED_FILES_V40_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_V40_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
}

async fn insert_watch(pool: &SqlitePool, wid: &str, tid: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
         VALUES (?1, '/tmp/test', 'projects', ?2, 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(wid).bind(tid).execute(pool).await.unwrap();
}

async fn insert_file(pool: &SqlitePool, wid: &str, path: &str, branches: &[&str], hash: &str) {
    let branches_json = serde_json::json!(branches).to_string();
    let primary = branches.first().copied().unwrap_or("main");
    let bp = wqm_common::hashing::compute_base_point("t1", path, hash);
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, relative_path, primary_branch, branches,
         file_mtime, file_hash, collection, base_point, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, '2025-01-01T00:00:00Z', ?5, 'projects', ?6,
                 '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(wid)
    .bind(path)
    .bind(primary)
    .bind(&branches_json)
    .bind(hash)
    .bind(&bp)
    .execute(pool)
    .await
    .unwrap();
}

fn get_branches(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

async fn get_file_branches(pool: &SqlitePool, path: &str) -> Vec<String> {
    let json: String =
        sqlx::query_scalar("SELECT branches FROM tracked_files WHERE relative_path = ?1")
            .bind(path)
            .fetch_one(pool)
            .await
            .unwrap();
    get_branches(&json)
}

/// Scenario 1: Content-hash dedup — same file on two branches shares one base_point.
#[tokio::test]
async fn test_content_hash_dedup_same_base_point() {
    let bp1 = wqm_common::hashing::compute_base_point("t1", "src/shared.rs", "same_hash");
    let bp2 = wqm_common::hashing::compute_base_point("t1", "src/shared.rs", "same_hash");
    assert_eq!(bp1, bp2);
}

/// Scenario 2: Branch-agnostic base_point (branch not in hash formula).
#[tokio::test]
async fn test_base_point_branch_agnostic() {
    let bp1 = wqm_common::hashing::compute_base_point("tenant", "src/file.rs", "hash123");
    let bp2 = wqm_common::hashing::compute_base_point("tenant", "src/file.rs", "hash123");
    assert_eq!(bp1, bp2);

    let bp3 = wqm_common::hashing::compute_base_point("tenant", "src/file.rs", "hash456");
    assert_ne!(bp1, bp3);
}

/// Scenario 3: SQLite branches JSON array supports multiple branches.
#[tokio::test]
async fn test_branches_json_array_multi_branch() {
    let pool = create_pool().await;
    setup(&pool).await;
    insert_watch(&pool, "w1", "t1").await;

    insert_file(
        &pool,
        "w1",
        "src/shared.rs",
        &["main", "feature", "dev"],
        "h1",
    )
    .await;

    let branches = get_file_branches(&pool, "src/shared.rs").await;
    assert_eq!(branches.len(), 3);
    assert!(branches.contains(&"main".to_string()));
    assert!(branches.contains(&"feature".to_string()));
    assert!(branches.contains(&"dev".to_string()));
}

/// Scenario 4: json_each query finds files by branch membership.
#[tokio::test]
async fn test_json_each_branch_query() {
    let pool = create_pool().await;
    setup(&pool).await;
    insert_watch(&pool, "w1", "t1").await;

    insert_file(&pool, "w1", "src/a.rs", &["main", "feature"], "h1").await;
    insert_file(&pool, "w1", "src/b.rs", &["main"], "h2").await;
    insert_file(&pool, "w1", "src/c.rs", &["feature"], "h3").await;

    let feature_files: Vec<String> = sqlx::query_scalar(
        "SELECT relative_path FROM tracked_files
         WHERE watch_folder_id = 'w1'
           AND EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = 'feature')
         ORDER BY relative_path",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    assert_eq!(feature_files, vec!["src/a.rs", "src/c.rs"]);

    let main_files: Vec<String> = sqlx::query_scalar(
        "SELECT relative_path FROM tracked_files
         WHERE watch_folder_id = 'w1'
           AND EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = 'main')
         ORDER BY relative_path",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    assert_eq!(main_files, vec!["src/a.rs", "src/b.rs"]);
}

/// Scenario 5: Branch removal via json_group_array.
#[tokio::test]
async fn test_branch_removal_from_json_array() {
    let pool = create_pool().await;
    setup(&pool).await;
    insert_watch(&pool, "w1", "t1").await;

    insert_file(
        &pool,
        "w1",
        "src/shared.rs",
        &["main", "feature", "dev"],
        "h1",
    )
    .await;

    sqlx::query(
        "UPDATE tracked_files
         SET branches = (
             SELECT json_group_array(j.value)
             FROM json_each(branches) AS j
             WHERE j.value != 'feature'
         )
         WHERE relative_path = 'src/shared.rs'",
    )
    .execute(&pool)
    .await
    .unwrap();

    let branches = get_file_branches(&pool, "src/shared.rs").await;
    assert_eq!(branches.len(), 2);
    assert!(branches.contains(&"main".to_string()));
    assert!(branches.contains(&"dev".to_string()));
    assert!(!branches.contains(&"feature".to_string()));
}

/// Scenario 6: Cross-branch search filter — wildcard omits branch predicate.
#[tokio::test]
async fn test_cross_branch_filter_wildcard() {
    use serde_json::json;

    let mut filter = HashMap::new();
    filter.insert("tenant_id".to_string(), json!("proj1"));
    filter.insert("branch".to_string(), json!("*"));

    let result = workspace_qdrant_core::storage::build_filter_from_json(&filter)
        .expect("should produce filter with just tenant_id");

    assert_eq!(result.must.len(), 1);
}

/// Scenario 7: Branch-scoped filter — branch becomes branches array match.
#[tokio::test]
async fn test_branch_scoped_filter() {
    use serde_json::json;

    let mut filter = HashMap::new();
    filter.insert("tenant_id".to_string(), json!("proj1"));
    filter.insert("branch".to_string(), json!("feature"));

    let result = workspace_qdrant_core::storage::build_filter_from_json(&filter)
        .expect("should produce filter with tenant_id + branches");

    assert_eq!(result.must.len(), 2);
}

/// Scenario 8: Discovery classification — shared vs novel files.
#[tokio::test]
async fn test_discovery_classification() {
    use workspace_qdrant_core::branch_discovery::db::KnownFile;
    use workspace_qdrant_core::branch_discovery::scanner::classify_files;

    let mut fs_files = HashMap::new();
    fs_files.insert("src/a.rs".to_string(), "hash_a".to_string());
    fs_files.insert("src/b.rs".to_string(), "hash_b_new".to_string());
    fs_files.insert("src/c.rs".to_string(), "hash_c".to_string());

    let mut known = HashMap::new();
    known.insert(
        ("src/a.rs".to_string(), "hash_a".to_string()),
        KnownFile {
            file_id: 1,
            branches: vec!["main".to_string()],
            base_point: Some("bp1".to_string()),
        },
    );
    known.insert(
        ("src/b.rs".to_string(), "hash_b_old".to_string()),
        KnownFile {
            file_id: 2,
            branches: vec!["main".to_string()],
            base_point: Some("bp2".to_string()),
        },
    );

    let (shared, novel) = classify_files(&fs_files, &known, "feature");

    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].file_id, 1);

    assert_eq!(novel.len(), 2);
    assert!(novel.contains(&"src/b.rs".to_string()));
    assert!(novel.contains(&"src/c.rs".to_string()));
}

/// Scenario 9: UNIQUE(watch_folder_id, relative_path, file_hash) allows same path different hash.
#[tokio::test]
async fn test_unique_constraint_allows_different_hashes() {
    let pool = create_pool().await;
    setup(&pool).await;
    insert_watch(&pool, "w1", "t1").await;

    insert_file(&pool, "w1", "src/file.rs", &["main"], "hash_v1").await;
    insert_file(&pool, "w1", "src/file.rs", &["feature"], "hash_v2").await;

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files WHERE relative_path = 'src/file.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count, 2);
}
