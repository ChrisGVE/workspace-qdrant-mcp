//! Tests for the tenant processing strategy.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sqlx::SqlitePool;
use tokio::sync::Semaphore;

use crate::allowed_extensions::AllowedExtensions;
use crate::context::ProcessingContext;
use crate::document_processor::DocumentProcessor;
use crate::embedding::{
    DenseEmbedding, DenseProvider, EmbeddingConfig, EmbeddingError, EmbeddingGenerator,
};
use crate::git::GitStatus;
use crate::lexicon::LexiconManager;
use crate::patterns::exclusion::should_exclude_file;
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_schema::{
    CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL, ItemType, ProjectPayload,
    QueueOperation, QueueStatus, UnifiedQueueItem,
};
use crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL;
use wqm_common::constants::COLLECTION_PROJECTS;

use super::project::{
    enqueue_project_scan, insert_watch_folder, WatchFolderInsertStatus,
};
use super::TenantStrategy;

#[test]
fn test_tenant_strategy_handles_tenant_items() {
    let strategy = TenantStrategy;
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Scan));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Delete));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Rename));
}

#[test]
fn test_tenant_strategy_handles_doc_items() {
    let strategy = TenantStrategy;
    assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Delete));
    assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Uplift));
}

#[test]
fn test_tenant_strategy_rejects_non_tenant_items() {
    let strategy = TenantStrategy;
    assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
    assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
    assert!(!strategy.handles(&ItemType::Folder, &QueueOperation::Scan));
}

#[test]
fn test_tenant_strategy_name() {
    let strategy = TenantStrategy;
    assert_eq!(strategy.name(), "tenant");
}

/// Test that the exclusion check logic correctly identifies files that should be cleaned up.
/// This tests the core decision logic used by cleanup_excluded_files without needing
/// Qdrant or SQLite connections.
#[test]
fn test_cleanup_exclusion_logic_identifies_hidden_files() {
    let project_root = Path::new("/home/user/project");

    // Simulate file paths as they would be stored in Qdrant (absolute paths)
    let qdrant_paths = vec![
        "/home/user/project/src/main.rs",
        "/home/user/project/.hidden_file",
        "/home/user/project/src/.secret",
        "/home/user/project/.git/config",
        "/home/user/project/src/lib.rs",
        "/home/user/project/node_modules/package/index.js",
        "/home/user/project/.env",
        "/home/user/project/README.md",
        "/home/user/project/src/.cache/data",
        "/home/user/project/.github/workflows/ci.yml",
    ];

    let mut should_delete = Vec::new();
    let mut should_keep = Vec::new();

    for qdrant_file in &qdrant_paths {
        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.to_string(),
        };

        if should_exclude_file(&rel_path) {
            should_delete.push(qdrant_file.to_string());
        } else {
            should_keep.push(qdrant_file.to_string());
        }
    }

    // Hidden files should be marked for deletion
    assert!(
        should_delete.contains(&"/home/user/project/.hidden_file".to_string()),
        "Expected .hidden_file to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/src/.secret".to_string()),
        "Expected src/.secret to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/.git/config".to_string()),
        "Expected .git/config to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/.env".to_string()),
        "Expected .env to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/src/.cache/data".to_string()),
        "Expected src/.cache/data to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/node_modules/package/index.js".to_string()),
        "Expected node_modules content to be excluded"
    );

    // Normal files should NOT be deleted
    assert!(
        should_keep.contains(&"/home/user/project/src/main.rs".to_string()),
        "Expected src/main.rs to be kept"
    );
    assert!(
        should_keep.contains(&"/home/user/project/src/lib.rs".to_string()),
        "Expected src/lib.rs to be kept"
    );
    assert!(
        should_keep.contains(&"/home/user/project/README.md".to_string()),
        "Expected README.md to be kept"
    );

    // .github/ should be whitelisted (not excluded)
    assert!(
        should_keep.contains(&"/home/user/project/.github/workflows/ci.yml".to_string()),
        "Expected .github/workflows/ci.yml to be kept (whitelisted)"
    );
}

#[test]
fn test_cleanup_exclusion_logic_with_non_strippable_paths() {
    // Test when Qdrant paths don't share the project root prefix
    let project_root = Path::new("/home/user/project");
    let qdrant_file = "/different/root/src/.hidden";

    let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
        Ok(stripped) => stripped.to_string_lossy().to_string(),
        Err(_) => qdrant_file.to_string(),
    };

    // Should still detect hidden component even with full path fallback
    assert!(
        should_exclude_file(&rel_path),
        "Expected .hidden to be excluded even when path can't be stripped"
    );
}

#[test]
fn test_cleanup_exclusion_logic_empty_paths() {
    // Verify no panic with edge cases
    let project_root = Path::new("/home/user/project");
    let qdrant_paths: Vec<String> = vec![];

    let mut count = 0u64;
    for qdrant_file in &qdrant_paths {
        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.clone(),
        };

        if should_exclude_file(&rel_path) {
            count += 1;
        }
    }

    assert_eq!(count, 0, "Empty path list should produce zero deletions");
}

// =============================================================================
// Tests for Unit 2 (audit issue #8): gate scan enqueue on watch_folder insert
// =============================================================================
//
// Background: `insert_watch_folder` silently returns Ok(()) when the
// project_root is a subdirectory of an already-registered project. Before
// this fix, `handle_project_add` then called `enqueue_project_scan`
// unconditionally, generating File/Add items with an orphan tenant_id (no
// watch_folder row exists). These tests verify the new gating contract.

/// No-op dense provider for tests that don't need actual embeddings.
///
/// `insert_watch_folder` never calls into the embedding pipeline, but the
/// `ProcessingContext::new` constructor requires a non-`None`
/// `EmbeddingGenerator`. This stub satisfies the signature without
/// downloading any models or touching ONNX Runtime.
#[derive(Debug)]
struct NoopDenseProvider;

#[async_trait]
impl DenseProvider for NoopDenseProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
        Ok(texts
            .iter()
            .map(|t| DenseEmbedding {
                vector: vec![0.0; 1],
                model_name: "noop".to_string(),
                sequence_length: t.len(),
            })
            .collect())
    }

    fn output_dim(&self) -> usize {
        1
    }

    fn provider_label(&self) -> &str {
        "noop"
    }

    fn metrics_label(&self) -> &'static str {
        "fastembed"
    }

    async fn probe(&self) -> Result<(), EmbeddingError> {
        Ok(())
    }
}

/// Build an in-memory SQLite pool with the schemas `insert_watch_folder`
/// and `enqueue_project_scan` actually touch.
async fn setup_test_pool() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");

    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(index_sql)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue index");
    }

    sqlx::query(CREATE_WATCH_FOLDERS_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create watch_folders table");

    pool
}

/// Build a minimal `ProcessingContext` for tenant-strategy unit tests.
///
/// The Qdrant `StorageClient` is constructed lazily (no network connection
/// is made up-front); tests must not invoke any method that would actually
/// talk to Qdrant. `insert_watch_folder` and `enqueue_project_scan` only
/// touch SQLite, so this is safe for our scope.
fn build_test_context(pool: SqlitePool) -> ProcessingContext {
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));
    let storage_client = Arc::new(StorageClient::new());
    let dense_provider = Arc::new(NoopDenseProvider);
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(EmbeddingConfig::default(), dense_provider)
            .expect("EmbeddingGenerator::new should succeed with NoopDenseProvider"),
    );
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_semaphore = Arc::new(Semaphore::new(1));
    let lexicon_manager = Arc::new(LexiconManager::new(pool.clone(), 1.2));
    let allowed_extensions = Arc::new(AllowedExtensions::default());

    ProcessingContext::new(
        pool,
        queue_manager,
        storage_client,
        embedding_generator,
        document_processor,
        embedding_semaphore,
        lexicon_manager,
        None,
        None,
        allowed_extensions,
    )
}

/// Insert a watch_folder row representing an already-registered project.
async fn insert_parent_watch_folder(pool: &SqlitePool, path: &str, tenant_id: &str) {
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        r#"INSERT INTO watch_folders (
            watch_id, path, collection, tenant_id, is_active,
            follow_symlinks, enabled, cleanup_on_disable,
            is_git_tracked, is_worktree,
            created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, 1, 0, 1, 0, 0, 0, ?5, ?5)"#,
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(path)
    .bind(COLLECTION_PROJECTS)
    .bind(tenant_id)
    .bind(&now)
    .execute(pool)
    .await
    .expect("seed parent watch_folder");
}

/// Build a synthetic `UnifiedQueueItem` for a Tenant/Add of `project_root`.
fn make_tenant_add_item(tenant_id: &str, project_root: &str) -> UnifiedQueueItem {
    let payload = ProjectPayload {
        project_root: project_root.to_string(),
        git_remote: None,
        project_type: None,
        old_tenant_id: None,
        is_active: Some(false),
    };
    let payload_json = serde_json::to_string(&payload).unwrap();
    let now = wqm_common::timestamps::now_utc();
    UnifiedQueueItem {
        queue_id: uuid::Uuid::new_v4().to_string(),
        idempotency_key: format!("test-{}", tenant_id),
        item_type: ItemType::Tenant,
        op: QueueOperation::Add,
        tenant_id: tenant_id.to_string(),
        collection: COLLECTION_PROJECTS.to_string(),
        status: QueueStatus::Pending,
        branch: "main".to_string(),
        payload_json,
        metadata: None,
        created_at: now.clone(),
        updated_at: now,
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: None,
        search_status: None,
        decision_json: None,
    }
}

/// Count Tenant/Scan rows enqueued for a given tenant in `unified_queue`.
async fn count_tenant_scans(pool: &SqlitePool, tenant_id: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue \
         WHERE item_type = 'tenant' AND op = 'scan' AND tenant_id = ?1",
    )
    .bind(tenant_id)
    .fetch_one(pool)
    .await
    .unwrap()
}

#[tokio::test]
async fn insert_watch_folder_returns_skipped_subdir_for_nested_path() {
    let pool = setup_test_pool().await;
    insert_parent_watch_folder(&pool, "/parent", "parent_tenant").await;

    let ctx = build_test_context(pool);
    let item = make_tenant_add_item("subdir_tenant", "/parent/sub");
    let payload: ProjectPayload = serde_json::from_str(&item.payload_json).unwrap();

    let status = insert_watch_folder(&ctx, &item, &payload, &GitStatus::not_git())
        .await
        .expect("insert_watch_folder must not error");

    assert_eq!(
        status,
        WatchFolderInsertStatus::SkippedSubdir,
        "subdir of registered project must return SkippedSubdir"
    );
}

#[tokio::test]
async fn insert_watch_folder_returns_inserted_for_new_top_level_path() {
    let pool = setup_test_pool().await;

    let ctx = build_test_context(pool);
    let item = make_tenant_add_item("fresh_tenant", "/fresh/project");
    let payload: ProjectPayload = serde_json::from_str(&item.payload_json).unwrap();

    let status = insert_watch_folder(&ctx, &item, &payload, &GitStatus::not_git())
        .await
        .expect("insert_watch_folder must not error");

    assert_eq!(
        status,
        WatchFolderInsertStatus::Inserted,
        "first registration of a new path must return Inserted"
    );
}

#[tokio::test]
async fn insert_watch_folder_returns_already_exists_on_idempotent_replay() {
    let pool = setup_test_pool().await;
    insert_parent_watch_folder(&pool, "/parent", "parent_tenant").await;

    let ctx = build_test_context(pool);
    let item = make_tenant_add_item("parent_tenant", "/parent");
    let payload: ProjectPayload = serde_json::from_str(&item.payload_json).unwrap();

    let status = insert_watch_folder(&ctx, &item, &payload, &GitStatus::not_git())
        .await
        .expect("insert_watch_folder must not error");

    assert_eq!(
        status,
        WatchFolderInsertStatus::AlreadyExists,
        "re-registering the exact same path must return AlreadyExists"
    );
}

/// End-to-end gating test: a Tenant/Add for a subdirectory of an
/// already-registered project MUST NOT enqueue a Tenant/Scan. Without the
/// fix in this unit, `handle_project_add` would still call
/// `enqueue_project_scan` after `insert_watch_folder` silently returned
/// `Ok(())`, leaving an orphan tenant_id in the queue.
#[tokio::test]
async fn subdir_does_not_enqueue_scan() {
    let pool = setup_test_pool().await;
    insert_parent_watch_folder(&pool, "/parent", "parent_tenant").await;

    let ctx = build_test_context(pool.clone());
    let item = make_tenant_add_item("subdir_tenant", "/parent/sub");
    let payload: ProjectPayload = serde_json::from_str(&item.payload_json).unwrap();

    // Drive the same decision the production `handle_project_add` makes.
    let status = insert_watch_folder(&ctx, &item, &payload, &GitStatus::not_git())
        .await
        .expect("insert_watch_folder must not error");

    match status {
        WatchFolderInsertStatus::Inserted | WatchFolderInsertStatus::AlreadyExists => {
            enqueue_project_scan(&ctx, &item, &payload).await;
        }
        WatchFolderInsertStatus::SkippedSubdir => {
            // Gate fires: no scan enqueued, matching the new
            // `handle_project_add` contract.
        }
    }

    let scan_count = count_tenant_scans(&pool, "subdir_tenant").await;
    assert_eq!(
        scan_count, 0,
        "Tenant/Scan must not be enqueued for an orphan tenant_id (subdir of registered project)"
    );

    // Sanity check: the parent's tenant_id must also not have a stray scan
    // (the gating decision was specifically about the subdir).
    let parent_scan_count = count_tenant_scans(&pool, "parent_tenant").await;
    assert_eq!(
        parent_scan_count, 0,
        "no scan should have been enqueued for the pre-existing parent tenant either"
    );
}

/// Companion to `subdir_does_not_enqueue_scan`: when the path is NOT a
/// subdir, the gating logic must enqueue the scan as before.
#[tokio::test]
async fn top_level_add_does_enqueue_scan() {
    let pool = setup_test_pool().await;

    let ctx = build_test_context(pool.clone());
    let item = make_tenant_add_item("fresh_tenant", "/fresh/project");
    let payload: ProjectPayload = serde_json::from_str(&item.payload_json).unwrap();

    let status = insert_watch_folder(&ctx, &item, &payload, &GitStatus::not_git())
        .await
        .expect("insert_watch_folder must not error");

    match status {
        WatchFolderInsertStatus::Inserted | WatchFolderInsertStatus::AlreadyExists => {
            enqueue_project_scan(&ctx, &item, &payload).await;
        }
        WatchFolderInsertStatus::SkippedSubdir => {
            panic!("top-level path must not be classified as SkippedSubdir");
        }
    }

    let scan_count = count_tenant_scans(&pool, "fresh_tenant").await;
    assert_eq!(
        scan_count, 1,
        "Tenant/Scan should be enqueued exactly once for a brand-new top-level project"
    );
}
