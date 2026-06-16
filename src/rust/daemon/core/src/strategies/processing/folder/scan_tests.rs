//! Regression tests for progressive single-level directory scanning.
//!
//! Covers the Gate-0 ignore cascade (#105): root-level `.gitignore` /
//! `.wqmignore` patterns must apply when scanning subdirectories, not just
//! when scanning the watch root itself. Without the cascade, a subdirectory
//! scan enqueued thousands of ignored directories (session-env scan storm,
//! #103).

use std::path::Path;
use std::sync::Arc;

use sqlx::Row;
use tempfile::tempdir;
use wqm_common::paths::CanonicalPath;
use wqm_common::timestamps::now_utc;

use crate::allowed_extensions::AllowedExtensions;
use crate::queue_config::QueueConnectionConfig;
use crate::queue_operations::tests::apply_sql_script;
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};

use super::scan::scan_directory_single_level;

/// Build a QueueManager backed by a fresh temp SQLite db with the full
/// watch_folders schema applied (mirrors queue_operations test harness).
async fn test_queue_manager(dir: &Path) -> Arc<QueueManager> {
    let db_path = dir.join("scan_test.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(
        &pool,
        include_str!("../../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();
    Arc::new(manager)
}

/// A minimal in-progress Folder/Scan queue item for driving the scanner.
fn scan_item(tenant_id: &str) -> UnifiedQueueItem {
    let now = now_utc();
    UnifiedQueueItem {
        queue_id: "test-scan-item".to_string(),
        idempotency_key: "test-scan-key".to_string(),
        item_type: ItemType::Folder,
        op: QueueOperation::Scan,
        tenant_id: tenant_id.to_string(),
        collection: "projects".to_string(),
        status: QueueStatus::InProgress,
        branch: "main".to_string(),
        payload_json: "{}".to_string(),
        metadata: None,
        created_at: now.clone(),
        updated_at: now,
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        size_bytes: None,
        qdrant_status: None,
        search_status: None,
        decision_json: None,
    }
}

/// Root `.wqmignore` patterns must cascade into subdirectory scans (#105):
/// a `session-env/` exclusion defined at the watch root applies when the
/// scanner enumerates `claude-max/`, so the ignored directory is never
/// enqueued while sibling content still is.
#[tokio::test]
async fn root_wqmignore_cascades_into_subdirectory_scan() {
    let temp = tempdir().unwrap();
    let root = temp.path();

    // Watch root: .wqmignore excludes session-env/ everywhere.
    std::fs::write(root.join(".wqmignore"), "session-env/\n").unwrap();

    // Subdirectory under scan: contains one ignored dir, one normal dir,
    // and one normal file.
    let sub = root.join("claude-max");
    std::fs::create_dir_all(sub.join("session-env")).unwrap();
    std::fs::write(sub.join("session-env").join("state.json"), "{}").unwrap();
    std::fs::create_dir_all(sub.join("keep")).unwrap();
    std::fs::write(sub.join("keep").join("kept.md"), "kept").unwrap();
    std::fs::write(sub.join("notes.md"), "notes").unwrap();

    let queue_manager = test_queue_manager(root).await;
    let allowed = Arc::new(AllowedExtensions::default());
    let watch_root = CanonicalPath::from_user_input(root.to_str().unwrap()).unwrap();
    let item = scan_item("tenant-scan-cascade");

    // Scan the SUBDIRECTORY (not the root): before #105 the root ignore
    // files were invisible here and session-env/ got enqueued.
    let dir_path = Path::new(watch_root.as_str()).join("claude-max");
    let (files_queued, dirs_queued, files_excluded, errors) = scan_directory_single_level(
        &dir_path,
        &watch_root,
        &item,
        &queue_manager,
        &allowed,
        None,
    )
    .await
    .unwrap();

    assert_eq!(errors, 0, "scan must not error");
    assert_eq!(files_queued, 1, "notes.md should be enqueued");
    assert_eq!(
        dirs_queued, 1,
        "only keep/ should be enqueued; session-env/ is ignored via root .wqmignore"
    );
    assert!(
        files_excluded >= 1,
        "session-env/ must be counted as excluded"
    );

    // Verify queue contents directly: no Folder/Scan payload references
    // session-env, exactly one references keep/.
    let rows = sqlx::query(
        "SELECT payload_json FROM unified_queue WHERE item_type = 'folder' AND op = 'scan'",
    )
    .fetch_all(queue_manager.pool())
    .await
    .unwrap();
    let payloads: Vec<String> = rows
        .iter()
        .map(|r| r.try_get::<String, _>("payload_json").unwrap())
        .collect();
    assert!(
        !payloads.iter().any(|p| p.contains("session-env")),
        "ignored directory must not be enqueued, got: {payloads:?}"
    );
    assert_eq!(
        payloads
            .iter()
            .filter(|p| p.contains("claude-max/keep"))
            .count(),
        1,
        "keep/ must be enqueued exactly once, got: {payloads:?}"
    );
}

/// A queued scan of a directory INSIDE an ignored tree must enqueue nothing
/// (#103/#105 storm self-sustain): before ancestor-aware matching, pattern
/// `session-env/` did not match `session-env/subdir`, so already-enqueued
/// scans of inner directories kept re-spawning children.
#[tokio::test]
async fn scan_inside_ignored_tree_enqueues_nothing() {
    let temp = tempdir().unwrap();
    let root = temp.path();

    std::fs::write(root.join(".wqmignore"), "session-env/\n").unwrap();

    // Simulate an already-enqueued scan targeting a dir INSIDE session-env.
    let inner = root.join("claude-max").join("session-env").join("envs");
    std::fs::create_dir_all(inner.join("child-dir")).unwrap();
    std::fs::write(inner.join("state.md"), "state").unwrap();

    let queue_manager = test_queue_manager(root).await;
    let allowed = Arc::new(AllowedExtensions::default());
    let watch_root = CanonicalPath::from_user_input(root.to_str().unwrap()).unwrap();
    let item = scan_item("tenant-scan-inner");

    let dir_path = Path::new(watch_root.as_str())
        .join("claude-max")
        .join("session-env")
        .join("envs");
    let (files_queued, dirs_queued, files_excluded, errors) = scan_directory_single_level(
        &dir_path,
        &watch_root,
        &item,
        &queue_manager,
        &allowed,
        None,
    )
    .await
    .unwrap();

    assert_eq!(errors, 0);
    assert_eq!(
        (files_queued, dirs_queued),
        (0, 0),
        "everything under an ignored tree must be excluded"
    );
    assert_eq!(files_excluded, 2, "child-dir and state.md both excluded");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue")
        .fetch_one(queue_manager.pool())
        .await
        .unwrap();
    assert_eq!(count, 0, "no items enqueued from inside ignored tree");
}

/// Same cascade applies to FILE patterns: a root `.gitignore` excluding
/// `draft-*.md` must suppress file enqueueing in subdirectory scans. The
/// pattern targets an allowlisted extension (.md) so the assertion proves
/// the ignore cascade, not the extension allowlist.
#[tokio::test]
async fn root_gitignore_file_pattern_cascades_into_subdirectory_scan() {
    let temp = tempdir().unwrap();
    let root = temp.path();

    std::fs::write(root.join(".gitignore"), "draft-*.md\n").unwrap();

    let sub = root.join("app");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("draft-notes.md"), "draft").unwrap();
    std::fs::write(sub.join("readme.md"), "docs").unwrap();

    let queue_manager = test_queue_manager(root).await;
    let allowed = Arc::new(AllowedExtensions::default());
    let watch_root = CanonicalPath::from_user_input(root.to_str().unwrap()).unwrap();
    let item = scan_item("tenant-scan-gitignore");

    let dir_path = Path::new(watch_root.as_str()).join("app");
    let (files_queued, dirs_queued, files_excluded, errors) = scan_directory_single_level(
        &dir_path,
        &watch_root,
        &item,
        &queue_manager,
        &allowed,
        None,
    )
    .await
    .unwrap();

    assert_eq!(errors, 0);
    assert_eq!(dirs_queued, 0);
    assert_eq!(files_queued, 1, "only readme.md should be enqueued");
    assert!(files_excluded >= 1, "draft-notes.md must be excluded");

    let rows = sqlx::query("SELECT payload_json FROM unified_queue WHERE item_type = 'file'")
        .fetch_all(queue_manager.pool())
        .await
        .unwrap();
    let payloads: Vec<String> = rows
        .iter()
        .map(|r| r.try_get::<String, _>("payload_json").unwrap())
        .collect();
    assert!(
        !payloads.iter().any(|p| p.contains("draft-notes.md")),
        "ignored file must not be enqueued, got: {payloads:?}"
    );
}
