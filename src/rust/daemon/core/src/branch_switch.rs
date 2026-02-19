//! Branch Switch Protocol (Task 9)
//!
//! Handles branch switch and commit events detected by the git watcher.
//! On branch switch:
//!   1. Uses `diff_tree` to find changed files between old and new commits
//!   2. Batch-updates unchanged files (branch column only, no re-ingestion)
//!   3. Enqueues changed files for re-ingestion via the unified queue
//!   4. Updates `last_commit_hash` in `watch_folders`
//!
//! On new commit (same branch):
//!   1. Uses `diff_tree` to find changed files since last known commit
//!   2. Enqueues changed files for update

use std::collections::HashSet;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::git_watcher::{diff_tree, FileChange, FileChangeStatus, GitEvent, GitEventType};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation};
use crate::watching_queue::get_current_branch;

/// Result of a branch switch operation
#[derive(Debug, Clone, Default)]
pub struct BranchSwitchStats {
    /// Files batch-updated (branch metadata only, no re-ingestion)
    pub batch_updated: u64,
    /// Files enqueued for re-ingestion (content changed)
    pub enqueued_changed: u64,
    /// Files enqueued for addition (new on target branch)
    pub enqueued_added: u64,
    /// Files enqueued for deletion (removed on target branch)
    pub enqueued_deleted: u64,
    /// Errors during processing
    pub errors: u64,
}

/// Handle a git event by dispatching to the appropriate handler.
pub async fn handle_git_event(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<BranchSwitchStats, String> {
    // Look up watch folder info
    let wf = sqlx::query_as::<_, (String, String, String)>(
        "SELECT path, collection, tenant_id FROM watch_folders WHERE watch_id = ?1"
    )
    .bind(&event.watch_folder_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folder: {}", e))?
    .ok_or_else(|| format!("Watch folder {} not found", event.watch_folder_id))?;

    let (project_root, collection, tenant_id) = wf;

    match &event.event_type {
        GitEventType::BranchSwitch => {
            handle_branch_switch(
                event, pool, queue_manager,
                &project_root, &collection, &tenant_id,
            ).await
        }
        GitEventType::Commit | GitEventType::Merge | GitEventType::Pull | GitEventType::Rebase => {
            handle_new_commit(
                event, pool, queue_manager,
                &project_root, &collection, &tenant_id,
            ).await
        }
        GitEventType::Reset => {
            // Reset can change arbitrary files — enqueue a full scan
            info!(
                "Git reset detected for {}, enqueueing full scan",
                event.watch_folder_id
            );
            enqueue_tenant_scan(queue_manager, &tenant_id, &collection, &project_root).await?;
            Ok(BranchSwitchStats::default())
        }
        GitEventType::Stash | GitEventType::Unknown => {
            debug!(
                "Ignoring git event {:?} for {}",
                event.event_type, event.watch_folder_id
            );
            Ok(BranchSwitchStats::default())
        }
    }
}

/// Handle a branch switch: diff-tree for changes, batch-update unchanged, enqueue changed.
async fn handle_branch_switch(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    project_root: &str,
    collection: &str,
    tenant_id: &str,
) -> Result<BranchSwitchStats, String> {
    let root = Path::new(project_root);
    let new_branch = event.branch.as_deref().unwrap_or("default");
    let old_branch = event.old_branch.as_deref()
        .unwrap_or_else(|| get_current_branch(root).leak());

    info!(
        "Branch switch: {} -> {} for {} (old_sha={:.8}, new_sha={:.8})",
        old_branch, new_branch, event.watch_folder_id,
        &event.old_sha, &event.new_sha
    );

    // Get changed files between old and new commits via diff-tree
    let changes = diff_tree(root, &event.old_sha, &event.new_sha)
        .map_err(|e| format!("diff_tree failed: {}", e))?;

    let changed_paths: HashSet<String> = changes.iter().map(|c| c.path.clone()).collect();

    let mut stats = BranchSwitchStats::default();

    // 1. Batch update unchanged files: update branch in tracked_files
    //    These files have identical content — only the branch metadata changes.
    match batch_update_branch(
        pool, &event.watch_folder_id, old_branch, new_branch, &changed_paths,
    ).await {
        Ok(count) => {
            stats.batch_updated = count;
            if count > 0 {
                info!(
                    "Batch-updated {} unchanged files: branch {} -> {}",
                    count, old_branch, new_branch
                );
            }
        }
        Err(e) => {
            warn!("Batch branch update failed: {}", e);
            stats.errors += 1;
        }
    }

    // 2. Enqueue changed files for re-ingestion
    for change in &changes {
        let result = enqueue_changed_file(
            queue_manager, change, tenant_id, collection, project_root, new_branch,
        ).await;
        match result {
            Ok(op) => match op {
                QueueOperation::Update => stats.enqueued_changed += 1,
                QueueOperation::Add => stats.enqueued_added += 1,
                QueueOperation::Delete => stats.enqueued_deleted += 1,
                _ => {}
            },
            Err(e) => {
                warn!("Failed to enqueue changed file {}: {}", change.path, e);
                stats.errors += 1;
            }
        }
    }

    // 3. Update last_commit_hash in watch_folders
    if let Err(e) = update_last_commit_hash(pool, &event.watch_folder_id, &event.new_sha).await {
        warn!("Failed to update last_commit_hash: {}", e);
        stats.errors += 1;
    }

    info!(
        "Branch switch complete for {}: {} batch-updated, {} changed, {} added, {} deleted, {} errors",
        event.watch_folder_id, stats.batch_updated, stats.enqueued_changed,
        stats.enqueued_added, stats.enqueued_deleted, stats.errors
    );

    Ok(stats)
}

/// Handle a new commit on the same branch: diff-tree vs parent, enqueue changed files.
async fn handle_new_commit(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    project_root: &str,
    collection: &str,
    tenant_id: &str,
) -> Result<BranchSwitchStats, String> {
    let root = Path::new(project_root);
    let branch = event.branch.as_deref().unwrap_or("default");

    info!(
        "New commit on branch {} for {} (old_sha={:.8}, new_sha={:.8})",
        branch, event.watch_folder_id,
        &event.old_sha, &event.new_sha
    );

    let changes = diff_tree(root, &event.old_sha, &event.new_sha)
        .map_err(|e| format!("diff_tree failed: {}", e))?;

    let mut stats = BranchSwitchStats::default();

    for change in &changes {
        let result = enqueue_changed_file(
            queue_manager, change, tenant_id, collection, project_root, branch,
        ).await;
        match result {
            Ok(op) => match op {
                QueueOperation::Update => stats.enqueued_changed += 1,
                QueueOperation::Add => stats.enqueued_added += 1,
                QueueOperation::Delete => stats.enqueued_deleted += 1,
                _ => {}
            },
            Err(e) => {
                warn!("Failed to enqueue changed file {}: {}", change.path, e);
                stats.errors += 1;
            }
        }
    }

    // Update last_commit_hash
    if let Err(e) = update_last_commit_hash(pool, &event.watch_folder_id, &event.new_sha).await {
        warn!("Failed to update last_commit_hash: {}", e);
        stats.errors += 1;
    }

    info!(
        "Commit processed for {}: {} changed, {} added, {} deleted, {} errors",
        event.watch_folder_id, stats.enqueued_changed,
        stats.enqueued_added, stats.enqueued_deleted, stats.errors
    );

    Ok(stats)
}

/// Enqueue a single changed file based on its diff-tree status.
/// Returns the operation type that was enqueued.
async fn enqueue_changed_file(
    queue_manager: &QueueManager,
    change: &FileChange,
    tenant_id: &str,
    collection: &str,
    project_root: &str,
    branch: &str,
) -> Result<QueueOperation, String> {
    let abs_path = Path::new(project_root).join(&change.path);
    let abs_str = abs_path.to_string_lossy().to_string();

    let (op, old_path) = match &change.status {
        FileChangeStatus::Modified => (QueueOperation::Update, None),
        FileChangeStatus::Added => (QueueOperation::Add, None),
        FileChangeStatus::Deleted => (QueueOperation::Delete, None),
        FileChangeStatus::Renamed { old_path, .. } => {
            // Delete old path, add new path
            let old_abs = Path::new(project_root).join(old_path);
            enqueue_file_op(
                queue_manager, tenant_id, collection,
                &old_abs.to_string_lossy(), QueueOperation::Delete, branch,
            ).await?;
            (QueueOperation::Add, Some(old_path.clone()))
        }
        FileChangeStatus::Copied { .. } => (QueueOperation::Add, None),
        FileChangeStatus::TypeChanged => (QueueOperation::Update, None),
    };

    enqueue_file_op(queue_manager, tenant_id, collection, &abs_str, op.clone(), branch).await?;

    // For rename, report as Update for stats (it's logically an update, just with path change)
    if old_path.is_some() {
        return Ok(QueueOperation::Update);
    }

    Ok(op)
}

/// Enqueue a file operation to the unified queue.
async fn enqueue_file_op(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    abs_file_path: &str,
    op: QueueOperation,
    branch: &str,
) -> Result<(), String> {
    let file_payload = FilePayload {
        file_path: abs_file_path.to_string(),
        file_type: None,
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload)
        .map_err(|e| format!("Failed to serialize FilePayload: {}", e))?;

    queue_manager.enqueue_unified(
        ItemType::File,
        op,
        tenant_id,
        collection,
        &payload_json,
        0, // Priority computed at dequeue time
        Some(branch),
        None,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue: {}", e))
}

/// Enqueue a full tenant scan (used for reset events).
async fn enqueue_tenant_scan(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    project_root: &str,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "project_root": project_root,
        "recovery": false,
    }).to_string();

    let branch = get_current_branch(Path::new(project_root));

    queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Scan,
        tenant_id,
        collection,
        &payload,
        0,
        Some(&branch),
        None,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue tenant scan: {}", e))
}

/// Batch update branch column in tracked_files for unchanged files.
///
/// Within a single transaction, updates all tracked files on the old branch
/// that are NOT in the changed_paths set. Also recomputes base_point since
/// it includes the branch in its hash input.
async fn batch_update_branch(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_branch: &str,
    new_branch: &str,
    changed_paths: &HashSet<String>,
) -> Result<u64, String> {
    let now = timestamps::now_utc();

    // Get all tracked files on the old branch for this watch folder
    let files: Vec<(i64, String, String, Option<String>)> = sqlx::query_as(
        "SELECT file_id, file_path, COALESCE(file_hash, ''), relative_path
         FROM tracked_files
         WHERE watch_folder_id = ?1 AND branch = ?2"
    )
    .bind(watch_folder_id)
    .bind(old_branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query tracked files: {}", e))?;

    if files.is_empty() {
        return Ok(0);
    }

    // Look up tenant_id for base_point computation
    let tenant_id: String = sqlx::query_scalar(
        "SELECT tenant_id FROM watch_folders WHERE watch_id = ?1"
    )
    .bind(watch_folder_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to query tenant_id: {}", e))?;

    let mut tx = pool.begin().await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut updated = 0u64;

    for (file_id, file_path, file_hash, relative_path) in &files {
        // Skip files that changed (those will be re-ingested)
        let rel = relative_path.as_deref().unwrap_or(file_path);
        if changed_paths.contains(rel) || changed_paths.contains(file_path.as_str()) {
            continue;
        }

        // Recompute base_point with new branch
        let new_bp = wqm_common::hashing::compute_base_point(
            &tenant_id, new_branch, rel, file_hash,
        );

        sqlx::query(
            "UPDATE tracked_files
             SET branch = ?1, base_point = ?2, updated_at = ?3
             WHERE file_id = ?4"
        )
        .bind(new_branch)
        .bind(&new_bp)
        .bind(&now)
        .bind(file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to update file_id={}: {}", file_id, e))?;

        updated += 1;
    }

    tx.commit().await
        .map_err(|e| format!("Failed to commit batch branch update: {}", e))?;

    Ok(updated)
}

/// Update last_commit_hash in watch_folders.
async fn update_last_commit_hash(
    pool: &SqlitePool,
    watch_folder_id: &str,
    commit_hash: &str,
) -> Result<(), String> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE watch_folders SET last_commit_hash = ?1, updated_at = ?2 WHERE watch_id = ?3"
    )
    .bind(commit_hash)
    .bind(&now)
    .bind(watch_folder_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to update last_commit_hash: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use sqlx::sqlite::SqlitePoolOptions;
    use crate::watch_folders_schema;
    use crate::tracked_files_schema::{CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL};
    use crate::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL};

    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    async fn setup_tables(pool: &SqlitePool) {
        sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await.unwrap();
        sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(pool).await.unwrap();
        sqlx::query(CREATE_TRACKED_FILES_SQL).execute(pool).await.unwrap();
        for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await.unwrap();
        for idx in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }
    }

    async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, tenant_id: &str, path: &str) {
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
             VALUES (?1, ?2, 'projects', ?3, 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(watch_id)
        .bind(path)
        .bind(tenant_id)
        .execute(pool).await.unwrap();
    }

    async fn insert_tracked_file(
        pool: &SqlitePool,
        watch_id: &str,
        file_path: &str,
        branch: &str,
        file_hash: &str,
        relative_path: &str,
        base_point: &str,
    ) {
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash,
             collection, base_point, relative_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, '2025-01-01T00:00:00Z', ?4, 'projects', ?5, ?6, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(watch_id)
        .bind(file_path)
        .bind(branch)
        .bind(file_hash)
        .bind(base_point)
        .bind(relative_path)
        .execute(pool).await.unwrap();
    }

    #[test]
    fn test_branch_switch_stats_default() {
        let stats = BranchSwitchStats::default();
        assert_eq!(stats.batch_updated, 0);
        assert_eq!(stats.enqueued_changed, 0);
        assert_eq!(stats.enqueued_added, 0);
        assert_eq!(stats.enqueued_deleted, 0);
        assert_eq!(stats.errors, 0);
    }

    #[tokio::test]
    async fn test_batch_update_branch_basic() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let tenant = "t1";
        let watch_id = "w1";
        insert_watch_folder(&pool, watch_id, tenant, "/tmp/project").await;

        // Insert 3 files on branch "main"
        let bp1 = wqm_common::hashing::compute_base_point(tenant, "main", "src/a.rs", "hash_a");
        let bp2 = wqm_common::hashing::compute_base_point(tenant, "main", "src/b.rs", "hash_b");
        let bp3 = wqm_common::hashing::compute_base_point(tenant, "main", "src/c.rs", "hash_c");

        insert_tracked_file(&pool, watch_id, "src/a.rs", "main", "hash_a", "src/a.rs", &bp1).await;
        insert_tracked_file(&pool, watch_id, "src/b.rs", "main", "hash_b", "src/b.rs", &bp2).await;
        insert_tracked_file(&pool, watch_id, "src/c.rs", "main", "hash_c", "src/c.rs", &bp3).await;

        // File b.rs changed on the target branch
        let mut changed = HashSet::new();
        changed.insert("src/b.rs".to_string());

        // Batch update from "main" to "feature"
        let count = batch_update_branch(&pool, watch_id, "main", "feature", &changed)
            .await.unwrap();

        // Only 2 files should be updated (a.rs, c.rs — b.rs is in changed set)
        assert_eq!(count, 2);

        // Verify branch was updated for unchanged files
        let branch_a: String = sqlx::query_scalar(
            "SELECT branch FROM tracked_files WHERE file_path = 'src/a.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(branch_a, "feature");

        let branch_c: String = sqlx::query_scalar(
            "SELECT branch FROM tracked_files WHERE file_path = 'src/c.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(branch_c, "feature");

        // b.rs should still be on "main"
        let branch_b: String = sqlx::query_scalar(
            "SELECT branch FROM tracked_files WHERE file_path = 'src/b.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(branch_b, "main");

        // Verify base_point was recomputed for updated files
        let new_bp_a = wqm_common::hashing::compute_base_point(tenant, "feature", "src/a.rs", "hash_a");
        let stored_bp: String = sqlx::query_scalar(
            "SELECT base_point FROM tracked_files WHERE file_path = 'src/a.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(stored_bp, new_bp_a);
    }

    #[tokio::test]
    async fn test_batch_update_branch_empty_changed_set() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let tenant = "t1";
        let watch_id = "w1";
        insert_watch_folder(&pool, watch_id, tenant, "/tmp/project").await;

        let bp = wqm_common::hashing::compute_base_point(tenant, "main", "src/a.rs", "hash_a");
        insert_tracked_file(&pool, watch_id, "src/a.rs", "main", "hash_a", "src/a.rs", &bp).await;

        let changed = HashSet::new();
        let count = batch_update_branch(&pool, watch_id, "main", "dev", &changed)
            .await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_batch_update_branch_no_files() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        insert_watch_folder(&pool, "w1", "t1", "/tmp/empty").await;

        let changed = HashSet::new();
        let count = batch_update_branch(&pool, "w1", "main", "dev", &changed)
            .await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_update_last_commit_hash() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        insert_watch_folder(&pool, "w1", "t1", "/tmp/project").await;

        update_last_commit_hash(&pool, "w1", "abc123def456").await.unwrap();

        let hash: Option<String> = sqlx::query_scalar(
            "SELECT last_commit_hash FROM watch_folders WHERE watch_id = 'w1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(hash.as_deref(), Some("abc123def456"));
    }

    #[tokio::test]
    async fn test_enqueue_file_op() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let qm = QueueManager::new(pool.clone());

        enqueue_file_op(&qm, "t1", "projects", "/tmp/project/src/main.rs", QueueOperation::Update, "main")
            .await.unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 't1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(count, 1);

        let op: String = sqlx::query_scalar(
            "SELECT op FROM unified_queue WHERE tenant_id = 't1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(op, "update");
    }
}
