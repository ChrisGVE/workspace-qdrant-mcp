//! Remote URL Change Detection and Cascade Rename
//!
//! Monitors active projects for git remote URL changes and triggers cascade renames
//! when a project's remote changes. This keeps Qdrant tenant_ids in sync with the
//! actual git remote, handling scenarios such as:
//!
//! - Repository transfer to a new GitHub organization
//! - Switching from HTTPS to SSH remote (or vice versa, if normalization changes)
//! - Repository rename on the hosting platform
//!
//! The detection runs during the daemon polling cycle. When a change is detected:
//! 1. SQLite `watch_folders` are updated atomically (tenant_id, git_remote_url, remote_hash)
//! 2. A cascade rename queue item is enqueued to update Qdrant point payloads

use git2::Repository;
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use wqm_common::constants::COLLECTION_PROJECTS;
use crate::project_disambiguation::ProjectIdCalculator;
use crate::queue_operations::QueueManager;

/// Result of a single remote URL change detection cycle
#[derive(Debug, Default)]
pub struct RemoteCheckResult {
    /// Number of projects checked
    pub projects_checked: u32,
    /// Number of remote changes detected and processed
    pub changes_detected: u32,
    /// Number of projects skipped due to errors (git failures, etc.)
    pub errors: u32,
}

/// Check all active projects for git remote URL changes.
///
/// For each active, non-archived project watch folder:
/// 1. Read current git remote URL from filesystem using git2
/// 2. Compare (normalized) with stored `git_remote_url`
/// 3. If changed: update SQLite and enqueue Qdrant cascade rename
///
/// # Arguments
///
/// * `pool` - SQLite connection pool
/// * `queue_manager` - Queue manager for enqueuing cascade renames
///
/// # Returns
///
/// Summary of the check cycle
pub async fn check_remote_url_changes(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<RemoteCheckResult, String> {
    let mut result = RemoteCheckResult::default();

    // Query active, non-archived, top-level project watch_folders that have git remotes
    let watches: Vec<(String, String, String, String, Option<String>, Option<String>)> =
        sqlx::query_as(
            r#"
            SELECT watch_id, path, tenant_id, git_remote_url,
                   remote_hash, disambiguation_path
            FROM watch_folders
            WHERE is_active = 1
              AND COALESCE(is_archived, 0) = 0
              AND collection = ?1
              AND parent_watch_id IS NULL
              AND git_remote_url IS NOT NULL
            "#,
        )
        .bind(COLLECTION_PROJECTS)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to query watch_folders: {}", e))?;

    let calculator = ProjectIdCalculator::new();

    for (watch_id, path, old_tenant_id, stored_url, _stored_hash, disambiguation_path) in &watches
    {
        result.projects_checked += 1;

        // Read current git remote URL from filesystem
        let current_url = match get_git_remote_url(path) {
            Ok(url) => url,
            Err(e) => {
                debug!(
                    "Skipping remote check for {} ({}): {}",
                    watch_id, path, e
                );
                result.errors += 1;
                continue;
            }
        };

        // Normalize both for comparison
        let normalized_stored = ProjectIdCalculator::normalize_git_url(stored_url);
        let normalized_current = ProjectIdCalculator::normalize_git_url(&current_url);

        if normalized_stored == normalized_current {
            continue; // No change
        }

        info!(
            "Remote URL changed for watch_id={}: {} -> {}",
            watch_id, normalized_stored, normalized_current
        );

        // Compute new tenant_id and remote_hash
        let new_remote_hash = calculator.calculate_remote_hash(&current_url);
        let new_tenant_id = calculator.calculate(
            std::path::Path::new(path),
            Some(&current_url),
            disambiguation_path.as_deref(),
        );

        // Update SQLite atomically: watch_folders (parent + submodules)
        if let Err(e) = update_watch_folders_remote(
            pool,
            watch_id,
            &current_url,
            &new_remote_hash,
            &new_tenant_id,
        )
        .await
        {
            warn!(
                "Failed to update watch_folders for {}: {}",
                watch_id, e
            );
            result.errors += 1;
            continue;
        }

        // Enqueue Qdrant cascade rename (update tenant_id in point payloads)
        let reason = format!(
            "Remote URL changed: {} -> {}",
            normalized_stored, normalized_current
        );
        match queue_manager
            .enqueue_cascade_rename(
                old_tenant_id,
                &new_tenant_id,
                &["projects", "memory"],
                &reason,
            )
            .await
        {
            Ok(queue_ids) => {
                info!(
                    "Enqueued {} cascade rename(s) for tenant {} -> {}",
                    queue_ids.len(),
                    old_tenant_id,
                    new_tenant_id
                );
            }
            Err(e) => {
                // SQLite is already updated - log error but don't fail
                warn!(
                    "Failed to enqueue cascade rename for {} -> {}: {}",
                    old_tenant_id, new_tenant_id, e
                );
            }
        }

        result.changes_detected += 1;
    }

    Ok(result)
}

/// Read the current git remote URL from a repository path using git2.
///
/// Tries "origin" first, then "upstream", then any available remote.
fn get_git_remote_url(repo_path: &str) -> Result<String, String> {
    let repo = Repository::open(repo_path)
        .map_err(|e| format!("Not a git repository: {}", e))?;

    // Try origin first, then upstream
    let remote = repo
        .find_remote("origin")
        .or_else(|_| repo.find_remote("upstream"))
        .map_err(|_| "No origin or upstream remote found".to_string())?;

    remote
        .url()
        .map(|url| url.to_string())
        .ok_or_else(|| "Remote URL is not valid UTF-8".to_string())
}

/// Update watch_folders in SQLite when remote URL changes.
///
/// Updates the parent project and all its submodules in a single transaction.
async fn update_watch_folders_remote(
    pool: &SqlitePool,
    parent_watch_id: &str,
    new_git_remote_url: &str,
    new_remote_hash: &str,
    new_tenant_id: &str,
) -> Result<(), String> {
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let now = wqm_common::timestamps::now_utc();

    // Update parent watch_folder
    let parent_result = sqlx::query(
        r#"
        UPDATE watch_folders
        SET tenant_id = ?1,
            git_remote_url = ?2,
            remote_hash = ?3,
            updated_at = ?4
        WHERE watch_id = ?5
        "#,
    )
    .bind(new_tenant_id)
    .bind(new_git_remote_url)
    .bind(new_remote_hash)
    .bind(&now)
    .bind(parent_watch_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("Failed to update parent watch_folder: {}", e))?;

    if parent_result.rows_affected() == 0 {
        return Err(format!(
            "Parent watch_folder not found: {}",
            parent_watch_id
        ));
    }

    // Update submodules: they share the parent's tenant_id prefix
    // Submodules keep their own git_remote_url but inherit parent's tenant_id
    let sub_result = sqlx::query(
        r#"
        UPDATE watch_folders
        SET tenant_id = ?1,
            updated_at = ?2
        WHERE parent_watch_id = ?3
        "#,
    )
    .bind(new_tenant_id)
    .bind(&now)
    .bind(parent_watch_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("Failed to update submodule watch_folders: {}", e))?;

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit transaction: {}", e))?;

    info!(
        "Updated watch_folders: parent={}, submodules={}",
        parent_watch_id,
        sub_result.rows_affected()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::Path;

    /// Helper to create in-memory SQLite database with watch_folders schema
    async fn create_test_database() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory database");

        sqlx::query(
            r#"
            CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
                tenant_id TEXT NOT NULL,
                parent_watch_id TEXT,
                is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1)),
                is_archived INTEGER DEFAULT 0 CHECK (is_archived IN (0, 1)),
                git_remote_url TEXT,
                remote_hash TEXT,
                disambiguation_path TEXT,
                patterns TEXT NOT NULL DEFAULT '[]',
                ignore_patterns TEXT NOT NULL DEFAULT '[]',
                auto_ingest BOOLEAN NOT NULL DEFAULT 1,
                recursive BOOLEAN NOT NULL DEFAULT 1,
                recursive_depth INTEGER NOT NULL DEFAULT 10,
                debounce_seconds REAL NOT NULL DEFAULT 2.0,
                enabled BOOLEAN NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_activity_at TEXT,
                FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&pool)
        .await
        .expect("Failed to create watch_folders table");

        // Also create unified_queue for QueueManager
        sqlx::query(
            r#"
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                lease_until TEXT,
                worker_id TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL DEFAULT '{}',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                error_message TEXT,
                last_error_at TEXT,
                branch TEXT DEFAULT 'main',
                metadata TEXT DEFAULT '{}',
                file_path TEXT UNIQUE
            )
            "#,
        )
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

        pool
    }

    /// Create a real git repo with a remote set
    fn create_git_repo_with_remote(dir: &Path, remote_url: &str) {
        let repo = Repository::init(dir).expect("Failed to init git repo");
        repo.remote("origin", remote_url)
            .expect("Failed to set remote");
    }

    #[tokio::test]
    async fn test_get_git_remote_url() {
        let temp = TempDir::new().unwrap();
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/user/repo.git",
        );

        let url = get_git_remote_url(temp.path().to_str().unwrap()).unwrap();
        assert_eq!(url, "https://github.com/user/repo.git");
    }

    #[tokio::test]
    async fn test_get_git_remote_url_not_git() {
        let temp = TempDir::new().unwrap();
        let result = get_git_remote_url(temp.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_watch_folders_remote() {
        let pool = create_test_database().await;

        // Insert parent project
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                git_remote_url, remote_hash, disambiguation_path)
            VALUES ('proj-1', '/tmp/repo', 'projects', 'old_tenant', 1,
                'https://github.com/old/repo.git', 'oldhash12345', NULL)
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert submodule
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                parent_watch_id, git_remote_url, remote_hash)
            VALUES ('sub-1', '/tmp/repo/lib', 'projects', 'old_tenant', 1,
                'proj-1', 'https://github.com/lib/sub.git', 'subhash12345')
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        // Execute remote update
        update_watch_folders_remote(
            &pool,
            "proj-1",
            "https://github.com/new/repo.git",
            "newhash12345",
            "new_tenant",
        )
        .await
        .unwrap();

        // Verify parent updated
        let (tid, url, hash): (String, String, String) = sqlx::query_as(
            "SELECT tenant_id, git_remote_url, remote_hash FROM watch_folders WHERE watch_id = 'proj-1'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(tid, "new_tenant");
        assert_eq!(url, "https://github.com/new/repo.git");
        assert_eq!(hash, "newhash12345");

        // Verify submodule tenant_id updated (but git_remote_url unchanged)
        let (sub_tid, sub_url): (String, String) = sqlx::query_as(
            "SELECT tenant_id, git_remote_url FROM watch_folders WHERE watch_id = 'sub-1'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(sub_tid, "new_tenant");
        assert_eq!(sub_url, "https://github.com/lib/sub.git"); // unchanged
    }

    #[tokio::test]
    async fn test_check_remote_url_changes_no_change() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Create git repo with known remote
        let temp = TempDir::new().unwrap();
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/user/repo.git",
        );

        // Insert watch_folder with same remote (normalized)
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                git_remote_url, remote_hash)
            VALUES ('proj-1', ?1, 'projects', 'some_tenant', 1,
                'https://github.com/user/repo.git', 'somehash12345')
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_remote_url_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.changes_detected, 0);
        assert_eq!(result.errors, 0);
    }

    #[tokio::test]
    async fn test_check_remote_url_changes_detects_change() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Create git repo with NEW remote
        let temp = TempDir::new().unwrap();
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/new-org/repo.git",
        );

        let calculator = ProjectIdCalculator::new();

        // Insert watch_folder with OLD remote
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                git_remote_url, remote_hash)
            VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
                'https://github.com/old-org/repo.git', 'oldhash12345')
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_remote_url_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.changes_detected, 1);
        assert_eq!(result.errors, 0);

        // Verify watch_folder was updated
        let (tid, url): (String, String) = sqlx::query_as(
            "SELECT tenant_id, git_remote_url FROM watch_folders WHERE watch_id = 'proj-1'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(url, "https://github.com/new-org/repo.git");
        // Verify tenant_id changed
        assert_ne!(tid, "old_tenant");
        // Verify it matches expected calculation
        let expected_tid = calculator.calculate(
            temp.path(),
            Some("https://github.com/new-org/repo.git"),
            None,
        );
        assert_eq!(tid, expected_tid);

        // Verify cascade rename was enqueued
        let count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'rename'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert!(count >= 1, "Expected at least 1 cascade rename queue item, got {}", count);
    }

    #[tokio::test]
    async fn test_check_remote_url_changes_skips_archived() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Create git repo with different remote
        let temp = TempDir::new().unwrap();
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/new-org/repo.git",
        );

        // Insert archived watch_folder with old remote
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_archived, git_remote_url, remote_hash)
            VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
                1, 'https://github.com/old-org/repo.git', 'oldhash')
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_remote_url_changes(&pool, &queue_manager)
            .await
            .unwrap();

        // Archived project should not be checked
        assert_eq!(result.projects_checked, 0);
        assert_eq!(result.changes_detected, 0);
    }

    #[tokio::test]
    async fn test_check_remote_url_changes_preserves_disambiguation() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Create git repo with NEW remote
        let temp = TempDir::new().unwrap();
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/new-org/repo.git",
        );

        let calculator = ProjectIdCalculator::new();

        // Insert watch_folder with OLD remote AND disambiguation_path
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                git_remote_url, remote_hash, disambiguation_path)
            VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
                'https://github.com/old-org/repo.git', 'oldhash12345', 'work/clone1')
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_remote_url_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.changes_detected, 1);

        // Verify new tenant_id preserves disambiguation_path
        let tid: String = sqlx::query_scalar(
            "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-1'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let expected_tid = calculator.calculate(
            temp.path(),
            Some("https://github.com/new-org/repo.git"),
            Some("work/clone1"),
        );
        assert_eq!(tid, expected_tid);
    }
}
