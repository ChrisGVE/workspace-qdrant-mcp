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

    // Update submodules via junction table (Task 14): they share the parent's tenant_id prefix
    // Submodules keep their own git_remote_url but inherit parent's tenant_id
    let sub_result = sqlx::query(
        r#"
        UPDATE watch_folders
        SET tenant_id = ?1,
            updated_at = ?2
        WHERE watch_id IN (
            SELECT child_watch_id FROM watch_folder_submodules
            WHERE parent_watch_id = ?3
        )
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

// ========== Git State Change Detection (Transitions 1-5) ==========

/// Result of a git state change detection cycle
#[derive(Debug, Default)]
pub struct GitStateCheckResult {
    /// Number of projects checked
    pub projects_checked: u32,
    /// Number of state transitions detected and processed
    pub transitions_detected: u32,
    /// Number of errors encountered
    pub errors: u32,
}

/// Check all active projects for git state changes (transitions 1-5).
///
/// Detects transitions between three states:
/// - **Local**: no `.git/` directory (`is_git_tracked=0, git_remote_url=NULL`)
/// - **Local Git**: `.git/` exists but no remote (`is_git_tracked=1, git_remote_url=NULL`)
/// - **Remote Git**: `.git/` with remote (`is_git_tracked=1, git_remote_url=URL`)
///
/// Transition 6 (remote URL change) is already handled by `check_remote_url_changes`.
///
/// For each detected transition, updates `watch_folders` and enqueues cascade renames
/// when tenant_id changes.
pub async fn check_git_state_changes(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<GitStateCheckResult, String> {
    let mut result = GitStateCheckResult::default();

    // Query ALL active, non-archived, top-level project watch_folders
    let watches: Vec<(String, String, String, i32, Option<String>, Option<String>)> =
        sqlx::query_as(
            r#"
            SELECT watch_id, path, tenant_id,
                   COALESCE(is_git_tracked, 0) AS is_git_tracked,
                   git_remote_url, disambiguation_path
            FROM watch_folders
            WHERE is_active = 1
              AND COALESCE(is_archived, 0) = 0
              AND collection = ?1
              AND parent_watch_id IS NULL
            "#,
        )
        .bind(COLLECTION_PROJECTS)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to query watch_folders for git state check: {}", e))?;

    let calculator = ProjectIdCalculator::new();

    for (watch_id, path, old_tenant_id, stored_git_tracked, stored_remote, disambiguation_path)
        in &watches
    {
        result.projects_checked += 1;
        let project_path = std::path::Path::new(path.as_str());

        // Detect current filesystem git state
        let git_status = crate::git_integration::detect_git_status(project_path);
        let current_remote = if git_status.is_git {
            get_git_remote_url(path).ok()
        } else {
            None
        };

        let was_git = *stored_git_tracked != 0;
        let is_now_git = git_status.is_git;
        let had_remote = stored_remote.is_some();
        let has_remote_now = current_remote.is_some();

        // Determine which transition, if any, occurred
        let transition = match (was_git, had_remote, is_now_git, has_remote_now) {
            // No change
            (false, false, false, false) => continue,          // local → local
            (true, false, true, false) => continue,            // local-git → local-git
            (true, true, true, true) => continue,              // remote-git → remote-git (URL change handled by check_remote_url_changes)

            // Transition 1: local → local-git (git init)
            (false, _, true, false) => "local → local-git",

            // Transition 2: local → remote-git (git init + remote add)
            (false, _, true, true) => "local → remote-git",

            // Transition 3: local-git → remote-git (remote add)
            (true, false, true, true) => "local-git → remote-git",

            // Transition 4: git → local (rm -rf .git)
            (true, _, false, _) => "git → local",

            // Transition 5: remote-git → local-git (remote remove)
            (true, true, true, false) => "remote-git → local-git",

            // Unexpected states — skip
            _ => {
                debug!(
                    "Unexpected git state for {}: was_git={}, had_remote={}, is_git={}, has_remote={}",
                    watch_id, was_git, had_remote, is_now_git, has_remote_now
                );
                continue;
            }
        };

        info!(
            "Git state transition detected for {}: {} (path={})",
            watch_id, transition, path
        );

        // Compute new tenant_id based on new state
        let new_tenant_id = match &current_remote {
            Some(url) => calculator.calculate(
                project_path,
                Some(url.as_str()),
                disambiguation_path.as_deref(),
            ),
            None => calculator.calculate(project_path, None, None),
        };

        let new_remote_hash = current_remote.as_ref().map(|url| calculator.calculate_remote_hash(url));

        // Determine what to update in SQLite
        let new_is_git_tracked: i32 = if is_now_git { 1 } else { 0 };

        // Update watch_folders atomically
        let now = wqm_common::timestamps::now_utc();
        let update_result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_git_tracked = ?1,
                git_remote_url = ?2,
                remote_hash = ?3,
                tenant_id = ?4,
                updated_at = ?5
            WHERE watch_id = ?6
            "#,
        )
        .bind(new_is_git_tracked)
        .bind(current_remote.as_deref())
        .bind(new_remote_hash.as_deref())
        .bind(&new_tenant_id)
        .bind(&now)
        .bind(watch_id)
        .execute(pool)
        .await;

        if let Err(e) = update_result {
            warn!(
                "Failed to update watch_folders for git state transition {}: {}",
                watch_id, e
            );
            result.errors += 1;
            continue;
        }

        // Enqueue cascade rename if tenant_id changed
        if new_tenant_id != *old_tenant_id {
            let reason = format!("Git state transition: {}", transition);
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
                        "Enqueued {} cascade rename(s) for tenant {} -> {} ({})",
                        queue_ids.len(),
                        old_tenant_id,
                        new_tenant_id,
                        transition
                    );
                }
                Err(e) => {
                    // SQLite already updated — log but don't fail
                    warn!(
                        "Failed to enqueue cascade rename for {} -> {}: {}",
                        old_tenant_id, new_tenant_id, e
                    );
                }
            }
        }

        result.transitions_detected += 1;
    }

    Ok(result)
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
                is_git_tracked INTEGER DEFAULT 0 CHECK (is_git_tracked IN (0, 1)),
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
        // Task 14: Junction table for submodule relationships
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS watch_folder_submodules (
                parent_watch_id TEXT NOT NULL,
                child_watch_id TEXT NOT NULL,
                submodule_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (parent_watch_id, child_watch_id),
                FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE,
                FOREIGN KEY (child_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&pool)
        .await
        .expect("Failed to create watch_folder_submodules table");

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

        // Task 14: Insert junction table row
        sqlx::query(
            "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, submodule_path, created_at) VALUES ('proj-1', 'sub-1', 'lib', datetime('now'))"
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
            "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'"
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

    // ========== Git State Change Detection Tests ==========

    #[tokio::test]
    async fn test_git_state_check_no_change_local() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Create a local (non-git) temp directory
        let temp = TempDir::new().unwrap();

        // Insert watch_folder as local (is_git_tracked=0, no remote)
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_git_tracked, git_remote_url)
            VALUES ('proj-local', ?1, 'projects', 'local_abc123', 1, 0, NULL)
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_git_state_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.transitions_detected, 0);
    }

    #[tokio::test]
    async fn test_git_state_check_local_to_local_git() {
        // Transition 1: local → local-git (git init, no remote)
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        let temp = TempDir::new().unwrap();
        // Create git repo WITHOUT remote
        Repository::init(temp.path()).expect("Failed to init git repo");

        // Insert as local project (stored: is_git_tracked=0)
        let calculator = ProjectIdCalculator::new();
        let local_tid = calculator.calculate(temp.path(), None, None);

        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_git_tracked, git_remote_url)
            VALUES ('proj-1', ?1, 'projects', ?2, 1, 0, NULL)
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .bind(&local_tid)
        .execute(&pool)
        .await
        .unwrap();

        let result = check_git_state_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.transitions_detected, 1);

        // Verify is_git_tracked updated to 1
        let is_git: i32 = sqlx::query_scalar(
            "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'proj-1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(is_git, 1);

        // tenant_id should NOT change (still local_ prefix, same path)
        let tid: String = sqlx::query_scalar(
            "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(tid, local_tid, "Transition 1 should not change tenant_id");
    }

    #[tokio::test]
    async fn test_git_state_check_local_git_to_remote_git() {
        // Transition 3: local-git → remote-git (remote add)
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        let temp = TempDir::new().unwrap();
        // Create git repo WITH remote
        create_git_repo_with_remote(
            temp.path(),
            "https://github.com/user/repo.git",
        );

        // Insert as local-git (stored: is_git_tracked=1, no remote)
        let calculator = ProjectIdCalculator::new();
        let local_tid = calculator.calculate(temp.path(), None, None);

        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_git_tracked, git_remote_url)
            VALUES ('proj-2', ?1, 'projects', ?2, 1, 1, NULL)
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .bind(&local_tid)
        .execute(&pool)
        .await
        .unwrap();

        let result = check_git_state_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.transitions_detected, 1);

        // Verify remote was stored
        let remote: Option<String> = sqlx::query_scalar(
            "SELECT git_remote_url FROM watch_folders WHERE watch_id = 'proj-2'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remote, Some("https://github.com/user/repo.git".to_string()));

        // tenant_id should have changed from local_ to remote-based
        let tid: String = sqlx::query_scalar(
            "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-2'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_ne!(tid, local_tid, "Transition 3 should change tenant_id");
        assert!(!tid.starts_with("local_"), "Should be remote-based tenant_id");

        // Verify cascade rename was enqueued
        let rename_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(rename_count >= 1, "Should have enqueued cascade rename");
    }

    #[tokio::test]
    async fn test_git_state_check_git_to_local() {
        // Transition 4: git → local (rm -rf .git simulated by non-git temp dir)
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        // Plain temp dir (no .git) — simulates .git removal
        let temp = TempDir::new().unwrap();

        let calculator = ProjectIdCalculator::new();
        let remote_tid = calculator.calculate(
            temp.path(),
            Some("https://github.com/user/repo.git"),
            None,
        );

        // Insert as remote-git project
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_git_tracked, git_remote_url, remote_hash)
            VALUES ('proj-3', ?1, 'projects', ?2, 1, 1,
                'https://github.com/user/repo.git', 'somehash')
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .bind(&remote_tid)
        .execute(&pool)
        .await
        .unwrap();

        let result = check_git_state_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 1);
        assert_eq!(result.transitions_detected, 1);

        // Verify is_git_tracked set to 0
        let is_git: i32 = sqlx::query_scalar(
            "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'proj-3'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(is_git, 0);

        // Verify git_remote_url cleared
        let remote: Option<String> = sqlx::query_scalar(
            "SELECT git_remote_url FROM watch_folders WHERE watch_id = 'proj-3'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(remote.is_none(), "Remote URL should be cleared");

        // tenant_id should have changed to local_
        let tid: String = sqlx::query_scalar(
            "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-3'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(tid.starts_with("local_"), "Should be local-based tenant_id after .git removal");

        // Verify cascade rename was enqueued
        let rename_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(rename_count >= 1);
    }

    #[tokio::test]
    async fn test_git_state_check_skips_inactive() {
        let pool = create_test_database().await;
        let queue_manager = QueueManager::new(pool.clone());

        let temp = TempDir::new().unwrap();

        // Insert as inactive project
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
                is_git_tracked)
            VALUES ('proj-inactive', ?1, 'projects', 'local_abc', 0, 0)
            "#,
        )
        .bind(temp.path().to_str().unwrap())
        .execute(&pool)
        .await
        .unwrap();

        let result = check_git_state_changes(&pool, &queue_manager)
            .await
            .unwrap();

        assert_eq!(result.projects_checked, 0);
        assert_eq!(result.transitions_detected, 0);
    }
}
