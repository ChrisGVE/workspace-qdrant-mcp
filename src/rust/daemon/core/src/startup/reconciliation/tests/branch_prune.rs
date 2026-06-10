//! Tests for `branch_prune::prune_orphaned_branches`.
//!
//! Verifies that documents for branches no longer present in git are enqueued
//! for deletion, while documents for live branches are left untouched — and
//! that an unreadable/non-git path is skipped rather than wiped.

use std::path::Path;
use std::sync::Arc;

use git2::{Repository, Signature};
use tempfile::TempDir;

use crate::queue_operations::QueueManager;

use super::super::branch_prune::prune_orphaned_branches;
use super::{create_test_pool, setup_schema};

/// Initialise a git repo with one commit, then create the named local branches.
fn init_repo(dir: &Path, extra_branches: &[&str]) {
    let repo = Repository::init(dir).expect("git init");
    let sig = Signature::now("Test", "test@example.com").unwrap();
    let tree_id = {
        let mut index = repo.index().unwrap();
        index.write_tree().unwrap()
    };
    let tree = repo.find_tree(tree_id).unwrap();
    let commit_id = repo
        .commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
        .unwrap();
    let commit = repo.find_commit(commit_id).unwrap();
    for b in extra_branches {
        repo.branch(b, &commit, true).unwrap();
    }
}

async fn insert_watch_folder(pool: &sqlx::SqlitePool, watch_id: &str, tenant: &str, path: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES (?1, ?2, 'projects', ?3, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_id)
    .bind(path)
    .bind(tenant)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_tracked_file(
    pool: &sqlx::SqlitePool,
    watch_id: &str,
    branch: &str,
    relative_path: &str,
) {
    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, branch, file_mtime, file_hash, relative_path, collection, created_at, updated_at) \
         VALUES (?1, ?2, '2025-01-01T00:00:00Z', 'hash', ?3, 'projects', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_id)
    .bind(branch)
    .bind(relative_path)
    .execute(pool)
    .await
    .unwrap();
}

async fn file_delete_count(pool: &sqlx::SqlitePool) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'file' AND op = 'delete'",
    )
    .fetch_one(pool)
    .await
    .unwrap()
}

#[tokio::test]
async fn prunes_genuine_orphan_feature_branch() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = Arc::new(QueueManager::new(pool.clone()));

    let repo_dir = TempDir::new().unwrap();
    init_repo(repo_dir.path(), &["keep-a", "keep-b"]);
    let repo_path = repo_dir.path().to_str().unwrap();

    insert_watch_folder(&pool, "w1", "t1", repo_path).await;
    // Live branches (present in git) — must be preserved. keep-a is the largest
    // tracked branch, so it is the corpus (never pruned regardless).
    insert_tracked_file(&pool, "w1", "keep-a", "src/a.rs").await;
    insert_tracked_file(&pool, "w1", "keep-a", "src/b.rs").await;
    insert_tracked_file(&pool, "w1", "keep-a", "src/c.rs").await;
    insert_tracked_file(&pool, "w1", "keep-b", "src/a.rs").await;
    // A genuine deleted feature branch: absent from git, a minor offshoot (fewer
    // files than the corpus), not a default name → pruned.
    insert_tracked_file(&pool, "w1", "feature/gone", "src/a.rs").await;
    insert_tracked_file(&pool, "w1", "feature/gone", "src/d.rs").await;

    let stats = prune_orphaned_branches(&pool, &qm).await.expect("prune");

    assert_eq!(stats.branches_pruned, 1, "only 'feature/gone' is orphaned");
    assert_eq!(stats.files_enqueued, 2, "2 files on 'feature/gone'");
    assert_eq!(
        file_delete_count(&pool).await,
        2,
        "exactly the 2 'feature/gone' files enqueued; live branches untouched"
    );
}

/// Regression for the bws-engineer / compress-mcp incident: a project's corpus
/// mislabeled under a non-existent branch must NEVER be deleted. The largest
/// tracked branch is protected even when absent from git.
#[tokio::test]
async fn never_prunes_largest_branch_even_if_absent_from_git() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = Arc::new(QueueManager::new(pool.clone()));

    let repo_dir = TempDir::new().unwrap();
    init_repo(repo_dir.path(), &["dev-clean"]);
    let repo_path = repo_dir.path().to_str().unwrap();

    insert_watch_folder(&pool, "w1", "t1", repo_path).await;
    // Corpus mislabeled under "ghost-default" (NOT a git branch), the largest set.
    for f in ["a", "b", "c", "d", "e"] {
        insert_tracked_file(&pool, "w1", "ghost-default", &format!("src/{f}.rs")).await;
    }
    // A small live branch present in git.
    insert_tracked_file(&pool, "w1", "dev-clean", "src/a.rs").await;

    let stats = prune_orphaned_branches(&pool, &qm).await.expect("prune");

    assert_eq!(stats.branches_pruned, 0, "largest branch is never pruned");
    assert_eq!(file_delete_count(&pool).await, 0, "no deletes enqueued");
}

/// A branch literally named `main`/`master` is never pruned even when git has no
/// such branch — the exact shape of the incident (content under fallback "main").
#[tokio::test]
async fn never_prunes_main_or_master_even_if_absent() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = Arc::new(QueueManager::new(pool.clone()));

    let repo_dir = TempDir::new().unwrap();
    init_repo(repo_dir.path(), &["dev-clean"]); // git default + dev-clean; no "main"
    let repo_path = repo_dir.path().to_str().unwrap();

    insert_watch_folder(&pool, "w1", "t1", repo_path).await;
    // "main" is absent from git but is a default name AND would be the corpus —
    // doubly protected. Add a larger live branch so it is NOT the largest, to
    // prove the name guard alone protects it.
    insert_tracked_file(&pool, "w1", "main", "src/a.rs").await;
    insert_tracked_file(&pool, "w1", "dev-clean", "src/a.rs").await;
    insert_tracked_file(&pool, "w1", "dev-clean", "src/b.rs").await;
    insert_tracked_file(&pool, "w1", "dev-clean", "src/c.rs").await;

    let stats = prune_orphaned_branches(&pool, &qm).await.expect("prune");

    assert_eq!(stats.branches_pruned, 0, "'main' is name-protected");
    assert_eq!(file_delete_count(&pool).await, 0, "no deletes enqueued");
}

#[tokio::test]
async fn skips_non_git_path_without_pruning() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = Arc::new(QueueManager::new(pool.clone()));

    // A real directory that is NOT a git repo — list_all_branches errors, so the
    // project must be skipped entirely (never prune on uncertain ground truth).
    let plain_dir = TempDir::new().unwrap();
    insert_watch_folder(&pool, "w1", "t1", plain_dir.path().to_str().unwrap()).await;
    insert_tracked_file(&pool, "w1", "some-branch", "src/a.rs").await;

    let stats = prune_orphaned_branches(&pool, &qm).await.expect("prune");

    assert_eq!(stats.branches_pruned, 0, "non-git path must be skipped");
    assert_eq!(file_delete_count(&pool).await, 0, "nothing enqueued");
}
