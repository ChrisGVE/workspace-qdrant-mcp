//! Git Integration Tests
//!
//! Comprehensive integration tests for Git-related features:
//! - Branch lifecycle detection (create, delete, rename)
//! - Project ID disambiguation for multi-clone scenarios
//! - Default branch detection and tracking
//! - Alias management for project ID transitions

use std::path::Path;
use tempfile::TempDir;
use git2::{Repository, Signature};
use tokio::time::Duration;

use workspace_qdrant_core::git_integration::{
    GitBranchDetector, BranchLifecycleDetector, BranchLifecycleConfig,
    BranchEvent,
};
use workspace_qdrant_core::project_disambiguation::{
    ProjectIdCalculator, DisambiguationPathComputer,
};

// ============================================================================
// Test Utilities
// ============================================================================

/// Helper to create a Git repository with initial commit
fn create_test_repo(path: &Path) -> Result<Repository, git2::Error> {
    let repo = Repository::init(path)?;

    // Create initial commit (required for branch to exist)
    let sig = Signature::now("Test User", "test@example.com")?;
    let tree_id = {
        let mut index = repo.index()?;
        index.write_tree()?
    };
    {
        let tree = repo.find_tree(tree_id)?;
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        )?;
    }

    Ok(repo)
}

/// Create a test repo with multiple branches
fn create_multi_branch_repo(path: &Path) -> Result<Repository, git2::Error> {
    let repo = create_test_repo(path)?;

    // Get HEAD commit OID
    let head_oid = {
        let head = repo.head()?;
        let commit = head.peel_to_commit()?;
        commit.id()
    };

    // Create additional branches using the OID in a scope
    {
        let commit = repo.find_commit(head_oid)?;
        repo.branch("develop", &commit, false)?;
        repo.branch("feature/test", &commit, false)?;
        repo.branch("release/1.0", &commit, false)?;
    }

    Ok(repo)
}

/// Add a remote to a repository
fn add_remote(repo: &Repository, name: &str, url: &str) -> Result<(), git2::Error> {
    repo.remote(name, url)?;
    Ok(())
}

// ============================================================================
// Branch Lifecycle Integration Tests
// ============================================================================

#[tokio::test]
async fn test_branch_lifecycle_full_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    // Create repository with initial branch
    let repo = create_test_repo(repo_path).unwrap();

    // Initialize detector
    let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
    detector.initialize().await.unwrap();

    // Verify initial state
    let branches = detector.get_tracked_branches().await;
    assert!(!branches.is_empty(), "Should have at least one branch");

    let default_branch = detector.get_default_branch().await;
    assert!(default_branch.is_some(), "Should have a default branch");

    // Create a new branch
    let head = repo.head().unwrap();
    let commit = head.peel_to_commit().unwrap();
    repo.branch("feature/workflow-test", &commit, false).unwrap();

    // Scan for changes
    let events = detector.scan_for_changes().await.unwrap();

    // Should detect new branch
    let created_event = events.iter().find(|e| {
        matches!(e, BranchEvent::Created { branch, .. } if branch == "feature/workflow-test")
    });
    assert!(created_event.is_some(), "Should detect branch creation");

    // Delete the branch
    let mut branch = repo.find_branch("feature/workflow-test", git2::BranchType::Local).unwrap();
    branch.delete().unwrap();

    // First scan registers the deletion
    detector.scan_for_changes().await.unwrap();

    // Wait for rename correlation timeout
    tokio::time::sleep(Duration::from_millis(600)).await;

    // Second scan emits delete event
    let events = detector.scan_for_changes().await.unwrap();

    let deleted_event = events.iter().find(|e| {
        matches!(e, BranchEvent::Deleted { branch } if branch == "feature/workflow-test")
    });
    assert!(deleted_event.is_some(), "Should detect branch deletion");
}

#[tokio::test]
async fn test_branch_rename_detection() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    let repo = create_test_repo(repo_path).unwrap();

    // Create a branch
    let head = repo.head().unwrap();
    let commit = head.peel_to_commit().unwrap();
    repo.branch("old-name", &commit, false).unwrap();

    // Initialize detector after branch exists
    let config = BranchLifecycleConfig {
        rename_correlation_timeout_ms: 2000, // Longer timeout for test
        ..Default::default()
    };
    let detector = BranchLifecycleDetector::new(repo_path.to_path_buf(), config);
    detector.initialize().await.unwrap();

    // "Rename" by creating new branch at same commit and deleting old
    repo.branch("new-name", &commit, false).unwrap();

    // Delete old branch
    let mut old_branch = repo.find_branch("old-name", git2::BranchType::Local).unwrap();
    old_branch.delete().unwrap();

    // Scan should detect this as rename (delete+create same commit within timeout)
    let events = detector.scan_for_changes().await.unwrap();

    // Should have either a Renamed event or separate Create+Delete
    let has_rename = events.iter().any(|e| {
        matches!(e, BranchEvent::Renamed { old_name, new_name }
            if old_name == "old-name" && new_name == "new-name")
    });
    let has_create_delete = events.iter().any(|e| {
        matches!(e, BranchEvent::Created { branch, .. } if branch == "new-name")
    });

    assert!(has_rename || has_create_delete,
        "Should detect branch rename or create/delete pair");
}

#[tokio::test]
async fn test_branch_lifecycle_stats() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    let _repo = create_multi_branch_repo(repo_path).unwrap();

    let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
    detector.initialize().await.unwrap();

    let stats = detector.stats().await;

    // Should have at least 4 branches: main/master + develop + feature/test + release/1.0
    assert!(stats.tracked_branches >= 4,
        "Expected at least 4 branches, got {}", stats.tracked_branches);
    assert_eq!(stats.pending_deletes, 0);
    assert!(stats.default_branch.is_some());
}

// ============================================================================
// Project ID Disambiguation Integration Tests
// ============================================================================

#[test]
fn test_project_id_calculator_integration() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    let repo = create_test_repo(repo_path).unwrap();
    add_remote(&repo, "origin", "https://github.com/user/test-repo.git").unwrap();

    let calculator = ProjectIdCalculator::new();

    // Calculate ID with remote
    let id = calculator.calculate(
        repo_path,
        Some("https://github.com/user/test-repo.git"),
        None,
    );

    // Should be a 12-character hash (not local_)
    assert_eq!(id.len(), 12);
    assert!(!id.starts_with("local_"));

    // Same remote should give same ID
    let id2 = calculator.calculate(
        repo_path,
        Some("git@github.com:user/test-repo.git"), // SSH format
        None,
    );

    // Both should normalize to same ID
    assert_eq!(id, id2, "Different URL formats should produce same ID");
}

#[test]
fn test_disambiguation_path_for_clones() {
    // Simulate two clones of the same repo
    let path1 = std::path::PathBuf::from("/home/user/work/my-project");
    let path2 = std::path::PathBuf::from("/home/user/personal/my-project");

    let existing = vec![path1.clone()];
    let disambig = DisambiguationPathComputer::compute(&path2, &existing);

    // Should compute disambiguation path from the differing point
    assert_eq!(disambig, "personal/my-project");

    // Verify disambiguation produces different project IDs
    let calculator = ProjectIdCalculator::new();

    let id1 = calculator.calculate(
        &path1,
        Some("https://github.com/user/my-project.git"),
        Some("work/my-project"),
    );

    let id2 = calculator.calculate(
        &path2,
        Some("https://github.com/user/my-project.git"),
        Some("personal/my-project"),
    );

    assert_ne!(id1, id2, "Different disambiguation should produce different IDs");
}

#[test]
fn test_local_project_id_generation() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    // Create a repo without remote
    let _repo = create_test_repo(repo_path).unwrap();

    let calculator = ProjectIdCalculator::new();

    // Calculate ID without remote
    let id = calculator.calculate(
        repo_path,
        None, // No remote
        None,
    );

    // Should have local_ prefix
    assert!(id.starts_with("local_"), "Local project should have local_ prefix");
    assert_eq!(id.len(), 18, "local_ (6) + 12-char hash = 18 characters");
}

// NOTE: test_project_alias_creation removed - AliasManager has been deprecated.
// Project aliases are no longer needed with the unified watch_folders table.

#[test]
fn test_git_url_normalization_comprehensive() {
    let test_cases = vec![
        ("https://github.com/user/repo.git", "github.com/user/repo"),
        ("git@github.com:user/repo.git", "github.com/user/repo"),
        ("ssh://git@github.com/user/repo.git", "github.com/user/repo"),
        ("http://github.com/user/repo", "github.com/user/repo"),
        ("https://gitlab.com/org/project.git", "gitlab.com/org/project"),
        ("git@bitbucket.org:team/app.git", "bitbucket.org/team/app"),
        ("HTTPS://GitHub.COM/User/Repo.GIT", "github.com/user/repo"),
    ];

    for (input, expected) in test_cases {
        let normalized = ProjectIdCalculator::normalize_git_url(input);
        assert_eq!(normalized, expected, "Failed for input: {}", input);
    }
}

#[test]
fn test_remote_hash_grouping() {
    let calculator = ProjectIdCalculator::new();

    // All variations of the same repo should produce same remote hash
    let urls = vec![
        "https://github.com/anthropics/claude-code.git",
        "git@github.com:anthropics/claude-code.git",
        "ssh://git@github.com/anthropics/claude-code",
        "http://github.com/anthropics/claude-code",
    ];

    let hashes: Vec<_> = urls.iter()
        .map(|u| calculator.calculate_remote_hash(u))
        .collect();

    // All hashes should be identical
    for (i, hash) in hashes.iter().enumerate().skip(1) {
        assert_eq!(hash, &hashes[0],
            "URL {} produced different hash than URL 0", i);
    }
}

// ============================================================================
// Cross-Feature Integration Tests
// ============================================================================

#[tokio::test]
async fn test_branch_detector_with_disambiguation() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    // Create repo with remote
    let repo = create_test_repo(repo_path).unwrap();
    add_remote(&repo, "origin", "https://github.com/user/test-repo.git").unwrap();

    // Calculate project ID
    let calculator = ProjectIdCalculator::new();
    let project_id = calculator.calculate(
        repo_path,
        Some("https://github.com/user/test-repo.git"),
        None,
    );

    // Initialize branch detector
    let branch_detector = GitBranchDetector::new();
    let branch = branch_detector.get_current_branch(repo_path).await.unwrap();

    // Verify we have both project ID and branch
    assert!(!project_id.is_empty(), "Project ID should not be empty");
    assert!(branch == "main" || branch == "master",
        "Expected default branch, got {}", branch);

    // Now initialize lifecycle detector
    let lifecycle_detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
    lifecycle_detector.initialize().await.unwrap();

    let default_branch = lifecycle_detector.get_default_branch().await;
    assert_eq!(default_branch.as_deref(), Some(branch.as_str()),
        "Lifecycle detector should agree on default branch");
}

#[tokio::test]
async fn test_event_channel_integration() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    let repo = create_test_repo(repo_path).unwrap();

    // Create channel for events
    let (tx, mut rx) = tokio::sync::mpsc::channel::<BranchEvent>(10);

    // Initialize detector with channel
    let mut detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
    detector.set_event_sender(tx);
    detector.initialize().await.unwrap();

    // Create a branch
    let head = repo.head().unwrap();
    let commit = head.peel_to_commit().unwrap();
    repo.branch("channel-test", &commit, false).unwrap();

    // Scan for changes (should send event to channel)
    detector.scan_for_changes().await.unwrap();

    // Try to receive event
    let event = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await;

    assert!(event.is_ok(), "Should receive event within timeout");
    if let Ok(Some(event)) = event {
        assert!(matches!(event, BranchEvent::Created { branch, .. } if branch == "channel-test"));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_non_git_directory_error() {
    let temp_dir = TempDir::new().unwrap();
    // Don't initialize git - just a regular directory

    let detector = BranchLifecycleDetector::with_defaults(temp_dir.path().to_path_buf());
    let result = detector.initialize().await;

    assert!(result.is_err(), "Should fail for non-git directory");
}

#[test]
fn test_calculator_with_invalid_path() {
    let calculator = ProjectIdCalculator::new();

    // Non-existent path should still produce a local ID
    let id = calculator.calculate(
        std::path::Path::new("/nonexistent/path/to/project"),
        None,
        None,
    );

    assert!(id.starts_with("local_"), "Should produce local ID for non-existent path");
}
