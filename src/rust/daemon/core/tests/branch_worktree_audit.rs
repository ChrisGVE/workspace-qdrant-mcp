//! Integration tests for the branch / worktree audit (issue #63).
//!
//! Each test drives a real on-disk git fixture through a concrete daemon
//! code path and asserts the observable outcome. Findings per scenario
//! are consolidated in `docs/specs/19-branch-worktree-audit.md`.
//!
//! These tests intentionally avoid spinning up a full daemon (gRPC,
//! queue processor, file watcher) -- they drive pure-ish functions and
//! structs. Scenarios that require full-stack integration are called
//! out as gaps in the audit report rather than faked here.

use std::path::PathBuf;
use std::time::Duration;

use shared_test_utils::git_fixtures::GitFixtures;
use tempfile::tempdir;

use workspace_qdrant_core::git::{
    detect_git_status, find_main_worktree_path, BranchEvent, BranchLifecycleConfig,
    BranchLifecycleDetector,
};
use workspace_qdrant_core::path_validator::{
    PathValidator, PathValidatorConfig, RegisteredProject,
};
use wqm_common::project_id::{DisambiguationPathComputer, ProjectIdCalculator};

// ─── Task 2: worktree detection ────────────────────────────────────────────

#[test]
fn task2_plain_clone_detects_as_main_repo() {
    let fx = GitFixtures::plain_clone().unwrap();
    let status = detect_git_status(&fx.repo_path);
    assert!(status.is_git);
    assert!(!status.is_worktree, ".git is a directory, not a file");
    assert_eq!(status.branch, "main");
    assert!(status.commit_hash.is_some());
}

#[test]
fn task2_linked_worktree_flagged_as_worktree() {
    let fx = GitFixtures::worktree("feature").unwrap();
    let main_status = detect_git_status(&fx.main_path);
    let wt_status = detect_git_status(&fx.worktree_path);

    assert!(main_status.is_git && !main_status.is_worktree);
    assert_eq!(main_status.branch, "main");

    assert!(wt_status.is_git, "worktree must still be recognised as git");
    assert!(
        wt_status.is_worktree,
        "worktree with .git file must set is_worktree=true"
    );
    assert_eq!(wt_status.branch, "feature");
}

#[test]
fn task2_nested_worktree_still_detects_as_worktree() {
    let fx = GitFixtures::nested_worktree().unwrap();
    let nested = fx.nested_worktree_path.as_ref().unwrap();
    let status = detect_git_status(nested);
    assert!(status.is_git);
    assert!(
        status.is_worktree,
        "nested worktree must be flagged as worktree"
    );
    assert_eq!(status.branch, "nested");
}

#[test]
fn task2_worktree_commondir_resolves_to_main() {
    let fx = GitFixtures::worktree("feature").unwrap();
    // .git file inside the worktree points at main/.git/worktrees/<name>/
    let gitdir_file = std::fs::read_to_string(fx.worktree_path.join(".git")).unwrap();
    let gitdir_line = gitdir_file.trim_start_matches("gitdir: ").trim();
    let wt_gitdir = PathBuf::from(gitdir_line);

    let resolved = find_main_worktree_path(&wt_gitdir).expect("commondir must resolve");
    let expected = fx.main_path.canonicalize().unwrap();
    assert_eq!(resolved, expected);
}

#[test]
fn task2_detached_head_uses_short_sha_for_branch() {
    let fx = GitFixtures::detached_head().unwrap();
    let status = detect_git_status(&fx.repo_path);
    assert!(status.is_git);
    assert!(!status.is_worktree);
    // detect_git_status emits first 8 chars of SHA for detached HEAD.
    assert_eq!(status.branch.len(), 8);
    assert_eq!(
        &status.commit_hash.as_ref().unwrap()[..8],
        status.branch.as_str()
    );
}

// ─── Task 3: orphan cleanup on path disappearance ──────────────────────────

#[tokio::test]
async fn task3_pathvalidator_flags_missing_worktree_after_grace() {
    let fx = GitFixtures::worktree("feature").unwrap();
    let wt_path = fx.worktree_path.clone();

    // Grace period = 0 so the first-pass detection immediately confirms.
    let validator = PathValidator::with_config(PathValidatorConfig {
        validation_interval_hours: 0,
        enabled: true,
        grace_period_minutes: 0,
        max_paths_per_cycle: 100,
    });

    let project = RegisteredProject {
        project_id: "wt-001".into(),
        path: wt_path.clone(),
        is_active: true,
    };

    // First pass: path exists -> no orphan.
    let orphans = validator
        .validate_projects(vec![project.clone()])
        .await
        .unwrap();
    assert!(orphans.is_empty(), "existing path must not be orphaned");

    // Simulate `git worktree remove`/prune by deleting the checkout.
    std::fs::remove_dir_all(&wt_path).unwrap();

    // Second pass with grace=0: registered twice -> confirmed orphan.
    let orphans1 = validator
        .validate_projects(vec![project.clone()])
        .await
        .unwrap();
    // Grace period of 0 still requires the "first_missing" entry to exist,
    // so the first post-deletion pass primes pending; the second confirms.
    let orphans2 = validator.validate_projects(vec![project]).await.unwrap();
    let total: Vec<_> = orphans1.into_iter().chain(orphans2.into_iter()).collect();
    assert_eq!(total.len(), 1, "orphan must be confirmed within two passes");
    assert_eq!(total[0].project_id, "wt-001");
}

// ─── Task 4: branch rename mid-session ─────────────────────────────────────

#[tokio::test]
async fn task4_branch_rename_emits_renamed_event_within_timeout() {
    let fx = GitFixtures::plain_clone().unwrap();

    let detector = BranchLifecycleDetector::new(
        fx.repo_path.clone(),
        BranchLifecycleConfig {
            enabled: true,
            auto_delete_on_branch_delete: true,
            scan_interval_seconds: 1,
            rename_correlation_timeout_ms: 2000,
        },
    );
    detector.initialize().await.unwrap();

    // Rename main -> trunk.
    std::process::Command::new("git")
        .args([
            "-c",
            "user.name=Fixture Bot",
            "-c",
            "user.email=fixture@example.invalid",
            "branch",
            "-m",
            "main",
            "trunk",
        ])
        .current_dir(&fx.repo_path)
        .status()
        .unwrap();

    // Post-#69: an atomic rename must produce a single Renamed event.
    // A DefaultChanged event is still expected because .git/HEAD now
    // points to the new branch name.
    let events = detector.scan_for_changes().await.unwrap();
    let renamed = events.iter().find_map(|e| match e {
        BranchEvent::Renamed { old_name, new_name } => Some((old_name.clone(), new_name.clone())),
        _ => None,
    });
    assert_eq!(
        renamed,
        Some(("main".to_string(), "trunk".to_string())),
        "expected a single Renamed main->trunk, got {:?}",
        events
    );
    let default_changed = events.iter().any(|e| {
        matches!(
            e,
            BranchEvent::DefaultChanged { old_default, new_default }
                if old_default == "main" && new_default == "trunk"
        )
    });
    assert!(
        default_changed,
        "expected DefaultChanged main->trunk, got {:?}",
        events
    );
    // Must NOT emit a bare Created for the new name.
    let has_created = events
        .iter()
        .any(|e| matches!(e, BranchEvent::Created { branch, .. } if branch == "trunk"));
    assert!(
        !has_created,
        "rename must not leak a Created event for the new name, got {:?}",
        events
    );
}

// ─── Task 5: branch deletion cleanup ───────────────────────────────────────

#[tokio::test]
async fn task5_branch_deletion_emits_deleted_after_rename_timeout() {
    let fx = GitFixtures::plain_clone().unwrap();

    // Create and commit on a feature branch, then switch back to main.
    run_git(&fx.repo_path, &["checkout", "-b", "feature"]);
    std::fs::write(fx.repo_path.join("b.txt"), "b").unwrap();
    run_git(&fx.repo_path, &["add", "b.txt"]);
    run_git(
        &fx.repo_path,
        &[
            "commit",
            "-m",
            "b",
            "--author=Fixture Bot <fixture@example.invalid>",
        ],
    );
    run_git(&fx.repo_path, &["checkout", "main"]);

    let detector = BranchLifecycleDetector::new(
        fx.repo_path.clone(),
        BranchLifecycleConfig {
            enabled: true,
            auto_delete_on_branch_delete: true,
            scan_interval_seconds: 1,
            // Tight timeout so expiry happens before the second scan.
            rename_correlation_timeout_ms: 50,
        },
    );
    detector.initialize().await.unwrap();

    // Force delete the branch.
    run_git(&fx.repo_path, &["branch", "-D", "feature"]);

    // First scan captures the delete into pending.
    let events1 = detector.scan_for_changes().await.unwrap();
    // Second scan after the timeout must emit Deleted.
    tokio::time::sleep(Duration::from_millis(120)).await;
    let events2 = detector.scan_for_changes().await.unwrap();

    let all: Vec<_> = events1.into_iter().chain(events2.into_iter()).collect();
    let deleted: Vec<_> = all
        .iter()
        .filter(|e| matches!(e, BranchEvent::Deleted { branch } if branch == "feature"))
        .collect();
    assert_eq!(
        deleted.len(),
        1,
        "expected one Deleted event, got {:?}",
        all
    );
}

// ─── Task 6: default branch change ─────────────────────────────────────────

#[tokio::test]
async fn task6_default_branch_change_via_head_rename_is_detected() {
    let fx = GitFixtures::plain_clone().unwrap();

    let detector =
        BranchLifecycleDetector::new(fx.repo_path.clone(), BranchLifecycleConfig::default());
    detector.initialize().await.unwrap();
    assert_eq!(detector.get_default_branch().await.as_deref(), Some("main"));

    // Rename the current branch (detector reads .git/HEAD for the default).
    run_git(&fx.repo_path, &["branch", "-m", "main", "trunk"]);

    let events = detector.scan_for_changes().await.unwrap();
    let default_changed = events
        .iter()
        .find(|e| matches!(e, BranchEvent::DefaultChanged { .. }))
        .expect("DefaultChanged event must fire when HEAD points to a new branch name");
    if let BranchEvent::DefaultChanged {
        old_default,
        new_default,
    } = default_changed
    {
        assert_eq!(old_default, "main");
        assert_eq!(new_default, "trunk");
    }
}

// ─── Task 7: rapid branch switch (basic, single-file) ──────────────────────

#[test]
fn task7_rapid_branch_switch_lands_on_final_branch() {
    // Exercises git's own correctness only: detector/switch handler need a
    // full daemon to assert zero queue duplication; captured as a gap.
    let fx = GitFixtures::plain_clone().unwrap();
    run_git(&fx.repo_path, &["checkout", "-b", "feature"]);
    std::fs::write(fx.repo_path.join("x.txt"), "f").unwrap();
    run_git(&fx.repo_path, &["add", "x.txt"]);
    run_git(
        &fx.repo_path,
        &[
            "commit",
            "-m",
            "x",
            "--author=Fixture Bot <fixture@example.invalid>",
        ],
    );
    run_git(&fx.repo_path, &["checkout", "main"]);
    run_git(&fx.repo_path, &["checkout", "feature"]);
    run_git(&fx.repo_path, &["checkout", "main"]);

    let status = detect_git_status(&fx.repo_path);
    assert_eq!(status.branch, "main");
}

// ─── Task 8: tenant_id disambiguation across clones ────────────────────────

#[test]
fn task8_multiple_clones_share_remote_hash_but_get_distinct_ids() {
    let fx = GitFixtures::multiple_clones(3).unwrap();
    let calc = ProjectIdCalculator::new();

    // remote_hash depends only on the normalised remote URL, so it must be
    // identical across all clones.
    let remote_hash = calc.calculate_remote_hash(&fx.remote_url);
    for p in &fx.clone_paths {
        let hash = calc.calculate_remote_hash(&fx.remote_url);
        assert_eq!(
            hash, remote_hash,
            "remote_hash must be stable for clone {:?}",
            p
        );
    }

    // Disambiguation paths must differ for clones under a shared ancestor.
    let disambig = DisambiguationPathComputer::recompute_all(&fx.clone_paths);
    let values: std::collections::HashSet<_> = disambig.values().collect();
    assert_eq!(values.len(), 3, "each clone must get a distinct disambig");

    // Final tenant_ids must differ.
    let ids: std::collections::HashSet<_> = fx
        .clone_paths
        .iter()
        .map(|p| calc.calculate(p, Some(&fx.remote_url), disambig.get(p).map(String::as_str)))
        .collect();
    assert_eq!(ids.len(), 3, "tenant_ids must be distinct per clone");
}

#[test]
fn task8_no_remote_yields_local_prefixed_id() {
    let fx = GitFixtures::no_remote().unwrap();
    let calc = ProjectIdCalculator::new();
    let id = calc.calculate(&fx.repo_path, None, None);
    assert!(
        id.starts_with("local_"),
        "no-remote projects must be local_*, got {id}"
    );
}

#[test]
fn task8_two_independent_clones_recomputed_get_stable_disambig() {
    // Sanity: if we re-run recompute_all on the same paths, the mapping is
    // deterministic (daemon re-registers a clone on restart and must not flap).
    let fx = GitFixtures::multiple_clones(2).unwrap();
    let first = DisambiguationPathComputer::recompute_all(&fx.clone_paths);
    let second = DisambiguationPathComputer::recompute_all(&fx.clone_paths);
    assert_eq!(first, second, "recompute_all must be deterministic");
}

// ─── Task 9: cross-cutting edge cases ──────────────────────────────────────

#[test]
fn task9_mid_rebase_still_reports_is_git() {
    let fx = GitFixtures::mid_rebase().unwrap();
    let status = detect_git_status(&fx.repo_path);
    assert!(
        status.is_git,
        "repo during rebase must still be is_git=true"
    );
    // During rebase, HEAD is detached; branch is short SHA or "HEAD".
    assert!(
        status.branch.len() == 8 || status.branch == "HEAD",
        "mid-rebase branch was {}",
        status.branch
    );
}

#[test]
fn task9_submodule_has_its_own_git_pointer_and_distinct_tenant() {
    let fx = GitFixtures::with_submodule().unwrap();
    let parent_status = detect_git_status(&fx.parent_path);
    let sub_status = detect_git_status(&fx.submodule_path);

    assert!(parent_status.is_git);
    assert!(sub_status.is_git);

    // Both paths are "git" but the submodule's tenant (derived from its own
    // remote) must not collide with the parent's.
    let calc = ProjectIdCalculator::new();
    let parent_id = calc.calculate(&fx.parent_path, Some(&fx.parent_remote_url), None);
    let sub_id = calc.calculate(&fx.submodule_path, Some(&fx.submodule_remote_url), None);
    assert_ne!(
        parent_id, sub_id,
        "submodule tenant_id must differ from parent"
    );
}

#[test]
fn task9_shallow_clone_is_git_and_has_commit() {
    let fx = GitFixtures::shallow_clone(1).unwrap();
    let status = detect_git_status(&fx.repo_path);
    assert!(status.is_git);
    assert_eq!(status.branch, "main");
    assert!(
        status.commit_hash.is_some(),
        "shallow clone must still expose HEAD commit"
    );
    assert!(fx.repo_path.join(".git/shallow").exists());
}

// ─── helpers ───────────────────────────────────────────────────────────────

fn run_git(cwd: &std::path::Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .args([
            "-c",
            "user.name=Fixture Bot",
            "-c",
            "user.email=fixture@example.invalid",
        ])
        .args(args)
        .current_dir(cwd)
        .status()
        .expect("git exec");
    assert!(status.success(), "git {:?} failed in {:?}", args, cwd);
}

// Silence unused-import warnings on platforms where tempdir is unused.
#[allow(dead_code)]
fn _force_tempdir_used() -> tempfile::TempDir {
    tempdir().unwrap()
}
