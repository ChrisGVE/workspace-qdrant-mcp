//! Tests for git fixture builders.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use git2::Repository;

use super::helpers::run_git_stdout;
use super::*;
use crate::TestResult;

fn has_git_cli() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn plain_clone_opens_and_has_remote() -> TestResult {
    let fx = GitFixtures::plain_clone()?;
    let repo = Repository::open(&fx.repo_path)?;
    assert!(repo.find_remote("origin").is_ok());
    assert!(fx.repo_path.join(".git").is_dir());
    assert!(fx.commit_hash.is_some());
    assert_eq!(fx.branch, "main");
    Ok(())
}

#[test]
fn no_remote_has_no_origin() -> TestResult {
    let fx = GitFixtures::no_remote()?;
    let repo = Repository::open(&fx.repo_path)?;
    assert!(repo.find_remote("origin").is_err());
    Ok(())
}

#[test]
fn detached_head_puts_head_off_branch() -> TestResult {
    let fx = GitFixtures::detached_head()?;
    let repo = Repository::open(&fx.repo_path)?;
    let head = repo.head()?;
    assert!(!head.is_branch(), "HEAD should be detached");
    Ok(())
}

#[test]
fn mid_rebase_leaves_rebase_dir() -> TestResult {
    if !has_git_cli() {
        eprintln!("skipping: git CLI unavailable");
        return Ok(());
    }
    let fx = GitFixtures::mid_rebase()?;
    let rebase_apply = fx.repo_path.join(".git/rebase-apply");
    let rebase_merge = fx.repo_path.join(".git/rebase-merge");
    assert!(
        rebase_apply.exists() || rebase_merge.exists(),
        "expected rebase state dir"
    );
    // Repository::open must still succeed mid-rebase.
    Repository::open(&fx.repo_path)?;
    Ok(())
}

#[test]
fn shallow_clone_restricts_history() -> TestResult {
    if !has_git_cli() {
        return Ok(());
    }
    let fx = GitFixtures::shallow_clone(1)?;
    let repo = Repository::open(&fx.repo_path)?;
    // A shallow file should exist in the .git dir.
    assert!(fx.repo_path.join(".git/shallow").exists());
    // rev-list --count HEAD should be 1.
    let out = run_git_stdout(&fx.repo_path, &["rev-list", "--count", "HEAD"])?;
    assert_eq!(out.trim(), "1");
    assert!(repo.find_remote("origin").is_ok());
    Ok(())
}

#[test]
fn multiple_clones_share_remote_differ_in_path() -> TestResult {
    if !has_git_cli() {
        return Ok(());
    }
    let fx = GitFixtures::multiple_clones(3)?;
    assert_eq!(fx.clone_paths.len(), 3);
    for p in &fx.clone_paths {
        let repo = Repository::open(p)?;
        let url = repo.find_remote("origin")?.url().map(str::to_string);
        assert_eq!(url.as_deref(), Some(fx.remote_url.as_str()));
    }
    // All clones must have distinct canonical paths.
    let mut canon: Vec<_> = fx
        .clone_paths
        .iter()
        .map(|p| p.canonicalize().unwrap())
        .collect();
    canon.sort();
    canon.dedup();
    assert_eq!(canon.len(), 3, "clone paths must be distinct");
    Ok(())
}

#[test]
fn worktree_produces_linked_checkout() -> TestResult {
    if !has_git_cli() {
        return Ok(());
    }
    let fx = GitFixtures::worktree("feature")?;
    // Worktree checkout has a .git *file* (not directory).
    let dot_git = fx.worktree_path.join(".git");
    assert!(dot_git.is_file(), "worktree .git must be a file");
    // Main repo retains a .git directory.
    assert!(fx.main_path.join(".git").is_dir());

    // The .git file points to main/.git/worktrees/<name>/ which contains
    // a commondir file.
    let content = fs::read_to_string(&dot_git)?;
    let gitdir_line = content.trim_start_matches("gitdir: ").trim();
    let commondir = PathBuf::from(gitdir_line).join("commondir");
    assert!(commondir.exists(), "worktree commondir must exist");

    // Repository::open must succeed on the worktree checkout.
    let repo = Repository::open(&fx.worktree_path)?;
    let head = repo.head()?;
    assert_eq!(head.shorthand(), Some("feature"));
    Ok(())
}

#[test]
fn nested_worktree_has_two_linked_checkouts() -> TestResult {
    if !has_git_cli() {
        return Ok(());
    }
    let fx = GitFixtures::nested_worktree()?;
    let nested = fx.nested_worktree_path.as_ref().expect("nested path");
    assert!(nested.join(".git").is_file());
    // Both nested and outer worktrees must open cleanly.
    Repository::open(&fx.worktree_path)?;
    Repository::open(nested)?;
    Ok(())
}

#[test]
fn with_submodule_populates_submodule_dir() -> TestResult {
    if !has_git_cli() {
        return Ok(());
    }
    let fx = GitFixtures::with_submodule()?;
    // Parent .gitmodules must exist.
    assert!(fx.parent_path.join(".gitmodules").exists());
    // Submodule checkout exists under parent.
    assert!(fx.submodule_path.exists());
    // Submodule has its own .git pointer.
    assert!(fx.submodule_path.join(".git").exists());
    Ok(())
}
