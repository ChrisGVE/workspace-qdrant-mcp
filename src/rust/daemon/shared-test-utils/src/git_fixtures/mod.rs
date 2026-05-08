//! Deterministic git fixtures for branch/worktree audit scenarios.
//!
//! Produces repository layouts that exercise every path the daemon may
//! encounter: plain clones, multiple clones sharing a remote, linked and
//! nested worktrees, detached HEAD, no-remote repos, mid-rebase state,
//! submodules, and shallow clones.
//!
//! All fixtures are self-contained inside a `TempDir` and use git2 where
//! convenient, falling back to the `git` CLI for operations git2 does not
//! expose cleanly (worktree add, rebase, submodules, shallow clones).

mod helpers;

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use git2::Repository;
use tempfile::TempDir;

use crate::TestResult;
use helpers::{run_git, run_git_stdout, seed_initial_commit, write_and_commit};

/// Generic fixture returned by single-repo builders.
pub struct GitFixture {
    /// Temp root owning all paths (drop cleans up).
    pub temp_dir: TempDir,
    /// Path to the primary working tree.
    pub repo_path: PathBuf,
    /// Initial branch name.
    pub branch: String,
    /// Initial commit SHA (None for empty repos or transient states).
    pub commit_hash: Option<String>,
    /// Remote URL stored in `origin`, if any.
    pub remote_url: Option<String>,
    /// Free-form metadata for assertions.
    pub metadata: HashMap<String, String>,
}

/// Fixture for multi-clone scenarios (same remote, multiple working trees).
pub struct MultiCloneFixture {
    pub temp_dir: TempDir,
    /// Path to the shared bare "remote" repo.
    pub remote_path: PathBuf,
    /// Clone paths (length == `n`).
    pub clone_paths: Vec<PathBuf>,
    /// Common remote URL embedded in each clone.
    pub remote_url: String,
    pub branch: String,
    pub commit_hash: String,
}

/// Fixture for worktree scenarios.
pub struct WorktreeFixture {
    pub temp_dir: TempDir,
    /// Main working tree path.
    pub main_path: PathBuf,
    /// Linked worktree path.
    pub worktree_path: PathBuf,
    /// Optional nested worktree path (nested scenario only).
    pub nested_worktree_path: Option<PathBuf>,
    pub main_branch: String,
    pub worktree_branch: String,
    pub commit_hash: String,
}

/// Fixture for submodule scenario.
pub struct SubmoduleFixture {
    pub temp_dir: TempDir,
    pub parent_path: PathBuf,
    pub submodule_path: PathBuf,
    pub parent_remote_url: String,
    pub submodule_remote_url: String,
    pub parent_commit: String,
    pub submodule_commit: String,
}

/// Builder for deterministic git fixtures.
pub struct GitFixtures;

impl GitFixtures {
    // ---- single-repo scenarios --------------------------------------------

    /// Plain clone: init working tree with one commit and an `origin` remote.
    pub fn plain_clone() -> TestResult<GitFixture> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path().join("repo");
        fs::create_dir_all(&repo_path)?;

        let repo = Repository::init(&repo_path)?;
        let commit = seed_initial_commit(&repo, "main", "README.md", "initial")?;

        let remote_url = "https://example.invalid/user/plain.git".to_string();
        repo.remote("origin", &remote_url)?;

        let mut metadata = HashMap::new();
        metadata.insert("scenario".to_string(), "plain_clone".to_string());

        Ok(GitFixture {
            temp_dir,
            repo_path,
            branch: "main".to_string(),
            commit_hash: Some(commit),
            remote_url: Some(remote_url),
            metadata,
        })
    }

    /// Repo with one commit but no remote configured.
    pub fn no_remote() -> TestResult<GitFixture> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path().join("repo");
        fs::create_dir_all(&repo_path)?;

        let repo = Repository::init(&repo_path)?;
        let commit = seed_initial_commit(&repo, "main", "README.md", "local-only")?;

        let mut metadata = HashMap::new();
        metadata.insert("scenario".to_string(), "no_remote".to_string());

        Ok(GitFixture {
            temp_dir,
            repo_path,
            branch: "main".to_string(),
            commit_hash: Some(commit),
            remote_url: None,
            metadata,
        })
    }

    /// Repo with HEAD detached at the initial commit.
    pub fn detached_head() -> TestResult<GitFixture> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path().join("repo");
        fs::create_dir_all(&repo_path)?;

        let repo = Repository::init(&repo_path)?;
        let commit = seed_initial_commit(&repo, "main", "README.md", "initial")?;
        let oid = git2::Oid::from_str(&commit)?;
        repo.set_head_detached(oid)?;

        let mut metadata = HashMap::new();
        metadata.insert("scenario".to_string(), "detached_head".to_string());

        Ok(GitFixture {
            temp_dir,
            repo_path,
            branch: "HEAD".to_string(),
            commit_hash: Some(commit),
            remote_url: None,
            metadata,
        })
    }

    /// Repo left in an interactive rebase state (`.git/rebase-apply/` present).
    pub fn mid_rebase() -> TestResult<GitFixture> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path().join("repo");
        fs::create_dir_all(&repo_path)?;

        let repo = Repository::init(&repo_path)?;
        seed_initial_commit(&repo, "main", "README.md", "initial")?;

        // Create a conflicting branch: diverging content on same file.
        run_git(&repo_path, &["checkout", "-b", "feature"])?;
        write_and_commit(&repo_path, "README.md", "feature-side", "feature commit")?;

        run_git(&repo_path, &["checkout", "main"])?;
        write_and_commit(&repo_path, "README.md", "main-side", "main commit")?;

        // Start a rebase that will conflict and leave rebase-apply behind.
        // Use `rebase main` on feature; expect non-zero exit due to conflict.
        run_git(&repo_path, &["checkout", "feature"])?;
        let out = Command::new("git")
            .args(["-c", "rebase.autoStash=false", "rebase", "main"])
            .current_dir(&repo_path)
            .output()?;
        // We intentionally expect rebase to stop; do not error on non-zero.
        // Validate rebase state exists (either rebase-apply or rebase-merge).
        let rebase_apply = repo_path.join(".git/rebase-apply");
        let rebase_merge = repo_path.join(".git/rebase-merge");
        if !rebase_apply.exists() && !rebase_merge.exists() {
            return Err(format!(
                "mid_rebase fixture failed to leave rebase state. status={:?}\nstderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }

        let head_sha = run_git_stdout(&repo_path, &["rev-parse", "HEAD"])?
            .trim()
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("scenario".to_string(), "mid_rebase".to_string());
        metadata.insert(
            "rebase_dir".to_string(),
            if rebase_apply.exists() {
                "rebase-apply".to_string()
            } else {
                "rebase-merge".to_string()
            },
        );

        Ok(GitFixture {
            temp_dir,
            repo_path,
            // During rebase, HEAD is typically detached on the rebased commit.
            branch: "HEAD".to_string(),
            commit_hash: Some(head_sha),
            remote_url: None,
            metadata,
        })
    }

    /// Shallow clone (depth 1 by default): realistic `git clone --depth` of a
    /// local bare repo.
    pub fn shallow_clone(depth: u32) -> TestResult<GitFixture> {
        if depth == 0 {
            return Err("shallow_clone: depth must be >= 1".into());
        }

        let temp_dir = TempDir::new()?;

        // Seed a source repo with multiple commits so the shallow clone is
        // meaningfully shallow.
        let source_path = temp_dir.path().join("source");
        fs::create_dir_all(&source_path)?;
        let src_repo = Repository::init(&source_path)?;
        seed_initial_commit(&src_repo, "main", "README.md", "c1")?;
        write_and_commit(&source_path, "README.md", "c2", "commit 2")?;
        write_and_commit(&source_path, "README.md", "c3", "commit 3")?;

        // Bare remote as the clone source so `file://` works cross-platform.
        let remote_path = temp_dir.path().join("remote.git");
        run_git(
            temp_dir.path(),
            &[
                "clone",
                "--bare",
                source_path.to_str().unwrap(),
                remote_path.to_str().unwrap(),
            ],
        )?;

        let clone_path = temp_dir.path().join("clone");
        let remote_url = format!("file://{}", remote_path.display());
        run_git(
            temp_dir.path(),
            &[
                "clone",
                "--depth",
                &depth.to_string(),
                &remote_url,
                clone_path.to_str().unwrap(),
            ],
        )?;

        let commit = run_git_stdout(&clone_path, &["rev-parse", "HEAD"])?
            .trim()
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("scenario".to_string(), "shallow_clone".to_string());
        metadata.insert("depth".to_string(), depth.to_string());

        Ok(GitFixture {
            temp_dir,
            repo_path: clone_path,
            branch: "main".to_string(),
            commit_hash: Some(commit),
            remote_url: Some(remote_url),
            metadata,
        })
    }

    // ---- multi-repo scenarios ---------------------------------------------

    /// N sibling clones of the same bare remote. All clones share remote URL
    /// but live at distinct canonical paths -- exercises tenant_id
    /// disambiguation logic.
    pub fn multiple_clones(n: usize) -> TestResult<MultiCloneFixture> {
        if n < 2 {
            return Err("multiple_clones: need n >= 2 to exercise disambiguation".into());
        }

        let temp_dir = TempDir::new()?;

        // Source + bare remote.
        let source_path = temp_dir.path().join("source");
        fs::create_dir_all(&source_path)?;
        let src_repo = Repository::init(&source_path)?;
        let commit = seed_initial_commit(&src_repo, "main", "README.md", "shared")?;

        let remote_path = temp_dir.path().join("remote.git");
        run_git(
            temp_dir.path(),
            &[
                "clone",
                "--bare",
                source_path.to_str().unwrap(),
                remote_path.to_str().unwrap(),
            ],
        )?;

        let remote_url = format!("file://{}", remote_path.display());

        let mut clone_paths = Vec::with_capacity(n);
        for i in 0..n {
            let p = temp_dir.path().join(format!("clone{}", i + 1));
            run_git(
                temp_dir.path(),
                &["clone", &remote_url, p.to_str().unwrap()],
            )?;
            clone_paths.push(p);
        }

        Ok(MultiCloneFixture {
            temp_dir,
            remote_path,
            clone_paths,
            remote_url,
            branch: "main".to_string(),
            commit_hash: commit,
        })
    }

    /// Linked worktree on a new branch off `main`.
    pub fn worktree(branch_name: &str) -> TestResult<WorktreeFixture> {
        let temp_dir = TempDir::new()?;
        let main_path = temp_dir.path().join("main");
        fs::create_dir_all(&main_path)?;

        let repo = Repository::init(&main_path)?;
        let commit = seed_initial_commit(&repo, "main", "README.md", "initial")?;

        let worktree_path = temp_dir.path().join("wt");
        run_git(
            &main_path,
            &[
                "worktree",
                "add",
                "-b",
                branch_name,
                worktree_path.to_str().unwrap(),
            ],
        )?;

        Ok(WorktreeFixture {
            temp_dir,
            main_path,
            worktree_path,
            nested_worktree_path: None,
            main_branch: "main".to_string(),
            worktree_branch: branch_name.to_string(),
            commit_hash: commit,
        })
    }

    /// Worktree containing another worktree (tests `find_main_worktree_path`
    /// recursion via `commondir`).
    pub fn nested_worktree() -> TestResult<WorktreeFixture> {
        let mut fx = Self::worktree("feature")?;

        // Create a second worktree *relative to the linked worktree*. Git will
        // still register it against the main repo's .git/worktrees metadata,
        // but from the daemon's perspective the nested checkout's .git file
        // must resolve through multiple hops.
        let nested = fx.temp_dir.path().join("nested");
        run_git(
            &fx.worktree_path,
            &["worktree", "add", "-b", "nested", nested.to_str().unwrap()],
        )?;
        fx.nested_worktree_path = Some(nested);
        Ok(fx)
    }

    /// Parent repo with one submodule, each backed by its own bare remote.
    pub fn with_submodule() -> TestResult<SubmoduleFixture> {
        let temp_dir = TempDir::new()?;

        // Submodule source + bare remote.
        let sub_source = temp_dir.path().join("sub-source");
        fs::create_dir_all(&sub_source)?;
        let sub_repo = Repository::init(&sub_source)?;
        let sub_commit = seed_initial_commit(&sub_repo, "main", "lib.rs", "submodule")?;

        let sub_remote = temp_dir.path().join("sub.git");
        run_git(
            temp_dir.path(),
            &[
                "clone",
                "--bare",
                sub_source.to_str().unwrap(),
                sub_remote.to_str().unwrap(),
            ],
        )?;
        let sub_remote_url = format!("file://{}", sub_remote.display());

        // Parent repo.
        let parent_path = temp_dir.path().join("parent");
        fs::create_dir_all(&parent_path)?;
        let parent_repo = Repository::init(&parent_path)?;
        seed_initial_commit(&parent_repo, "main", "README.md", "parent")?;

        // Allow file-based submodules (git >= 2.38 requires opt-in).
        run_git(
            &parent_path,
            &[
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                &sub_remote_url,
                "lib",
            ],
        )?;
        run_git(&parent_path, &["commit", "-m", "add submodule"])?;
        let parent_commit = run_git_stdout(&parent_path, &["rev-parse", "HEAD"])?
            .trim()
            .to_string();

        // Parent remote URL (configured, not cloned from).
        let parent_remote_url = "https://example.invalid/user/parent.git".to_string();
        parent_repo.remote("origin", &parent_remote_url)?;

        let submodule_path = parent_path.join("lib");

        Ok(SubmoduleFixture {
            temp_dir,
            parent_path,
            submodule_path,
            parent_remote_url,
            submodule_remote_url: sub_remote_url,
            parent_commit,
            submodule_commit: sub_commit,
        })
    }
}

#[cfg(test)]
mod tests;
