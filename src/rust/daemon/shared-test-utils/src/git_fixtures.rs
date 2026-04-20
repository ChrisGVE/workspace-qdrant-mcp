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

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use git2::{Repository, Signature};
use tempfile::TempDir;

use crate::TestResult;

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

        // Create a conflicting branch: diverging content on same file
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

        // Allow file-based submodules (git ≥ 2.38 requires opt-in).
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

// ---- helpers --------------------------------------------------------------

fn test_signature<'a>() -> git2::Signature<'a> {
    // Deterministic author/committer for reproducible commit SHAs across runs.
    Signature::new(
        "Fixture Bot",
        "fixture@example.invalid",
        &git2::Time::new(1_700_000_000, 0),
    )
    .expect("static signature")
}

/// Create the initial commit on `branch_name`, materializing a single file so
/// the working tree is non-empty.
fn seed_initial_commit(
    repo: &Repository,
    branch_name: &str,
    file_name: &str,
    content: &str,
) -> TestResult<String> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| "bare repo has no workdir".to_string())?
        .to_path_buf();
    fs::write(workdir.join(file_name), content)?;

    let mut index = repo.index()?;
    index.add_path(Path::new(file_name))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;

    let sig = test_signature();
    let ref_name = format!("refs/heads/{}", branch_name);
    let commit_id = repo.commit(Some(&ref_name), &sig, &sig, "initial", &tree, &[])?;

    // Point HEAD at the branch so subsequent ops see the right ref.
    repo.set_head(&ref_name)?;
    Ok(commit_id.to_string())
}

/// Write `content` to `file_name` and create a follow-up commit on the current
/// branch.
fn write_and_commit(
    repo_path: &Path,
    file_name: &str,
    content: &str,
    msg: &str,
) -> TestResult<String> {
    fs::write(repo_path.join(file_name), content)?;
    let repo = Repository::open(repo_path)?;
    let mut index = repo.index()?;
    index.add_path(Path::new(file_name))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;

    let parent = repo.head()?.peel_to_commit()?;
    let sig = test_signature();
    let commit_id = repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &[&parent])?;
    Ok(commit_id.to_string())
}

/// Run `git` in `cwd`, erroring with captured stderr on non-zero exit.
fn run_git(cwd: &Path, args: &[&str]) -> TestResult<Output> {
    let out = Command::new("git")
        // Force deterministic identity even if the host has no user.* config.
        .args([
            "-c",
            "user.name=Fixture Bot",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .current_dir(cwd)
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "git {:?} failed in {}: {}",
            args,
            cwd.display(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(out)
}

/// Run `git` and return captured stdout (utf-8).
fn run_git_stdout(cwd: &Path, args: &[&str]) -> TestResult<String> {
    let out = run_git(cwd, args)?;
    Ok(String::from_utf8(out.stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
