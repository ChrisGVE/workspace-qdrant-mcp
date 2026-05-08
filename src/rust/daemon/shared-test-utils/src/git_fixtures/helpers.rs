//! Internal git helper functions shared across fixture builders.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use git2::{Repository, Signature};

use crate::TestResult;

pub(super) fn test_signature<'a>() -> git2::Signature<'a> {
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
pub(super) fn seed_initial_commit(
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
pub(super) fn write_and_commit(
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
pub(super) fn run_git(cwd: &Path, args: &[&str]) -> TestResult<Output> {
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
pub(super) fn run_git_stdout(cwd: &Path, args: &[&str]) -> TestResult<String> {
    let out = run_git(cwd, args)?;
    Ok(String::from_utf8(out.stdout)?)
}
