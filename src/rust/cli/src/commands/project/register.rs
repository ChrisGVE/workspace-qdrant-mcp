//! Register a project for tracking

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::paths::CanonicalPath;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::RegisterProjectRequest;
use crate::output;

use super::resolver::calculate_project_id;

async fn call_daemon_register(
    abs_path: std::path::PathBuf,
    project_id: String,
    project_name: String,
    git_remote: Option<String>,
) {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = RegisterProjectRequest {
                path: abs_path.display().to_string(),
                project_id: project_id.clone(),
                name: Some(project_name),
                git_remote,
                register_if_new: true,
                priority: None,
            };

            match client.project().register_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.created {
                        output::success("Project registered successfully");
                    } else {
                        output::info("Project already registered");
                    }
                    output::kv("Active", if result.is_active { "Yes" } else { "No" });
                    if result.is_worktree {
                        output::info(format!(
                            "Note: This path is a git worktree. \
                             It is indexed under project {}",
                            result.project_id
                        ));
                        if let Some(watch_path) = &result.watch_path {
                            output::kv("Watch Path", watch_path);
                        }
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to register project: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }
}

/// Build a [`CanonicalPath`] from a CLI path argument, absolutizing
/// relative inputs syntactically against CWD. No fs canonicalize.
fn canonical_from_cli_path(path: &std::path::Path) -> Result<CanonicalPath> {
    let s = path.to_str().context("Path contains invalid UTF-8")?;
    if let Ok(cp) = CanonicalPath::from_user_input(s) {
        return Ok(cp);
    }
    let cwd = std::env::current_dir().context("Could not determine current directory")?;
    let joined = cwd.join(path);
    let joined_str = joined
        .to_str()
        .context("Path contains invalid UTF-8 after CWD join")?;
    CanonicalPath::from_user_input(joined_str)
        .map_err(|e| anyhow::anyhow!("Could not resolve path: {e}"))
}

/// Return the canonical git top-level directory containing `path`, or `None`
/// if `path` is not inside a git working tree (or git is unavailable).
///
/// Used to detect project boundaries: a git submodule (or any nested repo) has
/// its own top-level distinct from the enclosing project, so it must be allowed
/// to register as an independent project rather than being absorbed into the
/// ancestor it physically sits under (#86).
fn git_toplevel(path: &std::path::Path) -> Option<PathBuf> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let top = String::from_utf8(output.stdout).ok()?;
    let top = top.trim();
    if top.is_empty() {
        return None;
    }
    std::fs::canonicalize(top).ok()
}

/// Decide whether a directory is a distinct project boundary from a matched
/// ancestor project, given each one's git top-level.
///
/// A git submodule / nested repository has its own top-level distinct from the
/// enclosing project, so it should register independently. A plain subdirectory
/// of the same repository (and exact re-registration) shares the ancestor's
/// top-level and is not distinct. A non-git directory (`own_root == None`) is
/// never treated as a distinct boundary (#86).
fn is_distinct_git_boundary(
    own_root: Option<&std::path::Path>,
    ancestor_root: Option<&std::path::Path>,
) -> bool {
    own_root.is_some() && own_root != ancestor_root
}

pub(super) async fn register_project(
    path: Option<PathBuf>,
    name: Option<String>,
    yes: bool,
) -> Result<()> {
    let project_path = path.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let abs_canonical = canonical_from_cli_path(&project_path)?;
    let abs_path = PathBuf::from(abs_canonical.as_str());

    let project_name = name.unwrap_or_else(|| {
        abs_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    // Generate project ID using the same algorithm as the daemon
    let project_id = calculate_project_id(&abs_path);

    // Detect git remote (same function used by project ID calculation)
    let git_remote = wqm_common::project_id::detect_git_remote(&abs_path);

    // Check if the path is already part of a registered project. The lookup
    // returns the longest registered ancestor (or exact match) of `abs_path`.
    if let Ok(db_path) = crate::config::get_database_path_checked() {
        if let Some((existing_id, existing_path)) =
            wqm_common::project_id::resolve_path_to_project(&db_path, &abs_path)
        {
            // A git submodule / nested repo is its own project boundary even
            // though it physically lives under the ancestor project. Detect this
            // by comparing git top-levels: when `abs_path` belongs to a different
            // git repository than the matched ancestor, allow it to register as
            // an independent project instead of refusing (#86). Plain
            // subdirectories of the same repo (and exact re-registration) share
            // the ancestor's top-level and are still treated as "already part of".
            let existing_pb = PathBuf::from(&existing_path);
            let own_root = git_toplevel(&abs_path);
            let ancestor_root = git_toplevel(&existing_pb);

            if !is_distinct_git_boundary(own_root.as_deref(), ancestor_root.as_deref()) {
                output::section("Register Project");
                output::info(format!(
                    "This directory is already part of project '{}'",
                    existing_id
                ));
                output::kv("Existing Project ID", &existing_id);
                output::kv(
                    "Project Path",
                    crate::output::style::home_to_tilde(&existing_path),
                );
                return Ok(());
            }
        }
    }

    // Display summary
    output::section("Register Project");
    output::kv("Path", abs_canonical.as_str());
    output::kv("Name", &project_name);
    output::kv("Project ID", &project_id);
    if let Some(remote) = &git_remote {
        output::kv("Git Remote", remote);
    }
    output::separator();

    // Confirm unless --yes
    if !yes && !output::confirm("Register this project?") {
        output::info("Aborted");
        return Ok(());
    }

    call_daemon_register(abs_path, project_id, project_name, git_remote).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn git_init(dir: &Path) {
        // Minimal repo: init + identity so commits/operations don't warn.
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "t"],
        ] {
            let ok = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(&args)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            assert!(ok, "git {args:?} failed in {dir:?}");
        }
    }

    #[test]
    fn distinct_boundary_logic() {
        let repo_a = Path::new("/repo/a");
        let repo_b = Path::new("/repo/b");
        // Same repo (plain subdir / exact match) → not distinct.
        assert!(!is_distinct_git_boundary(Some(repo_a), Some(repo_a)));
        // Different git repo (submodule / nested) → distinct.
        assert!(is_distinct_git_boundary(Some(repo_b), Some(repo_a)));
        // Own repo but ancestor is a non-git registered dir → distinct.
        assert!(is_distinct_git_boundary(Some(repo_b), None));
        // Non-git directory is never a distinct boundary.
        assert!(!is_distinct_git_boundary(None, Some(repo_a)));
        assert!(!is_distinct_git_boundary(None, None));
    }

    #[test]
    fn git_toplevel_detects_nested_repo_boundary() {
        // A nested repository (the boundary a submodule creates) must report a
        // top-level distinct from its enclosing repository (#86).
        let parent = tempfile::tempdir().expect("tempdir");
        git_init(parent.path());
        let nested = parent.path().join("nested-repo");
        std::fs::create_dir_all(&nested).expect("mkdir nested");
        git_init(&nested);

        let parent_root = git_toplevel(parent.path());
        let nested_root = git_toplevel(&nested);

        assert!(parent_root.is_some(), "parent toplevel resolved");
        assert!(nested_root.is_some(), "nested toplevel resolved");
        assert_ne!(
            parent_root, nested_root,
            "nested repo must have a distinct git top-level"
        );
        // And the boundary decision treats the nested repo as registerable.
        assert!(is_distinct_git_boundary(
            nested_root.as_deref(),
            parent_root.as_deref()
        ));
        // A plain subdir of the parent (no nested .git) resolves to the parent
        // top-level → not a distinct boundary.
        let plain = parent.path().join("plain-subdir");
        std::fs::create_dir_all(&plain).expect("mkdir plain");
        let plain_root = git_toplevel(&plain);
        assert_eq!(plain_root, parent_root, "plain subdir maps to parent repo");
        assert!(!is_distinct_git_boundary(
            plain_root.as_deref(),
            parent_root.as_deref()
        ));
    }
}
