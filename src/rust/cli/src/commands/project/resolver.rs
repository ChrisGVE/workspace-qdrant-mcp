//! Project ID resolution and disambiguation logic

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::output;

/// Resolve a project argument to a project_id.
///
/// If the argument looks like a path (contains `/` or `.`), resolve it to an
/// absolute path and compute the project_id. Otherwise use it as a direct ID.
pub(crate) fn resolve_project_id(project: &str) -> String {
    if project.contains('/') || project.contains('.') || project == "~" {
        let path = PathBuf::from(project);
        match path.canonicalize() {
            Ok(abs_path) => calculate_project_id(&abs_path),
            Err(_) => project.to_string(),
        }
    } else {
        project.to_string()
    }
}

/// Resolve an optional project argument to a project_id.
///
/// If `project` is Some, delegates to `resolve_project_id`. If None,
/// auto-detects the project from the current working directory by looking up
/// registered projects in SQLite.
pub(crate) fn resolve_project_id_or_cwd(project: Option<&str>) -> Result<String> {
    let (id, auto_detected) = resolve_project_id_or_cwd_quiet(project)?;
    if auto_detected {
        output::info(format!("Auto-detected project: {}", id));
    }
    Ok(id)
}

/// Resolve an optional project argument quietly (no output).
///
/// Returns (project_id, was_auto_detected).
pub(crate) fn resolve_project_id_or_cwd_quiet(project: Option<&str>) -> Result<(String, bool)> {
    if let Some(p) = project {
        return Ok((resolve_project_id(p), false));
    }

    // Auto-detect from CWD
    let cwd =
        std::env::current_dir().context("Could not determine current directory")?;

    let db_path = crate::config::get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("Database not found: {}", e))?;

    match wqm_common::project_id::resolve_path_to_project(&db_path, &cwd) {
        Some((tenant_id, _path)) => Ok((tenant_id, true)),
        None => {
            anyhow::bail!(
                "Could not detect project from current directory.\n\
                 Run from within a registered project directory, or pass a project ID explicitly."
            );
        }
    }
}

/// Calculate the canonical project ID using the same algorithm as the daemon.
pub(crate) fn calculate_project_id(abs_path: &std::path::Path) -> String {
    use wqm_common::project_id::{detect_git_remote, ProjectIdCalculator};

    let git_remote = detect_git_remote(abs_path);
    let calculator = ProjectIdCalculator::new();
    calculator.calculate(abs_path, git_remote.as_deref(), None)
}
