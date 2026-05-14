//! Project ID resolution and disambiguation logic

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::paths::CanonicalPath;

use crate::output;

/// Make a path-like CLI argument absolute without touching the filesystem.
///
/// Per spec §3.1 rule 7 we do not call `std::fs::canonicalize`. Relative
/// inputs (e.g. `.`, `./foo`, `nonexistent.path`) are absolutized by
/// joining onto the process CWD — no symlink resolution. Returns `None`
/// when the input cannot be syntactically normalized into a canonical
/// form (relative without a CWD, `..` segments after absolutization,
/// non-UTF-8, …).
fn try_canonical_from_user_input(input: &str) -> Option<CanonicalPath> {
    // First try the fast path: input is already absolute / tilde-prefixed.
    if let Ok(cp) = CanonicalPath::from_user_input(input) {
        return Some(cp);
    }

    // Fall back: absolutize a relative input against the current working
    // directory. Purely syntactic — no fs canonicalization.
    let cwd = std::env::current_dir().ok()?;
    let mut joined = cwd;
    joined.push(input);
    let joined_str = joined.to_str()?;
    CanonicalPath::from_user_input(joined_str).ok()
}

/// Resolve a project argument to a project_id.
///
/// If the argument looks like a path (contains `/` or `.`), resolve it to an
/// absolute syntactically-canonical path and compute the project_id.
/// Otherwise use it as a direct ID.
pub(crate) fn resolve_project_id(project: &str) -> String {
    if project.contains('/') || project.contains('.') || project == "~" {
        match try_canonical_from_user_input(project) {
            Some(canonical) => calculate_project_id(std::path::Path::new(canonical.as_str())),
            None => project.to_string(),
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
/// Resolution order when `project` is Some:
/// 1. If it looks like a path (contains `/` or `.`), compute project_id from path
/// 2. Otherwise, try hint-based resolution (exact ID, exact path, name substring)
///
/// When `project` is None, auto-detects from the current working directory.
///
/// Returns (project_id, was_auto_detected).
pub(crate) fn resolve_project_id_or_cwd_quiet(project: Option<&str>) -> Result<(String, bool)> {
    if let Some(p) = project {
        // Path-like arguments: compute project_id directly
        if p.contains('/') || p.contains('.') || p == "~" {
            return Ok((resolve_project_id(p), false));
        }

        // Non-path argument: try hint-based resolution (ID, name substring)
        // If the database is unavailable (daemon not running), fall back to literal ID.
        let db_result = crate::config::get_database_path_checked()
            .ok()
            .and_then(|db_path| {
                rusqlite::Connection::open_with_flags(
                    &db_path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .ok()
            });

        if let Some(conn) = db_result {
            let _ = conn.execute_batch("PRAGMA busy_timeout=5000;");
            match resolve_tenant_by_hint(&conn, p) {
                Ok((tenant_id, _path)) => return Ok((tenant_id, false)),
                Err(_) => {
                    // Fall back to using the argument as a literal ID
                    return Ok((p.to_string(), false));
                }
            }
        } else {
            // No database available: use argument as literal ID
            return Ok((p.to_string(), false));
        }
    }

    // Auto-detect from CWD
    let cwd = std::env::current_dir().context("Could not determine current directory")?;

    let db_path = crate::config::get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("Database not found: {}", e))?;

    // Try direct path match first (works for main checkout and subdirectories)
    if let Some((tenant_id, _path)) =
        wqm_common::project_id::resolve_path_to_project(&db_path, &cwd)
    {
        return Ok((tenant_id, true));
    }

    // Fallback: compute project_id from git remote (works for worktrees)
    let project_id = calculate_project_id(&cwd);
    if let Ok(conn) =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
    {
        let _ = conn.execute_batch("PRAGMA busy_timeout=5000;");
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 LIMIT 1",
                rusqlite::params![&project_id],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if exists {
            return Ok((project_id, true));
        }
    }

    anyhow::bail!(
        "Could not detect project from current directory.\n\
         Run from within a registered project directory, or pass a project ID explicitly."
    );
}

/// Resolve a project hint (name, path, or tenant_id) to a `(tenant_id, path)` pair
/// by querying the `watch_folders` table.
///
/// Resolution order:
/// 1. Exact `tenant_id` match
/// 2. Exact `path` match
/// 3. Case-insensitive `path` substring match (e.g. "MunsellSpace")
///
/// Returns an error when no match is found or the hint is ambiguous.
pub(crate) fn resolve_tenant_by_hint(
    conn: &rusqlite::Connection,
    hint: &str,
) -> Result<(String, String)> {
    // 1. Exact tenant_id match
    let exact_tenant: Option<(String, String)> = conn
        .query_row(
            "SELECT tenant_id, path FROM watch_folders WHERE tenant_id = ?1 \
             AND parent_watch_id IS NULL LIMIT 1",
            rusqlite::params![hint],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();
    if let Some(result) = exact_tenant {
        return Ok(result);
    }

    // 2. Exact path match (syntactic-canonical form, no fs canonicalize).
    let path_str = match try_canonical_from_user_input(hint) {
        Some(c) => c.into_string(),
        None => PathBuf::from(hint).to_string_lossy().to_string(),
    };
    let exact_path: Option<(String, String)> = conn
        .query_row(
            "SELECT tenant_id, path FROM watch_folders WHERE path = ?1 \
             AND parent_watch_id IS NULL LIMIT 1",
            rusqlite::params![&path_str],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();
    if let Some(result) = exact_path {
        return Ok(result);
    }

    // 3. Case-insensitive path substring match
    let pattern = format!("%{}%", hint.to_lowercase());
    let mut stmt = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE lower(path) LIKE ?1 AND parent_watch_id IS NULL",
    )?;
    let matches: Vec<(String, String)> = stmt
        .query_map(rusqlite::params![&pattern], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    match matches.len() {
        0 => anyhow::bail!(
            "No registered project found matching {:?}.\n\
             Use a tenant ID, an absolute path, or a unique name substring.",
            hint
        ),
        1 => Ok(matches.into_iter().next().unwrap()),
        _ => {
            let paths: Vec<_> = matches.iter().map(|(_, p)| p.as_str()).collect();
            anyhow::bail!(
                "Ambiguous project hint {:?} — matched {} projects:\n  {}",
                hint,
                paths.len(),
                paths.join("\n  ")
            )
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
