//! Branch default for explicit cross-project tool calls (#99).
//!
//! `list`/`search` default their branch filter to the SESSION's current
//! branch. When an explicit `projectId` targets a DIFFERENT project, that
//! default belongs to the wrong repository and silently empties results
//! (e.g. a session on a `feature/x` branch listing a project whose rows are
//! all on `main`). These helpers resolve the TARGET project's branch instead.

use rusqlite::Connection;

/// Resolve the branch filter default for a tool call that explicitly targets
/// `watch_folder_id` / `project_path` (a project other than the session's).
///
/// Order:
/// 1. Detect the branch from the target's checkout (`detect_branch`, which
///    follows submodule/worktree `gitdir:` pointers).
/// 2. When the path is unknown or not a usable git checkout (`"default"`),
///    fall back to the most common `primary_branch` among the target's
///    tracked rows — always consistent with what a filter can match.
/// 3. `None` when neither resolves: callers omit the branch filter entirely,
///    which over-returns (all branches) rather than silently returning zero.
pub(crate) fn resolve_cross_project_branch(
    conn: Option<&Connection>,
    watch_folder_id: &str,
    project_path: Option<&str>,
) -> Option<String> {
    if let Some(path) = project_path {
        let detected = wqm_client::project::detect_branch(std::path::Path::new(path));
        if detected != "default" {
            return Some(detected);
        }
    }

    let conn = conn?;
    conn.query_row(
        "SELECT primary_branch FROM tracked_files \
         WHERE watch_folder_id = ?1 AND primary_branch IS NOT NULL \
         GROUP BY primary_branch ORDER BY COUNT(*) DESC LIMIT 1",
        [watch_folder_id],
        |row| row.get::<_, String>(0),
    )
    .ok()
}

/// `true` when the tool call's explicit `projectId` targets a project other
/// than the session's bound project.
pub(crate) fn is_cross_project(
    input_project_id: Option<&str>,
    session_project_id: Option<&str>,
) -> bool {
    match input_project_id {
        Some(p) => Some(p) != session_project_id,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db_with_rows(rows: &[(&str, &str)]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE tracked_files (
                 watch_folder_id TEXT NOT NULL,
                 primary_branch TEXT
             )",
        )
        .unwrap();
        for (wf, branch) in rows {
            conn.execute(
                "INSERT INTO tracked_files (watch_folder_id, primary_branch) VALUES (?1, ?2)",
                [wf, branch],
            )
            .unwrap();
        }
        conn
    }

    #[test]
    fn db_majority_fallback_used_when_no_path() {
        let conn = db_with_rows(&[("wf1", "main"), ("wf1", "main"), ("wf1", "dev")]);
        assert_eq!(
            resolve_cross_project_branch(Some(&conn), "wf1", None),
            Some("main".to_string())
        );
    }

    #[test]
    fn no_rows_no_path_yields_none() {
        let conn = db_with_rows(&[]);
        assert_eq!(resolve_cross_project_branch(Some(&conn), "wf1", None), None);
    }

    #[test]
    fn detected_branch_from_real_checkout_wins() {
        // A temp dir with a real .git dir pointing HEAD at a branch.
        let dir = tempfile::TempDir::new().unwrap();
        let git = dir.path().join(".git");
        std::fs::create_dir_all(&git).unwrap();
        std::fs::write(git.join("HEAD"), "ref: refs/heads/feature-z\n").unwrap();
        let conn = db_with_rows(&[("wf1", "main")]);
        assert_eq!(
            resolve_cross_project_branch(Some(&conn), "wf1", dir.path().to_str()),
            Some("feature-z".to_string())
        );
    }

    #[test]
    fn gitfile_submodule_pointer_followed() {
        // Layout: super/.git/modules/sub/HEAD + super/sub/.git (gitfile).
        let dir = tempfile::TempDir::new().unwrap();
        let modules = dir.path().join(".git").join("modules").join("sub");
        std::fs::create_dir_all(&modules).unwrap();
        std::fs::write(modules.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join(".git"), "gitdir: ../.git/modules/sub\n").unwrap();
        let conn = db_with_rows(&[]);
        assert_eq!(
            resolve_cross_project_branch(Some(&conn), "wf1", sub.to_str()),
            Some("main".to_string())
        );
    }

    #[test]
    fn cross_project_detection() {
        assert!(is_cross_project(Some("a"), Some("b")));
        assert!(is_cross_project(Some("a"), None));
        assert!(!is_cross_project(Some("a"), Some("a")));
        assert!(!is_cross_project(None, Some("b")));
        assert!(!is_cross_project(None, None));
    }
}
