//! Project ID calculation and disambiguation
//!
//! Provides the canonical project ID calculation algorithm used by both
//! the daemon and CLI to ensure consistent tenant IDs.

mod calculator;
mod disambiguation;
mod types;
mod utils;

pub use calculator::ProjectIdCalculator;
pub use disambiguation::DisambiguationPathComputer;
pub use types::DisambiguationConfig;
pub use utils::{calculate_tenant_id, detect_git_remote};

#[cfg(feature = "sqlite")]
pub use utils::resolve_path_to_project;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn test_normalize_git_url_https() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("https://github.com/user/repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_ssh() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("git@github.com:user/repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_http() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("http://github.com/user/repo"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_case_insensitive() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("https://GitHub.COM/User/Repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_calculate_project_id_with_remote() {
        let calc = ProjectIdCalculator::new();
        let id = calc.calculate(
            Path::new("/home/user/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        assert_eq!(id.len(), 12);
        assert!(!id.starts_with("local_"));
    }

    #[test]
    fn test_calculate_project_id_local() {
        let calc = ProjectIdCalculator::new();
        let id = calc.calculate(Path::new("/home/user/project"), None, None);

        assert!(id.starts_with("local_"));
        assert_eq!(id.len(), 6 + 12); // "local_" + 12 char hash
    }

    #[test]
    fn test_calculate_project_id_with_disambiguation() {
        let calc = ProjectIdCalculator::new();

        let id1 = calc.calculate(
            Path::new("/home/user/work/project"),
            Some("https://github.com/user/repo.git"),
            Some("work/project"),
        );

        let id2 = calc.calculate(
            Path::new("/home/user/personal/project"),
            Some("https://github.com/user/repo.git"),
            Some("personal/project"),
        );

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_same_remote_same_id_without_disambiguation() {
        let calc = ProjectIdCalculator::new();

        let id1 = calc.calculate(
            Path::new("/home/user/work/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        let id2 = calc.calculate(
            Path::new("/home/user/personal/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_disambiguation_path_compute_single() {
        let new_path = Path::new("/home/user/work/project");
        let existing: Vec<PathBuf> = vec![];

        let disambig = DisambiguationPathComputer::compute(new_path, &existing);
        assert!(disambig.is_empty());
    }

    #[test]
    fn test_disambiguation_path_compute_two_clones() {
        let new_path = Path::new("/home/user/work/project");
        let existing = vec![PathBuf::from("/home/user/personal/project")];

        let disambig = DisambiguationPathComputer::compute(new_path, &existing);
        assert_eq!(disambig, "work/project");
    }

    #[test]
    fn test_recompute_all_disambiguation() {
        let paths = vec![
            PathBuf::from("/home/user/work/project"),
            PathBuf::from("/home/user/personal/project"),
        ];

        let result = DisambiguationPathComputer::recompute_all(&paths);

        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&paths[0]).unwrap(), "work/project");
        assert_eq!(result.get(&paths[1]).unwrap(), "personal/project");
    }

    #[test]
    fn test_remote_hash_grouping() {
        let calc = ProjectIdCalculator::new();

        let hash1 = calc.calculate_remote_hash("https://github.com/user/repo.git");
        let hash2 = calc.calculate_remote_hash("git@github.com:user/repo.git");
        let hash3 = calc.calculate_remote_hash("http://GITHUB.COM/User/Repo");

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    // ─── resolve_path_to_project tests ──────────────────────────────────

    /// Helper: create a SQLite database with watch_folders table and rows
    #[cfg(feature = "sqlite")]
    fn setup_test_db(rows: &[(&str, &str)]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                tenant_id TEXT NOT NULL,
                path TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'projects'
            )",
        )
        .unwrap();
        for (tenant, path) in rows {
            conn.execute(
                "INSERT INTO watch_folders (tenant_id, path, collection) VALUES (?1, ?2, 'projects')",
                rusqlite::params![tenant, path],
            )
            .unwrap();
        }
        (dir, db_path)
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_exact() {
        let (_dir, db_path) = setup_test_db(&[("tid_abc", "/home/user/project-a")]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a"));
        assert_eq!(
            result,
            Some(("tid_abc".into(), "/home/user/project-a".into()))
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_subdirectory() {
        let (_dir, db_path) = setup_test_db(&[("tid_abc", "/home/user/project-a")]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a/src/lib"));
        assert_eq!(
            result,
            Some(("tid_abc".into(), "/home/user/project-a".into()))
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_longest_wins() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_parent", "/home/user"),
            ("tid_child", "/home/user/project-a"),
        ]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a/src"));
        assert_eq!(
            result,
            Some(("tid_child".into(), "/home/user/project-a".into()))
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_no_match() {
        let (_dir, db_path) = setup_test_db(&[("tid_abc", "/home/user/project-a")]);
        let result = resolve_path_to_project(&db_path, Path::new("/other/dir"));
        assert_eq!(result, None);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_no_false_prefix() {
        let (_dir, db_path) = setup_test_db(&[("tid_abc", "/home/user/project")]);
        // "/home/user/project-extra" should NOT match "/home/user/project"
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-extra"));
        assert_eq!(result, None);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_missing_db() {
        let result = resolve_path_to_project(
            Path::new("/nonexistent/state.db"),
            Path::new("/home/user/project"),
        );
        assert_eq!(result, None);
    }
}
