//! Log file auto-pruning.
//!
//! Prunes rotated and old log files from the canonical log directory,
//! keeping files newer than the configured retention period.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::daemon_state::{get_operational_state, set_operational_state, DaemonStateResult};

/// Key used in the operational_state table to track last prune time.
const STATE_KEY: &str = "last_log_prune";
/// Component identifier for daemon-initiated pruning.
const COMPONENT: &str = "daemon";

/// Result of a prune operation.
#[derive(Debug, Default)]
pub struct PruneResult {
    /// Number of files deleted.
    pub files_deleted: usize,
    /// Total bytes freed.
    pub bytes_freed: u64,
    /// Files that would be deleted (dry-run only).
    pub candidates: Vec<PruneCandidate>,
}

/// A file eligible for pruning.
#[derive(Debug, Clone)]
pub struct PruneCandidate {
    pub path: PathBuf,
    pub size: u64,
    pub age_hours: f64,
}

/// Check if pruning is due and run it if so.
///
/// Uses the `operational_state` table to track the last prune time.
/// Skips if the last prune was less than `check_interval_hours` ago.
pub async fn run_if_due(
    pool: &SqlitePool,
    log_dir: &Path,
    retention_hours: u64,
    check_interval_hours: u64,
) -> DaemonStateResult<Option<PruneResult>> {
    let last_prune = get_operational_state(pool, STATE_KEY, COMPONENT, None).await?;

    if let Some(ts) = &last_prune {
        if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(ts) {
            let elapsed = chrono::Utc::now() - parsed.to_utc();
            let elapsed_hours = elapsed.num_seconds() as f64 / 3600.0;
            if elapsed_hours < check_interval_hours as f64 {
                debug!(
                    elapsed_hours = elapsed_hours,
                    interval = check_interval_hours,
                    "Log pruning not due yet"
                );
                return Ok(None);
            }
        }
    }

    let result = prune_now(log_dir, retention_hours, false)?;

    // Record the prune time
    let now = wqm_common::timestamps::now_utc();
    set_operational_state(pool, STATE_KEY, COMPONENT, &now, None).await?;

    Ok(Some(result))
}

/// Prune log files older than `retention_hours`.
///
/// In dry-run mode, populates `candidates` without deleting.
/// Active log files (`daemon.jsonl`, `mcp-server.jsonl`, `workspace.log`)
/// are never deleted — only rotated/compressed variants.
pub fn prune_now(
    log_dir: &Path,
    retention_hours: u64,
    dry_run: bool,
) -> std::io::Result<PruneResult> {
    let mut result = PruneResult::default();

    if !log_dir.exists() {
        return Ok(result);
    }

    let cutoff = SystemTime::now() - Duration::from_secs(retention_hours * 3600);

    let entries = std::fs::read_dir(log_dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Never prune active log files (the ones currently being written to)
        if is_active_log(&path) {
            continue;
        }

        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let modified = match metadata.modified() {
            Ok(t) => t,
            Err(_) => continue,
        };

        if modified < cutoff {
            let size = metadata.len();
            let age_hours = SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default()
                .as_secs_f64()
                / 3600.0;

            let candidate = PruneCandidate {
                path: path.clone(),
                size,
                age_hours,
            };

            if dry_run {
                result.candidates.push(candidate);
            } else {
                match std::fs::remove_file(&path) {
                    Ok(()) => {
                        info!(path = %path.display(), size, "Pruned old log file");
                        result.files_deleted += 1;
                        result.bytes_freed += size;
                        result.candidates.push(candidate);
                    }
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Failed to prune log file");
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Returns true if this is an active (currently written) log file.
///
/// Active files are: `daemon.jsonl`, `mcp-server.jsonl`, `workspace.log`
/// Rotated files like `daemon.jsonl.1.gz`, `workspace.2026-01-24_*.log.gz` are pruneable.
fn is_active_log(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };
    matches!(name, "daemon.jsonl" | "mcp-server.jsonl" | "workspace.log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_file_with_age(dir: &Path, name: &str, age_hours: u64) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, "test log content\n").unwrap();
        // Set mtime to the past
        let mtime = filetime::FileTime::from_system_time(
            SystemTime::now() - Duration::from_secs(age_hours * 3600),
        );
        filetime::set_file_mtime(&path, mtime).unwrap();
        path
    }

    #[test]
    fn test_prune_old_files() {
        let dir = tempdir().unwrap();
        let log_dir = dir.path();

        // Create files: one old rotated, one recent rotated, one active
        create_file_with_age(log_dir, "daemon.jsonl.1.gz", 48);
        create_file_with_age(log_dir, "daemon.jsonl.2.gz", 12);
        create_file_with_age(log_dir, "daemon.jsonl", 48); // active — never pruned

        let result = prune_now(log_dir, 36, false).unwrap();
        assert_eq!(result.files_deleted, 1, "Should delete only the 48h old rotated file");
        assert!(!log_dir.join("daemon.jsonl.1.gz").exists());
        assert!(log_dir.join("daemon.jsonl.2.gz").exists());
        assert!(log_dir.join("daemon.jsonl").exists());
    }

    #[test]
    fn test_prune_dry_run() {
        let dir = tempdir().unwrap();
        let log_dir = dir.path();

        create_file_with_age(log_dir, "old.log.gz", 48);
        create_file_with_age(log_dir, "recent.log.gz", 12);

        let result = prune_now(log_dir, 36, true).unwrap();
        assert_eq!(result.files_deleted, 0, "Dry run should not delete");
        assert_eq!(result.candidates.len(), 1, "Should identify 1 candidate");
        assert!(log_dir.join("old.log.gz").exists(), "File should still exist in dry run");
    }

    #[test]
    fn test_active_logs_never_pruned() {
        let dir = tempdir().unwrap();
        let log_dir = dir.path();

        // All active logs, even if old
        create_file_with_age(log_dir, "daemon.jsonl", 100);
        create_file_with_age(log_dir, "mcp-server.jsonl", 100);
        create_file_with_age(log_dir, "workspace.log", 100);

        let result = prune_now(log_dir, 36, false).unwrap();
        assert_eq!(result.files_deleted, 0);
    }

    #[test]
    fn test_prune_nonexistent_dir() {
        let result = prune_now(Path::new("/nonexistent/log/dir"), 36, false).unwrap();
        assert_eq!(result.files_deleted, 0);
    }

    #[tokio::test]
    async fn test_run_if_due_skips_when_recent() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("prune_due.db");
        let manager = crate::daemon_state::DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();
        let pool = manager.pool();

        let log_dir = temp_dir.path().join("logs");
        fs::create_dir_all(&log_dir).unwrap();
        create_file_with_age(&log_dir, "old.log.gz", 48);

        // First call should prune (no previous timestamp)
        let result = run_if_due(pool, &log_dir, 36, 12).await.unwrap();
        assert!(result.is_some(), "First run should prune");
        assert_eq!(result.unwrap().files_deleted, 1);

        // Create another old file
        create_file_with_age(&log_dir, "old2.log.gz", 48);

        // Second call immediately should skip (interval not elapsed)
        let result = run_if_due(pool, &log_dir, 36, 12).await.unwrap();
        assert!(result.is_none(), "Should skip — interval not elapsed");

        // File should still exist
        assert!(log_dir.join("old2.log.gz").exists());
    }

    #[test]
    fn test_is_active_log() {
        assert!(is_active_log(Path::new("/logs/daemon.jsonl")));
        assert!(is_active_log(Path::new("/logs/mcp-server.jsonl")));
        assert!(is_active_log(Path::new("/logs/workspace.log")));
        assert!(!is_active_log(Path::new("/logs/daemon.jsonl.1.gz")));
        assert!(!is_active_log(Path::new("/logs/workspace.2026-01-24.log.gz")));
        assert!(!is_active_log(Path::new("/logs/mcp-server.1.jsonl")));
    }
}
