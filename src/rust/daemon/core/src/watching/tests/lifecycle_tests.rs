//! Tests for project auto-watch lifecycle and watcher restart behavior

use super::tests::{create_test_task_submitter, test_watcher_config};
use super::*;
use shared_test_utils::*;
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

#[cfg(test)]
mod project_auto_watch_tests {
    use super::*;

    #[tokio::test]
    async fn test_watcher_can_be_configured_for_startup() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let (task_submitter, _pipeline) = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        let stats = watcher.stats().await;
        assert_eq!(stats.watched_paths, 0);

        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_handles_multiple_project_paths() -> TestResult {
        init_test_tracing();

        let temp_dir1 = TempDir::new()?;

        let config = test_watcher_config();
        let (task_submitter, _pipeline) = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        watcher.watch_path(temp_dir1.path()).await?;

        let watched = watcher.watched_paths().await;
        assert!(watched.contains(&temp_dir1.path().to_path_buf()));

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_survives_restart_cycle() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let (task_submitter, _pipeline) = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config.clone(), task_submitter.clone())?;

        // Start watching
        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        // Stop (simulating daemon restart)
        watcher.stop_watching().await?;
        assert_eq!(watcher.watched_paths().await.len(), 0);

        // Create new watcher (simulating daemon restart)
        let watcher2 = FileWatcher::new(config, task_submitter)?;

        watcher2.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher2.watched_paths().await.len(), 1);

        watcher2.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_handles_nested_project_structures() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let project_root = temp_dir.path();

        // Create nested project structure
        let src_dir = project_root.join("src");
        let tests_dir = project_root.join("tests");
        fs::create_dir_all(&src_dir)?;
        fs::create_dir_all(&tests_dir)?;

        let config = test_watcher_config();
        let (task_submitter, _pipeline) = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        watcher.watch_path(project_root).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create files in nested directories
        fs::write(src_dir.join("main.rs"), "fn main() {}")?;
        fs::write(tests_dir.join("test.rs"), "#[test] fn it_works() {}")?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = watcher.stats().await;
        assert!(
            stats.events_received > 0,
            "Should detect events in nested structures"
        );

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_process_existing_files_on_startup() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;

        // Create files before watcher starts
        fs::write(temp_dir.path().join("existing1.txt"), "File 1")?;
        fs::write(temp_dir.path().join("existing2.md"), "File 2")?;
        fs::write(temp_dir.path().join("existing3.rs"), "File 3")?;

        // Configure with process_existing enabled
        let mut config = test_watcher_config();
        config.process_existing = true;

        let (task_submitter, _pipeline) = create_test_task_submitter().await;
        let watcher = FileWatcher::new(config, task_submitter)?;

        watcher.watch_path(temp_dir.path()).await?;

        // Give time for initial scan to complete
        tokio::time::sleep(Duration::from_millis(1000)).await;

        let stats = watcher.stats().await;
        assert!(
            stats.tasks_submitted >= 3,
            "Should process existing files on startup"
        );

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_startup_with_invalid_paths() -> TestResult {
        init_test_tracing();

        let config = test_watcher_config();
        let (task_submitter, _pipeline) = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        let result = watcher.watch_path(Path::new("/nonexistent/project")).await;
        assert!(result.is_err(), "Should fail gracefully with invalid path");

        assert_eq!(watcher.watched_paths().await.len(), 0);

        Ok(())
    }
}
