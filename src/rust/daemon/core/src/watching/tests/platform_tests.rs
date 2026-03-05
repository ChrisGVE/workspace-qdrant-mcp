//! Platform-specific tests for macOS, Windows, and Linux watchers

use shared_test_utils::TestResult;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Platform-specific tests for macOS FSEvents watcher
#[cfg(target_os = "macos")]
mod platform_macos_tests {
    use super::*;
    use crate::watching::platform::{MacOSConfig, MacOSWatcher, PlatformWatcher};

    fn test_macos_config() -> MacOSConfig {
        MacOSConfig {
            latency: 0.1,
            use_kqueue: false,
            stream_flags: 0,
            watch_file_events: true,
            watch_dir_events: true,
        }
    }

    #[tokio::test]
    async fn test_macos_watcher_creation() -> TestResult<()> {
        let config = test_macos_config();
        let watcher = MacOSWatcher::new(config, 4096)?;

        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_macos_watcher_watch_and_stop() -> TestResult<()> {
        let config = test_macos_config();
        let mut watcher = MacOSWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;
        let watch_path = temp_dir.path();

        // Start watching
        watcher.watch(watch_path).await?;
        assert!(watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 1);

        // Stop watching
        watcher.stop().await?;
        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_macos_watcher_nonexistent_path() -> TestResult<()> {
        let config = test_macos_config();
        let mut watcher = MacOSWatcher::new(config, 4096)?;

        let result = watcher
            .watch(Path::new("/nonexistent/path/that/does/not/exist"))
            .await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_macos_watcher_symlink_resolution() -> TestResult<()> {
        let config = test_macos_config();
        let mut watcher = MacOSWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;
        let actual_dir = temp_dir.path().join("actual");
        fs::create_dir(&actual_dir)?;

        // Create a symlink
        let symlink_path = temp_dir.path().join("symlink");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&actual_dir, &symlink_path)?;

        // Watch the symlink - should resolve to actual path
        watcher.watch(&symlink_path).await?;
        assert!(watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 1);

        watcher.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_macos_watcher_event_receiver() -> TestResult<()> {
        let config = test_macos_config();
        let mut watcher = MacOSWatcher::new(config, 4096)?;

        // Take the event receiver before watching
        let receiver = watcher.take_event_receiver();
        assert!(receiver.is_some());

        // Second call should return None
        let receiver2 = watcher.take_event_receiver();
        assert!(receiver2.is_none());

        Ok(())
    }
}

/// Platform-specific tests for Windows ReadDirectoryChangesW watcher
#[cfg(target_os = "windows")]
mod platform_windows_tests {
    use super::*;
    use crate::watching::platform::{PlatformWatcher, WindowsConfig, WindowsWatcher};

    fn test_windows_config() -> WindowsConfig {
        WindowsConfig {
            watch_subtree: true,
            buffer_size: 65536,
            filter_flags: 0,
            use_completion_ports: true,
            monitor_file_name: true,
            monitor_dir_name: true,
            monitor_size: true,
            monitor_last_write: true,
        }
    }

    #[tokio::test]
    async fn test_windows_watcher_creation() -> TestResult<()> {
        let config = test_windows_config();
        let watcher = WindowsWatcher::new(config, 4096)?;

        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_windows_watcher_watch_and_stop() -> TestResult<()> {
        let config = test_windows_config();
        let mut watcher = WindowsWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;
        let watch_path = temp_dir.path();

        // Start watching
        watcher.watch(watch_path).await?;
        assert!(watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 1);

        // Stop watching
        watcher.stop().await?;
        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_windows_watcher_nonexistent_path() -> TestResult<()> {
        let config = test_windows_config();
        let mut watcher = WindowsWatcher::new(config, 4096)?;

        let result = watcher
            .watch(Path::new("C:\\nonexistent\\path\\that\\does\\not\\exist"))
            .await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_windows_watcher_event_receiver() -> TestResult<()> {
        let config = test_windows_config();
        let mut watcher = WindowsWatcher::new(config, 4096)?;

        // Take the event receiver before watching
        let receiver = watcher.take_event_receiver();
        assert!(receiver.is_some());

        // Second call should return None
        let receiver2 = watcher.take_event_receiver();
        assert!(receiver2.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_windows_watcher_non_recursive() -> TestResult<()> {
        let mut config = test_windows_config();
        config.watch_subtree = false; // Non-recursive mode

        let mut watcher = WindowsWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;
        let watch_path = temp_dir.path();

        // Create a subdirectory
        let subdir = watch_path.join("subdir");
        fs::create_dir(&subdir)?;

        // Watch non-recursively
        watcher.watch(watch_path).await?;
        assert!(watcher.is_active());

        watcher.stop().await?;
        Ok(())
    }
}

/// Platform-specific tests for Linux inotify watcher
#[cfg(target_os = "linux")]
mod platform_linux_tests {
    use super::*;
    use crate::watching::platform::{LinuxConfig, LinuxWatcher, PlatformWatcher};

    fn test_linux_config() -> LinuxConfig {
        LinuxConfig {
            use_epoll: false,
            buffer_size: 4096,
            max_watches: 8192,
            track_moves: true,
            monitor_create: true,
            monitor_modify: true,
            monitor_delete: true,
        }
    }

    #[tokio::test]
    async fn test_linux_watcher_creation() -> TestResult<()> {
        let config = test_linux_config();
        let watcher = LinuxWatcher::new(config, 4096)?;

        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_watch_and_stop() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;

        // Start watching
        watcher.watch(temp_dir.path()).await?;
        assert!(watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 1);

        // Stop watching
        watcher.stop().await?;
        assert!(!watcher.is_active());
        assert_eq!(watcher.watched_path_count(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_nonexistent_path() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let result = watcher
            .watch(Path::new("/nonexistent/path/that/does/not/exist"))
            .await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_detects_file_create() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let mut receiver = watcher
            .take_event_receiver()
            .expect("Should have event receiver");

        let temp_dir = TempDir::new()?;
        watcher.watch(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create a file
        let test_file = temp_dir.path().join("test_create.txt");
        fs::write(&test_file, "hello inotify")?;

        // Wait for event
        let event = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .expect("Timed out waiting for create event")
            .expect("Channel closed");

        assert_eq!(event.path, test_file);
        assert!(
            matches!(event.event_kind, EventKind::Create(_)),
            "Expected Create event, got {:?}",
            event.event_kind
        );
        assert_eq!(
            event.metadata.get("platform").map(|s| s.as_str()),
            Some("linux")
        );

        watcher.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_detects_file_modify() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let mut receiver = watcher
            .take_event_receiver()
            .expect("Should have event receiver");

        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("test_modify.txt");
        fs::write(&test_file, "initial content")?;

        watcher.watch(temp_dir.path()).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Modify the file
        fs::write(&test_file, "modified content")?;

        // Wait for event (may receive MODIFY and/or CLOSE_WRITE)
        let event = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .expect("Timed out waiting for modify event")
            .expect("Channel closed");

        assert_eq!(event.path, test_file);
        assert!(
            matches!(event.event_kind, EventKind::Modify(_)),
            "Expected Modify event, got {:?}",
            event.event_kind
        );

        watcher.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_detects_file_rename() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let mut receiver = watcher
            .take_event_receiver()
            .expect("Should have event receiver");

        let temp_dir = TempDir::new()?;
        let old_file = temp_dir.path().join("old_name.txt");
        let new_file = temp_dir.path().join("new_name.txt");
        fs::write(&old_file, "content to rename")?;

        watcher.watch(temp_dir.path()).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Rename the file
        fs::rename(&old_file, &new_file)?;

        // Should receive MOVED_FROM and MOVED_TO events
        let mut got_moved_from = false;
        let mut got_moved_to = false;

        for _ in 0..10 {
            match tokio::time::timeout(Duration::from_secs(2), receiver.recv()).await {
                Ok(Some(event)) => {
                    if matches!(
                        event.event_kind,
                        EventKind::Modify(notify::event::ModifyKind::Name(
                            notify::event::RenameMode::From
                        ))
                    ) {
                        got_moved_from = true;
                    }
                    if matches!(
                        event.event_kind,
                        EventKind::Modify(notify::event::ModifyKind::Name(
                            notify::event::RenameMode::To
                        ))
                    ) {
                        got_moved_to = true;
                    }
                    if got_moved_from && got_moved_to {
                        break;
                    }
                }
                _ => break,
            }
        }

        assert!(got_moved_from, "Should detect MOVED_FROM event");
        assert!(got_moved_to, "Should detect MOVED_TO event");

        watcher.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_stop_terminates() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        let temp_dir = TempDir::new()?;
        watcher.watch(temp_dir.path()).await?;

        assert!(watcher.is_active());

        // Stop should complete within a reasonable time
        let result = tokio::time::timeout(Duration::from_secs(10), watcher.stop()).await;

        assert!(result.is_ok(), "stop() should complete within 10 seconds");
        assert!(!watcher.is_active());

        Ok(())
    }

    #[tokio::test]
    async fn test_linux_watcher_event_receiver() -> TestResult<()> {
        let config = test_linux_config();
        let mut watcher = LinuxWatcher::new(config, 4096)?;

        // Take the event receiver before watching
        let receiver = watcher.take_event_receiver();
        assert!(receiver.is_some());

        // Second call should return None
        let receiver2 = watcher.take_event_receiver();
        assert!(receiver2.is_none());

        Ok(())
    }
}
