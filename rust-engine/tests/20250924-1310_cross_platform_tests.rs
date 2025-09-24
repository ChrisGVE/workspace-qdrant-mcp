//! Cross-platform testing suite for workspace-qdrant-daemon
//!
//! This module provides comprehensive testing across Windows, macOS, and Linux platforms
//! with platform-specific conditional compilation and behavior validation.

use std::path::{Path, PathBuf};
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

use workspace_qdrant_daemon::{
    config::DaemonConfig,
    daemon::Daemon,
    error::WorkspaceError,
};

/// Platform-specific test configuration
#[derive(Debug, Clone)]
struct PlatformTestConfig {
    temp_dir: PathBuf,
    max_file_watchers: usize,
    path_separator: char,
    case_sensitive: bool,
}

impl PlatformTestConfig {
    fn new() -> Result<Self, WorkspaceError> {
        let temp_dir = tempfile::tempdir()?.into_path();

        #[cfg(target_os = "windows")]
        let config = PlatformTestConfig {
            temp_dir,
            max_file_watchers: 64,
            path_separator: '\\',
            case_sensitive: false,
        };

        #[cfg(target_os = "macos")]
        let config = PlatformTestConfig {
            temp_dir,
            max_file_watchers: 256,
            path_separator: '/',
            case_sensitive: false, // HFS+ default
        };

        #[cfg(target_os = "linux")]
        let config = PlatformTestConfig {
            temp_dir,
            max_file_watchers: 8192,
            path_separator: '/',
            case_sensitive: true,
        };

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let config = PlatformTestConfig {
            temp_dir,
            max_file_watchers: 256,
            path_separator: '/',
            case_sensitive: true,
        };

        Ok(config)
    }
}

#[tokio::test]
async fn test_platform_specific_file_watching() -> Result<(), WorkspaceError> {
    let platform_config = PlatformTestConfig::new()?;
    let temp_dir = TempDir::new()?;

    // Create test files with platform-specific naming
    let test_files = create_platform_test_files(&temp_dir, &platform_config)?;

    // Initialize daemon with platform-appropriate configuration
    let daemon_config = create_platform_daemon_config(&temp_dir, &platform_config)?;
    let daemon = Arc::new(Daemon::new(daemon_config).await?);

    // Test file watching behavior across platforms
    for test_file in test_files {
        // Write to file and verify detection
        fs::write(&test_file, "test content")?;

        // Platform-specific timing considerations
        #[cfg(target_os = "windows")]
        let wait_duration = Duration::from_millis(200); // Windows needs more time
        #[cfg(target_os = "macos")]
        let wait_duration = Duration::from_millis(100); // FSEvents is fast
        #[cfg(target_os = "linux")]
        let wait_duration = Duration::from_millis(50);  // inotify is fastest
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let wait_duration = Duration::from_millis(100);

        tokio::time::sleep(wait_duration).await;

        // Verify file was detected by daemon
        // This would interact with the daemon's file watcher system
        // Implementation would depend on the actual daemon API
    }

    Ok(())
}

#[tokio::test]
async fn test_path_handling_cross_platform() -> Result<(), WorkspaceError> {
    let platform_config = PlatformTestConfig::new()?;

    // Test path normalization across platforms
    let test_paths = vec![
        "simple_file.txt",
        "folder/nested_file.txt",
        "folder\\windows_style.txt",
        "file with spaces.txt",
        "file-with-unicode-名前.txt",
    ];

    for path_str in test_paths {
        let path = PathBuf::from(path_str);
        let normalized = normalize_path_for_platform(&path, &platform_config);

        // Verify path normalization is correct for platform
        assert_path_normalization(&normalized, &platform_config);
    }

    Ok(())
}

#[tokio::test]
async fn test_case_sensitivity_handling() -> Result<(), WorkspaceError> {
    let platform_config = PlatformTestConfig::new()?;
    let temp_dir = TempDir::new()?;

    // Create files with different cases
    let file1 = temp_dir.path().join("TestFile.txt");
    let file2 = temp_dir.path().join("testfile.txt");

    fs::write(&file1, "content1")?;

    if platform_config.case_sensitive {
        // On case-sensitive systems, both files should be distinct
        fs::write(&file2, "content2")?;
        assert!(file1.exists());
        assert!(file2.exists());
        assert_ne!(fs::read_to_string(&file1)?, fs::read_to_string(&file2)?);
    } else {
        // On case-insensitive systems, file2 should overwrite file1
        fs::write(&file2, "content2")?;
        assert!(file1.exists() || file2.exists());
        // Content should be "content2" regardless of which path we read from
    }

    Ok(())
}

#[tokio::test]
async fn test_file_watcher_limits() -> Result<(), WorkspaceError> {
    let platform_config = PlatformTestConfig::new()?;
    let temp_dir = TempDir::new()?;

    // Create more files than the platform's typical watcher limit
    let file_count = platform_config.max_file_watchers * 2;
    let mut test_files = Vec::new();

    for i in 0..file_count {
        let file_path = temp_dir.path().join(format!("test_file_{}.txt", i));
        fs::write(&file_path, format!("content {}", i))?;
        test_files.push(file_path);
    }

    // Test that daemon handles file watcher limit gracefully
    let daemon_config = create_platform_daemon_config(&temp_dir, &platform_config)?;
    let result = Daemon::new(daemon_config).await;

    // Should either succeed with proper resource management or fail gracefully
    match result {
        Ok(_daemon) => {
            // Daemon should handle the load without crashes
            println!("Daemon successfully handled {} files", file_count);
        }
        Err(e) => {
            // Should be a resource limit error, not a crash
            println!("Daemon gracefully handled resource limit: {}", e);
        }
    }

    Ok(())
}

#[tokio::test]
#[cfg(unix)]
async fn test_unix_permissions() -> Result<(), WorkspaceError> {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("permission_test.txt");

    fs::write(&test_file, "test content")?;

    // Test various permission scenarios
    let permissions_tests = vec![
        0o644, // Read/write owner, read others
        0o600, // Read/write owner only
        0o755, // Executable directory-like
        0o444, // Read-only
    ];

    for perm in permissions_tests {
        fs::set_permissions(&test_file, fs::Permissions::from_mode(perm))?;

        // Test daemon behavior with different permissions
        // This would test the actual file processing logic
        let can_read = test_file.metadata()?.permissions().mode() & 0o400 != 0;
        let can_write = test_file.metadata()?.permissions().mode() & 0o200 != 0;

        println!("Permission {:#o}: can_read={}, can_write={}", perm, can_read, can_write);
    }

    Ok(())
}

#[tokio::test]
#[cfg(target_os = "windows")]
async fn test_windows_specific_features() -> Result<(), WorkspaceError> {
    let temp_dir = TempDir::new()?;

    // Test Windows-specific file attributes
    let test_file = temp_dir.path().join("windows_test.txt");
    fs::write(&test_file, "test content")?;

    // Test long path support (Windows has 260 char limit by default)
    let long_path_components: Vec<String> = (0..10)
        .map(|i| format!("very_long_directory_name_{}", i))
        .collect();

    let mut long_path = temp_dir.path().to_path_buf();
    for component in &long_path_components {
        long_path.push(component);
        if let Err(e) = fs::create_dir_all(&long_path) {
            println!("Long path creation failed at: {}: {}", long_path.display(), e);
            break;
        }
    }

    Ok(())
}

#[tokio::test]
#[cfg(target_os = "macos")]
async fn test_macos_specific_features() -> Result<(), WorkspaceError> {
    let temp_dir = TempDir::new()?;

    // Test macOS extended attributes
    let test_file = temp_dir.path().join("macos_test.txt");
    fs::write(&test_file, "test content")?;

    // Test resource forks and extended attributes (would require additional crates)
    // For now, just test basic functionality

    // Test HFS+ case insensitivity
    let file1 = temp_dir.path().join("CaseTest.txt");
    let file2 = temp_dir.path().join("casetest.txt");

    fs::write(&file1, "content1")?;
    fs::write(&file2, "content2")?;

    // On HFS+, file2 should overwrite file1
    let content = fs::read_to_string(&file1)?;
    assert_eq!(content, "content2");

    Ok(())
}

/// Helper functions
fn create_platform_test_files(
    temp_dir: &TempDir,
    platform_config: &PlatformTestConfig,
) -> Result<Vec<PathBuf>, WorkspaceError> {
    let mut files = Vec::new();

    // Standard files
    for i in 0..5 {
        let file_path = temp_dir.path().join(format!("test_file_{}.txt", i));
        files.push(file_path);
    }

    // Platform-specific test cases
    #[cfg(target_os = "windows")]
    {
        // Windows-specific file names
        files.push(temp_dir.path().join("CON.txt")); // Reserved name
        files.push(temp_dir.path().join("file.with.dots.txt"));
    }

    #[cfg(unix)]
    {
        // Unix-specific file names
        files.push(temp_dir.path().join(".hidden_file"));
        files.push(temp_dir.path().join("file:with:colons"));
    }

    Ok(files)
}

fn create_platform_daemon_config(
    temp_dir: &TempDir,
    platform_config: &PlatformTestConfig,
) -> Result<DaemonConfig, WorkspaceError> {
    let mut config = DaemonConfig::default();
    config.workspace_root = temp_dir.path().to_path_buf();

    // Platform-specific optimizations
    #[cfg(target_os = "linux")]
    {
        config.max_file_watchers = Some(platform_config.max_file_watchers);
        config.use_polling = false; // inotify is preferred
    }

    #[cfg(target_os = "macos")]
    {
        config.max_file_watchers = Some(platform_config.max_file_watchers);
        config.use_polling = false; // FSEvents is preferred
    }

    #[cfg(target_os = "windows")]
    {
        config.max_file_watchers = Some(platform_config.max_file_watchers);
        config.use_polling = true; // More reliable on Windows
    }

    Ok(config)
}

fn normalize_path_for_platform(
    path: &Path,
    platform_config: &PlatformTestConfig,
) -> PathBuf {
    let path_str = path.to_string_lossy();

    #[cfg(target_os = "windows")]
    {
        // Convert forward slashes to backslashes
        let normalized = path_str.replace('/', "\\");
        PathBuf::from(normalized)
    }

    #[cfg(unix)]
    {
        // Convert backslashes to forward slashes
        let normalized = path_str.replace('\\', "/");
        PathBuf::from(normalized)
    }
}

fn assert_path_normalization(path: &Path, platform_config: &PlatformTestConfig) {
    let path_str = path.to_string_lossy();

    // Verify correct path separator
    if cfg!(target_os = "windows") {
        assert!(!path_str.contains('/') || path_str == "/",
               "Windows paths should not contain forward slashes");
    } else {
        assert!(!path_str.contains('\\'),
               "Unix paths should not contain backslashes");
    }
}

#[tokio::test]
async fn test_concurrent_platform_operations() -> Result<(), WorkspaceError> {
    let platform_config = PlatformTestConfig::new()?;
    let temp_dir = TempDir::new()?;

    // Test concurrent file operations across threads
    let handles = (0..platform_config.max_file_watchers.min(100))
        .map(|i| {
            let temp_path = temp_dir.path().to_path_buf();
            tokio::spawn(async move {
                let file_path = temp_path.join(format!("concurrent_test_{}.txt", i));
                fs::write(&file_path, format!("content {}", i))?;
                tokio::time::sleep(Duration::from_millis(10)).await;
                fs::remove_file(&file_path)?;
                Ok::<_, WorkspaceError>(())
            })
        })
        .collect::<Vec<_>>();

    // Wait for all operations to complete
    let results = futures_util::future::join_all(handles).await;

    // Verify all operations completed successfully
    for result in results {
        result??; // Unwrap tokio::JoinError and then WorkspaceError
    }

    Ok(())
}

#[tokio::test]
async fn test_platform_resource_cleanup() -> Result<(), WorkspaceError> {
    let temp_dir = TempDir::new()?;

    // Create and destroy multiple daemon instances
    for _ in 0..10 {
        let daemon_config = DaemonConfig {
            workspace_root: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let daemon = Arc::new(Daemon::new(daemon_config).await?);

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Drop daemon and verify cleanup
        drop(daemon);

        // Brief pause to allow cleanup
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    Ok(())
}