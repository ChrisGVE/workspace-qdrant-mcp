//! Comprehensive cross-platform file system compatibility tests
//!
//! This test suite validates file system operations across Windows, macOS, and Linux,
//! testing platform-specific features like case sensitivity, path separators, permissions,
//! symlinks, hard links, and junction points.

use workspace_qdrant_daemon::daemon::fs_compat::*;
use workspace_qdrant_daemon::error::DaemonError;
use tempfile::{tempdir, TempDir};
use std::path::PathBuf;
use tokio_test;
use std::time::{Duration, SystemTime};

/// Test suite for cross-platform file system compatibility
struct CrossPlatformFsTestSuite {
    fs_compat: FsCompat,
    temp_dir: TempDir,
}

impl CrossPlatformFsTestSuite {
    fn new() -> Self {
        Self {
            fs_compat: FsCompat::default(),
            temp_dir: tempdir().unwrap(),
        }
    }
    
    fn temp_path(&self, name: &str) -> PathBuf {
        self.temp_dir.path().join(name)
    }
    
    async fn create_test_file(&self, name: &str, content: &[u8]) -> PathBuf {
        let path = self.temp_path(name);
        tokio::fs::write(&path, content).await.unwrap();
        path
    }
    
    async fn create_test_dir(&self, name: &str) -> PathBuf {
        let path = self.temp_path(name);
        tokio::fs::create_dir(&path).await.unwrap();
        path
    }
}

// === Platform Detection Tests ===

#[tokio::test]
async fn test_platform_specific_defaults() {
    let fs_compat = FsCompat::default();
    
    // Test platform-specific path length limits
    #[cfg(windows)]
    {
        assert_eq!(fs_compat.max_path_length, 260);
        assert!(fs_compat.long_path_support);
    }
    
    #[cfg(not(windows))]
    {
        assert_eq!(fs_compat.max_path_length, 4096);
        assert!(!fs_compat.long_path_support);
    }
    
    assert!(fs_compat.normalize_unicode);
}

#[tokio::test]
async fn test_case_sensitivity_detection() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let normalized = test_suite.fs_compat.normalize_path("Test.txt").unwrap();
    
    #[cfg(target_os = "linux")]
    assert!(normalized.case_sensitive);
    
    #[cfg(any(target_os = "macos", windows))]
    assert!(!normalized.case_sensitive);
}

#[tokio::test]
async fn test_path_separator_detection() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let normalized = test_suite.fs_compat.normalize_path("test").unwrap();
    
    #[cfg(windows)]
    assert_eq!(normalized.separator, "\\");
    
    #[cfg(not(windows))]
    assert_eq!(normalized.separator, "/");
}

// === Path Normalization Tests ===

#[tokio::test]
async fn test_normalize_path_separators() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test mixed separators
    let mixed_paths = vec![
        "dir\\file.txt",
        "dir/file.txt",
        "dir\\subdir/file.txt",
        "dir/subdir\\file.txt",
    ];
    
    for path in mixed_paths {
        let normalized = test_suite.fs_compat.normalize_path(path).unwrap();
        let norm_str = normalized.normalized.to_string_lossy();
        
        #[cfg(windows)]
        {
            assert!(!norm_str.contains('/'), "Windows path should not contain forward slashes: {}", norm_str);
            assert!(norm_str.contains('\\') || !norm_str.contains('/'), "Windows path should use backslashes: {}", norm_str);
        }
        
        #[cfg(not(windows))]
        {
            assert!(!norm_str.contains('\\'), "Unix path should not contain backslashes: {}", norm_str);
            if norm_str.len() > 1 {
                assert!(norm_str.contains('/') || norm_str == ".", "Unix path should use forward slashes: {}", norm_str);
            }
        }
    }
}

#[tokio::test]
async fn test_unicode_filename_support() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let unicode_filenames = vec![
        "файл.txt",        // Russian (Cyrillic)
        "文件.txt",          // Chinese (Simplified)
        "ファイル.txt",     // Japanese (Hiragana)
        "파일.txt",         // Korean (Hangul)
        "αρχείο.txt",      // Greek
        "ملف.txt",          // Arabic
        "फ़ाइल.txt",       // Hindi (Devanagari)
        "dosya.txt",       // Turkish (Latin with special chars)
        "tệp.txt",         // Vietnamese
        "🚀rocket.txt",     // Emoji
        "test_ñoño.txt",   // Spanish with tildes
        "café_résumé.txt", // French with accents
    ];
    
    for filename in unicode_filenames {
        // Test normalization
        let result = test_suite.fs_compat.normalize_path(filename);
        assert!(result.is_ok(), "Failed to normalize Unicode filename: {}", filename);
        
        let normalized = result.unwrap();
        assert_eq!(normalized.original, PathBuf::from(filename));
        
        // Test file creation and access with Unicode names
        let file_path = test_suite.temp_path(filename);
        let content = format!("Content of {}", filename).into_bytes();
        
        // Create file
        let create_result = tokio::fs::write(&file_path, &content).await;
        if create_result.is_ok() {
            // Test reading back
            let read_content = tokio::fs::read(&file_path).await.unwrap();
            assert_eq!(read_content, content);
            
            // Test fs_compat entry retrieval
            let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
            assert_eq!(entry.entry_type, FsEntryType::File);
            assert_eq!(entry.size, content.len() as u64);
        }
    }
}

#[tokio::test]
async fn test_control_character_normalization() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let paths_with_controls = vec![
        "file\0with\0nulls.txt",
        "file\x01with\x02controls.txt",
        "file\twith\ttabs.txt",     // Tabs should be preserved
        "file\nwith\nlines.txt",   // Newlines should be preserved
        "file\x1fwith\x1ffs.txt",   // Unit separator (control)
    ];
    
    for path in paths_with_controls {
        let normalized = test_suite.fs_compat.normalize_path(path).unwrap();
        let norm_str = normalized.normalized.to_string_lossy();
        
        // Null bytes should be replaced
        assert!(!norm_str.contains('\0'), "Normalized path should not contain null bytes: {:?}", norm_str);
        
        // Other control characters (except tabs/newlines) should be replaced
        assert!(!norm_str.contains('\x01'), "Normalized path should not contain control character \x01: {:?}", norm_str);
        assert!(!norm_str.contains('\x02'), "Normalized path should not contain control character \x02: {:?}", norm_str);
        assert!(!norm_str.contains('\x1f'), "Normalized path should not contain control character \x1f: {:?}", norm_str);
    }
}

#[tokio::test]
async fn test_path_length_limits() {
    let mut fs_compat = FsCompat::new(true, 100, false); // Short limit for testing
    
    let short_path = "short.txt";
    let long_path = "a".repeat(150) + ".txt";
    
    // Short path should work
    assert!(fs_compat.normalize_path(short_path).is_ok());
    
    // Long path should fail without long path support
    assert!(fs_compat.normalize_path(&long_path).is_err());
    
    // Enable long path support
    fs_compat.long_path_support = true;
    assert!(fs_compat.normalize_path(&long_path).is_ok());
}

#[tokio::test]
async fn test_extremely_long_paths() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test paths that exceed typical limits
    let very_long_component = "a".repeat(255); // Max filename component on most systems
    let very_long_path = format!("{}/{}.txt", very_long_component, very_long_component);
    
    // This should work on systems with long path support
    let result = test_suite.fs_compat.normalize_path(&very_long_path);
    
    if cfg!(windows) && test_suite.fs_compat.long_path_support {
        assert!(result.is_ok(), "Long path should work on Windows with long path support");
    } else if !cfg!(windows) {
        assert!(result.is_ok(), "Long path should work on Unix systems");
    }
}

// === Case Sensitivity Tests ===

#[tokio::test]
async fn test_case_sensitivity_behavior() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let path1 = "Test.txt";
    let path2 = "test.txt";
    let path3 = "TEST.txt";
    
    let equal_12 = test_suite.fs_compat.paths_equal(path1, path2).unwrap();
    let equal_13 = test_suite.fs_compat.paths_equal(path1, path3).unwrap();
    let equal_23 = test_suite.fs_compat.paths_equal(path2, path3).unwrap();
    
    #[cfg(any(windows, target_os = "macos"))]
    {
        // Case-insensitive platforms
        assert!(equal_12, "Paths should be equal on case-insensitive platforms");
        assert!(equal_13, "Paths should be equal on case-insensitive platforms");
        assert!(equal_23, "Paths should be equal on case-insensitive platforms");
    }
    
    #[cfg(target_os = "linux")]
    {
        // Case-sensitive platform
        assert!(!equal_12, "Paths should not be equal on case-sensitive platforms");
        assert!(!equal_13, "Paths should not be equal on case-sensitive platforms");
        assert!(!equal_23, "Paths should not be equal on case-sensitive platforms");
    }
}

#[tokio::test]
async fn test_case_sensitivity_with_unicode() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let unicode_pairs = vec![
        ("Файл.txt", "файл.txt"),   // Russian uppercase/lowercase
        ("文件.TXT", "文件.txt"),     // Mixed case with Chinese
        ("CAFÉ.txt", "café.txt"),   // French with accents
    ];
    
    for (path1, path2) in unicode_pairs {
        let equal = test_suite.fs_compat.paths_equal(path1, path2).unwrap();
        
        #[cfg(any(windows, target_os = "macos"))]
        assert!(equal, "Unicode paths should be equal on case-insensitive platforms: {} vs {}", path1, path2);
        
        #[cfg(target_os = "linux")]
        assert!(!equal, "Unicode paths should not be equal on case-sensitive platforms: {} vs {}", path1, path2);
    }
}

// === File System Entry Tests ===

#[tokio::test]
async fn test_get_fs_entry_file() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let content = b"Test file content with some data";
    let file_path = test_suite.create_test_file("test.txt", content).await;
    
    let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
    
    assert_eq!(entry.entry_type, FsEntryType::File);
    assert_eq!(entry.size, content.len() as u64);
    assert!(entry.modified.is_some());
    assert!(entry.path.normalized.ends_with("test.txt"));
    
    // Test permissions
    #[cfg(unix)]
    {
        assert!(entry.permissions > 0, "Unix permissions should be set");
    }
    
    #[cfg(windows)]
    {
        assert!(entry.permissions == 0o666 || entry.permissions == 0o444, "Windows permissions should be readable: {:#o}", entry.permissions);
    }
}

#[tokio::test]
async fn test_get_fs_entry_directory() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let dir_path = test_suite.create_test_dir("testdir").await;
    
    let entry = test_suite.fs_compat.get_fs_entry(&dir_path).await.unwrap();
    
    assert_eq!(entry.entry_type, FsEntryType::Directory);
    assert!(entry.path.normalized.ends_with("testdir"));
    assert!(entry.modified.is_some());
}

#[tokio::test]
async fn test_hidden_file_detection() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test Unix-style hidden files (starting with dot)
    let hidden_file = test_suite.create_test_file(".hidden", b"hidden content").await;
    let visible_file = test_suite.create_test_file("visible.txt", b"visible content").await;
    
    let hidden_entry = test_suite.fs_compat.get_fs_entry(&hidden_file).await.unwrap();
    let visible_entry = test_suite.fs_compat.get_fs_entry(&visible_file).await.unwrap();
    
    #[cfg(unix)]
    {
        assert!(hidden_entry.hidden, "Dot files should be hidden on Unix");
        assert!(!visible_entry.hidden, "Regular files should not be hidden on Unix");
    }
    
    #[cfg(windows)]
    {
        // On Windows, we need to set the hidden attribute manually for testing
        // For now, just test that the function doesn't crash
        let _ = hidden_entry.hidden;
        let _ = visible_entry.hidden;
    }
}

// === Symlink and Junction Tests ===

#[cfg(unix)]
#[tokio::test]
async fn test_symlink_detection() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let target_file = test_suite.create_test_file("target.txt", b"target content").await;
    let link_path = test_suite.temp_path("link.txt");
    
    // Create symlink
    std::os::unix::fs::symlink(&target_file, &link_path).unwrap();
    
    let entry = test_suite.fs_compat.get_fs_entry(&link_path).await.unwrap();
    
    match entry.entry_type {
        FsEntryType::Symlink { target } => {
            assert_eq!(target, target_file, "Symlink target should match original");
        },
        _ => panic!("Expected symlink entry type, got: {:?}", entry.entry_type),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn test_symlink_to_directory() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let target_dir = test_suite.create_test_dir("target_dir").await;
    let link_path = test_suite.temp_path("dir_link");
    
    // Create directory symlink
    std::os::unix::fs::symlink(&target_dir, &link_path).unwrap();
    
    let entry = test_suite.fs_compat.get_fs_entry(&link_path).await.unwrap();
    
    match entry.entry_type {
        FsEntryType::Symlink { target } => {
            assert_eq!(target, target_dir, "Directory symlink target should match");
        },
        _ => panic!("Expected symlink entry type for directory link"),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn test_broken_symlink() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let nonexistent_target = test_suite.temp_path("nonexistent.txt");
    let link_path = test_suite.temp_path("broken_link.txt");
    
    // Create symlink to nonexistent target
    std::os::unix::fs::symlink(&nonexistent_target, &link_path).unwrap();
    
    let entry = test_suite.fs_compat.get_fs_entry(&link_path).await.unwrap();
    
    match entry.entry_type {
        FsEntryType::Symlink { target } => {
            assert_eq!(target, nonexistent_target, "Broken symlink should still report target");
        },
        _ => panic!("Expected symlink entry type for broken link"),
    }
}

#[cfg(windows)]
#[tokio::test]
async fn test_windows_junction_detection() {
    // Note: Creating junction points requires admin privileges on Windows
    // This test will be skipped in most CI environments
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test the junction detection logic without creating actual junctions
    // since that requires admin privileges
    let dummy_path = test_suite.temp_path("dummy");
    let result = test_suite.fs_compat.is_junction_point(&dummy_path);
    assert!(result.is_ok(), "Junction point detection should not fail");
}

// === Permission Tests ===

#[tokio::test]
async fn test_readonly_file_detection() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let file_path = test_suite.create_test_file("readonly.txt", b"readonly content").await;
    
    // Make file readonly
    let mut perms = tokio::fs::metadata(&file_path).await.unwrap().permissions();
    perms.set_readonly(true);
    tokio::fs::set_permissions(&file_path, perms).await.unwrap();
    
    let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
    assert!(entry.readonly, "File should be detected as readonly");
    
    #[cfg(windows)]
    {
        assert_eq!(entry.permissions, 0o444, "Windows readonly file should have 444 permissions");
    }
}

#[cfg(unix)]
#[tokio::test]
async fn test_unix_permission_extraction() {
    use std::os::unix::fs::PermissionsExt;
    
    let test_suite = CrossPlatformFsTestSuite::new();
    let file_path = test_suite.create_test_file("perms.txt", b"permission test").await;
    
    // Set specific permissions
    let mut perms = tokio::fs::metadata(&file_path).await.unwrap().permissions();
    perms.set_mode(0o755);
    tokio::fs::set_permissions(&file_path, perms).await.unwrap();
    
    let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
    assert_eq!(entry.permissions, 0o755, "Unix permissions should be extracted correctly");
}

// === Path Safety Tests ===

#[tokio::test]
async fn test_safe_join_security() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let base = "/safe/base/path";
    
    // Safe joins should work
    let safe_paths = vec![
        "file.txt",
        "subdir/file.txt",
        "deep/nested/structure/file.txt",
        "unicode_файл.txt",
    ];
    
    for safe_path in safe_paths {
        let result = test_suite.fs_compat.safe_join(base, safe_path);
        assert!(result.is_ok(), "Safe path should join successfully: {}", safe_path);
    }
    
    // Unsafe joins should fail
    let unsafe_paths = vec![
        "../escape.txt",
        "../../etc/passwd",
        "subdir/../../../escape.txt",
        "/absolute/path.txt",
        "C:\\Windows\\System32", // Windows absolute path
    ];
    
    for unsafe_path in unsafe_paths {
        let result = test_suite.fs_compat.safe_join(base, unsafe_path);
        assert!(result.is_err(), "Unsafe path should be rejected: {}", unsafe_path);
    }
}

#[tokio::test]
async fn test_directory_traversal_prevention() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let base = test_suite.temp_path("base");
    tokio::fs::create_dir(&base).await.unwrap();
    
    // Test various directory traversal attempts
    let traversal_attempts = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "subdir/../../secrets.txt",
        "normal/../../../bad.txt",
        "./../outside.txt",
    ];
    
    for attempt in traversal_attempts {
        let result = test_suite.fs_compat.safe_join(&base, attempt);
        assert!(result.is_err(), "Directory traversal should be prevented: {}", attempt);
        
        if let Err(e) = result {
            match e {
                DaemonError::FileIo { message, .. } => {
                    assert!(message.contains("directory traversal") || message.contains("absolute"), 
                           "Error should mention security issue: {}", message);
                },
                _ => panic!("Expected FileIo error for security violation"),
            }
        }
    }
}

// === Canonicalization Tests ===

#[tokio::test]
async fn test_canonicalize_path() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let file_path = test_suite.create_test_file("canon.txt", b"canonicalize test").await;
    
    let canonical = test_suite.fs_compat.canonicalize(&file_path).await.unwrap();
    
    assert!(canonical.is_absolute(), "Canonical path should be absolute");
    assert!(canonical.ends_with("canon.txt"), "Canonical path should end with filename");
    
    // Test that the canonical path exists
    assert!(test_suite.fs_compat.exists(&canonical).await, "Canonical path should exist");
}

#[tokio::test]
async fn test_canonicalize_relative_path() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Change to temp directory
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(test_suite.temp_dir.path()).unwrap();
    
    // Create file in current directory
    let rel_path = "relative.txt";
    tokio::fs::write(rel_path, b"relative path test").await.unwrap();
    
    let canonical = test_suite.fs_compat.canonicalize(rel_path).await.unwrap();
    
    assert!(canonical.is_absolute(), "Canonical path should be absolute");
    assert!(canonical.ends_with("relative.txt"), "Canonical path should end with filename");
    
    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

// === Existence Tests ===

#[tokio::test]
async fn test_file_existence_check() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let existing_file = test_suite.create_test_file("exists.txt", b"I exist").await;
    let nonexistent_file = test_suite.temp_path("does_not_exist.txt");
    
    assert!(test_suite.fs_compat.exists(&existing_file).await, "Existing file should be detected");
    assert!(!test_suite.fs_compat.exists(&nonexistent_file).await, "Nonexistent file should not be detected");
}

#[tokio::test]
async fn test_directory_existence_check() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let existing_dir = test_suite.create_test_dir("exists_dir").await;
    let nonexistent_dir = test_suite.temp_path("does_not_exist_dir");
    
    assert!(test_suite.fs_compat.exists(&existing_dir).await, "Existing directory should be detected");
    assert!(!test_suite.fs_compat.exists(&nonexistent_dir).await, "Nonexistent directory should not be detected");
}

// === Special Characters Tests ===

#[tokio::test]
async fn test_special_characters_in_filenames() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let special_filenames = vec![
        "file with spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "file@with#symbols$.txt",
        "file(with)parentheses.txt",
        "file[with]brackets.txt",
        "file{with}braces.txt",
        "file+with=equals&and+plus.txt",
        "file%20with%20encoded.txt",
    ];
    
    for filename in special_filenames {
        let result = test_suite.fs_compat.normalize_path(filename);
        assert!(result.is_ok(), "Should handle special characters in filename: {}", filename);
        
        // Try to create and access the file
        let file_path = test_suite.temp_path(filename);
        let create_result = tokio::fs::write(&file_path, b"test content").await;
        
        if create_result.is_ok() {
            let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
            assert_eq!(entry.entry_type, FsEntryType::File);
            assert_eq!(entry.size, 12); // "test content" length
        }
    }
}

// === Performance and Stress Tests ===

#[tokio::test]
async fn test_large_directory_structure() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let base_dir = test_suite.create_test_dir("large_structure").await;
    
    // Create nested directory structure
    let mut current_dir = base_dir.clone();
    for i in 0..10 {
        current_dir = current_dir.join(format!("level_{}", i));
        tokio::fs::create_dir(&current_dir).await.unwrap();
        
        // Create some files in each level
        for j in 0..5 {
            let file_path = current_dir.join(format!("file_{}.txt", j));
            tokio::fs::write(&file_path, format!("Content of file {} at level {}", j, i)).await.unwrap();
            
            // Test that fs_compat can handle the file
            let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
            assert_eq!(entry.entry_type, FsEntryType::File);
        }
    }
    
    // Test the deepest directory
    let entry = test_suite.fs_compat.get_fs_entry(&current_dir).await.unwrap();
    assert_eq!(entry.entry_type, FsEntryType::Directory);
}

#[tokio::test]
async fn test_concurrent_fs_operations() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    let mut handles = vec![];
    
    // Spawn multiple concurrent operations
    for i in 0..10 {
        let fs_compat = test_suite.fs_compat.clone();
        let temp_path = test_suite.temp_path(&format!("concurrent_{}.txt", i));
        
        let handle = tokio::spawn(async move {
            let content = format!("Concurrent content {}", i);
            tokio::fs::write(&temp_path, &content).await.unwrap();
            
            let entry = fs_compat.get_fs_entry(&temp_path).await.unwrap();
            assert_eq!(entry.entry_type, FsEntryType::File);
            assert_eq!(entry.size, content.len() as u64);
            
            let normalized = fs_compat.normalize_path(&temp_path).unwrap();
            assert!(!normalized.normalized.to_string_lossy().is_empty());
            
            i
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    let results = futures_util::future::join_all(handles).await;
    
    for (i, result) in results.into_iter().enumerate() {
        assert_eq!(result.unwrap(), i, "Concurrent operation {} should complete successfully", i);
    }
}

// === Error Handling Tests ===

#[tokio::test]
async fn test_error_handling_nonexistent_path() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let nonexistent = test_suite.temp_path("does_not_exist.txt");
    
    let result = test_suite.fs_compat.get_fs_entry(&nonexistent).await;
    assert!(result.is_err(), "Should error on nonexistent path");
    
    match result {
        Err(DaemonError::FileIo { message, path }) => {
            assert!(message.contains("Cannot access file metadata"), "Error message should be descriptive: {}", message);
            assert!(path.contains("does_not_exist.txt"), "Error should include path: {}", path);
        },
        _ => panic!("Expected FileIo error"),
    }
}

#[tokio::test]
async fn test_error_handling_invalid_path() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test with invalid Unicode sequences (if possible on platform)
    let invalid_paths = vec![
        "\0invalid\0path",
        "path\x01with\x02controls",
    ];
    
    for invalid_path in invalid_paths {
        let result = test_suite.fs_compat.normalize_path(invalid_path);
        // Should either normalize safely or return appropriate error
        match result {
            Ok(normalized) => {
                let norm_str = normalized.normalized.to_string_lossy();
                assert!(!norm_str.contains('\0'), "Normalized path should not contain null bytes");
            },
            Err(_) => {
                // Error is also acceptable for truly invalid paths
            },
        }
    }
}

// === Serialization Tests ===

#[tokio::test]
async fn test_fs_entry_serialization() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let file_path = test_suite.create_test_file("serialize.txt", b"serialization test").await;
    
    let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
    
    // Test JSON serialization
    let json = serde_json::to_string(&entry).unwrap();
    assert!(!json.is_empty(), "JSON serialization should produce non-empty result");
    
    let deserialized: FsEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(entry.entry_type, deserialized.entry_type);
    assert_eq!(entry.size, deserialized.size);
    assert_eq!(entry.readonly, deserialized.readonly);
    assert_eq!(entry.hidden, deserialized.hidden);
}

#[tokio::test]
async fn test_normalized_path_serialization() {
    let test_suite = CrossPlatformFsTestSuite::new();
    let path = "test/serialization/path.txt";
    let normalized = test_suite.fs_compat.normalize_path(path).unwrap();
    
    // Test JSON serialization
    let json = serde_json::to_string(&normalized).unwrap();
    let deserialized: NormalizedPath = serde_json::from_str(&json).unwrap();
    
    assert_eq!(normalized.original, deserialized.original);
    assert_eq!(normalized.normalized, deserialized.normalized);
    assert_eq!(normalized.case_sensitive, deserialized.case_sensitive);
    assert_eq!(normalized.separator, deserialized.separator);
}

// === Integration Tests ===

#[tokio::test]
async fn test_complete_file_lifecycle() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Create a complex file structure
    let base_dir = test_suite.create_test_dir("lifecycle").await;
    let sub_dir = base_dir.join("subdir");
    tokio::fs::create_dir(&sub_dir).await.unwrap();
    
    let file_path = sub_dir.join("test_file.txt");
    let content = b"Lifecycle test content";
    tokio::fs::write(&file_path, content).await.unwrap();
    
    // Test normalization
    let normalized = test_suite.fs_compat.normalize_path(&file_path).unwrap();
    assert!(!normalized.normalized.to_string_lossy().is_empty());
    
    // Test entry retrieval
    let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
    assert_eq!(entry.entry_type, FsEntryType::File);
    assert_eq!(entry.size, content.len() as u64);
    
    // Test existence
    assert!(test_suite.fs_compat.exists(&file_path).await);
    
    // Test canonicalization
    let canonical = test_suite.fs_compat.canonicalize(&file_path).await.unwrap();
    assert!(canonical.is_absolute());
    
    // Test safe join from base
    let relative_path = "subdir/test_file.txt";
    let joined = test_suite.fs_compat.safe_join(&base_dir, relative_path).unwrap();
    let joined_normalized = test_suite.fs_compat.normalize_path(&joined).unwrap();
    
    // Verify paths refer to the same file
    assert!(test_suite.fs_compat.paths_equal(&file_path, &joined).unwrap());
    
    // Test directory entry
    let dir_entry = test_suite.fs_compat.get_fs_entry(&sub_dir).await.unwrap();
    assert_eq!(dir_entry.entry_type, FsEntryType::Directory);
}

// === Platform-Specific Feature Tests ===

#[cfg(windows)]
mod windows_specific {
    use super::*;
    
    #[tokio::test]
    async fn test_windows_drive_letters() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        let paths_with_drives = vec![
            "C:\\Windows\\System32",
            "D:\\Data\\file.txt",
            "Z:\\Network\\share.txt",
        ];
        
        for path in paths_with_drives {
            let result = test_suite.fs_compat.normalize_path(path);
            assert!(result.is_ok(), "Windows drive path should normalize: {}", path);
            
            let normalized = result.unwrap();
            assert!(!normalized.case_sensitive, "Windows paths should be case-insensitive");
            assert_eq!(normalized.separator, "\\", "Windows should use backslash separator");
        }
    }
    
    #[tokio::test]
    async fn test_windows_unc_paths() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        let unc_paths = vec![
            "\\\\server\\share\\file.txt",
            "\\\\192.168.1.100\\public\\document.pdf",
        ];
        
        for path in unc_paths {
            let result = test_suite.fs_compat.normalize_path(path);
            assert!(result.is_ok(), "UNC path should normalize: {}", path);
        }
    }
    
    #[tokio::test]
    async fn test_windows_long_path_support() {
        let mut fs_compat = FsCompat::new(true, 260, true);
        
        // Test paths that exceed the traditional 260 character limit
        let long_component = "a".repeat(100);
        let long_path = format!("C:\\{0}\\{0}\\{0}\\{0}.txt", long_component);
        
        let result = fs_compat.normalize_path(&long_path);
        assert!(result.is_ok(), "Long path should be supported with long_path_support enabled");
        
        // Disable long path support
        fs_compat.long_path_support = false;
        let result = fs_compat.normalize_path(&long_path);
        assert!(result.is_err(), "Long path should fail without long_path_support");
    }
}

#[cfg(unix)]
mod unix_specific {
    use super::*;
    
    #[tokio::test]
    async fn test_unix_absolute_paths() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        let unix_paths = vec![
            "/usr/bin/bash",
            "/home/user/document.txt",
            "/var/log/system.log",
            "/tmp/temporary_file.tmp",
        ];
        
        for path in unix_paths {
            let result = test_suite.fs_compat.normalize_path(path);
            assert!(result.is_ok(), "Unix absolute path should normalize: {}", path);
            
            let normalized = result.unwrap();
            assert_eq!(normalized.separator, "/", "Unix should use forward slash separator");
        }
    }
    
    #[tokio::test]
    async fn test_unix_device_files() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        // Common device files (may not exist in test environment)
        let device_paths = vec![
            "/dev/null",
            "/dev/zero",
            "/dev/random",
        ];
        
        for path in device_paths {
            if test_suite.fs_compat.exists(path).await {
                let entry = test_suite.fs_compat.get_fs_entry(path).await.unwrap();
                // Device files are typically "Other" entry types
                match entry.entry_type {
                    FsEntryType::File | FsEntryType::Other(_) => {
                        // Both are acceptable for device files
                    },
                    _ => panic!("Device file should be File or Other type: {:?}", entry.entry_type),
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_specific {
    use super::*;
    
    #[tokio::test]
    async fn test_macos_case_insensitive_behavior() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        // macOS is typically case-insensitive but case-preserving
        let original_case = "TestFile.txt";
        let different_case = "testfile.txt";
        
        let normalized1 = test_suite.fs_compat.normalize_path(original_case).unwrap();
        let normalized2 = test_suite.fs_compat.normalize_path(different_case).unwrap();
        
        assert!(!normalized1.case_sensitive, "macOS should be case-insensitive");
        assert!(!normalized2.case_sensitive, "macOS should be case-insensitive");
        
        let equal = test_suite.fs_compat.paths_equal(original_case, different_case).unwrap();
        assert!(equal, "Paths should be equal on case-insensitive macOS");
    }
    
    #[tokio::test]
    async fn test_macos_special_directories() {
        let test_suite = CrossPlatformFsTestSuite::new();
        
        let macos_paths = vec![
            "/Applications",
            "/System/Library",
            "/Users",
            "/Library",
        ];
        
        for path in macos_paths {
            if test_suite.fs_compat.exists(path).await {
                let entry = test_suite.fs_compat.get_fs_entry(path).await.unwrap();
                assert_eq!(entry.entry_type, FsEntryType::Directory, "macOS system path should be directory: {}", path);
            }
        }
    }
}

// === Comprehensive Test Summary ===

#[tokio::test]
async fn test_comprehensive_compatibility() {
    let test_suite = CrossPlatformFsTestSuite::new();
    
    // Test all major features together
    let test_scenarios = vec![
        ("basic_file.txt", b"Basic file content" as &[u8]),
        ("unicode_файл.txt", b"Unicode filename content"),
        ("special chars!@#$.txt", b"Special characters in name"),
        (".hidden_file", b"Hidden file content"),
        ("UPPERCASE.TXT", b"Uppercase filename"),
        ("mixed_Case.Txt", b"Mixed case filename"),
    ];
    
    for (filename, content) in test_scenarios {
        // Create file
        let file_path = test_suite.temp_path(filename);
        if tokio::fs::write(&file_path, content).await.is_ok() {
            // Test all fs_compat operations
            let normalized = test_suite.fs_compat.normalize_path(&file_path).unwrap();
            let entry = test_suite.fs_compat.get_fs_entry(&file_path).await.unwrap();
            let exists = test_suite.fs_compat.exists(&file_path).await;
            let canonical = test_suite.fs_compat.canonicalize(&file_path).await.unwrap();
            
            // Verify results
            assert_eq!(entry.entry_type, FsEntryType::File, "File should be detected as file: {}", filename);
            assert_eq!(entry.size, content.len() as u64, "File size should match content length: {}", filename);
            assert!(exists, "File should exist: {}", filename);
            assert!(canonical.is_absolute(), "Canonical path should be absolute: {}", filename);
            
            // Test serialization
            let entry_json = serde_json::to_string(&entry).unwrap();
            let normalized_json = serde_json::to_string(&normalized).unwrap();
            
            assert!(!entry_json.is_empty(), "Entry serialization should work: {}", filename);
            assert!(!normalized_json.is_empty(), "Normalized path serialization should work: {}", filename);
        }
    }
}
