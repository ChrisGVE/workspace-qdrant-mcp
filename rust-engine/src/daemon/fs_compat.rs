//! Cross-platform file system compatibility layer
//!
//! This module provides a unified interface for file system operations that handles
//! platform-specific differences in file paths, case sensitivity, symlinks, and permissions.

use crate::error::{DaemonError, DaemonResult};
use std::path::{Path, PathBuf};
use std::fs::Metadata;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn, error};

/// Normalized path information across platforms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedPath {
    /// Original path as provided
    pub original: PathBuf,
    /// Normalized path for consistent operations
    pub normalized: PathBuf,
    /// Indicates if the path is case-sensitive on this platform
    pub case_sensitive: bool,
    /// Platform-specific path separator
    pub separator: String,
}

/// File system entry type with platform-specific information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FsEntryType {
    /// Regular file
    File,
    /// Directory
    Directory,
    /// Symbolic link (target path included)
    Symlink { target: PathBuf },
    /// Hard link (Windows junction points treated as hard links)
    #[cfg(windows)]
    Junction { target: PathBuf },
    /// Other platform-specific types
    Other(String),
}

/// Comprehensive file system entry information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsEntry {
    /// Normalized path information
    pub path: NormalizedPath,
    /// Entry type with platform-specific details
    pub entry_type: FsEntryType,
    /// File size in bytes
    pub size: u64,
    /// Unix-style permissions (converted on Windows)
    pub permissions: u32,
    /// Read-only status
    pub readonly: bool,
    /// Hidden status (platform-specific)
    pub hidden: bool,
    /// Creation time (if available)
    pub created: Option<SystemTime>,
    /// Last modification time
    pub modified: Option<SystemTime>,
    /// Last access time (if available)
    pub accessed: Option<SystemTime>,
}

/// Cross-platform file system compatibility handler
#[derive(Debug, Clone)]
pub struct FsCompat {
    /// Enable Unicode filename normalization
    pub normalize_unicode: bool,
    /// Maximum path length to handle
    pub max_path_length: usize,
    /// Enable long path support on Windows
    pub long_path_support: bool,
}

impl Default for FsCompat {
    fn default() -> Self {
        Self {
            normalize_unicode: true,
            max_path_length: if cfg!(windows) { 260 } else { 4096 },
            long_path_support: cfg!(windows),
        }
    }
}

impl FsCompat {
    /// Create a new file system compatibility handler
    pub fn new(normalize_unicode: bool, max_path_length: usize, long_path_support: bool) -> Self {
        Self {
            normalize_unicode,
            max_path_length,
            long_path_support,
        }
    }

    /// Normalize a path for cross-platform compatibility
    pub fn normalize_path<P: AsRef<Path>>(&self, path: P) -> DaemonResult<NormalizedPath> {
        let path = path.as_ref();
        let original = path.to_path_buf();
        
        debug!("Normalizing path: {}", path.display());
        
        // Check path length limits
        let path_str = path.to_string_lossy();
        if path_str.len() > self.max_path_length && !self.long_path_support {
            return Err(DaemonError::FileIo {
                message: format!(
                    "Path exceeds maximum length of {}: {} characters",
                    self.max_path_length,
                    path_str.len()
                ),
                path: path_str.to_string(),
            });
        }
        
        // Normalize path separators
        let normalized_str = if cfg!(windows) {
            // Convert forward slashes to backslashes on Windows
            path_str.replace('/', "\\")
        } else {
            // Convert backslashes to forward slashes on Unix-like systems
            path_str.replace('\\', "/")
        };
        
        // Handle Unicode normalization if enabled
        let final_str = if self.normalize_unicode {
            self.normalize_unicode_path(&normalized_str)?
        } else {
            normalized_str
        };
        
        let normalized = PathBuf::from(final_str);
        
        // Platform-specific case sensitivity
        let case_sensitive = self.is_case_sensitive_platform();
        
        // Platform-specific separator
        let separator = if cfg!(windows) {
            "\\".to_string()
        } else {
            "/".to_string()
        };
        
        Ok(NormalizedPath {
            original,
            normalized,
            case_sensitive,
            separator,
        })
    }
    
    /// Normalize Unicode characters in file paths
    fn normalize_unicode_path(&self, path: &str) -> DaemonResult<String> {
        // Simple Unicode normalization - replace problematic characters
        let normalized = path
            .chars()
            .map(|c| match c {
                // Replace null bytes that can cause issues
                '\0' => '_',
                // Replace control characters
                c if c.is_control() && c != '\t' && c != '\n' && c != '\r' => '_',
                // Keep other characters as-is
                c => c,
            })
            .collect();
        
        Ok(normalized)
    }
    
    /// Check if the current platform has case-sensitive file paths
    fn is_case_sensitive_platform(&self) -> bool {
        #[cfg(target_os = "linux")]
        return true;
        
        #[cfg(target_os = "macos")]
        return false; // HFS+ and APFS are case-insensitive by default
        
        #[cfg(windows)]
        return false; // Windows is case-insensitive
        
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        return true; // Default to case-sensitive for other Unix-like systems
    }
    
    /// Get detailed file system entry information
    pub async fn get_fs_entry<P: AsRef<Path>>(&self, path: P) -> DaemonResult<FsEntry> {
        let path = path.as_ref();
        let normalized_path = self.normalize_path(path)?;
        
        debug!("Getting fs entry for: {}", path.display());
        
        // Get metadata
        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            error!("Failed to get metadata for {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot access file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;
        
        // Determine entry type
        let entry_type = self.determine_entry_type(path, &metadata).await?;
        
        // Extract common metadata
        let size = metadata.len();
        let permissions = self.extract_permissions(&metadata);
        let readonly = metadata.permissions().readonly();
        let hidden = self.is_hidden(path)?;
        
        // Extract timestamps
        let created = metadata.created().ok();
        let modified = metadata.modified().ok();
        let accessed = metadata.accessed().ok();
        
        Ok(FsEntry {
            path: normalized_path,
            entry_type,
            size,
            permissions,
            readonly,
            hidden,
            created,
            modified,
            accessed,
        })
    }
    
    /// Determine the entry type (file, directory, symlink, etc.)
    async fn determine_entry_type<P: AsRef<Path>>(&self, path: P, metadata: &Metadata) -> DaemonResult<FsEntryType> {
        let path = path.as_ref();
        
        if metadata.is_file() {
            Ok(FsEntryType::File)
        } else if metadata.is_dir() {
            Ok(FsEntryType::Directory)
        } else if metadata.file_type().is_symlink() {
            // Get symlink target
            let target = tokio::fs::read_link(path).await.map_err(|e| {
                warn!("Failed to read symlink target for {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot read symlink target: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;
            
            #[cfg(windows)]
            {
                // On Windows, check if this is a junction point
                if self.is_junction_point(path)? {
                    return Ok(FsEntryType::Junction { target });
                }
            }
            
            Ok(FsEntryType::Symlink { target })
        } else {
            // Other file types (device files, pipes, etc.)
            let type_name = format!("{:?}", metadata.file_type());
            Ok(FsEntryType::Other(type_name))
        }
    }
    
    /// Extract Unix-style permissions from metadata
    fn extract_permissions(&self, metadata: &Metadata) -> u32 {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode()
        }
        
        #[cfg(windows)]
        {
            // Convert Windows permissions to Unix-style
            let readonly = metadata.permissions().readonly();
            if readonly {
                0o444 // Read-only
            } else {
                0o666 // Read-write
            }
        }
    }
    
    /// Check if a file is hidden (platform-specific)
    fn is_hidden<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let path = path.as_ref();
        
        #[cfg(unix)]
        {
            // On Unix, files starting with '.' are hidden
            if let Some(filename) = path.file_name() {
                Ok(filename.to_string_lossy().starts_with('.'))
            } else {
                Ok(false)
            }
        }
        
        #[cfg(windows)]
        {
            // On Windows, check the hidden attribute
            const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;

            match std::fs::metadata(path) {
                Ok(metadata) => {
                    #[cfg(windows)]
                    {
                        use std::os::windows::fs::MetadataExt;
                        let attributes = metadata.file_attributes();
                        Ok((attributes & FILE_ATTRIBUTE_HIDDEN) != 0)
                    }
                    #[cfg(not(windows))]
                    Ok(false)
                },
                Err(_) => Ok(false),
            }
        }
    }
    
    /// Check if a path is a Windows junction point
    #[cfg(windows)]
    fn is_junction_point<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let path = path.as_ref();
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;

        match std::fs::metadata(path) {
            Ok(metadata) => {
                use std::os::windows::fs::MetadataExt;
                let attributes = metadata.file_attributes();
                Ok((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
            },
            Err(_) => Ok(false),
        }
    }
    
    /// Check if two paths refer to the same file (handles case sensitivity)
    pub fn paths_equal<P1: AsRef<Path>, P2: AsRef<Path>>(&self, path1: P1, path2: P2) -> DaemonResult<bool> {
        let norm1 = self.normalize_path(path1)?;
        let norm2 = self.normalize_path(path2)?;
        
        if norm1.case_sensitive {
            Ok(norm1.normalized == norm2.normalized)
        } else {
            // Case-insensitive comparison
            let str1 = norm1.normalized.to_string_lossy().to_lowercase();
            let str2 = norm2.normalized.to_string_lossy().to_lowercase();
            Ok(str1 == str2)
        }
    }
    
    /// Get the canonical (absolute) path with all symlinks resolved
    pub async fn canonicalize<P: AsRef<Path>>(&self, path: P) -> DaemonResult<PathBuf> {
        let path = path.as_ref();
        
        tokio::fs::canonicalize(path).await.map_err(|e| {
            error!("Failed to canonicalize path {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot canonicalize path: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })
    }
    
    /// Check if a path exists (case-sensitive check on case-sensitive platforms)
    pub async fn exists<P: AsRef<Path>>(&self, path: P) -> bool {
        let path = path.as_ref();
        
        tokio::fs::try_exists(path).await.unwrap_or(false)
    }
    
    /// Safe path join that prevents directory traversal attacks
    pub fn safe_join<P1: AsRef<Path>, P2: AsRef<Path>>(&self, base: P1, path: P2) -> DaemonResult<PathBuf> {
        let base = base.as_ref();
        let path = path.as_ref();
        
        // Check for directory traversal attempts
        let path_str = path.to_string_lossy();
        if path_str.contains("..") || path.is_absolute() {
            return Err(DaemonError::FileIo {
                message: "Path contains directory traversal or is absolute".to_string(),
                path: path_str.to_string(),
            });
        }
        
        let joined = base.join(path);
        self.normalize_path(joined).map(|np| np.normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, TempDir};
    use std::fs;
    
    fn create_fs_compat() -> FsCompat {
        FsCompat::default()
    }
    
    #[tokio::test]
    async fn test_normalize_path_basic() {
        let fs_compat = create_fs_compat();
        
        let path = "/test/path/file.txt";
        let normalized = fs_compat.normalize_path(path).unwrap();
        
        assert_eq!(normalized.original, PathBuf::from(path));
        assert!(!normalized.normalized.to_string_lossy().is_empty());
        
        if cfg!(windows) {
            assert!(!normalized.case_sensitive);
            assert_eq!(normalized.separator, "\\");
        } else {
            assert_eq!(normalized.separator, "/");
        }
    }
    
    #[tokio::test]
    async fn test_normalize_path_separators() {
        let fs_compat = create_fs_compat();
        
        // Test path with mixed separators
        let mixed_path = "test\\mixed/separators\\file.txt";
        let normalized = fs_compat.normalize_path(mixed_path).unwrap();
        
        let norm_str = normalized.normalized.to_string_lossy();
        
        if cfg!(windows) {
            assert!(!norm_str.contains('/'));
            assert!(norm_str.contains('\\'));
        } else {
            assert!(!norm_str.contains('\\'));
            assert!(norm_str.contains('/'));
        }
    }
    
    #[tokio::test]
    async fn test_normalize_unicode_paths() {
        let fs_compat = create_fs_compat();
        
        let unicode_paths = vec![
            "test/файл.txt",  // Russian
            "test/文件.txt",   // Chinese
            "test/ファイル.txt", // Japanese
            "test/🚀.txt",    // Emoji
        ];
        
        for path in unicode_paths {
            let result = fs_compat.normalize_path(path);
            assert!(result.is_ok(), "Failed to normalize Unicode path: {}", path);
            
            let normalized = result.unwrap();
            assert_eq!(normalized.original, PathBuf::from(path));
        }
    }
    
    #[tokio::test]
    async fn test_normalize_path_with_control_chars() {
        let fs_compat = create_fs_compat();
        
        let path_with_controls = "test\0file\x01.txt";
        let normalized = fs_compat.normalize_path(path_with_controls).unwrap();
        
        let norm_str = normalized.normalized.to_string_lossy();
        assert!(!norm_str.contains('\0'));
        assert!(!norm_str.contains('\x01'));
    }
    
    #[tokio::test]
    async fn test_path_length_limits() {
        let mut fs_compat = create_fs_compat();
        fs_compat.max_path_length = 50;
        fs_compat.long_path_support = false;
        
        let short_path = "short.txt";
        let long_path = "a".repeat(100) + ".txt";
        
        assert!(fs_compat.normalize_path(short_path).is_ok());
        assert!(fs_compat.normalize_path(long_path).is_err());
    }
    
    #[tokio::test]
    async fn test_get_fs_entry_file() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        fs::write(&file_path, b"test content").unwrap();
        
        let entry = fs_compat.get_fs_entry(&file_path).await.unwrap();
        
        assert_eq!(entry.entry_type, FsEntryType::File);
        assert_eq!(entry.size, 12); // "test content" length
        assert!(entry.modified.is_some());
    }
    
    #[tokio::test]
    async fn test_get_fs_entry_directory() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().join("subdir");
        
        fs::create_dir(&dir_path).unwrap();
        
        let entry = fs_compat.get_fs_entry(&dir_path).await.unwrap();
        
        assert_eq!(entry.entry_type, FsEntryType::Directory);
    }
    
    #[cfg(unix)]
    #[tokio::test]
    async fn test_get_fs_entry_symlink() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("target.txt");
        let link_path = temp_dir.path().join("link.txt");
        
        fs::write(&file_path, b"target content").unwrap();
        std::os::unix::fs::symlink(&file_path, &link_path).unwrap();
        
        let entry = fs_compat.get_fs_entry(&link_path).await.unwrap();
        
        match entry.entry_type {
            FsEntryType::Symlink { target } => {
                assert_eq!(target, file_path);
            },
            _ => panic!("Expected symlink entry type"),
        }
    }
    
    #[tokio::test]
    async fn test_paths_equal_case_sensitive() {
        let fs_compat = create_fs_compat();
        
        let path1 = "Test.txt";
        let path2 = "test.txt";
        
        let equal = fs_compat.paths_equal(path1, path2).unwrap();
        
        if cfg!(windows) || cfg!(target_os = "macos") {
            assert!(equal); // Case-insensitive platforms
        } else {
            assert!(!equal); // Case-sensitive platforms
        }
    }
    
    #[tokio::test]
    async fn test_safe_join() {
        let fs_compat = create_fs_compat();
        
        let base = "/safe/base";
        let safe_path = "subdir/file.txt";
        let unsafe_path = "../../../etc/passwd";
        let absolute_path = "/absolute/path";
        
        // Safe join should work
        assert!(fs_compat.safe_join(base, safe_path).is_ok());
        
        // Unsafe joins should fail
        assert!(fs_compat.safe_join(base, unsafe_path).is_err());
        assert!(fs_compat.safe_join(base, absolute_path).is_err());
    }
    
    #[tokio::test]
    async fn test_is_hidden_unix_style() {
        let fs_compat = create_fs_compat();
        
        #[cfg(unix)]
        {
            assert!(fs_compat.is_hidden(".hidden").unwrap());
            assert!(fs_compat.is_hidden("path/.hidden").unwrap());
            assert!(!fs_compat.is_hidden("visible.txt").unwrap());
        }
    }
    
    #[tokio::test]
    async fn test_canonicalize() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        
        let canonical = fs_compat.canonicalize(temp_dir.path()).await.unwrap();
        assert!(canonical.is_absolute());
    }
    
    #[tokio::test]
    async fn test_exists() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("exists.txt");
        let missing_path = temp_dir.path().join("missing.txt");
        
        fs::write(&file_path, b"content").unwrap();
        
        assert!(fs_compat.exists(&file_path).await);
        assert!(!fs_compat.exists(&missing_path).await);
    }
    
    #[tokio::test]
    async fn test_fs_compat_clone_debug() {
        let fs_compat = create_fs_compat();
        let cloned = fs_compat.clone();
        
        assert_eq!(fs_compat.normalize_unicode, cloned.normalize_unicode);
        assert_eq!(fs_compat.max_path_length, cloned.max_path_length);
        assert_eq!(fs_compat.long_path_support, cloned.long_path_support);
        
        let debug_str = format!("{:?}", fs_compat);
        assert!(debug_str.contains("FsCompat"));
    }
    
    #[tokio::test]
    async fn test_normalized_path_serialization() {
        let fs_compat = create_fs_compat();
        let path = "test/file.txt";
        let normalized = fs_compat.normalize_path(path).unwrap();
        
        // Test JSON serialization
        let json = serde_json::to_string(&normalized).unwrap();
        let deserialized: NormalizedPath = serde_json::from_str(&json).unwrap();
        
        assert_eq!(normalized, deserialized);
    }
    
    #[tokio::test]
    async fn test_fs_entry_serialization() {
        let fs_compat = create_fs_compat();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("serialize.txt");
        
        fs::write(&file_path, b"content").unwrap();
        
        let entry = fs_compat.get_fs_entry(&file_path).await.unwrap();
        
        // Test JSON serialization
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: FsEntry = serde_json::from_str(&json).unwrap();
        
        assert_eq!(entry.entry_type, deserialized.entry_type);
        assert_eq!(entry.size, deserialized.size);
    }
    
    #[tokio::test]
    async fn test_platform_specific_behavior() {
        let fs_compat = create_fs_compat();
        
        // Test platform-specific defaults
        if cfg!(windows) {
            assert_eq!(fs_compat.max_path_length, 260);
            assert!(fs_compat.long_path_support);
        } else {
            assert_eq!(fs_compat.max_path_length, 4096);
            assert!(!fs_compat.long_path_support);
        }
        
        let normalized = fs_compat.normalize_path("test").unwrap();
        
        #[cfg(target_os = "linux")]
        assert!(normalized.case_sensitive);
        
        #[cfg(any(target_os = "macos", windows))]
        assert!(!normalized.case_sensitive);
    }
}