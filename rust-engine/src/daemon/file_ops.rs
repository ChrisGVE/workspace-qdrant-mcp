//! Async file operations for document processing
//!
//! This module provides comprehensive async file I/O operations with proper error handling,
//! stream processing, and integration with the file system monitoring capabilities.

use crate::error::{DaemonError, DaemonResult};
use blake3::Hasher;
use std::path::{Path, PathBuf};
use std::os::unix::fs::{MetadataExt, FileTypeExt};
use std::collections::HashMap;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use futures_util::stream::unfold;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

/// Maximum file size for async processing (100MB)
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Default buffer size for async operations
pub const DEFAULT_BUFFER_SIZE: usize = 8192;

/// Async file processor for document operations
#[derive(Debug, Clone)]
pub struct AsyncFileProcessor {
    max_file_size: u64,
    buffer_size: usize,
    enable_compression: bool,
}

impl Default for AsyncFileProcessor {
    fn default() -> Self {
        Self {
            max_file_size: MAX_FILE_SIZE,
            buffer_size: DEFAULT_BUFFER_SIZE,
            enable_compression: true,
        }
    }
}

impl AsyncFileProcessor {
    /// Create a new async file processor with custom configuration
    pub fn new(max_file_size: u64, buffer_size: usize, enable_compression: bool) -> Self {
        Self {
            max_file_size,
            buffer_size,
            enable_compression,
        }
    }

    /// Read file content asynchronously with size validation
    pub async fn read_file<P: AsRef<Path>>(&self, path: P) -> DaemonResult<Vec<u8>> {
        let path = path.as_ref();
        debug!("Reading file asynchronously: {}", path.display());

        // Validate file exists and get metadata
        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            error!("Failed to read metadata for {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot access file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        // Check file size limit
        if metadata.len() > self.max_file_size {
            return Err(DaemonError::FileTooLarge {
                path: path.to_string_lossy().to_string(),
                size: metadata.len(),
                max_size: self.max_file_size,
            });
        }

        // Open and read file
        let mut file = File::open(path).await.map_err(|e| {
            error!("Failed to open file {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot open file: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        file.read_to_end(&mut buffer).await.map_err(|e| {
            error!("Failed to read file {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot read file content: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        // Use compression setting for future enhancement
        if self.enable_compression {
            debug!("Compression enabled for file processing");
        }

        info!("Successfully read {} bytes from {}", buffer.len(), path.display());
        Ok(buffer)
    }

    /// Write file content asynchronously with atomic operations
    pub async fn write_file<P: AsRef<Path>>(&self, path: P, content: &[u8]) -> DaemonResult<()> {
        let path = path.as_ref();
        debug!("Writing {} bytes to file: {}", content.len(), path.display());

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create parent directories for {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot create parent directories: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;
        }

        // Use temporary file for atomic writes
        let temp_path = path.with_extension(format!("tmp_{}", uuid::Uuid::new_v4()));

        {
            let mut file = File::create(&temp_path).await.map_err(|e| {
                error!("Failed to create temporary file {}: {}", temp_path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot create temporary file: {}", e),
                    path: temp_path.to_string_lossy().to_string(),
                }
            })?;

            file.write_all(content).await.map_err(|e| {
                error!("Failed to write to temporary file {}: {}", temp_path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot write to temporary file: {}", e),
                    path: temp_path.to_string_lossy().to_string(),
                }
            })?;

            file.sync_all().await.map_err(|e| {
                error!("Failed to sync temporary file {}: {}", temp_path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot sync temporary file: {}", e),
                    path: temp_path.to_string_lossy().to_string(),
                }
            })?;
        }

        // Atomically move temporary file to final location
        tokio::fs::rename(&temp_path, path).await.map_err(|e| {
            error!("Failed to move temporary file {} to {}: {}", temp_path.display(), path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot move temporary file to final location: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        info!("Successfully wrote {} bytes to {}", content.len(), path.display());
        Ok(())
    }

    /// Append content to file asynchronously
    pub async fn append_file<P: AsRef<Path>>(&self, path: P, content: &[u8]) -> DaemonResult<()> {
        let path = path.as_ref();
        debug!("Appending {} bytes to file: {}", content.len(), path.display());

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create parent directories for {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot create parent directories: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|e| {
                error!("Failed to open file for append {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot open file for append: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;

        file.write_all(content).await.map_err(|e| {
            error!("Failed to append to file {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot append to file: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        file.sync_all().await.map_err(|e| {
            error!("Failed to sync file {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot sync file: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        info!("Successfully appended {} bytes to {}", content.len(), path.display());
        Ok(())
    }

    /// Calculate file hash asynchronously
    pub async fn calculate_hash<P: AsRef<Path>>(&self, path: P) -> DaemonResult<String> {
        let path = path.as_ref();
        debug!("Calculating hash for file: {}", path.display());

        let mut file = File::open(path).await.map_err(|e| {
            error!("Failed to open file for hashing {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot open file for hashing: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let mut hasher = Hasher::new();
        let mut buffer = vec![0u8; self.buffer_size];

        loop {
            let bytes_read = file.read(&mut buffer).await.map_err(|e| {
                error!("Failed to read file for hashing {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot read file for hashing: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        let hash_string = hash.to_hex().to_string();

        debug!("Calculated hash {} for file: {}", hash_string, path.display());
        Ok(hash_string)
    }

    /// Process file in chunks asynchronously
    pub async fn process_chunks<P, F, T>(&self, path: P, chunk_size: usize, processor: F) -> DaemonResult<Vec<T>>
    where
        P: AsRef<Path>,
        F: Fn(&[u8], usize) -> DaemonResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let path = path.as_ref();
        debug!("Processing file in chunks: {} (chunk_size: {})", path.display(), chunk_size);

        let file = File::open(path).await.map_err(|e| {
            error!("Failed to open file for chunk processing {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot open file for chunk processing: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let mut reader = BufReader::new(file);
        let mut results = Vec::new();
        let mut buffer = vec![0u8; chunk_size];
        let mut chunk_index = 0;

        loop {
            let bytes_read = reader.read(&mut buffer).await.map_err(|e| {
                error!("Failed to read chunk from file {}: {}", path.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot read chunk from file: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })?;

            if bytes_read == 0 {
                break;
            }

            let result = processor(&buffer[..bytes_read], chunk_index)?;
            results.push(result);
            chunk_index += 1;
        }

        info!("Processed {} chunks from file: {}", chunk_index, path.display());
        Ok(results)
    }

    /// Copy file asynchronously with progress tracking
    pub async fn copy_file<P1, P2>(&self, src: P1, dst: P2) -> DaemonResult<u64>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        let src = src.as_ref();
        let dst = dst.as_ref();
        debug!("Copying file from {} to {}", src.display(), dst.display());

        // Validate source file
        let metadata = tokio::fs::metadata(src).await.map_err(|e| {
            error!("Failed to read source metadata {}: {}", src.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot access source file metadata: {}", e),
                path: src.to_string_lossy().to_string(),
            }
        })?;

        if metadata.len() > self.max_file_size {
            return Err(DaemonError::FileTooLarge {
                path: src.to_string_lossy().to_string(),
                size: metadata.len(),
                max_size: self.max_file_size,
            });
        }

        // Create destination parent directories
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create destination parent directories for {}: {}", dst.display(), e);
                DaemonError::FileIo {
                    message: format!("Cannot create destination parent directories: {}", e),
                    path: dst.to_string_lossy().to_string(),
                }
            })?;
        }

        // Use buffered copy for large files
        let mut src_file = File::open(src).await.map_err(|e| {
            error!("Failed to open source file {}: {}", src.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot open source file: {}", e),
                path: src.to_string_lossy().to_string(),
            }
        })?;

        let dst_file = File::create(dst).await.map_err(|e| {
            error!("Failed to create destination file {}: {}", dst.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot create destination file: {}", e),
                path: dst.to_string_lossy().to_string(),
            }
        })?;

        let mut dst_writer = BufWriter::new(dst_file);
        let bytes_copied = tokio::io::copy(&mut src_file, &mut dst_writer).await.map_err(|e| {
            error!("Failed to copy file from {} to {}: {}", src.display(), dst.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot copy file content: {}", e),
                path: dst.to_string_lossy().to_string(),
            }
        })?;

        dst_writer.flush().await.map_err(|e| {
            error!("Failed to flush destination file {}: {}", dst.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot flush destination file: {}", e),
                path: dst.to_string_lossy().to_string(),
            }
        })?;

        info!("Successfully copied {} bytes from {} to {}", bytes_copied, src.display(), dst.display());
        Ok(bytes_copied)
    }

    /// Check if file exists and is readable
    pub async fn validate_file<P: AsRef<Path>>(&self, path: P) -> DaemonResult<FileInfo> {
        let path = path.as_ref();
        debug!("Validating file: {}", path.display());

        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            error!("Failed to read file metadata {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot access file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let file_info = FileInfo {
            path: path.to_path_buf(),
            size: metadata.len(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            is_readonly: metadata.permissions().readonly(),
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
        };

        debug!("File validation result for {}: {:?}", path.display(), file_info);
        Ok(file_info)
    }

    /// Create a backup of the file before modification
    pub async fn create_backup<P: AsRef<Path>>(&self, path: P) -> DaemonResult<PathBuf> {
        let path = path.as_ref();
        let backup_path = path.with_extension(format!("backup_{}", chrono::Utc::now().timestamp()));

        debug!("Creating backup of {} to {}", path.display(), backup_path.display());

        self.copy_file(path, &backup_path).await?;

        info!("Successfully created backup: {}", backup_path.display());
        Ok(backup_path)
    }

    /// Safe atomic file replacement
    pub async fn atomic_replace<P: AsRef<Path>>(&self, path: P, content: &[u8]) -> DaemonResult<()> {
        let path = path.as_ref();
        debug!("Performing atomic replacement of file: {}", path.display());

        // Create backup first if file exists
        let backup_path = if path.exists() {
            Some(self.create_backup(path).await?)
        } else {
            None
        };

        // Attempt to write new content
        match self.write_file(path, content).await {
            Ok(()) => {
                info!("Atomic replacement successful for: {}", path.display());
                // Remove backup on success
                if let Some(backup) = backup_path {
                    if let Err(e) = tokio::fs::remove_file(&backup).await {
                        warn!("Failed to remove backup file {}: {}", backup.display(), e);
                    }
                }
                Ok(())
            }
            Err(e) => {
                error!("Atomic replacement failed for {}: {}", path.display(), e);
                // Restore from backup on failure
                if let Some(backup) = backup_path {
                    if let Err(restore_err) = self.copy_file(&backup, path).await {
                        error!("Failed to restore from backup {}: {}", backup.display(), restore_err);
                    }
                    if let Err(cleanup_err) = tokio::fs::remove_file(&backup).await {
                        warn!("Failed to cleanup backup file {}: {}", backup.display(), cleanup_err);
                    }
                }
                Err(e)
            }
        }
    }
}

/// File information structure
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub is_readonly: bool,
    pub created: Option<std::time::SystemTime>,
    pub modified: Option<std::time::SystemTime>,
}

/// Special file types supported by the system
#[derive(Debug, Clone, PartialEq)]
pub enum SpecialFileType {
    NamedPipe,
    Socket,
    CharacterDevice,
    BlockDevice,
}

/// Symlink information structure
#[derive(Debug, Clone)]
pub struct SymlinkInfo {
    pub target: PathBuf,
    pub final_target: PathBuf,
    pub is_broken: bool,
    pub is_absolute: bool,
    pub depth: usize,
    pub is_cross_filesystem: bool,
}

/// Special file handler for symlinks, hard links, and special file types
#[derive(Debug, Clone)]
pub struct SpecialFileHandler {
    max_symlink_depth: usize,
    enable_cross_filesystem: bool,
}

impl Default for SpecialFileHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecialFileHandler {
    /// Create a new special file handler with default configuration
    pub fn new() -> Self {
        Self {
            max_symlink_depth: 32, // Standard Unix limit
            enable_cross_filesystem: true,
        }
    }

    /// Create a new special file handler with custom configuration
    pub fn with_config(max_symlink_depth: usize, enable_cross_filesystem: bool) -> Self {
        Self {
            max_symlink_depth,
            enable_cross_filesystem,
        }
    }

    /// Check if a path is a symlink
    pub async fn is_symlink<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let path = path.as_ref();

        let metadata = tokio::fs::symlink_metadata(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read symlink metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        Ok(metadata.file_type().is_symlink())
    }

    /// Resolve a symlink to its final target
    pub async fn resolve_symlink<P: AsRef<Path>>(&self, path: P) -> DaemonResult<PathBuf> {
        let path = path.as_ref();

        if !self.is_symlink(path).await? {
            return Err(DaemonError::FileIo {
                message: "Path is not a symlink".to_string(),
                path: path.to_string_lossy().to_string(),
            });
        }

        self.resolve_symlink_recursive(path, 0, &mut HashMap::new()).await
    }

    /// Recursive symlink resolution with cycle detection
    fn resolve_symlink_recursive<'a>(
        &'a self,
        path: &'a Path,
        depth: usize,
        visited: &'a mut HashMap<PathBuf, usize>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = DaemonResult<PathBuf>> + Send + 'a>> {
        Box::pin(async move {
        // Check depth limit
        if depth > self.max_symlink_depth {
            return Err(DaemonError::SymlinkDepthExceeded {
                link_path: path.to_string_lossy().to_string(),
                depth,
                max_depth: self.max_symlink_depth,
            });
        }

        // Check cross-filesystem access
        if !self.enable_cross_filesystem {
            debug!("Cross-filesystem symlink resolution disabled");
        }
            return Err(DaemonError::SymlinkDepthExceeded {
                link_path: path.to_string_lossy().to_string(),
                depth,
                max_depth: self.max_symlink_depth,
            });
        }

        // Check for circular references
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if let Some(&previous_depth) = visited.get(&canonical_path) {
            return Err(DaemonError::SymlinkCircular {
                link_path: path.to_string_lossy().to_string(),
                cycle_depth: depth - previous_depth,
            });
        }
        visited.insert(canonical_path, depth);

        // Read the symlink target
        let target = tokio::fs::read_link(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read symlink target: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        // Resolve relative paths
        let resolved_target = if target.is_absolute() {
            target
        } else {
            path.parent()
                .ok_or_else(|| DaemonError::FileIo {
                    message: "Cannot determine parent directory".to_string(),
                    path: path.to_string_lossy().to_string(),
                })?
                .join(target)
        };

        // Check if target exists
        if !resolved_target.exists() {
            return Err(DaemonError::SymlinkBroken {
                link_path: path.to_string_lossy().to_string(),
                target_path: resolved_target.to_string_lossy().to_string(),
            });
        }

        // If target is also a symlink, resolve recursively
        if self.is_symlink(&resolved_target).await? {
            self.resolve_symlink_recursive(&resolved_target, depth + 1, visited).await
        } else {
            Ok(resolved_target)
        }
        })
    }

    /// Get comprehensive symlink information
    pub async fn get_symlink_info<P: AsRef<Path>>(&self, path: P) -> DaemonResult<SymlinkInfo> {
        let path = path.as_ref();

        if !self.is_symlink(path).await? {
            return Err(DaemonError::FileIo {
                message: "Path is not a symlink".to_string(),
                path: path.to_string_lossy().to_string(),
            });
        }

        // Read immediate target
        let target = tokio::fs::read_link(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read symlink target: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        // Resolve relative target
        let resolved_target = if target.is_absolute() {
            target.clone()
        } else {
            path.parent()
                .ok_or_else(|| DaemonError::FileIo {
                    message: "Cannot determine parent directory".to_string(),
                    path: path.to_string_lossy().to_string(),
                })?
                .join(&target)
        };

        // Calculate depth and final target
        let (depth, final_target, is_broken) = match self.calculate_symlink_depth(path, self.max_symlink_depth).await {
            Ok(d) => {
                match self.resolve_symlink(path).await {
                    Ok(ft) => (d, ft, false),
                    Err(_) => (d, resolved_target.clone(), true),
                }
            }
            Err(_) => (0, resolved_target.clone(), true),
        };

        // Check if cross-filesystem (simplified check)
        let is_cross_filesystem = false; // TODO: Implement proper cross-filesystem detection

        Ok(SymlinkInfo {
            target: resolved_target,
            final_target,
            is_broken,
            is_absolute: target.is_absolute(),
            depth,
            is_cross_filesystem,
        })
    }

    /// Calculate the depth of a symlink chain
    pub async fn calculate_symlink_depth<P: AsRef<Path>>(&self, path: P, max_depth: usize) -> DaemonResult<usize> {
        let path = path.as_ref();
        let mut visited = HashMap::new();

        self.calculate_depth_recursive(path, 0, max_depth, &mut visited).await
    }

    /// Recursive depth calculation
    fn calculate_depth_recursive<'a>(
        &'a self,
        path: &'a Path,
        current_depth: usize,
        max_depth: usize,
        visited: &'a mut HashMap<PathBuf, usize>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = DaemonResult<usize>> + Send + 'a>> {
        Box::pin(async move {
        if current_depth > max_depth {
            return Err(DaemonError::SymlinkDepthExceeded {
                link_path: path.to_string_lossy().to_string(),
                depth: current_depth,
                max_depth,
            });
        }

        if !self.is_symlink(path).await? {
            return Ok(current_depth);
        }

        // Check for cycles
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if visited.contains_key(&canonical_path) {
            return Err(DaemonError::SymlinkCircular {
                link_path: path.to_string_lossy().to_string(),
                cycle_depth: current_depth,
            });
        }
        visited.insert(canonical_path, current_depth);

        // Read target and recurse
        let target = tokio::fs::read_link(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read symlink target: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let resolved_target = if target.is_absolute() {
            target
        } else {
            path.parent()
                .ok_or_else(|| DaemonError::FileIo {
                    message: "Cannot determine parent directory".to_string(),
                    path: path.to_string_lossy().to_string(),
                })?
                .join(target)
        };

        if !resolved_target.exists() {
            return Ok(current_depth + 1); // Broken link, but still counts
        }

        self.calculate_depth_recursive(&resolved_target, current_depth + 1, max_depth, visited).await
        })
    }

    /// Check if a path is a hard link (has multiple links to the same inode)
    pub async fn is_hard_link<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let path = path.as_ref();

        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        // A file is considered a hard link if it has more than 1 link
        Ok(metadata.nlink() > 1)
    }

    /// Get the number of hard links to a file
    pub async fn get_hard_link_count<P: AsRef<Path>>(&self, path: P) -> DaemonResult<u64> {
        let path = path.as_ref();

        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        Ok(metadata.nlink())
    }

    /// Find all hard links to the same inode within a directory tree
    pub async fn find_hard_links<P: AsRef<Path>>(&self, search_root: P, target_inode: u64) -> DaemonResult<Vec<PathBuf>> {
        let search_root = search_root.as_ref();
        let mut hard_links = Vec::new();

        // Use walkdir for efficient directory traversal
        for entry in WalkDir::new(search_root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();

                // Check if this file has the same inode
                if let Ok(metadata) = tokio::fs::metadata(path).await {
                    if metadata.ino() == target_inode {
                        hard_links.push(path.to_path_buf());
                    }
                }
            }
        }

        Ok(hard_links)
    }

    /// Check if a path is a special file (pipe, socket, device)
    pub async fn is_special_file<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let path = path.as_ref();

        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let file_type = metadata.file_type();
        Ok(file_type.is_fifo() || file_type.is_socket() || file_type.is_char_device() || file_type.is_block_device())
    }

    /// Get the type of a special file
    pub async fn get_file_type<P: AsRef<Path>>(&self, path: P) -> DaemonResult<SpecialFileType> {
        let path = path.as_ref();

        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            DaemonError::FileIo {
                message: format!("Cannot read file metadata: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let file_type = metadata.file_type();

        if file_type.is_fifo() {
            Ok(SpecialFileType::NamedPipe)
        } else if file_type.is_socket() {
            Ok(SpecialFileType::Socket)
        } else if file_type.is_char_device() {
            Ok(SpecialFileType::CharacterDevice)
        } else if file_type.is_block_device() {
            Ok(SpecialFileType::BlockDevice)
        } else {
            Err(DaemonError::FileIo {
                message: "File is not a special file type".to_string(),
                path: path.to_string_lossy().to_string(),
            })
        }
    }

    /// Determine if a special file should be processed for content
    pub async fn should_process_special_file<P: AsRef<Path>>(&self, path: P) -> DaemonResult<bool> {
        let file_type = self.get_file_type(path).await?;

        // Generally, special files should not be processed for content
        match file_type {
            SpecialFileType::NamedPipe => Ok(false),      // Pipes are streams, not files
            SpecialFileType::Socket => Ok(false),         // Sockets are communication endpoints
            SpecialFileType::CharacterDevice => Ok(false), // Device files
            SpecialFileType::BlockDevice => Ok(false),    // Block devices
        }
    }
}

/// Stream-based file processor for large files
pub struct AsyncFileStream {
    processor: AsyncFileProcessor,
}

impl AsyncFileStream {
    pub fn new(processor: AsyncFileProcessor) -> Self {
        Self { processor }
    }

    /// Process file as a stream of chunks
    pub async fn process_stream<P>(&self, path: P, chunk_size: usize) -> DaemonResult<std::pin::Pin<Box<dyn tokio_stream::Stream<Item = DaemonResult<Vec<u8>>> + Send>>>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        debug!("Creating stream processor for file: {}", path.display());

        let file = File::open(path).await.map_err(|e| {
            error!("Failed to open file for streaming {}: {}", path.display(), e);
            DaemonError::FileIo {
                message: format!("Cannot open file for streaming: {}", e),
                path: path.to_string_lossy().to_string(),
            }
        })?;

        let reader = BufReader::new(file);
        Ok(Box::pin(unfold((reader, chunk_size), |(mut reader, chunk_size)| async move {
            let mut buffer = vec![0u8; chunk_size];
            match reader.read(&mut buffer).await {
                Ok(0) => None, // EOF
                Ok(n) => {
                    buffer.truncate(n);
                    Some((Ok(buffer), (reader, chunk_size)))
                }
                Err(e) => Some((Err(DaemonError::FileIo {
                    message: format!("Stream read error: {}", e),
                    path: "stream".to_string(),
                }), (reader, chunk_size)))
            }
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, TempDir};
    use tokio_test;
    use std::fs;
    use futures_util;

    #[tokio::test]
    async fn test_read_file_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let content = b"Hello, async file operations!";

        // Create test file
        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let result = processor.read_file(&file_path).await.unwrap();

        assert_eq!(result, content);
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("nonexistent.txt");

        let processor = AsyncFileProcessor::default();
        let result = processor.read_file(&file_path).await;

        assert!(result.is_err());
        match result {
            Err(DaemonError::FileIo { message, path }) => {
                assert!(message.contains("Cannot access file metadata"));
                assert_eq!(path, file_path.to_string_lossy());
            }
            _ => panic!("Expected FileIo error"),
        }
    }

    #[tokio::test]
    async fn test_read_file_too_large() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("large.txt");
        let content = vec![0u8; 1000];

        fs::write(&file_path, &content).unwrap();

        let processor = AsyncFileProcessor::new(500, DEFAULT_BUFFER_SIZE, true);
        let result = processor.read_file(&file_path).await;

        assert!(result.is_err());
        match result {
            Err(DaemonError::FileTooLarge { path, size, max_size }) => {
                assert_eq!(path, file_path.to_string_lossy());
                assert_eq!(size, 1000);
                assert_eq!(max_size, 500);
            }
            _ => panic!("Expected FileTooLarge error"),
        }
    }

    #[tokio::test]
    async fn test_write_file_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("write_test.txt");
        let content = b"Write test content";

        let processor = AsyncFileProcessor::default();
        processor.write_file(&file_path, content).await.unwrap();

        let read_content = fs::read(&file_path).unwrap();
        assert_eq!(read_content, content);
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("subdir").join("deep").join("file.txt");
        let content = b"Deep directory test";

        let processor = AsyncFileProcessor::default();
        processor.write_file(&file_path, content).await.unwrap();

        assert!(file_path.exists());
        let read_content = fs::read(&file_path).unwrap();
        assert_eq!(read_content, content);
    }

    #[tokio::test]
    async fn test_append_file_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("append_test.txt");
        let initial_content = b"Initial content\n";
        let append_content = b"Appended content\n";

        let processor = AsyncFileProcessor::default();

        // Write initial content
        processor.write_file(&file_path, initial_content).await.unwrap();

        // Append more content
        processor.append_file(&file_path, append_content).await.unwrap();

        let read_content = fs::read(&file_path).unwrap();
        let expected = [initial_content.as_slice(), append_content.as_slice()].concat();
        assert_eq!(read_content, expected);
    }

    #[tokio::test]
    async fn test_append_file_creates_new() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("new_append.txt");
        let content = b"New file content";

        let processor = AsyncFileProcessor::default();
        processor.append_file(&file_path, content).await.unwrap();

        let read_content = fs::read(&file_path).unwrap();
        assert_eq!(read_content, content);
    }

    #[tokio::test]
    async fn test_calculate_hash_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("hash_test.txt");
        let content = b"Content for hashing";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let hash = processor.calculate_hash(&file_path).await.unwrap();

        // Verify hash is a valid hex string
        assert_eq!(hash.len(), 64); // BLAKE3 produces 32-byte hash = 64 hex chars
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn test_calculate_hash_consistent() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("consistent_hash.txt");
        let content = b"Consistent content";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let hash1 = processor.calculate_hash(&file_path).await.unwrap();
        let hash2 = processor.calculate_hash(&file_path).await.unwrap();

        assert_eq!(hash1, hash2);
    }

    #[tokio::test]
    async fn test_process_chunks_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("chunks_test.txt");
        let content = b"0123456789abcdefghijklmnopqrstuvwxyz";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let results = processor.process_chunks(&file_path, 10, |chunk, index| {
            Ok(format!("Chunk {}: {} bytes", index, chunk.len()))
        }).await.unwrap();

        assert_eq!(results.len(), 4); // 36 bytes / 10 = 3.6 -> 4 chunks
        assert_eq!(results[0], "Chunk 0: 10 bytes");
        assert_eq!(results[1], "Chunk 1: 10 bytes");
        assert_eq!(results[2], "Chunk 2: 10 bytes");
        assert_eq!(results[3], "Chunk 3: 6 bytes"); // Last chunk
    }

    #[tokio::test]
    async fn test_copy_file_success() {
        let temp_dir = tempdir().unwrap();
        let src_path = temp_dir.path().join("source.txt");
        let dst_path = temp_dir.path().join("destination.txt");
        let content = b"File copy test content";

        fs::write(&src_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let bytes_copied = processor.copy_file(&src_path, &dst_path).await.unwrap();

        assert_eq!(bytes_copied, content.len() as u64);
        assert!(dst_path.exists());

        let copied_content = fs::read(&dst_path).unwrap();
        assert_eq!(copied_content, content);
    }

    #[tokio::test]
    async fn test_copy_file_creates_parent_dirs() {
        let temp_dir = tempdir().unwrap();
        let src_path = temp_dir.path().join("source.txt");
        let dst_path = temp_dir.path().join("sub").join("dir").join("destination.txt");
        let content = b"Parent directory creation test";

        fs::write(&src_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        processor.copy_file(&src_path, &dst_path).await.unwrap();

        assert!(dst_path.exists());
        let copied_content = fs::read(&dst_path).unwrap();
        assert_eq!(copied_content, content);
    }

    #[tokio::test]
    async fn test_validate_file_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("validate_test.txt");
        let content = b"Validation test content";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let file_info = processor.validate_file(&file_path).await.unwrap();

        assert_eq!(file_info.path, file_path);
        assert_eq!(file_info.size, content.len() as u64);
        assert!(file_info.is_file);
        assert!(!file_info.is_dir);
        assert!(file_info.created.is_some());
        assert!(file_info.modified.is_some());
    }

    #[tokio::test]
    async fn test_validate_directory() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().join("subdir");

        fs::create_dir(&dir_path).unwrap();

        let processor = AsyncFileProcessor::default();
        let file_info = processor.validate_file(&dir_path).await.unwrap();

        assert_eq!(file_info.path, dir_path);
        assert!(!file_info.is_file);
        assert!(file_info.is_dir);
    }

    #[tokio::test]
    async fn test_create_backup_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("backup_test.txt");
        let content = b"Backup test content";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let backup_path = processor.create_backup(&file_path).await.unwrap();

        assert!(backup_path.exists());
        assert_ne!(backup_path, file_path);

        let backup_content = fs::read(&backup_path).unwrap();
        assert_eq!(backup_content, content);
    }

    #[tokio::test]
    async fn test_atomic_replace_new_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("atomic_new.txt");
        let content = b"New file atomic content";

        let processor = AsyncFileProcessor::default();
        processor.atomic_replace(&file_path, content).await.unwrap();

        assert!(file_path.exists());
        let read_content = fs::read(&file_path).unwrap();
        assert_eq!(read_content, content);
    }

    #[tokio::test]
    async fn test_atomic_replace_existing_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("atomic_existing.txt");
        let original_content = b"Original content";
        let new_content = b"Replacement content";

        fs::write(&file_path, original_content).unwrap();

        let processor = AsyncFileProcessor::default();
        processor.atomic_replace(&file_path, new_content).await.unwrap();

        let read_content = fs::read(&file_path).unwrap();
        assert_eq!(read_content, new_content);
    }

    #[tokio::test]
    async fn test_async_file_stream() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("stream_test.txt");
        let content = b"0123456789abcdefghijklmnopqrstuvwxyz";

        fs::write(&file_path, content).unwrap();

        let processor = AsyncFileProcessor::default();
        let stream_processor = AsyncFileStream::new(processor);
        let mut stream = stream_processor.process_stream(&file_path, 10).await.unwrap();

        let mut chunks = Vec::new();
        use futures_util::stream::StreamExt;
        while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream).await {
            chunks.push(chunk_result.unwrap());
        }

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].len(), 10);
        assert_eq!(chunks[1].len(), 10);
        assert_eq!(chunks[2].len(), 10);
        assert_eq!(chunks[3].len(), 6);

        let reconstructed: Vec<u8> = chunks.into_iter().flatten().collect();
        assert_eq!(reconstructed, content);
    }

    #[tokio::test]
    async fn test_concurrent_file_operations() {
        let temp_dir = tempdir().unwrap();
        let processor = AsyncFileProcessor::default();

        let mut handles = Vec::new();

        // Spawn multiple concurrent file operations
        for i in 0..10 {
            let file_path = temp_dir.path().join(format!("concurrent_{}.txt", i));
            let content = format!("Concurrent content {}", i).into_bytes();
            let processor_clone = processor.clone();

            let handle = tokio::spawn(async move {
                // Write file
                processor_clone.write_file(&file_path, &content).await.unwrap();

                // Read file back
                let read_content = processor_clone.read_file(&file_path).await.unwrap();
                assert_eq!(read_content, content);

                // Calculate hash
                let hash = processor_clone.calculate_hash(&file_path).await.unwrap();
                assert_eq!(hash.len(), 64);

                i
            });

            handles.push(handle);
        }

        // Wait for all operations to complete
        let results = futures_util::future::join_all(handles).await;
        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), i);
        }
    }

    #[tokio::test]
    async fn test_error_handling_permission_denied() {
        // This test might not work on all systems, skip if we can't create
        // a readonly directory
        let temp_dir = tempdir().unwrap();
        let readonly_dir = temp_dir.path().join("readonly");

        fs::create_dir(&readonly_dir).unwrap();

        // Try to make directory readonly (this might not work on all platforms)
        #[cfg(unix)]
        {
            use std::fs::Permissions;
            use std::os::unix::fs::PermissionsExt;

            let permissions = Permissions::from_mode(0o444); // Read-only
            if fs::set_permissions(&readonly_dir, permissions).is_ok() {
                let file_path = readonly_dir.join("test.txt");
                let processor = AsyncFileProcessor::default();

                let result = processor.write_file(&file_path, b"test").await;
                assert!(result.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_edge_case_empty_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.txt");

        fs::write(&file_path, b"").unwrap();

        let processor = AsyncFileProcessor::default();

        let content = processor.read_file(&file_path).await.unwrap();
        assert_eq!(content, b"");

        let hash = processor.calculate_hash(&file_path).await.unwrap();
        assert_eq!(hash.len(), 64);

        let chunks = processor.process_chunks(&file_path, 10, |chunk, index| {
            Ok(format!("Chunk {}: {} bytes", index, chunk.len()))
        }).await.unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[tokio::test]
    async fn test_large_file_chunked_processing() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("large.txt");

        // Create a larger file (1MB)
        let chunk_data = vec![0u8; 1024];
        let mut large_content = Vec::new();
        for i in 0..1024 {
            large_content.extend_from_slice(&chunk_data);
        }

        fs::write(&file_path, &large_content).unwrap();

        let processor = AsyncFileProcessor::new(10 * 1024 * 1024, DEFAULT_BUFFER_SIZE, true);

        // Test chunked processing
        let chunks = processor.process_chunks(&file_path, 8192, |chunk, _index| {
            Ok(chunk.len())
        }).await.unwrap();

        let total_bytes: usize = chunks.iter().sum();
        assert_eq!(total_bytes, large_content.len());
    }

    #[tokio::test]
    async fn test_file_processor_configuration() {
        let processor = AsyncFileProcessor::new(1024, 512, false);

        assert_eq!(processor.max_file_size, 1024);
        assert_eq!(processor.buffer_size, 512);
        assert!(!processor.enable_compression);

        let default_processor = AsyncFileProcessor::default();
        assert_eq!(default_processor.max_file_size, MAX_FILE_SIZE);
        assert_eq!(default_processor.buffer_size, DEFAULT_BUFFER_SIZE);
        assert!(default_processor.enable_compression);
    }

    #[tokio::test]
    async fn test_unicode_file_paths() {
        let temp_dir = tempdir().unwrap();
        let unicode_filename = "测试文件_тест_テスト_🚀.txt";
        let file_path = temp_dir.path().join(unicode_filename);
        let content = b"Unicode filename test";

        let processor = AsyncFileProcessor::default();

        processor.write_file(&file_path, content).await.unwrap();
        assert!(file_path.exists());

        let read_content = processor.read_file(&file_path).await.unwrap();
        assert_eq!(read_content, content);

        let file_info = processor.validate_file(&file_path).await.unwrap();
        assert_eq!(file_info.size, content.len() as u64);
    }

    #[tokio::test]
    async fn test_special_characters_in_content() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("special_content.txt");
        let special_content = "Special chars: \0\n\r\t\x01\x1f\x7f".as_bytes();

        let processor = AsyncFileProcessor::default();

        processor.write_file(&file_path, special_content).await.unwrap();
        let read_content = processor.read_file(&file_path).await.unwrap();
        assert_eq!(read_content, special_content);

        let hash = processor.calculate_hash(&file_path).await.unwrap();
        assert_eq!(hash.len(), 64);
    }

    #[tokio::test]
    async fn test_file_info_debug_format() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("debug_test.txt");

        fs::write(&file_path, b"test").unwrap();

        let processor = AsyncFileProcessor::default();
        let file_info = processor.validate_file(&file_path).await.unwrap();

        let debug_str = format!("{:?}", file_info);
        assert!(debug_str.contains("FileInfo"));
        assert!(debug_str.contains("path"));
        assert!(debug_str.contains("size"));
    }

    #[tokio::test]
    async fn test_processor_clone() {
        let processor1 = AsyncFileProcessor::new(1024, 512, false);
        let processor2 = processor1.clone();

        assert_eq!(processor1.max_file_size, processor2.max_file_size);
        assert_eq!(processor1.buffer_size, processor2.buffer_size);
        assert_eq!(processor1.enable_compression, processor2.enable_compression);
    }
}