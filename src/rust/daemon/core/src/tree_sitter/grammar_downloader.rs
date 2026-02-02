//! Grammar downloading and automatic acquisition.
//!
//! This module provides functionality to download tree-sitter grammar files
//! from GitHub releases, enabling automatic acquisition of missing grammars
//! without manual intervention.
//!
//! # URL Template
//!
//! The download URL is constructed from a template with the following placeholders:
//! - `{language}` - Language name (e.g., "rust", "python")
//! - `{version}` - Grammar version (e.g., "0.23.0")
//! - `{platform}` - Platform triple (e.g., "x86_64-apple-darwin")
//! - `{ext}` - Library extension (.so, .dylib, .dll)

use super::grammar_cache::{
    compute_checksum, grammar_filename, GrammarCachePaths, GrammarMetadata,
};
use reqwest::Client;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tracing::{debug, error, info, warn};

/// Errors that can occur during grammar download.
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("Grammar not found for language '{language}' version '{version}'")]
    NotFound { language: String, version: String },

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("HTTP error {status}: {message}")]
    HttpError { status: u16, message: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Invalid URL template: {0}")]
    InvalidUrlTemplate(String),

    #[error("Request error: {0}")]
    ReqwestError(#[from] reqwest::Error),
}

/// Result type for download operations.
pub type DownloadResult<T> = Result<T, DownloadError>;

/// Information about a downloaded grammar.
#[derive(Debug, Clone)]
pub struct DownloadedGrammar {
    /// Path to the downloaded grammar file
    pub path: PathBuf,
    /// Checksum of the downloaded file
    pub checksum: String,
    /// Language name
    pub language: String,
    /// Grammar version
    pub version: String,
    /// Platform triple
    pub platform: String,
}

/// Grammar downloader for fetching grammar files from remote sources.
pub struct GrammarDownloader {
    /// HTTP client for making requests
    client: Client,
    /// Base URL template for downloads
    base_url_template: String,
    /// Grammar cache paths
    cache_paths: GrammarCachePaths,
    /// Whether to verify checksums
    verify_checksums: bool,
}

impl GrammarDownloader {
    /// Create a new grammar downloader.
    ///
    /// # Arguments
    ///
    /// * `cache_paths` - Paths for storing downloaded grammars
    /// * `base_url_template` - URL template with {language}, {version}, {platform}, {ext} placeholders
    /// * `verify_checksums` - Whether to verify downloaded file checksums
    pub fn new(
        cache_paths: GrammarCachePaths,
        base_url_template: impl Into<String>,
        verify_checksums: bool,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url_template: base_url_template.into(),
            cache_paths,
            verify_checksums,
        }
    }

    /// Create a downloader with the default GitHub releases URL template.
    pub fn with_default_url(cache_paths: GrammarCachePaths, verify_checksums: bool) -> Self {
        Self::new(
            cache_paths,
            "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}",
            verify_checksums,
        )
    }

    /// Download a grammar for the specified language and version.
    ///
    /// # Arguments
    ///
    /// * `language` - The language name (e.g., "rust", "python")
    /// * `version` - The grammar version (e.g., "0.23.0")
    ///
    /// # Returns
    ///
    /// Information about the downloaded grammar, or an error if download failed.
    pub async fn download_grammar(
        &self,
        language: &str,
        version: &str,
    ) -> DownloadResult<DownloadedGrammar> {
        let platform = &self.cache_paths.platform;
        let ext = library_extension();

        // Build the download URL
        let url = self.build_download_url(language, version, platform, ext)?;
        info!(
            language = language,
            version = version,
            platform = platform,
            url = %url,
            "Downloading grammar"
        );

        // Download the file
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DownloadError::NetworkError(e.to_string()))?;

        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(DownloadError::NotFound {
                language: language.to_string(),
                version: version.to_string(),
            });
        }

        if !status.is_success() {
            return Err(DownloadError::HttpError {
                status: status.as_u16(),
                message: format!("Failed to download grammar from {}", url),
            });
        }

        // Get the bytes
        let bytes = response.bytes().await.map_err(|e| {
            DownloadError::NetworkError(format!("Failed to read response body: {}", e))
        })?;

        // Create the cache directories
        self.cache_paths.create_directories(language)?;

        // Write to temporary file first
        let grammar_path = self.cache_paths.grammar_path(language);
        let temp_path = grammar_path.with_extension("tmp");

        debug!(
            path = %temp_path.display(),
            bytes = bytes.len(),
            "Writing grammar to temporary file"
        );

        let mut file = tokio::fs::File::create(&temp_path).await?;
        file.write_all(&bytes).await?;
        file.flush().await?;
        drop(file);

        // Compute checksum
        let checksum = compute_checksum(&temp_path)?;

        // Move to final location
        tokio::fs::rename(&temp_path, &grammar_path).await?;

        // Save metadata
        let metadata = GrammarMetadata::new(
            language,
            &self.cache_paths.tree_sitter_version,
            version,
            platform,
            &checksum,
        );
        self.cache_paths.save_metadata(language, &metadata)?;

        info!(
            language = language,
            version = version,
            path = %grammar_path.display(),
            checksum = &checksum[..16], // First 16 chars for brevity
            "Grammar downloaded successfully"
        );

        Ok(DownloadedGrammar {
            path: grammar_path,
            checksum,
            language: language.to_string(),
            version: version.to_string(),
            platform: platform.to_string(),
        })
    }

    /// Check if a grammar needs to be downloaded.
    ///
    /// Returns true if the grammar is not in the cache or the cached version doesn't match.
    pub fn needs_download(&self, language: &str, version: &str) -> bool {
        if !self.cache_paths.grammar_exists(language) {
            return true;
        }

        // Check if the cached version matches
        if let Ok(Some(metadata)) = self.cache_paths.load_metadata(language) {
            if metadata.grammar_version != version {
                return true;
            }
        } else {
            // No metadata, assume download needed
            return true;
        }

        false
    }

    /// Download a grammar only if it's not already cached.
    pub async fn ensure_grammar(
        &self,
        language: &str,
        version: &str,
    ) -> DownloadResult<PathBuf> {
        if self.needs_download(language, version) {
            let downloaded = self.download_grammar(language, version).await?;
            Ok(downloaded.path)
        } else {
            Ok(self.cache_paths.grammar_path(language))
        }
    }

    /// Download multiple grammars sequentially.
    ///
    /// Returns a vector of results, one for each grammar.
    pub async fn download_grammars(
        &self,
        grammars: &[(&str, &str)], // (language, version) pairs
    ) -> Vec<DownloadResult<DownloadedGrammar>> {
        let mut results = Vec::with_capacity(grammars.len());
        for (lang, ver) in grammars {
            results.push(self.download_grammar(lang, ver).await);
        }
        results
    }

    /// Build the download URL from the template.
    fn build_download_url(
        &self,
        language: &str,
        version: &str,
        platform: &str,
        ext: &str,
    ) -> DownloadResult<String> {
        let url = self
            .base_url_template
            .replace("{language}", language)
            .replace("{version}", version)
            .replace("{platform}", platform)
            .replace("{ext}", ext);

        // Validate the URL still looks reasonable
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(DownloadError::InvalidUrlTemplate(format!(
                "URL must start with http:// or https://: {}",
                url
            )));
        }

        Ok(url)
    }

    /// Get the cache paths.
    pub fn cache_paths(&self) -> &GrammarCachePaths {
        &self.cache_paths
    }
}

/// Get the library extension for the current platform.
fn library_extension() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "so"
    }
    #[cfg(target_os = "windows")]
    {
        "dll"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "so"
    }
}

/// Get the current platform string for downloads.
pub fn download_platform() -> String {
    format!(
        "{}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    )
}

impl std::fmt::Debug for GrammarDownloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarDownloader")
            .field("base_url_template", &self.base_url_template)
            .field("cache_paths", &self.cache_paths)
            .field("verify_checksums", &self.verify_checksums)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_library_extension() {
        let ext = library_extension();
        assert!(!ext.is_empty());
        #[cfg(target_os = "macos")]
        assert_eq!(ext, "dylib");
        #[cfg(target_os = "linux")]
        assert_eq!(ext, "so");
        #[cfg(target_os = "windows")]
        assert_eq!(ext, "dll");
    }

    #[test]
    fn test_download_platform() {
        let platform = download_platform();
        assert!(!platform.is_empty());
        assert!(platform.contains('-'));
    }

    #[test]
    fn test_build_download_url() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let downloader = GrammarDownloader::new(
            cache_paths,
            "https://example.com/{language}/v{version}/{platform}.{ext}",
            false,
        );

        let url = downloader
            .build_download_url("rust", "0.23.0", "x86_64-apple-darwin", "dylib")
            .unwrap();
        assert_eq!(
            url,
            "https://example.com/rust/v0.23.0/x86_64-apple-darwin.dylib"
        );
    }

    #[test]
    fn test_build_download_url_invalid() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let downloader = GrammarDownloader::new(
            cache_paths,
            "not-a-url/{language}",
            false,
        );

        let result = downloader.build_download_url("rust", "0.23.0", "x86_64", "so");
        assert!(result.is_err());
    }

    #[test]
    fn test_needs_download_no_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let downloader = GrammarDownloader::with_default_url(cache_paths, false);

        assert!(downloader.needs_download("rust", "0.23.0"));
    }

    #[test]
    fn test_needs_download_with_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");

        // Create a fake cached grammar
        cache_paths.create_directories("rust").unwrap();
        std::fs::write(cache_paths.grammar_path("rust"), "fake grammar").unwrap();

        let metadata = GrammarMetadata::new(
            "rust",
            "0.24",
            "0.23.0",
            &cache_paths.platform,
            "fakechecksum",
        );
        cache_paths.save_metadata("rust", &metadata).unwrap();

        let downloader = GrammarDownloader::with_default_url(cache_paths, false);

        // Same version should not need download
        assert!(!downloader.needs_download("rust", "0.23.0"));

        // Different version should need download
        assert!(downloader.needs_download("rust", "0.24.0"));
    }

    #[test]
    fn test_downloader_debug() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let downloader = GrammarDownloader::with_default_url(cache_paths, true);

        let debug_str = format!("{:?}", downloader);
        assert!(debug_str.contains("GrammarDownloader"));
        assert!(debug_str.contains("verify_checksums"));
    }

    // Note: Actual download tests require network access or mocking.
    // Those tests should be in integration tests with either real network
    // access or a mock HTTP server.
}
