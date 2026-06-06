//! Grammar downloading, compilation, and automatic acquisition.
//!
//! Tree-sitter grammars are distributed as C source code, not pre-built binaries.
//! This module downloads source tarballs from GitHub releases, compiles them into
//! shared libraries (.dylib/.so/.dll), and caches the results.
//!
//! # Build process
//!
//! 1. Download source tarball from GitHub release (or archive fallback)
//! 2. Extract to a temp directory
//! 3. Compile `src/parser.c` (and optionally `src/scanner.c` or `src/scanner.cc`)
//!    into a shared library using the system C/C++ compiler
//! 4. Move the compiled library into the grammar cache

pub(crate) mod compile;
mod extract;
mod fetch;
#[cfg(test)]
mod tests;

pub use compile::{download_platform, library_extension};

use std::path::PathBuf;

use reqwest::Client;
use thiserror::Error;
use tracing::info;

use super::grammar_cache::{compute_checksum, GrammarCachePaths, GrammarMetadata};
use super::grammar_registry;

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

    #[error("Compilation failed for {language}: {message}")]
    CompilationFailed { language: String, message: String },

    #[error("No C compiler found. Install gcc or clang.")]
    NoCompiler,

    #[error("Unknown grammar source for language: {0}")]
    UnknownGrammar(String),

    #[error("Archive extraction failed: {0}")]
    ExtractionFailed(String),
}

/// Result type for download operations.
pub type DownloadResult<T> = Result<T, DownloadError>;

/// Verify downloaded tarball bytes against an expected SHA256 (hex).
///
/// Comparison is case-insensitive on the hex digits. Returns
/// [`DownloadError::ChecksumMismatch`] on any difference.
pub(crate) fn verify_tarball_checksum(bytes: &[u8], expected: &str) -> DownloadResult<()> {
    use sha2::{Digest, Sha256};

    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected.trim()) {
        Ok(())
    } else {
        Err(DownloadError::ChecksumMismatch {
            expected: expected.trim().to_lowercase(),
            actual,
        })
    }
}

/// Information about a downloaded grammar.
#[derive(Debug, Clone)]
pub struct DownloadedGrammar {
    /// Path to the compiled grammar library
    pub path: PathBuf,
    /// Checksum of the compiled library
    pub checksum: String,
    /// Language name
    pub language: String,
    /// Grammar version (tag or "main")
    pub version: String,
    /// Platform triple
    pub platform: String,
}

/// Grammar downloader that fetches source and compiles grammars.
pub struct GrammarDownloader {
    /// HTTP client for making requests
    client: Client,
    /// Grammar cache paths
    cache_paths: GrammarCachePaths,
    /// Whether to verify checksums
    verify_checksums: bool,
    /// Path to C compiler (resolved on creation)
    cc_path: Option<PathBuf>,
    /// Path to C++ compiler (resolved on creation)
    cxx_path: Option<PathBuf>,
}

impl GrammarDownloader {
    /// Create a new grammar downloader.
    ///
    /// Download URLs are derived from the grammar registry; the only knobs are
    /// the cache location and whether tarball checksums are verified.
    pub fn new(cache_paths: GrammarCachePaths, verify_checksums: bool) -> Self {
        let cc_path = compile::find_compiler("cc")
            .or_else(|| compile::find_compiler("gcc"))
            .or_else(|| compile::find_compiler("clang"));
        let cxx_path = compile::find_compiler("c++")
            .or_else(|| compile::find_compiler("g++"))
            .or_else(|| compile::find_compiler("clang++"));

        Self {
            client: Client::new(),
            cache_paths,
            verify_checksums,
            cc_path,
            cxx_path,
        }
    }

    /// Download and compile a grammar for the specified language.
    ///
    /// # Arguments
    ///
    /// * `language` - The language name (e.g., "rust", "python")
    /// * `version` - Ignored for now; version is determined by the grammar registry
    pub async fn download_grammar(
        &self,
        language: &str,
        version: &str,
    ) -> DownloadResult<DownloadedGrammar> {
        let source = grammar_registry::lookup(language)
            .ok_or_else(|| DownloadError::UnknownGrammar(language.to_string()))?;

        // Check compiler availability
        if source.has_cpp_scanner {
            if self.cxx_path.is_none() {
                return Err(DownloadError::NoCompiler);
            }
        } else if self.cc_path.is_none() {
            return Err(DownloadError::NoCompiler);
        }

        let platform = &self.cache_paths.platform;
        info!(language, %platform, repo = %source.repo, "Downloading grammar source");

        // Pinned ref when present, else release tarball then archive fallback
        let tarball_bytes = fetch::fetch_grammar_source(&self.client, language, &source).await?;

        // Verify the tarball against the registry's pinned sha256 (only
        // meaningful for pinned refs — moving targets have no stable hash).
        if self.verify_checksums {
            if let Some(ref expected) = source.sha256 {
                verify_tarball_checksum(&tarball_bytes, expected)?;
                info!(language, "Tarball sha256 verified against registry pin");
            }
        }

        // Extract and compile in a temp directory
        let temp_dir = tempfile::tempdir()?;
        let extract_dir = temp_dir.path();

        extract::extract_tarball(&tarball_bytes, extract_dir)?;

        // Find the src/ directory (may be in a subdirectory)
        let src_dir = extract::find_src_dir(extract_dir, source.src_subdir.as_deref())?;

        // Compile
        let grammar_lib = compile::compile_grammar(
            language,
            &src_dir,
            &source,
            extract_dir,
            self.cc_path.as_ref(),
            self.cxx_path.as_ref(),
        )
        .await?;

        // Move to cache
        self.cache_paths.create_directories(language)?;
        let grammar_path = self.cache_paths.grammar_path(language);
        std::fs::copy(&grammar_lib, &grammar_path)?;

        let checksum = compute_checksum(&grammar_path)?;

        // Record the pinned ref as the installed version when one exists —
        // it identifies the exact source bytes, unlike the caller's hint.
        let effective_version = source.git_ref.as_deref().unwrap_or(version);

        let metadata = GrammarMetadata::new(
            language,
            &self.cache_paths.tree_sitter_version,
            effective_version,
            platform,
            &checksum,
        );
        self.cache_paths.save_metadata(language, &metadata)?;

        info!(
            language,
            path = %grammar_path.display(),
            checksum = &checksum[..16],
            "Grammar compiled and cached successfully"
        );

        Ok(DownloadedGrammar {
            path: grammar_path,
            checksum,
            language: language.to_string(),
            version: effective_version.to_string(),
            platform: platform.to_string(),
        })
    }

    /// Check if a grammar needs to be downloaded.
    pub fn needs_download(&self, language: &str, _version: &str) -> bool {
        !self.cache_paths.grammar_exists(language)
    }

    /// Download a grammar only if it's not already cached.
    pub async fn ensure_grammar(&self, language: &str, version: &str) -> DownloadResult<PathBuf> {
        if self.needs_download(language, version) {
            let downloaded = self.download_grammar(language, version).await?;
            Ok(downloaded.path)
        } else {
            Ok(self.cache_paths.grammar_path(language))
        }
    }

    /// Download multiple grammars sequentially.
    pub async fn download_grammars(
        &self,
        grammars: &[(&str, &str)],
    ) -> Vec<DownloadResult<DownloadedGrammar>> {
        let mut results = Vec::with_capacity(grammars.len());
        for (lang, ver) in grammars {
            results.push(self.download_grammar(lang, ver).await);
        }
        results
    }

    /// Get the cache paths.
    pub fn cache_paths(&self) -> &GrammarCachePaths {
        &self.cache_paths
    }
}

impl std::fmt::Debug for GrammarDownloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarDownloader")
            .field("cache_paths", &self.cache_paths)
            .field("verify_checksums", &self.verify_checksums)
            .field("has_cc", &self.cc_path.is_some())
            .field("has_cxx", &self.cxx_path.is_some())
            .finish()
    }
}
