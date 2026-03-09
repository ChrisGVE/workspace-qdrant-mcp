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

use super::grammar_cache::{compute_checksum, GrammarCachePaths, GrammarMetadata};
use super::grammar_registry;
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
use tracing::{debug, info};

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
    /// The `_base_url_template` parameter is accepted for backward compatibility
    /// but ignored — URLs are now derived from the grammar registry.
    pub fn new(
        cache_paths: GrammarCachePaths,
        _base_url_template: impl Into<String>,
        verify_checksums: bool,
    ) -> Self {
        let cc_path = find_compiler("cc")
            .or_else(|| find_compiler("gcc"))
            .or_else(|| find_compiler("clang"));
        let cxx_path = find_compiler("c++")
            .or_else(|| find_compiler("g++"))
            .or_else(|| find_compiler("clang++"));

        Self {
            client: Client::new(),
            cache_paths,
            verify_checksums,
            cc_path,
            cxx_path,
        }
    }

    /// Create a downloader with default settings.
    pub fn with_default_url(cache_paths: GrammarCachePaths, verify_checksums: bool) -> Self {
        Self::new(cache_paths, "", verify_checksums)
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

        // Try release tarball first, then archive fallback
        let tarball_bytes = self.fetch_grammar_source(language, &source).await?;

        // Extract and compile in a temp directory
        let temp_dir = tempfile::tempdir()?;
        let extract_dir = temp_dir.path();

        extract_tarball(&tarball_bytes, extract_dir)?;

        // Find the src/ directory (may be in a subdirectory)
        let src_dir = find_src_dir(extract_dir, source.src_subdir)?;

        // Compile
        let grammar_lib = self
            .compile_grammar(language, &src_dir, &source, extract_dir)
            .await?;

        // Move to cache
        self.cache_paths.create_directories(language)?;
        let grammar_path = self.cache_paths.grammar_path(language);
        std::fs::copy(&grammar_lib, &grammar_path)?;

        let checksum = compute_checksum(&grammar_path)?;

        let metadata = GrammarMetadata::new(
            language,
            &self.cache_paths.tree_sitter_version,
            version,
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
            version: version.to_string(),
            platform: platform.to_string(),
        })
    }

    /// Fetch grammar source tarball, trying release URL first, then archive fallback.
    async fn fetch_grammar_source(
        &self,
        language: &str,
        source: &grammar_registry::GrammarSource,
    ) -> DownloadResult<Vec<u8>> {
        // Try the GitHub release tarball first (most grammars have releases)
        // We use the repo's latest release via the redirect URL
        let release_url = format!(
            "https://github.com/{}/{}/releases/latest/download/{}.tar.gz",
            source.owner, source.repo, source.repo
        );

        debug!(language, url = %release_url, "Trying release tarball");
        match self.fetch_bytes(&release_url, language, "latest").await {
            Ok(bytes) => {
                info!(language, "Downloaded from release tarball");
                return Ok(bytes);
            }
            Err(DownloadError::NotFound { .. }) | Err(DownloadError::HttpError { .. }) => {
                debug!(language, "No release tarball, trying archive fallback");
            }
            Err(e) => return Err(e),
        }

        // Fallback: download from branch archive
        // Some repos keep generated parser.c only on a specific branch
        let mut branches = Vec::new();
        if let Some(branch) = source.archive_branch {
            branches.push(branch);
        }
        branches.extend_from_slice(&["main", "master"]);
        for branch in &branches {
            let archive_url = source.archive_tarball_url(branch);
            debug!(language, url = %archive_url, "Trying archive tarball");
            match self.fetch_bytes(&archive_url, language, branch).await {
                Ok(bytes) => {
                    info!(language, branch, "Downloaded from archive");
                    return Ok(bytes);
                }
                Err(DownloadError::NotFound { .. }) | Err(DownloadError::HttpError { .. }) => {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(DownloadError::NotFound {
            language: language.to_string(),
            version: "latest".to_string(),
        })
    }

    /// Compile a grammar from its source directory.
    async fn compile_grammar(
        &self,
        language: &str,
        src_dir: &Path,
        _source: &grammar_registry::GrammarSource,
        work_dir: &Path,
    ) -> DownloadResult<PathBuf> {
        let output_path = work_dir.join(format!("grammar.{}", library_extension()));

        let parser_c = src_dir.join("parser.c");
        if !parser_c.exists() {
            return Err(DownloadError::CompilationFailed {
                language: language.to_string(),
                message: format!("parser.c not found in {}", src_dir.display()),
            });
        }

        // Collect source files
        let mut c_sources = vec![parser_c];
        let mut cpp_sources = Vec::new();

        let scanner_c = src_dir.join("scanner.c");
        let scanner_cc = src_dir.join("scanner.cc");

        if scanner_cc.exists() {
            cpp_sources.push(scanner_cc);
        } else if scanner_c.exists() {
            c_sources.push(scanner_c);
        }

        // Compile C sources
        let cc = self.cc_path.as_ref().ok_or(DownloadError::NoCompiler)?;
        let mut object_files = Vec::new();

        for (i, src) in c_sources.iter().enumerate() {
            let obj = work_dir.join(format!("c_{}.o", i));
            let status = Command::new(cc)
                .args(compile_c_args(src, &obj, src_dir))
                .status()
                .map_err(|e| DownloadError::CompilationFailed {
                    language: language.to_string(),
                    message: format!("Failed to run C compiler: {}", e),
                })?;

            if !status.success() {
                return Err(DownloadError::CompilationFailed {
                    language: language.to_string(),
                    message: format!("C compilation failed for {}", src.display()),
                });
            }
            object_files.push(obj);
        }

        // Compile C++ sources
        if !cpp_sources.is_empty() {
            let cxx = self.cxx_path.as_ref().ok_or(DownloadError::NoCompiler)?;
            for (i, src) in cpp_sources.iter().enumerate() {
                let obj = work_dir.join(format!("cpp_{}.o", i));
                let status = Command::new(cxx)
                    .args(compile_cxx_args(src, &obj, src_dir))
                    .status()
                    .map_err(|e| DownloadError::CompilationFailed {
                        language: language.to_string(),
                        message: format!("Failed to run C++ compiler: {}", e),
                    })?;

                if !status.success() {
                    return Err(DownloadError::CompilationFailed {
                        language: language.to_string(),
                        message: format!("C++ compilation failed for {}", src.display()),
                    });
                }
                object_files.push(obj);
            }
        }

        // Link into shared library
        let linker = if !cpp_sources.is_empty() {
            self.cxx_path.as_ref().ok_or(DownloadError::NoCompiler)?
        } else {
            cc
        };

        let status = Command::new(linker)
            .args(link_args(&object_files, &output_path))
            .status()
            .map_err(|e| DownloadError::CompilationFailed {
                language: language.to_string(),
                message: format!("Failed to link: {}", e),
            })?;

        if !status.success() {
            return Err(DownloadError::CompilationFailed {
                language: language.to_string(),
                message: "Linking failed".to_string(),
            });
        }

        info!(language, path = %output_path.display(), "Grammar compiled successfully");
        Ok(output_path)
    }

    /// Fetch raw bytes from a URL, mapping HTTP errors to `DownloadError`.
    async fn fetch_bytes(
        &self,
        url: &str,
        language: &str,
        version: &str,
    ) -> DownloadResult<Vec<u8>> {
        let response = self
            .client
            .get(url)
            .header("User-Agent", "workspace-qdrant-mcp grammar-downloader")
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
                message: format!("Failed to download from {}", url),
            });
        }

        let bytes = response.bytes().await.map_err(|e| {
            DownloadError::NetworkError(format!("Failed to read response body: {}", e))
        })?;
        Ok(bytes.to_vec())
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

/// Extract a gzipped tarball into a directory.
///
/// Uses entry-by-entry extraction to handle tarballs that lack directory
/// entries (common in GitHub release tarballs).
fn extract_tarball(bytes: &[u8], dest: &Path) -> DownloadResult<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let decoder = GzDecoder::new(std::io::Cursor::new(bytes));
    let mut archive = Archive::new(decoder);
    // Some release tarballs omit directory entries, so we extract
    // entry-by-entry and create parent directories as needed.
    for entry in archive
        .entries()
        .map_err(|e| DownloadError::ExtractionFailed(format!("Failed to read tarball: {}", e)))?
    {
        let mut entry = entry
            .map_err(|e| DownloadError::ExtractionFailed(format!("Failed to read entry: {}", e)))?;
        let path = entry
            .path()
            .map_err(|e| DownloadError::ExtractionFailed(format!("Invalid entry path: {}", e)))?;
        let full_path = dest.join(&*path);

        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&full_path).ok();
        } else if entry.header().entry_type().is_file() {
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    DownloadError::ExtractionFailed(format!(
                        "Failed to create directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
            entry.unpack(&full_path).map_err(|e| {
                DownloadError::ExtractionFailed(format!(
                    "Failed to unpack {}: {}",
                    full_path.display(),
                    e
                ))
            })?;
        }
    }
    Ok(())
}

/// Find the `src/` directory containing `parser.c` within an extracted tarball.
///
/// Handles multiple tarball formats:
/// - Files at root with `./src/parser.c` (release tarballs, e.g. tree-sitter-rust)
/// - Single top-level directory (archive tarballs, e.g. `tree-sitter-rust-main/src/parser.c`)
/// - Grammar in a subdirectory (monorepos, e.g. `typescript/src/parser.c`)
fn find_src_dir(extract_dir: &Path, subdir: Option<&str>) -> DownloadResult<PathBuf> {
    // Strategy 1: Check if src/parser.c exists directly in extract_dir
    // (release tarballs often extract files directly)
    let direct_src = if let Some(sub) = subdir {
        extract_dir.join(sub).join("src")
    } else {
        extract_dir.join("src")
    };
    if direct_src.join("parser.c").exists() {
        return Ok(direct_src);
    }

    // Strategy 2: Look for a single top-level directory (archive tarballs)
    let entries: Vec<_> = std::fs::read_dir(extract_dir)
        .map_err(|e| DownloadError::ExtractionFailed(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if entries.len() == 1 {
        let top_dir = entries[0].path();
        let grammar_root = if let Some(sub) = subdir {
            top_dir.join(sub)
        } else {
            top_dir
        };

        let src_dir = grammar_root.join("src");
        if src_dir.join("parser.c").exists() {
            return Ok(src_dir);
        }
        if grammar_root.join("parser.c").exists() {
            return Ok(grammar_root);
        }
    }

    // Strategy 3: Search recursively for parser.c (last resort)
    for e in walkdir::WalkDir::new(extract_dir)
        .max_depth(4)
        .into_iter()
        .flatten()
    {
        if e.file_name() == "parser.c" {
            if let Some(parent) = e.path().parent() {
                return Ok(parent.to_path_buf());
            }
        }
    }

    Err(DownloadError::ExtractionFailed(format!(
        "Could not find parser.c in extracted archive at {}",
        extract_dir.display()
    )))
}

/// Build C compiler arguments for compiling a source file to an object file.
fn compile_c_args(src: &Path, obj: &Path, include_dir: &Path) -> Vec<String> {
    let mut args = vec![
        "-c".to_string(),
        "-fPIC".to_string(),
        "-O2".to_string(),
        "-I".to_string(),
        include_dir.to_string_lossy().to_string(),
    ];

    // tree-sitter headers are in src/tree_sitter/
    let ts_include = include_dir.join("tree_sitter");
    if ts_include.exists() {
        args.push("-I".to_string());
        args.push(
            include_dir
                .parent()
                .unwrap_or(include_dir)
                .to_string_lossy()
                .to_string(),
        );
    }

    args.push("-o".to_string());
    args.push(obj.to_string_lossy().to_string());
    args.push(src.to_string_lossy().to_string());
    args
}

/// Build C++ compiler arguments for compiling a source file to an object file.
fn compile_cxx_args(src: &Path, obj: &Path, include_dir: &Path) -> Vec<String> {
    let mut args = compile_c_args(src, obj, include_dir);
    args.push("-std=c++14".to_string());
    args
}

/// Build linker arguments for creating a shared library from object files.
fn link_args(objects: &[PathBuf], output: &Path) -> Vec<String> {
    let mut args = vec!["-shared".to_string()];

    #[cfg(not(target_os = "windows"))]
    args.push("-fPIC".to_string());

    #[cfg(target_os = "macos")]
    {
        args.push("-undefined".to_string());
        args.push("dynamic_lookup".to_string());
    }

    args.push("-o".to_string());
    args.push(output.to_string_lossy().to_string());

    for obj in objects {
        args.push(obj.to_string_lossy().to_string());
    }

    args
}

/// Find a compiler on PATH.
fn find_compiler(name: &str) -> Option<PathBuf> {
    which::which(name).ok()
}

/// Get the library extension for the current platform.
pub fn library_extension() -> &'static str {
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

/// Get the current platform string.
pub fn download_platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
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
    fn test_needs_download_no_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");
        let downloader = GrammarDownloader::with_default_url(cache_paths, false);

        assert!(downloader.needs_download("rust", "0.24.0"));
    }

    #[test]
    fn test_needs_download_with_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");

        // Create a fake cached grammar
        cache_paths.create_directories("rust").unwrap();
        std::fs::write(cache_paths.grammar_path("rust"), "fake grammar").unwrap();

        let downloader = GrammarDownloader::with_default_url(cache_paths, false);

        // Cached grammar exists, no download needed
        assert!(!downloader.needs_download("rust", "0.24.0"));
    }

    #[test]
    fn test_downloader_debug() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");
        let downloader = GrammarDownloader::with_default_url(cache_paths, true);

        let debug_str = format!("{:?}", downloader);
        assert!(debug_str.contains("GrammarDownloader"));
        assert!(debug_str.contains("verify_checksums"));
    }

    #[test]
    fn test_find_compiler() {
        // On any dev machine, at least one of these should exist
        let has_compiler = find_compiler("cc").is_some()
            || find_compiler("gcc").is_some()
            || find_compiler("clang").is_some();
        assert!(has_compiler, "No C compiler found on PATH");
    }

    #[test]
    fn test_compile_c_args() {
        let src = Path::new("/tmp/src/parser.c");
        let obj = Path::new("/tmp/build/parser.o");
        let include = Path::new("/tmp/src");
        let args = compile_c_args(src, obj, include);
        assert!(args.contains(&"-c".to_string()));
        assert!(args.contains(&"-fPIC".to_string()));
        assert!(args.contains(&"-O2".to_string()));
    }

    #[test]
    fn test_link_args() {
        let objects = vec![PathBuf::from("/tmp/a.o"), PathBuf::from("/tmp/b.o")];
        let output = Path::new("/tmp/grammar.dylib");
        let args = link_args(&objects, output);
        assert!(args.contains(&"-shared".to_string()));
        assert!(args.contains(&"/tmp/a.o".to_string()));
        assert!(args.contains(&"/tmp/b.o".to_string()));
    }

    #[test]
    fn test_extract_tarball() {
        // Create a minimal tarball in memory
        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("test-grammar");
        std::fs::create_dir_all(src_dir.join("src")).unwrap();
        std::fs::write(src_dir.join("src/parser.c"), "// test").unwrap();

        // Create tarball — must drop builder+encoder before reading
        let tar_path = temp_dir.path().join("test.tar.gz");
        {
            let tar_file = std::fs::File::create(&tar_path).unwrap();
            let enc = flate2::write::GzEncoder::new(tar_file, flate2::Compression::default());
            let mut builder = tar::Builder::new(enc);
            builder.append_dir_all("test-grammar", &src_dir).unwrap();
            let enc = builder.into_inner().unwrap();
            enc.finish().unwrap();
        }

        // Extract
        let extract_dir = temp_dir.path().join("extracted");
        std::fs::create_dir_all(&extract_dir).unwrap();
        let bytes = std::fs::read(&tar_path).unwrap();
        extract_tarball(&bytes, &extract_dir).unwrap();

        // Find src dir
        let found = find_src_dir(&extract_dir, None).unwrap();
        assert!(found.join("parser.c").exists());
    }
}
