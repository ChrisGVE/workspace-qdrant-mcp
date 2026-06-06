//! Unit tests for the grammar_downloader module.

use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::compile::{compile_c_args, find_compiler, link_args};
use super::extract::{extract_tarball, find_src_dir};
use super::GrammarDownloader;
use crate::tree_sitter::grammar_cache::GrammarCachePaths;

#[test]
fn test_library_extension() {
    let ext = super::compile::library_extension();
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
    let platform = super::compile::download_platform();
    assert!(!platform.is_empty());
    assert!(platform.contains('-'));
}

#[test]
fn test_needs_download_no_cache() {
    let temp_dir = TempDir::new().unwrap();
    let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");
    let downloader = GrammarDownloader::new(cache_paths, false);

    assert!(downloader.needs_download("rust", "0.24.0"));
}

#[test]
fn test_needs_download_with_cache() {
    let temp_dir = TempDir::new().unwrap();
    let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");

    // Create a fake cached grammar
    cache_paths.create_directories("rust").unwrap();
    std::fs::write(cache_paths.grammar_path("rust"), "fake grammar").unwrap();

    let downloader = GrammarDownloader::new(cache_paths, false);

    // Cached grammar exists, no download needed
    assert!(!downloader.needs_download("rust", "0.24.0"));
}

#[test]
fn test_downloader_debug() {
    let temp_dir = TempDir::new().unwrap();
    let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.26");
    let downloader = GrammarDownloader::new(cache_paths, true);

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
