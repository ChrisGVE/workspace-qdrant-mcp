//! Grammar compilation: compiling C/C++ sources into shared libraries.

use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::info;
use which;

use super::{DownloadError, DownloadResult};
use crate::tree_sitter::grammar_registry;

/// Compile a grammar from its source directory into a shared library.
///
/// Collects `parser.c` (and optionally `scanner.c` / `scanner.cc`), compiles
/// each to an object file, then links everything into a shared library at
/// `work_dir/grammar.<ext>`.
pub(super) async fn compile_grammar(
    language: &str,
    src_dir: &Path,
    _source: &grammar_registry::GrammarSource,
    work_dir: &Path,
    cc_path: Option<&PathBuf>,
    cxx_path: Option<&PathBuf>,
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
    let (c_sources, cpp_sources) = collect_sources(src_dir, parser_c);

    let cc = cc_path.ok_or(DownloadError::NoCompiler)?;
    let mut object_files = Vec::new();

    // Compile C sources
    compile_c_sources(
        language,
        &c_sources,
        work_dir,
        cc,
        src_dir,
        &mut object_files,
    )?;

    // Compile C++ sources
    compile_cpp_sources(
        language,
        &cpp_sources,
        work_dir,
        cxx_path,
        src_dir,
        &mut object_files,
    )?;

    // Link into shared library — prefer C++ linker when C++ sources present
    link_shared_library(
        language,
        &cpp_sources,
        cc,
        cxx_path,
        &object_files,
        &output_path,
    )?;

    info!(language, path = %output_path.display(), "Grammar compiled successfully");
    Ok(output_path)
}

/// Collect C and C++ source files from the source directory.
fn collect_sources(src_dir: &Path, parser_c: PathBuf) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut c_sources = vec![parser_c];
    let mut cpp_sources = Vec::new();

    let scanner_c = src_dir.join("scanner.c");
    let scanner_cc = src_dir.join("scanner.cc");

    if scanner_cc.exists() {
        cpp_sources.push(scanner_cc);
    } else if scanner_c.exists() {
        c_sources.push(scanner_c);
    }

    (c_sources, cpp_sources)
}

/// Compile all C source files to object files.
fn compile_c_sources(
    language: &str,
    c_sources: &[PathBuf],
    work_dir: &Path,
    cc: &PathBuf,
    src_dir: &Path,
    object_files: &mut Vec<PathBuf>,
) -> DownloadResult<()> {
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
    Ok(())
}

/// Compile all C++ source files to object files.
fn compile_cpp_sources(
    language: &str,
    cpp_sources: &[PathBuf],
    work_dir: &Path,
    cxx_path: Option<&PathBuf>,
    src_dir: &Path,
    object_files: &mut Vec<PathBuf>,
) -> DownloadResult<()> {
    if cpp_sources.is_empty() {
        return Ok(());
    }
    let cxx = cxx_path.ok_or(DownloadError::NoCompiler)?;
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
    Ok(())
}

/// Link object files into a shared library.
fn link_shared_library(
    language: &str,
    cpp_sources: &[PathBuf],
    cc: &PathBuf,
    cxx_path: Option<&PathBuf>,
    object_files: &[PathBuf],
    output_path: &Path,
) -> DownloadResult<()> {
    let linker = if !cpp_sources.is_empty() {
        cxx_path.ok_or(DownloadError::NoCompiler)?
    } else {
        cc
    };

    let status = Command::new(linker)
        .args(link_args(object_files, output_path))
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
    Ok(())
}

/// Build C compiler arguments for compiling a source file to an object file.
pub(super) fn compile_c_args(src: &Path, obj: &Path, include_dir: &Path) -> Vec<String> {
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
pub(super) fn compile_cxx_args(src: &Path, obj: &Path, include_dir: &Path) -> Vec<String> {
    let mut args = compile_c_args(src, obj, include_dir);
    args.push("-std=c++14".to_string());
    args
}

/// Build linker arguments for creating a shared library from object files.
pub(super) fn link_args(objects: &[PathBuf], output: &Path) -> Vec<String> {
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

/// Find a compiler executable on PATH.
pub(super) fn find_compiler(name: &str) -> Option<PathBuf> {
    which::which(name).ok()
}

/// Get the shared-library file extension for the current platform.
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

/// Get the current platform string in `arch-os` format.
pub fn download_platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}
