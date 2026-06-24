//! External compressor detection and invocation for `wqm backup --full` (F20).
//!
//! Detection order: `zstd` -> `xz` -> `gzip` (AC-F20.3).  First binary found on
//! PATH (resolved to an absolute path) wins.  Compression and decompression are
//! invoked with a **fixed argv** -- `Command::new(<abs-path>).args([...])` --
//! never via `sh -c`, so CWE-78 / Guard-4 / AC-F12.6 are not violated.
//!
//! ## Compress (backup)
//!
//! The caller writes raw tar bytes to the compressor's **stdin**; the compressor
//! writes compressed bytes to its **stdout**; the caller redirects stdout to the
//! destination archive file.  This streams the archive without ever holding an
//! uncompressed copy on disk.
//!
//! ## Decompress (restore)
//!
//! The caller opens the archive file and pipes it into the decompressor's
//! **stdin**; the decompressor writes decompressed tar bytes to **stdout** which
//! the caller streams into `tar::Archive`.  Peak memory is bounded by pipe
//! buffers (PERF-NN-02).

use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

/// A resolved external compressor.
#[derive(Debug, Clone)]
pub(crate) struct Compressor {
    /// Absolute path to the compressor binary.
    pub bin: PathBuf,
    /// Human-readable name (`"zstd"`, `"xz"`, `"gzip"`).
    pub name: &'static str,
}

/// Probe PATH for the first available compressor in preference order.
///
/// Order: `zstd` -> `xz` -> `gzip`.  Returns `None` when none is installed.
pub(crate) fn detect() -> Option<Compressor> {
    for (name, candidates) in &[
        ("zstd", &["zstd"][..]),
        ("xz", &["xz"][..]),
        ("gzip", &["gzip"][..]),
    ] {
        for &candidate in *candidates {
            if let Ok(path) = which::which(candidate) {
                return Some(Compressor { bin: path, name });
            }
        }
    }
    None
}

impl Compressor {
    /// Fixed compress argv: reads stdin, writes compressed bytes to stdout.
    ///
    /// The compressor binary is never passed untrusted user input; the only
    /// argument is `-` (read from stdin).  Guard-4 / CWE-78 safe.
    fn compress_args(&self) -> Vec<&OsStr> {
        match self.name {
            "zstd" => vec![OsStr::new("-c"), OsStr::new("-")],
            "xz" => vec![OsStr::new("-c"), OsStr::new("-")],
            "gzip" => vec![OsStr::new("-c"), OsStr::new("-")],
            _ => vec![OsStr::new("-c"), OsStr::new("-")],
        }
    }

    /// Fixed decompress argv: reads compressed bytes from stdin, writes tar
    /// bytes to stdout.
    fn decompress_args(&self) -> Vec<&OsStr> {
        match self.name {
            "zstd" => vec![OsStr::new("-d"), OsStr::new("-c"), OsStr::new("-")],
            "xz" => vec![OsStr::new("-d"), OsStr::new("-c"), OsStr::new("-")],
            "gzip" => vec![OsStr::new("-d"), OsStr::new("-c"), OsStr::new("-")],
            _ => vec![OsStr::new("-d"), OsStr::new("-c"), OsStr::new("-")],
        }
    }

    /// Spawn a compression child: stdin=piped, stdout=inherited (caller has
    /// already redirected stdout to the destination file) or piped.
    pub(crate) fn spawn_compress(&self, stdout: Stdio) -> Result<std::process::Child> {
        Command::new(&self.bin)
            .args(self.compress_args())
            .stdin(Stdio::piped())
            .stdout(stdout)
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn compressor '{}'", self.name))
    }

    /// Spawn a decompression child: stdin=piped (caller feeds the archive),
    /// stdout=piped (caller reads decompressed tar bytes as a stream).
    pub(crate) fn spawn_decompress(&self, stdin: Stdio) -> Result<std::process::Child> {
        Command::new(&self.bin)
            .args(self.decompress_args())
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn decompressor '{}'", self.name))
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "compressor_tests.rs"]
mod tests;
