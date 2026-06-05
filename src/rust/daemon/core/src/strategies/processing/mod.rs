//! Per-item-type processing strategies.
//!
//! Each module implements `ProcessingStrategy` for a specific `ItemType`,
//! extracting the processing logic from `unified_queue_processor.rs` into
//! focused, testable units.

pub mod collection;
pub mod file;
pub mod folder;
pub mod tenant;
pub mod text;
pub mod url;
pub mod url_fetch;
pub mod url_security;
pub mod website;

/// Single source of truth for the maximum size of a file ingested during a
/// scan (chunked + embedded + keyword-indexed). Previously this was a 100 MB
/// constant duplicated across the folder-scan and library-tenant paths; it is
/// unified here and lowered to a saner default.
///
/// 10 MB is far above any real source/doc file but excludes the giant
/// generated/build/data files (minified bundles, vendored blobs, logs) that
/// inflate memory and pollute the sparse/keyword vocabulary with base64/hash
/// junk tokens. Override with `WQM_MAX_INGEST_FILE_MB` (megabytes, > 0).
pub fn max_ingest_file_bytes() -> u64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let mb = std::env::var("WQM_MAX_INGEST_FILE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(10);
        mb * 1024 * 1024
    })
}
