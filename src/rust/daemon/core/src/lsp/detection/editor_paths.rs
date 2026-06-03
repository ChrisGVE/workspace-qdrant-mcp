//! Editor-managed LSP server discovery.
//!
//! The implementation now lives in `wqm_common::lsp_detection` (WI-b1); this
//! module re-exports the public API so existing
//! `crate::lsp::detection::editor_paths::*` paths keep resolving unchanged.

pub use wqm_common::lsp_detection::{
    find_all_lsp_binaries, find_lsp_binary, DetectionSource, LspDetectionResult,
};
