//! LanguageServiceImpl struct definition and constructor.

use std::sync::Arc;

use tokio::sync::Mutex;
use workspace_qdrant_core::tree_sitter::GrammarManager;

/// LanguageService implementation backed by a shared `GrammarManager`.
///
/// The manager owns the on-disk grammar cache, downloader, and tree-sitter
/// loader. Install/remove operations need `&mut`, so the manager is wrapped in a
/// `tokio::sync::Mutex`; the grammar engine stays daemon-side (clients reach it
/// only via this RPC surface, never by linking `workspace-qdrant-core`).
pub struct LanguageServiceImpl {
    pub(crate) grammar_manager: Arc<Mutex<GrammarManager>>,
}

impl LanguageServiceImpl {
    /// Create a new LanguageService over a shared grammar manager.
    pub fn new(grammar_manager: Arc<Mutex<GrammarManager>>) -> Self {
        Self { grammar_manager }
    }
}
