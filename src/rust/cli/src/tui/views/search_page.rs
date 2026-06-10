//! Search page — multi-mode search explorer for the TUI.
//!
//! Located at: `src/rust/cli/src/tui/views/search_page.rs`
//!
//! # Architecture context
//! This view is tab 10 in the TUI tab bar, surfacing both the daemon's
//! TextSearchService (Grep and Semantic modes) and GraphService (Graph mode)
//! through three display modes selectable with keys 1-3:
//!
//!   1 Grep     — literal/regex line search via TextSearchService::Search
//!   2 Semantic — hybrid dense+sparse search via TextSearchService::Search
//!   3 Graph    — related symbols via GraphService::QueryRelated
//!
//! The TUI loop is synchronous; all gRPC work runs on a background thread
//! writing into an `Arc<Mutex<SearchSnapshot>>` (see `search_data.rs`). This
//! file contains state management and navigation logic.
//! Rendering lives in `search_render.rs` and `search_render_detail.rs`.
//!
//! Neighbors: `search_data.rs` (data model + background fetcher),
//! `search_render.rs` (primary render methods),
//! `search_render_detail.rs` (preview popup + query prompt render),
//! `app.rs` (tab integration), `app/key_handler.rs` (key routing),
//! `app/render.rs` (draw dispatch).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::search_data::{
    load_tenant_list, spawn_search_fetcher, FetchRequest, SearchSnapshot, TenantRef,
};

// ─── SearchMode ───────────────────────────────────────────────────────────────

/// The three display sub-modes within the Search page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Grep,
    Semantic,
    Graph,
}

impl SearchMode {
    /// All modes in key order (1–3).
    pub const ALL: [SearchMode; 3] = [SearchMode::Grep, SearchMode::Semantic, SearchMode::Graph];

    /// Short label shown in the mode bar.
    pub fn label(self) -> &'static str {
        match self {
            SearchMode::Grep => "Grep",
            SearchMode::Semantic => "Semantic",
            SearchMode::Graph => "Graph",
        }
    }

    /// One-based index (matches key 1–3).
    pub fn number(self) -> usize {
        SearchMode::ALL.iter().position(|m| *m == self).unwrap_or(0) + 1
    }

    /// Construct from a 1-based key digit (1–3); returns `None` for out-of-range.
    pub fn from_key(key: u8) -> Option<Self> {
        if key == 0 {
            return None;
        }
        Self::ALL.get((key - 1) as usize).copied()
    }
}

// ─── Query input state ────────────────────────────────────────────────────────

/// Minimal text-input state for the search query prompt.
///
/// Activated by `i` or `/`; confirmed by Enter (triggers a gRPC fetch);
/// cancelled by Esc.
#[derive(Debug, Clone, Default)]
pub struct QueryPrompt {
    /// Whether the prompt is currently capturing keystrokes.
    pub active: bool,
    /// Text the user is typing.
    pub query: String,
    /// The last query that was actually submitted to the background thread.
    pub last_submitted: String,
}

impl QueryPrompt {
    /// Open the prompt, keeping any text from the previous session.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Discard the in-progress text and close the prompt.
    pub fn cancel(&mut self) {
        self.active = false;
        self.query.clear();
    }

    /// Append one character to the query text.
    pub fn push_char(&mut self, c: char) {
        self.query.push(c);
    }

    /// Remove the last character from the query text.
    pub fn backspace(&mut self) {
        self.query.pop();
    }

    /// Confirm the current query; returns `Some(query)` when non-blank, or
    /// `None` when the prompt is empty (which just closes it silently).
    pub fn confirm(&mut self) -> Option<String> {
        let trimmed = self.query.trim().to_string();
        self.active = false;
        if trimmed.is_empty() {
            self.query.clear();
            return None;
        }
        self.last_submitted = trimmed.clone();
        self.query.clear();
        Some(trimmed)
    }

    /// Whether there is a last-submitted (non-empty) query to display.
    pub fn has_query(&self) -> bool {
        !self.last_submitted.is_empty()
    }
}

// ─── SearchPageView ───────────────────────────────────────────────────────────

/// State for the Search TUI page.
pub struct SearchPageView {
    /// Currently active display mode.
    pub mode: SearchMode,
    /// Index of the currently selected tenant in `tenants`.
    pub tenant_idx: usize,
    /// Cursor position in the active results list.
    pub selected: usize,
    /// Whether the preview popup for the selected result is open.
    pub preview_open: bool,
    /// Search query input prompt.
    pub prompt: QueryPrompt,
    /// Shared data snapshot updated by the background fetcher.
    pub snapshot: Arc<Mutex<SearchSnapshot>>,
    /// Channel to send fetch requests to the background thread.
    fetch_tx: std::sync::mpsc::Sender<FetchRequest>,
    /// When we last triggered a fetch (informational; not used for throttle here).
    last_fetch: Option<Instant>,
    /// Tenant list cached locally to avoid locking the snapshot on every draw.
    pub tenants: Vec<TenantRef>,
}

impl SearchPageView {
    /// Create a new Search page view: spawns the background fetcher and loads tenants.
    pub fn new() -> Self {
        let (snapshot, fetch_tx) = spawn_search_fetcher();
        let tenants = load_tenant_list();
        Self {
            mode: SearchMode::Grep,
            tenant_idx: 0,
            selected: 0,
            preview_open: false,
            prompt: QueryPrompt::default(),
            snapshot,
            fetch_tx,
            last_fetch: None,
            tenants,
        }
    }

    // ─── Tick (called every 250 ms by the app loop) ───────────────────────

    /// Periodic update: keep the tenant list in sync with registered projects.
    pub fn on_tick(&mut self) {
        self.tenants = load_tenant_list();
        if !self.tenants.is_empty() {
            self.tenant_idx = self.tenant_idx.min(self.tenants.len() - 1);
        }
    }

    // ─── Query dispatch ───────────────────────────────────────────────────

    /// Send a text-search (Grep or Semantic) fetch request to the background thread.
    ///
    /// Grep mode uses a literal or regex pattern; Semantic mode sends the same
    /// `TextSearchRequest` with `regex = false` (the daemon handles hybrid
    /// dense+sparse ranking transparently for non-regex queries).
    pub fn trigger_text_search(&mut self, query: String) {
        if let Some(tenant) = self.active_tenant() {
            let regex = self.mode == SearchMode::Grep;
            let _ = self.fetch_tx.send(FetchRequest::TextSearch {
                tenant_id: tenant.tenant_id.clone(),
                pattern: query,
                regex,
                case_sensitive: false,
            });
            self.last_fetch = Some(Instant::now());
        }
    }

    /// Send a graph-related-nodes fetch request to the background thread.
    pub fn trigger_graph_search(&mut self, node_id: String) {
        if let Some(tenant) = self.active_tenant() {
            let _ = self.fetch_tx.send(FetchRequest::GraphQuery {
                tenant_id: tenant.tenant_id.clone(),
                node_id,
            });
            self.last_fetch = Some(Instant::now());
        }
    }

    /// Dispatch the confirmed query to the appropriate fetcher for the active mode.
    pub fn dispatch_query(&mut self, query: String) {
        match self.mode {
            SearchMode::Grep | SearchMode::Semantic => self.trigger_text_search(query),
            SearchMode::Graph => self.trigger_graph_search(query),
        }
    }

    // ─── Mode / tenant navigation ─────────────────────────────────────────

    /// Switch to the given mode and reset the cursor.
    pub fn set_mode(&mut self, mode: SearchMode) {
        if self.mode != mode {
            self.mode = mode;
            self.selected = 0;
            self.preview_open = false;
        }
    }

    /// Cycle to the next registered tenant (wraps around).
    pub fn next_tenant(&mut self) {
        if self.tenants.len() > 1 {
            self.tenant_idx = (self.tenant_idx + 1) % self.tenants.len();
            self.selected = 0;
        }
    }

    /// Cycle to the previous registered tenant (wraps around).
    pub fn prev_tenant(&mut self) {
        if self.tenants.len() > 1 {
            let len = self.tenants.len();
            self.tenant_idx = (self.tenant_idx + len - 1) % len;
            self.selected = 0;
        }
    }

    /// The currently active tenant, or `None` when no tenants are registered.
    pub fn active_tenant(&self) -> Option<&TenantRef> {
        self.tenants.get(self.tenant_idx)
    }

    // ─── List navigation ──────────────────────────────────────────────────

    /// Length of the results list in the current snapshot.
    pub fn results_len(&self, snap: &SearchSnapshot) -> usize {
        match self.mode {
            SearchMode::Grep | SearchMode::Semantic => snap.matches.len(),
            SearchMode::Graph => snap.graph_nodes.len(),
        }
    }

    pub fn select_next(&mut self) {
        let snap = self.read_snapshot();
        let len = self.results_len(&snap);
        if len > 0 {
            self.selected = (self.selected + 1).min(len - 1);
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn page_down(&mut self, step: usize) {
        let snap = self.read_snapshot();
        let len = self.results_len(&snap);
        if len > 0 {
            self.selected = (self.selected + step).min(len - 1);
        }
    }

    pub fn page_up(&mut self, step: usize) {
        self.selected = self.selected.saturating_sub(step);
    }

    pub fn jump_first(&mut self) {
        self.selected = 0;
    }

    pub fn jump_last(&mut self) {
        let snap = self.read_snapshot();
        let len = self.results_len(&snap);
        if len > 0 {
            self.selected = len - 1;
        }
    }

    /// Open the file-content preview for the selected result.
    pub fn open_preview(&mut self) {
        let snap = self.read_snapshot();
        if self.results_len(&snap) > 0 {
            self.preview_open = true;
        }
    }

    /// Close the preview popup.
    pub fn close_preview(&mut self) {
        self.preview_open = false;
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Read a cloned snapshot without holding the lock across draws.
    pub fn read_snapshot(&self) -> SearchSnapshot {
        self.snapshot.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "search_page_tests.rs"]
mod tests;
