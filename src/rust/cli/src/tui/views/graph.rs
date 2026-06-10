//! Graph view — code-relationship graph explorer for the TUI.
//!
//! Located at: `src/rust/cli/src/tui/views/graph.rs`
//!
//! # Architecture context
//! This view is tab 9 in the TUI tab bar, surfacing the daemon's GraphService
//! via five display modes selectable with keys 1-5:
//!
//!   1 Stats       — node/edge type breakdown for the active tenant
//!   2 PageRank    — top-50 nodes ranked by PageRank score
//!   3 Communities — detected code clusters with member lists
//!   4 Betweenness — top-50 nodes ranked by betweenness centrality
//!   5 Impact      — callers/callees/affected files for a queried symbol
//!
//! The TUI loop is synchronous; all gRPC work runs on a background thread
//! writing into an `Arc<Mutex<GraphSnapshot>>` (see `graph_data.rs`). This
//! file contains state management and navigation logic.
//! Rendering lives in `graph_render.rs` and `graph_render_detail.rs`.
//!
//! Neighbors: `graph_data.rs` (data model + background fetcher),
//! `graph_render.rs` (primary render methods),
//! `graph_render_detail.rs` (community popup + impact + prompt render),
//! `app.rs` (tab integration), `app/key_handler.rs` (key routing),
//! `app/render.rs` (draw dispatch).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::graph_data::{
    load_tenant_list, spawn_graph_fetcher, FetchRequest, GraphSnapshot, TenantRef,
    REFRESH_INTERVAL_SECS,
};

// ─── GraphMode ───────────────────────────────────────────────────────────────

/// The five display sub-modes within the Graph page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    Stats,
    PageRank,
    Communities,
    Betweenness,
    Impact,
}

impl GraphMode {
    /// All modes in key order (1–5).
    pub const ALL: [GraphMode; 5] = [
        GraphMode::Stats,
        GraphMode::PageRank,
        GraphMode::Communities,
        GraphMode::Betweenness,
        GraphMode::Impact,
    ];

    /// Short label shown in the mode bar.
    pub fn label(self) -> &'static str {
        match self {
            GraphMode::Stats => "Stats",
            GraphMode::PageRank => "PageRank",
            GraphMode::Communities => "Communities",
            GraphMode::Betweenness => "Betweenness",
            GraphMode::Impact => "Impact",
        }
    }

    /// One-based index (matches key 1–5).
    pub fn number(self) -> usize {
        GraphMode::ALL.iter().position(|m| *m == self).unwrap_or(0) + 1
    }

    /// Construct from a 1-based key digit (1–5); returns `None` for out-of-range.
    pub fn from_key(key: u8) -> Option<Self> {
        if key == 0 {
            return None;
        }
        Self::ALL.get((key - 1) as usize).copied()
    }
}

// ─── Impact input state ───────────────────────────────────────────────────────

/// Minimal text-input state for the Impact symbol prompt.
///
/// Activated by `i`; confirmed by Enter (triggering an RPC); cancelled by Esc.
#[derive(Debug, Clone, Default)]
pub struct ImpactPrompt {
    /// Whether the prompt is currently capturing keystrokes.
    pub active: bool,
    /// Text the user is typing.
    pub query: String,
    /// The last symbol that was actually submitted for an RPC.
    pub last_submitted: String,
}

impl ImpactPrompt {
    pub fn activate(&mut self) {
        self.active = true;
        self.query.clear();
    }

    pub fn cancel(&mut self) {
        self.active = false;
        self.query.clear();
    }

    /// Feed a character to the prompt. Returns `false` always (caller triggers
    /// RPC on Enter via `confirm`, not via `push_char`).
    pub fn push_char(&mut self, c: char) -> bool {
        self.query.push(c);
        false
    }

    pub fn backspace(&mut self) {
        self.query.pop();
    }

    pub fn confirm(&mut self) -> Option<String> {
        if self.query.trim().is_empty() {
            self.active = false;
            return None;
        }
        let sym = self.query.trim().to_string();
        self.last_submitted = sym.clone();
        self.active = false;
        self.query.clear();
        Some(sym)
    }
}

// ─── GraphView ────────────────────────────────────────────────────────────────

/// State for the Graph TUI page.
pub struct GraphView {
    /// Currently active display mode.
    pub mode: GraphMode,
    /// Index of the currently selected tenant in `snapshot.tenants`.
    pub tenant_idx: usize,
    /// Cursor position in the active list mode.
    pub selected: usize,
    /// Expanded community index (for showing member details), if any.
    pub expanded_community: Option<usize>,
    /// Impact symbol text-input prompt.
    pub impact_prompt: ImpactPrompt,
    /// Shared data snapshot updated by the background fetcher.
    pub snapshot: Arc<Mutex<GraphSnapshot>>,
    /// Channel to send fetch requests to the background thread.
    fetch_tx: std::sync::mpsc::Sender<FetchRequest>,
    /// When we last triggered a reload (for auto-refresh throttle).
    last_reload: Option<Instant>,
    /// Tenant list cached locally to avoid locking the snapshot on every draw.
    pub tenants: Vec<TenantRef>,
}

impl GraphView {
    /// Create a new Graph view: spawns the background fetcher and loads tenants.
    pub fn new() -> Self {
        let (snapshot, fetch_tx) = spawn_graph_fetcher();
        let tenants = load_tenant_list();
        Self {
            mode: GraphMode::Stats,
            tenant_idx: 0,
            selected: 0,
            expanded_community: None,
            impact_prompt: ImpactPrompt::default(),
            snapshot,
            fetch_tx,
            last_reload: None,
            tenants,
        }
    }

    // ─── Tick (called every 250 ms by the app loop) ───────────────────────

    /// Periodic update: auto-reload stats when the interval has elapsed and
    /// keep the tenant list in sync with registered projects.
    pub fn on_tick(&mut self) {
        // Refresh the tenant list on every tick (cheap SQLite read; ≤200 rows).
        self.tenants = load_tenant_list();
        // Clamp the active tenant index after a possible list shrink.
        if !self.tenants.is_empty() {
            self.tenant_idx = self.tenant_idx.min(self.tenants.len() - 1);
        }

        // Auto-reload once the refresh interval elapses.
        let should_reload = self
            .last_reload
            .map_or(true, |t| t.elapsed().as_secs() >= REFRESH_INTERVAL_SECS);

        if should_reload {
            self.trigger_reload();
        }
    }

    /// Send a Reload request to the background thread for the active tenant.
    pub fn trigger_reload(&mut self) {
        if let Some(tenant) = self.active_tenant() {
            let _ = self.fetch_tx.send(FetchRequest::Reload {
                tenant_id: tenant.tenant_id.clone(),
            });
            self.last_reload = Some(Instant::now());
        }
    }

    /// Send an Impact request for the given symbol to the background thread.
    pub fn trigger_impact(&mut self, symbol: String) {
        if let Some(tenant) = self.active_tenant() {
            let _ = self.fetch_tx.send(FetchRequest::Impact {
                tenant_id: tenant.tenant_id.clone(),
                symbol,
            });
        }
    }

    // ─── Mode / tenant navigation ─────────────────────────────────────────

    /// Switch to the given mode and reset the cursor.
    pub fn set_mode(&mut self, mode: GraphMode) {
        if self.mode != mode {
            self.mode = mode;
            self.selected = 0;
            self.expanded_community = None;
        }
    }

    /// Cycle to the next registered tenant (wraps around).
    pub fn next_tenant(&mut self) {
        if self.tenants.len() > 1 {
            self.tenant_idx = (self.tenant_idx + 1) % self.tenants.len();
            self.selected = 0;
            self.trigger_reload();
        }
    }

    /// Cycle to the previous registered tenant (wraps around).
    pub fn prev_tenant(&mut self) {
        if self.tenants.len() > 1 {
            let len = self.tenants.len();
            self.tenant_idx = (self.tenant_idx + len - 1) % len;
            self.selected = 0;
            self.trigger_reload();
        }
    }

    /// The currently active tenant, or `None` when no tenants are registered.
    pub fn active_tenant(&self) -> Option<&TenantRef> {
        self.tenants.get(self.tenant_idx)
    }

    // ─── List navigation ──────────────────────────────────────────────────

    /// Length of the active list for the current mode.
    pub fn list_len(&self, snap: &GraphSnapshot) -> usize {
        match self.mode {
            GraphMode::Stats => 0, // not a scrollable list
            GraphMode::PageRank => snap.pagerank.len(),
            GraphMode::Communities => snap.communities.len(),
            GraphMode::Betweenness => snap.betweenness.len(),
            GraphMode::Impact => snap
                .impact
                .as_ref()
                .map_or(0, |r| r.direct.len() + r.transitive.len()),
        }
    }

    pub fn select_next(&mut self) {
        let snap = self.read_snapshot();
        let len = self.list_len(&snap);
        if len > 0 {
            self.selected = (self.selected + 1).min(len - 1);
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn page_down(&mut self, step: usize) {
        let snap = self.read_snapshot();
        let len = self.list_len(&snap);
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
        let len = self.list_len(&snap);
        if len > 0 {
            self.selected = len - 1;
        }
    }

    /// Toggle expansion of the selected community row (Communities mode only).
    pub fn toggle_community_expand(&mut self) {
        if self.mode == GraphMode::Communities {
            if self.expanded_community == Some(self.selected) {
                self.expanded_community = None;
            } else {
                self.expanded_community = Some(self.selected);
            }
        }
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Read a cloned snapshot without holding the lock across draws.
    pub fn read_snapshot(&self) -> GraphSnapshot {
        self.snapshot.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph_data::TenantRef;
    use super::*;

    // ── GraphMode ──────────────────────────────────────────────────────────

    #[test]
    fn graph_mode_labels_are_non_empty() {
        for m in GraphMode::ALL {
            assert!(!m.label().is_empty(), "{:?} has empty label", m);
        }
    }

    #[test]
    fn graph_mode_numbers_are_one_to_five() {
        for (i, m) in GraphMode::ALL.iter().enumerate() {
            assert_eq!(m.number(), i + 1);
        }
    }

    #[test]
    fn graph_mode_from_key_round_trips() {
        assert_eq!(GraphMode::from_key(1), Some(GraphMode::Stats));
        assert_eq!(GraphMode::from_key(2), Some(GraphMode::PageRank));
        assert_eq!(GraphMode::from_key(3), Some(GraphMode::Communities));
        assert_eq!(GraphMode::from_key(4), Some(GraphMode::Betweenness));
        assert_eq!(GraphMode::from_key(5), Some(GraphMode::Impact));
    }

    #[test]
    fn graph_mode_from_key_out_of_range() {
        assert_eq!(GraphMode::from_key(0), None);
        assert_eq!(GraphMode::from_key(6), None);
        assert_eq!(GraphMode::from_key(255), None);
    }

    // ── GraphView navigation ──────────────────────────────────────────────

    fn make_view_with_tenants(n: usize) -> GraphView {
        let mut v = GraphView::new();
        v.tenants = (0..n)
            .map(|i| TenantRef {
                tenant_id: format!("t{i}"),
                name: format!("project-{i}"),
            })
            .collect();
        v
    }

    #[test]
    fn set_mode_resets_cursor() {
        let mut v = GraphView::new();
        v.selected = 7;
        v.set_mode(GraphMode::PageRank);
        assert_eq!(v.mode, GraphMode::PageRank);
        assert_eq!(v.selected, 0);
    }

    #[test]
    fn set_mode_same_mode_keeps_cursor() {
        let mut v = GraphView::new();
        v.mode = GraphMode::PageRank;
        v.selected = 3;
        v.set_mode(GraphMode::PageRank); // no-op
        assert_eq!(v.selected, 3);
    }

    #[test]
    fn next_tenant_wraps_around() {
        let mut v = make_view_with_tenants(3);
        v.tenant_idx = 2;
        v.next_tenant();
        assert_eq!(v.tenant_idx, 0);
    }

    #[test]
    fn prev_tenant_wraps_around() {
        let mut v = make_view_with_tenants(3);
        v.tenant_idx = 0;
        v.prev_tenant();
        assert_eq!(v.tenant_idx, 2);
    }

    #[test]
    fn next_tenant_no_op_with_single_tenant() {
        let mut v = make_view_with_tenants(1);
        v.tenant_idx = 0;
        v.next_tenant();
        assert_eq!(v.tenant_idx, 0);
    }

    #[test]
    fn tenant_cycler_no_op_with_no_tenants() {
        let mut v = make_view_with_tenants(0);
        v.next_tenant(); // should not panic
        v.prev_tenant(); // should not panic
        assert_eq!(v.tenant_idx, 0);
    }

    #[test]
    fn active_tenant_returns_none_with_no_tenants() {
        let v = make_view_with_tenants(0);
        assert!(v.active_tenant().is_none());
    }

    #[test]
    fn active_tenant_returns_correct_entry() {
        let v = make_view_with_tenants(3);
        assert_eq!(v.active_tenant().unwrap().tenant_id, "t0");
    }

    #[test]
    fn select_prev_clamps_to_zero() {
        let mut v = GraphView::new();
        v.selected = 0;
        v.select_prev();
        assert_eq!(v.selected, 0);
    }

    #[test]
    fn jump_first_resets_selection() {
        let mut v = GraphView::new();
        v.selected = 5;
        v.jump_first();
        assert_eq!(v.selected, 0);
    }

    #[test]
    fn toggle_community_expand_sets_and_clears() {
        let mut v = GraphView::new();
        v.mode = GraphMode::Communities;
        v.selected = 2;
        v.toggle_community_expand();
        assert_eq!(v.expanded_community, Some(2));
        v.toggle_community_expand();
        assert_eq!(v.expanded_community, None);
    }

    #[test]
    fn toggle_community_expand_noop_outside_communities_mode() {
        let mut v = GraphView::new();
        v.mode = GraphMode::PageRank;
        v.selected = 1;
        v.toggle_community_expand();
        assert_eq!(v.expanded_community, None);
    }
}
