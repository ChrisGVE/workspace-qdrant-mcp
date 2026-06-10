//! Unit tests for `search_page.rs`.
//!
//! Extracted via `#[cfg(test)] #[path = "search_page_tests.rs"] mod tests;`
//! to keep `search_page.rs` under the 500-line limit.

use super::super::search_data::{GraphRelatedNode, SearchMatch, SearchSnapshot, TenantRef};
use super::*;

// ── SearchMode ────────────────────────────────────────────────────────

#[test]
fn search_mode_labels_are_non_empty() {
    for m in SearchMode::ALL {
        assert!(!m.label().is_empty(), "{:?} has empty label", m);
    }
}

#[test]
fn search_mode_numbers_are_one_to_three() {
    for (i, m) in SearchMode::ALL.iter().enumerate() {
        assert_eq!(m.number(), i + 1);
    }
}

#[test]
fn search_mode_from_key_round_trips() {
    assert_eq!(SearchMode::from_key(1), Some(SearchMode::Grep));
    assert_eq!(SearchMode::from_key(2), Some(SearchMode::Exact));
    assert_eq!(SearchMode::from_key(3), Some(SearchMode::Graph));
}

#[test]
fn search_mode_from_key_out_of_range() {
    assert_eq!(SearchMode::from_key(0), None);
    assert_eq!(SearchMode::from_key(4), None);
    assert_eq!(SearchMode::from_key(255), None);
}

// ── QueryPrompt ───────────────────────────────────────────────────────

#[test]
fn prompt_activate_sets_active() {
    let mut p = QueryPrompt::default();
    p.activate();
    assert!(p.active);
}

#[test]
fn prompt_cancel_clears_query() {
    let mut p = QueryPrompt::default();
    p.activate();
    p.push_char('x');
    p.cancel();
    assert!(!p.active);
    assert!(p.query.is_empty());
}

#[test]
fn prompt_confirm_returns_trimmed_query() {
    let mut p = QueryPrompt::default();
    p.activate();
    for c in "  fn main  ".chars() {
        p.push_char(c);
    }
    let result = p.confirm();
    assert_eq!(result, Some("fn main".to_string()));
    assert!(!p.active);
    assert!(p.query.is_empty());
    assert_eq!(p.last_submitted, "fn main");
}

#[test]
fn prompt_confirm_empty_returns_none() {
    let mut p = QueryPrompt::default();
    p.activate();
    let result = p.confirm();
    assert_eq!(result, None);
    assert!(!p.active);
}

#[test]
fn prompt_backspace_removes_last_char() {
    let mut p = QueryPrompt::default();
    p.push_char('a');
    p.push_char('b');
    p.backspace();
    assert_eq!(p.query, "a");
}

#[test]
fn prompt_has_query_after_submit() {
    let mut p = QueryPrompt::default();
    p.activate();
    p.push_char('x');
    p.confirm();
    assert!(p.has_query());
}

#[test]
fn prompt_has_no_query_initially() {
    let p = QueryPrompt::default();
    assert!(!p.has_query());
}

// ── SearchPageView navigation ─────────────────────────────────────────

fn make_view_with_tenants(n: usize) -> SearchPageView {
    let mut v = SearchPageView::new();
    v.tenants = (0..n)
        .map(|i| TenantRef {
            tenant_id: format!("t{i}"),
            name: format!("project-{i}"),
        })
        .collect();
    v
}

#[test]
fn set_mode_resets_cursor_and_preview() {
    let mut v = SearchPageView::new();
    v.selected = 5;
    v.preview_open = true;
    v.set_mode(SearchMode::Exact);
    assert_eq!(v.mode, SearchMode::Exact);
    assert_eq!(v.selected, 0);
    assert!(!v.preview_open);
}

#[test]
fn set_mode_same_mode_keeps_cursor() {
    let mut v = SearchPageView::new();
    v.mode = SearchMode::Grep;
    v.selected = 3;
    v.set_mode(SearchMode::Grep); // no-op
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
    v.next_tenant(); // must not panic
    v.prev_tenant(); // must not panic
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
    let mut v = SearchPageView::new();
    v.selected = 0;
    v.select_prev();
    assert_eq!(v.selected, 0);
}

#[test]
fn jump_first_resets_selection() {
    let mut v = SearchPageView::new();
    v.selected = 7;
    v.jump_first();
    assert_eq!(v.selected, 0);
}

#[test]
fn open_preview_no_op_when_empty() {
    let mut v = SearchPageView::new();
    v.open_preview();
    assert!(!v.preview_open);
}

#[test]
fn close_preview_clears_flag() {
    let mut v = SearchPageView::new();
    v.preview_open = true;
    v.close_preview();
    assert!(!v.preview_open);
}

#[test]
fn results_len_grep_counts_matches() {
    let mut v = SearchPageView::new();
    v.mode = SearchMode::Grep;
    let mut snap = SearchSnapshot::default();
    snap.matches = vec![
        SearchMatch {
            file_path: "a.rs".into(),
            line_number: 1,
            content: "x".into(),
            context_before: vec![],
            context_after: vec![],
        },
        SearchMatch {
            file_path: "b.rs".into(),
            line_number: 2,
            content: "y".into(),
            context_before: vec![],
            context_after: vec![],
        },
    ];
    assert_eq!(v.results_len(&snap), 2);
}

#[test]
fn results_len_graph_counts_nodes() {
    let mut v = SearchPageView::new();
    v.mode = SearchMode::Graph;
    let mut snap = SearchSnapshot::default();
    snap.graph_nodes = vec![GraphRelatedNode {
        symbol_name: "foo".into(),
        symbol_type: "fn".into(),
        file_path: "src/foo.rs".into(),
        edge_type: "CALLS".into(),
        depth: 1,
    }];
    assert_eq!(v.results_len(&snap), 1);
}
