//! Unit tests for the TUI project browser view.
//!
//! Extracted into a separate file to keep `projects.rs` under the 500-line
//! limit.

use ratatui::style::Color;

use super::super::projects_data::ProjectRow;
use super::*;

#[test]
fn project_browser_new_starts_empty() {
    let browser = ProjectBrowser::new();
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
    assert!(browser.detail.is_none());
    assert!(browser.last_refresh.is_none());
}

#[test]
fn select_clamps_at_boundaries() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(5);
    b.selected = 4;
    b.select_next();
    assert_eq!(b.selected, 4);
    b.selected = 0;
    b.select_prev();
    assert_eq!(b.selected, 0);
}

#[test]
fn select_advances_and_retreats() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(5);
    b.selected = 2;
    b.select_next();
    assert_eq!(b.selected, 3);
    b.select_prev();
    assert_eq!(b.selected, 2);
}

#[test]
fn page_navigation_clamps() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(50);
    b.selected = 5;
    b.page_up(20);
    assert_eq!(b.selected, 0);
    b.selected = 45;
    b.page_down(20);
    assert_eq!(b.selected, 49);
}

#[test]
fn close_detail_clears() {
    let mut b = ProjectBrowser::new();
    b.detail = Some(make_test_detail());
    assert!(b.detail_open());
    b.close_detail();
    assert!(!b.detail_open());
}

#[test]
fn select_on_empty_list() {
    let mut b = ProjectBrowser::new();
    b.select_next();
    b.select_prev();
    assert_eq!(b.selected, 0);
}

#[test]
fn status_color_mapping() {
    assert_eq!(status_color("done"), Color::Green);
    assert_eq!(status_color("pending"), Color::Yellow);
    assert_eq!(status_color("in_progress"), Color::Blue);
    assert_eq!(status_color("failed"), Color::Red);
    assert_eq!(status_color("unknown"), Color::Reset);
}

#[test]
fn truncate_str_behavior() {
    assert_eq!(truncate_str("hello", 10), "hello");
    let long = "a".repeat(40);
    let result = truncate_str(&long, 10);
    assert!(result.ends_with("..."));
    assert!(result.chars().count() <= 10);
}

#[test]
fn request_toggle_targets_inverted_enabled() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 0; // item 0 is enabled (i % 2 == 0)
    assert!(!b.confirm_open());
    b.request_toggle();
    assert!(b.confirm_open());
    // Confirming yields the watch_id and the *target* state (disable).
    let (wid, enable) = b.take_confirm().unwrap();
    assert_eq!(wid, "watch-0");
    assert!(!enable);
    assert!(!b.confirm_open());
}

#[test]
fn cancel_confirm_clears_modal() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(2);
    b.request_toggle();
    assert!(b.confirm_open());
    b.cancel_confirm();
    assert!(!b.confirm_open());
    assert!(b.take_confirm().is_none());
}

#[test]
fn force_refresh_clears_last_refresh() {
    let mut b = ProjectBrowser::new();
    b.last_refresh = Some(Instant::now());
    b.force_refresh();
    assert!(b.last_refresh.is_none());
}

// ── Nudge confirm state machine ───────────────────────────────────────────────

#[test]
fn nudge_confirm_not_open_by_default() {
    let b = ProjectBrowser::new();
    assert!(!b.nudge_confirm_open());
}

#[test]
fn request_nudge_on_empty_list_is_noop() {
    let mut b = ProjectBrowser::new();
    b.request_nudge();
    assert!(!b.nudge_confirm_open());
}

#[test]
fn request_nudge_opens_confirm() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 1;
    b.request_nudge();
    assert!(b.nudge_confirm_open());
    assert!(b.confirm_open());
    let ac = b.nudge_action_confirm().unwrap();
    let ActionConfirm::Simple(ref sc) = ac;
    assert_eq!(sc.verb, "Rescan");
    assert_eq!(sc.target, "project-1");
}

#[test]
fn take_nudge_returns_tenant_id_and_clears_modal() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 2;
    b.request_nudge();
    let tenant = b.take_nudge().unwrap();
    assert_eq!(tenant, "watch-2"); // watch_id used as tenant_id for projects
    assert!(!b.nudge_confirm_open());
}

#[test]
fn cancel_nudge_clears_modal() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(2);
    b.request_nudge();
    assert!(b.nudge_confirm_open());
    b.cancel_nudge();
    assert!(!b.nudge_confirm_open());
    assert!(b.take_nudge().is_none());
}

#[test]
fn nudge_confirm_is_separate_from_toggle_confirm() {
    // Both confirms can be tracked independently via confirm_open.
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(2);
    b.request_toggle();
    assert!(b.confirm_open());
    assert!(!b.nudge_confirm_open());
    b.cancel_confirm();
    b.request_nudge();
    assert!(b.nudge_confirm_open());
    assert!(b.confirm_open()); // confirm_open covers nudge too
}

fn make_test_rows(n: usize) -> Vec<ProjectRow> {
    (0..n)
        .map(|i| ProjectRow {
            watch_id: format!("watch-{i}"),
            name: format!("project-{i}"),
            display_path: format!("~/dev/project-{i}"),
            is_active: i % 2 == 0,
            enabled: i % 2 == 0,
            doc_count: (i * 10) as i64,
            queue_count: (i % 3) as i64,
            branch: "main".to_string(),
        })
        .collect()
}

fn make_test_detail() -> ProjectDetail {
    ProjectDetail {
        watch_id: "w1".into(),
        tenant_id: "t1".into(),
        name: "test-proj".into(),
        display_path: "~/test-proj".into(),
        collection: "projects".into(),
        is_active: true,
        is_paused: false,
        is_archived: false,
        git_remote_url: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T12:00:00Z".into(),
        last_scan: None,
        sub_watches: Vec::new(),
        queue_by_status: std::collections::HashMap::new(),
    }
}

/// A periodic refresh that rebuilds/reorders `items` after the nudge modal
/// opened must not retarget the rescan: the watch_id was captured at request
/// time.
#[test]
fn take_nudge_ignores_items_rebuild_after_open() {
    let mut b = ProjectBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 2;
    b.request_nudge();
    // Simulate a refresh replacing the list with different rows.
    b.items = make_test_rows(1);
    b.selected = 0;
    assert_eq!(b.take_nudge().unwrap(), "watch-2");
}
