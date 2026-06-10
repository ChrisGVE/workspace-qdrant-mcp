//! Unit tests for the TUI library browser view.
//!
//! Extracted into a separate file to keep `libraries.rs` under the 500-line
//! limit.

use super::*;

#[test]
fn library_browser_new_starts_empty() {
    let browser = LibraryBrowser::new();
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
    assert!(browser.detail.is_none());
    assert!(browser.last_refresh.is_none());
}

#[test]
fn select_next_clamps_to_bounds() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 4;
    browser.select_next();
    assert_eq!(browser.selected, 4);
}

#[test]
fn select_prev_clamps_to_zero() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 0;
    browser.select_prev();
    assert_eq!(browser.selected, 0);
}

#[test]
fn select_next_advances() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 2;
    browser.select_next();
    assert_eq!(browser.selected, 3);
}

#[test]
fn select_prev_retreats() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 3;
    browser.select_prev();
    assert_eq!(browser.selected, 2);
}

#[test]
fn page_up_clamps() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(50);
    browser.selected = 5;
    browser.page_up(20);
    assert_eq!(browser.selected, 0);
}

#[test]
fn page_down_clamps() {
    let mut browser = LibraryBrowser::new();
    browser.items = make_test_rows(50);
    browser.selected = 45;
    browser.page_down(20);
    assert_eq!(browser.selected, 49);
}

#[test]
fn close_detail_clears() {
    let mut browser = LibraryBrowser::new();
    browser.detail = Some(LibraryDetail {
        watch_id: "lib-test".into(),
        tag: "test".into(),
        display_path: "/tmp/lib".into(),
        enabled: true,
        is_active: false,
        mode: "sync".into(),
        doc_count: 5,
        follow_symlinks: false,
        cleanup_on_disable: false,
        is_paused: false,
        is_archived: false,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        last_scan: None,
        last_activity_at: None,
    });
    assert!(browser.detail_open());
    browser.close_detail();
    assert!(!browser.detail_open());
}

#[test]
fn select_on_empty_list() {
    let mut browser = LibraryBrowser::new();
    browser.select_next();
    assert_eq!(browser.selected, 0);
    browser.select_prev();
    assert_eq!(browser.selected, 0);
}

#[test]
fn request_toggle_skips_project_derived() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(2);
    b.items[0].source = Some("P:proj".into());
    b.selected = 0;
    b.request_toggle();
    // Project-derived library is not toggleable here: no modal, message set.
    assert!(!b.confirm_open());
    assert!(b.message.is_some());
}

#[test]
fn request_toggle_opens_for_top_level_library() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(2);
    b.items[0].source = None;
    b.items[0].enabled = true;
    b.selected = 0;
    b.request_toggle();
    assert!(b.confirm_open());
    let (wid, enable) = b.take_confirm().unwrap();
    assert_eq!(wid, "lib-tag-0");
    assert!(!enable); // toggles to disabled
}

fn make_test_rows(n: usize) -> Vec<LibraryRow> {
    (0..n)
        .map(|i| LibraryRow {
            watch_id: format!("lib-tag-{i}"),
            tag: format!("tag-{i}"),
            name: format!("lib-{i}"),
            display_path: format!("/tmp/lib-{i}"),
            enabled: true,
            is_active: i % 2 == 0,
            mode: "sync".into(),
            doc_count: i as u64 * 10,
            source: None,
        })
        .collect()
}
