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

// ── Nudge confirm state machine ───────────────────────────────────────────────

#[test]
fn nudge_confirm_not_open_by_default() {
    let b = LibraryBrowser::new();
    assert!(!b.nudge_confirm_open());
}

#[test]
fn request_nudge_on_empty_list_is_noop() {
    let mut b = LibraryBrowser::new();
    b.request_nudge();
    assert!(!b.nudge_confirm_open());
}

#[test]
fn request_nudge_opens_confirm_with_library_name() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 1;
    b.request_nudge();
    assert!(b.nudge_confirm_open());
    let ac = b.nudge_action_confirm().unwrap();
    let ActionConfirm::Simple(ref sc) = ac;
    assert_eq!(sc.verb, "Rescan");
    assert_eq!(sc.target, "lib-1");
}

#[test]
fn take_nudge_returns_tag_and_clears_modal() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 2;
    b.request_nudge();
    let tenant = b.take_nudge().unwrap();
    assert_eq!(tenant, "tag-2");
    assert!(!b.nudge_confirm_open());
}

#[test]
fn cancel_nudge_clears_modal() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(2);
    b.request_nudge();
    b.cancel_nudge();
    assert!(!b.nudge_confirm_open());
    assert!(b.take_nudge().is_none());
}

// ── Book-removal typed confirm ────────────────────────────────────────────────

#[test]
fn book_remove_confirm_not_open_by_default() {
    let b = LibraryBrowser::new();
    assert!(!b.book_remove_confirm_open());
}

#[test]
fn request_book_remove_opens_confirm_with_basename() {
    let mut b = LibraryBrowser::new();
    b.request_book_remove("/home/user/lib/chapter1.pdf".to_string());
    assert!(b.book_remove_confirm_open());
    assert!(b.confirm_open()); // confirm_open covers book-removal too
    let tc = b.book_remove_confirm_mut().unwrap();
    assert_eq!(tc.name, "chapter1.pdf");
}

#[test]
fn book_remove_rejected_when_input_wrong() {
    let mut b = LibraryBrowser::new();
    b.request_book_remove("/lib/doc.pdf".to_string());
    // Type wrong string.
    for c in "Delete wrong.pdf".chars() {
        b.book_remove_confirm_mut().unwrap().push_char(c);
    }
    let result = b.take_book_remove_if_confirmed();
    assert!(result.is_none());
    // Modal is still open after rejection.
    assert!(b.book_remove_confirm_open());
    // Rejected flag is set.
    assert!(b.book_remove_confirm_mut().unwrap().rejected);
}

#[test]
fn book_remove_accepted_when_input_exact() {
    let mut b = LibraryBrowser::new();
    b.request_book_remove("/lib/doc.pdf".to_string());
    for c in "Delete doc.pdf".chars() {
        b.book_remove_confirm_mut().unwrap().push_char(c);
    }
    let result = b.take_book_remove_if_confirmed();
    assert_eq!(result.unwrap(), "/lib/doc.pdf");
    assert!(!b.book_remove_confirm_open());
}

#[test]
fn cancel_book_remove_clears_confirm() {
    let mut b = LibraryBrowser::new();
    b.request_book_remove("/lib/x.pdf".to_string());
    assert!(b.book_remove_confirm_open());
    b.cancel_book_remove();
    assert!(!b.book_remove_confirm_open());
}

#[test]
fn selected_library_mode_incremental() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(2);
    b.items[0].mode = "incremental".to_string();
    b.selected = 0;
    assert_eq!(b.selected_library_mode(), LibraryMode::Incremental);
}

#[test]
fn selected_library_mode_sync() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(2);
    b.items[0].mode = "sync".to_string();
    b.selected = 0;
    assert_eq!(b.selected_library_mode(), LibraryMode::Sync);
}

#[test]
fn selected_library_mode_empty_list() {
    let b = LibraryBrowser::new();
    assert_eq!(b.selected_library_mode(), LibraryMode::NotLibrary);
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

/// A periodic refresh that rebuilds/reorders `items` after the nudge modal
/// opened must not retarget the rescan: the tag was captured at request time.
#[test]
fn take_nudge_ignores_items_rebuild_after_open() {
    let mut b = LibraryBrowser::new();
    b.items = make_test_rows(3);
    b.selected = 2;
    b.request_nudge();
    // Simulate a refresh replacing the list with different rows.
    b.items = make_test_rows(1);
    b.selected = 0;
    assert_eq!(b.take_nudge().unwrap(), "tag-2");
}
