//! Unit tests for the TUI queue browser view.
//!
//! Extracted into a separate file to keep `queue.rs` under the 500-line limit.

use super::*;

#[test]
fn queue_browser_new_starts_empty() {
    let browser = QueueBrowser::new();
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
    assert_eq!(browser.filter, StatusFilter::All);
    assert!(browser.detail.is_none());
    assert!(browser.last_refresh.is_none());
}

#[test]
fn select_next_clamps_to_bounds() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 4;
    browser.select_next();
    assert_eq!(browser.selected, 4);
}

#[test]
fn select_prev_clamps_to_zero() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 0;
    browser.select_prev();
    assert_eq!(browser.selected, 0);
}

#[test]
fn select_next_advances() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 2;
    browser.select_next();
    assert_eq!(browser.selected, 3);
}

#[test]
fn select_prev_retreats() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(5);
    browser.selected = 3;
    browser.select_prev();
    assert_eq!(browser.selected, 2);
}

#[test]
fn page_up_clamps() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(50);
    browser.selected = 5;
    browser.page_up(20);
    assert_eq!(browser.selected, 0);
}

#[test]
fn page_down_clamps() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(50);
    browser.selected = 45;
    browser.page_down(20);
    assert_eq!(browser.selected, 49);
}

#[test]
fn cycle_filter_resets_selection() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(10);
    browser.selected = 5;
    browser.cycle_filter();
    assert_eq!(browser.selected, 0);
    assert_eq!(browser.filter, StatusFilter::Pending);
    assert!(browser.last_refresh.is_none());
}

#[test]
fn jump_first_and_last() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(10);
    browser.selected = 4;
    browser.jump_first();
    assert_eq!(browser.selected, 0);
    browser.jump_last();
    assert_eq!(browser.selected, 9);
    // Empty list never panics or overflows.
    let mut empty = QueueBrowser::new();
    empty.jump_last();
    assert_eq!(empty.selected, 0);
}

#[test]
fn close_detail_clears() {
    let mut browser = QueueBrowser::new();
    browser.detail = Some(QueueDetail {
        queue_id: "test".into(),
        idempotency_key: "key".into(),
        item_type: "file".into(),
        op: "add".into(),
        collection: "projects".into(),
        status: "done".into(),
        project: "proj".into(),
        tenant_id: "t1".into(),
        object: "main.rs".into(),
        payload_json: "{}".into(),
        error_message: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        retry_count: 0,
    });
    assert!(browser.detail_open());
    browser.close_detail();
    assert!(!browser.detail_open());
}

#[test]
fn filter_accessor() {
    let browser = QueueBrowser::new();
    assert_eq!(browser.filter(), StatusFilter::All);
}

#[test]
fn page_filter_narrows_visible_items() {
    use crossterm::event::KeyCode;
    let mut browser = QueueBrowser::new();
    browser.all_items = vec![
        QueueRow {
            object: "main.rs".into(),
            ..make_test_rows(1)[0].clone()
        },
        QueueRow {
            object: "lib.py".into(),
            ..make_test_rows(1)[0].clone()
        },
    ];
    browser.recompute_visible();
    assert_eq!(browser.items.len(), 2); // no filter → all visible
    browser.page_filter_mut().activate();
    for c in "\\.rs".chars() {
        browser.page_filter_mut().handle_key(KeyCode::Char(c));
    }
    browser.page_filter_mut().handle_key(KeyCode::Enter);
    browser.recompute_visible();
    assert_eq!(browser.items.len(), 1);
    assert_eq!(browser.items[0].object, "main.rs");
}

#[test]
fn global_filter_composes_with_page_filter() {
    let mut browser = QueueBrowser::new();
    browser.all_items = vec![
        QueueRow {
            project: "alpha".into(),
            object: "main.rs".into(),
            ..make_test_rows(1)[0].clone()
        },
        QueueRow {
            project: "beta".into(),
            object: "main.rs".into(),
            ..make_test_rows(1)[0].clone()
        },
    ];
    // Global filter on project name narrows to alpha; both have main.rs.
    browser.set_global_filter(regex::Regex::new("(?i)alpha").ok());
    assert_eq!(browser.items.len(), 1);
    assert_eq!(browser.items[0].project, "alpha");
    // Clearing the global filter restores both.
    browser.set_global_filter(None);
    assert_eq!(browser.items.len(), 2);
}

#[test]
fn recompute_clamps_selection() {
    let mut browser = QueueBrowser::new();
    browser.all_items = make_test_rows(5);
    browser.recompute_visible();
    browser.selected = 4;
    browser.set_global_filter(regex::Regex::new("(?i)nomatchxyz").ok());
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
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
fn age_color_health() {
    // Failed is always red regardless of age.
    assert_eq!(age_color("failed", "2s ago"), Color::Red);
    // A pending item waiting hours/days is stale → yellow.
    assert_eq!(age_color("pending", "3h ago"), Color::Yellow);
    assert_eq!(age_color("in_progress", "1d ago"), Color::Yellow);
    // Fresh or completed items are a legible gray, not dim.
    assert_eq!(age_color("pending", "5s ago"), Color::Gray);
    assert_eq!(age_color("done", "2d ago"), Color::Gray);
}

#[test]
fn truncate_str_short() {
    assert_eq!(truncate_str("hello", 10), "hello");
}

#[test]
fn truncate_str_long() {
    let long = "a".repeat(40);
    let result = truncate_str(&long, 10);
    assert!(result.ends_with("..."));
    assert!(result.chars().count() <= 10);
}

#[test]
fn select_on_empty_list() {
    let mut browser = QueueBrowser::new();
    browser.select_next();
    assert_eq!(browser.selected, 0);
    browser.select_prev();
    assert_eq!(browser.selected, 0);
}

// ── Action confirm state machine ──────────────────────────────────────────────

#[test]
fn confirm_not_open_by_default() {
    let browser = QueueBrowser::new();
    assert!(!browser.confirm_open());
    assert!(browser.action_confirm().is_none());
}

#[test]
fn request_action_on_empty_list_is_noop() {
    let mut browser = QueueBrowser::new();
    browser.request_action(QueueAction::Retry);
    assert!(!browser.confirm_open());
}

#[test]
fn request_retry_opens_confirm() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(3);
    browser.selected = 1;
    browser.request_action(QueueAction::Retry);
    assert!(browser.confirm_open());
    let confirm = browser.action_confirm().unwrap();
    // ActionConfirm::Simple wraps the verb
    let ActionConfirm::Simple(ref sc) = confirm;
    assert_eq!(sc.verb, "Retry");
}

#[test]
fn request_cancel_opens_confirm_with_cancel_verb() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(2);
    browser.request_action(QueueAction::Cancel);
    assert!(browser.confirm_open());
    let ActionConfirm::Simple(ref sc) = browser.action_confirm().unwrap();
    assert_eq!(sc.verb, "Cancel pending items for");
}

#[test]
fn request_remove_opens_confirm_with_remove_verb() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(2);
    browser.request_action(QueueAction::Remove);
    assert!(browser.confirm_open());
    let ActionConfirm::Simple(ref sc) = browser.action_confirm().unwrap();
    assert_eq!(sc.verb, "Remove");
}

#[test]
fn take_action_returns_action_queue_id_and_tenant_id() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(3);
    browser.selected = 2;
    browser.request_action(QueueAction::Retry);
    let (action, queue_id, tenant_id) = browser.take_action().unwrap();
    assert_eq!(action, QueueAction::Retry);
    assert_eq!(queue_id, "id-2");
    assert_eq!(tenant_id, "tenant-2");
    assert!(!browser.confirm_open());
}

/// Cancel dispatches tenant-wide on the daemon (CancelItemsRequest takes a
/// tenant_id, not a queue_id) — the pending action must carry the tenant_id
/// captured from the selected row.
#[test]
fn cancel_action_carries_tenant_id_of_selected_row() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(3);
    browser.selected = 1;
    browser.request_action(QueueAction::Cancel);
    let (action, queue_id, tenant_id) = browser.take_action().unwrap();
    assert_eq!(action, QueueAction::Cancel);
    assert_eq!(queue_id, "id-1");
    assert_eq!(tenant_id, "tenant-1");
    assert_ne!(tenant_id, queue_id);
}

/// A refresh that reorders/rebuilds `items` after the modal opened must not
/// retarget the action: the ids were captured at request time.
#[test]
fn pending_action_survives_items_rebuild() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(3);
    browser.selected = 2;
    browser.request_action(QueueAction::Cancel);
    // Simulate a periodic refresh replacing the list with different rows.
    browser.items = make_test_rows(1);
    browser.selected = 0;
    let (_, queue_id, tenant_id) = browser.take_action().unwrap();
    assert_eq!(queue_id, "id-2");
    assert_eq!(tenant_id, "tenant-2");
}

#[test]
fn cancel_action_clears_pending() {
    let mut browser = QueueBrowser::new();
    browser.items = make_test_rows(1);
    browser.request_action(QueueAction::Remove);
    assert!(browser.confirm_open());
    browser.cancel_action();
    assert!(!browser.confirm_open());
    assert!(browser.action_confirm().is_none());
}

#[test]
fn set_message_and_force_refresh() {
    let mut browser = QueueBrowser::new();
    browser.last_refresh = Some(std::time::Instant::now());
    browser.set_message("done".to_string());
    assert_eq!(browser.message, Some("done".to_string()));
    browser.force_refresh();
    assert!(browser.last_refresh.is_none());
}

fn make_test_rows(n: usize) -> Vec<QueueRow> {
    (0..n)
        .map(|i| QueueRow {
            queue_id: format!("id-{i}"),
            tenant_id: format!("tenant-{i}"),
            short_id: format!("id-{i}"),
            project: "project".into(),
            object: "file.rs".into(),
            item_type: "file".into(),
            op: "add".into(),
            status: "pending".into(),
            age: "1m ago".into(),
            kind: 'P',
            size: Some(1024),
        })
        .collect()
}
