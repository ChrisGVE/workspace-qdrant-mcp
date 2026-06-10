//! Unit tests for the rules browser view.
//!
//! Extracted from `rules.rs` to keep that file under the 500-line limit.

use super::*;

#[test]
fn browser_initializes_empty() {
    let browser = RuleBrowser::new();
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
    assert!(!browser.detail_open());
}

#[test]
fn truncate_str_short() {
    assert_eq!(truncate_str("hello", 10), "hello");
}

#[test]
fn truncate_str_long() {
    let result = truncate_str("hello world foo bar", 10);
    assert!(result.chars().count() <= 10);
    assert!(result.contains('\u{2026}'));
}

#[test]
fn format_short_date_iso() {
    assert_eq!(format_short_date("2026-04-16T18:41:10+02:00"), "2026-04-16");
}

#[test]
fn format_short_date_short_input() {
    assert_eq!(format_short_date("2026"), "2026");
}

// ── Delete confirm: rule_id capture ───────────────────────────────────────────

fn make_rule(id: &str, text: &str) -> RuleRow {
    RuleRow {
        rule_id: id.to_string(),
        rule_text: text.to_string(),
        scope: "global".to_string(),
        tenant_id: String::new(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        updated_at: "2026-01-01T00:00:00Z".to_string(),
    }
}

/// Type the full `Delete <label>` confirmation phrase into the modal.
fn type_confirmation(browser: &mut RuleBrowser, label: &str) {
    let tc = browser.delete_confirm_mut().unwrap();
    for c in format!("Delete {label}").chars() {
        tc.push_char(c);
    }
}

#[test]
fn take_delete_returns_rule_id_captured_at_open() {
    let mut browser = RuleBrowser::new();
    browser.items = vec![make_rule("rule-a", "alpha"), make_rule("rule-b", "beta")];
    browser.selected = 1;
    browser.request_delete();
    assert!(browser.delete_confirm_open());
    type_confirmation(&mut browser, "beta");
    assert_eq!(
        browser.take_delete_if_confirmed(),
        Some("rule-b".to_string())
    );
    assert!(!browser.delete_confirm_open());
}

/// A periodic refresh can rebuild/reorder `items` while the user types the
/// confirmation. The delete must act on the rule_id captured when the modal
/// opened — not on whatever now sits at `items[selected]`.
#[test]
fn take_delete_ignores_items_rebuild_after_open() {
    let mut browser = RuleBrowser::new();
    browser.items = vec![make_rule("rule-a", "alpha"), make_rule("rule-b", "beta")];
    browser.selected = 1;
    browser.request_delete();
    // Simulate a refresh that reorders the list under the open modal.
    browser.items = vec![make_rule("rule-b", "beta"), make_rule("rule-a", "alpha")];
    type_confirmation(&mut browser, "beta");
    assert_eq!(
        browser.take_delete_if_confirmed(),
        Some("rule-b".to_string())
    );
}

#[test]
fn take_delete_rejects_wrong_input_and_keeps_modal() {
    let mut browser = RuleBrowser::new();
    browser.items = vec![make_rule("rule-a", "alpha")];
    browser.request_delete();
    type_confirmation(&mut browser, "wrong-name");
    assert_eq!(browser.take_delete_if_confirmed(), None);
    assert!(browser.delete_confirm_open());
}
