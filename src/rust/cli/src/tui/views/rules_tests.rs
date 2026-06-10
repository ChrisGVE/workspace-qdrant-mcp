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
