//! Unit tests for the TUI scratchpad browser view.
//!
//! Extracted into a separate file to keep `scratchpad.rs` under the 500-line
//! limit.

use super::*;

#[test]
fn browser_initializes_empty() {
    let browser = ScratchpadBrowser::new();
    assert!(browser.items.is_empty());
    assert_eq!(browser.selected, 0);
    assert!(!browser.detail_open());
}

#[test]
fn format_tags_empty() {
    assert_eq!(format_tags("[]"), "—");
    assert_eq!(format_tags(""), "—");
}

#[test]
fn format_tags_json() {
    assert_eq!(format_tags(r#"["rust","cli","tui"]"#), "rust, cli, tui");
}

#[test]
fn format_tags_single() {
    assert_eq!(format_tags(r#"["analysis"]"#), "analysis");
}

#[test]
fn detail_scroll() {
    let mut browser = ScratchpadBrowser::new();
    browser.detail_open = true;
    browser.scroll_detail_down();
    assert_eq!(browser.detail_scroll, 1);
    browser.scroll_detail_up();
    assert_eq!(browser.detail_scroll, 0);
    browser.scroll_detail_up();
    assert_eq!(browser.detail_scroll, 0); // no underflow
}
