//! Unit tests for the file-list popup module.
//!
//! Extracted into a separate file to keep `file_list.rs` under the 500-line
//! limit while maintaining full test coverage.

use super::*;

// ── FileEntry construction ────────────────────────────────────────────────────

#[test]
fn file_entry_stores_all_fields() {
    let entry = FileEntry {
        relative_path: "src/main.rs".to_string(),
        abs_path: "/home/user/proj/src/main.rs".to_string(),
        size: Some(4096),
        chunk_count: 3,
    };
    assert_eq!(entry.relative_path, "src/main.rs");
    assert_eq!(entry.size, Some(4096));
    assert_eq!(entry.chunk_count, 3);
}

#[test]
fn file_entry_optional_size() {
    let entry = FileEntry {
        relative_path: "deleted.txt".to_string(),
        abs_path: "/gone".to_string(),
        size: None,
        chunk_count: 0,
    };
    assert!(entry.size.is_none());
}

// ── Tab switching ─────────────────────────────────────────────────────────────

#[test]
fn state_starts_on_detail_tab() {
    let s = FileListState::new();
    assert_eq!(s.tab, PopupTab::Detail);
}

#[test]
fn activate_files_tab_switches_tab() {
    let mut s = FileListState::new();
    s.activate_files_tab();
    assert_eq!(s.tab, PopupTab::Files);
    s.activate_detail_tab();
    assert_eq!(s.tab, PopupTab::Detail);
}

#[test]
fn activate_detail_tab_clears_content() {
    let mut s = FileListState::new();
    s.tab = PopupTab::Files;
    s.content = Some("hello".to_string());
    s.content_scroll = 5;
    s.activate_detail_tab();
    assert_eq!(s.tab, PopupTab::Detail);
    assert!(s.content.is_none());
    assert_eq!(s.content_scroll, 0);
}

// ── File-list cursor navigation ───────────────────────────────────────────────

#[test]
fn cursor_down_advances() {
    let mut s = FileListState::new();
    s.files = make_entries(5);
    s.file_cursor = 2;
    s.cursor_down();
    assert_eq!(s.file_cursor, 3);
}

#[test]
fn cursor_down_clamps_at_end() {
    let mut s = FileListState::new();
    s.files = make_entries(3);
    s.file_cursor = 2;
    s.cursor_down();
    assert_eq!(s.file_cursor, 2);
}

#[test]
fn cursor_up_retreats() {
    let mut s = FileListState::new();
    s.files = make_entries(5);
    s.file_cursor = 3;
    s.cursor_up();
    assert_eq!(s.file_cursor, 2);
}

#[test]
fn cursor_up_clamps_at_zero() {
    let mut s = FileListState::new();
    s.files = make_entries(3);
    s.file_cursor = 0;
    s.cursor_up();
    assert_eq!(s.file_cursor, 0);
}

#[test]
fn cursor_first_and_last() {
    let mut s = FileListState::new();
    s.files = make_entries(10);
    s.file_cursor = 7;
    s.cursor_first();
    assert_eq!(s.file_cursor, 0);
    s.cursor_last();
    assert_eq!(s.file_cursor, 9);
}

#[test]
fn cursor_on_empty_list_no_panic() {
    let mut s = FileListState::new();
    s.cursor_down();
    s.cursor_up();
    s.cursor_last();
    assert_eq!(s.file_cursor, 0);
}

// ── Load resets cursor ────────────────────────────────────────────────────────

#[test]
fn load_resets_cursor_and_content() {
    let mut s = FileListState::new();
    s.file_cursor = 5;
    s.content = Some("old content".to_string());
    s.load(make_entries(3));
    assert_eq!(s.file_cursor, 0);
    assert!(s.content.is_none());
    assert_eq!(s.files.len(), 3);
}

// ── Content overlay open / close ──────────────────────────────────────────────

#[test]
fn close_content_clears_state() {
    let mut s = FileListState::new();
    s.content = Some("text".to_string());
    s.content_scroll = 4;
    s.close_content();
    assert!(s.content.is_none());
    assert_eq!(s.content_scroll, 0);
}

#[test]
fn content_scroll_increments_and_decrements() {
    let mut s = FileListState::new();
    s.content_scroll_down();
    s.content_scroll_down();
    assert_eq!(s.content_scroll, 2);
    s.content_scroll_up();
    assert_eq!(s.content_scroll, 1);
}

#[test]
fn content_scroll_up_clamps_at_zero() {
    let mut s = FileListState::new();
    s.content_scroll_up();
    assert_eq!(s.content_scroll, 0);
}

// ── render_file_content ───────────────────────────────────────────────────────

#[test]
fn render_plain_text_unchanged() {
    let raw = b"hello\nworld\n";
    let result = render_file_content("file.txt", raw);
    assert_eq!(result, "hello\nworld\n");
}

#[test]
fn render_binary_file_gives_notice() {
    // A sequence with many NUL bytes is clearly binary.
    let raw = vec![0u8; 100];
    let result = render_file_content("file.bin", &raw);
    assert!(
        result.starts_with(BINARY_NOTICE_PREFIX),
        "expected binary notice, got: {result}"
    );
}

#[test]
fn render_binary_notice_includes_byte_count() {
    let raw = vec![0u8; 42];
    let result = render_file_content("a.bin", &raw);
    assert!(result.contains("42"), "expected byte count in: {result}");
}

#[test]
fn render_empty_file_produces_empty_string() {
    let result = render_file_content("empty.txt", b"");
    assert_eq!(result, "");
}

// ── handle_popup_key ──────────────────────────────────────────────────────────

#[test]
fn detail_tab_esc_closes_popup() {
    let mut s = FileListState::new();
    assert_eq!(s.tab, PopupTab::Detail);
    let action = handle_popup_key(&mut s, crossterm::event::KeyCode::Esc);
    assert_eq!(action, FileListAction::ClosePopup);
}

#[test]
fn detail_tab_tab_switches_to_files() {
    let mut s = FileListState::new();
    let action = handle_popup_key(&mut s, crossterm::event::KeyCode::Tab);
    assert_eq!(action, FileListAction::Consumed);
    assert_eq!(s.tab, PopupTab::Files);
}

#[test]
fn files_tab_esc_closes_popup_when_no_content() {
    let mut s = FileListState::new();
    s.tab = PopupTab::Files;
    let action = handle_popup_key(&mut s, crossterm::event::KeyCode::Esc);
    assert_eq!(action, FileListAction::ClosePopup);
}

#[test]
fn files_tab_content_esc_closes_overlay_not_popup() {
    let mut s = FileListState::new();
    s.tab = PopupTab::Files;
    s.content = Some("text".to_string());
    let action = handle_popup_key(&mut s, crossterm::event::KeyCode::Esc);
    assert_eq!(action, FileListAction::Consumed);
    assert!(s.content.is_none());
    assert_eq!(s.tab, PopupTab::Files);
}

#[test]
fn files_tab_j_moves_cursor_down() {
    let mut s = FileListState::new();
    s.tab = PopupTab::Files;
    s.files = make_entries(5);
    handle_popup_key(&mut s, crossterm::event::KeyCode::Char('j'));
    assert_eq!(s.file_cursor, 1);
}

#[test]
fn files_tab_backtab_returns_to_detail() {
    let mut s = FileListState::new();
    s.tab = PopupTab::Files;
    let action = handle_popup_key(&mut s, crossterm::event::KeyCode::BackTab);
    assert_eq!(action, FileListAction::Consumed);
    assert_eq!(s.tab, PopupTab::Detail);
}

// ── wrap_line ─────────────────────────────────────────────────────────────────

#[test]
fn wrap_line_short_unchanged() {
    assert_eq!(wrap_line("hello", 80), vec!["hello".to_string()]);
}

#[test]
fn wrap_line_long_splits() {
    let result = wrap_line("abcdef", 3);
    assert_eq!(result, vec!["abc".to_string(), "def".to_string()]);
}

#[test]
fn wrap_line_zero_width_no_panic() {
    let result = wrap_line("hello", 0);
    assert_eq!(result.len(), 1);
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_entries(n: usize) -> Vec<FileEntry> {
    (0..n)
        .map(|i| FileEntry {
            relative_path: format!("src/file_{i}.rs"),
            abs_path: format!("/project/src/file_{i}.rs"),
            size: Some(1024 * (i as u64 + 1)),
            chunk_count: i as i64,
        })
        .collect()
}
