//! File-list popup tab and content overlay for the Projects and Libraries views.
//!
//! Owns state and rendering for the "Files" tab inside per-project/library
//! detail popups. Deliberately free of SQLite/I/O so navigation is unit-testable
//! in memory. Key types: [`FileEntry`], [`FileListState`], [`FileListAction`].
//! Extension point for content-type rendering: [`render_file_content`].

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use crate::tui::theme;
use crate::tui::util::{scroll_offset, truncate_path, visible_rows};

// ─── Data types ─────────────────────────────────────────────────────────────

/// A single tracked file displayed in the Files tab.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Path relative to the watch folder root (for display).
    pub relative_path: String,
    /// Reconstructed absolute path used for on-disk reads.
    pub abs_path: String,
    /// File size in bytes; `None` means the metadata could not be read.
    pub size: Option<u64>,
    /// Number of chunks indexed for this file.
    pub chunk_count: i64,
}

// ─── Which popup tab is active ───────────────────────────────────────────────

/// The two tabs inside a detail popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopupTab {
    /// Existing metadata view (unchanged).
    Detail,
    /// New scrollable file list with content overlay.
    Files,
}

impl PopupTab {}

// ─── State ───────────────────────────────────────────────────────────────────

/// Full popup-layer state for the Files tab.
///
/// Keeps the file list cursor and any open content overlay. Designed so every
/// navigation method can be tested on pure in-memory data.
#[derive(Debug, Default)]
pub struct FileListState {
    /// Which tab is currently shown in the detail popup.
    pub tab: PopupTab,
    /// Files for the currently selected project / library.
    pub files: Vec<FileEntry>,
    /// Selected row in the file list.
    pub file_cursor: usize,
    /// When `Some`, the content overlay is open, showing the file's text.
    pub content: Option<String>,
    /// Vertical scroll offset inside the content overlay.
    pub content_scroll: u16,
}

impl Default for PopupTab {
    fn default() -> Self {
        PopupTab::Detail
    }
}

impl FileListState {
    /// Create a new, empty state with the Detail tab active.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a fresh set of file entries and reset the cursor.
    ///
    /// Call this each time the popup opens or when the watch folder changes.
    pub fn load(&mut self, entries: Vec<FileEntry>) {
        self.files = entries;
        self.file_cursor = 0;
        self.content = None;
        self.content_scroll = 0;
    }

    /// Switch to the Detail tab, closing any open content overlay.
    pub fn activate_detail_tab(&mut self) {
        self.tab = PopupTab::Detail;
        self.content = None;
        self.content_scroll = 0;
    }

    /// Switch to the Files tab.
    pub fn activate_files_tab(&mut self) {
        self.tab = PopupTab::Files;
    }

    /// Whether the content overlay is currently open.
    pub fn content_open(&self) -> bool {
        self.content.is_some()
    }

    /// Move the file-list cursor one row up.
    pub fn cursor_up(&mut self) {
        self.file_cursor = self.file_cursor.saturating_sub(1);
    }

    /// Move the file-list cursor one row down.
    pub fn cursor_down(&mut self) {
        if !self.files.is_empty() {
            self.file_cursor = (self.file_cursor + 1).min(self.files.len() - 1);
        }
    }

    /// Jump to the first file.
    pub fn cursor_first(&mut self) {
        self.file_cursor = 0;
    }

    /// Jump to the last file.
    pub fn cursor_last(&mut self) {
        if !self.files.is_empty() {
            self.file_cursor = self.files.len() - 1;
        }
    }

    /// Open the content overlay for the currently selected file.
    ///
    /// Reads the file from disk. Binary files produce a short notice instead of
    /// raw bytes. Missing files produce a "not found" message rather than panicking.
    pub fn open_content(&mut self) {
        if let Some(entry) = self.files.get(self.file_cursor) {
            let text = read_file_for_display(&entry.abs_path);
            self.content = Some(text);
            self.content_scroll = 0;
        }
    }

    /// Close the content overlay, returning to the file list.
    pub fn close_content(&mut self) {
        self.content = None;
        self.content_scroll = 0;
    }

    /// Scroll the content overlay one line down.
    pub fn content_scroll_down(&mut self) {
        self.content_scroll = self.content_scroll.saturating_add(1);
    }

    /// Scroll the content overlay one line up.
    pub fn content_scroll_up(&mut self) {
        self.content_scroll = self.content_scroll.saturating_sub(1);
    }

    /// Close and reset all state; called when the parent popup is closed.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ─── Content rendering ───────────────────────────────────────────────────────

/// The binary-file notice prefix. Present in the returned string when the file
/// cannot be displayed as UTF-8 text.
pub const BINARY_NOTICE_PREFIX: &str = "(binary file,";

/// Convert raw file bytes to display text (task-4 extension point).
///
/// UTF-8 files are returned as-is. Binary files (NUL bytes or high replacement-
/// char ratio) get a one-line notice instead. `path` is reserved for task 4's
/// content-type dispatch (syntax highlighting, Markdown, etc.).
pub fn render_file_content(path: &str, raw: &[u8]) -> String {
    // Try interpreting as UTF-8. Lossily convert so partial UTF-8 files still
    // display rather than showing only a notice. Then check for a high ratio of
    // replacement characters as a binary heuristic.
    let text = String::from_utf8_lossy(raw);
    let replacement_count = text.chars().filter(|&c| c == '\u{FFFD}').count();
    let char_count = text.chars().count();

    let is_binary = char_count > 0 && replacement_count * 10 > char_count;
    if is_binary {
        return format!("{BINARY_NOTICE_PREFIX} {} bytes)", raw.len());
    }

    // NUL bytes are another binary indicator independent of UTF-8 validity.
    if raw.contains(&0u8) {
        return format!("{BINARY_NOTICE_PREFIX} {} bytes)", raw.len());
    }

    let _ = path; // reserved for task 4's content-type dispatch
    text.into_owned()
}

/// Read a file from disk; wraps `render_file_content` with I/O.
fn read_file_for_display(abs_path: &str) -> String {
    match std::fs::read(abs_path) {
        Ok(bytes) => render_file_content(abs_path, &bytes),
        Err(e) => format!("(could not read file: {e})"),
    }
}

// ─── Key action ──────────────────────────────────────────────────────────────

/// What the caller should do after a key event inside the popup layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileListAction {
    /// The key was consumed; caller does nothing further.
    Consumed,
    /// The key was not consumed; the caller may apply its own fallback.
    NotConsumed,
    /// The entire popup (not just the overlay) should be closed.
    ClosePopup,
}

/// Route a key event through the file-list state machine.
///
/// Call only when the detail popup is open (`browser.detail_open()`).
pub fn handle_popup_key(
    state: &mut FileListState,
    key: crossterm::event::KeyCode,
) -> FileListAction {
    use crossterm::event::KeyCode;

    match state.tab {
        PopupTab::Detail => match key {
            // Tab switches to Files.
            KeyCode::Tab => {
                state.activate_files_tab();
                FileListAction::Consumed
            }
            // BackTab stays in Detail (already there); close on Esc.
            KeyCode::BackTab => FileListAction::Consumed,
            KeyCode::Esc => FileListAction::ClosePopup,
            _ => FileListAction::NotConsumed,
        },

        PopupTab::Files => {
            if state.content_open() {
                // Content overlay captures j/k for scrolling, Esc to close.
                match key {
                    KeyCode::Char('j') | KeyCode::Down => {
                        state.content_scroll_down();
                        FileListAction::Consumed
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        state.content_scroll_up();
                        FileListAction::Consumed
                    }
                    KeyCode::Esc => {
                        state.close_content();
                        FileListAction::Consumed
                    }
                    _ => FileListAction::Consumed,
                }
            } else {
                match key {
                    // Tab / BackTab switch tab.
                    KeyCode::Tab | KeyCode::BackTab => {
                        state.activate_detail_tab();
                        FileListAction::Consumed
                    }
                    // Cursor navigation.
                    KeyCode::Char('j') | KeyCode::Down => {
                        state.cursor_down();
                        FileListAction::Consumed
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        state.cursor_up();
                        FileListAction::Consumed
                    }
                    KeyCode::Char('g') => {
                        state.cursor_first();
                        FileListAction::Consumed
                    }
                    KeyCode::Char('G') => {
                        state.cursor_last();
                        FileListAction::Consumed
                    }
                    // Enter opens the content overlay.
                    KeyCode::Enter => {
                        state.open_content();
                        FileListAction::Consumed
                    }
                    // Esc closes the whole popup (not just the overlay, since
                    // there is no overlay to close here).
                    KeyCode::Esc => FileListAction::ClosePopup,
                    _ => FileListAction::Consumed,
                }
            }
        }
    }
}

// ─── Rendering ───────────────────────────────────────────────────────────────

/// Draw the tab bar (1-row strip at top of popup). Active tab = bold + cyan.
pub fn draw_tab_bar(frame: &mut Frame, area: Rect, active: PopupTab) {
    let detail_style = if active == PopupTab::Detail {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let files_style = if active == PopupTab::Files {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let spans = vec![
        Span::raw(" "),
        Span::styled(" Detail ", detail_style),
        Span::styled("  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Files ", files_style),
        Span::styled(
            "  Tab/Shift+Tab to switch",
            Style::default().fg(Color::DarkGray),
        ),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Draw the file list table inside the Files tab content area.
pub fn draw_file_list_tab(frame: &mut Frame, inner: Rect, state: &FileListState) {
    // Split vertically: hint bar (1 row) + table (rest).
    let sections = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(inner);
    let hint_area = sections[0];
    let table_area = sections[1];

    // Key hint.
    let hint = Paragraph::new(Line::from(vec![Span::styled(
        "Enter: view content   j/k: navigate   g/G: first/last   Esc: close",
        Style::default().fg(Color::DarkGray),
    )]));
    frame.render_widget(hint, hint_area);

    if state.files.is_empty() {
        let msg =
            Paragraph::new("No tracked files found").style(Style::default().fg(Color::DarkGray));
        frame.render_widget(msg, table_area);
        return;
    }

    // Table: path, size, chunks.
    let header = Row::new(vec!["File", "Size", "Chunks"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    // Chrome: borders 2 + header 1 + header margin 1.
    let visible = visible_rows(table_area.height, 4);
    let offset = scroll_offset(state.file_cursor, visible);

    // Fixed: size 10, chunks 8; separator gaps 2; path fills remainder.
    let path_w = (table_area.width as usize)
        .saturating_sub(10 + 8 + 2 + 2)
        .max(10);

    let rows: Vec<Row> = state
        .files
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(i, entry)| {
            let row_style = if i == state.file_cursor {
                theme::selected_row_style()
            } else {
                Style::default()
            };

            let size_str = match entry.size {
                Some(b) => super::service_data::format_bytes(b),
                None => "\u{2014}".to_string(), // em dash for "unknown"
            };

            Row::new(vec![
                Span::styled(
                    truncate_path(&entry.relative_path, path_w),
                    Style::default().fg(Color::White),
                ),
                Span::styled(format!("{:>9}", size_str), Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{:>7}", crate::tui::util::fmt_count(entry.chunk_count)),
                    Style::default().fg(Color::Cyan),
                ),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Min(10),    // relative path (flex)
        Constraint::Length(10), // size
        Constraint::Length(8),  // chunks
    ];

    let block = Block::default().borders(Borders::ALL).title(" Files ");
    let table = Table::new(rows, widths).header(header).block(block);
    frame.render_widget(table, table_area);
}

/// Draw the content overlay (full-popup panel) for the selected file.
pub fn draw_content_overlay(frame: &mut Frame, area: Rect, state: &FileListState) {
    let Some(ref content) = state.content else {
        return;
    };

    // Use almost the full popup area, leaving a 1-cell margin on each side.
    let overlay_w = area.width.saturating_sub(2);
    let overlay_h = area.height.saturating_sub(2);
    let x = area.x + 1;
    let y = area.y + 1;
    let overlay_area = Rect::new(x, y, overlay_w, overlay_h);

    frame.render_widget(Clear, overlay_area);

    let filename = state
        .files
        .get(state.file_cursor)
        .map(|e| e.relative_path.as_str())
        .unwrap_or("file");

    // Inner area for text (inside borders).
    let inner_h = overlay_h.saturating_sub(2);
    let inner_w = overlay_w.saturating_sub(2) as usize;

    let all_lines: Vec<Line> = content
        .lines()
        .flat_map(|line| wrap_line(line, inner_w))
        .map(|s| Line::from(Span::raw(s)))
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {filename} "))
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .style(Style::default().bg(Color::Black))
        .title_bottom(Line::from(Span::styled(
            " j/k: scroll  Esc: back ",
            Style::default().fg(Color::DarkGray),
        )));

    let scroll_pos = state.content_scroll.min(
        all_lines
            .len()
            .saturating_sub(inner_h as usize)
            .try_into()
            .unwrap_or(u16::MAX),
    );

    let para = Paragraph::new(all_lines)
        .block(block)
        .scroll((scroll_pos, 0));

    frame.render_widget(para, overlay_area);
}

/// Wrap a line to `max_width` columns at character boundaries.
pub(crate) fn wrap_line(line: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![String::new()];
    }
    if line.chars().count() <= max_width {
        return vec![line.to_string()];
    }
    let mut result = Vec::new();
    let mut current = String::new();
    let mut col = 0;
    for ch in line.chars() {
        current.push(ch);
        col += 1;
        if col >= max_width {
            result.push(current.clone());
            current.clear();
            col = 0;
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[cfg(test)]
#[path = "file_list_tests.rs"]
mod tests;
