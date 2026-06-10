//! File-list popup tab and content overlay for the Projects and Libraries views.
//!
//! Owns state and rendering for the "Files" tab inside per-project/library
//! detail popups. Deliberately free of SQLite/I/O so navigation is unit-testable
//! in memory. Key types: [`FileEntry`], [`FileListState`], [`FileListAction`].
//!
//! Content rendering is delegated to [`crate::tui::render::content::render_for_path`],
//! which dispatches on file extension (Markdown, code with syntax hints, plain text)
//! and handles binary-file detection. The overlay stores pre-rendered
//! `Vec<Line<'static>>` so the draw path never re-parses.

use ratatui::text::Line;

// Rendering lives in `file_list_draw.rs`; re-export so the Projects/Libraries
// views keep importing the draw functions from `file_list`.
pub use super::file_list_draw::{draw_content_overlay, draw_file_list_tab, draw_tab_bar};
use crate::tui::theme;

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
    /// Pre-rendered styled lines for the content overlay, set when the overlay
    /// is open. `None` means the overlay is closed.
    pub content: Option<Vec<Line<'static>>>,
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
    /// Reads the file from disk and pre-renders it through
    /// [`crate::tui::render::content::render_for_path`].  Binary files get a
    /// one-line notice; missing files get an error notice; all other files are
    /// rendered with the appropriate content-type renderer (Markdown, code, plain).
    ///
    /// The rendered lines are stored in `self.content` so `draw_content_overlay`
    /// can hand them directly to `Paragraph::new` without re-parsing.
    ///
    /// `width` is used for word-wrapping; pass the current overlay inner width.
    pub fn open_content_with_width(&mut self, width: usize) {
        if let Some(entry) = self.files.get(self.file_cursor) {
            let lines = read_and_render(&entry.abs_path, width);
            self.content = Some(lines);
            self.content_scroll = 0;
        }
    }

    /// Open the content overlay using a default width (80 columns).
    ///
    /// Convenience wrapper; callers that know the current terminal width should
    /// prefer [`open_content_with_width`][Self::open_content_with_width].
    pub fn open_content(&mut self) {
        self.open_content_with_width(80);
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

/// The binary-file notice prefix.
///
/// Re-exported from [`crate::tui::render::content`] so tests that import it
/// from this module via `use super::*` continue to compile without changes.
#[cfg(test)]
pub use crate::tui::render::content::BINARY_NOTICE_PREFIX;

/// Read a file from disk and render it to styled lines.
///
/// Delegates to [`crate::tui::render::content::render_for_path`] for content-
/// type dispatch. Missing files produce a single error-notice line rather than
/// panicking.
fn read_and_render(abs_path: &str, width: usize) -> Vec<Line<'static>> {
    use ratatui::style::Style;
    use ratatui::text::Span;
    match std::fs::read(abs_path) {
        Ok(bytes) => crate::tui::render::content::render_for_path(abs_path, &bytes, width),
        Err(e) => vec![Line::from(Span::styled(
            format!("(could not read file: {e})"),
            Style::default().fg(theme::COLOR_DIM),
        ))],
    }
}

// ─── Key action ──────────────────────────────────────────────────────────────

/// What the caller should do after a key event inside the popup layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileListAction {
    /// The key was consumed; caller does nothing further.
    Consumed,
    /// The key was not consumed; the caller may apply its own fallback.
    NotConsumed,
    /// The entire popup (not just the overlay) should be closed.
    ClosePopup,
    /// The user pressed `d` on a file in the Files tab; caller should open a
    /// typed-name confirm for this book. The inner string is the absolute path.
    RequestBookRemove(String),
}

/// Context for the file-list popup, provided by the caller so the Files tab
/// can gate actions on library mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryMode {
    /// Incremental library: books may be removed.
    Incremental,
    /// Sync library: books are managed by the watcher; removal is blocked.
    Sync,
    /// Not a library (project popup): removal is blocked.
    NotLibrary,
}

/// Route a key event through the file-list state machine.
///
/// Call only when the detail popup is open (`browser.detail_open()`).
/// Pass `library_mode` so the Files tab can gate `d` (remove book) on the
/// library mode: only [`LibraryMode::Incremental`] allows removal.
pub fn handle_popup_key(
    state: &mut FileListState,
    key: crossterm::event::KeyCode,
    library_mode: LibraryMode,
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
                    // `d` removes a book — only for incremental libraries.
                    KeyCode::Char('d') => match library_mode {
                        LibraryMode::Incremental => {
                            if let Some(entry) = state.files.get(state.file_cursor) {
                                FileListAction::RequestBookRemove(entry.abs_path.clone())
                            } else {
                                FileListAction::Consumed
                            }
                        }
                        LibraryMode::Sync => {
                            // Sync libraries are managed by the watcher; signal
                            // the caller to show a status message rather than silently
                            // swallowing the key.
                            FileListAction::RequestBookRemove("__sync_blocked__".to_string())
                        }
                        LibraryMode::NotLibrary => FileListAction::Consumed,
                    },
                    // Esc closes the whole popup (not just the overlay, since
                    // there is no overlay to close here).
                    KeyCode::Esc => FileListAction::ClosePopup,
                    _ => FileListAction::Consumed,
                }
            }
        }
    }
}

/// Wrap a line to `max_width` columns at character boundaries.
#[cfg(test)]
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
