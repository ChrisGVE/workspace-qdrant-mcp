//! Log viewer widget with tail behavior and color-coded log levels.
//!
//! Reads the daemon log file and displays the last N lines with
//! auto-scroll to bottom. Supports scrolling with j/k and arrow keys.

use std::path::PathBuf;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use super::logs_data::{file_len, read_from_offset, read_tail_lines, Level, LogLine, MAX_LINES};
use crate::tui::theme;

/// State for the log viewer widget.
pub struct LogViewer {
    /// Lines currently in the buffer.
    lines: Vec<LogLine>,
    /// Path to the daemon log file.
    log_path: PathBuf,
    /// Scroll offset from the bottom (0 = pinned to bottom).
    scroll_offset: usize,
    /// Whether the user has scrolled away from the bottom.
    user_scrolled: bool,
    /// Last known file size for detecting new content.
    last_file_size: u64,
    /// Absolute index of the cursor-selected line (used when `cursor_active`).
    selected: usize,
    /// Whether the per-line cursor is engaged (vs. live tail).
    cursor_active: bool,
    /// Whether the formatted-entry popup is open.
    popup_open: bool,
    /// Scroll offset within the popup.
    popup_scroll: u16,
    /// Filter text (groundwork for future `/` key filtering).
    _filter: Option<String>,
}

impl LogViewer {
    /// Create a new log viewer that reads from the daemon log.
    pub fn new() -> Self {
        let log_dir = wqm_common::paths::get_canonical_log_dir();
        let log_path = log_dir.join("daemon.jsonl");

        let mut viewer = Self {
            lines: Vec::with_capacity(MAX_LINES),
            log_path,
            scroll_offset: 0,
            user_scrolled: false,
            last_file_size: 0,
            selected: 0,
            cursor_active: false,
            popup_open: false,
            popup_scroll: 0,
            _filter: None,
        };
        viewer.load_initial();
        viewer
    }

    /// Load the last `MAX_LINES` lines from the log file.
    fn load_initial(&mut self) {
        let (lines, file_size) = read_tail_lines(&self.log_path, MAX_LINES);
        self.lines = lines;
        self.last_file_size = file_size;
    }

    /// Called on each tick to check for new log lines.
    pub fn on_tick(&mut self) {
        let current_size = file_len(&self.log_path);

        if current_size == self.last_file_size {
            return;
        }

        if current_size < self.last_file_size {
            // File was truncated or rotated; reload from scratch
            self.load_initial();
            if !self.user_scrolled {
                self.scroll_offset = 0;
            }
            return;
        }

        self.append_new_lines(current_size);
        self.last_file_size = current_size;

        // Auto-scroll to bottom when not user-scrolled
        if !self.user_scrolled {
            self.scroll_offset = 0;
        }
    }

    /// Read and append lines written after the last known file position.
    fn append_new_lines(&mut self, current_size: u64) {
        if let Some(new_lines) = read_from_offset(&self.log_path, self.last_file_size) {
            for raw in new_lines {
                let level = Level::from_line(&raw);
                self.lines.push(LogLine { text: raw, level });
            }
            // Trim to MAX_LINES
            if self.lines.len() > MAX_LINES {
                let drain_count = self.lines.len() - MAX_LINES;
                self.lines.drain(..drain_count);
                // Adjust scroll offset if lines were removed
                self.scroll_offset = self.scroll_offset.saturating_sub(drain_count);
            }
        }
        // Update file size regardless so on_tick can set it after this returns.
        // (on_tick sets last_file_size = current_size after calling this)
        let _ = current_size;
    }

    /// Engage the cursor at the latest line if it is not already active.
    fn ensure_cursor(&mut self) {
        if !self.cursor_active {
            self.cursor_active = true;
            self.user_scrolled = true;
            self.selected = self.lines.len().saturating_sub(1);
        }
    }

    /// Move the cursor up (toward older entries) by one line.
    pub fn scroll_up(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        self.ensure_cursor();
        self.selected = self.selected.saturating_sub(1);
    }

    /// Move the cursor down (toward newer entries) by one line.
    pub fn scroll_down(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        self.ensure_cursor();
        self.selected = (self.selected + 1).min(self.lines.len() - 1);
    }

    /// Move the cursor up by a page (visible height).
    pub fn page_up(&mut self, visible_height: usize) {
        if self.lines.is_empty() {
            return;
        }
        self.ensure_cursor();
        self.selected = self.selected.saturating_sub(visible_height);
    }

    /// Move the cursor down by a page (visible height).
    pub fn page_down(&mut self, visible_height: usize) {
        if self.lines.is_empty() {
            return;
        }
        self.ensure_cursor();
        self.selected = (self.selected + visible_height).min(self.lines.len() - 1);
    }

    /// Jump to the bottom (latest entries) and resume live tailing.
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
        self.user_scrolled = false;
        self.cursor_active = false;
    }

    /// Open the formatted-entry popup for the cursor-selected line.
    pub fn open_popup(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        self.ensure_cursor();
        self.popup_open = true;
        self.popup_scroll = 0;
    }

    /// Close the formatted-entry popup.
    pub fn close_popup(&mut self) {
        self.popup_open = false;
        self.popup_scroll = 0;
    }

    /// Whether the popup is currently open.
    pub fn popup_open(&self) -> bool {
        self.popup_open
    }

    /// Scroll the popup contents down.
    pub fn popup_scroll_down(&mut self) {
        self.popup_scroll = self.popup_scroll.saturating_add(1);
    }

    /// Scroll the popup contents up.
    pub fn popup_scroll_up(&mut self) {
        self.popup_scroll = self.popup_scroll.saturating_sub(1);
    }

    /// Render the log viewer into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let inner_height = area.height.saturating_sub(2) as usize; // borders
        if inner_height == 0 || self.lines.is_empty() {
            self.draw_empty(frame, area);
            return;
        }

        let total = self.lines.len();

        // Calculate visible window. In live mode the window is anchored to the
        // bottom; with the cursor active it follows the selected line.
        let (start, end) = if self.cursor_active {
            let sel = self.selected.min(total - 1);
            let start = sel.saturating_sub(inner_height / 2);
            let end = (start + inner_height).min(total);
            let start = end.saturating_sub(inner_height);
            (start, end)
        } else {
            let end = total.saturating_sub(self.scroll_offset);
            (end.saturating_sub(inner_height), end)
        };

        let visible_lines: Vec<Line> = self.lines[start..end]
            .iter()
            .enumerate()
            .map(|(offset, log_line)| {
                let abs = start + offset;
                let level_style = log_line
                    .level
                    .map_or(Style::default().fg(Color::DarkGray), Level::style);
                if self.cursor_active && abs == self.selected.min(total - 1) {
                    // Cursor: row background spans the line; keep the level fg.
                    Line::from(Span::styled(log_line.text.clone(), level_style))
                        .style(theme::selected_row_style())
                } else {
                    Line::from(Span::styled(log_line.text.clone(), level_style))
                }
            })
            .collect();

        let scroll_indicator = if self.cursor_active {
            format!(" Logs [{}/{}] ", self.selected.min(total - 1) + 1, total)
        } else {
            " Logs [live] ".to_string()
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(scroll_indicator)
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        let paragraph = Paragraph::new(visible_lines).block(block);
        frame.render_widget(paragraph, area);

        if self.popup_open {
            self.draw_popup(frame, area);
        }
    }

    /// Draw the formatted-entry popup for the selected log line.
    fn draw_popup(&self, frame: &mut Frame, area: Rect) {
        let Some(log_line) = self
            .lines
            .get(self.selected.min(self.lines.len().saturating_sub(1)))
        else {
            return;
        };

        let popup_w = area.width.saturating_sub(4).min(100);
        let popup_h = area.height.saturating_sub(4).min(40);
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        let pretty = pretty_json(&log_line.text);
        let lines: Vec<Line> = pretty.lines().map(|l| Line::from(l.to_string())).collect();

        let popup = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((self.popup_scroll, 0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Log Entry (j/k scroll, Esc close) ")
                    .title_style(Style::default().add_modifier(Modifier::BOLD))
                    .style(theme::popup_style()),
            );
        frame.render_widget(popup, popup_area);
    }

    /// Render the empty-state placeholder (no lines or zero-height area).
    fn draw_empty(&self, frame: &mut Frame, area: Rect) {
        let empty_msg = if self.log_path.exists() {
            "No log entries found"
        } else {
            "Log file not found — is the daemon running?"
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Logs ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));
        let p = Paragraph::new(empty_msg)
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        frame.render_widget(p, area);
    }
}

/// Pretty-print a JSON log line. Falls back to the raw text when the line is
/// not valid JSON (e.g. a plain-text log).
fn pretty_json(raw: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn scroll_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..50 {
                writeln!(f, r#"{{"level":"INFO","msg":"line {}"}}"#, i).unwrap();
            }
        }

        let mut viewer = LogViewer {
            lines: Vec::new(),
            log_path: PathBuf::from(&path),
            scroll_offset: 0,
            user_scrolled: false,
            last_file_size: 0,
            selected: 0,
            cursor_active: false,
            popup_open: false,
            popup_scroll: 0,
            _filter: None,
        };
        viewer.load_initial();

        assert_eq!(viewer.lines.len(), 50);
        assert!(!viewer.cursor_active);

        // First scroll up engages the cursor at the last line, then moves up.
        viewer.scroll_up();
        assert!(viewer.cursor_active);
        assert_eq!(viewer.selected, 48);

        // Scroll down moves the cursor toward newer entries (clamped at end).
        viewer.scroll_down();
        assert_eq!(viewer.selected, 49);

        // Page up/down move by the visible height, clamped to bounds.
        viewer.page_up(10);
        assert_eq!(viewer.selected, 39);
        viewer.page_down(10);
        assert_eq!(viewer.selected, 49);

        // Opening the popup engages on the selected line.
        viewer.open_popup();
        assert!(viewer.popup_open());
        viewer.popup_scroll_down();
        viewer.close_popup();
        assert!(!viewer.popup_open());

        // Jumping to the bottom resumes live tailing.
        viewer.scroll_up();
        viewer.scroll_to_bottom();
        assert!(!viewer.cursor_active);
        assert_eq!(viewer.scroll_offset, 0);
        assert!(!viewer.user_scrolled);
    }

    #[test]
    fn pretty_json_formats_and_falls_back() {
        let pretty = pretty_json(r#"{"level":"INFO","msg":"hi"}"#);
        assert!(pretty.contains('\n'));
        assert!(pretty.contains("\"msg\""));
        assert_eq!(pretty_json("not json"), "not json");
    }

    #[test]
    fn on_tick_detects_new_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, r#"{{"level":"INFO","msg":"initial"}}"#).unwrap();
        }

        let mut viewer = LogViewer {
            lines: Vec::new(),
            log_path: PathBuf::from(&path),
            scroll_offset: 0,
            user_scrolled: false,
            last_file_size: 0,
            selected: 0,
            cursor_active: false,
            popup_open: false,
            popup_scroll: 0,
            _filter: None,
        };
        viewer.load_initial();
        assert_eq!(viewer.lines.len(), 1);

        // Append a new line
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(f, r#"{{"level":"WARN","msg":"appended"}}"#).unwrap();
        }

        viewer.on_tick();
        assert_eq!(viewer.lines.len(), 2);
        assert!(viewer.lines[1].text.contains("appended"));
        assert_eq!(viewer.lines[1].level, Some(Level::Warn));
    }

    #[test]
    fn on_tick_handles_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..10 {
                writeln!(f, r#"{{"level":"INFO","msg":"line {}"}}"#, i).unwrap();
            }
        }

        let mut viewer = LogViewer {
            lines: Vec::new(),
            log_path: PathBuf::from(&path),
            scroll_offset: 0,
            user_scrolled: false,
            last_file_size: 0,
            selected: 0,
            cursor_active: false,
            popup_open: false,
            popup_scroll: 0,
            _filter: None,
        };
        viewer.load_initial();
        assert_eq!(viewer.lines.len(), 10);

        // Truncate and write fewer lines
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, r#"{{"level":"INFO","msg":"fresh"}}"#).unwrap();
        }

        viewer.on_tick();
        assert_eq!(viewer.lines.len(), 1);
        assert!(viewer.lines[0].text.contains("fresh"));
    }
}
