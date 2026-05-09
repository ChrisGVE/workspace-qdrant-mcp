//! Log viewer widget with tail behavior and color-coded log levels.
//!
//! Reads the daemon log file and displays the last N lines with
//! auto-scroll to bottom. Supports scrolling with j/k and arrow keys.

use std::path::PathBuf;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::logs_data::{file_len, read_from_offset, read_tail_lines, Level, LogLine, MAX_LINES};

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

    /// Scroll up by one line.
    pub fn scroll_up(&mut self) {
        if self.scroll_offset < self.lines.len().saturating_sub(1) {
            self.scroll_offset += 1;
            self.user_scrolled = true;
        }
    }

    /// Scroll down by one line.
    pub fn scroll_down(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
        if self.scroll_offset == 0 {
            self.user_scrolled = false;
        }
    }

    /// Scroll up by a page (visible height).
    pub fn page_up(&mut self, visible_height: usize) {
        let max = self.lines.len().saturating_sub(1);
        self.scroll_offset = (self.scroll_offset + visible_height).min(max);
        self.user_scrolled = true;
    }

    /// Scroll down by a page (visible height).
    pub fn page_down(&mut self, visible_height: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(visible_height);
        if self.scroll_offset == 0 {
            self.user_scrolled = false;
        }
    }

    /// Jump to the bottom (latest entries).
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
        self.user_scrolled = false;
    }

    /// Render the log viewer into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let inner_height = area.height.saturating_sub(2) as usize; // borders
        if inner_height == 0 || self.lines.is_empty() {
            self.draw_empty(frame, area);
            return;
        }

        // Calculate visible window
        let total = self.lines.len();
        let end = total.saturating_sub(self.scroll_offset);
        let start = end.saturating_sub(inner_height);

        let visible_lines: Vec<Line> = self.lines[start..end]
            .iter()
            .map(|log_line| {
                let style = log_line
                    .level
                    .map_or(Style::default().fg(Color::DarkGray), Level::style);
                Line::from(Span::styled(&log_line.text, style))
            })
            .collect();

        let scroll_indicator = if self.scroll_offset > 0 {
            format!(" Logs [{} from bottom] ", self.scroll_offset)
        } else {
            " Logs [live] ".to_string()
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(scroll_indicator)
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        let paragraph = Paragraph::new(visible_lines).block(block);
        frame.render_widget(paragraph, area);
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
            _filter: None,
        };
        viewer.load_initial();

        assert_eq!(viewer.lines.len(), 50);
        assert_eq!(viewer.scroll_offset, 0);
        assert!(!viewer.user_scrolled);

        // Scroll up
        viewer.scroll_up();
        assert_eq!(viewer.scroll_offset, 1);
        assert!(viewer.user_scrolled);

        // Scroll down back to bottom
        viewer.scroll_down();
        assert_eq!(viewer.scroll_offset, 0);
        assert!(!viewer.user_scrolled);

        // Page up
        viewer.page_up(10);
        assert_eq!(viewer.scroll_offset, 10);
        assert!(viewer.user_scrolled);

        // Page down
        viewer.page_down(10);
        assert_eq!(viewer.scroll_offset, 0);
        assert!(!viewer.user_scrolled);

        // Scroll to bottom after scrolling up
        viewer.scroll_up();
        viewer.scroll_up();
        viewer.scroll_to_bottom();
        assert_eq!(viewer.scroll_offset, 0);
        assert!(!viewer.user_scrolled);
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
