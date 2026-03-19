//! Log viewer widget with tail behavior and color-coded log levels.
//!
//! Reads the daemon log file and displays the last N lines with
//! auto-scroll to bottom. Supports scrolling with j/k and arrow keys.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

/// Maximum number of lines to retain in the viewer buffer.
const MAX_LINES: usize = 200;

/// A parsed log line with its detected level for color rendering.
#[derive(Debug, Clone)]
struct LogLine {
    /// The raw text of the log line.
    text: String,
    /// Detected log level, if any.
    level: Option<Level>,
}

/// Log levels for color-coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    /// Parse the level from a JSONL log line by looking for `"level":` field.
    fn from_line(line: &str) -> Option<Self> {
        // Fast path: look for "level" key in JSON
        let level_val = extract_json_level(line)?;
        match level_val.to_uppercase().as_str() {
            "ERROR" | "ERR" | "FATAL" => Some(Self::Error),
            "WARN" | "WARNING" => Some(Self::Warn),
            "INFO" => Some(Self::Info),
            "DEBUG" => Some(Self::Debug),
            "TRACE" => Some(Self::Trace),
            _ => {
                // Numeric pino levels
                if let Ok(n) = level_val.parse::<u64>() {
                    match n {
                        50.. => Some(Self::Error),
                        40..=49 => Some(Self::Warn),
                        30..=39 => Some(Self::Info),
                        20..=29 => Some(Self::Debug),
                        _ => Some(Self::Trace),
                    }
                } else {
                    None
                }
            }
        }
    }

    fn style(self) -> Style {
        match self {
            Self::Error => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Self::Warn => Style::default().fg(Color::Yellow),
            Self::Info => Style::default().fg(Color::Reset),
            Self::Debug => Style::default().fg(Color::DarkGray),
            Self::Trace => Style::default().fg(Color::DarkGray),
        }
    }
}

/// Extract the string value of the "level" JSON field without pulling in
/// a full JSON parser. Handles both `"level":"INFO"` and `"level":30`.
fn extract_json_level(line: &str) -> Option<String> {
    let key = "\"level\"";
    let idx = line.find(key)?;
    let after_key = &line[idx + key.len()..];
    // Skip optional whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let trimmed = after_colon.trim_start();

    if trimmed.starts_with('"') {
        // String value: "level":"INFO"
        let end = trimmed[1..].find('"')?;
        Some(trimmed[1..1 + end].to_string())
    } else {
        // Numeric value: "level":30
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(trimmed.len());
        if end == 0 {
            return None;
        }
        Some(trimmed[..end].to_string())
    }
}

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

        // Read only the new bytes appended since last check
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

        self.last_file_size = current_size;

        // Auto-scroll to bottom when not user-scrolled
        if !self.user_scrolled {
            self.scroll_offset = 0;
        }
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
}

/// Read the last `n` lines from a file. Returns the lines and the file size.
fn read_tail_lines(path: &PathBuf, n: usize) -> (Vec<LogLine>, u64) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return (Vec::new(), 0),
    };

    let metadata = match file.metadata() {
        Ok(m) => m,
        Err(_) => return (Vec::new(), 0),
    };
    let file_size = metadata.len();

    let reader = BufReader::new(file);
    let mut all_lines: Vec<LogLine> = Vec::new();

    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let level = Level::from_line(&line);
        all_lines.push(LogLine { text: line, level });
    }

    // Keep only last n
    if all_lines.len() > n {
        let start = all_lines.len() - n;
        all_lines.drain(..start);
    }

    (all_lines, file_size)
}

/// Read new lines appended to the file starting from `offset`.
fn read_from_offset(path: &PathBuf, offset: u64) -> Option<Vec<String>> {
    let mut file = File::open(path).ok()?;
    file.seek(SeekFrom::Start(offset)).ok()?;

    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    Some(lines)
}

/// Get the file length, returning 0 if the file does not exist.
fn file_len(path: &PathBuf) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn level_from_json_string() {
        let line = r#"{"timestamp":"2025-01-01T00:00:00Z","level":"ERROR","msg":"fail"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Error));
    }

    #[test]
    fn level_from_json_warn() {
        let line = r#"{"level":"WARN","msg":"caution"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Warn));
    }

    #[test]
    fn level_from_json_info() {
        let line = r#"{"level":"INFO","msg":"ok"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Info));
    }

    #[test]
    fn level_from_json_debug() {
        let line = r#"{"level":"DEBUG","msg":"verbose"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Debug));
    }

    #[test]
    fn level_from_numeric_pino() {
        let line = r#"{"level":50,"msg":"error"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Error));

        let line = r#"{"level":40,"msg":"warn"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Warn));

        let line = r#"{"level":30,"msg":"info"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Info));

        let line = r#"{"level":20,"msg":"debug"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Debug));

        let line = r#"{"level":10,"msg":"trace"}"#;
        assert_eq!(Level::from_line(line), Some(Level::Trace));
    }

    #[test]
    fn level_from_unknown_returns_none() {
        let line = r#"{"msg":"no level here"}"#;
        assert_eq!(Level::from_line(line), None);
    }

    #[test]
    fn extract_json_level_with_spaces() {
        let line = r#"{ "level" : "INFO" , "msg": "ok" }"#;
        assert_eq!(extract_json_level(line), Some("INFO".to_string()));
    }

    #[test]
    fn read_tail_lines_from_tempfile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..10 {
                writeln!(f, r#"{{"level":"INFO","msg":"line {}"}}"#, i).unwrap();
            }
        }

        let pb = PathBuf::from(&path);
        let (lines, size) = read_tail_lines(&pb, 5);
        assert_eq!(lines.len(), 5);
        assert!(size > 0);
        // Should contain the last 5 lines
        assert!(lines[0].text.contains("line 5"));
        assert!(lines[4].text.contains("line 9"));
    }

    #[test]
    fn read_from_offset_gets_new_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let pb = PathBuf::from(&path);

        // Write initial content
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, r#"{{"level":"INFO","msg":"first"}}"#).unwrap();
        }
        let initial_size = file_len(&pb);

        // Append more content
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(f, r#"{{"level":"WARN","msg":"second"}}"#).unwrap();
        }

        let new_lines = read_from_offset(&pb, initial_size).unwrap();
        assert_eq!(new_lines.len(), 1);
        assert!(new_lines[0].contains("second"));
    }

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
