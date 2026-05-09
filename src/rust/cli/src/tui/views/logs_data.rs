//! Data types and file I/O logic for the log viewer.
//!
//! Separated from the view module to keep both files under the 500-line limit
//! and to allow unit-testing parsing logic independently from rendering.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;

use ratatui::style::{Color, Modifier, Style};

/// Maximum number of lines to retain in the viewer buffer.
pub(super) const MAX_LINES: usize = 200;

/// A parsed log line with its detected level for color rendering.
#[derive(Debug, Clone)]
pub(super) struct LogLine {
    /// The raw text of the log line.
    pub(super) text: String,
    /// Detected log level, if any.
    pub(super) level: Option<Level>,
}

/// Log levels for color-coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    /// Parse the level from a JSONL log line by looking for `"level":` field.
    pub(super) fn from_line(line: &str) -> Option<Self> {
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

    pub(super) fn style(self) -> Style {
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
pub(super) fn extract_json_level(line: &str) -> Option<String> {
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

/// Read the last `n` lines from a file. Returns the lines and the file size.
pub(super) fn read_tail_lines(path: &PathBuf, n: usize) -> (Vec<LogLine>, u64) {
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
pub(super) fn read_from_offset(path: &PathBuf, offset: u64) -> Option<Vec<String>> {
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
pub(super) fn file_len(path: &PathBuf) -> u64 {
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
}
