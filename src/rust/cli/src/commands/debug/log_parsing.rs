//! Log parsing infrastructure for debug commands
//!
//! Provides log level parsing, log entry extraction, filtering,
//! and log file discovery (including rotated files).

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Returns the canonical OS-specific log directory for workspace-qdrant logs.
///
/// Delegates to `wqm_common::paths::get_canonical_log_dir()`.
pub fn get_canonical_log_dir() -> PathBuf {
    wqm_common::paths::get_canonical_log_dir()
}

/// Parse a relative time string (e.g. "1h", "30m", "2d", "10s") into a chrono Duration.
pub fn parse_relative_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("Empty time string");
    }

    // Try to split into numeric part and unit
    let (num_str, unit) = if s.ends_with("ms") {
        (&s[..s.len() - 2], "ms")
    } else {
        let split_pos = s
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(s.len());
        (&s[..split_pos], &s[split_pos..])
    };

    let num: f64 = num_str
        .parse()
        .with_context(|| format!("Invalid number in time string: '{}'", s))?;

    if num < 0.0 {
        anyhow::bail!("Negative time value: '{}'", s);
    }

    match unit {
        "s" | "sec" | "secs" => Ok(Duration::seconds(num as i64)),
        "m" | "min" | "mins" => Ok(Duration::minutes(num as i64)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Ok(Duration::hours(num as i64)),
        "d" | "day" | "days" => Ok(Duration::days(num as i64)),
        "w" | "week" | "weeks" => Ok(Duration::weeks(num as i64)),
        "ms" => Ok(Duration::milliseconds(num as i64)),
        "" => {
            // Default to seconds if no unit
            Ok(Duration::seconds(num as i64))
        }
        _ => anyhow::bail!("Unknown time unit '{}' in '{}'", unit, s),
    }
}

/// Parse a --since argument into a UTC cutoff timestamp.
///
/// Accepts:
/// - Relative: "1h", "30m", "2d", "10s", "1w"
/// - Absolute: ISO 8601 / RFC 3339 timestamps
pub fn parse_since(value: &str) -> Result<DateTime<Utc>> {
    // Try relative first (starts with a digit)
    if value
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        if let Ok(dur) = parse_relative_duration(value) {
            return Ok(Utc::now() - dur);
        }
    }

    // Try ISO 8601 / RFC 3339
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Try without timezone (assume UTC)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.and_utc());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc());
    }

    anyhow::bail!(
        "Cannot parse '{}'. Use relative (e.g. '1h', '30m', '2d') or ISO 8601 timestamp.",
        value
    )
}

/// Log levels ordered by severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// Parse a log level from a JSON value (string or numeric).
    pub fn from_json(val: &serde_json::Value) -> Option<Self> {
        match val {
            serde_json::Value::String(s) => Self::from_str(s),
            serde_json::Value::Number(n) => {
                // pino numeric levels: 10=trace, 20=debug, 30=info, 40=warn, 50=error
                let num = n.as_u64()?;
                match num {
                    0..=10 => Some(LogLevel::Trace),
                    11..=20 => Some(LogLevel::Debug),
                    21..=30 => Some(LogLevel::Info),
                    31..=40 => Some(LogLevel::Warn),
                    _ => Some(LogLevel::Error),
                }
            }
            _ => None,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" | "ERR" | "FATAL" => Some(LogLevel::Error),
            _ => None,
        }
    }

    pub fn is_error_or_warn(self) -> bool {
        matches!(self, LogLevel::Error | LogLevel::Warn)
    }
}

/// Represents a parsed log entry with timestamp and component info.
#[derive(Debug)]
pub struct LogEntry {
    pub timestamp: String,
    pub parsed_time: Option<DateTime<Utc>>,
    pub level: Option<LogLevel>,
    pub component: String,
    pub session_id: Option<String>,
    pub raw_line: String,
}

impl LogEntry {
    /// Try to parse a JSON log line, extracting timestamp, level, and session_id.
    pub fn from_json_line(line: &str, component: &str) -> Option<Self> {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let timestamp = json
                .get("timestamp")
                .or_else(|| json.get("time"))
                .or_else(|| json.get("ts"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let parsed_time = if !timestamp.is_empty() {
                DateTime::parse_from_rfc3339(&timestamp)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            } else {
                None
            };

            let level = json
                .get("level")
                .or_else(|| json.get("severity"))
                .and_then(LogLevel::from_json);

            let session_id = json
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            Some(LogEntry {
                timestamp,
                parsed_time,
                level,
                component: component.to_string(),
                session_id,
                raw_line: line.to_string(),
            })
        } else {
            // Non-JSON line
            Some(LogEntry {
                timestamp: String::new(),
                parsed_time: None,
                level: None,
                component: component.to_string(),
                session_id: None,
                raw_line: line.to_string(),
            })
        }
    }
}

/// Log entry filter criteria.
pub struct LogFilter {
    pub errors_only: bool,
    pub since: Option<DateTime<Utc>>,
    pub session: Option<String>,
}

impl LogFilter {
    pub fn matches(&self, entry: &LogEntry) -> bool {
        if self.errors_only && !entry.level.map(|l| l.is_error_or_warn()).unwrap_or(false) {
            return false;
        }
        if let Some(cutoff) = &self.since {
            if let Some(ts) = &entry.parsed_time {
                if ts < cutoff {
                    return false;
                }
            }
            // Entries without a parseable timestamp are included (conservative)
        }
        if let Some(ref sid) = self.session {
            if !entry
                .session_id
                .as_ref()
                .map(|s| s.contains(sid))
                .unwrap_or(false)
            {
                return false;
            }
        }
        true
    }
}

/// Discover log files for a component, including rotated files.
///
/// Returns files sorted newest-first. Matches patterns:
/// - `<base>.jsonl` (current)
/// - `<base>.jsonl.1`, `<base>.jsonl.2`, ... (rotated by logroller)
pub fn discover_log_files(log_dir: &Path, base_name: &str) -> Vec<PathBuf> {
    let current = log_dir.join(format!("{}.jsonl", base_name));
    let mut files: Vec<PathBuf> = Vec::new();

    if current.exists() {
        files.push(current.clone());
    }

    // Look for rotated files: <base>.jsonl.1, <base>.jsonl.2, ...
    if let Ok(entries) = fs::read_dir(log_dir) {
        let prefix = format!("{}.jsonl.", base_name);
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && entry.path() != current {
                // Check that suffix is numeric (not .gz etc. which we can't read yet)
                let suffix = &name_str[prefix.len()..];
                if suffix.chars().all(|c| c.is_ascii_digit()) {
                    files.push(entry.path());
                }
            }
        }
    }

    // Sort: current file first, then rotated by number (ascending = oldest first)
    // We want newest-first for efficient tail-N reading
    files.sort_by(|a, b| {
        let num_a = rotated_index(a);
        let num_b = rotated_index(b);
        num_a.cmp(&num_b)
    });

    files
}

/// Extract rotated file index (0 for current, N for .N suffix).
fn rotated_index(path: &Path) -> u32 {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Read log entries from a file with streaming filter, returning last N matching entries.
pub fn read_log_file_filtered(
    path: &Path,
    component: &str,
    max_lines: usize,
    filter: &LogFilter,
) -> Vec<LogEntry> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = BufReader::new(file);
    // Use a ring buffer to keep only the last max_lines matching entries
    let mut ring: Vec<LogEntry> = Vec::with_capacity(max_lines + 1);

    for line in reader.lines().filter_map(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(entry) = LogEntry::from_json_line(&line, component) {
            if filter.matches(&entry) {
                ring.push(entry);
                if ring.len() > max_lines {
                    ring.remove(0);
                }
            }
        }
    }

    ring
}

/// Read from multiple log files (current + rotated) respecting filters.
pub fn read_log_files_filtered(
    log_dir: &Path,
    base_name: &str,
    component: &str,
    max_lines: usize,
    filter: &LogFilter,
) -> Vec<LogEntry> {
    let files = discover_log_files(log_dir, base_name);

    if files.is_empty() {
        return Vec::new();
    }

    // If no --since filter, only read current file for performance
    if filter.since.is_none() {
        if let Some(current) = files.first() {
            return read_log_file_filtered(current, component, max_lines, filter);
        }
        return Vec::new();
    }

    // With --since, we may need to read rotated files too (oldest first)
    let mut all: Vec<LogEntry> = Vec::new();
    // Read files in reverse order (oldest first) so we can accumulate chronologically
    for file in files.iter().rev() {
        let entries = read_log_file_filtered(file, component, max_lines, filter);
        all.extend(entries);
    }

    // Take last N
    if all.len() > max_lines {
        all.drain(..all.len() - max_lines);
    }

    all
}
