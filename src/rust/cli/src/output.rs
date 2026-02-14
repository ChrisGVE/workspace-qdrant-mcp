//! Output formatting module
//!
//! Provides JSON and table output formatting with colored terminal support.
//! Used across all CLI commands for consistent output.
//!
//! Note: Some formatting functions are infrastructure for future CLI commands.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fmt::Display;

use colored::Colorize;
use serde::Serialize;
use tabled::settings::peaker::{Peaker, PriorityMax};
use tabled::settings::{Style, Width};
use tabled::{Table, Tabled};

use crate::config::OutputFormat;

// ─── Column layout hints ─────────────────────────────────────────────────

thread_local! {
    /// Content column indices for custom peakers.
    static CONTENT_COLUMNS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// Content-aware minimum widths per column (computed from data).
    static COLUMN_MIN_WIDTHS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Peaker for `Width::wrap`: shrinks categorical columns first.
///
/// When the table exceeds terminal width, this shrinks the widest
/// non-content (categorical) column first. Only falls back to shrinking
/// content columns when all categorical columns are at minimum width.
#[derive(Debug, Default, Clone)]
struct ShrinkCategoricalFirst;

impl Peaker for ShrinkCategoricalFirst {
    fn create() -> Self {
        Self
    }

    fn peak(&mut self, _min_widths: &[usize], widths: &[usize]) -> Option<usize> {
        CONTENT_COLUMNS.with(|cc| {
            COLUMN_MIN_WIDTHS.with(|cmw| {
                let content_cols = cc.borrow();
                let col_mins = cmw.borrow();

                // Phase 1: shrink widest categorical column above its content minimum
                let cat = (0..widths.len())
                    .filter(|i| !content_cols.contains(i))
                    .filter(|&i| widths[i] > col_mins.get(i).copied().unwrap_or(0))
                    .max_by_key(|&i| widths[i]);
                if cat.is_some() {
                    return cat;
                }

                // Phase 2: shrink widest content column above its minimum
                let content = (0..widths.len())
                    .filter(|i| content_cols.contains(i))
                    .filter(|&i| widths[i] > col_mins.get(i).copied().unwrap_or(0))
                    .max_by_key(|&i| widths[i]);
                if content.is_some() {
                    return content;
                }

                // Phase 3: last resort — shrink any column above 1 char
                (0..widths.len())
                    .filter(|&i| widths[i] > 1)
                    .max_by_key(|&i| widths[i])
            })
        })
    }
}

/// Peaker for `Width::increase`: expands only content columns.
///
/// When the table is narrower than the terminal, this distributes extra
/// space exclusively to content columns, picking the narrowest one each
/// time for even distribution. Falls back to `PriorityMax` if no content
/// columns are configured.
#[derive(Debug, Default, Clone)]
struct ExpandContentOnly;

impl Peaker for ExpandContentOnly {
    fn create() -> Self {
        Self
    }

    fn peak(&mut self, _min_widths: &[usize], widths: &[usize]) -> Option<usize> {
        CONTENT_COLUMNS.with(|cc| {
            let content_cols = cc.borrow();

            if content_cols.is_empty() {
                // No hints — fall back to widest column
                return (0..widths.len()).max_by_key(|&i| widths[i]);
            }

            // Expand the narrowest content column (for even distribution)
            content_cols
                .iter()
                .filter(|&&i| i < widths.len())
                .min_by_key(|&&i| widths[i])
                .copied()
        })
    }
}

/// Format and print a success message
pub fn success(message: impl Display) {
    println!("{} {}", "✓".green(), message);
}

/// Format and print an error message
pub fn error(message: impl Display) {
    eprintln!("{} {}", "✗".red(), message);
}

/// Format and print a warning message
pub fn warning(message: impl Display) {
    eprintln!("{} {}", "!".yellow(), message);
}

/// Format and print an info message
pub fn info(message: impl Display) {
    println!("{} {}", "ℹ".blue(), message);
}

/// Format and print data based on output format
pub fn print_data<T: Serialize + Tabled>(data: &[T], format: OutputFormat) {
    match format {
        OutputFormat::Json => print_json(data),
        OutputFormat::Table => print_table(data),
        OutputFormat::Plain => print_plain(data),
    }
}

/// Print data as JSON
pub fn print_json<T: Serialize + ?Sized>(data: &T) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => println!("{}", json),
        Err(e) => error(format!("Failed to serialize to JSON: {}", e)),
    }
}

/// Print data as a formatted table that fills the full terminal width.
///
/// 1. Wraps content to fit within terminal width (`PriorityMax` shrinks the
///    widest column first, `keep_words` breaks at word/comma boundaries).
/// 2. Stretches the table to fill the full terminal width, giving extra
///    space to the widest columns (text/content) via `PriorityMax`.
pub fn print_table<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();
    let table = Table::new(data)
        .with(Style::rounded())
        .with(Width::wrap(width).priority::<PriorityMax>().keep_words())
        .with(Width::increase(width).priority::<PriorityMax>())
        .to_string();
    println!("{}", table);
}

/// Cell padding used by `Style::rounded()` (1 space each side).
const CELL_PADDING: usize = 2;

/// Compute content-aware minimum widths for each column.
///
/// For categorical columns: minimum = max(header width, widest word/segment)
/// where words are split at commas and whitespace boundaries across all cells.
/// This ensures categorical columns never shrink below their widest atomic value.
///
/// For content columns: minimum = header width (flexible, gets extra space).
///
/// The returned widths include `CELL_PADDING` to match tabled's internal
/// column width representation (content + padding).
fn compute_column_min_widths<T: Tabled>(data: &[T], content_columns: &[usize]) -> Vec<usize> {
    let headers = T::headers();
    let num_cols = headers.len();
    let mut min_widths = vec![0usize; num_cols];

    // Start with header widths for all columns
    for (i, header) in headers.iter().enumerate() {
        min_widths[i] = header.len();
    }

    // For categorical columns, find widest word/segment across all cells.
    // Split on whitespace only — this matches how tabled's `split_keeping_words`
    // processes text, so trailing punctuation (commas, semicolons) is included
    // in the word width measurement.
    for row in data {
        let fields = row.fields();
        for (i, field) in fields.iter().enumerate() {
            if i >= num_cols || content_columns.contains(&i) {
                continue; // Content columns keep header width as min
            }
            let widest = field
                .split_whitespace()
                .map(|s| s.len())
                .max()
                .unwrap_or(0);
            min_widths[i] = min_widths[i].max(widest);
        }
    }

    // Add cell padding — tabled's peaker widths include padding
    for w in &mut min_widths {
        *w += CELL_PADDING;
    }

    min_widths
}

/// Print a table with column layout hints for optimal width distribution.
///
/// `content_columns` lists 0-based indices of columns that contain
/// variable-length text (titles, descriptions, paths). These columns
/// receive the most available width. All other columns are treated as
/// categorical (IDs, dates, enums, comma-delimited tags) and are
/// constrained to the width of their widest atomic value (word/segment).
///
/// Extra space is distributed exclusively to content columns.
pub fn print_table_with_hints<T: Tabled>(data: &[T], content_columns: &[usize]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();

    // Compute content-aware minimums and set thread-locals for peakers
    let col_mins = compute_column_min_widths(data, content_columns);
    CONTENT_COLUMNS.with(|cc| {
        *cc.borrow_mut() = content_columns.to_vec();
    });
    COLUMN_MIN_WIDTHS.with(|cmw| {
        *cmw.borrow_mut() = col_mins;
    });

    let table = Table::new(data)
        .with(Style::rounded())
        .with(
            Width::wrap(width)
                .priority::<ShrinkCategoricalFirst>()
                .keep_words(),
        )
        .with(Width::increase(width).priority::<ExpandContentOnly>())
        .to_string();

    // Clean up thread-locals
    CONTENT_COLUMNS.with(|cc| cc.borrow_mut().clear());
    COLUMN_MIN_WIDTHS.with(|cmw| cmw.borrow_mut().clear());

    println!("{}", table);
}

/// Trait for structs that know which of their columns are content columns.
///
/// Implement this on `Tabled` structs to enable automatic layout hints
/// via [`print_table_auto`].
pub trait ColumnHints {
    /// Returns 0-based indices of content columns (variable-length text).
    /// Empty slice means all columns are categorical (no special treatment).
    fn content_columns() -> &'static [usize];
}

/// Print a table with layout hints derived from the `ColumnHints` trait.
///
/// Delegates to [`print_table_with_hints`] when content columns exist,
/// or [`print_table`] when all columns are categorical.
pub fn print_table_auto<T: Tabled + ColumnHints>(data: &[T]) {
    let hints = T::content_columns();
    if hints.is_empty() {
        print_table(data);
    } else {
        print_table_with_hints(data, hints);
    }
}

/// Get the current terminal width, falling back to 120 columns
pub fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(120)
}

/// Truncate a string to a maximum display width, appending "..." if truncated.
///
/// UTF-8 safe: finds valid char boundaries before slicing.
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let target = max_len.saturating_sub(3);
        // Find a valid char boundary at or before target
        let mut boundary = target;
        while boundary > 0 && !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}...", &s[..boundary])
    }
}

/// Format a timestamp for table display (date only).
///
/// Extracts the date portion (YYYY-MM-DD) from an ISO-8601 timestamp.
pub fn format_date(ts: &str) -> String {
    if ts.len() >= 10 {
        ts[..10].to_string()
    } else {
        ts.to_string()
    }
}

/// Print data as plain text (one item per line)
pub fn print_plain<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        return;
    }

    // Use table but with minimal style
    let table = Table::new(data).with(Style::blank()).to_string();
    println!("{}", table);
}

/// Print a single key-value pair
pub fn kv(key: impl Display, value: impl Display) {
    println!("{}: {}", key.to_string().bold(), value);
}

/// Print a status line with colored status indicator
pub fn status_line(label: impl Display, status: ServiceStatus) {
    let (indicator, color_status) = match status {
        ServiceStatus::Healthy => ("●".green(), "healthy".green()),
        ServiceStatus::Degraded => ("●".yellow(), "degraded".yellow()),
        ServiceStatus::Unhealthy => ("●".red(), "unhealthy".red()),
        ServiceStatus::Unknown => ("○".dimmed(), "unknown".dimmed()),
    };
    println!("{} {}: {}", indicator, label, color_status);
}

/// Service health status for colored output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl ServiceStatus {
    /// Create from proto ServiceStatus enum value
    pub fn from_proto(value: i32) -> Self {
        match value {
            1 => Self::Healthy,
            2 => Self::Degraded,
            3 => Self::Unhealthy,
            _ => Self::Unknown,
        }
    }
}

/// Prompt user for yes/no confirmation, defaulting to No.
///
/// Returns true only if the user types "y" or "yes" (case-insensitive).
pub fn confirm(message: &str) -> bool {
    use std::io::{self, Write};
    print!("{} [y/N] ", message);
    io::stdout().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Print a horizontal separator
pub fn separator() {
    println!("{}", "─".repeat(60).dimmed());
}

/// Print a section header
pub fn section(title: impl Display) {
    println!("\n{}", title.to_string().bold().underline());
}

/// Format bytes as human-readable size
pub fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string
pub fn format_duration(seconds: u64) -> String {
    if seconds >= 86400 {
        format!("{}d {}h", seconds / 86400, (seconds % 86400) / 3600)
    } else if seconds >= 3600 {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3700), "1h 1m");
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("exactly ten", 11), "exactly ten");
    }

    #[test]
    fn test_truncate_long_string() {
        let long = "this is a long string that should be truncated";
        let result = truncate(long, 20);
        assert!(result.len() <= 20);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_multibyte() {
        // Emoji are multi-byte; ensure no panic
        let emoji = "Hello 🌍🌎🌏 world";
        let result = truncate(emoji, 10);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_format_date_iso() {
        assert_eq!(format_date("2026-02-12T10:30:00.000Z"), "2026-02-12");
    }

    #[test]
    fn test_format_date_short() {
        assert_eq!(format_date("short"), "short");
        assert_eq!(format_date(""), "");
    }

    #[test]
    fn test_service_status_from_proto() {
        assert_eq!(ServiceStatus::from_proto(1), ServiceStatus::Healthy);
        assert_eq!(ServiceStatus::from_proto(2), ServiceStatus::Degraded);
        assert_eq!(ServiceStatus::from_proto(3), ServiceStatus::Unhealthy);
        assert_eq!(ServiceStatus::from_proto(0), ServiceStatus::Unknown);
        assert_eq!(ServiceStatus::from_proto(99), ServiceStatus::Unknown);
    }

    /// Test struct for column min width computation
    #[derive(Tabled)]
    struct TestRow {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Title")]
        title: String,
        #[tabled(rename = "Tags")]
        tags: String,
    }

    #[test]
    fn test_compute_column_min_widths_header_floor() {
        // When cells are shorter than headers, header width + padding is the minimum
        let data = vec![TestRow {
            id: "1".into(),
            title: "Hi".into(),
            tags: "a".into(),
        }];
        // Title(1) is content column
        let mins = compute_column_min_widths(&data, &[1]);
        // All values include CELL_PADDING (2)
        assert_eq!(mins[0], 2 + CELL_PADDING); // "ID" header = 2
        assert_eq!(mins[1], 5 + CELL_PADDING); // "Title" header = 5 (content col)
        assert_eq!(mins[2], 4 + CELL_PADDING); // "Tags" header = 4 > "a" = 1
    }

    #[test]
    fn test_compute_column_min_widths_whitespace_split() {
        // Splits on whitespace only — trailing punctuation stays with the word
        let data = vec![TestRow {
            id: "abc".into(),
            title: "A long title".into(),
            tags: "short, verylongtag, mid".into(),
        }];
        let mins = compute_column_min_widths(&data, &[1]);
        assert_eq!(mins[0], 3 + CELL_PADDING); // "abc" > "ID"(2)
        assert_eq!(mins[1], 5 + CELL_PADDING); // content column, stays at header width
        // "verylongtag," (with trailing comma) = 12 > "Tags"(4)
        assert_eq!(mins[2], 12 + CELL_PADDING);
    }

    #[test]
    fn test_compute_column_min_widths_across_rows() {
        // Min width should be the max across all rows
        let data = vec![
            TestRow { id: "1".into(), title: "x".into(), tags: "a, b".into() },
            TestRow { id: "42".into(), title: "y".into(), tags: "longvalue,".into() },
            TestRow { id: "7".into(), title: "z".into(), tags: "c".into() },
        ];
        let mins = compute_column_min_widths(&data, &[1]);
        assert_eq!(mins[0], 2 + CELL_PADDING); // "42" = 2, "ID" = 2
        // "longvalue," = 10 (no whitespace, so entire string is one word)
        assert_eq!(mins[2], 10 + CELL_PADDING);
    }
}
