//! Pure string and value formatting utilities.
//!
//! Provides ANSI stripping, safe string truncation, date extraction, and
//! human-readable size/duration formatters.  All functions are pure
//! (no I/O, no side effects).

use ansi_str::AnsiStr;

/// Strip ANSI escape sequences from a string.
///
/// Uses the `ansi-str` crate to remove all terminal formatting codes,
/// producing plain text suitable for machine consumption.
pub fn strip_ansi(s: &str) -> String {
    s.ansi_strip().to_string()
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
    if ts.len() >= 10 && ts.is_char_boundary(10) {
        ts[..10].to_string()
    } else {
        ts.to_string()
    }
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
    fn test_strip_ansi_plain() {
        assert_eq!(strip_ansi("hello world"), "hello world");
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn test_strip_ansi_colored() {
        // "\x1b[31mfailed\x1b[0m" is red "failed"
        assert_eq!(strip_ansi("\x1b[31mfailed\x1b[0m"), "failed");
        // Bold + green
        assert_eq!(strip_ansi("\x1b[1;32mdone\x1b[0m"), "done");
    }

    #[test]
    fn test_strip_ansi_mixed() {
        let s = "prefix \x1b[33myellow\x1b[0m suffix";
        assert_eq!(strip_ansi(s), "prefix yellow suffix");
    }
}
