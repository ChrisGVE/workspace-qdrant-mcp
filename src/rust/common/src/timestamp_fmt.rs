//! Localized timestamp formatting for CLI output.
//!
//! Converts UTC timestamps (from SQLite/Qdrant) to local timezone for display.
//! Internal storage remains UTC; this is purely for human-readable output.

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};

/// Parse a UTC timestamp string and format it in the user's local timezone.
///
/// Accepts:
/// - RFC 3339 / ISO 8601 with `Z` suffix: `2026-02-17T19:46:47Z`
/// - RFC 3339 with offset: `2026-02-17T19:46:47+00:00`
/// - SQLite datetime: `2026-02-17 19:46:47`
/// - SQLite datetime with fractional: `2026-02-17 19:46:47.123`
///
/// Returns the timestamp formatted as `YYYY-MM-DD HH:MM:SS` in local time.
/// On parse failure, returns the input unchanged.
pub fn format_local(utc_str: &str) -> String {
    // Try RFC 3339 first (handles both Z and +00:00)
    if let Ok(dt) = DateTime::parse_from_rfc3339(utc_str) {
        let local = dt.with_timezone(&Local);
        return local.format("%Y-%m-%d %H:%M:%S").to_string();
    }

    // Try SQLite datetime format (with optional fractional seconds)
    let trimmed = utc_str.trim();
    for fmt in &["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, fmt) {
            let utc_dt: DateTime<Utc> = Utc.from_utc_datetime(&naive);
            let local = utc_dt.with_timezone(&Local);
            return local.format("%Y-%m-%d %H:%M:%S").to_string();
        }
    }

    // Fallback: return as-is
    utc_str.to_string()
}

/// Format a `DateTime<Utc>` as a local time string.
pub fn format_datetime_local(dt: &DateTime<Utc>) -> String {
    dt.with_timezone(&Local)
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

/// Format an optional UTC timestamp string to local time.
/// Returns `"—"` for None.
pub fn format_optional_local(utc_str: Option<&str>) -> String {
    match utc_str {
        Some(s) => format_local(s),
        None => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfc3339_z_suffix() {
        let result = format_local("2026-02-17T19:46:47Z");
        // Can't assert exact value (depends on test machine timezone),
        // but format should match YYYY-MM-DD HH:MM:SS
        assert!(
            result.len() == 19,
            "Expected 19-char datetime string, got '{}' (len={})",
            result,
            result.len()
        );
        assert!(result.contains('-'));
        assert!(result.contains(':'));
    }

    #[test]
    fn test_rfc3339_offset() {
        let result = format_local("2026-02-17T19:46:47+00:00");
        assert_eq!(result.len(), 19);
    }

    #[test]
    fn test_sqlite_format() {
        let result = format_local("2026-02-17 19:46:47");
        assert_eq!(result.len(), 19);
    }

    #[test]
    fn test_sqlite_fractional() {
        let result = format_local("2026-02-17 19:46:47.123");
        assert_eq!(result.len(), 19);
    }

    #[test]
    fn test_fallback_invalid_input() {
        let result = format_local("not a date");
        assert_eq!(result, "not a date");
    }

    #[test]
    fn test_format_optional_some() {
        let result = format_optional_local(Some("2026-02-17T19:46:47Z"));
        assert_eq!(result.len(), 19);
    }

    #[test]
    fn test_format_optional_none() {
        assert_eq!(format_optional_local(None), "—");
    }

    #[test]
    fn test_format_datetime_local() {
        let dt = Utc::now();
        let result = format_datetime_local(&dt);
        assert_eq!(result.len(), 19);
    }

    #[test]
    fn test_output_format_pattern() {
        // Verify the output matches the expected pattern
        let result = format_local("2026-02-17T12:00:00Z");
        let parts: Vec<&str> = result.split(' ').collect();
        assert_eq!(parts.len(), 2);
        // Date part: YYYY-MM-DD
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date_parts.len(), 3);
        assert_eq!(date_parts[0].len(), 4); // year
        assert_eq!(date_parts[1].len(), 2); // month
        assert_eq!(date_parts[2].len(), 2); // day
        // Time part: HH:MM:SS
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        assert_eq!(time_parts.len(), 3);
    }
}
