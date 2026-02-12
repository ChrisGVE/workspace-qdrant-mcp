//! Canonical UTC timestamp formatting for workspace-qdrant-mcp.
//!
//! All timestamps stored in SQLite or used in comparisons MUST use ISO 8601
//! format with millisecond precision and the `Z` suffix (never `+00:00`).
//!
//! # Format
//! `YYYY-MM-DDTHH:MM:SS.mmmZ`  (e.g. `2026-02-12T14:30:00.123Z`)
//!
//! # Why not plain `to_rfc3339()`?
//! `chrono::DateTime<Utc>::to_rfc3339()` produces `+00:00` instead of `Z`.
//! While semantically identical, mixing both formats in the same SQLite column
//! causes confusion and makes string comparisons unreliable.

use chrono::{DateTime, SecondsFormat, Utc};

/// Return the current UTC time as an ISO 8601 string with `Z` suffix.
///
/// Format: `YYYY-MM-DDTHH:MM:SS.mmmZ`
pub fn now_utc() -> String {
    format_utc(&Utc::now())
}

/// Format a `DateTime<Utc>` as an ISO 8601 string with `Z` suffix.
///
/// Format: `YYYY-MM-DDTHH:MM:SS.mmmZ`
pub fn format_utc(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Format an optional `DateTime<Utc>`, returning `None` when the input is `None`.
pub fn format_optional_utc(dt: &Option<DateTime<Utc>>) -> Option<String> {
    dt.as_ref().map(format_utc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn now_utc_ends_with_z() {
        let ts = now_utc();
        assert!(ts.ends_with('Z'), "Expected Z suffix, got: {}", ts);
        assert!(
            !ts.contains("+00:00"),
            "Must not contain +00:00, got: {}",
            ts
        );
    }

    #[test]
    fn format_utc_produces_correct_format() {
        let dt = Utc.with_ymd_and_hms(2026, 2, 12, 14, 30, 0).unwrap();
        let ts = format_utc(&dt);
        assert_eq!(ts, "2026-02-12T14:30:00.000Z");
    }

    #[test]
    fn format_utc_never_contains_offset() {
        let dt = Utc::now();
        let ts = format_utc(&dt);
        assert!(ts.ends_with('Z'), "Expected Z suffix, got: {}", ts);
        assert!(
            !ts.contains("+00:00"),
            "Must not contain +00:00, got: {}",
            ts
        );
    }

    #[test]
    fn format_utc_matches_expected_pattern() {
        let ts = now_utc();
        // YYYY-MM-DDTHH:MM:SS.mmmZ
        let re = regex_lite::Regex::new(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        )
        .unwrap();
        assert!(re.is_match(&ts), "Timestamp doesn't match pattern: {}", ts);
    }

    #[test]
    fn format_optional_utc_none() {
        assert_eq!(format_optional_utc(&None), None);
    }

    #[test]
    fn format_optional_utc_some() {
        let dt = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let result = format_optional_utc(&Some(dt));
        assert_eq!(result, Some("2026-01-01T00:00:00.000Z".to_string()));
    }
}
