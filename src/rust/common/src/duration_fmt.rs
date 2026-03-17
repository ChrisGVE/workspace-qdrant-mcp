//! Unified duration formatting for CLI output.
//!
//! Formats durations as `D:HH:MM:SS` with intelligent column-consistent
//! fractional seconds when any value in a column is sub-second.

/// Format a duration in seconds to human-readable form.
///
/// - `D:HH:MM:SS` when days > 0
/// - `HH:MM:SS` when hours > 0
/// - `MM:SS` otherwise
///
/// `fractional_digits`: if > 0, display fractional seconds with that precision.
pub fn format_duration(secs: f64, fractional_digits: usize) -> String {
    if secs < 0.0 {
        return "—".to_string();
    }

    // Round to the requested precision first to avoid fractional carry issues
    let rounded = if fractional_digits > 0 {
        let scale = 10f64.powi(fractional_digits as i32);
        (secs * scale).round() / scale
    } else {
        secs.round()
    };

    let total_secs = rounded.floor() as u64;
    let frac = rounded - rounded.floor();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let s = total_secs % 60;

    let sec_str = if fractional_digits > 0 {
        let frac_val = (frac * 10f64.powi(fractional_digits as i32)).round() as u64;
        format!("{:02}.{:0>width$}", s, frac_val, width = fractional_digits)
    } else {
        format!("{:02}", s)
    };

    if days > 0 {
        format!("{}:{:02}:{:02}:{}", days, hours, mins, sec_str)
    } else if hours > 0 {
        format!("{:02}:{:02}:{}", hours, mins, sec_str)
    } else {
        format!("{:02}:{}", mins, sec_str)
    }
}

/// Determine the fractional precision needed for a column of durations.
///
/// If any value is less than 1 second (and non-negative), returns 1
/// (one significant fractional digit). Otherwise returns 0.
pub fn column_fractional_digits(durations: &[f64]) -> usize {
    if durations.iter().any(|&d| (0.0..1.0).contains(&d)) {
        1
    } else {
        0
    }
}

/// Format a column of durations with consistent precision.
///
/// If any duration in the slice is sub-second, all values are formatted
/// with fractional seconds at the same precision.
pub fn format_duration_column(durations: &[f64]) -> Vec<String> {
    let digits = column_fractional_digits(durations);
    durations
        .iter()
        .map(|&d| format_duration(d, digits))
        .collect()
}

/// Format a duration in seconds as an approximate human-readable string.
///
/// Inspired by Apple's adaptive time display:
/// - < 60 s  → "less than a minute"
/// - < 3600 s → "about N minute(s)"
/// - < 86400 s → "about N hour(s)"
/// - ≥ 86400 s → "about N day(s)" or "N day(s) and N hour(s)" when hours > 0
pub fn fmt_approx_duration(seconds: f64) -> String {
    if seconds < 0.0 {
        return "unknown".to_string();
    }

    let secs = seconds as u64;
    let minutes = secs / 60;
    let hours = secs / 3600;
    let days = secs / 86400;
    let remaining_hours = (secs % 86400) / 3600;

    if secs < 60 {
        "less than a minute".to_string()
    } else if hours < 1 {
        if minutes == 1 {
            "about 1 minute".to_string()
        } else {
            format!("about {} minutes", minutes)
        }
    } else if days < 1 {
        if hours == 1 {
            "about 1 hour".to_string()
        } else {
            format!("about {} hours", hours)
        }
    } else if remaining_hours == 0 {
        if days == 1 {
            "about 1 day".to_string()
        } else {
            format!("about {} days", days)
        }
    } else {
        let day_str = if days == 1 {
            "1 day".to_string()
        } else {
            format!("{} days", days)
        };
        let hour_str = if remaining_hours == 1 {
            "1 hour".to_string()
        } else {
            format!("{} hours", remaining_hours)
        };
        format!("{} and {}", day_str, hour_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(format_duration(0.0, 0), "00:00");
    }

    #[test]
    fn test_seconds_only() {
        assert_eq!(format_duration(45.0, 0), "00:45");
    }

    #[test]
    fn test_minutes_and_seconds() {
        assert_eq!(format_duration(90.0, 0), "01:30");
    }

    #[test]
    fn test_hours_minutes_seconds() {
        assert_eq!(format_duration(3661.0, 0), "01:01:01");
    }

    #[test]
    fn test_days() {
        assert_eq!(format_duration(90061.0, 0), "1:01:01:01");
    }

    #[test]
    fn test_fractional_seconds() {
        assert_eq!(format_duration(0.5, 1), "00:00.5");
    }

    #[test]
    fn test_fractional_with_minutes() {
        assert_eq!(format_duration(90.3, 1), "01:30.3");
    }

    #[test]
    fn test_negative_duration() {
        assert_eq!(format_duration(-1.0, 0), "—");
    }

    #[test]
    fn test_column_digits_all_above_one() {
        assert_eq!(column_fractional_digits(&[1.0, 2.0, 60.0]), 0);
    }

    #[test]
    fn test_column_digits_has_subsecond() {
        assert_eq!(column_fractional_digits(&[0.5, 2.0, 60.0]), 1);
    }

    #[test]
    fn test_column_digits_empty() {
        assert_eq!(column_fractional_digits(&[]), 0);
    }

    #[test]
    fn test_format_column_consistent() {
        let durations = vec![0.5, 120.0, 3600.0];
        let formatted = format_duration_column(&durations);
        assert_eq!(formatted[0], "00:00.5");
        assert_eq!(formatted[1], "02:00.0");
        assert_eq!(formatted[2], "01:00:00.0");
    }

    #[test]
    fn test_format_column_no_fractional() {
        let durations = vec![60.0, 120.0, 3600.0];
        let formatted = format_duration_column(&durations);
        assert_eq!(formatted[0], "01:00");
        assert_eq!(formatted[1], "02:00");
        assert_eq!(formatted[2], "01:00:00");
    }

    #[test]
    fn test_fractional_carry_over() {
        // When frac rounds up to 10, it should carry into seconds
        // 123.95 with 1 digit: 0.95 * 10 = 9.5 → rounds to 10 → carry
        assert_eq!(format_duration(123.95, 1), "02:04.0");
        // 59.99 with 1 digit: rounds to 60.0 → 01:00.0
        assert_eq!(format_duration(59.99, 1), "01:00.0");
    }

    #[test]
    fn test_large_duration() {
        // 2 days, 3 hours, 4 minutes, 5 seconds
        let secs = 2.0 * 86400.0 + 3.0 * 3600.0 + 4.0 * 60.0 + 5.0;
        assert_eq!(format_duration(secs, 0), "2:03:04:05");
    }

    // fmt_approx_duration tests

    #[test]
    fn test_approx_under_minute() {
        assert_eq!(fmt_approx_duration(0.0), "less than a minute");
        assert_eq!(fmt_approx_duration(30.0), "less than a minute");
        assert_eq!(fmt_approx_duration(59.9), "less than a minute");
    }

    #[test]
    fn test_approx_minutes() {
        assert_eq!(fmt_approx_duration(60.0), "about 1 minute");
        assert_eq!(fmt_approx_duration(90.0), "about 1 minute");
        assert_eq!(fmt_approx_duration(120.0), "about 2 minutes");
        assert_eq!(fmt_approx_duration(3599.0), "about 59 minutes");
    }

    #[test]
    fn test_approx_hours() {
        assert_eq!(fmt_approx_duration(3600.0), "about 1 hour");
        assert_eq!(fmt_approx_duration(7200.0), "about 2 hours");
        assert_eq!(fmt_approx_duration(86399.0), "about 23 hours");
    }

    #[test]
    fn test_approx_days_exact() {
        assert_eq!(fmt_approx_duration(86400.0), "about 1 day");
        assert_eq!(fmt_approx_duration(172800.0), "about 2 days");
    }

    #[test]
    fn test_approx_days_and_hours() {
        assert_eq!(fmt_approx_duration(86400.0 + 3600.0), "1 day and 1 hour");
        assert_eq!(fmt_approx_duration(86400.0 + 14400.0), "1 day and 4 hours");
        assert_eq!(fmt_approx_duration(172800.0 + 7200.0), "2 days and 2 hours");
    }

    #[test]
    fn test_approx_negative() {
        assert_eq!(fmt_approx_duration(-1.0), "unknown");
    }
}
