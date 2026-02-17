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

    let total_secs = secs.floor() as u64;
    let frac = secs - secs.floor();
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
    if durations.iter().any(|&d| d >= 0.0 && d < 1.0) {
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
    durations.iter().map(|&d| format_duration(d, digits)).collect()
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
    fn test_large_duration() {
        // 2 days, 3 hours, 4 minutes, 5 seconds
        let secs = 2.0 * 86400.0 + 3.0 * 3600.0 + 4.0 * 60.0 + 5.0;
        assert_eq!(format_duration(secs, 0), "2:03:04:05");
    }
}
