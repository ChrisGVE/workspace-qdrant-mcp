//! Formatting utilities for the status overview.

use colored::Colorize;

use crate::data::queries::HealthLevel;
use crate::output;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};

/// Format a health level with color (no gutter — inline colored text).
pub(super) fn format_health(level: HealthLevel) -> String {
    match level {
        HealthLevel::Healthy => "healthy".green().to_string(),
        HealthLevel::Degraded => "degraded".yellow().to_string(),
        HealthLevel::Unhealthy => "unhealthy".red().to_string(),
    }
}

/// Format milliseconds into a compact human-readable duration.
/// Examples: "1.2s", "45s", "12m", "3h 25m", "2d 5h"
pub(super) fn format_duration_short(ms: u64) -> String {
    let secs = ms / 1000;
    if secs < 60 {
        if ms < 10_000 {
            format!("{:.1}s", ms as f64 / 1000.0)
        } else {
            format!("{secs}s")
        }
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if m > 0 {
            format!("{h}h {m}m")
        } else {
            format!("{h}h")
        }
    } else {
        let d = secs / 86400;
        let h = (secs % 86400) / 3600;
        if h > 0 {
            format!("{d}d {h}h")
        } else {
            format!("{d}d")
        }
    }
}

/// Pad a string (which may contain ANSI codes) to a target visible width.
pub(super) fn pad_to(s: &str, target: usize) -> String {
    let visible = output::strip_ansi(s).chars().count();
    let padding = target.saturating_sub(visible);
    format!("{s}{}", " ".repeat(padding))
}

/// Format an entity header line: "name:" right-padded, then right-aligned total.
pub(super) fn format_entity_header(
    name: &str,
    total: usize,
    key_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) -> String {
    let label = format!("{name}:");
    let label_w = label.chars().count();
    let key_pad = key_w.saturating_sub(label_w);
    let total_str = format_usize(total, locale);
    let num_pad = num_w.saturating_sub(total_str.chars().count());
    format!(
        "{}{}{}{}{}",
        label.bold(),
        " ".repeat(key_pad),
        " ", // separator between key and value
        " ".repeat(num_pad),
        total_str,
    )
}

/// Format a decomposition line with gutter, label, and right-aligned number.
pub(super) fn format_decomp_line(
    gutter: Gutter,
    label: &str,
    value: usize,
    key_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) -> String {
    let key_str = format!("{label}:");
    let key_pad = key_w.saturating_sub(key_str.chars().count());
    let val_str = format_usize(value, locale);
    let num_pad = num_w.saturating_sub(val_str.chars().count());
    format!(
        "{} {}{} {}{}",
        gutter.colored(),
        key_str,
        " ".repeat(key_pad),
        " ".repeat(num_pad),
        val_str,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_short_sub_second() {
        assert_eq!(format_duration_short(500), "0.5s");
        assert_eq!(format_duration_short(1200), "1.2s");
        assert_eq!(format_duration_short(9999), "10.0s");
    }

    #[test]
    fn test_format_duration_short_seconds() {
        assert_eq!(format_duration_short(10_000), "10s");
        assert_eq!(format_duration_short(45_000), "45s");
        assert_eq!(format_duration_short(59_000), "59s");
    }

    #[test]
    fn test_format_duration_short_minutes() {
        assert_eq!(format_duration_short(60_000), "1m");
        assert_eq!(format_duration_short(720_000), "12m");
    }

    #[test]
    fn test_format_duration_short_hours() {
        assert_eq!(format_duration_short(3_600_000), "1h");
        assert_eq!(format_duration_short(3_600_000 + 25 * 60_000), "1h 25m");
        assert_eq!(format_duration_short(3 * 3_600_000), "3h");
    }

    #[test]
    fn test_format_duration_short_days() {
        assert_eq!(format_duration_short(2 * 86_400_000), "2d");
        assert_eq!(
            format_duration_short(2 * 86_400_000 + 5 * 3_600_000),
            "2d 5h"
        );
    }

    #[test]
    fn test_pad_to_no_ansi() {
        assert_eq!(pad_to("hi", 5), "hi   ");
        assert_eq!(pad_to("hello", 5), "hello");
    }
}
