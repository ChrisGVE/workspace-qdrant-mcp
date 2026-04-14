//! Locale-aware number formatting for CLI output.
//!
//! Formats integers and floats with thousands separators and configurable
//! decimal precision. Defaults: dot for decimal, apostrophe for thousands.

/// Locale settings for number formatting.
#[derive(Debug, Clone)]
pub struct NumberLocale {
    /// Thousands separator (default: `'`).
    pub thousands: char,
    /// Decimal separator (default: `.`).
    pub decimal: char,
}

impl Default for NumberLocale {
    fn default() -> Self {
        Self {
            thousands: '\'',
            decimal: '.',
        }
    }
}

/// Format an integer with thousands separators.
pub fn format_integer(value: i64, locale: &NumberLocale) -> String {
    let negative = value < 0;
    let abs = if negative {
        // Handle i64::MIN overflow
        if value == i64::MIN {
            return format!("-{}", format_integer_unsigned(value as u64, locale));
        }
        (-value) as u64
    } else {
        value as u64
    };
    let formatted = format_integer_unsigned(abs, locale);
    if negative {
        format!("-{formatted}")
    } else {
        formatted
    }
}

/// Format an unsigned integer with thousands separators.
fn format_integer_unsigned(value: u64, locale: &NumberLocale) -> String {
    let s = value.to_string();
    if s.len() <= 3 {
        return s;
    }

    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(locale.thousands);
        }
        result.push(ch);
    }
    result
}

/// Format a usize with thousands separators.
pub fn format_usize(value: usize, locale: &NumberLocale) -> String {
    format_integer(value as i64, locale)
}

/// Format a float with thousands separators and fixed decimal places.
pub fn format_float(value: f64, decimals: usize, locale: &NumberLocale) -> String {
    let negative = value < 0.0;
    let abs = value.abs();

    let integer_part = abs.trunc() as u64;
    let formatted_int = format_integer_unsigned(integer_part, locale);

    if decimals == 0 {
        return if negative {
            format!("-{formatted_int}")
        } else {
            formatted_int
        };
    }

    // Format decimal part with exact precision
    let factor = 10u64.pow(decimals as u32) as f64;
    let decimal_part = ((abs.fract() * factor).round() as u64).min(10u64.pow(decimals as u32) - 1);
    let decimal_str = format!("{:0>width$}", decimal_part, width = decimals);

    if negative {
        format!("-{formatted_int}{}{decimal_str}", locale.decimal)
    } else {
        format!("{formatted_int}{}{decimal_str}", locale.decimal)
    }
}

/// Format a percentage value with appropriate precision.
///
/// Follows the PRD rule: 0 decimals unless a value's significant digit
/// is in the decimal part, capped at `max_decimals`.
pub fn format_percentage(value: f64, max_decimals: usize) -> String {
    let locale = NumberLocale::default();
    let decimals = if value.abs() < 1.0 && value.abs() > 0.0 {
        // Find first significant decimal digit
        let mut d = 1;
        let mut test = value.abs() * 10.0;
        while test < 1.0 && d < max_decimals {
            d += 1;
            test *= 10.0;
        }
        d.min(max_decimals)
    } else {
        0
    };
    format!("{} %", format_float(value, decimals, &locale))
}

/// Format an ISO datetime string as dd-mmm-yyyy (e.g., "13-Apr-2026").
///
/// Tries RFC 3339, then `YYYY-MM-DD HH:MM:SS`, then `YYYY-MM-DD`.
/// Falls back to the first 10 characters if unparseable.
pub fn format_date_short(iso: &str) -> String {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(iso) {
        return dt.format("%d-%b-%Y").to_string();
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(iso, "%Y-%m-%d %H:%M:%S") {
        return dt.format("%d-%b-%Y").to_string();
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(iso, "%Y-%m-%d") {
        return d.format("%d-%b-%Y").to_string();
    }
    // Fallback: first 10 chars or original
    iso.get(..10).unwrap_or(iso).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_locale() -> NumberLocale {
        NumberLocale::default()
    }

    // ── Integer formatting ──────────────────────────────────────────────

    #[test]
    fn format_integer_small() {
        let l = default_locale();
        assert_eq!(format_integer(0, &l), "0");
        assert_eq!(format_integer(42, &l), "42");
        assert_eq!(format_integer(999, &l), "999");
    }

    #[test]
    fn format_integer_thousands() {
        let l = default_locale();
        assert_eq!(format_integer(1_000, &l), "1'000");
        assert_eq!(format_integer(12_345, &l), "12'345");
        assert_eq!(format_integer(1_234_567, &l), "1'234'567");
        assert_eq!(format_integer(1_000_000_000, &l), "1'000'000'000");
    }

    #[test]
    fn format_integer_negative() {
        let l = default_locale();
        assert_eq!(format_integer(-42, &l), "-42");
        assert_eq!(format_integer(-1_000, &l), "-1'000");
        assert_eq!(format_integer(-1_234_567, &l), "-1'234'567");
    }

    #[test]
    fn format_integer_custom_locale() {
        let l = NumberLocale {
            thousands: ',',
            decimal: '.',
        };
        assert_eq!(format_integer(1_234_567, &l), "1,234,567");
    }

    // ── Float formatting ────────────────────────────────────────────────

    #[test]
    fn format_float_no_decimals() {
        let l = default_locale();
        assert_eq!(format_float(1_234.567, 0, &l), "1'234");
    }

    #[test]
    fn format_float_with_decimals() {
        let l = default_locale();
        assert_eq!(format_float(1_234.5, 2, &l), "1'234.50");
        assert_eq!(format_float(0.123, 3, &l), "0.123");
        assert_eq!(format_float(99.9, 1, &l), "99.9");
    }

    #[test]
    fn format_float_negative() {
        let l = default_locale();
        assert_eq!(format_float(-1_234.5, 2, &l), "-1'234.50");
    }

    #[test]
    fn format_float_zero_padded() {
        let l = default_locale();
        assert_eq!(format_float(100.0, 2, &l), "100.00");
    }

    // ── Percentage formatting ───────────────────────────────────────────

    #[test]
    fn format_percentage_whole() {
        assert_eq!(format_percentage(85.0, 2), "85 %");
        assert_eq!(format_percentage(100.0, 2), "100 %");
    }

    #[test]
    fn format_percentage_small() {
        assert_eq!(format_percentage(0.5, 2), "0.5 %");
        assert_eq!(format_percentage(0.05, 2), "0.05 %");
    }

    #[test]
    fn format_percentage_zero() {
        assert_eq!(format_percentage(0.0, 2), "0 %");
    }

    // ── usize formatting ────────────────────────────────────────────────

    #[test]
    fn format_usize_basic() {
        let l = default_locale();
        assert_eq!(format_usize(52_583, &l), "52'583");
        assert_eq!(format_usize(0, &l), "0");
    }

    // ── Date formatting ────────────────────────────────────────────────

    #[test]
    fn format_date_rfc3339() {
        assert_eq!(
            format_date_short("2026-04-13T10:30:00+00:00"),
            "13-Apr-2026"
        );
    }

    #[test]
    fn format_date_naive_datetime() {
        assert_eq!(format_date_short("2025-12-25 14:00:00"), "25-Dec-2025");
    }

    #[test]
    fn format_date_naive_date() {
        assert_eq!(format_date_short("2024-01-01"), "01-Jan-2024");
    }

    #[test]
    fn format_date_fallback() {
        assert_eq!(format_date_short("not-a-date"), "not-a-date");
    }
}
