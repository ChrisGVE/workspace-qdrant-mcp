//! Shared TUI display helpers.
//!
//! Truncation here follows two TUI-wide rules: elision is always marked with
//! an ellipsis (`…`), and for path-like values the filename — which lives at
//! the end of the string — stays visible, with the dropped leading path marked
//! by a leading ellipsis.

/// Truncate `s` to at most `max` characters, appending a trailing ellipsis when
/// the string is too long. The ellipsis counts toward `max`.
pub fn truncate_end(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let kept: String = s.chars().take(max - 1).collect();
    format!("{kept}\u{2026}")
}

/// Truncate a path-like value to at most `max` characters while keeping the
/// filename (and as much trailing path as fits) visible. The dropped prefix is
/// marked with a leading ellipsis. Non-path values degrade gracefully to
/// showing their trailing portion.
pub fn truncate_path(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    if max <= 1 {
        return s.chars().rev().take(max).collect();
    }
    // Keep the last (max - 1) characters — the filename is at the end — and
    // mark the dropped prefix with a leading ellipsis.
    let tail: String = s.chars().skip(n - (max - 1)).collect();
    format!("\u{2026}{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_end_short_passthrough() {
        assert_eq!(truncate_end("hello", 10), "hello");
    }

    #[test]
    fn truncate_end_marks_elision() {
        let r = truncate_end("hello world", 8);
        assert_eq!(r.chars().count(), 8);
        assert!(r.ends_with('\u{2026}'));
    }

    #[test]
    fn truncate_path_short_passthrough() {
        assert_eq!(truncate_path("src/main.rs", 20), "src/main.rs");
    }

    #[test]
    fn truncate_path_keeps_filename() {
        let r = truncate_path("a/very/deep/path/to/main.rs", 12);
        assert_eq!(r.chars().count(), 12);
        assert!(r.starts_with('\u{2026}'));
        assert!(r.ends_with("main.rs"));
    }

    #[test]
    fn truncate_path_tiny_budget() {
        assert_eq!(truncate_path("abcdef", 1).chars().count(), 1);
    }
}
