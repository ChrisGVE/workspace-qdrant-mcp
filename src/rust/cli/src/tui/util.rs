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

/// Number of body rows a bordered table can show, given the total area height
/// and the count of chrome rows it reserves (top+bottom borders, the header
/// row, and any header bottom-margin). Centralized so every list view computes
/// the visible window the same way — a mismatch here is what lets the cursor
/// scroll off-screen.
pub fn visible_rows(area_height: u16, chrome: u16) -> usize {
    area_height.saturating_sub(chrome) as usize
}

/// Top scroll offset (index of the first rendered row) that keeps `selected`
/// inside a window of `visible` rows. Once the selection passes the bottom of
/// the window it pins to the last visible row rather than disappearing below it.
pub fn scroll_offset(selected: usize, visible: usize) -> usize {
    if visible > 0 && selected >= visible {
        selected - visible + 1
    } else {
        0
    }
}

/// Natural, case-insensitive string comparison.
///
/// Folds case and compares embedded digit runs numerically, so "item2" sorts
/// before "item10" and "Foo" sorts next to "foo".
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let mut ai = a.chars().flat_map(char::to_lowercase).peekable();
    let mut bi = b.chars().flat_map(char::to_lowercase).peekable();

    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let na = take_number(&mut ai);
                    let nb = take_number(&mut bi);
                    match na.cmp(&nb) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                } else {
                    match ca.cmp(&cb) {
                        Ordering::Equal => {
                            ai.next();
                            bi.next();
                        }
                        ord => return ord,
                    }
                }
            }
        }
    }
}

/// Consume a leading run of digits and parse it (saturating at u128::MAX).
fn take_number(it: &mut std::iter::Peekable<impl Iterator<Item = char>>) -> u128 {
    let mut n: u128 = 0;
    while let Some(c) = it.peek().copied() {
        if let Some(d) = c.to_digit(10) {
            n = n.saturating_mul(10).saturating_add(d as u128);
            it.next();
        } else {
            break;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn natural_cmp_case_insensitive() {
        assert_eq!(natural_cmp("Foo", "foo"), Ordering::Equal);
        assert_eq!(natural_cmp("Apple", "banana"), Ordering::Less);
    }

    #[test]
    fn natural_cmp_numeric_aware() {
        assert_eq!(natural_cmp("item2", "item10"), Ordering::Less);
        assert_eq!(natural_cmp("v1.9", "v1.10"), Ordering::Less);
    }

    #[test]
    fn natural_cmp_prefix() {
        assert_eq!(natural_cmp("foo", "foobar"), Ordering::Less);
    }

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

    #[test]
    fn visible_rows_subtracts_chrome() {
        assert_eq!(visible_rows(20, 3), 17);
        assert_eq!(visible_rows(20, 4), 16);
        // Never underflows below zero.
        assert_eq!(visible_rows(2, 4), 0);
    }

    #[test]
    fn scroll_offset_keeps_selection_visible() {
        // Selection within the window: no scroll.
        assert_eq!(scroll_offset(3, 10), 0);
        assert_eq!(scroll_offset(9, 10), 0);
        // Selection past the window: pin it to the last visible row.
        // With 10 visible rows and selected=10, offset 1 puts it at row index 9.
        assert_eq!(scroll_offset(10, 10), 1);
        assert_eq!(scroll_offset(25, 10), 16);
        // Degenerate window never panics.
        assert_eq!(scroll_offset(5, 0), 0);
    }
}
