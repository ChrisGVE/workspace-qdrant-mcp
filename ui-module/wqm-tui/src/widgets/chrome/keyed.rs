//! A label with ONE of its letters lit — the idiom two different elements need, stated once.
//!
//! A Dashboard cell heading lights the letter that focuses it; a sortable column header lights
//! the letter that sorts by it. Different hues, different elements, and the same three-way
//! split underneath: everything before the letter, the letter, everything after. The split was
//! written for [`crate::widgets::chrome::zone_heading`] first and copied nowhere — this is the
//! copy that was about to be made, made into a function instead.
//!
//! # Why the caller passes both styles
//!
//! What a lit letter looks like is not one thing. A heading's key is
//! [`crate::tokens::accent`] and bold on top of whatever rung the heading already wears; a
//! column's sort key is [`crate::tokens::selector`] at the header's own weight. Deciding it
//! here would mean this function knowing which element it was drawing, which is exactly the
//! knowledge that makes a shared helper stop being shared.

use ratatui::{style::Style, text::Span};

/// `title` split around the FIRST case-insensitive occurrence of `key`, which takes
/// `key_style`; everything else takes `rest_style`.
///
/// Found in the title rather than carried beside it as an index, so a label that gains a count
/// (`Projects` → `Projects (29)`) or is relabelled cannot leave the lit letter pointing at the
/// wrong character. A `key` of [`None`], or one that does not appear in the title, lights
/// nothing — which is the honest frame: there is no letter to press.
///
/// Returns as few spans as the split needs (one, two or three), so a title whose key is its
/// first letter does not carry an empty span in front of it.
pub fn keyed_spans(
    title: &str,
    key: Option<char>,
    key_style: Style,
    rest_style: Style,
) -> Vec<Span<'static>> {
    let at = key.and_then(|key| {
        title
            .char_indices()
            .find(|(_, c)| c.eq_ignore_ascii_case(&key))
    });
    let Some((at, letter)) = at else {
        return vec![Span::styled(title.to_string(), rest_style)];
    };
    let (before, after) = (&title[..at], &title[at + letter.len_utf8()..]);
    let mut spans = Vec::with_capacity(3);
    if !before.is_empty() {
        spans.push(Span::styled(before.to_string(), rest_style));
    }
    spans.push(Span::styled(letter.to_string(), key_style));
    if !after.is_empty() {
        spans.push(Span::styled(after.to_string(), rest_style));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::{Color, Modifier};

    fn styles() -> (Style, Style) {
        (
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            Style::default().fg(Color::White),
        )
    }

    /// The three shapes the split has, and the one it must NOT have: an empty leading span.
    #[test]
    fn the_split_carries_no_empty_span_and_lights_the_first_match() {
        let (key, rest) = styles();

        let leading = keyed_spans("Note", Some('n'), key, rest);
        assert_eq!(leading.len(), 2, "no empty span before a leading key: {leading:?}");
        assert_eq!(leading[0].content, "N");
        assert_eq!(leading[0].style, key);

        let middle = keyed_spans("Rule name", Some('n'), key, rest);
        assert_eq!(
            middle.iter().map(|s| s.content.as_ref()).collect::<Vec<_>>(),
            vec!["Rule ", "n", "ame"],
            "the FIRST `n` is the one lit, not the last"
        );
        assert_eq!(middle[1].style, key);
        assert_eq!(middle[0].style, rest);

        // A title with the key in it TWICE — the only shape that tells first from last apart.
        let twice = keyed_spans("Queue", Some('u'), key, rest);
        assert_eq!(
            twice.iter().map(|s| s.content.as_ref()).collect::<Vec<_>>(),
            vec!["Q", "u", "eue"],
            "the FIRST `u` of Queue is lit, not the second"
        );

        let trailing = keyed_spans("Sync", Some('c'), key, rest);
        assert_eq!(
            trailing.iter().map(|s| s.content.as_ref()).collect::<Vec<_>>(),
            vec!["Syn", "c"]
        );
    }

    /// A key that is not in the title lights nothing — and neither does no key at all.
    #[test]
    fn a_key_the_title_does_not_contain_lights_nothing() {
        let (key, rest) = styles();
        for absent in [None, Some('z')] {
            let spans = keyed_spans("Files", absent, key, rest);
            assert_eq!(spans.len(), 1, "{absent:?} lit something: {spans:?}");
            assert_eq!(spans[0].style, rest);
        }
    }

    /// The match ignores case, so a lowercase key finds the capital that starts a title.
    #[test]
    fn the_match_ignores_case() {
        let (key, rest) = styles();
        let spans = keyed_spans("Branch", Some('b'), key, rest);
        assert_eq!(spans[0].content, "B");
        assert_eq!(spans[0].style, key);
    }
}
