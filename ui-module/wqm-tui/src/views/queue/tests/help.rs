//! The help modal — the one place the paging keys are ever shown.

use super::*;
use crate::panes::list::NAV_HELP;

/// The lines the modal actually draws, taken from inside its own rect.
///
/// Read off the rendered screen rather than off the [`crate::widgets::modal::Modal`], because a
/// window that is built correctly and drawn off the edge is exactly as useless as one that is
/// not. Restricted to the window because the list behind it is full of `y`s and `c`s, and a
/// whole-page search would find every key whether or not the help mentioned it — the failure the
/// first cut of this guard had.
/// The help window is taller than the page it floats over — 45 rows to [`TALL`]'s 34 — so it
/// gets its own area here rather than being clipped to the page and dropping its bottom
/// sections.
const HELP_TALL: u16 = 50;

fn window() -> Vec<String> {
    let _restore = Restore::dark_truecolor();
    let buf = render(view(QueueState::default()).modal(Queue::help()), WIDE, HELP_TALL);
    let at = Queue::help().rect(Rect::new(0, 0, WIDE, HELP_TALL));
    (at.y..at.y + at.height)
        .map(|y| {
            (at.x..at.x + at.width)
                .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
                .collect::<String>()
        })
        .collect()
}

/// The five sections appear in the stated order and none is empty.
#[test]
fn sections_appear_in_order_and_are_not_empty() {
    let sections = Queue::help_sections();
    let titles: Vec<&str> = sections.iter().map(|s| s.title).collect();
    assert_eq!(
        titles,
        vec![
            "Navigation",
            "Sorting",
            "Focus",
            "Search / Filter",
            "List Item Selection",
            "Action",
            "General"
        ]
    );
    for section in &sections {
        assert!(
            !section.entries.is_empty(),
            "the {title} section has no entries",
            title = section.title
        );
    }
}

/// No key string repeats inside one section, except the intentional `f` in Search / Filter
/// which both opens and clears the filter.
#[test]
fn no_key_string_repeats_inside_a_section() {
    for section in Queue::help_sections() {
        let mut seen = std::collections::HashSet::new();
        for (key, _) in &section.entries {
            if section.title == "Search / Filter" && *key == "f" {
                // `f` is intentionally listed twice: once to open the filter and once to clear it.
                continue;
            }
            assert!(
                seen.insert(*key),
                "`{key}` appears more than once in the {} section",
                section.title
            );
        }
    }
}

/// The rendered key column is at least as wide as the widest key across all sections.
#[test]
fn rendered_key_column_is_as_wide_as_the_widest_key() {
    let sections = Queue::help_sections();
    let widest = sections
        .iter()
        .flat_map(|s| s.entries.iter())
        .map(|(key, _)| key.chars().count())
        .max()
        .unwrap_or(0);
    assert!(widest > 0, "there should be at least one key");

    // Flattened to plain text: this guard is about the key COLUMN's arithmetic, which is the
    // one property of a help line that survived ruling 6's styling unchanged.
    let rendered: Vec<String> = crate::panes::list::help::render(&sections, None)
        .iter()
        .map(|line| line.spans.iter().map(|span| span.content.as_ref()).collect::<String>())
        .collect();

    // Every entry line is "  <key-padded><what>". The padded key field must be at least
    // `widest + 2` characters wide (the measured width plus the gap before the meaning).
    for section in &sections {
        for (key, what) in &section.entries {
            let line = rendered
                .iter()
                .find(|line| line.ends_with(*what))
                .expect("every entry is drawn on its own line");
            let chars: Vec<char> = line.chars().collect();
            assert!(
                chars.starts_with(&[' ', ' ']),
                "entry line should be indented two spaces: {line:?}"
            );
            let key_chars: Vec<char> = key.chars().collect();
            assert!(
                chars[2..].starts_with(&key_chars),
                "entry line should start with its key after the indent: {line:?}"
            );
            let what_chars: Vec<char> = what.chars().collect();
            let what_at = chars.len() - what_chars.len();
            assert!(
                chars[what_at..] == what_chars[..],
                "entry line should end with its meaning: {line:?}"
            );
            assert!(
                what_at >= 2 + key_chars.len() + 2,
                "`{key}` runs into its meaning on {line:?}"
            );
            assert!(
                what_at >= 2 + widest,
                "the key column ({what_at}) is narrower than the widest key ({widest}) in {line:?}"
            );
        }
    }
}

/// The foot offers `y`, `c` and `x` for Retry, Cancel and Remove, and no longer offers `r`.
#[test]
fn foot_offers_retry_cancel_remove_and_not_retry_r() {
    let hints = view(QueueState::default()).hints();
    let keys: Vec<&str> = hints.iter().map(|(key, _)| *key).collect();
    assert!(keys.contains(&"y"), "foot should offer y for Retry: {hints:?}");
    assert!(keys.contains(&"c"), "foot should offer c for Cancel: {hints:?}");
    assert!(keys.contains(&"x"), "foot should offer x for Remove: {hints:?}");
    assert!(!keys.contains(&"r"), "foot should not offer r: {hints:?}");
}

/// The paging chords are in the help and are offered NOWHERE else.
///
/// Chris, 2026-09-07: *"shown only in the help… valid for all lists including the dashboard"*.
/// Checked against what the foot OFFERS rather than against what it draws — a chord pushed onto
/// an over-full hint row would vanish into the narrow-foot fallback and look compliant.
#[test]
fn the_paging_chords_are_offered_in_the_help_and_nowhere_else() {
    let mut declared: Vec<(&str, &str)> = Vec::new();
    for section in &Queue::help_sections() {
        declared.extend(section.entries.iter().copied());
    }
    for (chord, what) in NAV_HELP {
        assert!(
            declared.contains(&(chord, what)),
            "{chord:?} {what:?} is not in the help, which is the only place it is offered"
        );
    }

    for state in [
        QueueState::default(),
        QueueState {
            op: Some(Op::Delete),
            ..QueueState::default()
        },
    ] {
        for (key, _) in view(state).hints() {
            assert!(
                !NAV_HELP.iter().any(|(chord, _)| *chord == key),
                "{key:?} is on the foot, where Chris said it never goes"
            );
        }
    }
}

/// Every declared key reaches the screen, with its meaning beside it on one line.
///
/// The half [`sections_appear_in_order_and_are_not_empty`] cannot check: a modal too small for
/// its own body drops lines silently, and the paging chords are the longest entries in it.
#[test]
fn every_declared_key_is_drawn_with_its_meaning_beside_it() {
    let _serial = crate::global_state_lock();

    let drawn = window();
    for section in Queue::help_sections() {
        for (key, what) in &section.entries {
            let found = drawn
                .iter()
                .find(|row| row.contains(*key) && row.contains(*what));
            assert!(
                found.is_some(),
                "`{key}` and {what:?} are not on one line of the window: {drawn:#?}"
            );
        }
    }
}

/// The help window lists every key the foot offers, so a reader can always look up what a hint
/// means.
#[test]
fn every_key_on_the_foot_is_listed_in_the_help() {
    let sections = Queue::help_sections();
    let declared: String = sections
        .iter()
        .flat_map(|s| s.entries.iter().map(|(key, _)| *key))
        .collect::<Vec<_>>()
        .concat();

    for (key, label) in view(QueueState::default()).hints() {
        for letter in key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '?')
        {
            assert!(
                declared.contains(letter),
                "the foot offers `{key} {label}` and the help never mentions {letter:?}"
            );
        }
    }
}

/// A modal is over the page, and the page beneath it is quiet.
///
/// The sweep in [`super`] checks the page drawn WITHOUT its modal; this checks that opening one
/// does the same thing, which is the case a reader actually sees. The modal itself keeps its own
/// colours — it is the thing in focus — so the sweep is over what it does not cover.
#[test]
fn opening_the_help_quiets_the_page_under_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = crate::widgets::chrome::test_support::neutral_rungs();
    let with_modal = render(view(QueueState::default()).modal(Queue::help()), WIDE, TALL);
    let at = Queue::help().rect(Rect::new(0, 0, WIDE, TALL));

    let survivors: Vec<_> =
        crate::widgets::chrome::test_support::coloured_cells(&with_modal, &neutrals)
            .into_iter()
            .filter(|(x, y, _, _)| {
                !(at.x..at.x + at.width).contains(x) || !(at.y..at.y + at.height).contains(y)
            })
            .collect();
    assert!(
        survivors.is_empty(),
        "{} cells outside the modal kept a colour, first ten: {:?}",
        survivors.len(),
        &survivors[..survivors.len().min(10)]
    );
}

/// Ruling 6's presentation (Chris, 20260912): **keys in the accent hue, section titles bold,
/// complementary notes italic.**
///
/// Read off the built lines rather than off the drawn window, because what is being checked is
/// which SPAN carries which treatment — on the page a bold title and a bold key are two runs of
/// cells and nothing says which line either belongs to.
#[test]
fn a_title_is_bold_a_key_is_accented_and_a_note_is_italic() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let sections = Queue::help_sections();
    let lines = crate::panes::list::help::render(&sections, None);
    let text = |line: &ratatui::text::Line| -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    };

    for section in &sections {
        let title = lines
            .iter()
            .find(|line| text(line) == section.title)
            .unwrap_or_else(|| panic!("{:?} is not drawn on a line of its own", section.title));
        assert!(
            title.spans[0]
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD),
            "the {:?} title is not bold",
            section.title
        );

        for (key, what) in &section.entries {
            let line = lines
                .iter()
                .find(|line| text(line).ends_with(*what))
                .unwrap_or_else(|| panic!("{what:?} is not drawn"));
            // The indent, the key, its padding, the meaning — the key is the second span and
            // it is the only one wearing the accent.
            assert_eq!(line.spans[1].content, *key, "the key is not its own span");
            assert_eq!(
                line.spans[1].style.fg,
                Some(crate::tokens::accent()),
                "the key {key:?} is not in the accent hue"
            );
            assert!(
                line.spans
                    .iter()
                    .skip(2)
                    .all(|span| span.style.fg != Some(crate::tokens::accent())),
                "something past the key is accented on the {what:?} line"
            );
        }

        if let Some(note) = section.note {
            let line = lines
                .iter()
                .find(|line| text(line) == note)
                .unwrap_or_else(|| panic!("the {:?} note is not drawn", section.title));
            assert!(
                line.spans[0]
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::ITALIC),
                "the {:?} note is not italic",
                section.title
            );
        }
    }
}

/// The modifier legend is the LAST row of the window, it names all four modifiers, and its shift
/// glyph is not the arrow the arrow keys are spelled with.
///
/// The last clause is the only thing ruling 6 says about a specific character — *"the shift
/// glyph must NOT be the same arrow as the up-arrow key"* — so it is asserted against the arrow
/// the crate actually draws elsewhere rather than against a literal retyped here, which would
/// agree with itself if the foot's arrow ever changed.
#[test]
fn the_legend_closes_the_window_and_its_shift_is_not_the_up_arrow() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let lines = crate::panes::list::help::render(&Queue::help_sections(), Some("a tail"));
    let last: String = lines
        .last()
        .expect("the window has lines")
        .spans
        .iter()
        .map(|s| s.content.as_ref())
        .collect();

    for (glyph, name) in crate::panes::list::help::MODIFIERS {
        assert!(last.contains(glyph), "the legend omits {name}: {last:?}");
        assert!(last.contains(name), "the legend omits the word {name:?}: {last:?}");
    }

    // The tail sits above the legend, never below it — ruling 6 puts the legend on the bottom
    // row and a caller free to append after it is a caller free to get that wrong.
    let tail_at = lines
        .iter()
        .position(|line| {
            line.spans.iter().map(|s| s.content.as_ref()).collect::<String>() == "a tail"
        })
        .expect("the tail is drawn");
    assert!(tail_at < lines.len() - 1, "the legend is not the bottom row");

    let shift = crate::panes::list::help::MODIFIERS[1].0;
    assert_eq!(crate::panes::list::help::MODIFIERS[1].1, "shift");
    // The up-arrow KEY's own spelling, read out of the navigation section rather than retyped
    // here: a literal in this file would agree with itself the day the foot's arrow changed,
    // which is the one day this assertion has any work to do.
    let up = crate::panes::list::help::navigation()
        .entries
        .iter()
        .find(|(_, what)| *what == "Up one row")
        .map(|(key, _)| *key)
        .expect("the navigation section spells the up-arrow key");
    assert!(
        !up.contains(shift),
        "the shift glyph {shift:?} is the arrow the up-arrow key {up:?} is drawn with"
    );
}
