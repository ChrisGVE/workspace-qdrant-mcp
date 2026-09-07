//! The help modal — the one place the paging keys are ever shown.

use super::*;
use crate::panes::list::NAV_HELP;

/// The lines the modal actually draws, taken from inside its own rect.
///
/// Read off the rendered screen rather than off the [`crate::widgets::modal::Modal`], because a
/// window that is built correctly and drawn off the edge is exactly as useless as one that is
/// not. Restricted to the window because the list behind it is full of `r`s and `c`s, and a
/// whole-page search would find every key whether or not the help mentioned it — the failure the
/// first cut of this guard had.
fn window() -> Vec<String> {
    let _restore = Restore::dark_truecolor();
    let buf = render(view(QueueState::default()).modal(Queue::help()), WIDE, TALL);
    let at = Queue::help().rect(Rect::new(0, 0, WIDE, TALL));
    (at.y..at.y + at.height)
        .map(|y| {
            (at.x..at.x + at.width)
                .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
                .collect::<String>()
        })
        .collect()
}

/// The help declares every key of this view, in the ruled order, and the two the foot never
/// shows are among them.
///
/// Spelled out rather than derived, because the list IS the ruling. `?` and `q` are in it: a
/// window claiming to list every key while omitting the two on every screen would be the one
/// place a reader could not check.
#[test]
fn the_help_declares_every_key_of_this_view() {
    // Compared as PAIRS, not as keys: the paging entries come from `NAV_HELP` and a help that
    // transcribed their meanings would carry the right chords beside the wrong words.
    assert_eq!(
        Queue::help_keys(),
        vec![
            ("↓↑ / j k", "Move the cursor"),
            NAV_HELP[0],
            NAV_HELP[1],
            ("Enter", "Open the item, or load the next page"),
            ("/  n N", "Search; next and previous hit"),
            ("f", "Filter the list"),
            ("t", "Cycle the type"),
            ("s", "Cycle the status"),
            ("r  c  x", "Retry, cancel, remove"),
            ("Esc", "Leave search or filter"),
            ("?", "This window"),
            ("q", "Quit"),
        ]
    );
    let keys: Vec<&str> = Queue::help_keys().iter().map(|(key, _)| *key).collect();

    // Every letter the foot offers is one the help declares. Compared letter by letter, because
    // the two surfaces spell a pair differently on purpose — `r c x` is three hints on the foot
    // and one line here.
    let declared: String = keys.concat();
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

/// The paging chords are in the help and are offered NOWHERE else.
///
/// Chris, 2026-09-07: *"shown only in the help… valid for all lists including the dashboard"*.
/// Checked against what the foot OFFERS rather than against what it draws — a chord pushed onto
/// an over-full hint row would vanish into the narrow-foot fallback and look compliant.
#[test]
fn the_paging_chords_are_offered_in_the_help_and_nowhere_else() {
    let declared = Queue::help_keys();
    for (chord, what) in NAV_HELP {
        assert!(
            declared.contains(&(chord, what)),
            "{chord:?} is not in the help, which is the only place it is offered"
        );
    }

    for state in [
        QueueState::default(),
        QueueState {
            kind: Some(Kind::Library),
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
/// The half [`the_help_declares_every_key_of_this_view`] cannot check: a modal too small for its
/// own body drops lines silently, and the paging chords are the longest entries in it.
#[test]
fn every_declared_key_is_drawn_with_its_meaning_beside_it() {
    let _serial = crate::global_state_lock();

    let drawn = window();
    for (key, what) in Queue::help_keys() {
        let found = drawn
            .iter()
            .find(|row| row.contains(key) && row.contains(what));
        assert!(
            found.is_some(),
            "`{key}` and {what:?} are not on one line of the window: {drawn:#?}"
        );
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
