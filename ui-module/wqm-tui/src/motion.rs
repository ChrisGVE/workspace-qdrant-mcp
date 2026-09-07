//! The movement model every list shares: a typed count, a pending `g`, and the motions they
//! spell.
//!
//! Chris, 2026-09-07, defined one movement vocabulary for the lists — `Home`/`gg` to the top,
//! `End`/`G` to the bottom, `<n>g` to the row whose invariant `No` is `n`, and `<n>` in front of
//! a motion to repeat that motion `n` times. This module is that vocabulary as a pure state
//! machine: a [`Prefix`] accumulates the count and the pending `g`, and its [`Prefix::key`] turns
//! a [`Key`] into a [`Motion`] and a count — with no terminal library type anywhere in the
//! signature.
//!
//! # `<n>g` addresses the invariant `No`, never the drawn position
//!
//! A list's reference number is assigned at load and unmoved by every sort, filter and selector
//! the screen has (see [`crate::views::queue::frames::NO`]). `<n>g` names a row by that number, so
//! "go to row 137" means the same row however the list has been reordered or narrowed since the
//! number was read off it. [`Motion::Row`] therefore carries a `No`, and finding the row that
//! number names is the list's half of the job — this module only reports which number was asked
//! for.
//!
//! # The Dashboard does NOT use `<n>g` or counts
//!
//! The Dashboard's six cells are not lists: each is a narrow projection with its own cursor, and
//! none of them carries a reference number a `<n>g` could address, or a page of rows a count would
//! repeat. So this model belongs to the lists — the Queue today, and every later list tab — while
//! the Dashboard keeps its own simpler movement. One model is shared by every list; the Dashboard
//! is deliberately outside it.
//!
//! # No terminal library leaks in
//!
//! The crate is a storyboard with no event loop, so a motion cannot be handed the `KeyEvent` of
//! whichever terminal backend a future screen runs under. [`Key`] is the smallest shape the model
//! needs, spelled here, so a later keybinding layer maps its own events onto it instead of this
//! module depending on one.

/// The movement a reader asked for, in the vocabulary every list shares.
///
/// [`Motion::Top`], [`Motion::Bottom`] and [`Motion::Row`] already name a destination, so a count
/// means nothing to them — only [`Motion::Up`], [`Motion::Down`], [`Motion::PageUp`] and
/// [`Motion::PageDown`] repeat.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Motion {
    Up,
    Down,
    PageUp,
    PageDown,
    Top,
    Bottom,
    /// The row whose invariant `No` is this number (1-based), if the projection holds it.
    Row(usize),
}

/// A key, in the smallest shape the model needs.
///
/// [`Key::Char`] carries the letters and digits — `g`, `G`, `j`, `k` and `0`–`9` — while the
/// arrows, paging and `Home`/`End` are their own variants, so no terminal library's key type
/// appears in [`Prefix::key`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Char(char),
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
}

/// The typed count and the pending `g` a motion can be prefixed with.
///
/// Both fields are transient: the count is cleared by any non-digit key (consumed by a motion,
/// dropped by anything else), and the pending `g` survives only until the next key — another `g`
/// makes `gg`, anything else drops it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Prefix {
    count: Option<usize>,
    pending_g: bool,
}

impl Prefix {
    /// Feed one key through the prefix, returning what it becomes and the motion the key asked
    /// for, if any.
    ///
    /// The ruled spellings, each produced by this one function so no screen can invent its own:
    ///
    /// | keys | motion | count |
    /// |---|---|---|
    /// | `Home`, `gg` | `Top` | ignored |
    /// | `End`, `G` | `Bottom` | ignored |
    /// | `<n>g` | `Row(n)` | ignored |
    /// | `<n>` then `j`/`k`/`↓`/`↑`/paging | that motion | `n` |
    /// | a bare motion | that motion | `1` |
    /// | a digit | — | accumulate |
    /// | anything else | — | clears the digits |
    pub fn key(self, key: Key) -> (Prefix, Option<(Motion, usize)>) {
        match key {
            Key::Char(c) if c.is_ascii_digit() => {
                let digit = c.to_digit(10).expect("an ascii digit, guarded above") as usize;
                let count = self
                    .count
                    .unwrap_or(0)
                    .saturating_mul(10)
                    .saturating_add(digit);
                (
                    Prefix {
                        count: Some(count),
                        pending_g: self.pending_g,
                    },
                    None,
                )
            }
            Key::Char('g') => match self.count {
                Some(n) => (Prefix::default(), Some((Motion::Row(n), 1))),
                None if self.pending_g => (Prefix::default(), Some((Motion::Top, 1))),
                None => (
                    Prefix {
                        count: None,
                        pending_g: true,
                    },
                    None,
                ),
            },
            Key::Char('G') => (Prefix::default(), Some((Motion::Bottom, 1))),
            Key::Char('j') => (Prefix::default(), Some((Motion::Down, self.count.unwrap_or(1)))),
            Key::Char('k') => (Prefix::default(), Some((Motion::Up, self.count.unwrap_or(1)))),
            Key::Down => (Prefix::default(), Some((Motion::Down, self.count.unwrap_or(1)))),
            Key::Up => (Prefix::default(), Some((Motion::Up, self.count.unwrap_or(1)))),
            Key::PageDown => (
                Prefix::default(),
                Some((Motion::PageDown, self.count.unwrap_or(1))),
            ),
            Key::PageUp => (
                Prefix::default(),
                Some((Motion::PageUp, self.count.unwrap_or(1))),
            ),
            Key::Home => (Prefix::default(), Some((Motion::Top, 1))),
            Key::End => (Prefix::default(), Some((Motion::Bottom, 1))),
            _ => (Prefix::default(), None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Key, Motion, Prefix};

    /// Drive a key sequence through a fresh [`Prefix`], returning the motions it produced and the
    /// prefix left standing — the one shape every guard here reads, so none of them re-implements
    /// the feed.
    fn feed(keys: &[Key]) -> (Prefix, Vec<(Motion, usize)>) {
        let mut prefix = Prefix::default();
        let mut motions = Vec::new();
        for &key in keys {
            let (next, motion) = prefix.key(key);
            prefix = next;
            if let Some(motion) = motion {
                motions.push(motion);
            }
        }
        (prefix, motions)
    }

    #[test]
    fn home_and_gg_both_mean_top() {
        for keys in [vec![Key::Home], vec![Key::Char('g'), Key::Char('g')]] {
            assert_eq!(feed(&keys).1, vec![(Motion::Top, 1)], "{keys:?}");
        }
    }

    #[test]
    fn end_and_capital_g_both_mean_bottom() {
        for keys in [vec![Key::End], vec![Key::Char('G')]] {
            assert_eq!(feed(&keys).1, vec![(Motion::Bottom, 1)], "{keys:?}");
        }
    }

    #[test]
    fn a_number_then_g_names_the_row() {
        assert_eq!(
            feed(&[Key::Char('1'), Key::Char('2'), Key::Char('g')]).1,
            vec![(Motion::Row(12), 1)]
        );
    }

    #[test]
    fn a_number_then_a_motion_repeats_it_that_many_times() {
        assert_eq!(
            feed(&[Key::Char('3'), Key::Char('j')]).1,
            vec![(Motion::Down, 3)]
        );
        assert_eq!(
            feed(&[Key::Char('3'), Key::Down]).1,
            vec![(Motion::Down, 3)]
        );
    }

    #[test]
    fn a_bare_motion_has_a_count_of_one() {
        assert_eq!(feed(&[Key::Char('j')]).1, vec![(Motion::Down, 1)]);
        assert_eq!(feed(&[Key::Up]).1, vec![(Motion::Up, 1)]);
    }

    #[test]
    fn a_non_motion_key_clears_the_typed_digits() {
        // `12x` produces no motion, and the `x` forgets the `12`: the `j` that follows is bare.
        let (_, motions) = feed(&[Key::Char('1'), Key::Char('2'), Key::Char('x')]);
        assert!(motions.is_empty(), "a non-motion key is not a motion");
        let (_, after) = feed(&[
            Key::Char('1'),
            Key::Char('2'),
            Key::Char('x'),
            Key::Char('j'),
        ]);
        assert_eq!(
            after,
            vec![(Motion::Down, 1)],
            "the digit run was cleared, so `j` is bare"
        );
    }

    #[test]
    fn a_pending_g_with_no_digits_waits_for_its_pair() {
        let (_, motions) = feed(&[Key::Char('g')]);
        assert!(
            motions.is_empty(),
            "a lone `g` is pending, not a motion — only `gg` moves"
        );
    }
}
