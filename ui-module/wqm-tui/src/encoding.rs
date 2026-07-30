//! What the terminal can actually emit — the capability half of the palette design.
//!
//! [`crate::tokens::Palette`] answers *where a colour comes from*: the terminal's own
//! endpoints, the xterm 256-table ramp, or the sixteen theme slots. This module answers the
//! independent question of *what may be put on the wire at all*. `handover.md` §12 names
//! the two axes Source and Encoding and is explicit that they are different kinds of thing:
//! a source is a **preference**, an encoding is a **capability — probed and never chosen,
//! except by the instrument, which must be able to force any of them**.
//!
//! Conflating them is what made "are the palettes mutually exclusive?" an awkward question.
//! They are not: one ladder is generated from the source and rendered through the encoding.
//!
//! # The encoder picks a family once, never per rung
//!
//! [`Family`] is that decision, and it is ordered by capability, so the rule is a single
//! comparison: **a rung is emitted in the lesser of what the source wants and what the
//! encoding permits** ([`crate::tokens::family`]). Under `Ansi256` the derived ladder does
//! not become "the derived ladder, quantised" — it becomes the *authored* 256 ladder, whole.
//!
//! That is deliberate and measured. `handover.md` §12 quantised the tinted ladder into the
//! 256 palette per rung and recorded the result: the tint is lost on 4 rungs, 4 more
//! overshoot to `+40` where the ideal is `+26`…`+35`, and `faint` and `rule_frame` land on
//! the *same* index 103 — a new collision, worse than either palette mode already shipped.
//! The cube's dark end is too sparse (levels jump 0 → 95) for dark tinted greys to survive
//! it. So degradation drops to the next family's own ladder rather than approximating the
//! one above it.
//!
//! # What `termprofile` is adopted for, and what it is not
//!
//! Detection is [`termprofile`]'s: `TERM`, `COLORTERM`, `TERM_PROGRAM`, `NO_COLOR`,
//! `CLICOLOR`/`CLICOLOR_FORCE`/`FORCE_COLOR`, the CI and multiplexer special cases, and the
//! `is_terminal` check — an accumulation of pseudo-standards there is no point in
//! re-deriving here. Its [`TermProfile`] enum *is* this axis, including the `NoTty` row the
//! design did not have.
//!
//! Its `adapt_color` is **not** adopted: that function is exactly the per-rung quantisation
//! measured and rejected above. Adopting it wholesale would re-import a defect already paid
//! for. So this module uses the crate for *which encoding are we in* and keeps authorship of
//! *what each rung is in that encoding*.
//!
//! Detection needs none of the crate's features, so it costs no transitive dependency: every
//! one of `termprofile`'s seven dependencies is optional and all default features are off.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicU8, Ordering};

use termprofile::{DetectorSettings, EnvVarSource, TermProfile, TermVars};

/// How a colour is put on the wire. Ordered by capability, so `min` is the whole
/// source-meets-encoding rule.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Family {
    /// No colour at all. Shape and weight carry every distinction — which is what
    /// VISUAL-LANGUAGE r02 §3 asks for anyway ("structural signature first, colour
    /// reserved") and what [`crate::tokens::Health::glyph`] already implements.
    None,
    /// The sixteen theme slots. Four of them are neutral, against eleven specified rungs, so
    /// this family collapses roles by construction — see [`crate::tokens::Palette::Theme`].
    Slots,
    /// The xterm 256-table greyscale ramp. Eleven rungs survive; the tint does not.
    Ramp,
    /// Absolute RGB, interpolated between the terminal's own endpoints.
    Rgb,
}

/// What the output stream can emit, as probed.
///
/// The five rows are [`TermProfile`]'s, unchanged — including `NoTty`, which the design's
/// own table omitted. They are re-declared rather than re-exported so this crate's public
/// vocabulary does not name a dependency it uses only at one seam.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Encoding {
    /// The output is not a terminal; no escape sequence should be written.
    NoTty = 0,
    /// A terminal, but colour is refused — `NO_COLOR`, or a forcing variable that says so.
    /// Modifiers (bold, italic) are still legitimate; only colour is not.
    NoColor = 1,
    /// The first sixteen ANSI slots — a login shell, a plain `TERM`.
    Ansi16 = 2,
    /// The 256-colour indexed palette.
    Ansi256 = 3,
    /// 24-bit colour.
    TrueColor = 4,
}

impl From<TermProfile> for Encoding {
    fn from(profile: TermProfile) -> Self {
        match profile {
            TermProfile::NoTty => Encoding::NoTty,
            TermProfile::NoColor => Encoding::NoColor,
            TermProfile::Ansi16 => Encoding::Ansi16,
            TermProfile::Ansi256 => Encoding::Ansi256,
            TermProfile::TrueColor => Encoding::TrueColor,
        }
    }
}

/// The encoding in force. See [`Encoding::current`] for why the default is the permissive
/// end rather than a probe.
static ACTIVE: AtomicU8 = AtomicU8::new(Encoding::TrueColor as u8);

impl Encoding {
    /// Every row, weakest first — the order a degradation sheet compares them in.
    pub const ALL: [Encoding; 5] = [
        Encoding::NoTty,
        Encoding::NoColor,
        Encoding::Ansi16,
        Encoding::Ansi256,
        Encoding::TrueColor,
    ];

    /// The name this row is spelled with in the pantry.
    pub const fn label(self) -> &'static str {
        match self {
            Encoding::NoTty => "No TTY",
            Encoding::NoColor => "No Color",
            Encoding::Ansi16 => "ANSI 16",
            Encoding::Ansi256 => "ANSI 256",
            Encoding::TrueColor => "TrueColor",
        }
    }

    /// The strongest family this encoding can put on the wire.
    pub const fn family(self) -> Family {
        match self {
            // Both refuse colour, for different reasons; the rung is emitted the same way.
            // They stay distinct because what a *caller* should do about them differs — a
            // `NoTty` stream should carry no escape sequence at all, including modifiers.
            Encoding::NoTty | Encoding::NoColor => Family::None,
            Encoding::Ansi16 => Family::Slots,
            Encoding::Ansi256 => Family::Ramp,
            Encoding::TrueColor => Family::Rgb,
        }
    }

    /// The encoding every token currently resolves through.
    ///
    /// # Why the default is `TrueColor` and not a probe
    ///
    /// Rendering must never perform I/O, and a probe is I/O. More importantly, this crate is
    /// a design instrument before it is an application: `cargo pantry dump` writes ANSI
    /// *into a pipe on purpose*, and a probe would correctly report `NoTty` and strip the
    /// escape sequences that are the whole artifact. So detection is an act an application
    /// entry point performs ([`detect`]) rather than a default this module assumes, and the
    /// starting value is the one that renders the authored ladder unaltered.
    pub fn current() -> Encoding {
        match ACTIVE.load(Ordering::Relaxed) {
            0 => Encoding::NoTty,
            1 => Encoding::NoColor,
            2 => Encoding::Ansi16,
            3 => Encoding::Ansi256,
            _ => Encoding::TrueColor,
        }
    }

    /// Forces an encoding. This is the instrument's half of "probed, never chosen": a
    /// degradation frame has to be renderable on a truecolor machine, or the low rows are
    /// never looked at until a user without them reports the collapse.
    pub fn set(encoding: Encoding) {
        ACTIVE.store(encoding as u8, Ordering::Relaxed);
    }
}

/// Probes the process's own stdout.
pub fn detect() -> Encoding {
    detect_for(&std::io::stdout())
}

/// Probes a specific output stream.
pub fn detect_for<T: IsTerminal>(out: &T) -> Encoding {
    TermProfile::detect(out, DetectorSettings::default()).into()
}

/// Probes against variables supplied from memory rather than from the environment.
///
/// This is what makes the ladder *testable* rather than merely implementable: a degradation
/// can be asserted with no terminal, no environment mutation, and no interference between
/// concurrently running tests.
pub fn detect_from<S: EnvVarSource, T: IsTerminal>(source: &S, out: &T) -> Encoding {
    TermProfile::detect_with_vars(TermVars::from_source(
        source,
        out,
        DetectorSettings::default(),
    ))
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs::File;

    /// A stream that is definitively not a terminal, so a test's verdict comes from the
    /// variables under test rather than from wherever the harness sends its output.
    fn not_a_terminal() -> File {
        File::open("/dev/null").expect("/dev/null")
    }

    #[test]
    fn capability_order_is_the_degradation_order() {
        // The whole source-meets-encoding rule is `min` over this order, so the order being
        // right is the rule being right.
        assert!(Family::None < Family::Slots);
        assert!(Family::Slots < Family::Ramp);
        assert!(Family::Ramp < Family::Rgb);
    }

    #[test]
    fn a_pipe_is_no_tty_however_capable_the_terminal_claims_to_be() {
        let vars = HashMap::from_iter([("TERM", "xterm-256color"), ("COLORTERM", "truecolor")]);
        assert_eq!(
            detect_from(&vars, &not_a_terminal()),
            Encoding::NoTty,
            "a non-terminal stream must not be told it has colour"
        );
    }

    #[test]
    fn forcing_is_honoured_without_a_terminal() {
        // `CLICOLOR_FORCE` is what lets an instrument render a degradation frame on a
        // truecolor machine, and what lets this test exist at all.
        for (forced, expected) in [
            ("truecolor", Encoding::TrueColor),
            ("ansi256", Encoding::Ansi256),
            ("ansi16", Encoding::Ansi16),
            ("no_color", Encoding::NoColor),
        ] {
            let vars = HashMap::from_iter([("CLICOLOR_FORCE", forced)]);
            assert_eq!(
                detect_from(&vars, &not_a_terminal()),
                expected,
                "CLICOLOR_FORCE={forced}"
            );
        }
    }

    #[test]
    fn no_color_beats_a_capable_terminal() {
        // `TTY_FORCE` stands in for the terminal this test does not have; without it the
        // stream is a pipe and `NoTty` wins first, which is the neighbouring test.
        let vars = HashMap::from_iter([
            ("TERM", "xterm-256color"),
            ("COLORTERM", "truecolor"),
            ("TTY_FORCE", "1"),
            ("NO_COLOR", "1"),
        ]);
        assert_eq!(detect_from(&vars, &not_a_terminal()), Encoding::NoColor);
    }

    #[test]
    fn forcing_outranks_no_color_and_that_is_measured_not_assumed() {
        // Two published standards contradict each other here: no-color.org says `NO_COLOR`
        // disables colour, bixense's CLICOLOR spec says `CLICOLOR_FORCE` enables it "no
        // matter what". `termprofile` resolves the conflict in favour of forcing, which is
        // the opposite of what this test first asserted.
        //
        // Pinned rather than worked around, because the precedence is a *user-visible*
        // behaviour this crate now inherits: a user who sets `NO_COLOR` in their profile and
        // meets a tool that exports `CLICOLOR_FORCE` gets colour. If that ever changes
        // upstream, this fails and the change is noticed rather than discovered in a frame.
        let vars = HashMap::from_iter([
            ("TERM", "xterm-256color"),
            ("COLORTERM", "truecolor"),
            ("CLICOLOR_FORCE", "1"),
            ("NO_COLOR", "1"),
        ]);
        assert_eq!(detect_from(&vars, &not_a_terminal()), Encoding::TrueColor);
    }

    #[test]
    fn every_row_round_trips_through_the_global() {
        // The global is a `u8`, so a row added to the enum without a discriminant arm in
        // `current` would silently come back as `TrueColor`. This is what notices.
        let _serial = crate::global_state_lock();
        for encoding in Encoding::ALL {
            assert!(!encoding.label().is_empty());
            Encoding::set(encoding);
            assert_eq!(Encoding::current(), encoding);
        }
        Encoding::set(Encoding::TrueColor);
    }
}
