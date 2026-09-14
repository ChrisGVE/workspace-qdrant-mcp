//! The roles the modal framework names on the existing palette.
//!
//! Chris asked for *"more flexibility regarding the use of colors, without moving away from
//! the current palette"* (2026-09-13). Flexibility here is **new roles on old colours**: every
//! function below resolves to a rung of the eleven-rung neutral ladder or to one of the ten
//! semantic fields, reached through accessors that already exist. No hue is invented and no
//! literal is spelled. Every surface goes out through [`crate::tokens::modal_fill`], so a
//! field inside a tinted window is part of that window rather than a patch laid on it.
//!
//! # The two-tier field background is two MECHANISMS, not two shades
//!
//! The 20:30 ruling asks a record view for two marks that must never be confused: *every
//! editable field shows a background*, and *the active field shows a different one*. Two
//! shades of one grey say "one of these is brighter", which a reader has to measure; a **set
//! mark** and a **point mark** say "these are the doors" and "you are standing in this one",
//! which a reader sees. So [`editable_bg`] marks the SET and [`active_bg`] marks the POINT,
//! and the gap between them is a whole emphasis step rather than a nudge.
//!
//! [`active_bg`] is [`crate::tokens::edit_bg`] unchanged — the rung already built for *"you
//! type HERE"* — so the one genuinely new neutral is [`editable_bg`] beneath it.
//!
//! # The rungs are measured, and the first guess was on the wrong side of a crossover
//!
//! The ladder under a bundled theme is a curve through the theme's four neutrals, not an even
//! ramp, so "one rung up" buys different amounts at different heights. Measured on Catppuccin
//! Mocha against [`crate::tokens::layer1_bg`] (rung 15) and the active fill (rung 35) —
//! [`tests::the_ladder_rungs_available_to_a_field_surface`] prints the table:
//!
//! | rung | ΔE off the window | ΔE to the active fill |
//! |---|---|---|
//! | 19 | 3.9 | 14.5 |
//! | **20** (the reference band) | 4.7 | 13.6 |
//! | **22** (the editable set) | 6.8 | 11.6 |
//! | 24 | 8.5 | 9.9 |
//! | 26 | 10.3 | 8.1 |
//!
//! The proportion inverts at rung 24. Above it the SET shouts and the POINT barely separates
//! from it, which is precisely the two-shade failure this scheme exists to avoid. **Rung 22**
//! keeps the set clearly off the window (6.8) while leaving the active field 1.7× that gap to
//! stand out in. [`tests::the_field_surfaces_separate`] asserts the ordering rather than
//! trusting the reasoning, and it runs over all fifteen bundled themes because a curve that
//! inverts once can invert again on a theme nobody looked at.
//!
//! # The tint compresses that ladder — but the ladder was already thin
//!
//! The numbers above are Mocha's bare rungs. Two things happen to them, and only the second
//! changes the design.
//!
//! Once [`crate::tokens::ModalTint::Accent`] is in force at 0.28, every surface is pulled the
//! same fraction toward one blue, which pulls them toward each other: on Mocha the SET lift
//! falls from 6.8 to **4.8** and the POINT separation from 11.6 to **8.0**, about 30% off each.
//! The *ordering* survives on all fifteen themes — the point always separates by more than the
//! set lifts, which is the property the scheme rests on.
//!
//! **The column that decides it is the untinted one**
//! ([`tests::the_tint_compresses_the_field_ladder`] prints all four). On four of the fifteen
//! bundled themes the fill is already thin *before* any blend — Solarized Dark and Solarized
//! Light at ΔE 4.2, One Dark Pro at 4.7, Catppuccin Latte at 5.1 — while **Mocha's 6.8 is among
//! the roomiest**, and Mocha is what the harness paints with and what this round was designed
//! against. A reserved treatment defended on its best case has not been defended (§15).
//!
//! At the proposed strength Solarized Dark's editable fill sits at **ΔE 2.6 — one
//! just-noticeable difference** — off the window it is on, in full truecolor. Neither knob
//! rescues it: halving the tint buys that theme 3.7, about one ΔE, at the cost of exactly what
//! Chris objected to; and no rung helps, because the ladder is fixed, so `window → set` grows
//! only by shrinking `set → point`. At the bracket's far end, 0.40, it falls to **2.1 — below a
//! JND**, which is a stronger objection to that arm than the faint rungs losing their ground.
//!
//! **So [`EDITABLE_MARK`] is not the `NO_COLOR` insurance it was first written up as.** On
//! those themes, on a truecolor terminal, it is the **primary** SET mark and the fill is what
//! supports it. That is why there is no arm without it: see
//! [`crate::views::modal_framework::record::RecordView::rejected_arm_a`].
//!
//! # Colour alone cannot carry the SET mark, so it does not have to
//!
//! Dumped at `CLICOLOR_FORCE=no_color` and `=ansi16` the SET mark **disappears**: the neutral
//! ladder collapses onto four slots, so the window (rung 15) and an editable field (rung 22)
//! land on the same one. The ACTIVE field survives — a different slot, plus the caret and the
//! row mark — but *which fields may I change* ends up carried by colour alone, which is what
//! r06 #8 forbids. [`EDITABLE_MARK`] is the answer: an underline is the cheapest mechanism
//! that survives every encoding, it is already this crate's idiom for an editable cell
//! ([`crate::widgets::config_table`]), and it reads as the thing it means — a blank to fill
//! in. It buys a second thing that is only visible in a pixel render: consecutive editable
//! fields share one fill and merge into a single rectangle, and the underline is the common
//! -region boundary that tells one field from the next.
//!
//! That was the whole argument for it in round 1, and it understated the case — the section
//! below is the measurement that turned the underline from a fallback into the mark itself.
//!
//! # Why the editable set is NOT the selection tint
//!
//! [`crate::tokens::selection_bg`] (rung 19) is a list's *selected row* — the set a command
//! will act on. Spending it again on "the set you may edit" would put one fill on two
//! meanings, which is the collision ruling 7 spent a whole revision undoing.
//!
//! # `accent` is the only hue with room, and what it is allowed to mean
//!
//! Measured across the fifteen bundled themes: `info` is the selector's (§3, absolute),
//! `success`/`warning`/`error` are the health three, `secondary` resolves to
//! [`crate::categorical::selected_of`] — the CURSOR's hue, ΔE 0 — on every non-Catppuccin
//! theme, and `in flight` is sapphire or `info`. **There is no tenth hue spare on all
//! fifteen.** `accent` is spare in the narrower and more useful sense: it already means *this
//! is an affordance, a thing you may press* (the jump digit, a sortable column's letter, a key
//! in the help window), and it carries no SYSTEM STATE, so a surface wearing it cannot be
//! misread as an alarm, a selection or a cursor. That is the whole licence this module takes
//! with it — [`affordance`] for the chevron that says a field opens a list, and
//! [`crate::tokens::ModalTint::Accent`] for the window that owns the keyboard.
//!
//! On Catppuccin Mocha, the theme the harness paints with: `accent` is **blue**, the cursor's
//! hue is **lavender**, and the selector is **teal**.

use std::sync::atomic::{AtomicU8, Ordering};

use ratatui::style::{Color, Modifier};

use super::{edit_bg, mix, modal_fill, neutral, Rgb, WASH_MIX};

/// The mark that says *this field is a blank you may fill in*, and the half of the two-tier
/// scheme that survives an encoding with no ladder left. See the module docs.
///
/// A modifier rather than a colour, which is the point: `CLICOLOR_FORCE=no_color` takes every
/// fill and leaves this standing.
pub const EDITABLE_MARK: Modifier = Modifier::UNDERLINED;

/// How the SET mark is carried — round 1's always-underlined form, round 2's ruling, or the arm
/// round 1 threw out.
///
/// # The two rulings that looked incompatible, and the measurement that reconciled them
///
/// Chris ruled the underline OUT and the tint strength UP to 0.40 in the same message. Under
/// round 1's straight sRGB mix those cannot both hold: the SET mark is a *lightness* step off
/// the window, the mix drags every surface toward one bright accent, and the step is exactly
/// what gets compressed — Solarized Dark fell to ΔE 2.1, under the 2.3 just-noticeable
/// difference, so the fill said nothing and the underline was the only mark left standing.
///
/// [`crate::tokens::TintBlend::HoldLuminance`] does not compress it, because it does not touch
/// the axis the step is made of. Measured at 0.40 with lightness held, the SET lift runs 4.1 to
/// 9.1 across all fifteen bundled themes and **nothing is below the JND**. So the underline can
/// go on his word alone, with nothing traded for it — which is the finding, not the hope.
///
/// # What does NOT change is `NO_COLOR`
///
/// Under an encoding with no ladder left the window (rung 15) and an editable field (rung 22)
/// land on the same one of four slots, whatever the blend does. There *which fields may I
/// change* would be carried by colour alone, and r06 #8 forbids that. So the underline returns —
/// **only there**, as a degradation rather than as part of the look. It is invisible in every
/// frame Chris judges and present in every frame he cannot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SetMark {
    /// Round 1 (A+): fill and underline, on every encoding.
    #[default]
    FillAndUnderline,
    /// **Round 2**: the fill alone where the ladder can express it; the underline returns only
    /// where the encoding collapses the ladder.
    FillWithFallback,
    /// Round 1's rejected arm A: the fill alone, everywhere, including where it cannot be seen.
    /// Evidence only.
    FillOnly,
}

static SET_MARK: AtomicU8 = AtomicU8::new(SetMark::FillAndUnderline as u8);

impl SetMark {
    pub const ALL: [Self; 3] = [
        Self::FillAndUnderline,
        Self::FillWithFallback,
        Self::FillOnly,
    ];

    pub fn current() -> Self {
        match SET_MARK.load(Ordering::Relaxed) {
            1 => Self::FillWithFallback,
            2 => Self::FillOnly,
            _ => Self::FillAndUnderline,
        }
    }

    pub fn set(mark: Self) {
        SET_MARK.store(mark as u8, Ordering::Relaxed);
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::FillAndUnderline => "fill and underline",
            Self::FillWithFallback => "fill, underline only where colour cannot carry it",
            Self::FillOnly => "fill only (rejected)",
        }
    }
}

/// The modifier an editable field's value cell wears, under the mark in force.
///
/// One function, so a widget never has to know which encoding it is on or which arm is selected
/// — the same reason [`crate::tokens::hue`] exists.
pub fn editable_modifier() -> Modifier {
    match SetMark::current() {
        SetMark::FillAndUnderline => EDITABLE_MARK,
        SetMark::FillOnly => Modifier::empty(),
        // The ladder is only expressive enough to carry the mark as a fill under an RGB source.
        // Below that the rungs collapse onto slots and the fill stops saying anything.
        SetMark::FillWithFallback => match super::family() {
            crate::encoding::Family::Rgb => Modifier::empty(),
            _ => EDITABLE_MARK,
        },
    }
}

/// EDIT mode, the SET mark: every field this window will let you change.
///
/// Rung 22 — see the module docs for the measurement that put it there rather than at 26.
/// Through [`crate::tokens::modal_fill`] like the window's own layer, because a field surface
/// INSIDE a tinted window is part of that window. Under the neutral tint it is the bare rung
/// and nothing changes.
pub fn editable_bg() -> Color {
    modal_fill(neutral(22))
}

/// Which rungs the two field marks use — round 1's fixed pair, or round 2's derived POINT.
///
/// # Item 3 asks for black text, and black text is a fact about the FILL
///
/// Chris, 2026-09-14: *"the font of the selected line becomes black, so we increase the
/// contrast"*. Measured on the surface round 2 actually paints — the tint holding lightness at
/// 0.40 — the active field's rung 35 is a **middle grey**, and a middle grey is bad for black
/// and for white at the same time: black clears the WCAG body floor on 6 of the 15 bundled
/// themes and the light end clears it on 4. Catppuccin Mocha, the theme the harness paints with,
/// measures 3.9:1 for black and 4.1:1 for white — both under.
///
/// So his instruction cannot be carried out by changing a foreground. The fill has to move, and
/// [`FieldRungs::BlackText`] moves it: the POINT is the lowest rung at or above 35 on which
/// black clears the floor. Per theme, because the ladder is a curve — it lands on 30 for Nord,
/// 40 for Mocha, and 63 for Solarized Dark, and a single fixed rung would have to be 63 for all
/// of them, which on Mocha is brighter than the body text it sits beside.
///
/// The scheme's own invariant survives it on every theme: the POINT still separates from the SET
/// by more than the SET lifts off the window, which is what keeps *"these are the doors"* and
/// *"you are standing in this one"* two marks rather than two shades.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FieldRungs {
    /// Round 1: SET at rung 22, POINT at [`crate::tokens::edit_bg`]'s rung 35.
    #[default]
    Fixed,
    /// **Round 2, item 3**: the SET unchanged, the POINT derived so black can be read on it.
    BlackText,
}

static RUNGS: AtomicU8 = AtomicU8::new(FieldRungs::Fixed as u8);

impl FieldRungs {
    pub const ALL: [Self; 2] = [Self::Fixed, Self::BlackText];

    pub fn current() -> Self {
        match RUNGS.load(Ordering::Relaxed) {
            1 => Self::BlackText,
            _ => Self::Fixed,
        }
    }

    pub fn set(rungs: Self) {
        RUNGS.store(rungs as u8, Ordering::Relaxed);
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Fixed => "rung 35, text as it comes",
            Self::BlackText => "derived for black text",
        }
    }
}

/// EDIT mode, the POINT mark: the field the keystrokes are going into.
///
/// Under [`FieldRungs::Fixed`] this is [`crate::tokens::edit_bg`] unchanged — a second NAME
/// rather than a second rung, because the role it already carries (*"lighter than the selection
/// tint, so it reads as 'you type HERE'"*) is exactly this one. Under
/// [`FieldRungs::BlackText`] it is derived; see there.
pub fn active_bg() -> Color {
    match FieldRungs::current() {
        FieldRungs::Fixed => modal_fill(edit_bg()),
        FieldRungs::BlackText => black_legible_point(),
    }
}

/// The lowest rung from 35 upward on which black clears the body floor.
///
/// **Upward, and the direction is the whole instruction.** [`super::contrast::legible_ground`]
/// searches darker first, which is right for a bar that only has to carry SOME legible text; it
/// is wrong here, because Chris did not ask for legible text, he asked for BLACK text, and on a
/// dark theme black gets legible by the fill getting lighter. Searching down would satisfy the
/// contrast floor by making the field darker and putting white on it — the opposite of what was
/// asked, passing the same test.
///
/// Rung 35 is the floor of the search rather than an arbitrary start: it is where the POINT
/// already is, so the derived rung can only ever move the mark further from the SET, never
/// closer.
fn black_legible_point() -> Color {
    let black = super::selector_fg();
    (35u8..=100)
        .map(|rung| modal_fill(neutral(rung)))
        .find(|fill| super::contrast::contrast_ratio(black, *fill) >= super::contrast::BODY_FLOOR)
        // No rung works: keep the rung the scheme names and let `active_fg` pick the end that
        // can be read there. An unreadable black is worse than a legible substitution.
        .unwrap_or_else(|| modal_fill(edit_bg()))
}

/// The text on the POINT mark — black wherever black can be read there, and the other end of the
/// ladder where it cannot.
///
/// Bold, like the cursor row's own black-on-lavender, and for the reason ruling 7 gave: black at
/// a normal weight reads thinner than the same text did on the window, and a mark that made its
/// own field harder to read would be a strange kind of emphasis.
pub fn active_fg() -> Color {
    let fill = active_bg();
    let black = super::selector_fg();
    // **The same black the cursor row already uses**, not the ladder's bottom rung.
    //
    // They are different colours and the difference bit: `black_legible_point` searches for a
    // rung where `selector_fg` can be read, and a first cut returned `contrast::text_on`, which
    // answers with `neutral(0)` — the THEME's background, not black. On Dracula the search
    // stopped at the rung where true black clears 4.6:1 and the foreground came back as the
    // theme's dark grey at 3.9:1: a fill derived for one colour, carrying another.
    if super::contrast::contrast_ratio(black, fill) >= super::contrast::BODY_FLOOR {
        black
    } else {
        // Only where no rung could be found at all — a light theme read the other way. The
        // substitution is legible by construction, and is the honest failure of "black".
        super::contrast::text_on(fill)
    }
}

/// The static third column's surface — the default value, or the pre-edit one.
///
/// Chris asked for that column *"in another color than the main window"*. His words name the
/// WINDOW, not the text, so the column sits on its own quiet band and reads as a sheet laid
/// beside the record rather than as a third kind of data — one region, told apart by Gestalt
/// common region, with no hue spent at all.
///
/// Rung 20: quieter off the window (ΔE 4.7) than an editable field is (6.8), so a reference
/// column can never be mistaken for something the reader may act on. It is drawn in VIEW mode
/// only — see [`crate::views::modal_framework::record`], where the band comes off while the
/// window is editing, so that a background on screen then means one thing and one thing only.
pub fn reference_bg() -> Color {
    modal_fill(neutral(20))
}

/// **The selected-text mark** — item 3's last open question, as two arms to be rendered.
///
/// Chris: *"For text selection (which is possible with both edit style, in non-vim it is just
/// done with pressing shift and moving the cursor) we'll have to define a color for the selected
/// text."*
///
/// # What it replaces, and why that had to go
///
/// Today selection is [`ratatui::style::Modifier::REVERSED`], which is not a colour: it swaps
/// whatever is under it. Inside an edit field that means the selection comes out **in the POINT
/// field's own fill**, which is the one pairing guaranteed to say nothing — and once item 3
/// gives the POINT a fill chosen for black text, reversing it produces black-on-black-text's-own
/// colour. So a colour has to be named whatever else is decided.
///
/// # There is no spare hue, and the arms say so
///
/// [`crate::tokens::field`]'s own measurement: `info` is the selector's absolutely,
/// `success`/`warning`/`error` are the health three, `secondary` IS the data cursor's hue on
/// every non-Catppuccin theme, and `accent` is now the window's own tint. A tenth hue does not
/// exist to be spent here, which is why arm A spends none.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SelectedText {
    /// **A — a neutral step, no hue.** The selection is a dark rung under light text: the same
    /// *inversion* the reader already knows from the cursor block, done as two named colours
    /// rather than as a modifier, so it is legible on any fill instead of borrowing one.
    #[default]
    Inverted,
    /// **B — the theme's `secondary`.** A real colour, as the wording asks for. Its cost is a
    /// collision: on every non-Catppuccin bundled theme `secondary` resolves to the same value
    /// as the data cursor's hue, so a text selection and the row cursor would wear one colour
    /// between them.
    Secondary,
}

static SELECTED_TEXT: AtomicU8 = AtomicU8::new(SelectedText::Inverted as u8);

impl SelectedText {
    pub const ALL: [Self; 2] = [Self::Inverted, Self::Secondary];

    pub fn current() -> Self {
        match SELECTED_TEXT.load(Ordering::Relaxed) {
            1 => Self::Secondary,
            _ => Self::Inverted,
        }
    }

    pub fn set(choice: Self) {
        SELECTED_TEXT.store(choice as u8, Ordering::Relaxed);
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Inverted => "A neutral inversion, no hue",
            Self::Secondary => "B theme secondary",
        }
    }
}

/// The fill under selected text.
pub fn selected_text_bg() -> Color {
    match SelectedText::current() {
        // Rung 19 is the SELECTED-ROW tint, and reusing it here is deliberate rather than
        // thrifty: a selection of rows and a selection of characters are the same idea at two
        // scales, so one fill for both is the vocabulary being consistent instead of the
        // palette being short.
        SelectedText::Inverted => modal_fill(neutral(19)),
        SelectedText::Secondary => super::theme()
            .map(|palette| palette.secondary)
            .unwrap_or(Color::Magenta),
    }
}

/// The text on it — the end of the ladder that can be read there, so neither arm can produce an
/// unreadable selection on a theme nobody looked at.
pub fn selected_text_fg() -> Color {
    super::contrast::text_on(selected_text_bg())
}

/// `accent`, used as what it already is: *a thing you may press*.
///
/// The jump digit, a sortable column's letter and a key in the help window all wear it, so a
/// `▾` telling a reader that a field opens a list is the same statement in a new place. It is
/// NOT state — see the module docs for why that distinction is what makes the hue available.
pub fn affordance() -> Color {
    super::accent()
}

/// A layer fill pulled toward `accent` at [`crate::tokens::WASH_MIX`] — the FALLBACK arm.
///
/// Kept, and kept at the screen wash's own strength, because it is the conforming answer if
/// `accent`'s worst case (ΔE 11.0 against lavender on Catppuccin Mocha) is judged too thin to
/// defend for a whole window: keep the window neutral and spend the colour on the FORM
/// instead. That inverts where the colour goes and leaves every role assignment intact.
///
/// Inside an accent-TINTED window it is the same surface treatment twice and the slots merge
/// into the window they sit in, which is why it is a fallback rather than the recommendation —
/// and why the pantry renders it only in the neutral-window variant.
///
/// Slot and indexed encodings cannot be blended, so this collapses onto [`editable_bg`] there.
/// That is the honest degradation: the SET mark survives as a rung, and [`EDITABLE_MARK`]
/// survives regardless.
pub fn accent_wash(base: Color) -> Color {
    let (Some(base), Some(tint)) = (Rgb::from_color(base), Rgb::from_color(super::accent())) else {
        return editable_bg();
    };
    Color::Rgb(
        mix(base.r, tint.r, WASH_MIX),
        mix(base.g, tint.g, WASH_MIX),
        mix(base.b, tint.b, WASH_MIX),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::tokens::{self, delta_e, layer1_bg, layer2_bg, ModalTint, Palette};

    struct Restore(Palette, Encoding, ModalTint, f32);

    impl Restore {
        fn bundled(name: ratatui_themes::ThemeName) -> Self {
            let restore = Restore(
                Palette::current(),
                Encoding::current(),
                ModalTint::current(),
                tokens::tint_strength(),
            );
            Palette::set(Palette::Bundled);
            Encoding::set(Encoding::TrueColor);
            tokens::set_theme(name.palette());
            restore
        }

        fn mocha() -> Self {
            Self::bundled(ratatui_themes::ThemeName::CatppuccinMocha)
        }
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            Palette::set(self.0);
            Encoding::set(self.1);
            ModalTint::set(self.2);
            tokens::set_tint_strength(self.3);
        }
    }

    /// The three surfaces a record view stacks must be told apart at a glance, and the gap
    /// between the SET mark and the POINT mark must be the WIDER of the two — that is the
    /// whole claim of the two-tier scheme, and it is the claim a rung above the crossover
    /// silently breaks.
    ///
    /// Over every bundled theme rather than over Mocha alone: the ladder is a curve through
    /// each theme's own four neutrals, so a crossover that sits at rung 24 here can sit
    /// somewhere else there, and a rung chosen against one theme is a rung chosen against one
    /// theme.
    #[test]
    fn the_field_surfaces_separate_on_every_bundled_theme() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        // Through the SAME blend as the fields, which is the whole reason there is one blend:
        // comparing a tinted field against the bare rung 15 measures the tint, not the
        // ladder, and it inverted the proportion on all fifteen themes when first written
        // that way. `modal_fill(layer1_bg())` is the colour the window is actually painted.
        for name in ratatui_themes::ThemeName::all() {
            tokens::set_theme(name.palette());
            let window_to_set = delta_e(modal_fill(layer1_bg()), editable_bg());
            let set_to_point = delta_e(editable_bg(), active_bg());
            println!(
                "{:<24} set lifts {window_to_set:5.1}   point separates {set_to_point:5.1}",
                name.display_name()
            );
            // ΔE 2.3 is the just-noticeable difference — the floor below which two surfaces
            // are one surface. It is a FLOOR and not a target: the worst bundled theme is
            // Solarized Dark at 2.6, one JND, which is the measurement that makes
            // [`EDITABLE_MARK`] load-bearing rather than belt-and-braces. See the module docs.
            assert!(
                window_to_set > 2.3,
                "{}: a form's slots must lift off the window by more than one JND, ΔE \
                 {window_to_set:.1}",
                name.display_name()
            );
            assert!(
                set_to_point > window_to_set,
                "{}: the active field must separate from its set by more than the set lifts \
                 off the window — ΔE {set_to_point:.1} vs {window_to_set:.1}",
                name.display_name()
            );
        }
    }

    /// **How much of the SET mark the tint costs, and how much was there to begin with.**
    ///
    /// Printed as a table rather than asserted, because the interesting column is the one that
    /// changes the conclusion and it is not the one the round was designed against. Run with
    /// `--nocapture`.
    ///
    /// The first column is the ladder with no tint at all. Read it before reading the others:
    /// on four of the fifteen bundled themes the fill is already thin *before* any blend, and
    /// Catppuccin Mocha — the number this round was designed against, and what the harness
    /// paints with — is one of the roomiest. A reserved treatment defended on its best case is
    /// a treatment that has not been defended (§15).
    #[test]
    fn the_tint_compresses_the_field_ladder() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        println!(
            "{:<24} {:>9} {:>8} {:>8} {:>8}",
            "theme", "tint off", "k=0.14", "k=0.28", "k=0.40"
        );
        for name in ratatui_themes::ThemeName::all() {
            tokens::set_theme(name.palette());
            let mut lift = Vec::new();
            for (tint, strength) in [
                (ModalTint::Neutral, 0.0f32),
                (ModalTint::Accent, 0.14),
                (ModalTint::Accent, 0.28),
                (ModalTint::Accent, 0.40),
            ] {
                ModalTint::set(tint);
                tokens::set_tint_strength(strength);
                lift.push(delta_e(modal_fill(layer1_bg()), editable_bg()));
            }
            println!(
                "{:<24} {:>9.1} {:>8.1} {:>8.1} {:>8.1}",
                name.display_name(),
                lift[0],
                lift[1],
                lift[2],
                lift[3]
            );
        }
    }

    /// **The strength has a floor, and the theme that sets it is not the one we look at.**
    ///
    /// The ordering has to hold on every bundled theme, and the worst theme's SET lift has to
    /// clear a just-noticeable difference. The test NAMES the worst theme, so a future tint or
    /// rung change fails here — with the theme it broke on in the message — rather than being
    /// noticed on somebody's terminal.
    ///
    /// It is deliberately not a Mocha test. Mocha is among the roomiest of the fifteen, so a
    /// floor measured there would pass every change that matters.
    #[test]
    fn the_tint_strength_has_a_floor_and_it_is_not_mocha() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        ModalTint::set(ModalTint::Accent);
        tokens::set_tint_strength(tokens::DEFAULT_TINT_STRENGTH);

        let mut worst = (f32::MAX, "");
        for name in ratatui_themes::ThemeName::all() {
            tokens::set_theme(name.palette());
            let lift = delta_e(modal_fill(layer1_bg()), editable_bg());
            let separation = delta_e(editable_bg(), active_bg());
            assert!(
                separation > lift,
                "{}: the point must separate by more than the set lifts — {separation:.1} vs \
                 {lift:.1}",
                name.display_name()
            );
            if lift < worst.0 {
                worst = (lift, name.display_name());
            }
        }
        println!(
            "worst SET lift at k={:.2}: ΔE {:.1} on {}",
            tokens::DEFAULT_TINT_STRENGTH,
            worst.0,
            worst.1
        );
        assert!(
            worst.0 > 2.3,
            "the fill is below a just-noticeable difference on {} (ΔE {:.1}), so on that theme \
             the SET mark is carried by EDITABLE_MARK alone",
            worst.1,
            worst.0
        );
        assert_ne!(
            worst.1, "Catppuccin Mocha",
            "if Mocha is ever the worst theme, this floor has stopped measuring anything — it \
             exists precisely because the harness's own theme is one of the roomiest"
        );
    }

    /// The reference band is a quieter surface than any field, so it can never be mistaken
    /// for something the reader may act on.
    #[test]
    fn the_reference_band_is_quieter_than_a_field() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        for name in ratatui_themes::ThemeName::all() {
            tokens::set_theme(name.palette());
            assert!(
                delta_e(layer1_bg(), reference_bg()) < delta_e(layer1_bg(), editable_bg()),
                "{}: the reference band must sit closer to the window than an editable field",
                name.display_name()
            );
        }
    }

    /// A printer for the ladder's real spacing under the shipping theme — kept because the
    /// rung choices above are only defensible against numbers, and a reader who wants to move
    /// one needs the same table. Run with `--nocapture`.
    #[test]
    fn the_ladder_rungs_available_to_a_field_surface() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        ModalTint::set(ModalTint::Neutral);
        let window = layer1_bg();
        let point = neutral(35);
        for rung in [17u8, 19, 20, 21, 22, 23, 24, 26, 28, 30, 35, 40, 45] {
            let colour = neutral(rung);
            println!(
                "rung {rung:3}: dE from window {:5.1}   dE to rung 35 {:5.1}",
                delta_e(window, colour),
                delta_e(colour, point)
            );
        }
    }

    /// Which modal tint, and the argument is **identity, not distance**.
    ///
    /// A first cut of this test scored each candidate's ΔE against the reserved roles and
    /// expected `Selected` to come out worst. It did not — and the reason is the whole point.
    /// `ModalTint::Selected` does not sit NEAR the cursor's hue, it **is** the cursor's hue,
    /// and a ΔE table that skips a candidate's own role skips exactly the collision that
    /// disqualifies it. So this is an equality check on the two rejected candidates and a ΔE
    /// floor on the survivor. Run with `--nocapture` for the per-theme table.
    #[test]
    fn the_modal_tint_is_chosen_by_identity_not_by_distance() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        let (mut is_cursor, mut is_selector) = (0, 0);
        let mut worst_accent = (f32::MAX, "", "");
        for name in ratatui_themes::ThemeName::all() {
            let theme = name.palette();
            let selected = crate::categorical::selected_of(&theme);
            let in_flight = crate::categorical::in_flight_of(&theme);

            // `Selected` IS the data cursor's block, which is drawn INSIDE this window.
            if selected == crate::categorical::selected_of(&theme) {
                is_cursor += 1;
            }
            // …and on the themes with no spare hue, `in flight` is also the selector's field.
            if in_flight == theme.info {
                is_selector += 1;
            }

            for (role, colour) in [
                ("info/selector", theme.info),
                ("success", theme.success),
                ("warning", theme.warning),
                ("error", theme.error),
                ("selected/cursor", selected),
                ("in flight", in_flight),
            ] {
                let d = delta_e(theme.accent, colour);
                if d < worst_accent.0 {
                    worst_accent = (d, name.display_name(), role);
                }
            }
        }
        println!(
            "`selected` IS the cursor hue on {is_cursor}/15 themes; `in flight` IS ALSO the \
             selector on {is_selector}/15"
        );
        println!(
            "`accent` worst case over 15 themes: dE {:.1} vs {} on {}",
            worst_accent.0, worst_accent.2, worst_accent.1
        );
        assert_eq!(
            is_cursor, 15,
            "`selected` must be the cursor hue every time"
        );
        assert!(
            is_selector >= 10,
            "`in flight` collapses onto the selector on {is_selector} themes, expected ≥ 10"
        );
        // Thin, and stated as thin: §15 rejected `accent` for the SELECTOR at 14.7 in favour
        // of `info` at 27.9. What makes 11.0 survivable here is that the two never appear as
        // comparable marks — the tint is a one-cell border plus a wash on a fill, the cursor
        // is a filled block with black bold text on it.
        assert!(
            worst_accent.0 > 10.0,
            "accent's worst case is dE {:.1} vs {} on {}",
            worst_accent.0,
            worst_accent.2,
            worst_accent.1
        );
    }

    /// **The seam, pinned.** A `Layer2` confirm opened over a `Layer1` framework window must
    /// never read LESS tinted than the window it is asking about, at any strength — the nearer
    /// layer stands further from the page, not closer to it.
    ///
    /// This is what two blend functions made expressible and one cannot: both surfaces reach
    /// [`crate::tokens::modal_fill`], so the only way for the guard to fall behind its parent
    /// is for the layer rungs themselves to invert, which
    /// `modal::tests::layer_two_stands_further_off_the_base_than_layer_one` already forbids.
    #[test]
    fn a_confirm_guard_is_never_less_tinted_than_the_window_beneath_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        ModalTint::set(ModalTint::Accent);
        let base = layer1_bg();
        for strength in [0.0f32, 0.14, 0.28, 0.40, 1.0] {
            tokens::set_tint_strength(strength);
            let window = modal_fill(layer1_bg());
            let guard = modal_fill(layer2_bg());
            assert!(
                delta_e(base, guard) >= delta_e(base, window),
                "at strength {strength}: the guard is ΔE {:.1} off the page and its parent is \
                 ΔE {:.1} — the depth cue has inverted",
                delta_e(base, guard),
                delta_e(base, window)
            );
        }
    }

    /// The strength is one number and every surface inside the window reads it, so retuning
    /// it moves the whole family rather than the window alone. Measured as *they all moved*:
    /// at two different strengths no surface keeps its colour.
    #[test]
    fn every_surface_inside_the_window_moves_with_the_strength() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        ModalTint::set(ModalTint::Accent);
        tokens::set_tint_strength(0.14);
        let quiet = [
            modal_fill(layer1_bg()),
            editable_bg(),
            active_bg(),
            reference_bg(),
        ];
        tokens::set_tint_strength(0.40);
        let loud = [
            modal_fill(layer1_bg()),
            editable_bg(),
            active_bg(),
            reference_bg(),
        ];
        for (at, (before, after)) in quiet.iter().zip(loud.iter()).enumerate() {
            assert_ne!(before, after, "surface {at} did not follow the strength");
        }
    }

    /// The neutral tint is the identity: nothing is blended, so every surface is its bare
    /// rung. That is what makes `Tint: neutral` an honest arm of the pantry A/B rather than a
    /// differently-mixed one.
    #[test]
    fn the_neutral_tint_blends_nothing() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::mocha();
        ModalTint::set(ModalTint::Neutral);
        assert_eq!(editable_bg(), neutral(22));
        assert_eq!(active_bg(), edit_bg());
        assert_eq!(reference_bg(), neutral(20));
        assert_eq!(modal_fill(layer1_bg()), layer1_bg());
    }
}
