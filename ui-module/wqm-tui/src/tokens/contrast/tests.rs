//! The round-2 measurement tables — run with `--nocapture`.
//!
//! These are **instruments, not assertions**, with two exceptions at the end. Chris's item 4
//! says the text is unreadable and item 3 asks for black text on the selected field; neither is
//! a question a ΔE floor can answer, and both are decided by the tables below rather than by
//! anybody's reading of a frame.

use ratatui::style::Color;

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{
    self, delta_e, edit_bg, layer1_bg, modal_fill, neutral_at, selector_fg, ModalTint, Palette,
    TintBlend,
};

/// The strength Chris ruled on 2026-09-14: *"I think your Tint: blue 0.40 is much better"*.
const RULED_STRENGTH: f32 = 0.40;

/// Every process-global these tables touch, put back on the way out of the scope.
///
/// [`TintBlend`] is in here rather than saved and restored by hand at the end of a test, and
/// that is not tidiness: the hand-written version leaked on the first `assert!` that fired, and
/// a leaked blend repainted every later test in the run — two unrelated tests failed with it,
/// which is precisely the failure `modal_tint`'s own `with_tint` guard exists to prevent.
struct Restore(Palette, Encoding, ModalTint, f32, TintBlend);

impl Restore {
    /// Every table below is taken under the RULED look — accent at 0.40 — because a number
    /// taken under the old default would be a number about a design nobody is proposing.
    fn ruled() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
            TintBlend::current(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        ModalTint::set(ModalTint::Accent);
        tokens::set_tint_strength(RULED_STRENGTH);
        restore
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        Palette::set(self.0);
        Encoding::set(self.1);
        ModalTint::set(self.2);
        tokens::set_tint_strength(self.3);
        TintBlend::set(self.4);
    }
}

/// Every bundled theme, with the ruled look in force for each.
fn each_theme(mut run: impl FnMut(&str)) {
    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        run(name.display_name());
    }
}

/// **ITEM 4, the defect itself.** The text rungs a window spends, on the fill they land on.
///
/// `faint` (rung 50) is the third column's values. `muted` (62) is the help rows' labels AND a
/// non-editable field's value — the three surfaces Chris named. `normal` (85) is the body
/// baseline and the help rows' keys, printed as the control: if the baseline also fails, the
/// problem is the fill and not the rungs.
#[test]
fn item_4_the_text_rungs_on_the_ruled_window_fill() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    println!(
        "\nITEM 4 — text on the window fill, accent tint at {RULED_STRENGTH:.2}\n\
         ratio is WCAG 2.2; body text needs {BODY_FLOOR:.1}:1. `x` marks a rung that fails.\n"
    );
    println!(
        "{:<24} {:>15} {:>15} {:>15} {:>15}",
        "theme", "faint (50)", "muted (62)", "row (75)", "normal (85)"
    );
    let mut failures = 0usize;
    each_theme(|theme| {
        let ground = modal_fill(layer1_bg());
        let cells: Vec<String> = [50u8, 62, 75, 85]
            .iter()
            .map(|rung| {
                let m = Legibility::measure(*rung, ground);
                if !m.passes() {
                    failures += 1;
                }
                format!(
                    "dE{:4.1} {:4.1}:1{}",
                    m.delta_e,
                    m.ratio,
                    if m.passes() { " " } else { "x" }
                )
            })
            .collect();
        println!(
            "{theme:<24} {:>15} {:>15} {:>15} {:>15}",
            cells[0], cells[1], cells[2], cells[3]
        );
    });
    println!(
        "\n{failures} of {} (theme x rung) cells fall below the body floor.",
        ratatui_themes::ThemeName::all().len() * 4
    );
}

/// **ITEM 4, the proposal.** The lowest rung that clears the body floor on the ruled fill.
///
/// This is the answer to *"raise the rungs (or re-derive them from the tinted fill)"*: a text
/// rung named as a **contrast target** lands on a different number per theme, because the
/// ladder is a curve through each theme's own neutrals. Naming the target rather than the rung
/// is what makes the window legible on all fifteen instead of on the one we look at.
#[test]
fn item_4_the_rung_a_legible_window_would_have_to_use() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    println!(
        "\nITEM 4 — lowest rung clearing each floor on the ruled window fill\n\
         `none` means no rung on the ladder clears it, which is itself the finding.\n"
    );
    println!(
        "{:<24} {:>12} {:>12} {:>14}",
        "theme", "UI 3.0:1", "body 4.5:1", "faint today"
    );
    each_theme(|theme| {
        let ground = modal_fill(layer1_bg());
        let show = |r: Option<u8>| match r {
            Some(rung) => format!("{rung}"),
            None => "none".to_string(),
        };
        println!(
            "{theme:<24} {:>12} {:>12} {:>14.1}",
            show(rung_for_contrast(ground, UI_FLOOR)),
            show(rung_for_contrast(ground, BODY_FLOOR)),
            contrast_ratio(neutral_at(50), ground),
        );
    });
}

/// **ITEM 3, the black-text question.** What a fill has to be for black text to be legible on it.
///
/// Chris: *"the font of the selected line becomes black, so we increase the contrast"*. The
/// active field's fill today is [`edit_bg`], rung 35 — a DARK neutral — so black on it is not a
/// contrast increase, it is the worst pairing on the window. This prints the ratio of black on
/// every candidate fill so the consequence of his instruction is a number rather than an
/// argument: to get black text, the POINT mark has to become a LIGHT block.
#[test]
fn item_3_black_text_needs_a_light_fill_and_this_is_where_it_starts() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let black = selector_fg();
    println!(
        "\nITEM 3 — black ({black:?}) on a candidate POINT fill, accent tint at \
         {RULED_STRENGTH:.2}\nbody floor {BODY_FLOOR:.1}:1. Rung 35 is today's active fill.\n"
    );
    print!("{:<24}", "theme");
    for rung in [35u8, 55, 65, 70, 75, 80, 85] {
        print!("{:>8}", format!("r{rung}"));
    }
    println!();
    each_theme(|theme| {
        print!("{theme:<24}");
        for rung in [35u8, 55, 65, 70, 75, 80, 85] {
            let fill = modal_fill(neutral_at(rung));
            print!("{:>8}", format!("{:.1}", contrast_ratio(black, fill)));
        }
        println!();
    });
    println!(
        "\nfor reference, black on the VIEW-mode cursor block (lavender), which already \
         carries black bold:"
    );
    each_theme(|theme| {
        println!(
            "  {theme:<22} {:.1}:1",
            contrast_ratio(black, tokens::cursor_bg())
        );
    });
}

/// **ITEM 3, black text on the ACTIVE field, under the blend round 2 proposes.**
///
/// The table above is taken under round 1's straight mix, which lifts every fill toward a bright
/// accent and so flatters black text. [`TintBlend::HoldLuminance`] does not, and the whole point
/// of it is that a rung keeps the lightness it was given — so the question *"can black be read
/// on the active field"* has to be asked again on the surface that will actually be painted.
///
/// The second column is the same question asked of the ROLE rather than of the colour: whichever
/// end of the ladder is legible there. It is black on every dark theme, and it is what stops a
/// light theme — where the ladder runs the other way — from getting unreadable dark-on-dark.
#[test]
fn item_3_black_on_the_active_field_under_the_proposed_blend() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    TintBlend::set(TintBlend::HoldLuminance);
    let black = selector_fg();
    println!(
        "\nITEM 3 — text on the ACTIVE field, lightness held, k={RULED_STRENGTH:.2}\n\
         body floor {BODY_FLOOR:.1}:1. `role` is `contrast::text_on`, which picks the end that \
         can be read.\n"
    );
    println!(
        "{:<24} {:>14} {:>14} {:>10}",
        "theme", "black (asked)", "role (picked)", "picks"
    );
    let mut black_fails = Vec::new();
    each_theme(|theme| {
        let point = tokens::field::active_bg();
        let by_role = text_on(point);
        let (as_black, as_role) = (contrast_ratio(black, point), contrast_ratio(by_role, point));
        if as_black < BODY_FLOOR {
            black_fails.push((theme.to_string(), as_black));
        }
        println!(
            "{theme:<24} {:>14} {:>14} {:>10}",
            format!(
                "{as_black:.1}:1{}",
                if as_black >= BODY_FLOOR { "" } else { "x" }
            ),
            format!("{as_role:.1}:1"),
            if by_role == tokens::neutral_at(0) {
                "dark"
            } else {
                "light"
            }
        );
    });
    println!("\nliteral black falls below the floor on: {black_fails:?}");
}

/// **ITEM 3, the consequence: where the ACTIVE field has to sit for black to be readable on it.**
///
/// The table above says rung 35 is a dead zone — black fails on nine themes and the light end
/// fails on most of the rest, because rung 35 is a MIDDLE grey and a middle grey is bad for both
/// ends at once. So *"the font of the selected line becomes black"* is not a change of
/// foreground: it is a change of the fill under it, and this finds the rung.
///
/// The second half is the constraint that keeps the two-tier scheme intact: wherever the POINT
/// lands, it must still separate from the SET by more than the SET lifts off the window, or the
/// *"these are the doors / you are standing in this one"* pair collapses into two shades a
/// reader has to measure.
#[test]
fn item_3_the_rung_a_black_texted_active_field_needs() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    TintBlend::set(TintBlend::HoldLuminance);
    let black = selector_fg();
    println!(
        "\nITEM 3 — lowest rung where BLACK clears {BODY_FLOOR:.1}:1 on the active fill\n\
         lightness held, k={RULED_STRENGTH:.2}. `set→point` is measured at that rung.\n"
    );
    println!(
        "{:<24} {:>8} {:>10} {:>12} {:>12}",
        "theme", "rung", "black", "window→set", "set→point"
    );
    let mut highest = 0u8;
    each_theme(|theme| {
        let found = (16u8..=100)
            .find(|rung| contrast_ratio(black, modal_fill(neutral_at(*rung))) >= BODY_FLOOR);
        match found {
            Some(rung) => {
                highest = highest.max(rung);
                let point = modal_fill(neutral_at(rung));
                let set = tokens::field::editable_bg();
                println!(
                    "{theme:<24} {:>8} {:>10.1} {:>12.1} {:>12.1}",
                    rung,
                    contrast_ratio(black, point),
                    delta_e(modal_fill(layer1_bg()), set),
                    delta_e(set, point),
                );
            }
            // A light theme reads the ladder the other way: its rung 16 is already near the
            // background it sits on, so black clears the floor there and the search stops at
            // once. A `none` here would mean the ladder has no black-legible rung at all.
            None => println!("{theme:<24} {:>8}", "none"),
        }
    });
    println!(
        "\nthe highest rung any bundled theme needs is {highest}; one rung serves all fifteen \
         only if it is at least that."
    );
}

/// **ITEM 3, the SET mark without its underline.** What the fill alone is worth at 0.40.
///
/// Chris ruled the underline out and the strength up in the same breath, and those two pull
/// against each other: [`crate::tokens::field`]'s own measurement put Solarized Dark's SET fill
/// at ΔE 2.1 at this strength, below the 2.3 just-noticeable difference. This prints the fixed
/// rung beside a **derived** one — the lowest rung that clears a stated lift off the fill the
/// window is actually painted — which is the conforming alternative.
#[test]
fn item_3_the_set_mark_carried_by_the_fill_alone() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    println!(
        "\nITEM 3 — SET fill lift off the window, accent tint at {RULED_STRENGTH:.2}\n\
         JND is {JND:.1}. `fixed` is rung 22 as shipped; `derived` is the lowest rung \
         clearing the target.\n"
    );
    for target in [4.0f32, 5.0, 6.0] {
        println!("  target lift ΔE {target:.1}");
        println!(
            "  {:<24} {:>10} {:>10} {:>10}",
            "theme", "fixed r22", "derived", "its lift"
        );
        each_theme(|theme| {
            let ground = modal_fill(layer1_bg());
            let fixed = delta_e(ground, modal_fill(neutral_at(22)));
            match rung_for_delta_e_blended(ground, target) {
                Some(rung) => println!(
                    "  {theme:<24} {fixed:>10.1} {:>10} {:>10.1}",
                    format!("r{rung}"),
                    delta_e(ground, modal_fill(neutral_at(rung)))
                ),
                None => println!("  {theme:<24} {fixed:>10.1} {:>10} {:>10}", "none", "-"),
            }
        });
        println!();
    }
}

/// **ITEM 3, the question the underline ruling turns on.** Does the fill alone carry the SET
/// mark once the blend stops moving lightness?
///
/// Chris ruled the underline out and the strength up to 0.40 in one message. Under round 1's
/// straight mix those pull against each other: the SET fill is a LIGHTNESS step off the window,
/// the mix drags both surfaces toward one bright blue, and the step is what gets compressed —
/// `tokens::field` measured Solarized Dark at ΔE 2.1, under the 2.3 JND, so on that theme the
/// fill alone says nothing and the underline was the only mark left.
///
/// [`TintBlend::HoldLuminance`] does not compress it, because it does not touch the axis the
/// step is made of. If the right-hand column clears the JND everywhere, the underline can go on
/// Chris's word alone and nothing has to be traded for it.
#[test]
fn item_3_whether_the_fill_alone_survives_once_lightness_is_held() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    println!(
        "\nITEM 3 — SET fill lift off the window at k={RULED_STRENGTH:.2}, by blend\n\
         JND is {JND:.1}; `x` marks a fill that says nothing on its own.\n"
    );
    println!(
        "{:<24} {:>12} {:>12} {:>14} {:>14}",
        "theme", "straight", "held", "straight point", "held point"
    );
    let (mut straight_fail, mut held_fail) = (0usize, 0usize);
    each_theme(|theme| {
        let mut cells = Vec::new();
        for blend in [TintBlend::Straight, TintBlend::HoldLuminance] {
            TintBlend::set(blend);
            let lift = delta_e(modal_fill(layer1_bg()), tokens::field::editable_bg());
            let point = delta_e(tokens::field::editable_bg(), tokens::field::active_bg());
            if lift <= JND {
                match blend {
                    TintBlend::Straight => straight_fail += 1,
                    TintBlend::HoldLuminance => held_fail += 1,
                }
            }
            cells.push((lift, point));
        }
        let mark = |v: f32| {
            if v <= JND {
                format!("{v:.1}x")
            } else {
                format!("{v:.1} ")
            }
        };
        println!(
            "{theme:<24} {:>12} {:>12} {:>14.1} {:>14.1}",
            mark(cells[0].0),
            mark(cells[1].0),
            cells[0].1,
            cells[1].1
        );
    });
    println!(
        "\nfills below the JND: straight mix {straight_fail}/15, lightness held {held_fail}/15."
    );
    assert_eq!(
        held_fail, 0,
        "if the held blend also loses the fill on some theme, the underline cannot simply be \
         dropped and item 3 needs a second mark that is not colour"
    );
}

/// The derived-rung search, taken through the SAME blend the field surfaces go through.
///
/// [`super::rung_for_delta_e`] compares bare rungs; a field is painted through
/// [`modal_fill`], and comparing a bare rung against a tinted ground measures the tint rather
/// than the ladder — the exact error `tokens::field`'s own test records having made.
fn rung_for_delta_e_blended(ground: Color, target: f32) -> Option<u8> {
    (16u8..=100).find(|rung| delta_e(modal_fill(neutral_at(*rung)), ground) >= target)
}

/// **ITEM 3, the selected-text colour.** Two candidates, measured rather than eyeballed.
///
/// Chris: *"For text selection … we'll have to define a color for the selected text."* Today it
/// is [`ratatui::style::Modifier::REVERSED`], which is not a colour and inverts whatever is
/// under it — so inside an edit field it produces the POINT fill's own colour as text, which is
/// the one pairing guaranteed to vanish.
///
/// **A** — [`crate::tokens::selection_bg`], rung 19: the fill a *selected row* already wears, so
/// the vocabulary gains nothing new. **B** — the theme's `secondary`, the one semantic field not
/// yet spent inside a window.
#[test]
fn item_3_the_selected_text_colour_candidates() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    println!(
        "\nITEM 3 — selected-text fill, on the POINT field, accent tint at {RULED_STRENGTH:.2}\n\
         `vs point` is ΔE off the fill it sits on; `text` is the ratio of the text ON it.\n"
    );
    println!(
        "{:<24} {:>22} {:>22}",
        "theme", "A rung-19 selection", "B theme secondary"
    );
    each_theme(|theme| {
        let point = modal_fill(edit_bg());
        let a = modal_fill(neutral_at(19));
        let b = tokens::theme()
            .map(|t| t.secondary)
            .unwrap_or(Color::Magenta);
        let cell = |candidate: Color| {
            format!(
                "dE{:5.1} text {:4.1}:1",
                delta_e(candidate, point),
                contrast_ratio(tokens::neutral_at(85), candidate)
            )
        };
        println!("{theme:<24} {:>22} {:>22}", cell(a), cell(b));
    });
}

/// The transfer function is WCAG's and not a weighted byte sum — pinned on the two ends and on
/// the mid-grey where the two measures disagree most.
///
/// Without this the module is a plausible-looking number generator: a byte-space luminance
/// would give `#808080` a relative luminance of 0.50 where the real answer is 0.216, and every
/// ratio in every table above would be wrong in the same direction.
#[test]
fn the_luminance_is_wcags_and_the_ratios_are_the_published_ones() {
    let white = Color::Rgb(255, 255, 255);
    let black = Color::Rgb(0, 0, 0);
    let grey = Color::Rgb(128, 128, 128);

    assert!((relative_luminance(white) - 1.0).abs() < 0.001);
    assert!(relative_luminance(black).abs() < 0.001);
    // The value that separates this from a byte-space sum: 0.216, not 0.5.
    assert!(
        (relative_luminance(grey) - 0.2158).abs() < 0.001,
        "mid-grey linearises to {:.4}, expected 0.2158 — this is a byte sum, not WCAG",
        relative_luminance(grey)
    );
    // Black on white is the maximum the formula can express.
    assert!((contrast_ratio(black, white) - 21.0).abs() < 0.01);
    // Symmetric: a caller never has to know which argument is the text.
    assert!((contrast_ratio(black, white) - contrast_ratio(white, black)).abs() < 0.001);
    // Identical colours are 1:1, the floor of the scale.
    assert!((contrast_ratio(grey, grey) - 1.0).abs() < 0.001);
}

/// The search returns the LOWEST qualifying rung, and returns nothing rather than lying when the
/// ladder has no answer.
///
/// The second half is the one worth a test: a search that clamped to 100 would report a pass on
/// a light theme, where the ladder runs *down* from the background and no bright rung exists.
#[test]
fn the_rung_search_is_monotone_and_can_come_up_empty() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let ground = modal_fill(layer1_bg());

    // 1.5:1 rather than the body floor, because on the RULED window fill Mocha has no rung
    // that clears 4.5:1 at all — which is the finding of
    // `item_4_the_rung_a_legible_window_would_have_to_use`, not a reason for this test to fail.
    // A monotonicity check needs two targets the ladder can actually answer.
    let (low_target, high_target) = (1.5f32, UI_FLOOR);
    let low = rung_for_contrast(ground, low_target).expect("the ladder answers 1.5:1");
    let high = rung_for_contrast(ground, high_target).expect("the ladder answers 3:1 on Mocha");
    assert!(
        low <= high,
        "a higher target must need a higher rung: {low} for {low_target} vs {high} for \
         {high_target}"
    );
    assert!(
        contrast_ratio(neutral_at(high), ground) >= high_target,
        "the rung the search returned does not itself clear the target"
    );
    if high > 16 {
        assert!(
            contrast_ratio(neutral_at(high - 1), ground) < high_target,
            "rung {} also clears it, so {high} was not the lowest",
            high - 1
        );
    }
    // 21:1 is the maximum the formula can express, so nothing can clear more than that.
    assert_eq!(
        rung_for_contrast(ground, 22.0),
        None,
        "no rung can clear a ratio above the scale's own ceiling"
    );
}

/// **ITEM 4, the cause.** Body text legibility against the tint strength, both blends.
///
/// The control column is `k=0.00`, which is the untinted window — the look that was legible
/// before any of this. Read across a row: under [`TintBlend::Straight`] the ratio falls as the
/// strength rises, because `accent` is brighter than the fill and an sRGB mix carries that
/// brightness into the surface. Under [`TintBlend::HoldLuminance`] the row is FLAT, which is the
/// whole claim: the window gets its colour and the text keeps its ground.
#[test]
fn item_4_the_tint_strength_is_what_moved_the_text() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let restore_blend = TintBlend::current();
    println!(
        "\nITEM 4 — `normal` (rung 85, the body baseline) on the window fill\n\
         WCAG ratio; body floor {BODY_FLOOR:.1}:1. k=0.00 is the untinted control.\n"
    );
    println!(
        "{:<24} {:>34} {:>26}",
        "", "straight sRGB mix (round 1)", "lightness held (round 2)"
    );
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "theme", "k=0.00", "k=0.14", "k=0.28", "k=0.40", "k=0.14", "k=0.28", "k=0.40"
    );
    let mut straight_pass = 0usize;
    let mut held_pass = 0usize;
    each_theme(|theme| {
        let mut cells: Vec<String> = Vec::new();
        TintBlend::set(TintBlend::Straight);
        tokens::set_tint_strength(0.0);
        let control = contrast_ratio(neutral_at(85), modal_fill(layer1_bg()));
        cells.push(format!("{control:.1}"));
        for blend in [TintBlend::Straight, TintBlend::HoldLuminance] {
            TintBlend::set(blend);
            for k in [0.14f32, 0.28, RULED_STRENGTH] {
                tokens::set_tint_strength(k);
                let ratio = contrast_ratio(neutral_at(85), modal_fill(layer1_bg()));
                if (k - RULED_STRENGTH).abs() < f32::EPSILON && ratio >= BODY_FLOOR {
                    match blend {
                        TintBlend::Straight => straight_pass += 1,
                        TintBlend::HoldLuminance => held_pass += 1,
                    }
                }
                cells.push(format!("{ratio:.1}"));
            }
        }
        println!(
            "{theme:<24} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6]
        );
    });
    TintBlend::set(restore_blend);
    println!(
        "\nat the RULED k={RULED_STRENGTH:.2}: straight mix clears the body floor on \
         {straight_pass}/15 themes, lightness held on {held_pass}/15."
    );
}

/// **The claim of [`TintBlend::HoldLuminance`], asserted rather than printed.**
///
/// Two properties, and the pair is what makes it a tint rather than a no-op: the surface's
/// luminance does not move at any strength, and its *colour* does. One without the other is a
/// mode worth nothing — holding luminance while also holding hue is simply the neutral tint,
/// and the test would pass on it.
#[test]
fn holding_the_lightness_keeps_the_ground_and_still_changes_the_colour() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let restore_blend = TintBlend::current();
    TintBlend::set(TintBlend::HoldLuminance);

    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        tokens::set_tint_strength(0.0);
        let bare = modal_fill(layer1_bg());
        // The SHIPPING bracket only. `set_tint_strength` clamps to `0..=1`, but nothing above
        // 0.40 is proposed, and the ceiling is where the claim stops being exactly true — see
        // the gamut note below.
        // **Measured as a contrast RATIO, not as an absolute luminance**, and the difference is
        // not pedantry. Luminance is absolute and the ratio is relative, so a fixed absolute
        // tolerance is strict on a dark theme and loose on a light one: Catppuccin Latte's
        // window sits near the top of the scale, where a 1% relative wobble is five times the
        // absolute figure Mocha's produces. The design's claim is about the ratio — *text keeps
        // the ground it had* — so that is what is pinned.
        // Two clauses, and the second is the deliverable. The first bounds the drift; the
        // second is the claim itself — a theme whose window was legible before the tint is
        // legible after it. A percentage band alone would let a theme cross the floor while
        // still reading as a small move, and the crossing is the only thing Chris asked about.
        let control = contrast_ratio(neutral_at(85), bare);
        for k in [0.14f32, 0.28, 0.40] {
            tokens::set_tint_strength(k);
            let ratio = contrast_ratio(neutral_at(85), modal_fill(layer1_bg()));
            let drift = (ratio - control).abs() / control;
            assert!(
                drift < 0.10,
                "{}: at k={k} body text went from {control:.2}:1 to {ratio:.2}:1 ({:.1}% off) \
                 — the blend is not holding the ground",
                name.display_name(),
                drift * 100.0
            );
            assert!(
                control < BODY_FLOOR || ratio >= BODY_FLOOR,
                "{}: at k={k} the tint pushed body text from a legible {control:.2}:1 to \
                 {ratio:.2}:1, below the {BODY_FLOOR:.1}:1 floor",
                name.display_name()
            );
        }

        // **The limit, stated rather than avoided.** Pushed to full strength the mix asks for
        // the tint's own chroma at the fill's much darker lightness, and on a saturated accent
        // that pair is outside sRGB — `lab_to_color` clamps per channel, and clamping moves
        // luminance a little. Nothing above 0.40 is proposed; this pins that the ceiling
        // degrades gently rather than falling off it.
        tokens::set_tint_strength(1.0);
        let ratio = contrast_ratio(neutral_at(85), modal_fill(layer1_bg()));
        assert!(
            (ratio - control).abs() / control < 0.15,
            "{}: at full strength body text went from {control:.2}:1 to {ratio:.2}:1, which is \
             more than the gamut clamp alone should cost",
            name.display_name()
        );
        // …and at full strength it really is the tint's hue, not a trace of it. Without this
        // the test above would pass on a blend that did nothing at all.
        tokens::set_tint_strength(1.0);
        let full = modal_fill(layer1_bg());
        assert!(
            delta_e(full, bare) > 5.0,
            "{}: at full strength the fill is only ΔE {:.1} from the bare rung — plainly a \
             different colour is the point",
            name.display_name(),
            delta_e(full, bare)
        );
    }
    TintBlend::set(restore_blend);
}
