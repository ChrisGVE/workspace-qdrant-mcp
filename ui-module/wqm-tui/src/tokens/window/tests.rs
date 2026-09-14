//! The window's shorter ladder — that it applies inside and only inside, and that it buys what
//! item 4 asked for.

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{self, contrast, ModalTint, Palette, TintBlend};

struct Restore(Palette, Encoding, ModalTint, f32, TintBlend, WindowText);

impl Restore {
    fn ruled() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
            TintBlend::current(),
            WindowText::current(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        ModalTint::set(ModalTint::Accent);
        tokens::set_tint_strength(0.40);
        TintBlend::set(TintBlend::HoldLuminance);
        WindowText::set(WindowText::Raised);
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
        WindowText::set(self.5);
    }
}

/// **The rule applies inside a window and nowhere else.**
///
/// Both halves, because either alone is passable by a mistake: a rule that never fired would
/// satisfy the second, and one that fired everywhere would satisfy the first.
#[test]
fn the_quiet_rungs_rise_inside_a_window_and_only_there() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();

    let outside = (tokens::faint(), tokens::muted());
    {
        let _window = WindowScope::enter();
        assert!(in_window());
        let inside = (tokens::faint(), tokens::muted());
        assert_ne!(inside.0, outside.0, "faint did not rise inside the window");
        assert_ne!(inside.1, outside.1, "muted did not rise inside the window");
        assert_eq!(
            inside.0, inside.1,
            "inside a window there is ONE quiet rung"
        );
        assert_eq!(inside.0, tokens::neutral_at(RAISED_QUIET));
    }
    assert!(!in_window());
    assert_eq!(
        (tokens::faint(), tokens::muted()),
        outside,
        "the page's own ladder must come back when the window's draw ends"
    );
}

/// Nesting is counted, so a drop-down drawn inside a window does not switch the ladder back on
/// underneath it when it finishes.
#[test]
fn a_nested_scope_restores_the_outer_one_rather_than_clearing_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let outer = WindowScope::enter();
    {
        let _inner = WindowScope::enter();
        assert!(in_window());
    }
    assert!(in_window(), "the outer window is still drawing");
    drop(outer);
    assert!(!in_window());
}

/// Under [`WindowText::Ladder`] the scope is inert, which is what makes the A/B an A/B rather
/// than two different mechanisms.
#[test]
fn the_round_one_arm_is_unaffected_by_the_scope() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    WindowText::set(WindowText::Ladder);
    let outside = (tokens::faint(), tokens::muted());
    let _window = WindowScope::enter();
    assert_eq!((tokens::faint(), tokens::muted()), outside);
}

/// **What item 4 buys, counted.** The three surfaces Chris named clear the body floor on far
/// more themes raised than they do on the ladder.
///
/// A count rather than a blanket assertion, because two bundled themes cannot clear the floor at
/// ANY rung — Solarized Dark and Solarized Light put their own foreground 3.7:1 and 3.4:1 from
/// their own background, so the theme is under WCAG before we draw anything. A test demanding
/// 15 of 15 would be demanding we out-contrast the palette.
#[test]
fn raising_the_quiet_rung_makes_the_windows_text_legible_on_most_themes() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();

    let count = |mode: WindowText| {
        WindowText::set(mode);
        let _window = WindowScope::enter();
        ratatui_themes::ThemeName::all()
            .iter()
            .filter(|name| {
                tokens::set_theme(name.palette());
                let ground = tokens::modal_fill(tokens::layer1_bg());
                contrast::contrast_ratio(tokens::faint(), ground) >= contrast::BODY_FLOOR
                    && contrast::contrast_ratio(tokens::muted(), ground) >= contrast::BODY_FLOOR
            })
            .count()
    };

    let ladder = count(WindowText::Ladder);
    let raised = count(WindowText::Raised);
    println!(
        "themes where the window's quiet text is legible: ladder {ladder}/15, raised {raised}/15"
    );
    assert_eq!(
        ladder, 0,
        "on the ladder, `faint` clears the floor on no bundled theme at all — that is the \
         defect item 4 names"
    );
    assert!(
        raised >= 10,
        "raised, it should clear on at least ten of the fifteen; got {raised}"
    );

    // …and every theme that still fails is NAMED, with both numbers, so the residue is a
    // reported limit rather than an unexamined remainder. `coding.md#measurement`: the residual
    // class is where a missing class hides.
    WindowText::set(WindowText::Raised);
    let _window = WindowScope::enter();
    let mut residue = Vec::new();
    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        let ground = tokens::modal_fill(tokens::layer1_bg());
        let quiet = contrast::contrast_ratio(tokens::faint(), ground);
        if quiet < contrast::BODY_FLOOR {
            residue.push((
                name.display_name(),
                quiet,
                contrast::contrast_ratio(tokens::neutral_at(85), ground),
            ));
        }
    }
    println!("still under the floor at the ceiling rung: {residue:?}");
    // **The claim is that the palette is the limit, not our rung choice**, and that is what is
    // asserted: on every theme that still fails, the quiet rung has come within a tenth of the
    // theme's OWN baseline — the brightest body text that theme can produce on that fill. Two of
    // the four (both Solarized flavours) put their own foreground under the floor before we draw
    // anything, and the other two miss it by less than a tenth of a ratio point.
    //
    // A bare count would have hidden that: four failures reads the same whether we are one step
    // short or nowhere near.
    for (theme, quiet, baseline) in &residue {
        assert!(
            quiet / baseline >= 0.90,
            "{theme}: the quiet rung reaches {quiet:.2}:1 against a baseline of \
             {baseline:.2}:1 — that gap is ours to close, not the palette's"
        );
    }
}

/// The quiet rung stays BELOW the baseline, or the window has one level instead of two and the
/// third column stops reading as reference.
#[test]
fn the_raised_rung_is_still_quieter_than_the_body_text() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::ruled();
    let _window = WindowScope::enter();
    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        let ground = tokens::modal_fill(tokens::layer1_bg());
        let quiet = contrast::contrast_ratio(tokens::faint(), ground);
        let body = contrast::contrast_ratio(tokens::neutral_at(85), ground);
        assert!(
            quiet < body,
            "{}: the quiet rung is {quiet:.1}:1 against a baseline of {body:.1}:1 — there is \
             no hierarchy left",
            name.display_name()
        );
        assert!(
            quiet_colour(50) != tokens::neutral_at(85),
            "{}: the quiet rung has collapsed onto the baseline",
            name.display_name()
        );
    }
}
