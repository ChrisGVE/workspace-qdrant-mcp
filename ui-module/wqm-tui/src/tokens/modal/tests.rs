//! What the modal switch is pinned to.
//!
//! These guards are about the **switch** — every hue answers muted, the block loses its fill,
//! the neutral ladder is untouched, the scope restores and nests. The two guards that sweep a
//! whole 125 × 34 frame live beside the screens they sweep (`views::dashboard::tests::modal`
//! and `views::shell`), because they are readings of those screens rather than of this module.
//!
//! # Every one of these reads its tokens OUTSIDE the scope first
//!
//! Inside a scope `tokens::accent()` *is* the muted rung, so an assertion taken there compares
//! muted with muted and passes however wrong the switch is. So each guard reads the live value
//! first and asserts it is not already muted — otherwise the guard is checking nothing, which
//! is the failure mode a colour test is most prone to.

use ratatui::style::{Color, Modifier};

use crate::categorical::Categorical;
use crate::encoding::Encoding;
use crate::tokens::{self, Health, ModalScope, Palette, DISC};
use crate::widgets::chrome::test_support::{neutral_rungs, Restore};

/// Every hue the vocabulary can emit, by the accessor a widget would reach for.
///
/// Named pairs rather than a bare list, so a failure says *which* colour survived.
/// `selector_fg` is here with the rest: it is a hue slot (`Color::Black`), not a rung, and
/// although [`tokens::inverted`] no longer reaches it under a modal, a rule with one silent
/// exception is a rule the next caller gets wrong.
fn hue_accessors() -> Vec<(&'static str, Color)> {
    vec![
        ("accent", tokens::accent()),
        ("selector", tokens::selector()),
        ("selector_fg", tokens::selector_fg()),
        ("in_flight", tokens::in_flight()),
        ("healthy", tokens::healthy()),
        ("degraded", tokens::degraded()),
        ("offline", tokens::offline()),
    ]
}

/// Every hue in the vocabulary is the muted rung under a modal — checked against the rung
/// itself rather than against "not what it was", which would pass on any drift.
#[test]
fn every_hue_the_vocabulary_can_emit_goes_muted_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = hue_accessors();
    let muted = tokens::muted();
    assert!(
        live.iter().all(|(_, colour)| *colour != muted),
        "every hue is already muted with no modal open — this guard would check nothing: {live:?}"
    );

    let _modal = ModalScope::enter();
    for (name, _) in &live {
        let under = hue_accessors()
            .into_iter()
            .find(|(n, _)| n == name)
            .expect("the same accessors, re-read inside the scope")
            .1;
        assert_eq!(under, muted, "`{name}` survived a modal");
    }
}

/// The neutral rungs are NOT touched — the other half of the rule, and the half a heavy-handed
/// implementation would break. A page under a modal is grey, not blank: the emphasis ladder,
/// the two rule weights and the data cursor's tint are structure, and structure survives (§1).
#[test]
fn the_neutral_ladder_is_the_same_ladder_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = neutral_rungs();
    let _modal = ModalScope::enter();
    assert_eq!(neutral_rungs(), live, "a modal moved a neutral rung");
}

/// Every health state's disc keeps its SHAPE and loses its hue (VL §4).
#[test]
fn every_health_state_paints_muted_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let states = [Health::Healthy, Health::Degraded, Health::Offline];
    let live: Vec<Color> = states.iter().map(|state| state.color()).collect();
    let muted = tokens::muted();
    assert!(
        live.iter().all(|colour| *colour != muted),
        "the three states are already muted — this guard would check nothing"
    );

    let _modal = ModalScope::enter();
    for state in states {
        assert_eq!(state.color(), muted, "{state:?} kept its hue under a modal");
        assert_eq!(
            state.glyph(),
            DISC,
            "{state:?} changed shape under a modal — only the hue goes"
        );
    }
}

/// A data hue is a hue: the categorical tier mutes with the roles.
///
/// Measured on Mocha, which is the one bundled flavour that carries a tier worth the name —
/// on the eleven non-Catppuccin themes the tier is a single entry and "they all went muted"
/// would be one comparison dressed as eight.
#[test]
fn the_categorical_tier_mutes_with_the_roles() {
    let _serial = crate::global_state_lock();
    let previous = (Palette::current(), Encoding::current(), tokens::theme());
    Palette::set(Palette::Bundled);
    Encoding::set(Encoding::TrueColor);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    let tier = Categorical::current();
    assert!(
        tier.len() > 1,
        "Mocha carries a real categorical tier, or this guard is checking nothing"
    );
    let live: Vec<Color> = tier.iter().map(|(_, colour)| colour).collect();

    let under: Vec<Color> = {
        let _modal = ModalScope::enter();
        let read: Vec<Color> = tier.iter().map(|(_, colour)| colour).collect();
        for (i, colour) in read.iter().enumerate() {
            assert_eq!(
                tier.color(i),
                Some(*colour),
                "`color` and `iter` disagree about entry {i} under a modal"
            );
        }
        read
    };

    assert_ne!(live, under, "the tier painted straight through a modal");
    let muted = tokens::muted();
    assert!(
        under.iter().all(|colour| *colour == muted),
        "a data hue survived a modal: {under:?}"
    );

    Palette::set(previous.0);
    Encoding::set(previous.1);
    if let Some(theme) = previous.2 {
        tokens::set_theme(theme);
    }
}

/// The inverted block loses its fill and keeps its weight.
///
/// `bg: None` rather than a darker fill: a block behind a page a modal has taken the input from
/// is precisely the highlight §6 removes, and the padding stays so nothing on the row moves.
#[test]
fn an_inverted_block_under_a_modal_has_no_fill_at_all() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let selector = tokens::selector();
    assert_eq!(
        tokens::inverted(selector).bg,
        Some(selector),
        "with nothing over it the block IS the fill"
    );

    let _modal = ModalScope::enter();
    let under = tokens::inverted(selector);
    assert_eq!(under.bg, None, "a fill survived a modal");
    assert_eq!(under.fg, Some(tokens::muted()));
    assert!(
        under.add_modifier.contains(Modifier::BOLD),
        "weight is what is left to say which one is selected"
    );
    assert!(
        !under.add_modifier.contains(Modifier::REVERSED),
        "reverse video is a fill by another name"
    );
}

/// The scope restores on drop, and nests — an inner scope's end must not switch the page back
/// on underneath an outer one.
#[test]
fn the_scope_restores_on_drop_and_nests() {
    let _serial = crate::global_state_lock();

    assert!(!tokens::under_modal(), "a test left a scope open");
    {
        let outer = ModalScope::enter();
        assert!(tokens::under_modal());
        {
            let _inner = ModalScope::enter();
            assert!(tokens::under_modal());
        }
        assert!(
            tokens::under_modal(),
            "the inner scope's drop switched the page back on under the outer one"
        );
        drop(outer);
        assert!(!tokens::under_modal(), "the outer scope outlived its value");
    }
    assert!(!tokens::under_modal());
}
